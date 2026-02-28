#!/usr/bin/env bash
#
# Orchestrates a full training run on Fly.io:
#   1. Create Fly app (if needed)
#   2. Create volume (if needed)
#   3. fly deploy — builds image, pushes, creates machine with volume
#   4. Wait for training to complete (machine exits after CMD finishes)
#   5. Download artifacts via temp machine + ssh tar
#   6. Clean up
#
set -euo pipefail

APP_NAME="dynasty-model-train"
REGION="sjc"
VOLUME_NAME="model_artifacts"
VOLUME_SIZE_GB=5
LOCAL_ARTIFACT_DIR="./artifacts"
LOG_FILE="./artifacts/training.log"

echo "=== Dynasty Model — Fly.io Training ==="

# ---------- 1. Create app if needed ----------
if ! fly apps list --json 2>/dev/null | grep -q "\"$APP_NAME\""; then
    echo "[1/6] Creating Fly app: $APP_NAME"
    fly apps create "$APP_NAME" --org personal
else
    echo "[1/6] App $APP_NAME already exists"
fi

# ---------- 2. Create volume if needed ----------
EXISTING_VOL=$(fly volumes list --app "$APP_NAME" --json 2>/dev/null | grep -c "\"$VOLUME_NAME\"" || true)
if [ "$EXISTING_VOL" -eq 0 ]; then
    echo "[2/6] Creating ${VOLUME_SIZE_GB}GB volume: $VOLUME_NAME"
    fly volumes create "$VOLUME_NAME" \
        --app "$APP_NAME" \
        --region "$REGION" \
        --size "$VOLUME_SIZE_GB" \
        --yes
else
    echo "[2/6] Volume $VOLUME_NAME already exists"
fi

# ---------- 3. Deploy (build + push + run) ----------
echo "[3/6] Building image and deploying..."
echo "    This builds the Docker image, pushes it, and starts a machine."
echo "    The machine will run training and exit when done."
fly deploy --app "$APP_NAME" --ha=false

# ---------- 4. Wait for machine to finish training ----------
echo "[4/6] Waiting for training to complete..."
while true; do
    # Check if any machines are still running
    RUNNING=$(fly machine list --app "$APP_NAME" --json 2>/dev/null | python3 -c "
import json, sys
machines = json.load(sys.stdin)
running = [m for m in machines if m.get('state') in ('created', 'starting', 'started')]
print(len(running))
" 2>/dev/null || echo "1")

    if [ "$RUNNING" -eq 0 ]; then
        echo "    All machines stopped — training complete."
        break
    fi
    echo "    $(date '+%H:%M:%S') $RUNNING machine(s) still running..."
    sleep 30
done

# Save logs
mkdir -p "$(dirname "$LOG_FILE")"
fly logs --app "$APP_NAME" --no-tail 2>/dev/null > "$LOG_FILE" || true
LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
echo "    Saved $LINES log lines to $LOG_FILE"
echo ""
echo "--- Last 30 lines of training output ---"
tail -30 "$LOG_FILE" 2>/dev/null || true
echo "---"

# ---------- 5. Download artifacts ----------
echo "[5/6] Downloading artifacts..."

# Get image ref from the deployed machine
IMAGE_REF=$(fly machine list --app "$APP_NAME" --json 2>/dev/null | python3 -c "
import json, sys
machines = json.load(sys.stdin)
if machines:
    print(machines[0].get('config', {}).get('image', ''))
" 2>/dev/null)

if [ -z "$IMAGE_REF" ]; then
    echo "    WARNING: Could not get image ref, trying registry..."
    IMAGE_REF="registry.fly.io/$APP_NAME:latest"
fi

# Get volume ID
VOL_ID=$(fly volumes list --app "$APP_NAME" --json | python3 -c "
import json, sys
vols = json.load(sys.stdin)
for v in vols:
    if v['name'] == '$VOLUME_NAME':
        print(v['id'])
        break
")

# Destroy the stopped training machine first to free the volume
echo "    Freeing volume from stopped machine..."
fly machine list --app "$APP_NAME" --json 2>/dev/null | python3 -c "
import json, sys
machines = json.load(sys.stdin)
for m in machines:
    print(m['id'])
" 2>/dev/null | while read -r mid; do
    fly machine destroy "$mid" --app "$APP_NAME" --force 2>/dev/null || true
done
sleep 5

# Spin up a cheap temp machine to access the volume
echo "    Starting temp machine to download artifacts..."
fly machine run "$IMAGE_REF" \
    --app "$APP_NAME" \
    --region "$REGION" \
    --vm-size "shared-cpu-1x" \
    --volume "${VOL_ID}:/output" \
    --entrypoint "sleep 3600" \
    --detach 2>&1

echo "    Waiting for temp machine to start..."
sleep 15

# Download via ssh tar
fly ssh console \
    --app "$APP_NAME" \
    --select \
    --command "tar czf - -C /output artifacts" \
    > /tmp/dynasty_artifacts.tar.gz 2>/dev/null

# Extract locally
mkdir -p "$LOCAL_ARTIFACT_DIR"
tar xzf /tmp/dynasty_artifacts.tar.gz -C "$LOCAL_ARTIFACT_DIR" --strip-components=1 2>/dev/null || \
    tar xzf /tmp/dynasty_artifacts.tar.gz -C "$(dirname "$LOCAL_ARTIFACT_DIR")" 2>/dev/null
rm -f /tmp/dynasty_artifacts.tar.gz

# ---------- 6. Cleanup ----------
echo "[6/6] Cleaning up..."
fly machine list --app "$APP_NAME" --json 2>/dev/null | python3 -c "
import json, sys
machines = json.load(sys.stdin)
for m in machines:
    print(m['id'])
" 2>/dev/null | while read -r mid; do
    fly machine destroy "$mid" --app "$APP_NAME" --force 2>/dev/null || true
done

echo ""
echo "=== Complete ==="
echo "Artifacts: $LOCAL_ARTIFACT_DIR"
echo "Logs:      $LOG_FILE"
if [ -f "$LOCAL_ARTIFACT_DIR/metadata.json" ]; then
    echo ""
    echo "Metadata:"
    python3 -c "
import json
with open('$LOCAL_ARTIFACT_DIR/metadata.json') as f:
    m = json.load(f)
print(f'  Git SHA:    {m.get(\"git_sha\", \"unknown\")}')
print(f'  Timestamp:  {m.get(\"timestamp\", \"unknown\")}')
print(f'  Folds:      {m.get(\"walk_forward_folds\", [])}')
"
fi
