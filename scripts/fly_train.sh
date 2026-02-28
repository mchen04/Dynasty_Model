#!/usr/bin/env bash
#
# Orchestrates a full training run on Fly.io.
#
# Usage:
#   ./scripts/fly_train.sh              # Deploy + wait + download (keep terminal open)
#   ./scripts/fly_train.sh deploy       # Deploy only (safe to close laptop)
#   ./scripts/fly_train.sh status       # Check if training is still running
#   ./scripts/fly_train.sh download     # Download artifacts after training finishes
#
set -euo pipefail

APP_NAME="dynasty-model-train"
REGION="sjc"
VOLUME_NAME="model_artifacts"
VOLUME_SIZE_GB=5
LOCAL_ARTIFACT_DIR="./artifacts"
LOG_FILE="./artifacts/training.log"

COMMAND="${1:-all}"

# ============================================================
# Shared helpers
# ============================================================
_ensure_app() {
    if ! fly apps list --json 2>/dev/null | grep -q "\"$APP_NAME\""; then
        echo "[setup] Creating Fly app: $APP_NAME"
        fly apps create "$APP_NAME" --org personal
    else
        echo "[setup] App $APP_NAME already exists"
    fi
}

_ensure_volume() {
    EXISTING_VOL=$(fly volumes list --app "$APP_NAME" --json 2>/dev/null | grep -c "\"$VOLUME_NAME\"" || true)
    if [ "$EXISTING_VOL" -eq 0 ]; then
        echo "[setup] Creating ${VOLUME_SIZE_GB}GB volume: $VOLUME_NAME"
        fly volumes create "$VOLUME_NAME" \
            --app "$APP_NAME" \
            --region "$REGION" \
            --size "$VOLUME_SIZE_GB" \
            --yes
    else
        echo "[setup] Volume $VOLUME_NAME already exists"
    fi
}

_is_running() {
    fly machine list --app "$APP_NAME" --json 2>/dev/null | python3 -c "
import json, sys
machines = json.load(sys.stdin)
running = [m for m in machines if m.get('state') in ('created', 'starting', 'started')]
print(len(running))
" 2>/dev/null || echo "0"
}

_wait_for_completion() {
    echo "Waiting for training to complete..."
    echo "(You can Ctrl-C here and run './scripts/fly_train.sh status' later)"
    echo ""
    while true; do
        RUNNING=$(_is_running)
        if [ "$RUNNING" -eq 0 ]; then
            echo "All machines stopped — training complete."
            break
        fi
        echo "  $(date '+%H:%M:%S') $RUNNING machine(s) still running..."
        sleep 30
    done
}

_download_logs() {
    mkdir -p "$(dirname "$LOG_FILE")"
    fly logs --app "$APP_NAME" --no-tail 2>/dev/null > "$LOG_FILE" || true
    LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
    echo "Saved $LINES log lines to $LOG_FILE"
    echo ""
    echo "--- Last 30 lines of training output ---"
    tail -30 "$LOG_FILE" 2>/dev/null || true
    echo "---"
}

_download_artifacts() {
    echo "Downloading artifacts from volume..."

    # Get image ref from the deployed machine
    IMAGE_REF=$(fly machine list --app "$APP_NAME" --json 2>/dev/null | python3 -c "
import json, sys
machines = json.load(sys.stdin)
if machines:
    print(machines[0].get('config', {}).get('image', ''))
" 2>/dev/null)

    if [ -z "$IMAGE_REF" ]; then
        echo "  WARNING: Could not get image ref, trying registry..."
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

    # Destroy stopped machines to free the volume
    echo "  Freeing volume from stopped machine..."
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
    echo "  Starting temp machine to download artifacts..."
    fly machine run "$IMAGE_REF" \
        --app "$APP_NAME" \
        --region "$REGION" \
        --vm-size "shared-cpu-1x" \
        --volume "${VOL_ID}:/output" \
        --entrypoint "sleep 3600" \
        --detach 2>&1

    echo "  Waiting for temp machine to start..."
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
}

_cleanup() {
    echo "Cleaning up temp machines..."
    fly machine list --app "$APP_NAME" --json 2>/dev/null | python3 -c "
import json, sys
machines = json.load(sys.stdin)
for m in machines:
    print(m['id'])
" 2>/dev/null | while read -r mid; do
        fly machine destroy "$mid" --app "$APP_NAME" --force 2>/dev/null || true
    done
}

_print_summary() {
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
}

# ============================================================
# Commands
# ============================================================
case "$COMMAND" in
    deploy)
        echo "=== Dynasty Model — Fly.io Deploy ==="
        _ensure_app
        _ensure_volume
        echo "[deploy] Building image and deploying..."
        fly deploy --app "$APP_NAME" --ha=false
        echo ""
        echo "Training is running on Fly.io. Safe to close your laptop."
        echo "Check progress:    ./scripts/fly_train.sh status"
        echo "Download results:  ./scripts/fly_train.sh download"
        ;;

    status)
        RUNNING=$(_is_running)
        if [ "$RUNNING" -eq 0 ]; then
            echo "Training complete (no machines running)."
            echo "Run './scripts/fly_train.sh download' to get artifacts."
        else
            echo "$RUNNING machine(s) still running."
            echo ""
            echo "--- Recent logs ---"
            fly logs --app "$APP_NAME" --no-tail 2>/dev/null | tail -20
        fi
        ;;

    download)
        echo "=== Dynasty Model — Download Artifacts ==="
        RUNNING=$(_is_running)
        if [ "$RUNNING" -gt 0 ]; then
            echo "WARNING: $RUNNING machine(s) still running. Training may not be done."
            read -p "Download anyway? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 0
            fi
        fi
        _download_logs
        _download_artifacts
        _cleanup
        _print_summary
        ;;

    all)
        echo "=== Dynasty Model — Fly.io Training ==="
        _ensure_app
        _ensure_volume
        echo "[1/4] Building image and deploying..."
        fly deploy --app "$APP_NAME" --ha=false
        echo ""
        echo "[2/4] Waiting for training..."
        _wait_for_completion
        echo ""
        echo "[3/4] Downloading results..."
        _download_logs
        _download_artifacts
        echo ""
        echo "[4/4] Cleaning up..."
        _cleanup
        _print_summary
        ;;

    *)
        echo "Usage: $0 {deploy|status|download|all}"
        echo ""
        echo "  deploy    Build + deploy to Fly.io (safe to close laptop after)"
        echo "  status    Check if training is still running"
        echo "  download  Download artifacts + logs after training finishes"
        echo "  all       Deploy + wait + download (keep terminal open)"
        exit 1
        ;;
esac
