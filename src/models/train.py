import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from src.models.quantile_regressor import MultiTargetQuantilePool
from src.models.serialization import ensure_dir, save_metadata
from src.models.transformer import TimeSeriesTransformer
from src.models.ensembler import Stage1Ensembler
from src.preprocessing.sequence_builder import build_player_sequences
from src.utils.config import load_config, resolve_path


def create_targets(df, config=None):
    """
    Creates T+h targets for all stats and horizons by shifting within player groups.
    NaN targets are left as NaN (represent career endings).
    """
    if config is None:
        config = load_config()

    target_stats = config["model"]["target_stats"]
    horizons = config["model"]["horizons"]

    print(f"Creating multi-horizon targets: {len(target_stats)} stats x {len(horizons)} horizons...")
    df = df.sort_values(by=["PLAYER_ID", "SEASON"])

    for stat in target_stats:
        if stat not in df.columns:
            continue
        for h in horizons:
            col_name = f"{stat}_T+{h}"
            df[col_name] = df.groupby("PLAYER_ID")[stat].shift(-h)

    return df


def _get_feature_cols(df, config):
    """Get feature columns (exclude targets, IDs, string cols)."""
    target_cols = []
    for stat in config["model"]["target_stats"]:
        for h in config["model"]["horizons"]:
            target_cols.append(f"{stat}_T+{h}")

    id_cols = ["PLAYER_ID", "SEASON", "PLAYER_NAME", "TEAM_ID", "Tm", "Player", "TEAM"]
    exclude = set(target_cols + id_cols)

    # Only use numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


def _train_transformer(
    df_train, feature_cols, config, max_epochs=50, lr=1e-3, batch_size=64
):
    """Train the time-series transformer on player sequences."""
    t_cfg = config["model"]["transformer"]

    # Build sequences — pass feature_cols to avoid target-column leakage
    seq_array, pad_masks, seq_feat_cols, player_ids, row_map = build_player_sequences(
        df_train, max_seq_length=t_cfg["max_seq_len"], feature_cols=feature_cols,
    )

    num_features = seq_array.shape[2]
    model = TimeSeriesTransformer(
        num_features=num_features,
        embedding_dim=t_cfg["embedding_dim"],
    )

    # Simple reconstruction objective: predict last season's stats from sequence
    X_seq = torch.FloatTensor(seq_array)
    X_seq = torch.nan_to_num(X_seq, nan=0.0)  # replace NaN features for transformer

    # Standardize features to prevent gradient explosion (compute from valid positions)
    valid_positions = ~torch.BoolTensor(pad_masks)  # True = valid
    flat_valid = X_seq[valid_positions]  # (total_valid_timesteps, num_features)
    feat_mean = flat_valid.mean(dim=0)
    feat_std = flat_valid.std(dim=0).clamp(min=1e-6)
    X_seq = (X_seq - feat_mean) / feat_std

    masks = torch.BoolTensor(pad_masks)

    # Target: last valid timestep features (self-supervised, also standardized)
    # Skip all-padded players (no valid positions)
    target_dim = min(num_features, t_cfg["embedding_dim"])
    targets = []
    valid_player_indices = []
    for i in range(len(player_ids)):
        valid_pos = ~pad_masks[i]
        valid_indices = np.where(valid_pos)[0]
        if len(valid_indices) == 0:
            continue  # skip all-padded players
        last_valid_idx = valid_indices[-1]
        # Use standardized values for targets to match input scale
        raw = torch.FloatTensor(seq_array[i, last_valid_idx, :target_dim])
        raw = torch.nan_to_num(raw, nan=0.0)
        standardized = (raw - feat_mean[:target_dim]) / feat_std[:target_dim]
        targets.append(standardized.numpy())
        valid_player_indices.append(i)

    if len(valid_player_indices) == 0:
        print("    No valid players for transformer training, skipping.")
        return None, [], []

    # Filter to only valid players
    valid_player_indices = np.array(valid_player_indices)
    X_seq = X_seq[valid_player_indices]
    masks = masks[valid_player_indices]
    player_ids = [player_ids[i] for i in valid_player_indices]

    y_targets = torch.FloatTensor(np.array(targets))

    # Projection from embedding to target reconstruction
    recon_head = nn.Linear(t_cfg["embedding_dim"], target_dim)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(recon_head.parameters()), lr=lr
    )
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(X_seq, masks, y_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(max_epochs):
        total_loss = 0
        for batch_x, batch_mask, batch_y in loader:
            optimizer.zero_grad()
            embeddings = model(batch_x, src_key_padding_mask=batch_mask)
            recon = recon_head(embeddings)
            loss = loss_fn(recon, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(recon_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"    Transformer epoch {epoch+1}/{max_epochs}, loss: {avg_loss:.4f}")

    model.eval()
    return model, seq_feat_cols, player_ids, (feat_mean, feat_std)


def _get_transformer_embeddings(model, df, config, feature_cols=None,
                                norm_stats=None):
    """Extract transformer embeddings for all players in df.

    Uses the full player history present in *df* to build multi-season
    sequences.  Callers should pass a context DataFrame that includes all
    seasons up to (and including) the target season so that each player gets
    their complete career trajectory, not just one season.

    norm_stats: (feat_mean, feat_std) tensors from training, used to apply
    the same standardization.
    """
    t_cfg = config["model"]["transformer"]

    seq_array, pad_masks, feat_cols, player_ids, row_map = build_player_sequences(
        df, max_seq_length=t_cfg["max_seq_len"], feature_cols=feature_cols,
    )

    with torch.no_grad():
        X_seq = torch.nan_to_num(torch.FloatTensor(seq_array), nan=0.0)
        if norm_stats is not None:
            feat_mean, feat_std = norm_stats
            X_seq = (X_seq - feat_mean) / feat_std
        masks = torch.BoolTensor(pad_masks)
        embeddings = model(X_seq, src_key_padding_mask=masks)

    return embeddings.numpy(), player_ids, row_map


def _get_git_sha():
    """Get current git SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_versions():
    """Capture dependency versions for reproducibility."""
    import lightgbm
    versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import mapie
        versions["mapie"] = mapie.__version__
    except Exception:
        pass
    try:
        import optuna
        versions["optuna"] = optuna.__version__
    except Exception:
        pass
    return versions


def _train_ensembler(transformer, pool, cal_df, feature_cols, y_cal_targets,
                     config, context_df=None, norm_stats=None,
                     max_epochs=300, lr=1e-3, patience=30):
    """Train Stage1Ensembler on calibration set using LightGBM preds + transformer embeddings.

    context_df: full history DataFrame (all seasons up to cal) so the
    transformer gets multi-season sequences for each player.
    norm_stats: (feat_mean, feat_std) from transformer training.
    """
    target_stats = config["model"]["target_stats"]
    num_stats = len(target_stats)

    # Get LightGBM T+1 predictions for cal set
    X_cal = cal_df[feature_cols].values
    lgbm_preds = pool.predict_all(X_cal)

    # Get transformer embeddings using full history (one per player)
    emb_df = context_df if context_df is not None else cal_df
    cal_embeddings, cal_pids, _ = _get_transformer_embeddings(
        transformer, emb_df, config, feature_cols=feature_cols,
        norm_stats=norm_stats,
    )
    pid_to_emb = {pid: cal_embeddings[i] for i, pid in enumerate(cal_pids)}

    # Match each cal row to its player's embedding
    cal_player_ids = cal_df["PLAYER_ID"].values

    tree_preds_list = []
    emb_list = []
    target_list = []

    for row_idx, pid in enumerate(cal_player_ids):
        if pid not in pid_to_emb:
            continue

        tree_pred = []
        target_vals = []
        skip = False
        for stat in target_stats:
            key = f"{stat}_T+1"
            if key not in lgbm_preds or key not in pool.pools or not pool.pools[key].is_trained:
                skip = True
                break
            preds = lgbm_preds[key]
            tree_pred.append([preds["floor"][row_idx], preds["median"][row_idx],
                              preds["ceiling"][row_idx]])
            target_vals.append(y_cal_targets.get(key, np.full(len(cal_df), np.nan))[row_idx])

        if skip:
            continue

        tree_preds_list.append(tree_pred)
        emb_list.append(pid_to_emb[pid])
        target_list.append(target_vals)

    if len(tree_preds_list) < 20:
        print("    Too few matched samples for ensembler training.")
        return None

    tree_tensor = torch.FloatTensor(np.array(tree_preds_list))   # (n, num_stats, 3)
    emb_tensor = torch.FloatTensor(np.array(emb_list))           # (n, embedding_dim)
    target_tensor = torch.FloatTensor(np.array(target_list))     # (n, num_stats)

    ensembler = Stage1Ensembler(num_stats=num_stats)
    optimizer = torch.optim.AdamW(ensembler.parameters(), lr=lr)

    ensembler.train()
    best_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        output = ensembler(tree_tensor, emb_tensor)  # (n, num_stats, 3)

        # Loss on median predictions, masking NaN targets
        pred_median = output[:, :, 1]
        mask = ~torch.isnan(target_tensor)
        if mask.sum() == 0:
            break
        loss = nn.functional.mse_loss(pred_median[mask], target_tensor[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ensembler.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss - 1e-4:
            best_loss = current_loss
            best_state = {k: v.clone() for k, v in ensembler.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 50 == 0:
            print(f"    Ensembler epoch {epoch+1}/{max_epochs}, loss: {current_loss:.4f}")

        if epochs_no_improve >= patience:
            print(f"    Ensembler early stop at epoch {epoch+1}, best loss: {best_loss:.4f}")
            break

    if best_state is not None:
        ensembler.load_state_dict(best_state)
    ensembler.eval()
    return ensembler


def _ensemble_predictions(ensembler, transformer, test_df, feature_cols,
                          lgbm_preds, config, context_df=None,
                          norm_stats=None):
    """Blend T+1 LightGBM predictions with transformer embeddings via ensembler.

    Other horizons pass through unchanged.

    context_df: full history DataFrame (all seasons up to test year) so the
    transformer gets multi-season sequences.
    norm_stats: (feat_mean, feat_std) from transformer training.
    """
    target_stats = config["model"]["target_stats"]
    num_stats = len(target_stats)
    emb_dim = config["model"]["transformer"]["embedding_dim"]

    # Get transformer embeddings using full history
    emb_df = context_df if context_df is not None else test_df
    test_embeddings, test_pids, _ = _get_transformer_embeddings(
        transformer, emb_df, config, feature_cols=feature_cols,
        norm_stats=norm_stats,
    )

    # NaN guard: if embeddings are bad, fall back to LightGBM
    if np.isnan(test_embeddings).any():
        nan_pct = np.isnan(test_embeddings).mean() * 100
        print(f"  WARNING: {nan_pct:.0f}% NaN in transformer embeddings, using LightGBM only")
        return lgbm_preds

    pid_to_emb = {pid: test_embeddings[i] for i, pid in enumerate(test_pids)}

    test_player_ids = test_df["PLAYER_ID"].values
    n_rows = len(test_player_ids)

    # Build tree_preds tensor and match embeddings
    tree_preds = np.zeros((n_rows, num_stats, 3))
    emb_array = np.zeros((n_rows, emb_dim))
    has_embedding = np.zeros(n_rows, dtype=bool)

    for stat_idx, stat in enumerate(target_stats):
        key = f"{stat}_T+1"
        if key in lgbm_preds:
            tree_preds[:, stat_idx, 0] = lgbm_preds[key]["floor"]
            tree_preds[:, stat_idx, 1] = lgbm_preds[key]["median"]
            tree_preds[:, stat_idx, 2] = lgbm_preds[key]["ceiling"]

    for row_idx, pid in enumerate(test_player_ids):
        if pid in pid_to_emb:
            emb_array[row_idx] = pid_to_emb[pid]
            has_embedding[row_idx] = True

    if has_embedding.sum() == 0:
        return lgbm_preds

    with torch.no_grad():
        tree_tensor = torch.FloatTensor(tree_preds[has_embedding])
        emb_tensor = torch.FloatTensor(emb_array[has_embedding])
        ensembled = ensembler(tree_tensor, emb_tensor).numpy()  # (n_matched, num_stats, 3)

    # Write ensembled T+1 predictions back
    ensembled_preds = {}
    for key, val in lgbm_preds.items():
        ensembled_preds[key] = {k: v.copy() for k, v in val.items()}

    for stat_idx, stat in enumerate(target_stats):
        key = f"{stat}_T+1"
        if key not in ensembled_preds:
            continue
        ensembled_preds[key]["floor"][has_embedding] = ensembled[:, stat_idx, 0]
        ensembled_preds[key]["median"][has_embedding] = ensembled[:, stat_idx, 1]
        ensembled_preds[key]["ceiling"][has_embedding] = ensembled[:, stat_idx, 2]

    return ensembled_preds


def run_training(save_artifacts=False, artifact_dir="artifacts"):
    print("Starting Walk-Forward Cross Validation Pipeline...")
    config = load_config()

    processed_path = resolve_path(config["project"]["processed_data"])
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_path}. Run preprocessing pipeline first."
        )

    df = pd.read_csv(processed_path)
    print(f"Loaded processed data, shape: {df.shape}")

    # Ensure PLAYER_ID exists
    if "PLAYER_ID" not in df.columns:
        df["PLAYER_ID"] = df.groupby("Player").ngroup() if "Player" in df.columns else range(len(df))

    # Drop non-numeric columns except IDs
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    id_cols_to_keep = {"PLAYER_ID", "SEASON"}
    drop_str_cols = [c for c in str_cols if c not in id_cols_to_keep]
    df = df.drop(columns=drop_str_cols, errors="ignore")

    # DO NOT fill NaN with 0 — LightGBM handles NaN natively
    df = create_targets(df, config)

    feature_cols = _get_feature_cols(df, config)
    print(f"Using {len(feature_cols)} features.")

    # Walk-forward split parameters
    min_year = int(df["SEASON"].min())
    max_year = int(df["SEASON"].max())
    initial_train_years = config["model"]["initial_train_years"]
    start_test_year = min_year + initial_train_years

    if start_test_year >= max_year:
        start_test_year = max_year - 3

    print(f"\n--- Walk-Forward Validation (Testing {start_test_year} to {max_year}) ---")

    all_metrics = []
    tuned_group_params = None  # persisted across folds

    for test_year in range(start_test_year, max_year + 1):
        print(f"\n=== Evaluating Season: {test_year} ===")

        train_df = df[df["SEASON"] < test_year].copy()
        test_df = df[df["SEASON"] == test_year].copy()

        if len(test_df) == 0:
            print(f"  No test data for {test_year}, skipping.")
            continue

        # Split train into fit + calibration (last ~15% of training seasons)
        train_seasons = sorted(train_df["SEASON"].unique())
        cal_split = max(1, int(len(train_seasons) * 0.15))
        cal_seasons = train_seasons[-cal_split:]
        fit_seasons = train_seasons[:-cal_split]

        fit_df = train_df[train_df["SEASON"].isin(fit_seasons)].sort_values("SEASON")
        cal_df = train_df[train_df["SEASON"].isin(cal_seasons)].sort_values("SEASON")

        print(f"  Fit: {len(fit_df)} rows ({len(fit_seasons)} seasons)")
        print(f"  Cal: {len(cal_df)} rows ({len(cal_seasons)} seasons)")
        print(f"  Test: {len(test_df)} rows")

        X_fit = fit_df[feature_cols].values
        X_cal = cal_df[feature_cols].values
        X_test = test_df[feature_cols].values

        # Build target dicts
        y_fit_targets = {}
        y_cal_targets = {}
        y_test_targets = {}

        for stat in config["model"]["target_stats"]:
            for h in config["model"]["horizons"]:
                key = f"{stat}_T+{h}"
                if key in fit_df.columns:
                    y_fit_targets[key] = fit_df[key].values
                    y_cal_targets[key] = cal_df[key].values
                    y_test_targets[key] = test_df[key].values

        # Compute sample weights from games played
        fit_weights = None
        if "G" in fit_df.columns:
            fit_weights = np.clip(fit_df["G"].values / 82.0, 0.1, 1.0)

        # 1. Train MultiTargetQuantilePool
        print("  Training quantile regressors...")
        pool = MultiTargetQuantilePool()

        # Optuna tuning on first fold only, reuse params on subsequent folds
        tune = (test_year == start_test_year)
        if tune:
            print("  (Tuning hyperparameters on first fold...)")
        tuned_group_params = pool.fit_all(
            X_fit, y_fit_targets, tune_first_only=tune,
            sample_weight=fit_weights, group_params=tuned_group_params,
        )

        # 2. Calibrate with MAPIE
        print("  Calibrating with MAPIE...")
        pool.calibrate_all(X_cal, y_cal_targets)

        # 3. Train transformer on full training history
        print("  Training transformer...")
        transformer = None
        norm_stats = None
        try:
            transformer, seq_feats, train_pids, norm_stats = _train_transformer(
                train_df, feature_cols, config, max_epochs=30
            )
        except Exception as e:
            print(f"  Transformer training failed: {e}")

        # 4. Train ensembler on calibration set (full train history for embeddings)
        ensembler = None
        if transformer is not None:
            print("  Training ensembler...")
            try:
                ensembler = _train_ensembler(
                    transformer, pool, cal_df, feature_cols, y_cal_targets, config,
                    context_df=train_df, norm_stats=norm_stats,
                )
            except Exception as e:
                print(f"  Ensembler training failed: {e}")

        # Save fold artifacts
        if save_artifacts:
            fold_dir = os.path.join(artifact_dir, "models", f"fold_{test_year}")
            ensure_dir(fold_dir)
            print(f"  Saving fold artifacts to {fold_dir}...")
            pool.save_all(fold_dir)
            if transformer is not None:
                transformer.save(os.path.join(fold_dir, "transformer.pt"))
                if norm_stats is not None:
                    torch.save(
                        {"mean": norm_stats[0], "std": norm_stats[1]},
                        os.path.join(fold_dir, "norm_stats.pt"),
                    )
            if ensembler is not None:
                ensembler.save(os.path.join(fold_dir, "ensembler.pt"))

        # 5. Predict on test set (LightGBM baseline)
        all_preds = pool.predict_all(X_test)

        # Ensemble T+1 predictions if ensembler available
        if ensembler is not None:
            print("  Ensembling predictions...")
            # Use all data up to and including test year for full player histories
            context_df = df[df["SEASON"] <= test_year].copy()
            try:
                all_preds = _ensemble_predictions(
                    ensembler, transformer, test_df, feature_cols, all_preds, config,
                    context_df=context_df, norm_stats=norm_stats,
                )
            except Exception as e:
                print(f"  Ensemble prediction failed, using LightGBM only: {e}")

        # 6. Evaluate all stats x all horizons
        year_metrics = {"year": test_year}

        eval_pairs = [
            (stat, h)
            for stat in config["model"]["target_stats"]
            for h in config["model"]["horizons"]
        ]

        key_print_stats = {"PTS", "REB", "AST", "MP", "STL", "BLK", "TOV"}
        for stat, h in eval_pairs:
            key = f"{stat}_T+{h}"
            if key in all_preds and key in y_test_targets:
                preds = all_preds[key]
                y_true = y_test_targets[key]
                y_pred_median = preds["median"]
                valid = ~np.isnan(y_true) & ~np.isnan(y_pred_median)

                if valid.sum() > 0:
                    rmse = np.sqrt(mean_squared_error(y_true[valid], preds["median"][valid]))
                    mae = mean_absolute_error(y_true[valid], preds["median"][valid])
                    above_floor = np.mean(y_true[valid] >= preds["floor"][valid]) * 100
                    below_ceil = np.mean(y_true[valid] <= preds["ceiling"][valid]) * 100

                    year_metrics[f"{stat}_T+{h}_RMSE"] = rmse
                    year_metrics[f"{stat}_T+{h}_MAE"] = mae
                    year_metrics[f"{stat}_T+{h}_floor_cal"] = above_floor
                    year_metrics[f"{stat}_T+{h}_ceil_cal"] = below_ceil

                    # Only print key stats to avoid log spam
                    if stat in key_print_stats and h == 1:
                        print(f"    {stat} T+{h}: RMSE={rmse:.2f}, MAE={mae:.2f}, "
                              f"Floor Cal={above_floor:.1f}%, Ceil Cal={below_ceil:.1f}%")

        all_metrics.append(year_metrics)

    # Summary
    print("\n--- Walk-Forward Validation Complete ---")
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)

        # Print key stats summary at T+1
        print("\n  T+1 Averages:")
        for stat in config["model"]["target_stats"]:
            rmse_col = f"{stat}_T+1_RMSE"
            if rmse_col in metrics_df.columns:
                avg_rmse = metrics_df[rmse_col].mean()
                avg_floor = metrics_df[f"{stat}_T+1_floor_cal"].mean()
                avg_ceil = metrics_df[f"{stat}_T+1_ceil_cal"].mean()
                print(f"    {stat:>3s}: RMSE={avg_rmse:.2f}, "
                      f"Floor Cal={avg_floor:.1f}%, Ceil Cal={avg_ceil:.1f}%")

        # Print longer-horizon summary
        for h in config["model"]["horizons"]:
            if h == 1:
                continue
            has_any = False
            for stat in config["model"]["target_stats"]:
                rmse_col = f"{stat}_T+{h}_RMSE"
                if rmse_col in metrics_df.columns:
                    if not has_any:
                        print(f"\n  T+{h} Averages:")
                        has_any = True
                    avg_rmse = metrics_df[rmse_col].mean()
                    print(f"    {stat:>3s}: RMSE={avg_rmse:.2f}")

    # Save metrics and metadata
    if save_artifacts:
        metrics_dir = os.path.join(artifact_dir, "metrics")
        ensure_dir(metrics_dir)
        metrics_path = os.path.join(metrics_dir, "walk_forward_results.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"\nMetrics saved to {metrics_path}")

        save_metadata(artifact_dir, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _get_git_sha(),
            "versions": _get_versions(),
            "feature_cols": feature_cols,
            "config_snapshot": config,
            "walk_forward_folds": list(range(start_test_year, max_year + 1)),
        })
        print(f"Metadata saved to {os.path.join(artifact_dir, 'metadata.json')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Dynasty Model training pipeline")
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        default=False,
        help="Save trained model artifacts to disk",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts",
        help="Directory to save artifacts (default: artifacts/)",
    )
    return parser.parse_args()


class _TeeWriter:
    """Write to both stdout and a log file."""
    def __init__(self, log_path):
        self._stdout = sys.stdout
        ensure_dir(os.path.dirname(log_path))
        self._file = open(log_path, "w")

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._stdout


if __name__ == "__main__":
    args = parse_args()

    # Tee stdout to a log file inside the artifact dir
    tee = None
    if args.save_artifacts:
        log_path = os.path.join(args.artifact_dir, "training.log")
        tee = _TeeWriter(log_path)
        sys.stdout = tee

    try:
        run_training(
            save_artifacts=args.save_artifacts,
            artifact_dir=args.artifact_dir,
        )
    finally:
        if tee is not None:
            tee.close()
