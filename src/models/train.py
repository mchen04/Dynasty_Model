import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from src.models.quantile_regressor import MultiTargetQuantilePool
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

    # Build sequences
    seq_array, pad_masks, seq_feat_cols, player_ids, row_map = build_player_sequences(
        df_train, max_seq_length=t_cfg["max_seq_len"]
    )

    num_features = seq_array.shape[2]
    model = TimeSeriesTransformer(
        num_features=num_features,
        embedding_dim=t_cfg["embedding_dim"],
    )

    # Simple reconstruction objective: predict last season's stats from sequence
    X_seq = torch.FloatTensor(seq_array)
    masks = torch.BoolTensor(pad_masks)

    # Target: last valid timestep features (self-supervised)
    targets = []
    for i in range(len(player_ids)):
        valid_positions = ~pad_masks[i]
        last_valid_idx = np.where(valid_positions)[0][-1]
        targets.append(seq_array[i, last_valid_idx, :t_cfg["embedding_dim"]])

    # Pad/truncate to embedding_dim
    target_dim = min(num_features, t_cfg["embedding_dim"])
    y_targets = torch.FloatTensor(np.array([t[:target_dim] for t in targets]))

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
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"    Transformer epoch {epoch+1}/{max_epochs}, loss: {avg_loss:.4f}")

    model.eval()
    return model, seq_feat_cols, player_ids


def _get_transformer_embeddings(model, df, config):
    """Extract transformer embeddings for all players in df."""
    t_cfg = config["model"]["transformer"]

    seq_array, pad_masks, feat_cols, player_ids, row_map = build_player_sequences(
        df, max_seq_length=t_cfg["max_seq_len"]
    )

    with torch.no_grad():
        X_seq = torch.FloatTensor(seq_array)
        masks = torch.BoolTensor(pad_masks)
        embeddings = model(X_seq, src_key_padding_mask=masks)

    return embeddings.numpy(), player_ids, row_map


def run_training():
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

        fit_df = train_df[train_df["SEASON"].isin(fit_seasons)]
        cal_df = train_df[train_df["SEASON"].isin(cal_seasons)]

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

        # 1. Train MultiTargetQuantilePool
        print("  Training quantile regressors...")
        pool = MultiTargetQuantilePool()

        # Optuna tuning on first fold only
        tune = (test_year == start_test_year)
        if tune:
            print("  (Tuning hyperparameters on first fold...)")
        pool.fit_all(X_fit, y_fit_targets, tune_first_only=tune)

        # 2. Calibrate with MAPIE
        print("  Calibrating with MAPIE...")
        pool.calibrate_all(X_cal, y_cal_targets)

        # 3. Train transformer
        print("  Training transformer...")
        try:
            transformer, seq_feats, train_pids = _train_transformer(
                fit_df, feature_cols, config, max_epochs=30
            )
        except Exception as e:
            print(f"  Transformer training failed: {e}")
            transformer = None

        # 4. Predict and evaluate on test set
        all_preds = pool.predict_all(X_test)

        # Evaluate T+1 predictions for key stats
        year_metrics = {"year": test_year}
        for stat in ["PTS", "REB", "AST", "MP"]:
            key = f"{stat}_T+1"
            if key in all_preds and key in y_test_targets:
                preds = all_preds[key]
                y_true = y_test_targets[key]
                valid = ~np.isnan(y_true)

                if valid.sum() > 0:
                    rmse = np.sqrt(mean_squared_error(y_true[valid], preds["median"][valid]))
                    mae = mean_absolute_error(y_true[valid], preds["median"][valid])
                    above_floor = np.mean(y_true[valid] >= preds["floor"][valid]) * 100
                    below_ceil = np.mean(y_true[valid] <= preds["ceiling"][valid]) * 100

                    year_metrics[f"{stat}_RMSE"] = rmse
                    year_metrics[f"{stat}_MAE"] = mae
                    year_metrics[f"{stat}_floor_cal"] = above_floor
                    year_metrics[f"{stat}_ceil_cal"] = below_ceil

                    print(f"    {stat} T+1: RMSE={rmse:.2f}, MAE={mae:.2f}, "
                          f"Floor Cal={above_floor:.1f}%, Ceil Cal={below_ceil:.1f}%")

        all_metrics.append(year_metrics)

    # Summary
    print("\n--- Walk-Forward Validation Complete ---")
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        for stat in ["PTS", "REB", "AST", "MP"]:
            rmse_col = f"{stat}_RMSE"
            if rmse_col in metrics_df.columns:
                print(f"  {stat} T+1 Average RMSE: {metrics_df[rmse_col].mean():.2f}")
                print(f"  {stat} T+1 Average Floor Cal: {metrics_df[f'{stat}_floor_cal'].mean():.1f}%")
                print(f"  {stat} T+1 Average Ceil Cal: {metrics_df[f'{stat}_ceil_cal'].mean():.1f}%")


if __name__ == "__main__":
    run_training()
