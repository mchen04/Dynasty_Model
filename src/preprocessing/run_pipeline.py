import numpy as np
import pandas as pd

from src.preprocessing.advanced_feature_engineer import engineer_advanced_features
from src.preprocessing.imputer import BackcastImputer
from src.preprocessing.network_effects import apply_network_effects, calculate_teammate_efficiency
from src.preprocessing.zscores import calculate_era_adjusted_zscores
from src.utils.config import load_config, resolve_path
from src.utils.data_loader import load_raw_dataset


def run():
    print("Starting Preprocessing Pipeline...")
    config = load_config()

    raw_path = resolve_path(config["project"]["raw_data"])
    processed_path = resolve_path(config["project"]["processed_data"])

    print(f"Loading raw data from {raw_path}")
    df = load_raw_dataset(str(raw_path))
    print(f"Raw shape: {df.shape}")

    # 1. Advanced Features (Experience, Injuries, Combine, Coaches, Age, Momentum)
    print("\n--- Step 1: Advanced Feature Engineering ---")
    df = engineer_advanced_features(df)

    df["SEASON"] = df["SEASON"].astype(int)

    # 2. Z-scores (Era Adjustment)
    print("\n--- Step 2: Era Adjustment (Z-Scores) ---")
    zscore_stats = config["preprocessing"]["zscore_stats"]
    # Only z-score stats that actually exist in the dataframe
    available_zscore_stats = [s for s in zscore_stats if s in df.columns]
    df = calculate_era_adjusted_zscores(df, available_zscore_stats, season_col="SEASON")

    # 3. Multi-metric Imputation (POTENTIAL_AST, PASSES_MADE, AST_POINTS_CREATED)
    print("\n--- Step 3: Backcast Imputation ---")
    imputation_targets = config["preprocessing"]["imputation_targets"]
    for target in imputation_targets:
        if target not in df.columns:
            print(f"  {target} column not found in dataset, skipping imputation.")
            continue
        if df[target].isna().all():
            print(f"  {target} is entirely NaN, skipping imputation.")
            continue
        try:
            imputer = BackcastImputer(target_metric=target)
            imputer.train(df)
            df = imputer.impute(df)
        except ValueError as e:
            print(f"  Skipping imputation for {target}: {e}")

    # 4. Network Effects (Usage Vacuums and Teammate Efficiency)
    print("\n--- Step 4: Network Effects ---")
    # Ensure required columns exist
    team_col = "TEAM" if "TEAM" in df.columns else "Tm"
    if "TEAM_ID" not in df.columns:
        df["TEAM_ID"] = df[team_col]
    if "PLAYER_ID" not in df.columns:
        df["PLAYER_ID"] = df.groupby("Player").ngroup()
    if "USG_PCT" not in df.columns:
        df["USG_PCT"] = df.get("USG%", 20.0)
    if "TS_PCT" not in df.columns:
        df["TS_PCT"] = df.get("TS%", 0.55)
    if "MIN" not in df.columns:
        df["MIN"] = df.get("MP", 20.0)

    seasons = sorted(df["SEASON"].unique())
    network_dfs = []

    for i, season_t in enumerate(seasons):
        df_season_t = df[df["SEASON"] == season_t].copy()

        if i == 0:
            df_season_t["VACATED_USAGE"] = 0.0
            teammate_ts = df_season_t.apply(
                lambda row: calculate_teammate_efficiency(
                    row, df_season_t[df_season_t["TEAM_ID"] == row["TEAM_ID"]]
                ),
                axis=1,
            )
            df_season_t["TEAMMATE_TS_PCT"] = teammate_ts
            network_dfs.append(df_season_t)
        else:
            season_t_minus_1 = seasons[i - 1]
            df_season_t_minus_1 = df[df["SEASON"] == season_t_minus_1]
            enhanced_season_t = apply_network_effects(df_season_t_minus_1, df_season_t)
            network_dfs.append(enhanced_season_t)

    df_final = pd.concat(network_dfs, ignore_index=True)

    # Save
    import os
    os.makedirs(processed_path.parent, exist_ok=True)
    print(f"\nSaving processed dataset to {processed_path}")
    df_final.to_csv(processed_path, index=False)
    print(f"Final shape: {df_final.shape}")
    print("Preprocessing complete!")


if __name__ == "__main__":
    run()
