import pandas as pd
import numpy as np


def calculate_era_adjusted_zscores(df, stat_columns, season_col="SEASON"):
    """
    Era Adjustment Logic (Standardization per Season).

    10 assists in 2004 operates in a fundamentally different ecosystem than
    10 assists in 2025. By converting stats to 'within-season z-scores',
    the model seamlessly handles pace-inflation and the 3-point revolution.

    Z = (Player Stat - Season Mean) / Season Standard Deviation
    """
    print(
        f"Calculating Era-Adjusted Z-Scores for {len(stat_columns)} stats across {df[season_col].nunique()} seasons..."
    )

    df_adjusted = df.copy()

    for stat in stat_columns:
        if stat not in df.columns:
            continue

        # We calculate the mean and standard dev restricted to the specific season.
        # This groups the players against only their actual contemporaries.
        z_col_name = f"Z_{stat}"

        # Using transform allows us to broadcast the z-score calculation straight back
        # to the same shape as the original dataframe.
        df_adjusted[z_col_name] = df_adjusted.groupby(season_col)[stat].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1)
        )

        # Handle cases where std is 0 (e.g., a stat that nobody recorded, or a season with 1 player)
        df_adjusted[z_col_name] = df_adjusted[z_col_name].fillna(0.0)

    return df_adjusted


def normalize_trade_value(vorp_series, premium=1.2):
    """
    Stage 2 Valuations Method preview:
    Normalizes VORP (Value Over Replacement Player) using the non-linear power curve.
    This creates higher valuation multipliers for top-end assets due to roster limits (2-for-1 trades).
    """
    # Ensure VORP is minimum 0
    safe_vorp = np.maximum(0, vorp_series)

    # Apply consolidation premium (power curve)
    normalized_vorp = safe_vorp**premium

    # Scale to 0-100 where the max VORP player in the league = 100
    max_vorp = normalized_vorp.max()
    if max_vorp > 0:
        trade_value = 100 * (normalized_vorp / max_vorp)
    else:
        trade_value = pd.Series(0, index=vorp_series.index)

    return trade_value.round(2)


if __name__ == "__main__":
    print("Testing Era Adjustment (Z-Score) logic...")

    mock_data = pd.DataFrame(
        {
            "PLAYER_NAME": ["Iverson", "Kobe", "Nash", "Harden", "Curry", "Luka"],
            "SEASON": [2005, 2005, 2005, 2020, 2020, 2020],
            "PTS": [30.7, 27.6, 15.5, 34.3, 20.8, 28.8],
            "3PM": [1.4, 2.0, 1.3, 4.4, 2.4, 2.8],  # Deep discrepancy across eras
            "PACE": [90.5, 90.5, 90.5, 100.2, 100.2, 100.2],
        }
    )

    stats_to_adjust = ["PTS", "3PM"]

    era_adjusted = calculate_era_adjusted_zscores(mock_data, stats_to_adjust)

    print("\nOriginal vs Z-Scored Stats (Notice how Harden in 2020 getting 4.4 3s")
    print("is less standard deviations above his peers than you might think):")
    print(era_adjusted[["PLAYER_NAME", "SEASON", "3PM", "Z_3PM", "PTS", "Z_PTS"]])

    print("\nTesting Stage 2 Trade Normalization (Power Curve):")
    vorp_mock = pd.Series([-1.5, 0.5, 2.0, 5.0, 9.5], name="raw_VORP")
    trade_values = normalize_trade_value(vorp_mock, premium=1.2)

    val_table = pd.DataFrame({"Raw_VORP": vorp_mock, "Trade_Value_0_100": trade_values})
    print(val_table)
