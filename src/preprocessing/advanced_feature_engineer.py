import io
import re
import time

import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import draftcombinestats

from src.utils.config import load_config, get_season_games
from src.utils.constants import BBR_TEAM_MAPPING


def _calculate_experience(df):
    """Years since a player's first NBA season."""
    min_season = df.groupby("Player")["SEASON"].transform("min")
    df["EXPERIENCE"] = df["SEASON"] - min_season
    return df


def _calculate_injury_history(df, config):
    """Games missed and played percentage using correct season lengths."""
    season_games_map = config["preprocessing"]["season_games"]

    df["SEASON_GAMES"] = df["SEASON"].map(
        lambda s: season_games_map.get(s, season_games_map["default"])
    )
    df["MISSED_GAMES"] = np.maximum(df["SEASON_GAMES"] - df["G"], 0)
    df["GAMES_PLAYED_PCT"] = df["G"] / df["SEASON_GAMES"]
    df = df.drop(columns=["SEASON_GAMES"])
    return df


def _calculate_rolling_injury_features(df):
    """Rolling injury features with proper lag to prevent leakage."""
    df = df.sort_values(["Player", "SEASON"]).copy()

    # Shifted (lagged) missed games from prior seasons
    df["MISSED_GAMES_LAST_1"] = df.groupby("Player")["MISSED_GAMES"].shift(1)
    df["MISSED_GAMES_LAST_2"] = (
        df.groupby("Player")["MISSED_GAMES"]
        .rolling(2, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .groupby(df["Player"])
        .shift(1)
    )
    df["MISSED_GAMES_LAST_3"] = (
        df.groupby("Player")["MISSED_GAMES"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .groupby(df["Player"])
        .shift(1)
    )
    df["INJURY_PRONE_FLAG"] = (df["MISSED_GAMES_LAST_3"] >= 40).astype(int)
    return df


def _calculate_age_features(df, config):
    """Age-related features including peak distance and contract year proxy."""
    peak_ages = config["preprocessing"]["peak_ages"]
    contract_years = config["preprocessing"]["contract_years"]

    age_col = "AGE" if "AGE" in df.columns else "Age"
    df["AGE_SQUARED"] = df[age_col] ** 2

    # Position-specific peak age
    def _get_peak(pos):
        if pd.isna(pos):
            return peak_ages["default"]
        pos_str = str(pos).split("-")[0].strip()
        return peak_ages.get(pos_str, peak_ages["default"])

    df["PEAK_AGE"] = df["Pos"].apply(_get_peak)
    df["YEARS_FROM_PEAK"] = df[age_col] - df["PEAK_AGE"]
    df["IS_PRE_PEAK"] = (df["YEARS_FROM_PEAK"] < 0).astype(int)
    df = df.drop(columns=["PEAK_AGE"])

    # Contract year proxy
    df["CONTRACT_YEAR_PROXY"] = df["EXPERIENCE"].isin(contract_years).astype(int)
    return df


def _calculate_rolling_momentum(df, config):
    """Season-over-season momentum features for key stats."""
    momentum_stats = config["preprocessing"]["momentum_stats"]
    df = df.sort_values(["Player", "SEASON"]).copy()

    for stat in momentum_stats:
        if stat not in df.columns:
            continue

        # Year-over-year delta (lagged to prevent leakage)
        df[f"{stat}_DELTA"] = df.groupby("Player")[stat].diff().groupby(df["Player"]).shift(1)

        # Exponentially weighted 3-season rolling average (lagged)
        df[f"{stat}_EWMA_3"] = (
            df.groupby("Player")[stat]
            .transform(lambda x: x.ewm(span=3, min_periods=1).mean())
            .groupby(df["Player"])
            .shift(1)
        )

        # Breakout flag: delta > 1 league-wide std dev of deltas
        raw_delta = df.groupby("Player")[stat].diff()
        season_std = raw_delta.groupby(df["SEASON"]).transform("std")
        lagged_delta = raw_delta.groupby(df["Player"]).shift(1)
        lagged_season_std = season_std.groupby(df["Player"]).shift(1)
        df[f"{stat}_BREAKOUT"] = (lagged_delta > lagged_season_std).astype(int)

    return df


def _calculate_depth_chart(df):
    """Rank by minutes within team and position per season."""
    df["DEPTH_CHART_RANK"] = df.groupby(["SEASON", "TEAM", "Pos"])["MP"].rank(
        ascending=False, method="min"
    )
    return df


def _fetch_combine_measurements(df):
    """Merge wingspan and standing reach from NBA combine data."""
    print("Fetching Combine Stats...")
    try:
        combine = draftcombinestats.DraftCombineStats(
            season_all_time="All Time"
        ).get_data_frames()[0]
        combine = combine[["PLAYER_NAME", "WINGSPAN", "STANDING_REACH"]].drop_duplicates(
            subset=["PLAYER_NAME"]
        )

        # Drop pre-existing columns to prevent _x/_y suffixes
        for col in ["WINGSPAN", "STANDING_REACH"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = pd.merge(df, combine, left_on="Player", right_on="PLAYER_NAME", how="left")
        df.drop(columns=["PLAYER_NAME"], inplace=True, errors="ignore")
    except Exception as e:
        print("Combine fetch error:", e)
        if "WINGSPAN" not in df.columns:
            df["WINGSPAN"] = np.nan
        if "STANDING_REACH" not in df.columns:
            df["STANDING_REACH"] = np.nan
    return df


def _fetch_coach_data(df, config):
    """Scrape head coach data and add coach encoding."""
    print("Fetching Coach Data...")
    headers = {"User-Agent": config["scraping"]["user_agent"]}
    coach_delay = config["scraping"]["coach_delay"]
    coach_data = []

    for year in df["SEASON"].unique():
        try:
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_coaches.html"
            req = requests.get(url, headers=headers)
            html_content = re.sub(r"<!--(.*?)-->", r"\1", req.text, flags=re.DOTALL)
            cdf = pd.read_html(io.StringIO(html_content))[0]
            if isinstance(cdf.columns, pd.MultiIndex):
                cdf.columns = cdf.columns.get_level_values(1)

            cdf = cdf[["Coach", "Tm"]].dropna().copy()
            cdf["Tm"] = cdf["Tm"].map(BBR_TEAM_MAPPING)
            cdf = cdf.drop_duplicates(subset=["Tm"], keep="first")
            cdf["SEASON"] = year
            coach_data.append(cdf)
            time.sleep(coach_delay)
        except Exception:
            pass

    if coach_data:
        all_coaches = pd.concat(coach_data, ignore_index=True)
        # Rename Tm to TEAM for merge consistency
        if "TEAM" in df.columns:
            all_coaches = all_coaches.rename(columns={"Tm": "TEAM"})
            df = pd.merge(df, all_coaches, on=["SEASON", "TEAM"], how="left")
        else:
            df = pd.merge(df, all_coaches, on=["SEASON", "Tm"], how="left")
    else:
        df["Coach"] = "Unknown"

    # ML-ready categorical encoding
    df["COACH_ID"] = df["Coach"].factorize()[0]
    return df


def engineer_advanced_features(df):
    """
    Computes experience, injury features, age features, momentum,
    depth chart, wingspan, and coach features.
    """
    config = load_config()
    df = df.copy()

    # Handle TEAM column name (may be Tm or TEAM depending on pipeline stage)
    team_col = "TEAM" if "TEAM" in df.columns else "Tm"

    print("Calculating Experience...")
    df = _calculate_experience(df)

    print("Calculating Games Missed & Played %...")
    df = _calculate_injury_history(df, config)

    print("Calculating Rolling Injury Features...")
    df = _calculate_rolling_injury_features(df)

    print("Calculating Age Features...")
    df = _calculate_age_features(df, config)

    print("Calculating Rolling Momentum Features...")
    df = _calculate_rolling_momentum(df, config)

    print("Calculating Depth Chart...")
    if team_col == "Tm":
        # Temporarily rename for depth chart calc
        df = df.rename(columns={"Tm": "TEAM"})
    df = _calculate_depth_chart(df)
    if team_col == "Tm":
        df = df.rename(columns={"TEAM": "Tm"})

    df = _fetch_combine_measurements(df)
    df = _fetch_coach_data(df, config)

    print("Advanced feature engineering complete.")
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/historical_nba_dataset.csv")
    df = engineer_advanced_features(df)
    df.to_csv("data/raw/historical_nba_dataset.csv", index=False)
