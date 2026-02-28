import os

import pandas as pd

from src.scraping.bball_ref_scraper import (
    scrape_bref_advanced_stats_page,
    scrape_bref_per_game_stats_page,
    scrape_bref_per_poss_stats_page,
    scrape_bref_team_stats_page,
)
from src.scraping.nba_api_client import get_nba_tracking_stats, get_nba_bio_stats
from src.utils.config import load_config, resolve_path
from src.utils.constants import BBR_TEAM_MAPPING


def build_complete_dataset(start_year=None, end_year=None):
    """
    Builds the complete NBA raw dataset as specified in the Dynasty Model spec.
    Fetches Core Box Score Stats + Advanced Metrics for every player, every season.
    Also merges Tracking Data (Potential Assists, etc.) from NBA API for post-2013 eras.
    """
    config = load_config()
    if start_year is None:
        start_year = config["scraping"]["start_year"]
    if end_year is None:
        end_year = config["scraping"]["end_year"]

    output_path = resolve_path(config["project"]["raw_data"])
    os.makedirs(output_path.parent, exist_ok=True)
    all_seasons_data = []

    for year in range(start_year, end_year + 1):
        print(f"\n--- Gathering Data for {year - 1}-{str(year)[-2:]} Season ---")

        per_game_df = scrape_bref_per_game_stats_page(year)
        if per_game_df is None:
            print(f"Failed to get per_game stats for {year}")
            continue

        advanced_df = scrape_bref_advanced_stats_page(year)
        if advanced_df is not None:
            shared_cols = [
                c
                for c in per_game_df.columns
                if c in advanced_df.columns and c not in ["Player", "Tm", "Age", "Pos"]
            ]
            adv_reduced = advanced_df.drop(columns=shared_cols)
            season_df = pd.merge(
                per_game_df, adv_reduced, on=["Player", "Tm", "Age", "Pos"], how="left"
            )
        else:
            season_df = per_game_df

        per_poss_df = scrape_bref_per_poss_stats_page(year)
        if per_poss_df is not None:
            shared_cols = [
                c
                for c in season_df.columns
                if c in per_poss_df.columns and c not in ["Player", "Tm", "Age", "Pos"]
            ]
            poss_reduced = per_poss_df[["Player", "Tm", "Age", "Pos"] + shared_cols]
            rename_map = {c: f"{c}_per100" for c in shared_cols}
            poss_reduced = poss_reduced.rename(columns=rename_map)
            season_df = pd.merge(
                season_df, poss_reduced, on=["Player", "Tm", "Age", "Pos"], how="left"
            )

        team_df = scrape_bref_team_stats_page(year)
        if team_df is not None:
            team_df["Tm"] = team_df["Team"].map(BBR_TEAM_MAPPING)
            team_df = team_df.drop(columns=["Team"])
            season_df = pd.merge(season_df, team_df, on="Tm", how="left")

        season_df["SEASON"] = year

        api_season_str = f"{year - 1}-{str(year)[-2:]}"

        bio_df = get_nba_bio_stats(season=api_season_str)
        if bio_df is not None:
            bio_reduced = bio_df[
                [
                    "PLAYER_NAME",
                    "PLAYER_HEIGHT",
                    "PLAYER_WEIGHT",
                    "COLLEGE",
                    "COUNTRY",
                    "DRAFT_YEAR",
                    "DRAFT_ROUND",
                    "DRAFT_NUMBER",
                ]
            ]
            season_df = pd.merge(
                season_df,
                bio_reduced,
                left_on="Player",
                right_on="PLAYER_NAME",
                how="left",
            )
            season_df = season_df.drop(columns=["PLAYER_NAME"], errors="ignore")

        if year >= 2014:
            tracking_pass = get_nba_tracking_stats(season=api_season_str)
            if tracking_pass is not None:
                tracking_reduced = tracking_pass[
                    [
                        "PLAYER_NAME",
                        "PASSES_MADE",
                        "POTENTIAL_AST",
                        "AST_POINTS_CREATED",
                    ]
                ]
                season_df = pd.merge(
                    season_df,
                    tracking_reduced,
                    left_on="Player",
                    right_on="PLAYER_NAME",
                    how="left",
                )
                season_df.drop(columns=["PLAYER_NAME"], inplace=True)

        all_seasons_data.append(season_df)

    if not all_seasons_data:
        print("No data collected!")
        return

    final_df = pd.concat(all_seasons_data, ignore_index=True)
    final_df.to_csv(output_path, index=False)

    print("\n=== Dataset Building Complete ===")
    print(f"Saved {len(final_df)} player-season records to {output_path}")
    print(final_df.head())
    print("\nColumns included:")
    print(final_df.columns.tolist())


if __name__ == "__main__":
    build_complete_dataset()
