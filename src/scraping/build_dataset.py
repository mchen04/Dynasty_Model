import os
import pandas as pd
from bball_ref_scraper import (
    scrape_bref_advanced_stats_page,
    scrape_bref_per_game_stats_page,
    scrape_bref_per_poss_stats_page,
    scrape_bref_team_stats_page,
)
from nba_api_client import get_nba_tracking_stats, get_nba_bio_stats

BBR_TEAM_MAPPING = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Bobcats": "CHA",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Jersey Nets": "NJN",
    "New Orleans Hornets": "NOH",
    "New Orleans Pelicans": "NOP",
    "New Orleans/Oklahoma City Hornets": "NOK",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Seattle SuperSonics": "SEA",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def build_complete_dataset(start_year=2002, end_year=2025):
    """
    Builds the complete NBA raw dataset as specified in the Dynasty Model spec.
    Fetches Core Box Score Stats + Advanced Metrics for every player, every season.
    Also merges Tracking Data (Potential Assists, etc.) from NBA API for post-2013 eras.
    """
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/raw")
    )
    os.makedirs(output_dir, exist_ok=True)
    all_seasons_data = []

    for year in range(start_year, end_year + 1):
        print(f"\n--- Gathering Data for {year - 1}-{str(year)[-2:]} Season ---")

        # 1. Get Per Game Box Score Stats
        per_game_df = scrape_bref_per_game_stats_page(year)
        if per_game_df is None:
            print(f"Failed to get per_game stats for {year}")
            continue

        # 2. Get Advanced Metrics
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

        # 3. Get Per-100 Possessions
        per_poss_df = scrape_bref_per_poss_stats_page(year)
        if per_poss_df is not None:
            shared_cols = [
                c
                for c in season_df.columns
                if c in per_poss_df.columns and c not in ["Player", "Tm", "Age", "Pos"]
            ]
            poss_reduced = per_poss_df[["Player", "Tm", "Age", "Pos"] + shared_cols]
            # Rename the stats so they don't overwrite per-game metrics
            rename_map = {c: f"{c}_per100" for c in shared_cols}
            poss_reduced = poss_reduced.rename(columns=rename_map)
            season_df = pd.merge(
                season_df, poss_reduced, on=["Player", "Tm", "Age", "Pos"], how="left"
            )

        # 4. Get Team Context
        team_df = scrape_bref_team_stats_page(year)
        if team_df is not None:
            team_df["Tm"] = team_df["Team"].map(BBR_TEAM_MAPPING)
            team_df = team_df.drop(columns=["Team"])
            season_df = pd.merge(season_df, team_df, on="Tm", how="left")

        season_df["SEASON"] = year

        # 5. Get Tracking & Bio Data from NBA API
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
            # NBA API season format is "2013-14" for year=2014
            api_season_str = f"{year - 1}-{str(year)[-2:]}"
            # Passing Tracking
            tracking_pass = get_nba_tracking_stats(season=api_season_str)
            if tracking_pass is not None:
                # Match names (this is fragile due to suffixes like Jr., III, but good enough for a base dataset)
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

    # Concatenate everything into one massive DataFrame
    final_df = pd.concat(all_seasons_data, ignore_index=True)

    # Save to CSV
    output_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../data/raw/historical_nba_dataset.csv"
        )
    )
    final_df.to_csv(output_path, index=False)

    print("\n=== Dataset Building Complete ===")
    print(f"Saved {len(final_df)} player-season records to {output_path}")
    print(final_df.head())
    print("\nColumns included:")
    print(final_df.columns.tolist())


if __name__ == "__main__":
    # Run the full 2001-2002 to 2025-2026 extraction process
    build_complete_dataset(start_year=2002, end_year=2026)
