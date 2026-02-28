import re
import io

import pandas as pd
import requests
from basketball_reference_scraper.players import get_stats

from src.utils.config import load_config


def _get_headers():
    config = load_config()
    return {"User-Agent": config["scraping"]["user_agent"]}


def _get_delay(delay_key="bref_delay"):
    import time
    config = load_config()
    time.sleep(config["scraping"][delay_key])


def get_player_season_stats(
    player_name, stat_type="PER_GAME", playoffs=False, career=False
):
    """
    Fetches season-level stats for a given player from Basketball-Reference.
    Supported stat types: 'PER_GAME', 'PER_MINUTE', 'PER_POSS', 'ADVANCED'
    """
    try:
        print(f"Fetching {stat_type} stats for {player_name}...")
        df = get_stats(
            player_name, stat_type=stat_type, playoffs=playoffs, career=career
        )
        _get_delay("bref_player_delay")
        return df
    except Exception as e:
        print(f"Error fetching stats for {player_name}: {e}")
        return None


def _scrape_bref_table(url, table_attrs=None):
    """Shared scraping logic for Basketball-Reference HTML tables."""
    headers = _get_headers()
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    html_content = re.sub(r"<!--(.*?)-->", r"\1", response.text, flags=re.DOTALL)

    if table_attrs:
        dfs = pd.read_html(io.StringIO(html_content), attrs=table_attrs)
    else:
        dfs = pd.read_html(io.StringIO(html_content))

    return dfs, html_content


def scrape_bref_advanced_stats_page(year):
    """Scrapes the advanced stats league leader table for a given year."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    print(f"Scraping advanced stats for {year} season from: {url}")

    try:
        dfs, _ = _scrape_bref_table(url)
        dfs = [d for d in dfs if "Player" in d.columns]
        if not dfs:
            return None
        df = dfs[0]

        df = df[df["Rk"] != "Rk"].copy()
        df.drop(
            columns=["Unnamed: 19", "Unnamed: 24"], inplace=True, errors="ignore"
        )
        if "Team" in df.columns:
            df.rename(columns={"Team": "Tm"}, inplace=True)

        cols_to_ignore = ["Player", "Pos", "Tm", "Awards"]
        numeric_cols = [c for c in df.columns if c not in cols_to_ignore]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        _get_delay()
        return df
    except Exception as e:
        print(
            f"Error scraping advanced stats: {type(e).__name__} (skipping to next...)"
        )
        return None


def scrape_bref_per_game_stats_page(year):
    """Scrapes the per_game stats league leader table for a given year."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    print(f"Scraping per-game stats for {year} season from: {url}")

    try:
        dfs, _ = _scrape_bref_table(url)
        dfs = [d for d in dfs if "Player" in d.columns]
        if not dfs:
            return None
        df = dfs[0]

        df = df[df["Rk"] != "Rk"].copy()
        if "Team" in df.columns:
            df.rename(columns={"Team": "Tm"}, inplace=True)

        cols_to_ignore = ["Player", "Pos", "Tm", "Awards"]
        numeric_cols = [c for c in df.columns if c not in cols_to_ignore]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        _get_delay()
        return df
    except Exception as e:
        print(
            f"Error scraping per-game stats: {type(e).__name__} (skipping to next...)"
        )
        return None


def scrape_bref_per_poss_stats_page(year):
    """Scrapes the per_100 poss stats league leader table for a given year."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_poss.html"
    print(f"Scraping per-100 possession stats for {year} season from: {url}")

    try:
        dfs, _ = _scrape_bref_table(url)
        dfs = [d for d in dfs if "Player" in d.columns]
        if not dfs:
            return None
        df = dfs[0]

        df = df[df["Rk"] != "Rk"].copy()
        if "Team" in df.columns:
            df.rename(columns={"Team": "Tm"}, inplace=True)

        cols_to_ignore = ["Player", "Pos", "Tm", "Awards"]
        numeric_cols = [c for c in df.columns if c not in cols_to_ignore]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        _get_delay()
        return df
    except Exception as e:
        print(f"Error scraping per-100 stats: {type(e).__name__} (skipping to next...)")
        return None


def scrape_bref_team_stats_page(year):
    """Scrapes the team advanced stats (Net Rating, Pace, W-L) for a given year."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}.html"
    print(f"Scraping team advanced stats for {year} season from: {url}")
    try:
        dfs, _ = _scrape_bref_table(url, table_attrs={"id": "advanced-team"})
        if not dfs:
            return None
        df = dfs[0]

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)

        df = df[df["Team"] != "League Average"].copy()
        df["Team"] = df["Team"].str.replace("*", "", regex=False)

        cols_to_keep = ["Team", "W", "L", "ORtg", "DRtg", "NRtg", "Pace"]
        df = df[[c for c in cols_to_keep if c in df.columns]]

        _get_delay()
        return df
    except Exception as e:
        print(
            f"Error scraping team advanced stats: {type(e).__name__} (skipping to next...)"
        )
        return None


if __name__ == "__main__":
    lebron_adv = get_player_season_stats("LeBron James", stat_type="ADVANCED")
    if lebron_adv is not None:
        print(lebron_adv.head())

    season_2025_adv = scrape_bref_advanced_stats_page(2025)
    if season_2025_adv is not None:
        print(f"Successfully scraped {len(season_2025_adv)} player records for 2025.")
        print(season_2025_adv.head())
