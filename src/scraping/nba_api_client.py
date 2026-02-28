import time

from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashptstats

from src.utils.config import load_config


def _api_delay():
    config = load_config()
    time.sleep(config["scraping"]["nba_api_delay"])


def get_nba_tracking_stats(season):
    """
    Fetches active player tracking stats (e.g., potential assists, passes)
    from stats.nba.com.
    Note: Tracking stats are only reliably available post-2013.
    """
    print(f"Fetching tracking stats (Passing) for {season}...")
    try:
        tracking = leaguedashptstats.LeagueDashPtStats(
            season=season,
            pt_measure_type="Passing",
            player_or_team="Player",
            per_mode_simple="PerGame",
        )
        df = tracking.get_data_frames()[0]
        _api_delay()
        return df
    except Exception as e:
        print(f"Error fetching tracking stats: {e}")
        return None


def get_nba_base_stats(season, per_mode="Per100Possessions"):
    """
    Fetches base or advanced stats from stats.nba.com.
    per_mode options: "PerGame", "Per100Possessions", "Totals"
    """
    print(f"Fetching base stats ({per_mode}) for {season}...")
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed=per_mode
        )
        df = stats.get_data_frames()[0]
        _api_delay()
        return df
    except Exception as e:
        print(f"Error fetching base stats: {e}")
        return None


def get_nba_bio_stats(season):
    """
    Fetches biological and draft data (Height, Weight, College, Draft Pick) from stats.nba.com.
    """
    print(f"Fetching bio/draft stats for {season}...")
    try:
        from nba_api.stats.endpoints import leaguedashplayerbiostats

        stats = leaguedashplayerbiostats.LeagueDashPlayerBioStats(season=season)
        df = stats.get_data_frames()[0]
        _api_delay()
        return df
    except Exception as e:
        print(f"Error fetching bio stats: {e}")
        return None


if __name__ == "__main__":
    tracking_df = get_nba_tracking_stats("2024-25")
    if tracking_df is not None:
        print(
            f"Successfully fetched passing tracking stats for {len(tracking_df)} players."
        )
        print(
            tracking_df[
                [
                    "PLAYER_NAME",
                    "TEAM_ABBREVIATION",
                    "PASSES_MADE",
                    "AST",
                    "POTENTIAL_AST",
                ]
            ].head()
        )

    per100_df = get_nba_base_stats("2024-25", "Per100Possessions")
    if per100_df is not None:
        print(f"Successfully fetched Per 100 stats for {len(per100_df)} players.")
        print(
            per100_df[["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB"]].head()
        )
