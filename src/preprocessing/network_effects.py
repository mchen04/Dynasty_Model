import pandas as pd


def calculate_usage_vacuum(
    roster_df_t_minus_1,
    roster_df_t,
    player_id_col="PLAYER_ID",
    usage_col="USG_PCT",
    team_col="TEAM_ID",
):
    """
    Calculates the 'Usage Vacuum' for each team in Season T.

    Usage Vacuum = the sum of Usage Rates from players who were on the team in Season T-1
    but are no longer on the team in Season T.
    """
    vacuums = []

    teams = roster_df_t[team_col].unique()
    for team in teams:
        team_t_minus_1 = roster_df_t_minus_1[roster_df_t_minus_1[team_col] == team]
        team_t = roster_df_t[roster_df_t[team_col] == team]

        departed_players = team_t_minus_1[
            ~team_t_minus_1[player_id_col].isin(team_t[player_id_col])
        ]

        vacated_usage = departed_players[usage_col].sum()
        vacuums.append({team_col: team, "VACATED_USAGE": vacated_usage})

    return pd.DataFrame(vacuums)


def calculate_teammate_efficiency(
    player_row,
    team_roster_df,
    player_id_col="PLAYER_ID",
    ts_pct_col="TS_PCT",
    min_col="MIN",
):
    """
    Calculates the minute-weighted True Shooting percentage of a player's teammates.
    """
    teammates = team_roster_df[
        team_roster_df[player_id_col] != player_row[player_id_col]
    ]

    if len(teammates) == 0:
        return 0.0

    total_teammate_mins = teammates[min_col].sum()
    if total_teammate_mins == 0:
        return 0.0

    weighted_ts = teammates[ts_pct_col] * (teammates[min_col] / total_teammate_mins)
    return weighted_ts.sum()


def apply_network_effects(df_season_t_minus_1, df_season_t):
    """
    Main orchestration function to join network effect features onto the Season T dataframe.
    """
    print("Calculating Usage Vacuums...")
    vacuums_df = calculate_usage_vacuum(df_season_t_minus_1, df_season_t)

    df_season_t = df_season_t.merge(vacuums_df, on="TEAM_ID", how="left")

    print("Calculating Teammate Efficiencies...")
    teammate_ts = df_season_t.apply(
        lambda row: calculate_teammate_efficiency(
            row, df_season_t[df_season_t["TEAM_ID"] == row["TEAM_ID"]]
        ),
        axis=1,
    )
    df_season_t["TEAMMATE_TS_PCT"] = teammate_ts

    return df_season_t


if __name__ == "__main__":
    print("Testing Network Effects calculation...")

    t_minus_1 = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3, 4],
            "PLAYER_NAME": ["Shai", "Jalen", "Chet", "Giddey"],
            "TEAM_ID": ["OKC", "OKC", "OKC", "OKC"],
            "USG_PCT": [32.0, 25.0, 20.0, 18.0],
            "TS_PCT": [0.65, 0.60, 0.62, 0.52],
            "MIN": [34.0, 32.0, 30.0, 25.0],
        }
    )

    t = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3, 5],
            "PLAYER_NAME": ["Shai", "Jalen", "Chet", "Caruso"],
            "TEAM_ID": ["OKC", "OKC", "OKC", "OKC"],
            "USG_PCT": [33.0, 26.0, 22.0, 12.0],
            "TS_PCT": [0.65, 0.61, 0.63, 0.58],
            "MIN": [34.0, 33.0, 32.0, 28.0],
        }
    )

    enhanced_db = apply_network_effects(t_minus_1, t)

    print("\nSeason T with Network Effects:")
    print(enhanced_db[["PLAYER_NAME", "TEAM_ID", "VACATED_USAGE", "TEAMMATE_TS_PCT"]])
