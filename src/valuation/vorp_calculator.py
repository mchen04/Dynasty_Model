import pandas as pd


class DynastyValuationEngine:
    """
    Stage 2 - Configurable Valuation Layer

    Converts ML-projected stat lines into actionable Dynasty Trade values and
    Auction Draft dollars via deterministic math.
    """

    def __init__(self, num_teams=12, roster_size=13, total_budget=200):
        self.num_teams = num_teams
        self.roster_size = roster_size
        self.total_budget = total_budget

        # Calculate the total number of players rostered in the league
        self.total_rostered = self.num_teams * self.roster_size

    def _calculate_fantasy_points(self, projected_stats_df, custom_weights=None):
        """
        Converts projected stats into fantasy points.
        Supports standard ESPN H2H Points or custom weighting setups (roster-context punting).
        """
        if custom_weights is None:
            # Default ESPN Standard H2H Points Weights
            weights = {
                "PTS": 1,
                "REB": 1,
                "AST": 2,
                "STL": 4,
                "BLK": 4,
                "TOV": -2,
                "FGM": 2,
                "FGA": -1,
                "FTM": 1,
                "FTA": -1,
                "3PM": 1,
            }
        else:
            weights = custom_weights

        fpts = pd.Series(0, index=projected_stats_df.index)
        for stat, weight in weights.items():
            if stat in projected_stats_df.columns:
                fpts += projected_stats_df[stat] * weight

        return fpts

    def get_replacement_level(self, projected_fpts_series):
        """
        Finds the Replacement Level player (the best player freely available on the waiver wire).
        Assumes the top `total_rostered` players are taken.
        """
        # Sort players by fantasy points descending
        sorted_fpts = projected_fpts_series.sort_values(ascending=False)

        # The replacement level is the Nth + 1 player
        if len(sorted_fpts) > self.total_rostered:
            replacement_fpts = sorted_fpts.iloc[self.total_rostered]
        else:
            # If the league is so deep everybody is rostered, replacement level is 0
            replacement_fpts = 0

        return replacement_fpts

    def calculate_vorp_and_dollars(self, df_projections, custom_weights=None):
        """
        Main calculation engine that generates VORP and translates it to Draft Auction Dollars.
        """
        print(
            f"Calculating Values for {self.num_teams}-team league ($str{self.total_budget} budget)..."
        )

        # 1. Get Fantasy Points
        df_projections["FPTS"] = self._calculate_fantasy_points(
            df_projections, custom_weights
        )

        # 2. Get Replacement Level
        repl_level = self.get_replacement_level(df_projections["FPTS"])
        print(f"League Replacement Level found at: {repl_level:.1f} FPTS")

        # 3. Calculate VORP (can be negative for waiver wire players)
        df_projections["VORP"] = df_projections["FPTS"] - repl_level

        # 4. Calculate Auction Dollars
        # Only players with positive VORP get allocated money from the budget pool
        total_positive_vorp = df_projections.loc[
            df_projections["VORP"] > 0, "VORP"
        ].sum()
        total_league_dollars = self.num_teams * self.total_budget

        # We reserve $1 for every roster spot (minimum bid)
        total_guaranteed_dollars = self.total_rostered * 1
        available_dollar_pool = total_league_dollars - total_guaranteed_dollars

        # Assign values
        df_projections["AUCTION_VALUE"] = 0.0

        positive_mask = df_projections["VORP"] > 0
        df_projections.loc[positive_mask, "AUCTION_VALUE"] = 1.0 + (
            (df_projections.loc[positive_mask, "VORP"] / total_positive_vorp)
            * available_dollar_pool
        )

        return df_projections.sort_values(by="AUCTION_VALUE", ascending=False)


if __name__ == "__main__":
    print("Testing Stage 2 Valuation Logic...")

    # Mock projections from Stage 1 ML Model (Per Game)
    projections = pd.DataFrame(
        {
            "PLAYER_NAME": [
                "Jokic",
                "Giannis",
                "Shai",
                "Wemby",
                "RolePlayer1",
                "WaiverWire1",
                "WaiverWire2",
            ],
            "PTS": [26.0, 30.0, 31.0, 24.0, 10.0, 5.0, 4.0],
            "REB": [12.0, 11.0, 5.0, 11.0, 4.0, 2.0, 1.0],
            "AST": [9.0, 6.0, 6.0, 4.0, 2.0, 1.0, 0.5],
            "STL": [1.3, 1.2, 2.1, 1.2, 0.8, 0.5, 0.2],
            "BLK": [0.7, 1.0, 0.9, 3.6, 0.3, 0.1, 0.0],
            "TOV": [3.0, 3.5, 2.5, 3.5, 1.0, 0.5, 0.5],
        }
    )

    # We pretend it's a very shallow league to force the waiver wire logic
    # 2 teams, 2 players each = 4 total rostered players.
    engine = DynastyValuationEngine(num_teams=2, roster_size=2, total_budget=200)

    valuations = engine.calculate_vorp_and_dollars(projections)

    print("\nCalculated Auction Values (Standard ESPN Points):")
    print(valuations[["PLAYER_NAME", "FPTS", "VORP", "AUCTION_VALUE"]].round(1))

    # Test Roster Context logic (Punting/Custom weighting)
    print("\nLet's test dynamic roster context.")
    # Manager doesn't care about blocks or rebounds
    custom = {"PTS": 1, "AST": 2, "STL": 4, "TOV": -2, "BLK": 0, "REB": 0}
    custom_valuations = engine.calculate_vorp_and_dollars(
        projections.copy(), custom_weights=custom
    )
    print("Calculated Auction Values (Punting REB + BLK):")
    print(custom_valuations[["PLAYER_NAME", "FPTS", "VORP", "AUCTION_VALUE"]].round(1))
