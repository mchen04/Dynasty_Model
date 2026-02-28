import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.preprocessing.zscores import normalize_trade_value


class DynastyValuationEngine:
    """
    Stage 2 - Configurable Valuation Layer

    Converts ML-projected stat lines into actionable Dynasty Trade values and
    Auction Draft dollars via deterministic math. Supports multi-year dynasty
    discounting, positional replacement levels, and shutdown risk.
    """

    # B-Ref column aliases for fantasy point calculation
    _STAT_ALIASES = {
        "FGM": ["FGM", "FG"],
        "FTM": ["FTM", "FT"],
        "3PM": ["3PM", "3P"],
        "REB": ["REB", "TRB"],
    }

    def __init__(self, config=None):
        if config is None:
            config = load_config()

        league = config["league"]
        self.num_teams = league["num_teams"]
        self.roster_size = league["roster_size"]
        self.total_budget = league["total_budget"]
        self.dynasty_horizon = league["dynasty_horizon"]
        self.consolidation_premium = league["consolidation_premium"]
        self.discount_curve = {
            int(k): v for k, v in league["dynasty_discount_curve"].items()
        }
        self.default_weights = league["scoring_weights"]
        self.shutdown_cfg = config["preprocessing"]["shutdown"]

        self.total_rostered = self.num_teams * self.roster_size

    def _resolve_stat(self, df, stat_name):
        """Resolve a stat column, checking aliases if primary name missing."""
        if stat_name in df.columns:
            return df[stat_name]
        aliases = self._STAT_ALIASES.get(stat_name, [])
        for alias in aliases:
            if alias in df.columns:
                return df[alias]
        return pd.Series(0, index=df.index)

    def _calculate_fantasy_points(self, projected_stats_df, custom_weights=None):
        """Converts projected stats into fantasy points per game."""
        weights = custom_weights if custom_weights is not None else self.default_weights

        fpts = pd.Series(0.0, index=projected_stats_df.index)
        for stat, weight in weights.items():
            fpts += self._resolve_stat(projected_stats_df, stat) * weight

        return fpts

    def get_replacement_level(self, projected_fpts_series, position=None):
        """
        Finds the replacement level player.
        If position is specified, calculates position-specific replacement level.
        """
        sorted_fpts = projected_fpts_series.sort_values(ascending=False)

        if len(sorted_fpts) > self.total_rostered:
            return sorted_fpts.iloc[self.total_rostered]
        return 0.0

    def get_positional_replacement_levels(self, df, fpts_col="FPTS"):
        """
        Calculate replacement level per position based on league size.
        Multi-position eligible players get blended replacement levels.
        """
        positions = ["PG", "SG", "SF", "PF", "C"]
        pos_col = "Pos" if "Pos" in df.columns else "POSITION"
        if pos_col not in df.columns:
            return {p: 0.0 for p in positions}

        repl_levels = {}
        slots_per_pos = max(1, self.num_teams)  # At least 1 starter per position

        for pos in positions:
            pos_mask = df[pos_col].fillna("").str.contains(pos, na=False)
            pos_fpts = df.loc[pos_mask, fpts_col].sort_values(ascending=False)
            if len(pos_fpts) > slots_per_pos:
                repl_levels[pos] = pos_fpts.iloc[slots_per_pos]
            else:
                repl_levels[pos] = 0.0

        return repl_levels

    def _get_blended_replacement(self, position_str, repl_levels):
        """Average replacement level across all eligible positions."""
        if pd.isna(position_str) or not position_str:
            return np.mean(list(repl_levels.values()))

        positions = [p.strip() for p in str(position_str).replace("/", "-").split("-")]
        levels = [repl_levels.get(p, 0.0) for p in positions if p in repl_levels]
        return np.mean(levels) if levels else np.mean(list(repl_levels.values()))

    def calculate_shutdown_factor(self, row, year_offset=0):
        """
        Age/team-context-based multiplier for fantasy playoff reliability.
        """
        cfg = self.shutdown_cfg
        factor = 1.0

        age = row.get("AGE", row.get("Age", 25))
        wins = row.get("W", 41)

        if age >= cfg["age_load_management"]:
            factor -= cfg["load_management_discount"]
        elif age >= cfg["age_contender_rest"] and wins > cfg["contender_wins"]:
            factor -= cfg["contender_rest_discount"]

        if wins < cfg["tanking_wins"]:
            factor -= cfg["tanking_discount"]

        # Future year uncertainty discount
        if year_offset > 0:
            factor -= cfg["future_year_discount"] * year_offset

        return max(0.1, factor)

    def calculate_dynasty_value(self, df_projections_by_year, custom_weights=None):
        """
        Multi-year dynasty discounting.

        dynasty_fpts = SUM(discount[t] * fpts_per_game[t] * projected_gp[t] * shutdown_factor[t])
        for t=1..horizon

        df_projections_by_year: dict mapping horizon (1-5) -> DataFrame with projected per-game stats
        """
        dynasty_fpts = pd.Series(0.0, dtype=float)
        initialized = False

        for t in range(1, self.dynasty_horizon + 1):
            if t not in df_projections_by_year:
                continue

            df_t = df_projections_by_year[t]
            if not initialized:
                dynasty_fpts = pd.Series(0.0, index=df_t.index)
                initialized = True

            fpts_per_game = self._calculate_fantasy_points(df_t, custom_weights)

            # Projected games played (use G if available, else assume 70)
            projected_gp = self._resolve_stat(df_t, "G")
            projected_gp = projected_gp.fillna(70)

            discount = self.discount_curve.get(t, 0.5)

            shutdown_factors = df_t.apply(
                lambda row: self.calculate_shutdown_factor(row, year_offset=t - 1),
                axis=1,
            )

            dynasty_fpts += discount * fpts_per_game * projected_gp * shutdown_factors

        return dynasty_fpts

    def calculate_vorp_and_dollars(self, df_projections, custom_weights=None):
        """
        Single-year valuation (backward compatible wrapper).
        Calculates VORP and auction dollar values.
        """
        print(
            f"Calculating Values for {self.num_teams}-team league (${self.total_budget} budget)..."
        )

        df = df_projections.copy()

        # Fantasy points
        df["FPTS"] = self._calculate_fantasy_points(df, custom_weights)

        # Positional replacement levels
        pos_col = "Pos" if "Pos" in df.columns else "POSITION"
        if pos_col in df.columns:
            repl_levels = self.get_positional_replacement_levels(df)
            df["REPL_LEVEL"] = df[pos_col].apply(
                lambda p: self._get_blended_replacement(p, repl_levels)
            )
        else:
            repl_level = self.get_replacement_level(df["FPTS"])
            df["REPL_LEVEL"] = repl_level

        print(f"  Replacement levels: {df['REPL_LEVEL'].describe().to_dict()}")

        # VORP
        df["VORP"] = df["FPTS"] - df["REPL_LEVEL"]

        # Auction Dollars
        total_positive_vorp = df.loc[df["VORP"] > 0, "VORP"].sum()
        total_league_dollars = self.num_teams * self.total_budget
        total_guaranteed_dollars = self.total_rostered * 1
        available_pool = total_league_dollars - total_guaranteed_dollars

        df["AUCTION_VALUE"] = 0.0
        positive_mask = df["VORP"] > 0
        if total_positive_vorp > 0:
            df.loc[positive_mask, "AUCTION_VALUE"] = 1.0 + (
                (df.loc[positive_mask, "VORP"] / total_positive_vorp) * available_pool
            )

        # Trade value (power curve)
        df["TRADE_VALUE"] = normalize_trade_value(
            df["VORP"], premium=self.consolidation_premium
        )

        return df.sort_values(by="AUCTION_VALUE", ascending=False)

    def calculate_roster_context_value(
        self, df_projections, market_weights=None, roster_weights=None
    ):
        """
        Produces market value vs roster-specific value comparison.
        Useful for identifying undervalued buy targets in punt builds.

        Returns DataFrame with MARKET_VALUE, ROSTER_VALUE, VALUE_GAP columns.
        """
        # Market value = standard scoring
        df_market = self.calculate_vorp_and_dollars(df_projections.copy(), market_weights)
        df_market = df_market.rename(columns={"AUCTION_VALUE": "MARKET_VALUE"})

        # Roster value = punt-build-specific scoring
        df_roster = self.calculate_vorp_and_dollars(df_projections.copy(), roster_weights)
        df_roster = df_roster.rename(columns={"AUCTION_VALUE": "ROSTER_VALUE"})

        # Merge
        id_col = "Player" if "Player" in df_market.columns else df_market.index
        result = df_market[["MARKET_VALUE"]].copy()
        result["ROSTER_VALUE"] = df_roster["ROSTER_VALUE"].values
        result["VALUE_GAP"] = result["ROSTER_VALUE"] - result["MARKET_VALUE"]

        return result.sort_values("VALUE_GAP", ascending=False)


if __name__ == "__main__":
    print("Testing Stage 2 Valuation Logic...")

    projections = pd.DataFrame(
        {
            "Player": ["Jokic", "Giannis", "Shai", "Wemby", "Role1", "Waiver1", "Waiver2"],
            "Pos": ["C", "PF", "PG", "PF-C", "SG", "PG", "SF"],
            "PTS": [26.0, 30.0, 31.0, 24.0, 10.0, 5.0, 4.0],
            "REB": [12.0, 11.0, 5.0, 11.0, 4.0, 2.0, 1.0],
            "AST": [9.0, 6.0, 6.0, 4.0, 2.0, 1.0, 0.5],
            "STL": [1.3, 1.2, 2.1, 1.2, 0.8, 0.5, 0.2],
            "BLK": [0.7, 1.0, 0.9, 3.6, 0.3, 0.1, 0.0],
            "TOV": [3.0, 3.5, 2.5, 3.5, 1.0, 0.5, 0.5],
            "FGM": [10.0, 11.0, 11.0, 9.0, 4.0, 2.0, 1.5],
            "FGA": [18.0, 20.0, 20.0, 19.0, 9.0, 5.0, 4.0],
            "FTM": [5.0, 7.0, 8.0, 5.0, 2.0, 1.0, 0.5],
            "FTA": [6.0, 10.0, 9.0, 7.0, 3.0, 1.5, 1.0],
            "3PM": [1.0, 1.0, 2.0, 1.5, 1.0, 0.5, 0.5],
            "AGE": [29, 29, 25, 21, 28, 30, 26],
            "W": [50, 48, 60, 45, 35, 20, 42],
            "G": [75, 70, 78, 68, 72, 55, 60],
        }
    )

    engine = DynastyValuationEngine()
    valuations = engine.calculate_vorp_and_dollars(projections)

    print("\nCalculated Auction Values (Standard ESPN Points):")
    print(
        valuations[["Player", "FPTS", "VORP", "AUCTION_VALUE", "TRADE_VALUE"]].round(1)
    )
