import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class BackcastImputer:
    """
    Stage 0 - Data Harmonization & Imputation.

    Trains on Modern Era data (2014-Present) where tracking stats exist.
    Learns the relationship between standard box score stats and advanced metrics,
    then backcasts predictions to pre-2014 era for unbroken historical dataset.

    Supports multiple target metrics trained independently.
    """

    # Expanded feature set (~27 features)
    CORE_FEATURES = [
        "PTS", "AST", "REB", "STL", "BLK", "TOV",
        "FGA", "FGM", "FTA", "FTM", "3PA", "3PM",
        "MP", "AGE", "GS", "ORB", "DRB", "PF",
        "PER", "TS_PCT", "USG_PCT", "AST_PCT", "TOV_PCT",
        "EFG_PCT", "BPM", "WS",
    ]

    # Fallback column name aliases for features that may have B-Ref names
    FEATURE_ALIASES = {
        "FGM": "FG", "FTM": "FT", "3PM": "3P",
        "REB": "TRB", "AGE": "Age",
        "TS_PCT": "TS%", "USG_PCT": "USG%", "AST_PCT": "AST%",
        "TOV_PCT": "TOV%", "EFG_PCT": "eFG%",
    }

    def __init__(self, target_metric):
        self.target_metric = target_metric
        self.model = HistGradientBoostingRegressor(
            max_iter=200, random_state=42, early_stopping=True
        )
        self.is_trained = False
        self._feature_cols = None

    def _select_features(self, df):
        """
        Selects available features from the expanded core set.
        Falls back to aliases if canonical names aren't present.
        """
        if self._feature_cols is not None:
            return df[self._feature_cols]

        available = []
        for feat in self.CORE_FEATURES:
            if feat in df.columns:
                available.append(feat)
            elif feat in self.FEATURE_ALIASES and self.FEATURE_ALIASES[feat] in df.columns:
                available.append(self.FEATURE_ALIASES[feat])

        self._feature_cols = available
        return df[available]

    def train(self, df):
        """Train the imputation model using rows where the target_metric is NOT null."""
        print(f"Training imputer for missing metric: {self.target_metric}")

        train_df = df.dropna(subset=[self.target_metric])
        if len(train_df) == 0:
            raise ValueError(
                f"No training data found containing the target metric {self.target_metric}"
            )

        X = self._select_features(train_df)
        y = train_df[self.target_metric]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"  Model trained on {len(X_train)} samples with {len(self._feature_cols)} features.")
        print(f"  Validation RMSE = {rmse:.4f}, R2 = {r2:.4f}")
        self.is_trained = True
        return self

    def impute(self, df):
        """Predict the target metric for rows where it is currently NaN."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before imputation.")

        imputed_df = df.copy()

        missing_mask = imputed_df[self.target_metric].isna()
        num_missing = missing_mask.sum()

        if num_missing == 0:
            print(f"  No missing values found for {self.target_metric}.")
            return imputed_df

        print(f"  Imputing {num_missing} missing rows for {self.target_metric}...")

        X_missing = self._select_features(imputed_df[missing_mask])
        predictions = self.model.predict(X_missing)
        imputed_df.loc[missing_mask, self.target_metric] = predictions

        return imputed_df


class MultiMetricImputer:
    """Trains and applies BackcastImputer for multiple target metrics."""

    def __init__(self, target_metrics):
        self.target_metrics = target_metrics
        self.imputers = {m: BackcastImputer(target_metric=m) for m in target_metrics}

    def train_all(self, df):
        """Train a separate imputer for each target metric."""
        for metric, imputer in self.imputers.items():
            if metric not in df.columns:
                print(f"  {metric} not in dataset, skipping.")
                continue
            if df[metric].isna().all():
                print(f"  {metric} is entirely NaN, skipping.")
                continue
            try:
                imputer.train(df)
            except ValueError as e:
                print(f"  Skipping {metric}: {e}")

    def impute_all(self, df):
        """Apply all trained imputers sequentially."""
        for metric, imputer in self.imputers.items():
            if imputer.is_trained:
                df = imputer.impute(df)
        return df


if __name__ == "__main__":
    print("Testing Stage 0 Multi-Metric Imputer...")

    modern_data = pd.DataFrame(
        {
            "PLAYER_NAME": ["LeBron", "Curry", "Gobert", "Harden"],
            "PTS": [25.0, 30.0, 15.0, 36.0],
            "AST": [8.0, 6.5, 1.2, 7.5],
            "REB": [8.0, 5.0, 13.0, 6.6],
            "MP": [35.0, 34.0, 32.0, 36.5],
            "AGE": [35, 32, 28, 30],
            "POTENTIAL_AST": [15.2, 12.0, 2.1, 14.8],
            "PASSES_MADE": [50.1, 42.0, 20.5, 55.3],
        }
    )

    retro_data = pd.DataFrame(
        {
            "PLAYER_NAME": ["Nash", "Kobe", "Shaq"],
            "PTS": [15.5, 35.4, 22.9],
            "AST": [11.5, 4.5, 2.5],
            "REB": [3.3, 5.3, 10.4],
            "MP": [34.3, 41.0, 30.0],
            "AGE": [31, 27, 33],
            "POTENTIAL_AST": [np.nan, np.nan, np.nan],
            "PASSES_MADE": [np.nan, np.nan, np.nan],
        }
    )

    full_dataset = pd.concat([modern_data, retro_data], ignore_index=True)

    multi_imputer = MultiMetricImputer(["POTENTIAL_AST", "PASSES_MADE"])
    multi_imputer.train_all(full_dataset)
    clean_dataset = multi_imputer.impute_all(full_dataset)

    print("\nResulting Dataset with Imputed Values:")
    print(clean_dataset[["PLAYER_NAME", "AST", "POTENTIAL_AST", "PASSES_MADE"]])
