import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class BackcastImputer:
    """
    Stage 0 - Data Harmonization & Imputation.

    This model trains exclusively on Modern Era data (2014-Present) where tracking
    stats (like Potential Assists) exist. It learns the relationship between standard
    play-by-play/box score stats and these advanced metrics.

    It is then used to predict (backfill) these tracking metrics for players from
    2001-2013, ensuring an unbroken long-term dataset for the temporal model.
    """

    def __init__(self, target_metric):
        self.target_metric = target_metric
        # HistGradientBoosting handles NaN values natively, exactly what we want
        self.model = HistGradientBoostingRegressor(
            max_iter=200, random_state=42, early_stopping=True
        )
        self.is_trained = False

    def _select_features(self, df):
        """
        Selects standard box score and play-by-play features available since 2001.
        """
        # In a full implementation, this list would be comprehensive.
        # Here we use the core statistical footprint that proxies tracking data.
        core_features = [
            "PTS",
            "AST",
            "REB",
            "STL",
            "BLK",
            "TOV",
            "FGA",
            "FGM",
            "FTA",
            "FTM",
            "3PA",
            "3PM",
            "MIN",
            "AGE",
        ]

        # Keep only features that actually exist in the dataframe
        available_features = [f for f in core_features if f in df.columns]
        return df[available_features]

    def train(self, df):
        """
        Train the imputation model using rows where the target_metric is NOT null.
        """
        print(f"Training imputer for missing metric: {self.target_metric}")

        # Filter to rows where target exists (e.g., post-2014 players)
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

        # Evaluate
        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"Model trained. Validation RMSE = {rmse:.4f}, R2 = {r2:.4f}")
        self.is_trained = True
        return self

    def impute(self, df):
        """
        Predict the target metric for rows where it is currently NaN.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before imputation.")

        imputed_df = df.copy()

        # Find rows missing the target metric (e.g., pre-2014 players)
        missing_mask = imputed_df[self.target_metric].isna()
        num_missing = missing_mask.sum()

        if num_missing == 0:
            print("No missing values found to impute.")
            return imputed_df

        print(f"Imputing {num_missing} missing rows for {self.target_metric}...")

        X_missing = self._select_features(imputed_df[missing_mask])

        # Fill missing values with predictions
        predictions = self.model.predict(X_missing)
        imputed_df.loc[missing_mask, self.target_metric] = predictions

        return imputed_df


if __name__ == "__main__":
    # --- Integration Test / Demonstration ---
    print("Testing Stage 0 Imputer Logic...")

    # Simulate some 2014+ data (has potential assists)
    modern_data = pd.DataFrame(
        {
            "PLAYER_NAME": ["LeBron", "Curry", "Gobert", "Harden"],
            "PTS": [25.0, 30.0, 15.0, 36.0],
            "AST": [8.0, 6.5, 1.2, 7.5],
            "REB": [8.0, 5.0, 13.0, 6.6],
            "MIN": [35.0, 34.0, 32.0, 36.5],
            "AGE": [35, 32, 28, 30],
            "POTENTIAL_AST": [15.2, 12.0, 2.1, 14.8],  # Valid targets
        }
    )

    # Simulate some 2005 data (tracking didn't exist, potential assists are NaN)
    retro_data = pd.DataFrame(
        {
            "PLAYER_NAME": ["Nash", "Kobe", "Shaq"],
            "PTS": [15.5, 35.4, 22.9],
            "AST": [
                11.5,
                4.5,
                2.5,
            ],  # Nash had huge AST, should predict huge POTENTIAL_AST
            "REB": [3.3, 5.3, 10.4],
            "MIN": [34.3, 41.0, 30.0],
            "AGE": [31, 27, 33],
            "POTENTIAL_AST": [np.nan, np.nan, np.nan],  # Missing!
        }
    )

    # Combine
    full_dataset = pd.concat([modern_data, retro_data], ignore_index=True)

    # Train imputer
    imputer = BackcastImputer(target_metric="POTENTIAL_AST")
    # Use modern data to train since it has the target
    imputer.train(modern_data)

    # Impute the missing retro data
    clean_dataset = imputer.impute(full_dataset)

    print("\nResulting Dataset with Imputed Values:")
    print(clean_dataset[["PLAYER_NAME", "AST", "POTENTIAL_AST"]])
