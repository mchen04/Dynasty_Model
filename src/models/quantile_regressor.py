import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class QuantileRegressorPool:
    """
    Stage 1 - Cross-Sectional Snapshot Processor (Probabilistic)

    Instead of predicting a single point estimate (Mean) for T+1, this ensemble
    trains unique Gradient Boosters for the 10th (Floor), 50th (Median),
    and 90th (Ceiling) percentiles.

    Crucial for Dynasty Value: Rebuilding teams want to gamble on players with
    high 90th percentile ceilings. Contenders want safe 10th percentile floors.
    """

    def __init__(self, target="PTS"):
        self.target = target
        # In a production environment, you'd use LightGBM or NGBoost for speed/native quantile support.
        # Scikit-learn's GBR supports quantile loss natively as well.
        self.models = {
            "floor_10th": GradientBoostingRegressor(
                loss="quantile", alpha=0.1, n_estimators=100, max_depth=4
            ),
            "median_50th": GradientBoostingRegressor(
                loss="quantile", alpha=0.5, n_estimators=100, max_depth=4
            ),
            "ceiling_90th": GradientBoostingRegressor(
                loss="quantile", alpha=0.9, n_estimators=100, max_depth=4
            ),
        }
        self.is_trained = False

    def fit(self, X_train, y_train):
        """
        Trains all three percentile regressors on the provided dataset.
        """
        print(f"Training Quantile Distribution Models for {self.target}...")

        for name, model in self.models.items():
            print(f"  Fitting {name} model...")
            model.fit(X_train, y_train)

        self.is_trained = True
        return self

    def predict(self, X_test):
        """
        Generates Floor, Median, and Ceiling predictions for a player.
        Returns a dictionary or dataframe of these bounds.
        """
        if not self.is_trained:
            raise RuntimeError("Models must be trained before predicting.")

        predictions = {
            "10th_Percentile": self.models["floor_10th"].predict(X_test),
            "50th_Percentile": self.models["median_50th"].predict(X_test),
            "90th_Percentile": self.models["ceiling_90th"].predict(X_test),
        }

        return predictions


if __name__ == "__main__":
    print("Testing Quantile Regression Pool...")

    # 1. Feature Matrix (e.g., PTS, AGE, MIN, USG_PCT)
    X = np.random.rand(500, 4) * 30  # 500 historical player-seasons

    # 2. Target Variable (Their PTS outcome in year T+1)
    # Give it some variance so ceilings and floors exist
    noise = np.random.normal(0, 5, 500)
    y = X[:, 0] * 1.5 + noise

    # Init and Train
    q_pool = QuantileRegressorPool(target="T+1_PTS")
    q_pool.fit(X, y)

    # Test a rookie prospect
    rookie_features = np.array([[15.0, 19.0, 25.0, 22.0]])

    # Forecast
    forecasts = q_pool.predict(rookie_features)

    print(
        "\nDynasty T+1 Forecast for Rookie Prospect [Base Stats: 15.0 PTS, 19yrs old, 25 Mins, 22 USG%]:"
    )
    print(f"Floor (10th)   : {forecasts['10th_Percentile'][0]:.1f} projected pts")
    print(f"Median (50th)  : {forecasts['50th_Percentile'][0]:.1f} projected pts")
    print(f"Ceiling (90th) : {forecasts['90th_Percentile'][0]:.1f} projected pts")
