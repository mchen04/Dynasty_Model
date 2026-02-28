import numpy as np

from src.utils.config import load_config


class QuantileRegressorPool:
    """
    Stage 1 - Cross-Sectional Snapshot Processor (Probabilistic)

    Trains LightGBM quantile regressors for floor (10th), median (50th),
    and ceiling (90th) percentile predictions for a single (stat, horizon) pair.

    Uses Optuna for hyperparameter tuning and MAPIE for conformal calibration.
    """

    def __init__(self, target_name, horizon=1):
        self.target_name = target_name
        self.horizon = horizon
        self.models = {}
        self.calibrated_models = {}
        self.is_trained = False
        self.is_calibrated = False
        self._best_params = None

    def tune_hyperparams(self, X_train, y_train, n_trials=None):
        """Tune LightGBM hyperparameters with Optuna using TimeSeriesSplit CV."""
        import optuna
        from lightgbm import LGBMRegressor
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        config = load_config()
        if n_trials is None:
            n_trials = config["model"]["optuna_n_trials"]
        cv_splits = config["model"]["optuna_cv_splits"]

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            model = LGBMRegressor(
                objective="quantile", alpha=0.5, verbose=-1, **params
            )

            tscv = TimeSeriesSplit(n_splits=cv_splits)
            scores = cross_val_score(
                model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error"
            )
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self._best_params = study.best_params
        return self._best_params

    def fit(self, X_train, y_train, params=None):
        """
        Train floor/median/ceiling LightGBM quantile regressors.
        """
        from lightgbm import LGBMRegressor

        config = load_config()
        alphas = config["model"]["quantile_alphas"]

        if params is None and self._best_params is not None:
            params = self._best_params
        elif params is None:
            params = config["model"]["lgbm_defaults"]

        print(f"  Training quantile models for {self.target_name} T+{self.horizon}...")

        labels = ["floor", "median", "ceiling"]
        for alpha, label in zip(alphas, labels):
            model = LGBMRegressor(
                objective="quantile", alpha=alpha, verbose=-1, **params
            )
            model.fit(X_train, y_train)
            self.models[label] = model

        self.is_trained = True
        return self

    def calibrate(self, X_cal, y_cal):
        """Wrap with MAPIE for guaranteed coverage intervals."""
        try:
            from mapie.quantile_regression import MapieQuantileRegressor

            mapie = MapieQuantileRegressor(
                estimator=[
                    self.models["floor"],
                    self.models["median"],
                    self.models["ceiling"],
                ],
                cv="prefit",
                alpha=0.2,  # 80% coverage target
            )
            mapie.fit(X_cal, y_cal)
            self.calibrated_models["mapie"] = mapie
            self.is_calibrated = True
            print(f"  Calibrated {self.target_name} T+{self.horizon} with MAPIE.")
        except Exception as e:
            print(f"  MAPIE calibration failed for {self.target_name}: {e}")
            print("  Falling back to uncalibrated quantile predictions.")
        return self

    def predict(self, X_test):
        """Return {floor, median, ceiling} arrays."""
        if not self.is_trained:
            raise RuntimeError("Models must be trained before predicting.")

        if self.is_calibrated and "mapie" in self.calibrated_models:
            mapie = self.calibrated_models["mapie"]
            y_pred, y_pis = mapie.predict(X_test)
            return {
                "floor": y_pis[:, 0, 0],
                "median": y_pred,
                "ceiling": y_pis[:, 1, 0],
            }

        return {
            "floor": self.models["floor"].predict(X_test),
            "median": self.models["median"].predict(X_test),
            "ceiling": self.models["ceiling"].predict(X_test),
        }


class MultiTargetQuantilePool:
    """
    Manages all (stat, horizon) QuantileRegressorPool combinations.
    14 stats x 5 horizons = 70 pools, each with 3 quantile models = 210 total.
    """

    def __init__(self, target_names=None, horizons=None):
        config = load_config()
        if target_names is None:
            target_names = config["model"]["target_stats"]
        if horizons is None:
            horizons = config["model"]["horizons"]

        self.target_names = target_names
        self.horizons = horizons
        self.pools = {}

        for stat in target_names:
            for h in horizons:
                key = f"{stat}_T+{h}"
                self.pools[key] = QuantileRegressorPool(target_name=stat, horizon=h)

    def fit_all(self, X_train, y_targets, tune_first_only=True):
        """
        Train all pools. y_targets is a dict mapping 'STAT_T+H' -> target array.
        If tune_first_only, Optuna tuning runs on the first pool and params are reused.
        """
        shared_params = None

        for key, pool in self.pools.items():
            if key not in y_targets:
                continue

            y = y_targets[key]
            valid = ~np.isnan(y)
            if valid.sum() < 50:
                print(f"  Skipping {key}: insufficient non-NaN targets ({valid.sum()})")
                continue

            X_valid = X_train[valid]
            y_valid = y[valid]

            if tune_first_only and shared_params is None:
                print(f"  Tuning hyperparams on {key}...")
                shared_params = pool.tune_hyperparams(X_valid, y_valid)

            pool.fit(X_valid, y_valid, params=shared_params)

    def calibrate_all(self, X_cal, y_cal_targets):
        """Calibrate all trained pools with MAPIE."""
        for key, pool in self.pools.items():
            if not pool.is_trained:
                continue
            if key not in y_cal_targets:
                continue

            y = y_cal_targets[key]
            valid = ~np.isnan(y)
            if valid.sum() < 20:
                continue

            pool.calibrate(X_cal[valid], y[valid])

    def predict_all(self, X_test):
        """Return dict of predictions: key -> {floor, median, ceiling}."""
        results = {}
        for key, pool in self.pools.items():
            if pool.is_trained:
                results[key] = pool.predict(X_test)
        return results


if __name__ == "__main__":
    print("Testing Quantile Regression Pool with LightGBM...")

    X = np.random.rand(500, 10) * 30
    noise = np.random.normal(0, 5, 500)
    y = X[:, 0] * 1.5 + noise

    pool = QuantileRegressorPool(target_name="PTS", horizon=1)
    pool.fit(X, y)

    rookie = np.array([[15.0, 19.0, 25.0, 22.0, 1.0, 5.0, 3.0, 2.0, 0.5, 0.3]])
    forecasts = pool.predict(rookie)

    print(f"\nFloor (10th):  {forecasts['floor'][0]:.1f}")
    print(f"Median (50th): {forecasts['median'][0]:.1f}")
    print(f"Ceiling (90th): {forecasts['ceiling'][0]:.1f}")
