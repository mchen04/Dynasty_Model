import numpy as np
import pandas as pd
from scipy import stats


class DynastyEvaluator:
    """
    Evaluation & backtesting for the Dynasty Model.

    Provides real metrics instead of mocked values:
    - Quantile coverage calibration
    - Spearman rank correlation for dynasty value predictions
    - Top-N hit rate
    - Pairwise trade decision accuracy
    """

    def evaluate_quantile_coverage(self, y_true, y_pred_floor, y_pred_median, y_pred_ceiling):
        """
        Tests if probabilistic forecasting is properly calibrated.
        Roughly 80% of actual outcomes should fall between floor and ceiling.
        """
        valid = ~(np.isnan(y_true) | np.isnan(y_pred_floor) | np.isnan(y_pred_ceiling))
        y_t = y_true[valid]
        y_f = y_pred_floor[valid]
        y_c = y_pred_ceiling[valid]

        within_bounds = np.logical_and(y_t >= y_f, y_t <= y_c)
        coverage_pct = np.mean(within_bounds) * 100

        above_floor_pct = np.mean(y_t >= y_f) * 100
        below_ceiling_pct = np.mean(y_t <= y_c) * 100

        print(f"  Coverage (10th-90th): {coverage_pct:.1f}% (target ~80%)")
        print(f"  Above floor: {above_floor_pct:.1f}%, Below ceiling: {below_ceiling_pct:.1f}%")

        return {
            "coverage": coverage_pct,
            "above_floor": above_floor_pct,
            "below_ceiling": below_ceiling_pct,
            "n_samples": int(valid.sum()),
        }

    def backtest_dynasty_accuracy(
        self, predicted_dynasty_values, realized_fpts_3yr, top_n=30, top_actual_n=50
    ):
        """
        Real backtesting of dynasty value predictions against realized 3-year FPTS.

        Returns:
        - Spearman rank correlation
        - RMSE of predicted vs actual total FPTS
        - Top-N hit rate: % of model's top-N that were actually top-M
        """
        valid = ~(
            np.isnan(predicted_dynasty_values) | np.isnan(realized_fpts_3yr)
        )

        pred = predicted_dynasty_values[valid]
        actual = realized_fpts_3yr[valid]

        if len(pred) < 10:
            print("  Insufficient data for backtesting.")
            return None

        # Spearman rank correlation
        spearman_r, spearman_p = stats.spearmanr(pred, actual)

        # RMSE
        rmse = np.sqrt(np.mean((pred - actual) ** 2))

        # Top-N hit rate
        pred_top_indices = set(np.argsort(pred)[-top_n:])
        actual_top_indices = set(np.argsort(actual)[-top_actual_n:])
        hit_rate = len(pred_top_indices & actual_top_indices) / top_n * 100

        print(f"  Spearman rank correlation: {spearman_r:.3f} (p={spearman_p:.4f})")
        print(f"  RMSE (predicted vs realized 3yr FPTS): {rmse:.1f}")
        print(f"  Top-{top_n} hit rate (in actual top-{top_actual_n}): {hit_rate:.1f}%")

        return {
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "rmse": rmse,
            "top_n_hit_rate": hit_rate,
            "n_samples": int(valid.sum()),
        }

    def evaluate_trade_decisions(
        self, predicted_values, realized_values, value_threshold=0.30
    ):
        """
        Samples pairs of similarly-valued players (within threshold of each other)
        and checks if the model correctly identified which produced more.

        Returns pairwise accuracy percentage.
        """
        valid = ~(np.isnan(predicted_values) | np.isnan(realized_values))
        pred = predicted_values[valid]
        actual = realized_values[valid]
        n = len(pred)

        if n < 20:
            print("  Insufficient data for trade decision evaluation.")
            return None

        correct = 0
        total_pairs = 0

        # Sample pairs where predicted values are within threshold
        indices = np.arange(n)
        np.random.seed(42)

        for _ in range(min(5000, n * (n - 1) // 2)):
            i, j = np.random.choice(indices, size=2, replace=False)

            # Only compare similarly-valued players
            max_val = max(abs(pred[i]), abs(pred[j]), 1e-6)
            if abs(pred[i] - pred[j]) / max_val > value_threshold:
                continue

            total_pairs += 1

            # Did the model correctly rank who would produce more?
            pred_prefers_i = pred[i] > pred[j]
            actual_better_i = actual[i] > actual[j]

            if pred_prefers_i == actual_better_i:
                correct += 1

        if total_pairs == 0:
            print("  No similar-value pairs found for trade evaluation.")
            return None

        accuracy = correct / total_pairs * 100
        print(f"  Trade decision accuracy: {accuracy:.1f}% ({correct}/{total_pairs} pairs)")

        return {
            "accuracy": accuracy,
            "correct_pairs": correct,
            "total_pairs": total_pairs,
        }

    def full_evaluation(self, results_dict):
        """
        Run all evaluation metrics given a results dictionary.

        Expected keys:
        - y_true, y_pred_floor, y_pred_median, y_pred_ceiling (per-stat arrays)
        - predicted_dynasty_values, realized_fpts_3yr (for backtesting)
        """
        print("\n=== Dynasty Model Evaluation ===\n")

        metrics = {}

        if all(k in results_dict for k in ["y_true", "y_pred_floor", "y_pred_ceiling"]):
            print("Quantile Coverage:")
            metrics["coverage"] = self.evaluate_quantile_coverage(
                results_dict["y_true"],
                results_dict["y_pred_floor"],
                results_dict.get("y_pred_median", np.zeros_like(results_dict["y_true"])),
                results_dict["y_pred_ceiling"],
            )

        if all(k in results_dict for k in ["predicted_dynasty_values", "realized_fpts_3yr"]):
            print("\nDynasty Value Backtesting:")
            metrics["backtest"] = self.backtest_dynasty_accuracy(
                results_dict["predicted_dynasty_values"],
                results_dict["realized_fpts_3yr"],
            )

            print("\nTrade Decision Evaluation:")
            metrics["trade_decisions"] = self.evaluate_trade_decisions(
                results_dict["predicted_dynasty_values"],
                results_dict["realized_fpts_3yr"],
            )

        return metrics


if __name__ == "__main__":
    print("Testing Dynasty Evaluator...\n")

    evaluator = DynastyEvaluator()

    np.random.seed(42)
    n = 200

    # Simulated predictions and actuals
    true_talent = np.random.normal(20, 8, n)
    noise = np.random.normal(0, 3, n)

    y_true = true_talent + noise
    y_pred_floor = true_talent - 5
    y_pred_median = true_talent
    y_pred_ceiling = true_talent + 5

    print("1. Quantile Coverage:")
    evaluator.evaluate_quantile_coverage(y_true, y_pred_floor, y_pred_median, y_pred_ceiling)

    print("\n2. Dynasty Value Backtesting:")
    predicted_dynasty = true_talent * 3 + np.random.normal(0, 5, n)
    realized_3yr = true_talent * 3 + np.random.normal(0, 8, n)
    evaluator.backtest_dynasty_accuracy(predicted_dynasty, realized_3yr)

    print("\n3. Trade Decision Evaluation:")
    evaluator.evaluate_trade_decisions(predicted_dynasty, realized_3yr)
