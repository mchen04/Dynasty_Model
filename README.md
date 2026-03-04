# Dynasty Model - Results

NBA dynasty fantasy basketball projection model. Walk-forward cross-validation across 10 seasons (2017-2026).

## Architecture

- **Stage 0**: Feature imputation (tracking stats for pre-2014 seasons)
- **Stage 1**: LightGBM quantile regressors (14 stats x 5 horizons x 3 quantiles = 210 models) + Transformer trajectory embeddings + learned ensembler
- **Stage 2**: VORP-based dynasty valuation with positional replacement levels

Conformal calibration via MAPIE on held-out calibration sets.

## T+1 Results (Next Season)

| Stat | RMSE | MAE | Floor Cal | Ceil Cal |
|------|------|-----|-----------|----------|
| PTS | 3.276 | 2.534 | 79.2% | 69.6% |
| REB | 1.329 | 0.980 | 78.5% | 77.8% |
| AST | 0.873 | 0.615 | 81.7% | 79.1% |
| STL | 0.307 | 0.228 | 81.2% | 78.9% |
| BLK | 0.249 | 0.169 | 80.3% | 83.7% |
| TOV | 0.451 | 0.341 | 80.8% | 80.2% |
| FGM | 1.153 | 0.898 | 77.5% | 74.5% |
| FGA | 2.418 | 1.867 | 73.1% | 74.5% |
| FTM | 0.718 | 0.517 | 83.7% | 79.0% |
| FTA | 0.872 | 0.632 | 82.3% | 79.7% |
| 3PM | 0.477 | 0.348 | 85.7% | 86.2% |
| 3PA | 1.173 | 0.873 | 79.2% | 81.6% |
| MP | 6.058 | 4.659 | 87.5% | 80.8% |
| G | 23.675 | 18.612 | 87.8% | 90.8% |

## T+2 Results (2 Seasons Out)

| Stat | RMSE | MAE | Floor Cal | Ceil Cal |
|------|------|-----|-----------|----------|
| PTS | 5.426 | 4.533 | 89.0% | 90.2% |
| REB | 2.149 | 1.715 | 88.6% | 89.9% |
| AST | 1.456 | 1.159 | 89.2% | 90.0% |
| MP | 10.321 | 8.303 | 89.1% | 90.0% |

## T+3 Results (3 Seasons Out)

| Stat | RMSE | MAE | Floor Cal | Ceil Cal |
|------|------|-----|-----------|----------|
| PTS | 6.591 | 5.621 | 87.0% | 91.2% |
| REB | 2.329 | 1.908 | 86.9% | 91.4% |
| AST | 1.557 | 1.258 | 90.7% | 88.0% |
| MP | 11.372 | 9.276 | 87.3% | 91.4% |

## vs Baseline (T+1 Only, LightGBM-only, Feb 27 2025)

| Stat | Baseline MAE | Current MAE | Delta |
|------|-------------|-------------|-------|
| PTS | 2.246 | 2.534 | +0.288 |
| REB | 0.894 | 0.980 | +0.086 |
| AST | 0.577 | 0.615 | +0.038 |
| MP | 4.039 | 4.659 | +0.620 |

## Per-Fold PTS T+1

| Year | RMSE | MAE | Floor Cal | Ceil Cal |
|------|------|-----|-----------|----------|
| 2017 | 3.260 | 2.527 | 83.4% | 65.2% |
| 2018 | 3.383 | 2.558 | 79.4% | 74.1% |
| 2019 | 3.363 | 2.608 | 76.6% | 67.8% |
| 2020 | 3.451 | 2.657 | 84.7% | 71.6% |
| 2021 | 3.089 | 2.382 | 68.9% | 75.6% |
| 2022 | 3.392 | 2.654 | 68.1% | 75.7% |
| 2023 | 3.206 | 2.520 | 84.2% | 69.8% |
| 2024 | 2.961 | 2.359 | 81.5% | 68.1% |
| 2025 | 3.549 | 2.713 | 81.2% | 67.9% |
| 2026 | 3.103 | 2.361 | 83.7% | 59.7% |

## Notes

- **Calibration target**: Floor and ceiling should both be ~90% for proper 80% prediction intervals
- **T+1 calibration is too tight**: Floor avg 81.3%, ceiling avg 79.7% — model is overconfident at 1-year horizon
- **T+2/T+3 calibration is good**: Both ~89-91%, right on target
- **Games (G) is the weakest stat**: MAE ~19 games due to inherent injury unpredictability
- **Baseline comparison**: Current model trades slightly worse point-prediction accuracy for multi-horizon capability (14 stats x 5 horizons vs baseline's 4 stats x 1 horizon)
- Run date: March 1, 2026
