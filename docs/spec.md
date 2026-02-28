# Dynasty Model — NBA Fantasy Dynasty Player Valuation Engine

## Overview

Predicts two values for any NBA player:

1. **Auction Startup Value** ($1–$200) — startup dynasty auction price
2. **Trade Value** (0–100) — relative dynasty asset value, 100 = most valuable

Both are **dynasty-aware**: a 22-year-old averaging 18 PPG is worth more than a 33-year-old averaging 22 PPG. League-configurable with ESPN H2H Points defaults.

---

## Architecture: Three-Stage Pipeline

### Stage 0 — Data Harmonization & Imputation

Backcasts missing tracking stats (Potential Assists, Passes Made, Assist Points Created) for the pre-2014 era. HistGradientBoosting models train on 2014+ data (~27 features) and predict backwards to 2001, producing an unbroken dataset. Separate models per metric.

### Stage 1 — Probabilistic Stat Projection (League-Agnostic)

Predicts floor (10th), median (50th), and ceiling (90th) percentile outcomes for **14 stats across 5 horizons** (T+1 through T+5):
- **Efficiency**: PTS, REB, AST, STL, BLK, TOV, FGM, FGA, FTM, FTA, 3PM, 3PA
- **Opportunity**: MP, G

**LightGBM quantile regressors** handle the cross-sectional snapshot (210 models: 14 stats × 5 horizons × 3 quantiles). Tuned via Optuna with TimeSeriesSplit CV, calibrated with MAPIE for guaranteed coverage intervals. NaN-native — no zero-filling.

**Time-Series Transformer** (4-head, 2-layer, GELU, 64-dim) processes career sequences via self-attention, outputting a trajectory embedding that captures breakouts and decline.

**Ensembler** blends tree predictions with trajectory corrections via per-stat learned alpha weights and an adjustment MLP. Monotonicity enforcement (sort) guarantees floor ≤ median ≤ ceiling.

### Stage 2 — Valuation Layer (League-Specific)

Pure math, no ML. Converts projections to dollars/trade values:

1. Apply scoring weights → fantasy points per game
2. Multiply by projected GP with **shutdown risk discounts**:
   - Age ≥ 34: −15% (load management)
   - Contender (W > 55) + age ≥ 30: −10% (resting)
   - Tanking (W < 25): −10% (shutdown)
   - Future years: −3% per year beyond T+1
3. Discount future seasons (year 1: 1.0, year 2: 0.85, year 3: 0.72, year 4: 0.61, year 5: 0.52)
4. Sum → dynasty FPTS. Subtract **positional replacement level** (blended for multi-position eligibility) → VORP
5. Auction dollars: `player_$ = (VORP / total_VORP_pool) × league_budget`
6. Trade value: power curve `100 × max(0, VORP)^1.2 / max_vorp^1.2` (consolidation premium)
7. **Roster-context mode**: re-weight for punt builds, yielding Market Value, Roster Value, and VALUE_GAP (buy targets)

---

## League Configuration

| Parameter | Default | Range |
|-----------|---------|-------|
| `num_teams` | 10 | 8–20 |
| `budget_per_team` | $200 | $100–$500 |
| `roster_size` | 13 | 10–20 |
| `dynasty_horizon` | 5 years | 1–8 |
| `dynasty_discount_rate` | 0.85/yr | 0.5–1.0 |
| `consolidation_premium` | 1.2 | 1.0–1.5 |

### ESPN H2H Points Scoring

| Stat | Pts | Stat | Pts | Stat | Pts |
|------|-----|------|-----|------|-----|
| PTS | +1 | FGM | +2 | FTM | +1 |
| REB | +1 | FGA | −1 | FTA | −1 |
| AST | +2 | 3PM | +1 | | |
| STL | +4 | TOV | −2 | | |
| BLK | +4 | | | | |

---

## Feature Set

| Group | Features |
|-------|----------|
| **Box Score** | PTS, REB, AST, STL, BLK, TOV, FGM/A, 3PM/A, FTM/A, MP, G, GS, PF — per-game and per-100 |
| **Advanced** | PER, TS%, eFG%, BPM/OBPM/DBPM, VORP, WS/OWS/DWS/WS48, USG%, AST%/TOV%/REB%, FTr, 3PAr |
| **Era-Adjusted** | Within-season z-scores for all counting stats: `z = (stat − season_mean) / season_std` |
| **Biometric** | Age, height, weight, wingspan, standing reach, draft year/round/pick, experience, college |
| **Injury** | Games missed (season-length-aware: 66/73/72 for lockout/bubble/COVID), GP%, rolling 1/2/3-season missed games, injury-prone flag (40+ missed in 3yr) |
| **Age Curve** | Age², years from position-specific peak (PG/SG/SF: 27, PF/C: 26), pre-peak flag, contract year proxy |
| **Team Context** | W-L, ORtg/DRtg/NRtg, Pace (decayed toward league average for T+2+) |
| **Network** | Depth chart rank, usage vacuums (departed teammate USG%), teammate TS% (minute-weighted), coach ID |
| **Trajectory** | Full career season sequences (pre-padded, masked) for transformer input |
| **Momentum** | Season-over-season deltas, 3yr EWMA, breakout flags (>1σ delta) — all lagged |

---

## Training

**Data**: 2001-02 to present ("Modern NBA" — zone defense legalized). All players included regardless of minutes to prevent survivorship bias.

**Walk-forward CV**: For each test year, train on all prior seasons. Fit/calibration split within training data (last 15% of seasons held for MAPIE calibration). Optuna tuning on first fold only, params reused.

**Targets**: NaN left as NaN (career endings are structurally informative). LightGBM handles natively; transformer uses padding masks.

---

## Evaluation

- **Quantile calibration**: ~80% of actuals within floor–ceiling, per stat and horizon
- **Dynasty backtesting**: Spearman rank correlation (predicted dynasty value vs realized 3yr FPTS), RMSE, top-30 hit rate (% in actual top-50)
- **Trade decisions**: Pairwise accuracy on similarly-valued players (within 30%) — did the model correctly identify who produced more?

---

## Project Structure

```
Dynasty_Model/
├── configs/
│   └── config.yaml               # All paths, league defaults, hyperparameters
├── docs/
│   └── spec.md
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── utils/
│   │   ├── config.py              # load_config(), resolve_path()
│   │   ├── constants.py           # BBR_TEAM_MAPPING, COLUMN_RENAME_MAP
│   │   └── data_loader.py         # load_raw_dataset() with auto-renames
│   ├── scraping/
│   │   ├── bball_ref_scraper.py
│   │   ├── build_dataset.py
│   │   └── nba_api_client.py
│   ├── preprocessing/
│   │   ├── run_pipeline.py
│   │   ├── advanced_feature_engineer.py
│   │   ├── imputer.py             # BackcastImputer, MultiMetricImputer
│   │   ├── zscores.py
│   │   ├── network_effects.py
│   │   └── sequence_builder.py
│   ├── models/
│   │   ├── train.py               # Walk-forward CV
│   │   ├── quantile_regressor.py  # LightGBM + MAPIE
│   │   ├── transformer.py
│   │   └── ensembler.py
│   ├── valuation/
│   │   └── vorp_calculator.py     # DynastyValuationEngine
│   └── evaluation/
│       └── evaluator.py
├── pyproject.toml
└── requirements.txt
```
