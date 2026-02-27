# Dynasty Model — NBA Fantasy Dynasty Player Valuation Engine

## Overview

An ML model trained from scratch that predicts two values for any NBA player:

1. **Auction Startup Value** — what a player is worth in a fresh ESPN dynasty auction draft ($0–$200 scale)
2. **Current Trade Value** — what a player is worth mid-season in an existing dynasty league (normalized 0–100 scale)

The model is **league-configurable**: users can adjust team count, scoring system, and roster construction. Defaults to ESPN standard Head-to-Head (H2H) Points settings.

---

## Target Outputs

| Output | Scale | Description |
|--------|-------|-------------|
| `auction_value` | $1–$200 | Startup auction price (sum across all players = $200 × num_teams) |
| `trade_value` | 0–100 | Relative dynasty asset value, where 100 = most valuable player in the league |

Both outputs are **dynasty-aware**: they weight long-term future production, not just current-season projections. A 22-year-old averaging 18 PPG is worth more than a 33-year-old averaging 22 PPG.

---

## Architecture: Three-Stage Pipeline

### Stage 0 — Data Harmonization & Imputation

Handles missing positional data and advanced tracking stats for the pre-2014 era.
- **Backcast Imputation Model**: Trains an auxiliary model exclusively on 2014–Present data predicting advanced metrics (RAPTOR, EPM, Tracking metrics like potential assists) from traditional play-by-play and box scores. 
- Applies these predictions backwards to impute accurate spatial/impact data for players from 2001-2013, ensuring an unbroken long-term dataset for the temporal model.

### Stage 1 — Probabilistic Stat Projection Model (League-Agnostic)

Predicts **probabilistic distributions** of per-game stats across future seasons, yielding 10th (Floor), 50th (Median), and 90th (Ceiling) percentile outcomes.

Crucially, **Stage 1 predicts Efficiency (per-100 possessions) and Opportunity (Minutes Share) separately.**
Raw per-game stats are derived by multiplying these two predictions.

- **Input**: player features (see Feature Set below)
- **Output**: Multi-quantile projected per-100 stats, minutes, and GP for seasons T+1 through T+5.
- **Model**: Ensemble of Quantile Regression (e.g. NGBoost/LightGBM) + Time-Series Transformer
  - The tree model handles the cross-sectional snapshot.
  - The **Time-Series Transformer** elegantly processes a player's temporal career arc, using attention mechanisms to weigh specific developmental leaps and injury-shortened seasons simultaneously.
  - Final prediction = probabilistic blend with ceiling/floor confidence intervals.
- Trained once, shared across all league formats

### Stage 2 — Configurable Valuation Layer (League-Specific)

Converts projected stat lines into dollar/trade values based on league settings.

- **Inputs**: Stage 1 projections + league configuration parameters
- **Process**:
  1. Apply scoring weights to projected stats → fantasy points per game
  2. Weight fantasy points by projected games played, discounting for 'silly season' shutdown risks based on age and team context.
  3. Discount future seasons using a dynasty time-value decay (year 1 = 1.0, year 2 = 0.85, year 3 = 0.72, year 4 = 0.61, year 5 = 0.52 — tunable)
  4. Sum discounted values → total dynasty asset value
  5. Compute positional replacement level based on league size, roster slots, and positional fluidity.
  6. Calculate Value Over Replacement Player (VORP)
  7. **Roster-Context Evaluation**: Recalculate VORP specifically for arbitrary team builds (Punt categories) by applying category weights. Yields two metrics: *Market Value* and *Roster Value*.
  8. Convert VORP to auction dollars: `player_$ = (player_VORP / total_VORP_pool) × total_league_budget`
  9. For trade value: normalize using a **non-linear power curve** to account for roster consolidation premiums (2-for-1 trades).
- No ML training required — pure math, instantly recalculated for any league config based on owner's exact roster context.

---

## League Configuration Parameters

| Parameter | Default (ESPN H2H Points) | Configurable Range |
|-----------|------------------------|-------------------|
| `num_teams` | 10 | 8, 10, 12, 14, 16, 20 |
| `scoring_type` | `"points"` | `"points"`, `"custom"` |
| `budget_per_team` | $200 | $100–$500 |
| `roster_size` | 13 | 10–20 |
| `roster_slots` | PG, SG, SF, PF, C, G, F, UTIL×3, BN×3 | Custom |
| `dynasty_horizon` | 5 years | 1–8 years |
| `dynasty_discount_rate` | 0.85 per year | 0.5–1.0 |
| `consolidation_premium` | 1.2 | 1.0 - 1.5 |

### ESPN Default Points Scoring

| Stat | Points | Net Effect |
|------|--------|------------|
| PTS | +1 | — |
| REB | +1 | — |
| AST | +2 | — |
| STL | +4 | — |
| BLK | +4 | — |
| 3PM | +1 | — |
| TO | -2 | — |
| FGM | +2 | Made 2pt FG net = +3 (1 PTS + 2 FGM - 1 FGA) |
| FGA | -1 | Made 3pt FG net = +5 (3 PTS + 1 3PM + 2 FGM - 1 FGA) |
| FTM | +1 | Made FT net = +1 (1 PTS + 1 FTM - 1 FTA) |
| FTA | -1 | Missed FT net = -1 (0 PTS + 0 FTM - 1 FTA) |
| DD | +2 | Double-double bonus |
| TD | +5 | Triple-double bonus |

---

## Feature Set (Stage 1 Inputs)

Everything. Organized into feature groups:

### A. Core Box Score Stats (Per-Game + Per-100-Possessions)
- Points, rebounds (offensive + defensive), assists, steals, blocks, turnovers
- FGM/FGA/FG%, 3PM/3PA/3P%, FTM/FTA/FT%
- Minutes per game, games played, games started
- Personal fouls
- All stats in both raw per-game AND per-100-possessions (pace-adjusted) forms

### B. Advanced Individual Metrics
- **Efficiency**: PER, True Shooting % (TS%), Effective FG% (eFG%)
- **Impact**: BPM (Box Plus/Minus), OBPM, DBPM, VORP, Win Shares (WS), OWS, DWS, WS/48
- **Usage**: Usage Rate (USG%), Assist Rate, Turnover Rate, Rebound Rate (ORB%, DRB%, TRB%)
- **Shooting**: Free Throw Rate (FTr), 3-Point Attempt Rate (3PAr)
- **Modern Plus-Minus**: RAPTOR (offensive, defensive, total) from FiveThirtyEight, EPM from DunksAndThrees (where available, 2014+)

### C. Era-Adjusted Stats
All counting stats are also stored as **within-season z-scores** — how many standard deviations above/below the league average for that season. This handles scoring inflation, pace changes, and the 3-point revolution automatically.

Formula: `z_stat = (player_stat - season_mean) / season_std`

### D. Biometric & Draft Profile
- Age (exact, in years + days)
- Height, weight, wingspan, standing reach (from combine data where available)
- NBA draft year, pick number, draft position (1st round, 2nd round, undrafted)
- Years of NBA experience
- College (or international origin)
- Years in college before declaring (1-and-done vs 4-year player)

### E. Injury History
- Career games missed (total and by season)
- Games-played percentage: `games_played / possible_games` per season
- Rolling games missed in last 1, 2, 3 seasons
- Injury type encoding (one-hot by body region)
- Injury severity encoding
- Career-ending injury risk flag (ACL, Achilles)

### F. Team Context (Decayed for Long-Term Forecasting)
- Team win-loss record (current season)
- Team offensive/defensive/net rating
- Team pace
- **Note:** For multi-year forecasting (T+2 and beyond), these features are artificially decayed back towards the league average, as team context is highly volatile long-term.

### G. Role, Opportunity & Network Effects
- Depth chart position
- Minutes share: `player_minutes / team_total_minutes`
- Usage rate & Passer Rating (tracker data)
- **Usage Vacuums (Network Effect)**: `team_vacated_usage` measuring how much total usage rate from the prior season's roster is leaving in free agency/trades, creating a "development vacuum" for remaining young players.
- **Coaching Tags**: One-hot/Categorical embedding of the Head Coach and offensive system to boost/penalize pace and rotation tightness.

### H. Historical Trajectory Features (Transformer Sequence Inputs)
For the LSTM component, feed a **sequence of season-level feature vectors** representing the player's entire career:
- Per-season: all stats from groups A–G aggregated at the season level
- Padded/masked for players with fewer seasons

### I. Rolling Momentum Features
- Last 10/20/40 game rolling averages for key stats
- Trend direction and breakout detection.

---

## Training Data

### Era Definition: "Modern NBA" = 2001-02 to Present
- Zone defense was legalized in 2001-02.
- Fits seamlessly with Era-Adjusted stats (Z-scores per season).

### Data Pipeline & Survivor Bias Prevention
The training dataset includes **all players** who logged an official NBA minute. Filtering out low-minute players or rapid wash-outs creates extreme survivorship bias, destroying the LSTM's ability to recognize "bust" trajectories. The target population is the entire drafting pool.

### Target Variable Strategy
The Stage 1 model predicts **raw stats via Efficiency and Opportunity**, not fantasy points or dollar values directly.

Multiple Output Regression Strategy:
1. Predict `Per-100 Stats` (Efficiency/Skill)
2. Predict `Minutes Share` (Opportunity)
3. Predict `Current Team Context` (Pace)
4. Derive `Per-Game Stats` mathematically.

---

## Model Training Details

### Quantile Regression Component
- **Architecture**: NGBoost, CatBoost, or LightGBM with quantile error functions.
- **Features**: all tabular features from groups A–I.
- **Hyperparameter tuning**: Bayesian optimization (Optuna) with 5-fold time-series cross-validation.

### Time-Series Transformer Component
- **Architecture**: Transformer encoder blocks (multi-head self-attention) adapted for tabular series.
- **Training Strategy**: Uses positional encoding to gracefully handle missing/shortened injury seasons. Explicitly fed historical busts to prevent survivorship bias and train accurate 10th-percentile (floor) outcomes.

### Ensemble Blend
- `final_prediction = α × xgb_prediction + (1 - α) × lstm_prediction`

---

## Dynasty-Specific Modeling Considerations

### Age Curves and The "Wash Out" Risk
The model implicitly learns that players peak at **age 26-28**. Because the LSTM is fed full distributions of all players (no minute minimums), a 21-year-old performing poorly is aggressively penalized toward a "wash out" trajectory, rather than erroneously mapped to a late bloomer who simply survived. 

### Shutdown Risk & "Silly Season"
Fantasy values are directly impacted by when games are played. Stage 2 applies a **Playoff Likelihood Discount**. Older players on contending 'locked' seeds or tanking teams receive a fractional value discount mirroring their likelihood to be rested during fantasy playoffs (Weeks 20-23).

---

## Valuation Math (Stage 2)

### VORP Calculation
Positional eligibility is treated dynamically. A player with PF/C eligibility is compared against a blended replacement level of both positions, creating inherent value boosts for multi-positional elasticity.

### Consolidative Trade Normalization
A simple linear map of VORP produces poor 2-for-1 trade logic. Trade value is normalized using a non-linear power curve:

```
normalized_vorp = max(0, player_dynasty_VORP)^consolidation_premium
trade_value = 100 × (normalized_vorp / max_normalized_vorp)
```
Where `consolidation_premium` (e.g., 1.2) appropriately values top-tier assets requiring massive packages in return.

---

## Project Structure

```
Dynasty_Model/
├── docs/
│   ├── spec.md                  # This file
│   └── spec_review.md           # Critical review and methodologies
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── src/
│   ├── scraping/
│   ├── preprocessing/
│   ├── models/
│   ├── valuation/
│   └── evaluation/
├── notebooks/
├── configs/
├── tests/
├── requirements.txt
└── README.md
```
