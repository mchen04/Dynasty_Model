# Dynasty Model — NBA Fantasy Dynasty Player Valuation Engine

## Overview

An ML model trained from scratch that predicts two values for any NBA player:

1. **Auction Startup Value** — what a player is worth in a fresh ESPN dynasty auction draft ($0–$200 scale)
2. **Current Trade Value** — what a player is worth mid-season in an existing dynasty league (normalized 0–100 scale)

The model is **league-configurable**: users can adjust team count, scoring system, and roster construction. Defaults to ESPN standard settings.

---

## Target Outputs

| Output | Scale | Description |
|--------|-------|-------------|
| `auction_value` | $1–$200 | Startup auction price (sum across all players = $200 × num_teams) |
| `trade_value` | 0–100 | Relative dynasty asset value, where 100 = most valuable player in the league |

Both outputs are **dynasty-aware**: they weight long-term future production, not just current-season projections. A 22-year-old averaging 18 PPG is worth more than a 33-year-old averaging 22 PPG.

---

## Architecture: Two-Stage Pipeline

### Stage 1 — Stat Projection Model (League-Agnostic)

Predicts per-game stat lines for each player across multiple future seasons.

- **Input**: player features (see Feature Set below)
- **Output**: projected per-game stats for seasons T+1 through T+5 (points, rebounds, assists, steals, blocks, 3PM, FG%, FT%, TO, minutes, games played)
- **Model**: ensemble of XGBoost (tabular features) + LSTM (career trajectory sequences)
  - XGBoost handles the cross-sectional snapshot: "given everything we know about this player right now, what are his stats next year?"
  - LSTM handles the longitudinal trajectory: "given this player's career arc so far, where is the trend going?"
  - Final prediction = weighted blend of both, with SHAP values for interpretability
- Trained once, shared across all league formats

### Stage 2 — Configurable Valuation Layer (League-Specific)

Converts projected stat lines into dollar/trade values based on league settings.

- **Inputs**: Stage 1 projections + league configuration parameters
- **Process**:
  1. Apply scoring weights to projected stats → fantasy points per game (for points leagues) or per-category z-scores (for category leagues)
  2. Discount future seasons using a dynasty time-value decay (year 1 = 1.0, year 2 = 0.85, year 3 = 0.72, year 4 = 0.61, year 5 = 0.52 — tunable)
  3. Sum discounted values → total dynasty asset value
  4. Compute positional replacement level based on league size and roster slots
  5. Calculate Value Over Replacement Player (VORP)
  6. Convert VORP to auction dollars: `player_$ = (player_VORP / total_VORP_pool) × total_league_budget`
  7. For trade value: normalize to 0–100 scale
- No ML training required — pure math, instantly recalculated for any league config

---

## League Configuration Parameters

| Parameter | Default (ESPN Standard) | Configurable Range |
|-----------|------------------------|-------------------|
| `num_teams` | 10 | 8, 10, 12, 14, 16, 20 |
| `scoring_type` | `"points"` | `"points"`, `"9cat"`, `"8cat"`, `"custom"` |
| `budget_per_team` | $200 | $100–$500 |
| `roster_size` | 13 | 10–20 |
| `roster_slots` | PG, SG, SF, PF, C, G, F, UTIL×3, BN×3 | Custom |
| `dynasty_horizon` | 5 years | 1–8 years |
| `dynasty_discount_rate` | 0.85 per year | 0.5–1.0 |

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

### ESPN Default 9-Category (H2H Each Category)

FG%, FT%, 3PM, PTS, REB, AST, STL, BLK, TO

**Note on category scoring**: FG% and FT% are volume-weighted in practice — a player's impact on your team's FG% depends on their attempts, not just their percentage. The z-score for FG% should be calculated as: `(player_FG% - league_FG%) × player_FGA` to capture this.

### ESPN Default Rotisserie

Same 9 categories as H2H, but cumulative season-long totals ranked across all teams (1st through last in each category; total category points determine standings).

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
- **Offensive/Defensive Rating**: points per 100 possessions produced/allowed

### C. Era-Adjusted Stats
All counting stats are also stored as **within-season z-scores** — how many standard deviations above/below the league average for that season. This handles scoring inflation, pace changes, and the 3-point revolution automatically.

Formula: `z_stat = (player_stat - season_mean) / season_std`

This means 25 PPG in the 2003 grind-it-out era and 28 PPG in the 2024 pace-and-space era can be compared on equal footing.

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
- Injury type encoding (one-hot by body region: knee, ankle, back, shoulder, hand/wrist, foot, hip, head/concussion, illness, other)
- Injury severity encoding (minor: 1-7 days, moderate: 8-30 days, major: 31-90 days, severe: 90+ days)
- Days since last injury
- Career-ending injury risk flag (ACL, Achilles — binary indicators if player has had one)
- Load management indicator (did team rest healthy player?)

### F. Team Context
- Team win-loss record (current season and trailing 3 seasons)
- Team offensive rating, defensive rating, net rating
- Team pace
- Team 3-point attempt rate
- Playoff team indicator (binary)
- Conference and division
- Coach (encoded — some coaches run faster pace, use different rotation patterns)
- Team salary cap situation (are they competitive or rebuilding?)

### G. Role & Opportunity
- Depth chart position (starter, 6th man, rotation, deep bench)
- Minutes share: `player_minutes / team_total_minutes`
- Usage rate
- Touches per game, time of possession (from NBA tracking data, 2013-14+)
- Passes per game, potential assists (tracking data)
- Distance traveled per game (tracking data)
- Average speed, average speed on offense/defense (tracking data)
- On/off court net rating differential

### H. Teammate & Competition Context
- Quality of teammates: average BPM of other starters on the team
- Minutes available at position: are there minutes to be had or is the depth chart locked?
- Star teammate indicator: does the team have a top-15 player? (affects usage distribution)
- Upcoming free agency / contract year indicator

### I. Historical Trajectory Features (LSTM Sequence Inputs)
For the LSTM component, feed a **sequence of season-level feature vectors** representing the player's entire career:
- Per-season: all stats from groups A–H aggregated at the season level
- The sequence length = number of NBA seasons played (1 to 20+)
- Padded/masked for players with fewer seasons

### J. Rolling Momentum Features
- Last 10/20/40 game rolling averages for key stats
- Trend direction: are stats improving or declining? (slope of rolling average)
- Volatility: standard deviation of game-to-game performance over last 20 games
- Breakout detection: has the player's last-20-game average exceeded their season average by >1 std dev?

---

## Training Data

### Era Definition: "Modern NBA" = 2001-02 to Present

**Why 2001-02:**
- Zone defense was legalized in 2001-02, fundamentally changing offensive and defensive strategy
- BPM and VORP data available from 1973-74, but the game before zone defense is structurally different
- FiveThirtyEight's RAPTOR historical data uses box-score-plus-RPM hybrid from 2001-02 onward
- NBA tracking data (SportVU/Second Spectrum) available from 2013-14, giving us ~12 seasons of rich tracking features
- 2001-02 gives ~24 seasons of data, covering thousands of player-seasons and multiple complete career arcs

**Even with 2001-02 as the cutoff, all stats are era-adjusted via z-scores per season** — so a player from 2003 and a player from 2024 are measured relative to their contemporaries, not in absolute terms.

### The 3-Point Revolution & Scoring Inflation (Why Era Adjustment Matters)

| Season | 3PA/game | Pace | PPG | Notes |
|--------|----------|------|-----|-------|
| 2001-02 | 14.7 | 90.7 | 95.5 | Zone defense legalized |
| 2010-11 | 18.0 | 92.1 | 99.6 | Pre-revolution baseline |
| 2014-15 | 22.4 | 93.9 | 100.0 | Curry MVP, revolution begins |
| 2016-17 | 27.0 | 96.4 | 105.6 | Rapid acceleration |
| 2018-19 | 32.0 | 100.0 | 111.2 | Pace-and-space fully adopted |
| 2021-22 | 35.2 | 98.2 | 110.6 | Plateau phase |
| 2025-26 | 37.0 | 99.5 | 115.4 | Current |

3PA nearly tripled from 2001 to 2025. A "25 PPG scorer" means something very different in 2003 vs 2024. The z-score normalization + per-100-possession adjustment handles this — the model sees "how good is this player relative to his peers" rather than raw counting stats.

### Data Sources

| Source | What It Provides | Availability | Access Method |
|--------|-----------------|--------------|---------------|
| **Basketball-Reference** | Box scores, advanced stats (PER, WS, BPM, VORP), per-100-poss, shooting splits, draft data | 1947–present (advanced from 1973-74) | Web scraping (respect rate limits, check ToS) |
| **nba_api** (Python) | Official NBA.com data: box scores, player tracking, shooting dashboard, hustle stats, matchup data | 2001–present (tracking from 2013-14) | Free Python package, REST API |
| **FiveThirtyEight RAPTOR** | Historical RAPTOR ratings (offensive, defensive, total) | 1977–present (full tracking-based from 2014) | Free CSV download from GitHub |
| **DunksAndThrees EPM** | Estimated Plus-Minus, optimized skills | 2001–present | Web scraping |
| **NBA Injury Data** | Official injury reports, games missed, injury type | Structured from 2021-22; partial before via Basketball-Reference transaction logs | `nbainjuries` Python package + Basketball-Reference scraping |
| **Spotrac / Basketball-Reference** | Contract data, salary, cap holds | 2000–present | Web scraping |
| **ESPN Fantasy API** | Auction draft results, fantasy point calculations, roster settings | Varies | Hidden API — base: `lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}/...`; public leagues no auth, private needs `espn_s2`+`SWID` cookies; Python wrapper: `cwendt94/espn-api` |
| **FantasyPros / Hashtag Basketball** | Consensus rankings, trade value charts, auction values | Current + some historical | Web scraping for historical consensus |
| **BALLDONTLIE API** | NBA box scores, player stats, team data | Historical + current | Free tier available at balldontlie.io |
| **NBA Draft Combine** | Height, weight, wingspan, standing reach, hand size, vertical leap, lane agility, sprint | 2000–present | `nba_api` `DraftCombineStats` endpoint |
| **BBall Index LEBRON** | LEBRON metric (impact + luck-adjusted) | 2009–present | Free database at bball-index.com |
| **Cleaning the Glass** | Filtered stats (no garbage time), on/off, shot location by position | Recent seasons | $7.50/month subscription |

### Training Set Construction

Each training example is a **(player, season)** pair:

- **Features (X)**: everything from feature groups A–J as of the *start* of season T (using data from seasons T-1, T-2, ..., T-N)
- **Target (Y)**: actual per-game stats produced in season T (and optionally T+1 through T+4 for multi-year projection)

**Example**: For predicting the 2024-25 season, features come from 2023-24 and prior seasons. The target is the actual 2024-25 stat line.

This produces approximately **8,000–12,000 player-season examples** from 2001-02 to present (filtering out players with <200 minutes in a season).

### Target Variable Strategy

The Stage 1 model predicts **raw per-game stat lines**, not fantasy points or dollar values directly. This is critical because:

1. Raw stats are league-agnostic — the same prediction works for any scoring system
2. Fantasy point values and dollar conversions are deterministic functions of raw stats + league settings (handled in Stage 2)
3. Historical "ground truth" for raw stats is objective and abundant; historical auction values are sparse, noisy, and league-specific

**Multi-output regression**: the model predicts all stat categories simultaneously (PTS, REB, AST, STL, BLK, 3PM, FG%, FT%, TO, MIN, GP) as a vector output.

---

## Model Training Details

### XGBoost Component

- **Algorithm**: XGBoost with multi-output regression (one model per target stat, or a single multi-output model)
- **Features**: all tabular features from groups A–H, J (flattened to a single vector per player-season)
- **Hyperparameter tuning**: Bayesian optimization (Optuna) with 5-fold time-series cross-validation
  - Walk-forward validation: train on seasons 2001 through T-1, validate on season T, for T in 2018 through 2024
- **Feature importance**: SHAP values for interpretability
- **Regularization**: L1 + L2 (alpha, lambda), max_depth, min_child_weight, subsample, colsample_bytree

### LSTM Component

- **Architecture**: 2-layer bidirectional LSTM with attention
  - Input: sequence of season-level feature vectors (one per season in the player's career)
  - Hidden size: 128
  - Dropout: 0.3
  - Output: projected stat vector for next season(s)
- **Handles variable-length careers** via padding + masking
- **Captures**: aging curves, development trajectories, post-injury recovery patterns
- **Training**: PyTorch, Adam optimizer, MSE loss, early stopping on validation set

### Ensemble Blend

- `final_prediction = α × xgb_prediction + (1 - α) × lstm_prediction`
- α is learned via a simple linear meta-model on the validation set
- Expected α ≈ 0.6–0.7 (XGBoost typically dominates for next-season prediction; LSTM adds value for multi-year horizons and young player trajectories)

### Evaluation Metrics

- **MAE** (Mean Absolute Error) per stat category
- **MAPE** (Mean Absolute Percentage Error) for minutes and games played
- **R²** for overall prediction quality
- **Spearman rank correlation** between predicted and actual fantasy value rankings (most important for dynasty use case — we care more about ordering players correctly than exact stat predictions)
- **Auction value MAE**: after running Stage 2, compare predicted auction values to historical consensus values

---

## Dynasty-Specific Modeling Considerations

### Age Curves

Research consensus (Vaci et al. 2019, Bryant University, Dartmouth Sports Analytics): NBA players peak at **age 26-28** on average, with position-specific variation:

- **Guards**: peak slightly later (~27-28); skill-dependent positions, less athleticism decline. Technical performance stable or improves with age. Tend to exit the league starting at ~32.
- **Forwards**: peak ~27; slow decline until early-to-mid 30s. Versatility (shooting + size) provides longer careers.
- **Centers**: peak ~27 but with the **widest variance** of any position. Physical bigs peak earlier; skilled bigs maintain longer. Size provides a value floor even as athleticism declines. Defensive peak comes later than offensive peak.
- **General pattern**: offense peaks earlier than defense. Players who excelled early retain more skill and decline more slowly. High-minute players develop faster and decline slower.

Dynasty value windows:
- **Ages 20-24**: development phase; high variance, high upside. Rookies often struggle initially.
- **Ages 22-26**: sweet spot for dynasty value — rising trajectory with years of production ahead.
- **Ages 27-29**: peak production but declining dynasty value (sell window opens).
- **Ages 30-32**: still productive but declining; trade aging stars while value remains.
- **Ages 33+**: significant decline risk; replacement-level value unless exceptional player.

The model learns these curves implicitly from the data. The LSTM component is especially important here — it sees the full career trajectory and can distinguish a player who is still developing from one who is declining.

The dynasty discount rate in Stage 2 (default 0.85/year) approximates the *league-wide average* aging curve, but the model's per-player projections handle individual variation (a 30-year-old LeBron-type is projected differently than a 30-year-old athletic wing).

### Injury Risk Modeling

Injuries are the biggest source of uncertainty in dynasty valuation. Research-backed severity tiers:

| Injury | Avg Return Time | Non-Return Rate | Avg PER Drop | Dynasty Impact |
|--------|----------------|-----------------|--------------|----------------|
| **Achilles rupture** | ~11 months | 39% never return | -3.5 vs pre-injury, -5.4 vs healthy controls | Catastrophic — 63% of non-All-Stars out within 3 years |
| **ACL tear** | 9-12 months | 14% | Significant but less than Achilles | Severe — but modern recovery much better |
| **Lisfranc (foot)** | ~11 months | High re-injury risk | Varies | Severe — chronic concern |
| **Meniscus/knee** | 2-6 weeks | Low | Minimal if managed | Moderate — watch for recurring issues |
| **Ankle sprains** | 1-4 weeks | Very low | Minimal | Low — but chronic ankle issues compound |

The model captures injury risk through:

1. **Games-played prediction**: the model explicitly predicts GP (games played) as one of its target stats, using injury history features
2. **Post-injury trajectory**: the LSTM sees historical cases of players returning from ACL tears, Achilles ruptures, etc., and learns typical recovery curves
3. **Risk discount**: a player projected for 25 PPG but only 55 games is worth less than one projected for 22 PPG over 75 games. Stage 2 accounts for this by using `value = per_game_value × projected_GP / 82`
4. **Workload risk signals**: players with >100 minutes in last 5 games, older players with high usage, and players with prior same-body-part injuries are at elevated risk

### Rookie & Young Player Valuation

Players with <2 NBA seasons have thin data. The model handles this via:

- **Draft capital features**: pick number, college stats, combine measurements are highly predictive of early-career trajectory
- **Comparable player matching**: the LSTM can identify similar early-career profiles from historical data
- **Higher uncertainty bounds**: for young players, the model should output wider confidence intervals (future work: quantile regression or MC dropout for uncertainty)

---

## Project Structure

```
Dynasty_Model/
├── docs/
│   └── spec.md                  # This file
├── data/
│   ├── raw/                     # Raw scraped data (gitignored)
│   ├── processed/               # Cleaned, merged datasets
│   └── features/                # Final feature matrices for training
├── src/
│   ├── scraping/                # Data collection scripts
│   │   ├── basketball_ref.py    # Basketball-Reference scraper
│   │   ├── nba_api_fetch.py     # NBA.com official API data
│   │   ├── raptor_fetch.py      # FiveThirtyEight RAPTOR download
│   │   ├── injury_fetch.py      # Injury data collection
│   │   └── espn_fantasy.py      # ESPN fantasy API data
│   ├── preprocessing/           # Data cleaning and feature engineering
│   │   ├── clean.py             # Data validation, deduplication
│   │   ├── era_adjust.py        # Z-score normalization, pace adjustment
│   │   ├── features.py          # Feature engineering pipeline
│   │   └── sequences.py         # Build LSTM career sequences
│   ├── models/                  # Model definitions and training
│   │   ├── xgboost_model.py     # XGBoost multi-output regression
│   │   ├── lstm_model.py        # LSTM trajectory model (PyTorch)
│   │   ├── ensemble.py          # Meta-learner for blending
│   │   └── train.py             # Training orchestration
│   ├── valuation/               # Stage 2: league-specific valuation
│   │   ├── scoring.py           # Fantasy point calculation
│   │   ├── replacement.py       # Replacement level computation
│   │   ├── auction.py           # VORP → auction dollar conversion
│   │   └── trade_value.py       # Dynasty trade value normalization
│   └── evaluation/              # Model evaluation and analysis
│       ├── metrics.py           # MAE, R², Spearman correlation
│       ├── backtest.py          # Historical backtesting
│       └── shap_analysis.py     # Feature importance visualization
├── notebooks/                   # Exploration and analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_valuation_demo.ipynb
├── configs/
│   ├── espn_default.yaml        # ESPN default league settings
│   └── model_config.yaml        # Model hyperparameters
├── tests/
├── requirements.txt
└── README.md
```

---

## Implementation Phases

### Phase 1 — Data Collection & Processing
- Scrape Basketball-Reference for all player-season stats (2001-02 to present)
- Pull NBA tracking data via nba_api (2013-14 to present)
- Download FiveThirtyEight RAPTOR CSVs
- Collect injury data, draft data, contract data
- Build unified player-season dataset with all feature groups
- Implement era adjustment (z-scores per season, per-100-possession conversion)

### Phase 2 — Feature Engineering & Dataset Construction
- Engineer all features from groups A–J
- Build training/validation/test splits (time-series: train on older seasons, test on recent)
- Build LSTM career sequences
- Handle missing data (tracking stats missing pre-2014; impute or use indicator flags)

### Phase 3 — Model Training
- Train XGBoost multi-output model with Optuna hyperparameter search
- Train LSTM career trajectory model
- Train ensemble meta-learner
- Evaluate on held-out seasons (2022-23, 2023-24, 2024-25)
- SHAP analysis for interpretability

### Phase 4 — Valuation Layer
- Implement Stage 2 scoring/valuation pipeline
- Support ESPN default points and 9-cat scoring
- Support custom scoring weights
- Implement positional replacement level computation
- Implement auction dollar and trade value conversion
- Validate against published consensus rankings and auction values

### Phase 5 — Productionization
- CLI or simple web interface for querying player values
- Input: player name + league config → output: auction value + trade value
- Ability to compare two trade packages
- Auto-update pipeline when new game data comes in

---

## Stage 2 Valuation Math (Detailed)

### Points League Valuation

```
fantasy_pts_per_game = Σ (projected_stat_i × scoring_weight_i)
season_value = fantasy_pts_per_game × projected_games_played
dynasty_value = Σ_{t=1}^{horizon} season_value_t × discount_rate^(t-1)
```

### Category League Valuation (Z-Score Method)

For each of the 9 categories:
```
z_cat = (player_projected_stat - pool_mean) / pool_std
```

For volume-weighted percentage categories (FG%, FT%):
```
z_FG% = (player_FG% - pool_FG%) × player_FGA / pool_std_of_FG_impact
```

Composite: `total_z = Σ z_cat` (across all categories; TO is inverted since fewer is better)

### Replacement Level

```
replacement_rank = num_teams × positional_roster_slots_at_position + 1
replacement_value = total_z of the player at replacement_rank for that position
```

For UTIL/flex spots, replacement level is computed across all positions combined.

### VORP to Auction Dollars

```
player_VORP = player_total_z - replacement_value
total_VORP_pool = Σ max(0, player_VORP) for all above-replacement players
player_auction_$ = (player_VORP / total_VORP_pool) × (num_teams × budget_per_team)
```

Minimum auction value = $1 for any rosterable player.

### Dynasty Time-Value Discount

For multi-year valuation:
```
dynasty_VORP = Σ_{t=1}^{horizon} VORP_t × discount_rate^(t-1)
```

Where `VORP_t` uses the projected stat line for season t (from Stage 1).

### Trade Value Normalization

```
trade_value = 100 × (player_dynasty_VORP - min_VORP) / (max_VORP - min_VORP)
```

Clamped to 0–100. The #1 dynasty asset gets 100, replacement-level gets ~0.

---

## Open Questions & Future Work

- **Uncertainty quantification**: add prediction intervals (quantile regression or MC dropout) so users know confidence in projections
- **Rookie projections from college stats**: extend Stage 1 to accept college stat features for incoming rookies pre-NBA-debut
- **Schedule effects**: account for remaining schedule strength, back-to-back density, rest days
- **Real-time updates**: re-run projections after each game with updated rolling features
- **Sentiment / news features**: transformer-based analysis of injury reports and beat reporter tweets (showed promise in FPL research)
- **Graph neural network extension**: model teammate interactions and lineup synergies (NBA2Vec-style embeddings or GATv2 architecture)
