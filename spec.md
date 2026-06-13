# SDD Spec: mundial-predictor

## Requirements

### R1: Data Pipeline
- R1.1: Download Kaggle international football results dataset (all countries, 1872-present)
- R1.2: Filter official matches only (FIFA World Cup, continental cups, qualifiers, confederations cup, FIFA Confederations Cup, Nations League)
- R1.3: Filter date >= 1993-01-01
- R1.4: Remove walkovers, abandoned matches, and matches with missing goals
- R1.5: Normalize team names (handle name changes, country code standardization)
- R1.6: Add match_id (unique, sequential by date)
- R1.7: Mark neutral venue matches (no home advantage)
- R1.8: Export clean dataset as data/processed/clean_matches.csv

### R2: ELO Rating System
- R2.1: Initial rating: seeded by confederation (CONMEBOL=1700, UEFA=1650, CONCACAF=1550, CAF=1500, AFC=1450, OFC=1400)
- R2.2: K-factor: 40 for World Cup, 30 for qualifiers, 20 for continental tournaments
- R2.3: Provisional K = 2x standard K for teams with < 15 matches
- R2.4: Expected score formula: E = 1 / (1 + 10^((Elo_B - Elo_A) / 400))
- R2.5: Update: Elo_new = Elo_old + K x (S - E) where S = 1 for win, 0.5 for draw, 0 for loss
- R2.6: Goal margin adjustment: margin_mult = ln(abs(goals_for - goals_against) + 1) x (2.2 / (Elo_diff x 0.001 + 2.2))
- R2.7: Elo history persists to data/processed/elo_history.csv (match_id, team, elo_pre, elo_post, tournament_type)
- R2.8: Unit tests verify known rating calculations against published FIFA ELO ratings

### R3: Feature Store
- R3.1: elo_diff = elo_home - elo_away (from R2)
- R3.2: form_goals_for_5, form_goals_against_5, form_goals_for_10, form_goals_against_10 (rolling average goals in last N matches)
- R3.3: h2h_last_5: average goal differential in last 5 head-to-head matches
- R3.4: home_advantage: 1 for true home, 0.5 for neutral (continental tournament host), 0 for neutral venue
- R3.5: rest_days: days since team's last match (capped at 30)
- R3.6: implied_home_prob, implied_draw_prob, implied_away_prob from odds (overround removed via normalization)
- R3.7: Odds from football-data.co.uk CSV files for historical matches (2014/2018/2022)
- R3.8: Feature store exported as data/processed/feature_store.csv

### R4: XGBoost Model (1X2)
- R4.1: Target: 1 (home win), X (draw), 2 (away win)
- R4.2: Train/val split: temporal, not random (train <= date threshold, val > date threshold)
- R4.3: TimeSeriesSplit with n_splits=5
- R4.4: Use all features from R3
- R4.5: CalibratedClassifierCV with isotonic regression
- R4.6: Evaluation: Brier score, log-loss, reliability diagram
- R4.7: Feature importance analysis (SHAP values)
- R4.8: Model saved as models/xgb_model.pkl + xgb_model.json
- R4.9: Feature schema saved as models/feature_schema.json

### R5: Bivariate Poisson (Dixon-Coles)
- R5.1: lambda_home = exp(alpha + beta_home x X), lambda_away = exp(alpha + beta_away x X) where X is the feature vector
- R5.2: Dixon-Coles tau adjustment for low-scoring draws (0-0, 1-1):
  - tau(x, y) = 1 - rho x x x y for x = y = 0 or 1
  - tau(x, y) = 1 + rho for x = y >= 2
- R5.3: rho estimated via MLE on training data
- R5.4: P(exact score) = tau(x, y) x exp(-lambda_home) x lambda_home^x / x! x exp(-lambda_away) x lambda_away^y / y!
- R5.5: Validation: compare P(score) vs actual frequency on 2022 hold-out
- R5.6: Model params saved as models/poisson_params.json

### R6: Monte Carlo Simulation
- R6.1: Parse 2026 World Cup fixture (groups + bracket) from data/raw/fixture_2026.csv
- R6.2: Group stage: each team plays 3 matches, top 2 advance
- R6.3: Knockout stage: standard single-elimination (Round of 16 -> QF -> SF -> Final)
- R6.4: Draws in KO stage -> penalties (no golden goal)
- R6.5: Each simulation: update ELO path-dependently match by match
- R6.6: Each simulation: save final bracket + champion
- R6.7: Run 10,000 iterations
- R6.8: Output: data/processed/champion_probs.csv (team, champion_count, pct)
- R6.9: Output: data/processed/match_probs.csv (match_id, home_team, away_team, P_home, P_draw, P_away, P_exact_score_matrix)
- R6.10: Convergence check: running mean stabilizes within +/- 1%

### R7: Backtesting
- R7.1: Walk-forward on 2014/2018/2022 World Cups as hold-out sets
- R7.2: For each tournament, train on ALL data before the tournament, predict tournament matches
- R7.3: Metrics: Brier score, RPS (Ranked Probability Score), log-loss
- R7.4: ROI simulation using fractional Kelly criterion (f = 0.25)
- R7.5: Calibration curves (binned by predicted probability, 10 bins)
- R7.6: Confusion matrix by match type (group, KO, final)
- R7.7: Home/away bias analysis
- R7.8: Results saved to backtesting/results/ directory

### R8: Streamlit Dashboard
- R8.1: Page 1: Upcoming matches table with 1X2 probs, Poisson score matrix, edge vs bookies
- R8.2: Page 2: Champion ranking (bar chart + table), group-stage probabilities
- R8.3: Page 3: Backtesting results (Brier/RPS/ROI over time, calibration curves, error analysis)
- R8.4: Model card (feature list, training date, performance, limitations)
- R8.5: Download predictions as CSV button
- R8.6: Deploy to Streamlit Cloud

## Scenarios

### SC1: First-time run
1. User clones repo
2. Runs pipeline.py
3. System downloads Kaggle dataset
4. System cleans and processes data
5. System calculates ELO for all matches
6. System builds feature store
7. System trains XGBoost + Poisson models
8. Success: data/processed/ contains all CSVs, models/ contains .pkl files

### SC2: Incremental update (new matches)
1. pipeline.py detects existing processed data
2. System loads from cache
3. System appends only new matches since last run
4. System recalculates ELO from last checkpoint
5. System retrains models
6. Success: pipeline runs in < 5min on update

### SC3: Predict 2026 World Cup
1. User places fixture_2026.csv in data/raw/
2. Runs monte_carlo.py
3. System loads trained models
4. System runs 10k simulations
5. Output: champion_probs.csv + match_probs.csv
6. User opens dashboard to see results

### SC4: Backtest on past World Cup
1. Runs backtesting/evaluator.py --tournament 2022
2. System trains model on pre-2022 data
3. System predicts all 2022 matches
4. System compares with actual results
5. Output: Brier, RPS, ROI metrics
6. Success: validates model performance before 2026

## Data Contracts

### clean_matches.csv
Fields: match_id, date, home_team, away_team, home_goals, away_goals, tournament_type, neutral_venue, elo_home_pre, elo_away_pre, elo_home_post, elo_away_post

### feature_store.csv
Fields: match_id, date, home_team, away_team, elo_diff, form_home_5f, form_home_5a, form_away_5f, form_away_5a, form_home_10f, form_home_10a, form_away_10f, form_away_10a, h2h_avg_diff, home_advantage, rest_days_home, rest_days_away, implied_home, implied_draw, implied_away, target

### fixture_2026.csv
Fields: match_id, group, round, date, home_team, away_team, neutral_venue

## Constraints
- Python 3.11+, xgboost 2.0+, streamlit 1.28+
- All models must be reproducible (random_seed=42)
- Pipeline must support partial re-runs (checkpoint system)
- ELO history must be append-only (no retroactive changes)
