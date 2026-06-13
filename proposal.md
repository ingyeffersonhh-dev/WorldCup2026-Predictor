# SDD Proposal: mundial-predictor

## Problem Statement
Build a probabilistic World Cup 2026 match outcome and tournament winner prediction system using historical international football data from 1993-present, ELO ratings, machine learning (XGBoost, Bivariate Poisson), and Monte Carlo simulation.

## Goals
1. Accurately predict 1X2 match outcomes with well-calibrated probabilities
2. Predict exact scores using Bivariate Poisson with Dixon-Coles adjustment
3. Simulate the full 2026 World Cup tournament (64+ matches) 10,000+ times
4. Estimate champion probability for each qualified team
5. Backtest on 2014/2018/2022 World Cups as hold-out sets
6. Interactive Streamlit dashboard with predictions, edge vs bookies, and champion rankings

## Non-Goals
- Predicting individual player performance or injuries
- Squad composition or lineup prediction
- In-play / live betting
- Scouting or transfer market analysis
- Non-official matches (friendlies excluded)

## Success Criteria
- Brier score ≤ 0.22 on 2022 hold-out
- RPS (Ranked Probability Score) ≤ 0.18
- Positive simulated ROI (using fractional Kelly) across 2014/2018/2022 backtests
- Calibration curve within 5% of diagonal
- Monte Carlo simulation converges within ±1% champion probability at 10k iterations

## Scope
- Data: Kaggle international football results dataset (1993-present, official matches only)
- Features: ELO differential, recent form (5/10 matches goals), H2H, home advantage, rest days, implied odds
- Models: XGBoost (1X2) + Bivariate Poisson with Dixon-Coles (exact score)
- Simulation: Path-dependent ELO Monte Carlo (10k iterations)
- Backtesting: Walk-forward on 2014/2018/2022
- Dashboard: Streamlit (3 pages: matches, scores, backtesting)

## Sprint Breakdown

### Sprint 1 (Days 1-5): Data Foundation + ELO
- Day 1: Download Kaggle + audit columns
- Day 2: Clean data (official matches, 1993-present, deduplicate, walkovers)
- Day 3: ELO init (base 1500 + confederation adjustment + provisional K)
- Day 4: Cumulative ELO calculation with dynamic K
- Day 5: Unit tests + elo_history.csv

### Sprint 2 (Days 6-11): Feature Store + XGBoost
- Day 6: Feature engineering (elo_diff, form_5/10, h2h, home_adv, rest_days)
- Day 7: Odds API verification + fallback
- Day 8: Target encoding (1X2 + implied probs)
- Day 9: XGBoost + TimeSeriesSplit + CalibratedClassifierCV
- Day 10: Evaluation (Brier, log-loss, reliability diagram)
- Day 11: Model persistence

### Sprint 3 (Days 12-17): Poisson + Monte Carlo
- Day 12: Bivariate Poisson (Dixon-Coles)
- Day 13: Validation on 2022 hold-out
- Day 14: Fixture 2026 parser (groups + bracket)
- Day 15: Monte Carlo (10k sims, path-dependent ELO)
- Day 16: champion_probs.csv + match_probs.csv
- Day 17: Sensitivity tests

### Sprint 4 (Days 18-23): Backtesting + Dashboard
- Day 18: Walk-forward backtesting
- Day 19: Metrics (Brier, RPS, ROI, calibration curves)
- Day 20: Error analysis
- Day 21: Streamlit Page 1 (matches + edge)
- Day 22: Streamlit Page 2 (score matrix + champion ranking)
- Day 23: Streamlit Page 3 (backtesting + model card)

### Sprint 5 (Days 24-26): Hardening
- CI/CD (GitHub Actions)
- Documentation (README, MODEL_CARD, DATA_CARD)
- Deploy (Streamlit Cloud / Cloudflare Pages)

## Key Decisions (to resolve in spec phase)
1. The Odds API historical access vs football-data.co.uk fallback
2. ELO initial rating: flat 1500 vs confederation-seeded
3. Poisson: Dixon-Coles (τ) vs simple bivariate
4. Monte Carlo: path-dependent ELO vs fixed initial ELO
5. Deploy target: Streamlit Cloud vs Cloudflare Workers

## Project Structure
mundial-predictor/
├── data/
│   ├── raw/               # Kaggle CSV + fixture 2026
│   ├── processed/         # Feature store generated
│   └── odds/              # Odds from The Odds API / football-data.co.uk
├── models/
│   ├── elo.py             # ELO calculation and updates
│   ├── xgboost_model.py   # Train + predict 1X2
│   ├── poisson_model.py   # Goal prediction (Dixon-Coles)
│   └── monte_carlo.py     # Champion simulation
├── backtesting/
│   └── evaluator.py       # Brier, RPS, simulated ROI
├── pipeline.py            # Orchestrates full cycle
└── dashboard.py           # Streamlit app
