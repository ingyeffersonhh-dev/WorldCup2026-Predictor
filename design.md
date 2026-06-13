# SDD Design: mundial-predictor

## Architecture Overview

The system follows a pipeline architecture with 5 sequential stages and a dashboard for visualization. Each stage consumes the output of the previous stage.

```
Raw Data (Kaggle + Odds + Fixture)
    |
    v
[Stage 1: Data Pipeline] --> clean_matches.csv
    |
    v
[Stage 2: ELO Engine] --> elo_history.csv
    |
    v
[Stage 3: Feature Store] --> feature_store.csv
    |
    v
[Stage 4a: XGBoost] --> xgb_model.pkl
[Stage 4b: Poisson DC] --> poisson_params.json
    |
    v
[Stage 5a: Monte Carlo] --> champion_probs.csv, match_probs.csv
[Stage 5b: Backtesting] --> backtesting/results/*.csv
    |
    v
[Stage 6: Streamlit Dashboard] --> UI
```

## Component Design

### C1: data_pipeline.py - Data Ingestion & Cleaning
- Class: DataPipeline
- Methods:
  - download_kaggle(): Downloads from Kaggle API (kagglehub)
  - load_raw(path: str) -> pd.DataFrame: Reads CSV
  - audit_columns(df) -> Dict: Checks required columns exist (date, home_team, away_team, home_score, away_score, tournament, neutral)
  - filter_official(df, min_date) -> pd.DataFrame: Removes friendlies, filters date >= 1993
  - clean_matches(df) -> pd.DataFrame: Removes walkovers, deduplicates, normalizes names
  - add_match_ids(df) -> pd.DataFrame: Sequential integer IDs
  - add_venue_type(df) -> pd.DataFrame: Marks neutral venues
  - export(df, path): Saves to CSV
- Dependencies: pandas, kagglehub
- Config: data/raw/, data/processed/

### C2: elo.py - ELO Engine
- Class: EloEngine
- Constants:
  - BASE_RATINGS: {conf: rating}
  - K_FACTORS: {"World Cup": 40, "Qualifiers": 30, "others": 20}
  - PROVISIONAL_MATCH_THRESHOLD: 15
  - PROVISIONAL_K_MULTIPLIER: 2
- Methods:
  - __init__(base_ratings: Dict[str, int]): Loads confederation seed ratings
  - expected_score(rating_a: float, rating_b: float) -> float: E = 1/(1 + 10^((rb-ra)/400))
  - goal_margin_mult(goals_a: int, goals_b: int, elo_diff: float) -> float: ln(|gd|+1) * (2.2/(elo_diff*0.001+2.2))
  - k_factor(tournament_type: str, team_matches: int) -> int: Returns dynamic K
  - update_ratings(home_goals, away_goals, home_rating, away_rating, tournament, team_matches) -> Tuple[float, float]: Returns new ratings
  - process_matches(matches_df: pd.DataFrame) -> pd.DataFrame: Processes sequentially by date
  - validate_ratings(known_ratings: Dict) -> bool: Unit test helper
- State: In-memory dict tracking {team: {rating, match_count}}
- Output: elo_history.csv with match_id, team, elo_pre, elo_post

### C3: feature_store.py - Feature Engineering
- Class: FeatureStore
- Methods:
  - build(matches_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame: Orchestrates all features
  - elo_diff(home_elo, away_elo) -> float
  - rolling_goals(team_games: pd.Series, n: int) -> pd.Series: Rolling avg goals for/against
  - head_to_head(team_a: str, team_b: str, history: pd.DataFrame, n: int) -> float
  - home_advantage(match) -> float
  - rest_days(team: str, match_date, all_matches) -> int
  - load_odds_from_football_data(year: int) -> pd.DataFrame
  - merge_odds(matches_df, odds_df) -> pd.DataFrame
  - remove_overround(probs: List[float]) -> List[float]: Normalizes to sum=1
  - export_feature_store(df, path)
- Dependencies: pandas, numpy

### C4: xgboost_model.py - XGBoost Classifier
- Class: XGBoostModel
- Methods:
  - prepare_data(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]: X = features, y = target (1/X/2)
  - temporal_split(X, y, split_date) -> train/val sets
  - time_series_cv(X, y, n_splits) -> list of (train_idx, val_idx)
  - train(X_train, y_train, X_val, y_val) -> xgb.Booster: Early stopping
  - calibrate(model, X_calib, y_calib) -> CalibratedClassifierCV: Isotonic regression
  - evaluate(model, X_test, y_test) -> Dict: Brier, log-loss, reliability data
  - feature_importance(model) -> pd.DataFrame: SHAP values
  - save(model, path)
  - load(path) -> model
- Hyperparameters:
  - n_estimators: 1000 (early stopping)
  - max_depth: 6
  - learning_rate: 0.01
  - subsample: 0.8
  - colsample_bytree: 0.8
  - early_stopping_rounds: 20
  - random_state: 42
- Dependencies: xgboost, sklearn, shap, pickle

### C5: poisson_model.py - Bivariate Poisson with Dixon-Coles
- Class: DixonColesPoisson
- Methods:
  - prepare_data(feature_df: pd.DataFrame) -> Tuple: X features, y = (home_goals, away_goals)
  - dc_log_likelihood(params, X, y) -> float: Negative log-likelihood
  - fit(X_train, y_train) -> Dict: MLE estimation of alpha, beta_home, beta_away, rho
  - predict_lambdas(X, params) -> Tuple[float, float]: lambda_home, lambda_away
  - tau(x, y, rho) -> float: Dixon-Coles adjustment
  - exact_score_prob(lambda_h, lambda_a, rho, max_goals=6) -> np.array: (max_goals+1 x max_goals+1) matrix
  - match_1x2_from_score_matrix(score_matrix) -> Tuple[float, float, float]: Marginal P(1), P(X), P(2)
  - evaluate(y_true, y_pred_probs) -> Dict: RPS, log-loss
  - save(params, path)
  - load(path) -> Dict
- Dependencies: numpy, scipy.optimize, pickle

### C6: monte_carlo.py - Tournament Simulation
- Class: MonteCarloSimulator
- Methods:
  - load_fixture(path) -> pd.DataFrame: Parses fixture_2026.csv
  - build_bracket(fixture_df) -> Dict: Group stage + KO bracket structure
  - init_group_stage(fixture_df) -> Dict: {group: {teams, matches}}
  - simulate_group(group_data, models, elo_state) -> Dict: Simulates all group matches, returns standings
  - resolve_group(group_results) -> List[str]: Top 2 advance
  - simulate_ko_match(home, away, models, elo_state) -> Tuple[winner, loser, scores, is_penalty]
  - simulate_tournament(fixture, models, initial_elo, n_sims=10000) -> pd.DataFrame: Run all sims
  - track_champion(all_sim_results) -> pd.DataFrame: champion_probs.csv
  - track_match_probs(all_sim_results) -> pd.DataFrame: match_probs.csv
  - convergence_check(champion_df) -> Dict: Running mean stability
  - save_results(champion_df, match_df)
- State management: ELO updates path-dependently within each simulation
- Dependencies: numpy, pandas, tqdm
- Performance: 10k sims should run in < 30 min (target: 10 min with optimization)

### C7: evaluator.py - Backtesting
- Class: BacktestEvaluator
- Methods:
  - walk_forward(data, tournament_year) -> pd.DataFrame: Train on pre-tournament, predict tournament
  - brier_score(y_true, y_pred) -> float
  - ranked_probability_score(y_true, y_pred) -> float
  - log_loss(y_true, y_pred) -> float
  - fractional_kelly_roi(predictions, odds_implied, results, f=0.25) -> float
  - calibration_curve(y_true, y_pred, n_bins=10) -> pd.DataFrame
  - confusion_matrix_by_round(predictions, actuals, rounds) -> pd.DataFrame
  - plot_results(results_df): Generates plots to backtesting/results/
- Dependencies: numpy, pandas, matplotlib, sklearn.metrics

### C8: pipeline.py - Orchestrator
- Class: Pipeline
- Methods:
  - run(stages: List[str] = None): Run specified stages (default: all)
  - check_cache(stage: str) -> bool: Checkpoints
  - run_all(): Sequential execution
  - run_from(stage: str): Resume from checkpoint
- CLI: pipeline.py --stages data,elo,features,xgb,poisson,mc,backtest
- Dependencies: All components above

### C9: dashboard.py - Streamlit UI
- Pages: 3 (side by side in sidebar)
- Page 1: upcoming_matches_view()
- Page 2: champion_ranking_view()
- Page 3: backtesting_view()
- Data loading: Reads from data/processed/ and backtesting/results/
- Dependencies: streamlit, pandas, plotly, matplotlib

## File Structure
```
mundial-predictor/
├── data/
│   ├── raw/
│   │   ├── kaggle_results.csv     # Original Kaggle download
│   │   ├── fixture_2026.csv       # World Cup 2026 fixture
│   │   └── odds/
│   │       ├── odds_2014.csv      # football-data.co.uk
│   │       ├── odds_2018.csv
│   │       └── odds_2022.csv
│   └── processed/
│       ├── clean_matches.csv
│       ├── elo_history.csv
│       ├── feature_store.csv
│       ├── champion_probs.csv
│       └── match_probs.csv
├── models/
│   ├── elo.py
│   ├── xgb_model.pkl
│   ├── xgb_model.json
│   ├── feature_schema.json
│   ├── poisson_model.py
│   └── poisson_params.json
├── backtesting/
│   ├── evaluator.py
│   └── results/
│       ├── metrics_2014.csv
│       ├── metrics_2018.csv
│       ├── metrics_2022.csv
│       ├── calibration.png
│       └── confusion_matrix.png
├── monte_carlo.py
├── pipeline.py
├── dashboard.py
└── requirements.txt
```

## Data Flow

1. Kaggle CSV -> DataPipeline.clean() -> clean_matches.csv
2. clean_matches.csv -> EloEngine.process() -> elo_history.csv
3. clean_matches.csv + elo_history.csv + odds CSVs -> FeatureStore.build() -> feature_store.csv
4. feature_store.csv -> XGBoostModel.train() -> xgb_model.pkl
5. feature_store.csv -> DixonColesPoisson.fit() -> poisson_params.json
6. fixture_2026.csv + xgb_model.pkl + poisson_params.json + initial_elo -> MonteCarloSimulator.simulate() -> champion_probs.csv + match_probs.csv
7. feature_store.csv + xgb_model.pkl -> BacktestEvaluator.walk_forward() -> backtesting/results/*.csv
8. All processed data -> Dashboard -> UI

## Key Design Decisions

1. **Sequential processing**: Each stage reads from disk and writes to disk (checkpoint-safe)
2. **No database**: CSV-based storage (simplicity, reproducibility, git-friendly for small files)
3. **Reproducibility**: random_seed=42 on all stochastic processes
4. **Append-only ELO**: No retroactive changes to historical ratings
5. **Cache system**: pipeline.py checks for existing outputs before re-running stages
6. **Path-dependent Monte Carlo**: ELO updates during tournament simulation for realistic momentum effects

## Performance Targets
- Full pipeline (first run): < 45 min
- Incremental update: < 5 min
- Monte Carlo (10k sims): < 15 min
- Dashboard load: < 3 sec
- Backtesting (3 tournaments): < 10 min

## Error Handling
- Invalid or missing columns: raise ValueError with column names
- Missing Kaggle data: fallback to download, raise if no network
- Odds file not found: log warning, continue with implied probs from ELO only
- Model file corruption: delete and retrain
- Pipeline stage failure: log error, save partial state, exit non-zero
