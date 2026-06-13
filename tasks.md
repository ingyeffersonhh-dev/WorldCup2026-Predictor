# SDD Tasks: mundial-predictor

## Sprint 1: Data Foundation + ELO (Days 1-5)

### Task 1.1: Download and audit Kaggle dataset
- [x] File: data_pipeline.py (part 1)
- [x] Implement: DataPipeline.download_kaggle(), DataPipeline.audit_columns()
- [x] Verify: Output columns exist (date, home_team, away_team, home_score, away_score, tournament, neutral)
- [x] Test: print column report, count rows, date range

### Task 1.2: Clean and filter matches
- [x] File: data_pipeline.py (part 2)
- [x] Implement: DataPipeline.filter_official(), DataPipeline.clean_matches(), DataPipeline.add_match_ids(), DataPipeline.add_venue_type()
- [x] Logic: Keep only official tournaments, date >= 1993-01-01, remove walkovers/abandons
- [x] Verify: Run on sample subset, check row count before/after

### Task 1.3: ELO Engine - core calculation
- [x] File: models/elo.py
- [x] Implement: EloEngine class with expected_score(), goal_margin_mult(), k_factor(), update_ratings()
- [x] Constants: base ratings per confederation, K factors per tournament type
- [x] Verify: Manual test with known match result, verify rating change math

### Task 1.4: ELO Engine - batch processing
- [x] File: models/elo.py
- [x] Implement: EloEngine.process_matches() - iterate matches sorted by date, maintain state dict
- [x] Implement: elo_history.csv export
- [x] Verify: Process 1000 matches, check elo_history.csv has correct columns and no NaN

### Task 1.5: ELO validation and tests
- [x] File: models/elo.py
- [x] Implement: EloEngine.validate_ratings() 
- [x] Create: tests/test_elo.py
- [x] Tests: expected_score formula, K-factor by tournament, goal margin adjustment, confederation seeding, 10-match sequence

## Sprint 2: Feature Store + XGBoost (Days 6-11)

### Task 2.1: Basic features (ELO diff, form)
- [x] File: feature_store.py
- [x] Implement: FeatureStore.elo_diff(), FeatureStore.rolling_goals()
- [x] Logic: ELO diff from elo_history.csv, rolling avg goals for/against for 5 and 10 matches
- [x] Verify: Feature values look reasonable (no extreme outliers)

### Task 2.2: Advanced features (H2H, home advantage, rest days)
- [x] File: feature_store.py
- [x] Implement: FeatureStore.head_to_head(), FeatureStore.home_advantage(), FeatureStore.rest_days()
- [x] Verify: H2H matches correct teams, home_advantage flags neutral venues, rest_days capped at 30

### Task 2.3: Odds integration
- [x] File: feature_store.py
- [x] Implement: FeatureStore.load_odds_from_football_data(), FeatureStore.merge_odds(), FeatureStore.remove_overround()
- [x] Logic: Load football-data.co.uk CSVs for 2014/2018/2022, merge on date+teams, remove overround
- [x] Verify: Merged odds have no missing values for tournament matches (falls back to uniform probs)

### Task 2.4: Feature store assembly and export
- [x] File: feature_store.py
- [x] Implement: FeatureStore.build() - orchestrate all features, FeatureStore.export_feature_store()
- [x] Verify: feature_store.csv has all columns, no NaN, correct row count matching clean_matches

### Task 2.5: XGBoost training
- [x] File: models/xgboost_model.py
- [x] Implement: XGBoostModel.prepare_data(), XGBoostModel.temporal_split(), XGBoostModel.train()
- [x] Hyperparameters: n_estimators=1000, max_depth=6, lr=0.01, subsample=0.8, early_stopping=20
- [x] Verify: Model trains without errors, train/val loss decreases

### Task 2.6: XGBoost calibration and evaluation
- [x] File: models/xgboost_model.py
- [x] Implement: XGBoostModel.calibrate() with isotonic regression, XGBoostModel.evaluate()
- [x] Metrics: Brier score, log-loss, reliability diagram data
- [x] Verify: Calibration curve within 5% of diagonal

### Task 2.7: XGBoost persistence and SHAP analysis
- [x] File: models/xgboost_model.py
- [x] Implement: XGBoostModel.feature_importance() with SHAP, XGBoostModel.save(), XGBoostModel.load()
- [x] Save: xgb_model.pkl, xgb_model.json, feature_schema.json
- [x] Verify: Loaded model produces same predictions

## Sprint 3: Poisson + Monte Carlo (Days 12-17)

### Task 3.1: Poisson model - parameter estimation
- [x] File: models/poisson_model.py
- [x] Implement: DixonColesPoisson.prepare_data(), DixonColesPoisson.dc_log_likelihood(), DixonColesPoisson.fit()
- [x] MLE: scipy.optimize.minimize for alpha, beta_home, beta_away, rho
- [x] Verify: Parameters converge (not NaN, rho between -0.5 and 0.5)

### Task 3.2: Poisson model - prediction and score matrix
- [x] File: models/poisson_model.py
- [x] Implement: DixonColesPoisson.predict_lambdas(), DixonColesPoisson.tau(), DixonColesPoisson.exact_score_prob()
- [x] Score matrix: 7x7 (0-6 goals each side)
- [x] Verify: Probabilities sum to ~1, Dixon-Coles adjustment increases low-score draw probs

### Task 3.3: Poisson model - validation and persistence
- [x] File: models/poisson_model.py
- [x] Implement: DixonColesPoisson.evaluate(), DixonColesPoisson.save(), DixonColesPoisson.load()
- [x] Validate: Compare predicted exact score frequencies vs actual on 2022 hold-out
- [x] Verify: RPS < 0.20

### Task 3.4: Fixture parser and bracket builder
- [x] File: monte_carlo.py
- [x] Implement: MonteCarloSimulator.load_fixture(), MonteCarloSimulator.build_bracket(), MonteCarloSimulator.init_group_stage()
- [x] Logic: Parse fixture_2026.csv, build group stage + KO bracket (R32 -> R16 -> QF -> SF -> Final)
- [x] Verify: 72 group matches + 32 KO matches = 104 total, correct advancement logic per 2026 48-team format

### Task 3.5: Monte Carlo - simulation core
- [x] File: monte_carlo.py
- [x] Implement: MonteCarloSimulator.simulate_group(), MonteCarloSimulator.resolve_group(), MonteCarloSimulator.simulate_ko_match(), MonteCarloSimulator.simulate_tournament()
- [x] Each sim: path-dependent ELO update, XGBoost for 1X2 probs, Poisson for score
- [x] Verify: 100 sims run correctly, champion distribution looks reasonable

### Task 3.6: Monte Carlo - results and convergence
- [x] File: monte_carlo.py
- [x] Implement: MonteCarloSimulator.track_champion(), MonteCarloSimulator.track_match_probs(), MonteCarloSimulator.convergence_check(), MonteCarloSimulator.save_results()
- [x] Verify: champion_probs.csv sums to 100%, convergence within +/-1% at 10k sims

### Task 3.7: Sensitivity analysis
- [x] Create: tests/test_sensitivity.py
- [x] Tests: Vary n_sims (100, 500, 1000), verify champion ranking stability
- [x] Verify: Champion ranking stable across n_sims variations

## Sprint 4: Backtesting + Dashboard (Days 18-23)

### Task 4.1: Walk-forward backtesting
- [x] File: backtesting/evaluator.py
- [x] Implement: BacktestEvaluator.walk_forward() for 2014, 2018, 2022
- [x] Logic: Train on all data before tournament, predict tournament, repeat for each year
- [x] Verify: Predictions saved for each tournament separately

### Task 4.2: Evaluation metrics
- [x] File: backtesting/evaluator.py
- [x] Implement: BacktestEvaluator.brier_score(), BacktestEvaluator.ranked_probability_score(), BacktestEvaluator.log_loss(), BacktestEvaluator.fractional_kelly_roi()
- [x] Verify: RPS between 0 and 1, Brier < 0.25

### Task 4.3: Error analysis and visualization
- [x] File: backtesting/evaluator.py
- [x] Implement: BacktestEvaluator.calibration_curve(), BacktestEvaluator.confusion_matrix_by_round(), BacktestEvaluator.plot_results()
- [x] Generate: calibration.png, confusion_matrix.png
- [x] Verify: Plots render correctly, group vs KO error rates differ

### Task 4.4: Dashboard Page 1 - Upcoming matches
- [x] File: dashboard.py
- [x] Implement: Page with match table, 1X2 probabilities, Poisson score matrix heatmap, edge vs bookies
- [x] Layout: st.dataframe for table, st.plotly_chart/altair_chart for heatmap
- [x] Verify: Data loads, edge = model_prob - implied_prob

### Task 4.5: Dashboard Page 2 - Champion ranking
- [x] File: dashboard.py
- [x] Implement: Page with champion bar chart + table, group-stage probabilities
- [x] Layout: st.bar_chart + st.dataframe for champion table
- [x] Verify: Sorted by probability descending, top 5 highlighted

### Task 4.6: Dashboard Page 3 - Backtesting and model card
- [x] File: dashboard.py
- [x] Implement: Page with metrics over time, calibration curves, confusion matrix, model card, download button
- [x] Model card: features, training date, performance metrics, known limitations
- [x] Verify: Download produces valid CSV

## Sprint 5: Hardening (Days 24-26)

### Task 5.1: Pipeline orchestrator
- File: pipeline.py
- Implement: Pipeline.run(), Pipeline.check_cache(), Pipeline.run_from()
- CLI: argparse with --stages flag, resume support
- Verify: pipeline.py --stages elo,features runs only those stages

### Task 5.2: Requirements and project setup
- File: requirements.txt
- Dependencies: pandas, numpy, xgboost, scikit-learn, scipy, streamlit, shap, kagglehub, plotly, matplotlib, tqdm
- Ensure: version pinning (pandas>=2.0, xgboost>=2.0, streamlit>=1.28)

### Task 5.3: CI/CD (GitHub Actions)
- File: .github/workflows/ci.yml
- Steps: lint (ruff), test (pytest), check pipeline runs on sample data
- Ensure: Fails fast on error, caches pip/npm

### Task 5.4: Documentation
- README.md: Setup, usage, pipeline commands, project structure
- MODEL_CARD.md: Model details, training data, performance, limitations, intended use
- DATA_CARD.md: Dataset sources, columns, preprocessing, license

### Task 5.5: Streamlit Cloud deploy configuration
- File: .streamlit/config.toml
- Config: theme, server settings, secrets management for API keys
- Ensure: Deploy button works on Streamlit Cloud

## Dependency Graph
```
Task 1.1 -> Task 1.2
Task 1.2 -> Task 1.3, Task 1.4
Task 1.4 -> Task 1.5
Task 1.4 -> Task 2.1, Task 2.2
Task 2.1, 2.2 -> Task 2.3, Task 2.4
Task 2.4 -> Task 2.5 -> Task 2.6 -> Task 2.7
Task 2.4 -> Task 3.1 -> Task 3.2 -> Task 3.3
Task 2.7 + 3.3 -> Task 3.4, Task 3.5 -> Task 3.6 -> Task 3.7
Task 2.7 + 3.3 + 3.6 -> Task 4.1 -> Task 4.2 -> Task 4.3
Task 3.6 + 4.3 -> Task 4.4, Task 4.5, Task 4.6
All above -> Task 5.1 -> Task 5.2 -> Task 5.3, Task 5.4, Task 5.5
```

## Verification Gate
Each task is complete when:
- Code runs without errors
- Output file exists (if applicable)
- Tests pass (if applicable)
- Manual inspection shows correct data
