"""
pipeline.py — End-to-End Orchestrator

Run all training and simulation stages for mundial-predictor.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from data_pipeline import DataPipeline
from models.elo import EloEngine
from feature_store import FeatureStore
from models.xgboost_model import XGBoostModel
from models.poisson_model import DixonColesPoisson
from monte_carlo import MonteCarloSimulator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def run_data():
    logger.info("=== STAGE: Data Pipeline ===")
    dp = DataPipeline()
    dp.run_pipeline(min_date="2014-01-01", download=True)

def run_elo():
    logger.info("=== STAGE: ELO Engine ===")
    df = pd.read_csv("data/processed/clean_matches.csv", parse_dates=["date"])
    elo = EloEngine()
    elo.process_matches(df)
    elo.export_history()

def run_features():
    logger.info("=== STAGE: Feature Store ===")
    matches_df = pd.read_csv("data/processed/clean_matches.csv", parse_dates=["date"])
    elo_df = pd.read_csv("data/processed/elo_history.csv")
    fs = FeatureStore()
    df = fs.build(matches_df, elo_df)
    fs.export_feature_store(df, "data/processed/feature_store.csv")

def run_xgb():
    logger.info("=== STAGE: XGBoost Model ===")
    df = pd.read_csv("data/processed/feature_store.csv", parse_dates=["date"])
    date_index = df["date"]
    
    X, y = XGBoostModel.prepare_data(df)
    split_date = "2021-01-01"
    
    X_train, y_train, X_val, y_val = XGBoostModel.temporal_split(X, y, date_index, split_date)
    
    model = XGBoostModel()
    logger.info("Training XGBoost...")
    model.train(X_train, y_train, X_val, y_val)
    
    logger.info("Calibrating...")
    model.calibrate(X_val, y_val)
    
    logger.info("Evaluating...")
    metrics = model.evaluate(X_val, y_val)
    logger.info(f"Metrics: {metrics}")
    
    logger.info("Saving model...")
    # Need to pass MODELS_DIR
    model.save(Path("models"))

def run_poisson():
    logger.info("=== STAGE: Poisson Model ===")
    df = pd.read_csv("data/processed/feature_store.csv", parse_dates=["date"])
    matches = pd.read_csv("data/processed/clean_matches.csv", parse_dates=["date"])
    df = df.merge(matches[["match_id", "home_goals", "away_goals"]], on="match_id", how="left")
    
    model = DixonColesPoisson()
    X, y = model.prepare_data(df)
    
    logger.info("Fitting Poisson (MLE)...")
    params = model.fit(X, y)
    
    logger.info("Saving model...")
    model.save(params, Path("models") / "poisson_params.json")

def main():
    parser = argparse.ArgumentParser(description="Mundial Predictor Pipeline Orchestrator")
    parser.add_argument("--stages", type=str, default="all",
                        help="Comma-separated list of stages to run (data,elo,features,xgb,poisson,mc) or 'all'")
    parser.add_argument("--sims", type=int, default=10000, help="Number of Monte Carlo simulations to run")
    parser.add_argument("--closest-only", action="store_true", help="Simulate only closest matches for speed")
    args = parser.parse_args()

    stages = args.stages.split(",") if args.stages != "all" else ["data", "elo", "features", "xgb", "poisson", "mc"]

    if "data" in stages:
        run_data()
    if "elo" in stages:
        run_elo()
    if "features" in stages:
        run_features()
    if "xgb" in stages:
        run_xgb()
    if "poisson" in stages:
        run_poisson()
    if "mc" in stages:
        logger.info(f"=== STAGE: Monte Carlo Simulation ({args.sims} sims) ===")
        import subprocess
        cmd = ["python", "monte_carlo.py", "--n-sims", str(args.sims)]
        if args.closest_only:
            cmd.append("--closest-only")
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    logger.info("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
