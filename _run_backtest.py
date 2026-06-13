"""Run backtesting for 2014, 2018, 2022 World Cups."""
import logging
import pandas as pd
from backtesting.evaluator import BacktestEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

fs = pd.read_csv("data/processed/feature_store.csv", parse_dates=["date"])
bt = BacktestEvaluator()

for year in [2014, 2018, 2022]:
    print(f"\n{'='*40}")
    print(f"  {year} World Cup")
    print('='*40)
    results = bt.walk_forward(fs, year)
    if results is not None and not results.empty:
        print(f"  Matches: {len(results)}")
        if "brier" in results.columns:
            print(f"  Brier: {results['brier'].mean():.4f}")
        if "rps" in results.columns:
            print(f"  RPS: {results['rps'].mean():.4f}")
        if "correct" in results.columns:
            acc = results["correct"].mean()
            print(f"  Accuracy: {acc*100:.1f}%")
        results.to_csv(f"backtesting/results/metrics_{year}.csv", index=False)
        print(f"  Saved to backtesting/results/metrics_{year}.csv")
    else:
        print(f"  No results for {year}")

print("\nBacktesting complete!")
