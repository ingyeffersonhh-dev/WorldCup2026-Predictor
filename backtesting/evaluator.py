"""
backtesting/evaluator.py — Stage 5b: Walk-Forward Backtesting & Analysis

Walk-forward backtesting for 2014, 2018, 2022 World Cups with:

- Brier score, RPS, log-loss metrics
- Fractional Kelly ROI simulation (f=0.25)
- Calibration curves and calibration-by-round confusion matrices
- Results plots saved to backtesting/results/

Based on spec R7.1-R7.8 and design C7.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# World Cup year → date ranges (tournament start dates)
# ---------------------------------------------------------------------------
WC_START_DATES: Dict[int, str] = {
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}

# Tournament match counts per format (32-team: 64 matches, 48-team: 104)
WC_GROUP_MATCHES: Dict[int, int] = {
    2014: 48,
    2018: 48,
    2022: 48,
}

# Feature columns used by the XGBoost model (mirrors xgboost_model.py)
FEATURE_COLUMNS: List[str] = [
    "elo_diff",
    "form_home_5f",
    "form_home_5a",
    "form_away_5f",
    "form_away_5a",
    "form_home_10f",
    "form_home_10a",
    "form_away_10f",
    "form_away_10a",
    "h2h_avg_diff",
    "home_advantage",
    "rest_days_home",
    "rest_days_away",
    "implied_home",
    "implied_draw",
    "implied_away",
]

TARGET_LABELS = {0: "draw", 1: "home", 2: "away"}


# ===================================================================
# BacktestEvaluator
# ===================================================================
class BacktestEvaluator:
    """Walk-forward backtesting for historical World Cups.

    For each tournament year (2014, 2018, 2022):
      1. Train an XGBoost model on ALL data *before* the tournament.
      2. Predict each tournament match.
      3. Compare with actual results.
      4. Compute and store evaluation metrics.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing model artifacts (used for param defaults).
    results_dir : str | Path
        Output directory for metrics CSVs and plots.
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        results_dir: str | Path = "backtesting/results",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API — full backtest run
    # ------------------------------------------------------------------

    def run_all(
        self,
        feature_store_path: str | Path = "data/processed/feature_store.csv",
        clean_matches_path: str | Path = "data/processed/clean_matches.csv",
        years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Run walk-forward backtesting for all specified World Cup years.

        Parameters
        ----------
        feature_store_path : str | Path
            Path to ``feature_store.csv``.
        clean_matches_path : str | Path
            Path to ``clean_matches.csv`` (needed for tournament_type).
        years : list[int] | None
            World Cup years to evaluate.  Defaults to ``[2014, 2018, 2022]``.

        Returns
        -------
        pd.DataFrame
            Consolidated metrics for all years.
        """
        from models.xgboost_model import XGBoostModel

        if years is None:
            years = [2014, 2018, 2022]

        # ── Load and merge data ──────────────────────────────────────
        logger.info("Loading feature store from %s …", feature_store_path)
        fs = pd.read_csv(feature_store_path, parse_dates=["date"])

        logger.info("Loading clean matches from %s …", clean_matches_path)
        cm = pd.read_csv(clean_matches_path, parse_dates=["date"])

        # Merge tournament_type and actual goals onto feature store
        data = fs.merge(
            cm[["match_id", "tournament_type", "home_goals", "away_goals"]],
            on="match_id",
            how="left",
        )

        if "tournament_type" not in data.columns:
            raise ValueError(
                "Could not merge tournament_type. "
                "Check that clean_matches.csv has match_id column."
            )

        logger.info("Merged data: %d rows", len(data))

        # Identify World Cup matches
        data["is_wc"] = data["tournament_type"].str.strip().str.lower() == "fifa world cup"

        # ── Walk-forward per year ────────────────────────────────────
        all_metrics: List[Dict[str, Any]] = []
        all_predictions: List[pd.DataFrame] = []

        for year in years:
            logger.info("=" * 60)
            logger.info("Walk-forward: %d World Cup", year)

            year_metrics, year_preds = self._walk_forward_single(
                data, year, XGBoostModel
            )
            all_metrics.append(year_metrics)
            all_predictions.append(year_preds)

            # Save per-year predictions
            pred_path = self.results_dir / f"predictions_{year}.csv"
            year_preds.to_csv(pred_path, index=False)
            logger.info("Saved %d predictions → %s", len(year_preds), pred_path)

        # ── Consolidate metrics ──────────────────────────────────────
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = self.results_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Saved metrics summary → %s", metrics_path)

        # ── Generate plots ───────────────────────────────────────────
        combined_preds = pd.concat(all_predictions, ignore_index=True)
        self.plot_results(metrics_df, combined_preds)

        return metrics_df

    # ------------------------------------------------------------------
    # Single-year walk-forward (Task 4.1)
    # ------------------------------------------------------------------

    def _walk_forward_single(
        self,
        data: pd.DataFrame,
        year: int,
        XGBoostModel: type,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Run walk-forward for one World Cup year.

        Steps:
          1. Filter World Cup matches for *year*.
          2. Train model on data *before* the first match of that WC.
          3. Predict WC match outcomes.
          4. Compute and return metrics + predictions.
        """
        # ── Identify WC matches for this year ────────────────────────
        wc_mask = data["is_wc"] & (data["date"].dt.year == year)
        wc_matches = data[wc_mask].copy()

        if wc_matches.empty:
            logger.warning("No FIFA World Cup matches found for %d — skipping", year)
            return (
                {
                    "year": year,
                    "n_matches": 0,
                    "brier_score": float("nan"),
                    "rps": float("nan"),
                    "log_loss": float("nan"),
                    "accuracy": float("nan"),
                    "n_teams_trained": 0,
                },
                pd.DataFrame(),
            )

        first_match_date = wc_matches["date"].min()
        logger.info(
            "  %d WC matches, starts %s",
            len(wc_matches), first_match_date.date(),
        )

        # ── Pre-tournament training data ─────────────────────────────
        train_data = data[data["date"] < first_match_date].copy()
        logger.info("  Training data: %d matches before %s", len(train_data), first_match_date.date())

        if len(train_data) < 100:
            logger.warning("  Too few train samples (%d) — skipping %d", len(train_data), year)
            return (
                {
                    "year": year,
                    "n_matches": 0,
                    "brier_score": float("nan"),
                    "rps": float("nan"),
                    "log_loss": float("nan"),
                    "accuracy": float("nan"),
                    "n_teams_trained": 0,
                },
                pd.DataFrame(),
            )

        # ── Train model ──────────────────────────────────────────────
        model = XGBoostModel()

        # Prepare features
        X_all, y_all = XGBoostModel.prepare_data(train_data)

        # Temporal split: 80% train, 20% val (for early stopping + calibration)
        train_dates = train_data["date"]
        split_idx = int(len(X_all) * 0.8)
        split_date = train_dates.iloc[split_idx]

        X_train, y_train, X_val, y_val = XGBoostModel.temporal_split(
            X_all, y_all, train_dates, str(split_date.date())
        )

        if len(X_train) < 50 or len(X_val) < 10:
            logger.warning("  Insufficient training data after split — skipping %d", year)
            return (
                {
                    "year": year,
                    "n_matches": 0,
                    "brier_score": float("nan"),
                    "rps": float("nan"),
                    "log_loss": float("nan"),
                    "accuracy": float("nan"),
                    "n_teams_trained": 0,
                },
                pd.DataFrame(),
            )

        logger.info("  Training: %d samples, Validation: %d samples", len(X_train), len(X_val))
        model.train(X_train, y_train, X_val, y_val)

        # Calibrate
        try:
            model.calibrate(X_val, y_val)
            logger.info("  Calibration complete (isotonic)")
        except Exception as exc:
            logger.warning("  Calibration failed (%s) — using raw probabilities", exc)

        # ── Predict on WC matches ────────────────────────────────────
        X_wc, y_wc = XGBoostModel.prepare_data(wc_matches)
        y_pred = model.predict_proba(X_wc)

        # y_pred shape: (n_matches, 3), order: [P(draw), P(home), P(away)]
        # y_wc values: 0=draw, 1=home, 2=away

        # Build predictions DataFrame
        pred_df = wc_matches[["match_id", "date", "home_team", "away_team",
                              "home_goals", "away_goals", "target"]].copy()
        pred_df["p_draw"] = y_pred[:, 0]
        pred_df["p_home"] = y_pred[:, 1]
        pred_df["p_away"] = y_pred[:, 2]
        pred_df["predicted_class"] = np.argmax(y_pred, axis=1)
        pred_df["correct"] = (pred_df["predicted_class"] == pred_df["target"]).astype(int)

        # Predicted winner name
        pred_df["predicted_winner"] = pred_df.apply(
            lambda r: (
                "draw" if r["predicted_class"] == 0
                else r["home_team"] if r["predicted_class"] == 1
                else r["away_team"]
            ),
            axis=1,
        )
        pred_df["actual_winner"] = pred_df.apply(
            lambda r: (
                "draw" if r["target"] == 0
                else r["home_team"] if r["target"] == 1
                else r["away_team"]
            ),
            axis=1,
        )
        pred_df["year"] = year

        # Classify match round (group vs KO for confusion matrices)
        pred_df["match_type"] = self._classify_match_round(pred_df, year)

        # ── Compute metrics ──────────────────────────────────────────
        n_matches = len(pred_df)
        if n_matches == 0:
            return (
                {"year": year, "n_matches": 0, "brier_score": float("nan"),
                 "rps": float("nan"), "log_loss": float("nan"),
                 "accuracy": float("nan"), "n_teams_trained": 0},
                pred_df,
            )

        # Brier score
        brier = self.brier_score(pred_df["target"].values, y_pred)

        # RPS
        rps = self.ranked_probability_score(pred_df["target"].values, y_pred)

        # Log loss
        ll = self.log_loss(pred_df["target"].values, y_pred)

        # Accuracy
        acc = float(pred_df["correct"].mean())

        # Fractional Kelly ROI (if implied probs exist)
        roi = self.fractional_kelly_roi(
            y_pred,
            wc_matches[["implied_home", "implied_draw", "implied_away"]].values,
            pred_df["target"].values,
            f=0.25,
        )

        # Calibration curve
        calib = self.calibration_curve(pred_df["target"].values, y_pred)

        # Confusion matrix by round
        cm = self.confusion_matrix_by_round(
            pred_df["predicted_class"].values,
            pred_df["target"].values,
            pred_df["match_type"].values,
        )

        logger.info(
            "  %d: Brier=%.4f  RPS=%.4f  LogLoss=%.4f  Acc=%.1f%%  ROI=%.1f%%",
            year, brier, rps, ll, acc * 100, roi * 100,
        )

        metrics = {
            "year": year,
            "n_matches": n_matches,
            "brier_score": round(brier, 4),
            "rps": round(rps, 4),
            "log_loss": round(ll, 4),
            "accuracy": round(acc, 4),
            "roi": round(roi, 4),
            "calibration_data": calib,
            "confusion_matrix": cm,
            "n_teams_trained": len(model.feature_names),
        }

        # Save detailed metrics
        metrics_path = self.results_dir / f"metrics_{year}.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        return metrics, pred_df

    # ------------------------------------------------------------------
    # Match round classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_match_round(
        pred_df: pd.DataFrame,
        year: int,
    ) -> np.ndarray:
        """Classify each match as 'group' or 'knockout'.

        Uses chronological ordering: for a 32-team WC (all historical),
        matches 1-48 are group stage, 49-64 are KO.
        For 2026 (48-team), adjust to 72 group + 32 KO.
        """
        n_group = WC_GROUP_MATCHES.get(year, 48)
        n_total = len(pred_df)

        # Sort by date to get chronological order
        sorted_idx = pred_df["date"].argsort()
        result = np.full(n_total, "knockout", dtype=object)
        if n_total > 0:
            result[sorted_idx[:min(n_group, n_total)]] = "group"
        return result

    # ------------------------------------------------------------------
    # Evaluation metrics (Task 4.2)
    # ------------------------------------------------------------------

    @staticmethod
    def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Multi-class Brier score (Task 4.2).

        ``BS = (1/N) * sum_i sum_k (y_ik - p_ik)^2``

        Parameters
        ----------
        y_true : np.ndarray
            True class labels (0=draw, 1=home, 2=away).
        y_pred : np.ndarray
            Predicted probability matrix ``(N, 3)`` in order
            ``[P(draw), P(home), P(away)]``.

        Returns
        -------
        float
            Brier score (lower is better, range 0-2).
        """
        n = len(y_true)
        if n == 0:
            return float("nan")

        # One-hot encode true labels
        y_true_onehot = np.zeros((n, 3))
        y_true_onehot[np.arange(n), y_true.astype(int)] = 1.0

        return float(np.mean(np.sum((y_true_onehot - y_pred) ** 2, axis=1)))

    @staticmethod
    def ranked_probability_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Ranked Probability Score (RPS) for ordered outcomes (Task 4.2).

        The ordered categories for football are:
            home_win > draw > away_win

        So we reorder from ``[P(draw), P(home), P(away)]`` to
        ``[P(home), P(draw), P(away)]`` for the cumulative comparison.

        ``RPS = (1/(K-1)) * sum_{k=1}^{K-1} (cumsum(p)_k - cumsum(o)_k)^2``

        Parameters
        ----------
        y_true : np.ndarray
            True classes (0=draw, 1=home, 2=away).
        y_pred : np.ndarray
            Predicted probas ``[P(draw), P(home), P(away)]``.

        Returns
        -------
        float
            RPS (0 = perfect, 1 = worst).
        """
        n = len(y_true)
        if n == 0:
            return float("nan")

        # Reorder to [P(home), P(draw), P(away)] for ordered cumulative
        # Input:  [P(draw), P(home), P(away)]
        # Output: [P(home), P(draw), P(away)]
        pred_ordered = y_pred[:, [1, 0, 2]]

        # One-hot true in same order
        true_ordered = np.zeros((n, 3))
        # Map: 0(draw)→1, 1(home)→0, 2(away)→2
        remap = {0: 1, 1: 0, 2: 2}
        for i in range(n):
            col = remap[int(y_true[i])]
            true_ordered[i, col] = 1.0

        # Cumulative sums (excluding the last which is always 1)
        pred_cumsum = np.cumsum(pred_ordered, axis=1)[:, :2]
        true_cumsum = np.cumsum(true_ordered, axis=1)[:, :2]

        # RPS = mean of squared differences / (K-1)
        rps_per_match = np.sum((pred_cumsum - true_cumsum) ** 2, axis=1) / (3 - 1)
        return float(np.mean(rps_per_match))

    @staticmethod
    def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Multi-class logarithmic loss (Task 4.2).

        Parameters
        ----------
        y_true : np.ndarray
            True class labels.
        y_pred : np.ndarray
            Predicted probability matrix ``(N, 3)``.

        Returns
        -------
        float
            Log loss (lower is better).
        """
        from sklearn.metrics import log_loss as sk_log_loss

        n = len(y_true)
        if n == 0:
            return float("nan")

        # Clip to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(sk_log_loss(y_true, y_pred_clipped))

    @staticmethod
    def fractional_kelly_roi(
        predictions: np.ndarray,
        odds_implied: np.ndarray,
        results: np.ndarray,
        f: float = 0.25,
    ) -> float:
        """Simulated ROI using fractional Kelly criterion (Task 4.2).

        For each match, if the model's probability exceeds the implied
        probability from odds, we bet a fraction ``f`` of the bankroll
        on that outcome.

        ``edge = model_prob - implied_prob``

        Parameters
        ----------
        predictions : np.ndarray
            Model probability matrix ``(N, 3)``.
        odds_implied : np.ndarray
            Implied probability matrix ``(N, 3)`` from odds.
        results : np.ndarray
            True outcomes (0=draw, 1=home, 2=away).
        f : float
            Fraction of bankroll to risk per bet (default 0.25).

        Returns
        -------
        float
            Net ROI as a fraction (e.g., 0.15 = 15% return).
        """
        n = len(predictions)
        if n == 0 or odds_implied is None or len(odds_implied) != n:
            return float("nan")

        # Check if odds data is available (any non-uniform implied probs)
        uniform = (np.abs(odds_implied - 1 / 3) < 0.01).all(axis=1).all()
        if uniform:
            logger.info("  Odds appear uniform — skipping Kelly ROI")
            return float("nan")

        bankroll = 1.0
        initial_bankroll = 1.0

        for i in range(n):
            model_probs = predictions[i]
            implied_probs = odds_implied[i]
            true_outcome = int(results[i])

            # Find the best edge across the three outcomes
            for k in range(3):
                edge = model_probs[k] - implied_probs[k]
                if edge > 0 and implied_probs[k] > 0:
                    # Positive edge — place bet
                    # Decimal odds = 1 / implied_prob
                    decimal_odds = 1.0 / implied_probs[k]
                    # Kelly stake = f * edge / (decimal_odds - 1)
                    stake = f * (edge / (decimal_odds - 1.0))
                    stake = min(stake, 0.5)  # cap at 50% of bankroll

                    if stake > 0 and stake <= bankroll:
                        bet_amount = bankroll * stake
                        if k == true_outcome:
                            bankroll += bet_amount * (decimal_odds - 1.0)
                        else:
                            bankroll -= bet_amount

        total_roi = (bankroll - initial_bankroll) / initial_bankroll
        return float(total_roi)

    # ------------------------------------------------------------------
    # Error analysis (Task 4.3)
    # ------------------------------------------------------------------

    @staticmethod
    def calibration_curve(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, List[Dict[str, float]]]:
        """Compute calibration curve data per class (Task 4.3).

        Parameters
        ----------
        y_true : np.ndarray
            True class labels.
        y_pred : np.ndarray
            Predicted probability matrix ``(N, 3)``.
        n_bins : int
            Number of equal-width bins (default 10).

        Returns
        -------
        dict
            ``{class_name: [{"bin_center": ..., "predicted": ..., "observed": ...,
            "count": ...}, ...]}``
        """
        from sklearn.calibration import calibration_curve

        result: Dict[str, List[Dict[str, float]]] = {}
        class_names = ["draw", "home", "away"]

        for k in range(3):
            y_binary = (y_true == k).astype(int)
            prob_pred, prob_true = calibration_curve(
                y_binary, y_pred[:, k], n_bins=n_bins, strategy="uniform"
            )

            records = []
            for i in range(len(prob_pred)):
                records.append({
                    "bin_center": round(float(prob_pred[i]), 4),
                    "predicted": round(float(prob_pred[i]), 4),
                    "observed": round(float(prob_true[i]), 4),
                    "count": 0,  # sklearn doesn't return counts in this API
                })
            result[class_names[k]] = records

        return result

    @staticmethod
    def confusion_matrix_by_round(
        predictions: np.ndarray,
        actuals: np.ndarray,
        match_types: np.ndarray,
    ) -> Dict[str, Any]:
        """Confusion matrix broken down by group vs knockout (Task 4.3).

        Parameters
        ----------
        predictions : np.ndarray
            Predicted class labels.
        actuals : np.ndarray
            True class labels.
        match_types : np.ndarray
            Array of ``"group"`` or ``"knockout"`` per match.

        Returns
        -------
        dict
            ``{"group": {confusion matrix data}, "knockout": {...}}``
        """
        from sklearn.metrics import confusion_matrix

        unique_types = np.unique(match_types)
        result: Dict[str, Any] = {}

        for mtype in unique_types:
            mask = match_types == mtype
            if mask.sum() == 0:
                continue

            cm = confusion_matrix(actuals[mask], predictions[mask], labels=[0, 1, 2])
            result[str(mtype)] = {
                "matrix": cm.tolist(),
                "n": int(mask.sum()),
                "labels": ["draw", "home", "away"],
            }

        # Overall
        cm_all = confusion_matrix(actuals, predictions, labels=[0, 1, 2])
        result["overall"] = {
            "matrix": cm_all.tolist(),
            "n": len(actuals),
            "labels": ["draw", "home", "away"],
        }

        return result

    # ------------------------------------------------------------------
    # Plots (Task 4.3)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_results(
        metrics_df: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None,
        output_dir: str | Path = "backtesting/results",
    ) -> Dict[str, str]:
        """Generate and save result plots (Task 4.3).

        Creates:
          1. ``calibration.png`` — Per-class calibration curves
          2. ``confusion_matrix.png`` — Confusion matrices (group vs KO)
          3. ``metrics_over_time.png`` — Brier/RPS/ROI by year

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Metrics summary with one row per year.
        predictions : pd.DataFrame | None
            Combined predictions (used for calibration and confusion plots).
        output_dir : str | Path
            Output directory for plot files.

        Returns
        -------
        dict[str, str]
            ``{"calibration": path, "confusion": path, "metrics": path}``
        """
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        # ── 1. Metrics over time ─────────────────────────────────────
        if not metrics_df.empty and "year" in metrics_df.columns:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            years = metrics_df["year"].values

            for ax, metric, title, color in [
                (axes[0], "brier_score", "Brier Score (↓better)", "steelblue"),
                (axes[1], "rps", "RPS (↓better)", "forestgreen"),
                (axes[2], "accuracy", "Accuracy (↑better)", "darkorange"),
            ]:
                if metric in metrics_df.columns:
                    values = metrics_df[metric].values
                    ax.bar([str(y) for y in years], values, color=color, alpha=0.7)
                    ax.set_title(title, fontsize=12)
                    ax.set_ylim(0, max(1.0, np.nanmax(values) * 1.2))
                    for i, v in enumerate(values):
                        if not np.isnan(v):
                            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

            fig.suptitle("Walk-Forward Backtesting Metrics", fontsize=14)
            fig.tight_layout()
            metrics_path = output_dir / "metrics_over_time.png"
            fig.savefig(metrics_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths["metrics"] = str(metrics_path)
            logger.info("Saved metrics plot → %s", metrics_path)

        # ── 2. Calibration curve ─────────────────────────────────────
        if predictions is not None and len(predictions) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            class_names = ["Draw", "Home Win", "Away Win"]

            for k in range(3):
                from sklearn.calibration import calibration_curve
                y_binary = (predictions["target"].values == k).astype(int)
                prob_pred, prob_true = calibration_curve(
                    y_binary,
                    predictions[["p_draw", "p_home", "p_away"]].values[:, k],
                    n_bins=8,
                    strategy="uniform",
                )
                ax = axes[k]
                ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
                ax.plot(prob_pred, prob_true, "o-", color="C0", markersize=6)
                ax.set_xlabel("Predicted probability")
                ax.set_ylabel("Observed frequency")
                ax.set_title(class_names[k], fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)

            fig.suptitle("Calibration Curves (all years combined)", fontsize=14)
            fig.tight_layout()
            cal_path = output_dir / "calibration.png"
            fig.savefig(cal_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths["calibration"] = str(cal_path)
            logger.info("Saved calibration plot → %s", cal_path)

        # ── 3. Confusion matrix ──────────────────────────────────────
        if predictions is not None and len(predictions) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            titles = ["All Matches", "Group Stage", "Knockout"]
            subsets = [
                predictions,
                predictions[predictions["match_type"] == "group"],
                predictions[predictions["match_type"] == "knockout"],
            ]
            labels = ["Draw", "Home", "Away"]

            for ax, subset, title in zip(axes, subsets, titles):
                if len(subset) == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(title)
                    continue

                cm = confusion_matrix(
                    subset["target"].values,
                    subset["predicted_class"].values,
                    labels=[0, 1, 2],
                )
                im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
                ax.set_xticks(range(3))
                ax.set_yticks(range(3))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"{title} (n={len(subset)})")

                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(cm[i, j]),
                                ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() * 0.6 else "black")

            fig.suptitle("Confusion Matrix by Match Type", fontsize=14)
            fig.tight_layout()
            cm_path = output_dir / "confusion_matrix.png"
            fig.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths["confusion"] = str(cm_path)
            logger.info("Saved confusion matrix plot → %s", cm_path)

        return paths


# ===================================================================
# CLI entry point
# ===================================================================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Run walk-forward backtesting for World Cup years"
    )
    parser.add_argument(
        "--features",
        default="data/processed/feature_store.csv",
        help="Path to feature_store.csv",
    )
    parser.add_argument(
        "--matches",
        default="data/processed/clean_matches.csv",
        help="Path to clean_matches.csv",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2014, 2018, 2022],
        help="World Cup years to evaluate",
    )
    parser.add_argument(
        "--output",
        default="backtesting/results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    evaluator = BacktestEvaluator(results_dir=args.output)
    metrics = evaluator.run_all(
        feature_store_path=args.features,
        clean_matches_path=args.matches,
        years=args.years,
    )

    print(f"\nBacktesting complete. Results saved to {args.output}/")
    print(f"\n{'Year':>6}  Brier    RPS     LogLoss  Acc     ROI")
    print("-" * 55)
    for _, row in metrics.iterrows():
        print(
            f"{int(row['year']):>6}  "
            f"{row['brier_score']:.4f}  {row['rps']:.4f}  "
            f"{row['log_loss']:.4f}  {row['accuracy']:.2%}  "
            f"{row['roi']:.2%}" if not pd.isna(row.get('roi')) else "N/A"
        )
