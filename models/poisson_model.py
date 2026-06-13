"""
models/poisson_model.py — Stage 4b: Bivariate Poisson with Dixon-Coles Adjustment

Implements a bivariate Poisson regression model for predicting exact football
scores, using the Dixon-Coles (1997) adjustment for dependence in low-scoring
draws.

The model:
  lambda_home = exp(alpha + beta_home @ X)
  lambda_away = exp(alpha + beta_away @ X)

  P(score = x,y) = tau(x,y) x Pois(x | lambda_home) x Pois(y | lambda_away)

  tau(x,y) = 1 - rho*x*y   for x=y in {0,1}
  tau(x,y) = 1 + rho       for x=y >= 2
  tau(x,y) = 1             otherwise

Based on spec R5.1-R5.6 and design C5.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Features to use (same as XGBoost — all non-target, non-identity columns)
FEATURE_COLUMNS: List[str] = [
    "elo_diff",
    "elo_diff_sq",
    "form_home_3f",
    "form_home_3a",
    "form_away_3f",
    "form_away_3a",
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
    "streak_home",
    "streak_away",
    "tournament_importance",
    "has_real_odds",
    "implied_home",
    "implied_draw",
    "implied_away",
]


# ===================================================================
# DixonColesPoisson
# ===================================================================
class DixonColesPoisson:
    """Bivariate Poisson regression with Dixon-Coles adjustment (R5.1-R5.6).

    Parameters
    ----------
    max_goals : int
        Maximum goals per team considered in the score matrix (default 6).
    feature_columns : list[str] | None
        Which feature columns to use.  Defaults to ``FEATURE_COLUMNS``.
    """

    def __init__(
        self,
        max_goals: int = 6,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        self.max_goals: int = max_goals
        self.feature_columns: List[str] = feature_columns or FEATURE_COLUMNS.copy()
        self.params_: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Data preparation (Task 3.1)
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        feature_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Separate features (X) from score targets (y).

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature store DataFrame containing feature columns plus
            ``home_goals`` and ``away_goals``.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (available columns from ``self.feature_columns``).
        y : np.ndarray
            Target array of shape ``(n_matches, 2)`` with columns
            ``[home_goals, away_goals]``.
        """
        available = [c for c in self.feature_columns if c in feature_df.columns]
        if len(available) < 1:
            raise ValueError(
                f"No feature columns found. Need at least 1 of "
                f"{self.feature_columns}, got {list(feature_df.columns)}"
            )
        if len(available) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available)
            logger.warning(
                "Missing feature columns: %s — using %d/%d features",
                missing, len(available), len(self.feature_columns),
            )

        X = feature_df[available].copy()
        y = feature_df[["home_goals", "away_goals"]].values.astype(np.float64)

        # Validate scores are non-negative integers
        if (y < 0).any():
            raise ValueError("Negative scores found in target data.")
        if not np.all(y == y.astype(int)):
            logger.warning(
                "Some scores appear to be non-integer — coercing to int."
            )

        return X, y

    # ------------------------------------------------------------------
    # Dixon-Coles log-likelihood (R5.2, R5.3)
    # ------------------------------------------------------------------

    def dc_log_likelihood(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Negative log-likelihood of the Dixon-Coles model.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector:
            ``[alpha, beta_home[0..k-1], beta_away[0..k-1], rho]``.
        X : np.ndarray
            Standardised feature matrix ``(n, k)``.
        y : np.ndarray
            Goal array ``(n, 2) = [home_goals, away_goals]``.

        Returns
        -------
        float
            Negative log-likelihood (minimisation target).
        """
        n_features = X.shape[1]
        alpha = params[0]
        beta_home = params[1 : 1 + n_features]
        beta_away = params[1 + n_features : 1 + 2 * n_features]
        rho = params[-1]

        # Expected goals (R5.1)
        eta_home = alpha + X @ beta_home
        eta_away = alpha + X @ beta_away

        # Clip to avoid overflow
        eta_home = np.clip(eta_home, -10, 10)
        eta_away = np.clip(eta_away, -10, 10)

        lambda_h = np.exp(eta_home)
        lambda_a = np.exp(eta_away)

        # Small epsilon to avoid log(0)
        eps = 1e-12
        lambda_h = np.maximum(lambda_h, eps)
        lambda_a = np.maximum(lambda_a, eps)

        home_goals = y[:, 0].astype(int)
        away_goals = y[:, 1].astype(int)

        # Poisson log-probability (log-likelihood per match, before tau)
        # log(P) = -lambda + x*log(lambda) - log(x!)
        log_prob = (
            -lambda_h + home_goals * np.log(lambda_h) - gammaln(home_goals + 1)
            + -lambda_a + away_goals * np.log(lambda_a) - gammaln(away_goals + 1)
        )

        # Dixon-Coles tau adjustment (R5.2)
        # tau(x,y) = 1 - rho*x*y  for x=y in {0,1}
        # tau(x,y) = 1 + rho      for x=y >= 2
        # tau(x,y) = 1            otherwise
        is_draw = home_goals == away_goals
        is_low_draw = is_draw & (home_goals <= 1)
        is_high_draw = is_draw & (home_goals >= 2)

        log_tau = np.zeros(len(home_goals))
        if np.any(is_low_draw):
            tau_val = 1.0 - rho * home_goals[is_low_draw] * away_goals[is_low_draw]
            tau_val = np.maximum(tau_val, eps)
            log_tau[is_low_draw] = np.log(tau_val)
        if np.any(is_high_draw):
            tau_val = 1.0 + rho
            tau_val = max(tau_val, eps)
            log_tau[is_high_draw] = np.log(tau_val)

        log_prob += log_tau

        nll = -np.sum(log_prob)

        if np.isnan(nll) or np.isinf(nll):
            return 1e12  # large penalty for invalid params

        return float(nll)

    # ------------------------------------------------------------------
    # MLE fitting (Task 3.1)
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        method: str = "L-BFGS-B",
        maxiter: int = 5000,
    ) -> np.ndarray:
        """Estimate Dixon-Coles parameters via MLE (R5.3).

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : np.ndarray
            Training targets ``(n, 2) = [home_goals, away_goals]``.
        method : str
            SciPy optimisation method (default ``L-BFGS-B``).
        maxiter : int
            Maximum optimisation iterations.

        Returns
        -------
        np.ndarray
            Fitted parameter vector ``[alpha, beta_home..., beta_away..., rho]``.
        """
        n_features = X_train.shape[1]
        n_params = 1 + n_features + n_features + 1  # alpha, βh, βa, rho

        # Standardise features (MLE converges faster)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Initial parameter guesses
        # alpha ~ log(mean goals per match) ≈ log(1.4) ≈ 0.34
        mean_goals = y_train.mean()
        alpha_init = np.log(max(mean_goals.mean(), 0.5))
        # beta near zero (features have small normalised effects)
        beta_init = np.zeros(n_features)
        # rho starts at 0 (no adjustment)
        rho_init = 0.0

        x0 = np.concatenate([[alpha_init], beta_init, beta_init, [rho_init]])

        # Bounds
        # alpha: unconstrained
        # beta_home, beta_away: moderate range (standardised features)
        # rho: must be in (-1, 1) for tau > 0
        bounds = (
            [(-5.0, 5.0)]                          # alpha
            + [(-3.0, 3.0)] * n_features           # beta_home
            + [(-3.0, 3.0)] * n_features           # beta_away
            + [(-0.99, 0.99)]                      # rho
        )

        logger.info(
            "Fitting Dixon-Coles Poisson: %d params, %d samples, %d features",
            n_params, len(X_train), n_features,
        )

        result = minimize(
            self.dc_log_likelihood,
            x0,
            args=(X_scaled, y_train),
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-8},
        )

        if not result.success:
            logger.warning(
                "Optimisation did not converge: %s", result.message
            )

        self.params_ = result.x
        rho_estimate = self.params_[-1]

        logger.info(
            "MLE complete: rho=%.4f, neg-log-lik=%.2f, %s",
            rho_estimate, result.fun, result.message,
        )

        # Verify rho range (R5.3 verification)
        if abs(rho_estimate) > 0.5:
            logger.warning(
                "rho=%.4f is outside expected range [-0.5, 0.5] — "
                "model may be overfitted",
                rho_estimate,
            )

        return self.params_

    # ------------------------------------------------------------------
    # Prediction helpers (Task 3.2)
    # ------------------------------------------------------------------

    def predict_lambdas(
        self,
        X: pd.DataFrame,
        params: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute expected goals for each match.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        params : np.ndarray | None
            Parameter vector.  Uses ``self.params_`` if ``None``.

        Returns
        -------
        lambda_home : np.ndarray
            Expected home goals per match.
        lambda_away : np.ndarray
            Expected away goals per match.
        """
        if params is None:
            if self.params_ is None:
                raise ValueError("Model not fitted — call fit() or load() first.")
            params = self.params_

        if self.scaler is None:
            raise ValueError("No scaler found — model was not fitted correctly.")

        # Filter to the features this model was trained on
        if hasattr(self, 'feature_names_') and self.feature_names_:
            available = [c for c in self.feature_names_ if c in X.columns]
            X = X[available]
        elif hasattr(self.scaler, 'feature_names_in_'):
            available = [c for c in self.scaler.feature_names_in_ if c in X.columns]
            X = X[available]

        X_scaled = self.scaler.transform(X)

        n_features = X_scaled.shape[1]
        alpha = params[0]
        beta_home = params[1 : 1 + n_features]
        beta_away = params[1 + n_features : 1 + 2 * n_features]

        eta_home = alpha + X_scaled @ beta_home
        eta_away = alpha + X_scaled @ beta_away

        lambda_h = np.exp(np.clip(eta_home, -10, 10))
        lambda_a = np.exp(np.clip(eta_away, -10, 10))

        return lambda_h, lambda_a

    @staticmethod
    def tau(x: int, y: int, rho: float) -> float:
        """Dixon-Coles adjustment factor (R5.2).

        Parameters
        ----------
        x : int
            Home goals.
        y : int
            Away goals.
        rho : float
            Dependence parameter.

        Returns
        -------
        float
            tau(x, y) adjustment factor.
        """
        if x == y:
            if x <= 1:
                result = 1.0 - rho * x * y
            else:
                result = 1.0 + rho
        else:
            result = 1.0
        return max(result, 1e-12)

    def exact_score_prob(
        self,
        lambda_h: float,
        lambda_a: float,
        rho: float,
        max_goals: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the full score probability matrix (R5.4).

        Returns a ``(max_goals+1) x (max_goals+1)`` matrix where
        ``M[i, j] = P(home=i, away=j)``.

        Parameters
        ----------
        lambda_h : float
            Expected home goals.
        lambda_a : float
            Expected away goals.
        rho : float
            Dixon-Coles dependence parameter.
        max_goals : int | None
            Maximum goals to consider.  Defaults to ``self.max_goals``.

        Returns
        -------
        np.ndarray
            ``(max_goals+1, max_goals+1)`` score probability matrix.
        """
        if max_goals is None:
            max_goals = self.max_goals

        goals = np.arange(max_goals + 1)

        # Poisson probabilities via log-space (numerically stable)
        log_poisson_h = -lambda_h + goals * np.log(lambda_h) - gammaln(goals + 1)
        log_poisson_a = -lambda_a + goals * np.log(lambda_a) - gammaln(goals + 1)
        poisson_h = np.exp(log_poisson_h)
        poisson_a = np.exp(log_poisson_a)

        # Outer product for independent Poisson
        score_matrix = np.outer(poisson_h, poisson_a)

        # Apply Dixon-Coles tau adjustment
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                score_matrix[i, j] *= self.tau(i, j, rho)

        # Normalise to ensure sum = 1
        total = score_matrix.sum()
        if total > 0:
            score_matrix /= total

        return score_matrix

    @staticmethod
    def match_1x2_from_score_matrix(
        score_matrix: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute marginal 1X2 probabilities from a score matrix (R5.4).

        Parameters
        ----------
        score_matrix : np.ndarray
            ``(max_goals+1, max_goals+1)`` score probability matrix.

        Returns
        -------
        p_home : float
            Probability of home win.
        p_draw : float
            Probability of draw.
        p_away : float
            Probability of away win.
        """
        n = score_matrix.shape[0]
        goals = np.arange(n)

        # Home win: i > j
        home_mask = goals[:, None] > goals[None, :]   # (n, n)
        p_home = score_matrix[home_mask].sum()

        # Draw: i == j
        draw_mask = goals[:, None] == goals[None, :]
        p_draw = score_matrix[draw_mask].sum()

        # Away win: i < j
        away_mask = goals[:, None] < goals[None, :]
        p_away = score_matrix[away_mask].sum()

        return float(p_home), float(p_draw), float(p_away)

    # ------------------------------------------------------------------
    # Evaluation (Task 3.3)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate exact-score predictions.

        Parameters
        ----------
        y_true : np.ndarray
            True scores ``(n, 2) = [home_goals, away_goals]``.
        y_pred_probs : np.ndarray
            Predicted score matrices ``(n, max_goals+1, max_goals+1)``.

        Returns
        -------
        dict
            Metrics:
            - ``rps``: Ranked Probability Score on marginal 1X2
            - ``log_loss``: Multi-class log-loss over exact scores
            - ``n_matches``: number of matches evaluated
            - ``top_score_accuracy``: proportion where modal predicted score matches
        """
        n_matches = len(y_true)
        max_goals = y_pred_probs.shape[1] - 1

        log_loss_total = 0.0
        rps_total = 0.0
        top_score_hits = 0

        for idx in range(n_matches):
            score_matrix = y_pred_probs[idx]
            hg = min(int(y_true[idx, 0]), max_goals)
            ag = min(int(y_true[idx, 1]), max_goals)

            # Log-loss: -log P(true score | model)
            # Clip zero probabilities
            prob_true = max(score_matrix[hg, ag], 1e-15)
            log_loss_total += -np.log(prob_true)

            # 1X2 marginal probabilities
            p_h, p_d, p_a = self.match_1x2_from_score_matrix(score_matrix)
            pred_1x2 = np.array([p_h, p_d, p_a])
            pred_1x2 = np.maximum(pred_1x2, 1e-15)

            # 1X2 actual outcome
            if hg > ag:
                actual_1x2 = np.array([1, 0, 0])
            elif hg == ag:
                actual_1x2 = np.array([0, 1, 0])
            else:
                actual_1x2 = np.array([0, 0, 1])

            # RPS for 1X2 (cumulative difference squared)
            rps_total += float(
                np.sum((np.cumsum(pred_1x2) - np.cumsum(actual_1x2)) ** 2)
                / (3 - 1)  # number of categories - 1
            )

            # Top-score accuracy
            max_idx = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
            if int(max_idx[0]) == hg and int(max_idx[1]) == ag:
                top_score_hits += 1

        avg_log_loss = log_loss_total / n_matches
        avg_rps = rps_total / n_matches
        top_score_acc = top_score_hits / n_matches

        logger.info(
            "Poisson evaluation: RPS=%.4f, LogLoss=%.4f, TopScoreAcc=%.2f%% (n=%d)",
            avg_rps, avg_log_loss, top_score_acc * 100, n_matches,
        )

        return {
            "rps": round(avg_rps, 4),
            "log_loss": round(avg_log_loss, 4),
            "top_score_accuracy": round(top_score_acc, 4),
            "n_matches": n_matches,
        }

    # ------------------------------------------------------------------
    # Persistence (Task 3.3)
    # ------------------------------------------------------------------

    def save(self, params: np.ndarray, path: str | Path) -> Path:
        """Save fitted parameters and scaler to JSON (R5.6).

        Parameters
        ----------
        params : np.ndarray
            Parameter vector to save.
        path : str | Path
            Output path (typically ``models/poisson_params.json``).

        Returns
        -------
        Path
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Mean goals from training (for reference)
        alpha = float(params[0])
        n_features = (len(params) - 2) // 2
        beta_home = params[1 : 1 + n_features].tolist()
        beta_away = params[1 + n_features : 1 + 2 * n_features].tolist()
        rho = float(params[-1])

        scaler_data = None
        if self.scaler is not None:
            scaler_data = {
                "mean_": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") and self.scaler.mean_ is not None else None,
                "scale_": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") and self.scaler.scale_ is not None else None,
                "var_": self.scaler.var_.tolist() if hasattr(self.scaler, "var_") and self.scaler.var_ is not None else None,
                "n_features_in_": self.scaler.n_features_in_ if hasattr(self.scaler, "n_features_in_") else len(beta_home),
                "feature_names_in_": list(self.scaler.feature_names_in_) if hasattr(self.scaler, "feature_names_in_") else None,
            }

        data = {
            "alpha": alpha,
            "beta_home": beta_home,
            "beta_away": beta_away,
            "rho": rho,
            "n_features": n_features,
            "max_goals": self.max_goals,
            "feature_columns": self.feature_columns,
            "scaler": scaler_data,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved Poisson params → %s", path)
        logger.info("  alpha=%.4f, rho=%.4f, %d features", alpha, rho, n_features)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DixonColesPoisson":
        """Load fitted parameters from JSON.

        Parameters
        ----------
        path : str | Path
            Path to ``poisson_params.json``.

        Returns
        -------
        DixonColesPoisson
            Restored instance with ``params_`` and ``scaler``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Poisson params not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(
            max_goals=data.get("max_goals", 6),
            feature_columns=data.get("feature_columns"),
        )

        n_features = data["n_features"]
        alpha = data["alpha"]
        beta_home = data["beta_home"]
        beta_away = data["beta_away"]
        rho = data["rho"]

        instance.params_ = np.array(
            [alpha] + beta_home + beta_away + [rho]
        )

        # Restore scaler
        scaler_data = data.get("scaler")
        if scaler_data is not None and scaler_data.get("mean_") is not None:
            instance.scaler = StandardScaler()
            instance.scaler.mean_ = np.array(scaler_data["mean_"])
            instance.scaler.scale_ = np.array(scaler_data["scale_"])
            if scaler_data.get("var_") is not None:
                instance.scaler.var_ = np.array(scaler_data["var_"])
            if scaler_data.get("n_features_in_") is not None:
                instance.scaler.n_features_in_ = scaler_data["n_features_in_"]
            if scaler_data.get("feature_names_in_") is not None:
                instance.scaler.feature_names_in_ = scaler_data["feature_names_in_"]

        logger.info(
            "Loaded Poisson params from %s (alpha=%.4f, rho=%.4f)",
            path, alpha, rho,
        )
        return instance

    # ------------------------------------------------------------------
    # Sampling helpers for Monte Carlo
    # ------------------------------------------------------------------

    def sample_score(
        self,
        lambda_h: float,
        lambda_a: float,
        rho: float,
        rng: Optional[np.random.Generator] = None,
        max_goals: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Sample a single exact score from the Poisson model.

        Parameters
        ----------
        lambda_h : float
            Expected home goals.
        lambda_a : float
            Expected away goals.
        rho : float
            Dixon-Coles rho parameter.
        rng : np.random.Generator | None
            Random number generator.  Uses a default if None.
        max_goals : int | None
            Maximum goals to consider.

        Returns
        -------
        home_goals : int
            Sampled home goals.
        away_goals : int
            Sampled away goals.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        if max_goals is None:
            max_goals = self.max_goals

        score_matrix = self.exact_score_prob(lambda_h, lambda_a, rho, max_goals)
        flat = score_matrix.ravel()
        flat = np.maximum(flat, 0)

        # Normalise (guard against numerical drift)
        flat /= flat.sum()

        idx = rng.choice(len(flat), p=flat)
        home_goals = idx // (max_goals + 1)
        away_goals = idx % (max_goals + 1)
        return int(home_goals), int(away_goals)


# ===================================================================
# CLI entry point
# ===================================================================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Train Dixon-Coles Poisson model"
    )
    parser.add_argument(
        "--features",
        default="data/processed/feature_store.csv",
        help="Path to feature_store.csv",
    )
    parser.add_argument(
        "--split-date",
        default="2023-01-01",
        help="Temporal split threshold (train <= date, val > date)",
    )
    parser.add_argument(
        "--output",
        default="models/poisson_params.json",
        help="Output path for parameters",
    )
    args = parser.parse_args()

    # Load feature store
    logger.info("Loading feature store from %s …", args.features)
    fs = pd.read_csv(args.features, parse_dates=["date"])

    model = DixonColesPoisson()

    # Prepare data — need home_goals and away_goals columns
    # (feature_store.csv doesn't have these by default; load clean_matches for scores)
    logger.info("Loading clean matches for goal data …")
    matches = pd.read_csv(
        "data/processed/clean_matches.csv",
        parse_dates=["date"],
    )
    # Merge scores onto features
    fs = fs.merge(
        matches[["match_id", "home_goals", "away_goals"]],
        on="match_id",
        how="left",
    )

    X, y = model.prepare_data(fs)

    # Temporal split
    train_mask = fs["date"] <= pd.Timestamp(args.split_date)
    X_train, y_train = X[train_mask].values, y[train_mask]
    X_val, y_val = X[~train_mask].values, y[~train_mask]

    logger.info("Train: %d, Val: %d", len(X_train), len(X_val))

    # Fit
    params = model.fit(pd.DataFrame(X_train, columns=X.columns), y_train)

    # Evaluate on validation
    val_lambdas = model.predict_lambdas(
        pd.DataFrame(X_val, columns=X.columns)
    )
    val_score_matrices = np.array([
        model.exact_score_prob(val_lambdas[0][i], val_lambdas[1][i], params[-1])
        for i in range(len(X_val))
    ])
    metrics = model.evaluate(y_val, val_score_matrices)
    print(f"\nValidation metrics:")
    print(f"  RPS:      {metrics['rps']:.4f}")
    print(f"  LogLoss:  {metrics['log_loss']:.4f}")
    print(f"  TopScAcc: {metrics['top_score_accuracy']:.2%}")

    # Save
    model.save(params, args.output)
    print(f"\nParams saved to {args.output}")
    print(f"  alpha={params[0]:.4f}, rho={params[-1]:.4f}")
