"""
models/xgboost_model.py — Stage 4a: XGBoost Classifier (1X2)

Implements a multi-class XGBoost classifier for match outcome prediction
(1 = home win, 0 = draw, 2 = away win) with:

- Temporal train/val split (not random — respects match chronology)
- XGBoost with early stopping and regularisation
- CalibratedClassifierCV with isotonic regression
- Evaluation: Brier score, log-loss, reliability diagram data
- Feature importance via SHAP values
- Model persistence (pickle + XGBoost JSON + feature schema)

Based on spec R4.1-R4.9 and design C4.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import _CalibratedClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters (design C4 / Task 2.5)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "early_stopping_rounds": 20,
    "eval_metric": "mlogloss",
    "verbosity": 1,
}

# Feature columns expected by the model (in order)
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

TARGET_MAPPING = {1: "home_win", 0: "draw", 2: "away_win"}


# ===================================================================
# XGBoostModel
# ===================================================================
class XGBoostModel:
    """Multi-class match outcome predictor using XGBoost.

    Parameters
    ----------
    params : dict | None
        Hyperparameter overrides.  See ``DEFAULT_PARAMS``.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrated_model: Optional[_CalibratedClassifier] = None
        self.feature_names: List[str] = []
        self._fit_date: Optional[str] = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_data(
        feature_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features (X) from target (y).

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature store DataFrame containing feature columns and ``target``.
        feature_columns : list[str] | None
            Which columns to use as features.  Defaults to ``FEATURE_COLUMNS``.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector (1=home, 0=draw, 2=away).
        """
        cols = feature_columns or FEATURE_COLUMNS
        available = [c for c in cols if c in feature_df.columns]
        if len(available) < len(cols):
            missing = set(cols) - set(available)
            logger.warning(
                "Missing feature columns: %s — proceeding with %d/%d features",
                missing, len(available), len(cols),
            )

        X = feature_df[available].copy()
        y = feature_df["target"].copy()
        return X, y

    # ------------------------------------------------------------------
    # Temporal split (design C4 — not random)
    # ------------------------------------------------------------------

    @staticmethod
    def temporal_split(
        X: pd.DataFrame,
        y: pd.Series,
        date_index: pd.Series,
        split_date: str,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Train/validation split by chronological date threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        date_index : pd.Series
            Match dates aligned with ``X`` and ``y``.
        split_date : str
            Date threshold.  Rows with ``date <= split_date`` go to train,
            the rest to validation.

        Returns
        -------
        X_train, y_train, X_val, y_val
        """
        train_mask = date_index <= pd.Timestamp(split_date)
        val_mask = ~train_mask

        X_train = X[train_mask.values].copy() if hasattr(train_mask, "values") else X[train_mask].copy()
        y_train = y[train_mask.values].copy() if hasattr(train_mask, "values") else y[train_mask].copy()
        X_val = X[val_mask.values].copy() if hasattr(val_mask, "values") else X[val_mask].copy()
        y_val = y[val_mask.values].copy() if hasattr(val_mask, "values") else y[val_mask].copy()

        logger.info(
            "Temporal split at %s: %d train, %d val",
            split_date, len(X_train), len(X_val),
        )
        return X_train, y_train, X_val, y_val

    @staticmethod
    def time_series_cv(
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time-series cross-validation indices.

        Uses ``sklearn.model_selection.TimeSeriesSplit`` which respects
        temporal order (train always before validation).

        Returns
        -------
        list[tuple[ndarray, ndarray]]
            List of ``(train_idx, val_idx)`` pairs.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(X, y))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> xgb.XGBClassifier:
        """Train the XGBoost classifier with early stopping.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training targets.
        X_val : pd.DataFrame
            Validation features.
        y_val : pd.Series
            Validation targets.

        Returns
        -------
        xgb.XGBClassifier
            Trained classifier.
        """
        self.feature_names = list(X_train.columns)
        logger.info(
            "Training XGBoost: %d samples × %d features",
            len(X_train), len(self.feature_names),
        )

        self.model = xgb.XGBClassifier(**self.params, objective="multi:softprob")

        # Ensure integer target classes
        y_train_int = y_train.astype(int)
        y_val_int = y_val.astype(int)

        # Set num_class explicitly for multi-class
        self.model.set_params(num_class=len(np.unique(y_train_int)))

        self.model.fit(
            X_train,
            y_train_int,
            eval_set=[(X_train, y_train_int), (X_val, y_val_int)],
            verbose=self.params.get("verbosity", 0) > 0,
        )

        best_iter = self.model.best_iteration + 1  # 0-indexed
        logger.info(
            "Training complete: best iteration = %d, best mlogloss = %.4f",
            best_iter,
            self.model.best_score,
        )

        # Store fit date
        from datetime import datetime
        self._fit_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return self.model

    # ------------------------------------------------------------------
    # Calibration (R4.5)
    # ------------------------------------------------------------------

    def calibrate(
        self,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
    ) -> "_CalibratedClassifier":
        """Calibrate probabilities using isotonic regression.

        Uses sklearn's internal ``_fit_calibrator`` to fit isotonic
        calibrators per class on held-out (calibration) predictions.

        Parameters
        ----------
        X_calib : pd.DataFrame
            Calibration features (typically the validation set).
        y_calib : pd.Series
            Calibration targets.

        Returns
        -------
        _CalibratedClassifier
            Fitted pipeline of base estimator + calibrators.
        """
        if self.model is None:
            raise ValueError("Train the model before calibrating.")

        from sklearn.calibration import _fit_calibrator

        logger.info("Calibrating with isotonic regression on %d samples", len(X_calib))

        y_calib_int = y_calib.astype(int)
        raw_preds = self.model.predict_proba(X_calib)

        self.calibrated_model = _fit_calibrator(
            clf=self.model,
            predictions=raw_preds,
            y=y_calib_int.values,
            classes=self.model.classes_,
            method="isotonic",
            xp=None,
        )

        logger.info("Calibration complete")
        return self.calibrated_model

    # ------------------------------------------------------------------
    # Evaluation (R4.6)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Compute prediction metrics.

        Metrics returned:
        - ``brier_score``: Multi-class Brier score (average of per-class scores)
        - ``log_loss``:  Multi-class logarithmic loss
        - ``accuracy``:  Overall accuracy
        - ``reliability``: Reliability diagram data (predicted vs observed freq)

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test targets.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        model: Any = self.calibrated_model if self.calibrated_model is not None else self.model
        if model is None:
            raise ValueError("No trained model available.")

        y_test_int = y_test.astype(int)
        y_pred_proba = model.predict_proba(X_test)

        # Handle predict: _CalibratedClassifier has no .predict(), use argmax
        if hasattr(model, "predict"):
            y_pred = model.predict(X_test)
        else:
            y_pred = np.asarray(y_pred_proba).argmax(axis=1)

        n_classes = y_pred_proba.shape[1]

        # --- Per-class Brier scores (averaged) ---
        brier_scores: List[float] = []
        for i in range(n_classes):
            y_binary = (y_test_int == i).astype(int)
            brier_scores.append(brier_score_loss(y_binary, y_pred_proba[:, i]))
        brier = float(np.mean(brier_scores))

        # --- Log-loss ---
        ll = float(log_loss(y_test_int, y_pred_proba))

        # --- Accuracy ---
        acc = float((y_pred == y_test_int).mean())

        # --- Reliability diagram data ---
        reliability = self._reliability_curve(y_test_int, y_pred_proba, n_bins=10)

        result = {
            "brier_score": brier,
            "log_loss": ll,
            "accuracy": acc,
            "n_test": len(y_test_int),
            "reliability": reliability,
        }

        logger.info(
            "Evaluation: Brier=%.4f, LogLoss=%.4f, Acc=%.2f%% (n=%d)",
            brier, ll, acc * 100, len(y_test_int),
        )
        return result

    # ------------------------------------------------------------------
    # Prediction (exposed for downstream consumers like Monte Carlo)
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Return probability estimates for the input feature matrix.

        Delegates to the calibrated model if available, otherwise falls
        back to the raw XGBoost classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (must contain the same columns as training).

        Returns
        -------
        np.ndarray
            Probability array of shape ``(n_samples, n_classes)``.
            Class ordering follows ``classes_`` (typically ``[0, 1, 2]``
            mapping to draw, home_win, away_win).
        """
        model: Any = self.calibrated_model if self.calibrated_model is not None else self.model
        if model is None:
            raise ValueError("No trained model available — train or load first.")
        return model.predict_proba(X)

    # ------------------------------------------------------------------
    # Feature importance (R4.7)
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        X: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute feature importance via SHAP values.

        Uses ``shap.TreeExplainer`` on the raw (pre-calibration) XGBoost model.

        Parameters
        ----------
        X : pd.DataFrame | None
            Feature matrix to explain.  If ``None``, uses a small random sample
            for efficiency.

        Returns
        -------
        pd.DataFrame
            Columns ``feature``, ``importance`` (mean |SHAP|), sorted descending.
        """
        if self.model is None:
            raise ValueError("Model not trained yet — train before computing importance.")

        if X is None:
            raise ValueError("X is required for SHAP computation — pass the feature matrix.")

        import shap

        # Use a sample for efficiency (TreeExplainer can be slow on 10k+ rows)
        n_sample = min(5000, len(X))
        if n_sample < len(X):
            X_sample = X.sample(n=n_sample, random_state=42)
            logger.info("SHAP: using %d-row sample (of %d)", n_sample, len(X))
        else:
            X_sample = X

        logger.info("Computing SHAP values via TreeExplainer …")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        # shap_values shape for multi-class: (n_samples, n_features, n_classes)
        # Mean absolute SHAP per feature (averaged across classes and samples)
        if isinstance(shap_values, list):
            # Some shap versions return list of arrays per class
            mean_shap = np.abs(np.array(shap_values)).mean(axis=(0, 2))  # (n_features,)
        else:
            mean_shap = np.abs(shap_values).mean(axis=(0, 2))  # (n_features,)

        imp_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": mean_shap,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        logger.info("Top 5 features: %s", ", ".join(
            f"{r.feature}={r.importance:.4f}" for _, r in imp_df.head(5).iterrows()
        ))
        return imp_df

    # ------------------------------------------------------------------
    # Persistence (R4.8, R4.9)
    # ------------------------------------------------------------------

    def save(self, model_dir: str | Path = "models") -> Dict[str, str]:
        """Save model and metadata to disk.

        Creates three files:
        1. ``xgb_model.pkl`` — Pickled model + calibrator + feature names
        2. ``xgb_model.json`` — XGBoost booster in JSON format
        3. ``feature_schema.json`` — Feature names and metadata

        Parameters
        ----------
        model_dir : str | Path
            Output directory.

        Returns
        -------
        dict[str, str]
            ``{"pkl": …, "json": …, "schema": …}`` paths.
        """
        if self.model is None:
            raise ValueError("No model to save — train first.")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pickle full object state
        pkl_path = model_dir / "xgb_model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "calibrated_model": self.calibrated_model,
                    "feature_names": self.feature_names,
                    "params": self.params,
                    "fit_date": self._fit_date,
                },
                f,
            )
        logger.info("Saved model → %s", pkl_path)

        # 2. XGBoost JSON (booster format for cross-language portability)
        json_path = model_dir / "xgb_model.json"
        self.model.get_booster().save_model(str(json_path))
        logger.info("Saved booster → %s", json_path)

        # 3. Feature schema
        schema_path = model_dir / "feature_schema.json"
        schema = {
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "target_mapping": {str(k): v for k, v in TARGET_MAPPING.items()},
            "params": {k: str(v) if isinstance(v, (Path, type)) else v
                       for k, v in self.params.items()},
            "fit_date": self._fit_date,
        }
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        logger.info("Saved schema → %s", schema_path)

        return {
            "pkl": str(pkl_path),
            "json": str(json_path),
            "schema": str(schema_path),
        }

    @classmethod
    def load(cls, model_dir: str | Path = "models") -> "XGBoostModel":
        """Load pickled model and metadata from disk.

        Parameters
        ----------
        model_dir : str | Path
            Directory containing ``xgb_model.pkl``.

        Returns
        -------
        XGBoostModel
            Restored instance with model, calibrator, and feature names.
        """
        model_dir = Path(model_dir)
        pkl_path = model_dir / "xgb_model.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Model not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        instance = cls(params=data.get("params"))
        instance.model = data["model"]
        instance.calibrated_model = data.get("calibrated_model")
        instance.feature_names = data.get("feature_names", [])
        instance._fit_date = data.get("fit_date")

        logger.info("Loaded model from %s (%d features)", pkl_path, len(instance.feature_names))
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reliability_curve(
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> List[Dict[str, Any]]:
        """Generate reliability diagram data.

        For each class and each probability bin, compute the predicted mean
        probability versus the observed frequency.

        Returns
        -------
        list[dict]
            Each dict: ``{"class": int, "bin_center": float, "predicted": float,
            "observed": float, "count": int}``.
        """
        n_classes = y_pred_proba.shape[1]
        records: List[Dict[str, Any]] = []

        for cls in range(n_classes):
            y_binary = (y_true == cls).astype(int).values
            probs = y_pred_proba[:, cls]

            bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            for i in range(n_bins):
                in_bin = (probs >= bins[i]) & (probs < bins[i + 1])
                # Edge case: include upper boundary in the last bin
                if i == n_bins - 1:
                    in_bin |= probs == bins[i + 1]

                count = int(in_bin.sum())
                if count > 0:
                    predicted = float(probs[in_bin].mean())
                    observed = float(y_binary[in_bin].mean())
                else:
                    predicted = float(bin_centers[i])
                    observed = float("nan")

                records.append({
                    "class": int(cls),
                    "bin_center": float(bin_centers[i]),
                    "predicted": round(predicted, 4),
                    "observed": round(observed, 4) if not np.isnan(observed) else None,
                    "count": count,
                })

        return records


# ===================================================================
# CLI entry point
# ===================================================================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument(
        "--features",
        default="data/processed/feature_store.csv",
        help="Path to feature_store.csv",
    )
    parser.add_argument(
        "--split-date",
        default="2023-01-01",
        help="Temporal split threshold (train <= date < val)",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Output directory for model files",
    )
    parser.add_argument(
        "--skip-calibrate",
        action="store_true",
        help="Skip calibration step",
    )
    args = parser.parse_args()

    # Load feature store
    logger.info("Loading feature store from %s …", args.features)
    fs = pd.read_csv(args.features, parse_dates=["date"])

    # Separate features and target
    X, y = XGBoostModel.prepare_data(fs)

    # Temporal split
    X_train, y_train, X_val, y_val = XGBoostModel.temporal_split(
        X, y, fs["date"], args.split_date
    )

    # Train
    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train, X_val, y_val)

    # Calibrate
    if not args.skip_calibrate:
        xgb_model.calibrate(X_val, y_val)

    # Evaluate
    metrics = xgb_model.evaluate(X_val, y_val)
    print(f"\nValidation metrics:")
    print(f"  Brier score: {metrics['brier_score']:.4f}")
    print(f"  Log loss:    {metrics['log_loss']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.2%}")

    # Feature importance
    imp = xgb_model.feature_importance(X_val)
    print(f"\nTop 5 features:")
    for _, row in imp.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Save
    paths = xgb_model.save(args.model_dir)
    print(f"\nSaved to: {paths}")
