"""
dashboard.py — Stage 6: Streamlit Dashboard

Three-page Streamlit dashboard for the mundial-predictor:

- Page 1: Upcoming matches with 1X2 probabilities, Poisson score matrix
- Page 2: Champion ranking from Monte Carlo simulation
- Page 3: Backtesting results and model card

Based on spec R8.1-R8.6 and design C9.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
BACKTESTING_RESULTS = PROJECT_ROOT / "backtesting" / "results"
FIXTURE_PATH = DATA_RAW / "fixture_2026.csv"
ODDS_2026_PATH = DATA_RAW / "odds_2026.csv"
FEATURE_STORE_PATH = DATA_PROCESSED / "feature_store.csv"
CHAMPION_PROBS_PATH = DATA_PROCESSED / "champion_probs.csv"
MATCH_PROBS_PATH = DATA_PROCESSED / "match_probs.csv"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

# Feature columns (mirrors xgboost_model.py)
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

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mundial Predictor 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Background and Text */
    .stApp {
        background-color: #0D0F14;
        color: #E8EAF0;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #C9A84C !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        background-color: #161B27;
        border-radius: 8px;
        border: 1px solid #2A3347;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #E8EAF0 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #5E6C84 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161B27;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #C9A84C !important;
        color: #C9A84C !important;
    }
</style>
""", unsafe_allow_html=True)



# ===================================================================
# Data loading (cached)
# ===================================================================

@st.cache_data(show_spinner="Cargando fixture…")
def load_fixture() -> pd.DataFrame:
    """Load the 2026 fixture CSV."""
    if not FIXTURE_PATH.exists():
        st.error(f"Fixture file not found: {FIXTURE_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(FIXTURE_PATH, parse_dates=["date"])
    return df


@st.cache_data(show_spinner="Cargando probabilidades de campeón…")
def load_champion_probs() -> pd.DataFrame:
    """Load champion probabilities from MC simulation."""
    if not CHAMPION_PROBS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CHAMPION_PROBS_PATH)
    # Normalise column names
    rename = {}
    if "pct" in df.columns and "champion_pct" not in df.columns:
        rename["pct"] = "champion_pct"
    if "champion_count" not in df.columns and "count" in df.columns:
        rename["count"] = "champion_count"
    if rename:
        df = df.rename(columns=rename)
    return df


@st.cache_data(show_spinner="Cargando probabilidades de partidos…")
def load_match_probs() -> pd.DataFrame:
    """Load match probabilities from MC simulation."""
    if not MATCH_PROBS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MATCH_PROBS_PATH)


@st.cache_data(show_spinner="Cargando modelos…")
def load_models() -> Tuple[Any, Any, Optional[Dict[str, Any]]]:
    """Load the trained XGBoost and Poisson models.

    Returns
    -------
    xgb_model : XGBoostModel or None
    poisson : DixonColesPoisson or None
    feature_schema : dict or None
    """
    xgb_model = None
    poisson = None
    feature_schema = None

    # Load XGBoost
    try:
        from models.xgboost_model import XGBoostModel
        xgb_model = XGBoostModel.load(MODELS_DIR)
    except Exception as exc:
        st.warning(f"Could not load XGBoost model: {exc}")

    # Load Poisson
    try:
        from models.poisson_model import DixonColesPoisson
        poisson = DixonColesPoisson.load(MODELS_DIR / "poisson_params.json")
    except Exception as exc:
        st.warning(f"Could not load Poisson model: {exc}")

    # Load feature schema
    if FEATURE_SCHEMA_PATH.exists():
        try:
            with open(FEATURE_SCHEMA_PATH) as f:
                feature_schema = json.load(f)
        except Exception:
            pass

    return xgb_model, poisson, feature_schema


@st.cache_data(show_spinner="Cargando resultados de backtesting…")
def load_backtesting_results() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load backtesting metrics and predictions.

    Returns
    -------
    metrics_df : pd.DataFrame
        Aggregated metrics per year.
    predictions_df : pd.DataFrame
        Combined per-match predictions.
    calibration_data : dict
        Calibration curve data (if available).
    """
    metrics_path = BACKTESTING_RESULTS / "metrics_summary.csv"
    metrics_df = pd.DataFrame()
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)

    # Load individual year predictions
    pred_dfs = []
    for year in [2014, 2018, 2022]:
        p = BACKTESTING_RESULTS / f"predictions_{year}.csv"
        if p.exists() and p.stat().st_size > 10:
            pred_dfs.append(pd.read_csv(p, parse_dates=["date"]))

    predictions_df = pd.concat(pred_dfs, ignore_index=True) if pred_dfs else pd.DataFrame()

    # Load calibration data from individual metrics files
    calibration_data: Dict[str, Any] = {}
    for year in [2014, 2018, 2022]:
        p = BACKTESTING_RESULTS / f"metrics_{year}.csv"
        if p.exists():
            try:
                mdf = pd.read_csv(p)
                if "calibration_data" in mdf.columns:
                    raw = mdf.iloc[0]["calibration_data"]
                    if isinstance(raw, str):
                        calibration_data[str(year)] = json.loads(raw)
            except Exception:
                pass

    return metrics_df, predictions_df, calibration_data


@st.cache_data(show_spinner="Cargando feature store…")
def load_feature_store() -> pd.DataFrame:
    """Load the feature store for model predictions."""
    if not FEATURE_STORE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_STORE_PATH, parse_dates=["date"])


@st.cache_data(show_spinner="Cargando cuotas de TheOddsAPI…")
def load_odds_2026() -> pd.DataFrame:
    """Load the 2026 World Cup odds from TheOddsAPI fetch."""
    if not ODDS_2026_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(ODDS_2026_PATH)


def _build_odds_map(odds_df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Build a lookup dict from odds DataFrame.

    Returns
    -------
    dict
        ``{(home_team, away_team): {"implied_home": ..., "implied_draw": ..., "implied_away": ...}}``
    """
    odds_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    if odds_df.empty:
        return odds_map
    for _, row in odds_df.iterrows():
        key = (row["home_team"], row["away_team"])
        odds_map[key] = {
            "implied_home": float(row["implied_home"]),
            "implied_draw": float(row["implied_draw"]),
            "implied_away": float(row["implied_away"]),
        }
    return odds_map


# ===================================================================
# Prediction helpers
# ===================================================================

def build_feature_vector(
    home_team: str,
    away_team: str,
    xgb_model: Any,
    feature_store: pd.DataFrame,
    elo_ratings: Optional[Dict[str, float]] = None,
    odds_map: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Build a feature vector for a single fixture match.

    Uses the last known feature values from the feature store for each team,
    combined with current ELO ratings if available.  When real bookmaker odds
    are available in *odds_map*, they are used for the implied_* features
    instead of the uniform 1/3 fallback.

    Parameters
    ----------
    home_team : str
    away_team : str
    xgb_model : XGBoostModel
        Used to get feature names.
    feature_store : pd.DataFrame
        Historical feature data.
    elo_ratings : dict | None
        Current ELO ratings {team: rating}.
    odds_map : dict | None
        Real bookmaker odds lookup ``{(home, away): {implied_home, ...}}``.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with feature columns.
    """
    # Default feature values
    features = {
        "elo_diff": 0.0,
        "elo_diff_sq": 0.0,
        "form_home_3f": 0.0,
        "form_home_3a": 0.0,
        "form_away_3f": 0.0,
        "form_away_3a": 0.0,
        "form_home_5f": 0.0,
        "form_home_5a": 0.0,
        "form_away_5f": 0.0,
        "form_away_5a": 0.0,
        "form_home_10f": 0.0,
        "form_home_10a": 0.0,
        "form_away_10f": 0.0,
        "form_away_10a": 0.0,
        "h2h_avg_diff": 0.0,
        "home_advantage": 0.0,  # neutral venue
        "rest_days_home": 7.0,
        "rest_days_away": 7.0,
        "streak_home": 0.0,
        "streak_away": 0.0,
        "tournament_importance": 1.0,  # World Cup
        "has_real_odds": 0.0,
        "implied_home": 1.0 / 3.0,
        "implied_draw": 1.0 / 3.0,
        "implied_away": 1.0 / 3.0,
    }

    # Use real bookmaker odds when available
    if odds_map is not None:
        key = (home_team, away_team)
        if key in odds_map:
            real_odds = odds_map[key]
            features["implied_home"] = real_odds["implied_home"]
            features["implied_draw"] = real_odds["implied_draw"]
            features["implied_away"] = real_odds["implied_away"]
            features["has_real_odds"] = 1.0

    # Get last known form for home team
    home_matches = feature_store[
        (feature_store["home_team"] == home_team) |
        (feature_store["away_team"] == home_team)
    ]
    if not home_matches.empty:
        last = home_matches.iloc[-1]
        if last.get("home_team") == home_team:
            features["form_home_3f"] = float(last.get("form_home_3f", 0))
            features["form_home_3a"] = float(last.get("form_home_3a", 0))
            features["form_home_5f"] = float(last.get("form_home_5f", 0))
            features["form_home_5a"] = float(last.get("form_home_5a", 0))
            features["form_home_10f"] = float(last.get("form_home_10f", 0))
            features["form_home_10a"] = float(last.get("form_home_10a", 0))
            features["streak_home"] = float(last.get("streak_home", 0))
        else:
            features["form_home_3f"] = float(last.get("form_away_3f", 0))
            features["form_home_3a"] = float(last.get("form_away_3a", 0))
            features["form_home_5f"] = float(last.get("form_away_5f", 0))
            features["form_home_5a"] = float(last.get("form_away_5a", 0))
            features["form_home_10f"] = float(last.get("form_away_10f", 0))
            features["form_home_10a"] = float(last.get("form_away_10a", 0))
            features["streak_home"] = float(last.get("streak_away", 0))

    # Get last known form for away team
    away_matches = feature_store[
        (feature_store["home_team"] == away_team) |
        (feature_store["away_team"] == away_team)
    ]
    if not away_matches.empty:
        last = away_matches.iloc[-1]
        if last.get("home_team") == away_team:
            features["form_away_3f"] = float(last.get("form_home_3f", 0))
            features["form_away_3a"] = float(last.get("form_home_3a", 0))
            features["form_away_5f"] = float(last.get("form_home_5f", 0))
            features["form_away_5a"] = float(last.get("form_home_5a", 0))
            features["form_away_10f"] = float(last.get("form_home_10f", 0))
            features["form_away_10a"] = float(last.get("form_home_10a", 0))
            features["streak_away"] = float(last.get("streak_home", 0))
        else:
            features["form_away_3f"] = float(last.get("form_away_3f", 0))
            features["form_away_3a"] = float(last.get("form_away_3a", 0))
            features["form_away_5f"] = float(last.get("form_away_5f", 0))
            features["form_away_5a"] = float(last.get("form_away_5a", 0))
            features["form_away_10f"] = float(last.get("form_away_10f", 0))
            features["form_away_10a"] = float(last.get("form_away_10a", 0))
            features["streak_away"] = float(last.get("streak_away", 0))

    # ELO ratings
    if elo_ratings is not None:
        home_elo = elo_ratings.get(home_team, 1500.0)
        away_elo = elo_ratings.get(away_team, 1500.0)
        features["elo_diff"] = home_elo - away_elo
        features["elo_diff_sq"] = (home_elo - away_elo) ** 2

    # H2H
    h2h_matches = feature_store[
        ((feature_store["home_team"] == home_team) & (feature_store["away_team"] == away_team)) |
        ((feature_store["home_team"] == away_team) & (feature_store["away_team"] == home_team))
    ]
    if not h2h_matches.empty:
        h2h_last5 = h2h_matches.tail(5)
        diffs = []
        for _, m in h2h_last5.iterrows():
            if m["home_team"] == home_team:
                diffs.append(m.get("h2h_avg_diff", 0))
            else:
                diffs.append(-m.get("h2h_avg_diff", 0))
        features["h2h_avg_diff"] = float(np.mean(diffs)) if diffs else 0.0

    # Build DataFrame with correct column order
    feature_names = xgb_model.feature_names if xgb_model and hasattr(xgb_model, 'feature_names') else FEATURE_COLUMNS
    row = {col: features.get(col, 0.0) for col in feature_names}
    return pd.DataFrame([row])


def compute_predictions(
    fixture_df: pd.DataFrame,
    xgb_model: Any,
    poisson: Any,
    feature_store: pd.DataFrame,
    elo_ratings: Optional[Dict[str, float]] = None,
    odds_map: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Compute 1X2 and Poisson predictions for all fixture matches.

    Parameters
    ----------
    fixture_df : pd.DataFrame
        Fixture data with home_team, away_team.
    xgb_model : XGBoostModel
        Trained XGBoost model.
    poisson : DixonColesPoisson
        Trained Poisson model.
    feature_store : pd.DataFrame
        Historical feature store.
    elo_ratings : dict | None
        Current ELO ratings.
    odds_map : dict | None
        Real bookmaker odds lookup ``{(home, away): {implied_home, ...}}``.

    Returns
    -------
    pd.DataFrame
        Fixture with added prediction columns.
    """
    if xgb_model is None and poisson is None:
        return fixture_df.copy()

    results = fixture_df.copy()
    p_home_list: List[float] = []
    p_draw_list: List[float] = []
    p_away_list: List[float] = []
    score_matrices: List[Optional[np.ndarray]] = []
    implied_home_list: List[float] = []
    implied_draw_list: List[float] = []
    implied_away_list: List[float] = []

    for _, row in fixture_df.iterrows():
        features = build_feature_vector(
            row["home_team"], row["away_team"],
            xgb_model, feature_store, elo_ratings, odds_map,
        )

        # Store the implied probs that were used as features
        implied_home_list.append(float(features["implied_home"].iloc[0]))
        implied_draw_list.append(float(features["implied_draw"].iloc[0]))
        implied_away_list.append(float(features["implied_away"].iloc[0]))

        ph, pd_, pa = 1 / 3, 1 / 3, 1 / 3  # default uniform
        score_matrix = None

        # XGBoost prediction
        if xgb_model is not None:
            try:
                probs = xgb_model.predict_proba(features)[0]
                # probs order: [P(draw), P(home), P(away)]
                pd_ = float(probs[0])
                ph = float(probs[1])
                pa = float(probs[2])
            except Exception as exc:
                logger.warning("XGBoost prediction failed for %s vs %s: %s",
                               row["home_team"], row["away_team"], exc)

        # Poisson score matrix & prediction
        if poisson is not None:
            try:
                lambda_h, lambda_a = poisson.predict_lambdas(features)
                rho = float(poisson.params_[-1]) if poisson.params_ is not None else 0.0
                lambda_h_val = float(lambda_h.item() if hasattr(lambda_h, "item") else lambda_h)
                lambda_a_val = float(lambda_a.item() if hasattr(lambda_a, "item") else lambda_a)
                score_matrix = poisson.exact_score_prob(lambda_h_val, lambda_a_val, rho)

                # Get Poisson 1X2 marginal probabilities
                p_home_pois, p_draw_pois, p_away_pois = poisson.match_1x2_from_score_matrix(score_matrix)

                # Ensemble (using 50/50 weighting like in Monte Carlo)
                ensemble_alpha = 0.5
                ph = ensemble_alpha * ph + (1.0 - ensemble_alpha) * p_home_pois
                pd_ = ensemble_alpha * pd_ + (1.0 - ensemble_alpha) * p_draw_pois
                pa = ensemble_alpha * pa + (1.0 - ensemble_alpha) * p_away_pois

                # Adjust the score matrix so it perfectly matches the ensemble 1X2 probabilities
                if score_matrix is not None and p_home_pois > 0 and p_draw_pois > 0 and p_away_pois > 0:
                    adj_matrix = np.zeros_like(score_matrix)
                    max_g = score_matrix.shape[0]
                    for i in range(max_g):  # local goals
                        for j in range(max_g):  # away goals
                            if i > j:
                                adj_matrix[i, j] = score_matrix[i, j] * (ph / p_home_pois)
                            elif i == j:
                                adj_matrix[i, j] = score_matrix[i, j] * (pd_ / p_draw_pois)
                            else:
                                adj_matrix[i, j] = score_matrix[i, j] * (pa / p_away_pois)
                    
                    # Normalize to ensure sum is exactly 1.0
                    adj_matrix = adj_matrix / np.sum(adj_matrix)
                    score_matrix = adj_matrix

            except Exception as exc:
                logger.warning("Poisson prediction failed for %s vs %s: %s",
                               row["home_team"], row["away_team"], exc)

        p_home_list.append(ph)
        p_draw_list.append(pd_)
        p_away_list.append(pa)
        score_matrices.append(score_matrix)

    results["p_home"] = p_home_list
    results["p_draw"] = p_draw_list
    results["p_away"] = p_away_list
    results["predicted_winner"] = results.apply(
        lambda r: (
            r["home_team"] if r["p_home"] >= r["p_draw"] and r["p_home"] >= r["p_away"]
            else r["away_team"] if r["p_away"] >= r["p_draw"] and r["p_away"] >= r["p_home"]
            else "draw"
        ),
        axis=1,
    )
    results["confidence"] = results.apply(
        lambda r: max(r["p_home"], r["p_draw"], r["p_away"]),
        axis=1,
    )
    results["score_matrix"] = score_matrices

    # Real bookmaker implied probabilities
    results["implied_home"] = implied_home_list
    results["implied_draw"] = implied_draw_list
    results["implied_away"] = implied_away_list

    # Edge vs real bookmaker odds
    results["edge_home"] = results["p_home"] - results["implied_home"]
    results["edge_draw"] = results["p_draw"] - results["implied_draw"]
    results["edge_away"] = results["p_away"] - results["implied_away"]

    # Kelly criterion (quarter-Kelly for conservative sizing)
    def _kelly(p: float, q: float) -> float:
        if q <= 0 or q >= 1 or p <= q:
            return 0.0
        return (p - q) * q / (1 - q) * 0.25

    results["kelly_home"] = results.apply(lambda r: _kelly(r["p_home"], r["implied_home"]), axis=1)
    results["kelly_draw"] = results.apply(lambda r: _kelly(r["p_draw"], r["implied_draw"]), axis=1)
    results["kelly_away"] = results.apply(lambda r: _kelly(r["p_away"], r["implied_away"]), axis=1)

    def _best_bet(row: pd.Series) -> str:
        bets = {
            row["home_team"]: row["kelly_home"],
            "Empate": row["kelly_draw"],
            row["away_team"]: row["kelly_away"],
        }
        best = max(bets, key=bets.get)
        stake = bets[best]
        return best if stake > 0 else "Sin apuesta"

    results["best_bet"] = results.apply(_best_bet, axis=1)
    results["best_kelly"] = results[["kelly_home", "kelly_draw", "kelly_away"]].max(axis=1)

    return results

def get_elo_ratings(feature_store: pd.DataFrame = None) -> Dict[str, float]:
    """Extract the latest ELO ratings from elo_history.csv."""
    path = Path("data/processed/elo_history.csv")
    if not path.exists():
        return {}
    
    try:
        df = pd.read_csv(path)
        # Get the last row for each team and extract elo_post
        return df.groupby("team")["elo_post"].last().to_dict()
    except Exception as e:
        logger.warning(f"Failed to load ELO ratings: {e}")
        return {}


# ===================================================================
# Page renderers
# ===================================================================

def upcoming_matches_view(
    fixture_df: pd.DataFrame,
    xgb_model: Any,
    poisson: Any,
    feature_store: pd.DataFrame,
    elo_ratings: Optional[Dict[str, float]],
) -> None:
    """Page 1: Upcoming matches with predictions (R8.1)."""
    st.header("📊 Próximos Partidos — Mundial 2026")

    if fixture_df.empty:
        st.warning("No hay fixture disponible. Colocá fixture_2026.csv en data/raw/.")
        return

    if xgb_model is None:
        st.warning("Modelo XGBoost no cargado. Las predicciones no estarán disponibles.")
        return

    with st.spinner("Calculando predicciones …"):
        odds_df = load_odds_2026()
        odds_map = _build_odds_map(odds_df)
        preds = compute_predictions(
            fixture_df, xgb_model, poisson, feature_store, elo_ratings, odds_map
        )

    # ── Group filter ────────────────────────────────────────────────
    groups = sorted(preds["group"].unique())
    selected_group = st.selectbox("Filtrar por grupo:", ["Todos"] + groups)

    if selected_group != "Todos":
        display_df = preds[preds["group"] == selected_group].copy()
    else:
        display_df = preds.copy()

    # Sort chronologically by date, group, and match_id
    display_df = display_df.sort_values(by=["date", "group", "match_id"]).copy()

    # ── Tabs Setup ──────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📋 Tabla General", "🔢 Matriz Poisson"])

    with tab1:
        st.subheader("Predicciones 1X2")

        table_cols = [
            "date", "group", "home_team", "away_team",
            "p_home", "p_draw", "p_away", "predicted_winner", "confidence",
        ]
        table_df = display_df[table_cols].copy()

        # Format probabilities as percentages
        for col in ["p_home", "p_draw", "p_away"]:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.1%}")
        table_df["confidence"] = table_df["confidence"].apply(lambda x: f"{x:.1%}")
        table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")

        # Rename for display
        table_df = table_df.rename(columns={
            "date": "Fecha",
            "group": "Grupo",
            "home_team": "Local",
            "away_team": "Visitante",
            "p_home": "P(1)",
            "p_draw": "P(X)",
            "p_away": "P(2)",
            "predicted_winner": "Pronóstico",
            "confidence": "Confianza",
        })

        # Dynamically highlight the winner probability using pandas style
        def highlight_winner(row):
            styles = [''] * len(row)
            winner = row['Pronóstico']
            if winner != 'draw':
                if winner == row['Local']:
                    styles[table_df.columns.get_loc('P(1)')] = 'background-color: rgba(46, 204, 113, 0.2)'
                elif winner == row['Visitante']:
                    styles[table_df.columns.get_loc('P(2)')] = 'background-color: rgba(46, 204, 113, 0.2)'
            return styles

        st.dataframe(
            table_df.style.apply(highlight_winner, axis=1),
            width='stretch',
            hide_index=True,
        )

        # ── Edge vs real bookmaker odds ──────────────────────────────
        has_real_odds = "implied_home" in display_df.columns and display_df["implied_home"].notna().any()
        bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10, max_value=100000, value=100, step=10)

        if has_real_odds:
            st.subheader("📈 Value Bets — Kelly Criterion")
            st.caption(
                "Kelly fraction = (p − q) × q / (1 − q) × 25%. "
                "Valores positivos indican valor. El stake sugerido usa quarter-Kelly para sizing conservador."
            )

            # Build per-match table: for each match show the best single bet
            kelly_rows = []
            for _, r in display_df.iterrows():
                home = r["home_team"]
                away = r["away_team"]
                best = r["best_bet"]
                best_k = r["best_kelly"]
                if best == home:
                    p_model, p_casa = r["p_home"], r["implied_home"]
                    edge = r["edge_home"]
                elif best == "Empate":
                    p_model, p_casa = r["p_draw"], r["implied_draw"]
                    edge = r["edge_draw"]
                elif best == away:
                    p_model, p_casa = r["p_away"], r["implied_away"]
                    edge = r["edge_away"]
                else:
                    p_model, p_casa, edge = 0.0, 0.0, 0.0

                kelly_rows.append({
                    "Local": home,
                    "Visitante": away,
                    "P(Modelo)": p_model,
                    "P(Casa)": p_casa,
                    "Edge": edge,
                    "Kelly%": best_k,
                    "$ Sugerido": best_k * bankroll,
                    "Mejor Apuesta": best,
                })

            edge_display = pd.DataFrame(kelly_rows)

        else:
            st.subheader("📈 Ventaja vs Línea Base Uniforme")
            st.caption("Sin cuotas reales disponibles. Edge = probabilidad del modelo − 33.3%.")

            kelly_rows = []
            for _, r in display_df.iterrows():
                home = r["home_team"]
                away = r["away_team"]
                best = r["best_bet"]
                best_k = r["best_kelly"]
                if best == home:
                    p_model = r["p_home"]
                    edge = r["edge_home"]
                elif best == "Empate":
                    p_model = r["p_draw"]
                    edge = r["edge_draw"]
                elif best == away:
                    p_model = r["p_away"]
                    edge = r["edge_away"]
                else:
                    p_model, edge = 0.0, 0.0

                kelly_rows.append({
                    "Local": home,
                    "Visitante": away,
                    "P(Modelo)": p_model,
                    "Edge": edge,
                    "Kelly%": best_k,
                    "$ Sugerido": best_k * bankroll,
                    "Mejor Apuesta": best,
                })

            edge_display = pd.DataFrame(kelly_rows)

        # Style: green for strong, yellow for small, gray for none
        def highlight_kelly(val):
            if isinstance(val, (int, float)):
                if val > 0.02:
                    return 'background-color: rgba(46, 204, 113, 0.25);'
                elif val > 0.005:
                    return 'background-color: rgba(241, 196, 15, 0.2);'
                elif val > 0:
                    return 'background-color: rgba(149, 165, 166, 0.15);'
            return ''

        def highlight_edges(val):
            if isinstance(val, float):
                if val > 0.05:
                    return 'color: #2ecc71;'
                elif val < -0.05:
                    return 'color: #e74c3c;'
            return ''

        # Format percentages and currency
        fmt = {}
        for c in edge_display.columns:
            if c in ("P(Modelo)", "P(Casa)", "Edge", "Kelly%"):
                fmt[c] = "{:.3f}".format
            elif c == "$ Sugerido":
                fmt[c] = "${:,.2f}".format

        kelly_cols = [c for c in edge_display.columns if c == "Kelly%"]
        edge_cols = [c for c in edge_display.columns if c == "Edge"]

        styled = edge_display.style.format(fmt)
        if kelly_cols:
            styled = styled.map(highlight_kelly, subset=kelly_cols)
        if edge_cols:
            styled = styled.map(highlight_edges, subset=edge_cols)

        st.dataframe(styled, width='stretch', hide_index=True)

    with tab2:
        # ── Poisson score matrix heatmap ───────────────────────────────
        st.subheader("🔢 Matriz de Puntajes Poisson")
        st.caption("Seleccioná un partido para ver el mapa de calor de probabilidad de puntaje exacto.")

        # Match selector
        match_labels = display_df.apply(
            lambda r: f"{r['home_team']} vs {r['away_team']}", axis=1
        ).tolist()
        selected_match_idx = st.selectbox(
            "Partido:", range(len(match_labels)),
            format_func=lambda i: match_labels[i] if i < len(match_labels) else "",
        )

        if selected_match_idx < len(display_df):
            selected = display_df.iloc[selected_match_idx]
            sm = selected.get("score_matrix")

            if sm is not None and isinstance(sm, np.ndarray):
                import plotly.express as px

                max_g = sm.shape[0] - 1
                labels = [str(i) for i in range(max_g + 1)]

                fig = px.imshow(
                    sm,
                    x=labels,
                    y=labels,
                    labels={"x": "Goles Visitante", "y": "Goles Local", "color": "Probabilidad"},
                    title=f"{selected['home_team']} vs {selected['away_team']} — Probabilidad de Puntaje",
                    color_continuous_scale="Viridis",
                    text_auto=".1%",
                    aspect="equal",
                )
                fig.update_layout(
                    width=600,
                    height=600,
                    font=dict(size=12),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, width='stretch')

                # Marginal 1X2 from Poisson
                from models.poisson_model import DixonColesPoisson
                p_h, p_d, p_a = DixonColesPoisson.match_1x2_from_score_matrix(sm)
                col1, col2, col3 = st.columns(3)
                col1.metric(f"P({selected['home_team']})", f"{p_h:.1%}")
                col2.metric("P(Empate)", f"{p_d:.1%}")
                col3.metric(f"P({selected['away_team']})", f"{p_a:.1%}")

                # ── Recomendación y Análisis del Partido ──
                st.markdown("---")
                
                edge_home = selected.get("edge_home", 0.0)
                edge_away = selected.get("edge_away", 0.0)
                edge_draw = selected.get("edge_draw", 0.0)
                p_home = selected.get("p_home", 1/3)
                p_draw = selected.get("p_draw", 1/3)
                p_away = selected.get("p_away", 1/3)
                imp_home = selected.get("implied_home", 1/3)
                imp_draw = selected.get("implied_draw", 1/3)
                imp_away = selected.get("implied_away", 1/3)
                
                local = selected['home_team']
                visitante = selected['away_team']
                
                recommendations = []
                # Value bet detection — use real odds threshold if available
                has_real = selected.get("implied_home", 0) != 1/3
                edge_threshold = 0.05 if has_real else 0.15
                if edge_home > edge_threshold:
                    rec_label = "Value Bet Local" if has_real else "Ventaja Local"
                    recommendations.append(f"🟢 **{rec_label}:** **{local}** (edge +{edge_home:.1%})")
                elif edge_away > edge_threshold:
                    rec_label = "Value Bet Visitante" if has_real else "Ventaja Visitante"
                    recommendations.append(f"🟢 **{rec_label}:** **{visitante}** (edge +{edge_away:.1%})")
                
                # Close match detection
                max_p = max(p_home, p_draw, p_away)
                if max_p < 0.40:
                    recommendations.append("🟡 **Alto Riesgo:** Sin favorito claro")
                elif p_draw > 0.32:
                    recommendations.append(f"🔵 **Empate Probable** ({p_draw:.1%})")
                    
                if not recommendations:
                    recommendations.append("⚪ **Partido Equilibrado:** Sin ventajas claras vs casas de apuestas")
                    
                st.info(" | ".join(recommendations))
            else:
                st.info("Matriz de puntajes no disponible para este partido. Ejecutá el modelo Poisson primero.")


def champion_ranking_view() -> None:
    """Page 2: Champion ranking and group-stage probabilities (R8.2)."""
    st.header("🏆 Ranking de Campeones — Mundial 2026")

    # ── Botón de actualización ──────────────────────────────────────
    col_title, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("🔄 Actualizar Datos", type="primary", use_container_width=True):
            status_placeholder = st.empty()
            progress_bar = st.progress(0, text="Iniciando...")

            # Step 1: Fetch from Wikipedia
            progress_bar.progress(10, text="Paso 1/2: Buscando resultados en Wikipedia...")
            try:
                r1 = subprocess.run(
                    [sys.executable, "scripts/fetch_wikipedia_results.py"],
                    capture_output=True, text=True, timeout=60,
                    cwd=PROJECT_ROOT,
                )
                if r1.returncode == 0:
                    for line in r1.stdout.strip().split("\n"):
                        if line.strip():
                            status_placeholder.success(line)
                else:
                    status_placeholder.warning(f"Wikipedia fetch: {r1.stderr[:300]}")
            except subprocess.TimeoutExpired:
                status_placeholder.warning("Wikipedia fetch timed out (continuing anyway)")
            except Exception as e:
                status_placeholder.warning(f"Error en fetch: {e}")

            # Step 2: Run Monte Carlo
            progress_bar.progress(50, text="Paso 2/2: Simulando Monte Carlo (1000 sims)...")
            try:
                r2 = subprocess.run(
                    [sys.executable, "monte_carlo.py", "--n-sims", "1000", "--closest-only"],
                    capture_output=True, text=True, timeout=300,
                    cwd=PROJECT_ROOT,
                )
                if r2.returncode == 0:
                    for line in r2.stdout.strip().split("\n"):
                        if "Top 10" in line or "Converged" in line or line.strip().startswith("  "):
                            status_placeholder.info(line.strip())
                else:
                    status_placeholder.warning(f"MC sim: {r2.stderr[:300]}")
            except subprocess.TimeoutExpired:
                status_placeholder.warning("Simulación timed out (intentá con menos sims)")
            except Exception as e:
                status_placeholder.warning(f"Error en simulación: {e}")

            progress_bar.progress(100, text="✅ Listo. Refrescando datos...")
            st.cache_data.clear()
            st.rerun()

    champion_df = load_champion_probs()

    if champion_df.empty:
        st.warning(
            "No se encontraron probabilidades de campeón. "
            "Ejecutá `monte_carlo.py` primero para generar champion_probs.csv."
        )
        st.info(
            "Archivo esperado: `data/processed/champion_probs.csv`\n\n"
            "Comando: `python monte_carlo.py --n-sims 10000`"
        )
        return

    # Determine probability column
    prob_col = "champion_pct" if "champion_pct" in champion_df.columns else "pct"
    count_col = "champion_count" if "champion_count" in champion_df.columns else "count"
    runner_up_col = "runner_up_pct" if "runner_up_pct" in champion_df.columns else None

    # Top N filter
    top_n = st.slider("Mostrar top N equipos:", 5, min(50, len(champion_df)), 15)

    top_df = champion_df.head(top_n).copy()

    # ── Bar chart ──────────────────────────────────────────────────
    st.subheader(f"Top {top_n} — Probabilidades de Campeón")

    import plotly.express as px

    fig = px.bar(
        top_df,
        x="team",
        y=prob_col,
        title=f"Probabilidad de Campeón (n={int(champion_df[count_col].sum())} simulaciones)",
        labels={"team": "Equipo", prob_col: "Probabilidad (%)"},
        color=prob_col,
        color_continuous_scale="Viridis",
        text_auto=".1f",
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width='stretch')

    # ── Data table ─────────────────────────────────────────────────
    st.subheader("Ranking Completo")

    display_cols = ["team", prob_col, count_col]
    if runner_up_col and runner_up_col in champion_df.columns:
        display_cols.append(runner_up_col)

    rank_df = champion_df[display_cols].copy()
    rank_df[prob_col] = rank_df[prob_col].apply(lambda x: f"{x:.2f}%")
    if runner_up_col and runner_up_col in rank_df.columns:
        rank_df[runner_up_col] = rank_df[runner_up_col].apply(lambda x: f"{x:.2f}%")

    rank_df = rank_df.rename(columns={
        "team": "Equipo",
        prob_col: "Prob. Campeón",
        count_col: "Victorias",
    })
    if runner_up_col and runner_up_col in rank_df.columns:
        rank_df = rank_df.rename(columns={runner_up_col: "% Subcampeón"})

    rank_df.index = range(1, len(rank_df) + 1)
    rank_df.index.name = "Ranking"

    st.dataframe(rank_df, width='stretch')

    # ── Group-stage probabilities ──────────────────────────────────
    st.subheader("📋 Probabilidades — Fase de Grupos")

    match_df = load_match_probs()
    if not match_df.empty:
        groups = sorted(match_df["group"].unique())
        selected_g = st.selectbox("Filtrar por grupo (Fase de grupos):", ["Todos"] + groups)
        if selected_g != "Todos":
            df_g = match_df[match_df["group"] == selected_g].copy()
        else:
            df_g = match_df.copy()
            
        df_g["p_home"] = df_g["p_home"].apply(lambda x: f"{x:.1%}")
        df_g["p_draw"] = df_g["p_draw"].apply(lambda x: f"{x:.1%}")
        df_g["p_away"] = df_g["p_away"].apply(lambda x: f"{x:.1%}")
        
        df_g = df_g.rename(columns={
            "group": "Grupo",
            "home_team": "Local",
            "away_team": "Visitante",
            "p_home": "P(1)",
            "p_draw": "P(X)",
            "p_away": "P(2)",
            "avg_home_goals": "Goles Local",
            "avg_away_goals": "Goles Visitante"
        })
        st.dataframe(df_g[["Grupo", "Local", "Visitante", "P(1)", "P(X)", "P(2)", "Goles Local", "Goles Visitante"]], width='stretch', hide_index=True)
    else:
        st.info("No hay probabilidades de fase de grupos disponibles. Ejecutá la simulación.")


def backtesting_view(feature_schema: Optional[Dict[str, Any]]) -> None:
    """Page 3: Backtesting results and model card (R8.3, R8.4, R8.5)."""
    st.header("📈 Resultados de Backtesting")

    st.markdown("""
    El **Backtesting** es un "examen de historia" para nuestro predictor. Evaluamos el rendimiento del modelo simulando cómo habría pronosticado los Mundiales pasados (2014, 2018, 2022) entrenándolo **exclusivamente con datos anteriores** a cada torneo. Esto nos da una medida realista de la confiabilidad de los pronósticos.
    """)

    metrics_df, predictions_df, calibration_data = load_backtesting_results()

    if not metrics_df.empty:
        avg_acc = metrics_df["accuracy"].mean() if "accuracy" in metrics_df.columns else 0.0
        avg_rps = metrics_df["rps"].mean() if "rps" in metrics_df.columns else 0.0
        avg_roi = metrics_df["roi"].mean() if "roi" in metrics_df.columns else 0.0

        st.subheader("📊 Resumen de Rendimiento General")
        
        # Qualitative ratings
        acc_rating = "⭐ Excelente" if avg_acc > 0.50 else "⭐ Fuerte" if avg_acc > 0.40 else "⭐ Regular"
        rps_rating = "🎯 Muy Preciso (Error Bajo)" if avg_rps < 0.22 else "🎯 Normal"
        roi_rating = "🟢 Rentable (Retorno Positivo)" if avg_roi > 0 else "🔴 No Rentable (Estrategia Kelly pasiva)"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precisión Histórica (1X2)", f"{avg_acc:.1%}", help="Porcentaje de partidos donde el modelo acertó el ganador o empate.")
            st.caption(f"Calificación: **{acc_rating}** (el azar es 33.3%)")
        with col2:
            st.metric("Margen de Error (RPS)", f"{avg_rps:.3f}", help="Indica qué tan cerca estuvieron las probabilidades estimadas de los resultados reales. Menor es mejor.")
            st.caption(f"Calificación: **{rps_rating}**")
        with col3:
            st.metric("Retorno Kelly (ROI)", f"{avg_roi:.1%}", help="Retorno de inversión promedio aplicando la fórmula de Kelly según las ventajas del modelo.")
            st.caption(f"Resultado: **{roi_rating}**")

        st.markdown("---")

        # ── 1. EXPLICACIÓN SENCILLA DE MÉTRICAS ──
        with st.expander("❓ ¿Cómo entender estas métricas en palabras simples?", expanded=False):
            st.markdown("""
            - **Precisión (1X2):** Indica el porcentaje de veces que el modelo acertó la tendencia final (ej. si predijo victoria local y el local ganó). Un valor arriba del 50% es muy alto en fútbol.
            - **RPS (Ranked Probability Score):** Mide la precisión de las probabilidades asignadas. Si el modelo le da 90% a un equipo y este pierde, recibe una penalización alta. Si le da 40% y pierde, la penalización es menor. Un RPS promedio menor a 0.220 indica que el modelo calibra muy bien sus dudas.
            - **Kelly ROI:** Es el resultado financiero simulado. Mide si seguir las probabilidades del modelo contra las casas de apuestas habría generado ganancias a largo plazo.
            """)

        # ── 2. DETALLE POR MUNDIAL (Tabla y Gráficos) ──
        with st.expander("📅 Detalle de Métricas por Mundial", expanded=True):
            display_metrics = metrics_df[[c for c in ["year", "n_matches", "brier_score", "rps", "log_loss", "accuracy", "roi"] if c in metrics_df.columns]].copy()
            display_metrics = display_metrics.rename(columns={
                "year": "Año",
                "n_matches": "Partidos",
                "brier_score": "Brier Score",
                "rps": "RPS",
                "log_loss": "Log Loss",
                "accuracy": "Precisión",
                "roi": "Kelly ROI",
            })
            # Format percentages
            if "Precisión" in display_metrics.columns:
                display_metrics["Precisión"] = display_metrics["Precisión"].apply(lambda x: f"{x:.1%}")
            if "Kelly ROI" in display_metrics.columns:
                display_metrics["Kelly ROI"] = display_metrics["Kelly ROI"].apply(lambda x: f"{x:.1%}")
                
            st.dataframe(display_metrics, width='stretch', hide_index=True)

            # Optional bar chart
            import plotly.express as px
            metric_cols = [c for c in ["rps", "accuracy", "roi"] if c in metrics_df.columns]
            if metric_cols:
                melt_df = metrics_df.melt(
                    id_vars=["year"],
                    value_vars=metric_cols,
                    var_name="Metric",
                    value_name="Value",
                )
                rename_map = {
                    "rps": "RPS (Error)",
                    "accuracy": "Precisión",
                    "roi": "Kelly ROI",
                }
                melt_df["Metric"] = melt_df["Metric"].map(rename_map).fillna(melt_df["Metric"])
                fig = px.bar(
                    melt_df,
                    x="year",
                    y="Value",
                    color="Metric",
                    barmode="group",
                    title="Comparativa de Rendimiento Histórico por Mundial",
                    labels={"year": "Año", "Value": "Valor"},
                    text_auto=".3f",
                )
                fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, width='stretch')

        # ── 3. GRÁFICOS TÉCNICOS AVANZADOS ──
        with st.expander("📈 Gráficos Avanzados: Calibración y Matriz de Confusión"):
            col_img1, col_img2 = st.columns(2)
            cal_img = BACKTESTING_RESULTS / "calibration.png"
            if cal_img.exists():
                col_img1.image(str(cal_img), caption="Curvas de calibración (cercanía a las probabilidades reales)", use_container_width=True)
            else:
                col_img1.info("Gráfico de calibración no disponible.")

            cm_img = BACKTESTING_RESULTS / "confusion_matrix.png"
            if cm_img.exists():
                col_img2.image(str(cm_img), caption="Matriz de confusión (aciertos/errores de predicción)", use_container_width=True)
            else:
                col_img2.info("Matriz de confusión no disponible.")

        # ── 4. MODEL CARD / FICHA TÉCNICA ──
        with st.expander("📇 Ficha Técnica del Modelo (Model Card)"):
            if feature_schema:
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown("**Tipo de Modelo**")
                    st.write("Classifier XGBoost (multi:softprob) + Dixon-Coles Poisson")
                    st.markdown("**Fecha de Entrenamiento**")
                    st.write(feature_schema.get("fit_date", "N/A"))
                with col_c2:
                    st.markdown("**Mapeo de Clases Objetivo**")
                    st.write(str(feature_schema.get("target_mapping", {})))
                    st.markdown("**Características de ELO**")
                    st.write("Diferencia de ELO dinámico (elo_diff), Rest Days")

                st.markdown("**Lista Completa de Variables de Entrada (Features)**")
                feature_names = feature_schema.get("feature_names", [])
                if feature_names:
                    cols = st.columns(3)
                    for idx, name in enumerate(feature_names):
                        cols[idx % 3].write(f"- `{name}`")
            else:
                st.info("Schema del modelo no encontrado en `models/feature_schema.json`.")

        # ── 5. DESCARGAR DATOS ──
        with st.expander("📥 Descargar Historial de Predicciones"):
            if not predictions_df.empty:
                csv = predictions_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Descargar predicciones como CSV",
                    data=csv,
                    file_name="wc_backtest_predictions.csv",
                    mime="text/csv",
                )
    else:
        st.info("Métricas de backtesting no encontradas. Ejecutá `backtesting/evaluator.py` primero.")


def live_results_view(fixture_df: pd.DataFrame) -> None:
    st.title("🔴 Resultados en Vivo")
    st.markdown("Ingresá los resultados reales de los partidos a medida que ocurren. Esto disparará el re-entrenamiento automático del modelo.")
    
    live_path = Path("data/raw/live_results.csv")
    live_results = pd.DataFrame()
    if live_path.exists():
        try:
            live_results = pd.read_csv(live_path)
            st.success(f"Hay {len(live_results)} partidos registrados.")
            
            st.markdown("**Tabla de Resultados Guardados** (Modificá los goles en la tabla o borrá filas y apretá el botón para guardar)")
            
            # Tabla editable con altura fija para tener scroll y no ocupar toda la pantalla
            edited_df = st.data_editor(
                live_results,
                column_config={
                    "home_score": st.column_config.NumberColumn("Goles Local", min_value=0, max_value=20),
                    "away_score": st.column_config.NumberColumn("Goles Visitante", min_value=0, max_value=20),
                    "date": st.column_config.Column(disabled=True),
                    "home_team": st.column_config.Column(disabled=True),
                    "away_team": st.column_config.Column(disabled=True),
                    "tournament": st.column_config.Column(disabled=True),
                    "city": st.column_config.Column(disabled=True),
                    "country": st.column_config.Column(disabled=True),
                    "neutral": st.column_config.Column(disabled=True),
                },
                hide_index=True,
                height=250,
                num_rows="dynamic"  # Permite al usuario borrar un resultado si lo cargó mal
            )
            
            if st.button("💾 Guardar Cambios de la Tabla"):
                edited_df.to_csv(live_path, index=False)
                st.success("¡Cambios guardados con éxito!")
                st.rerun()
                
        except Exception as e:
            st.warning(f"Error cargando resultados: {e}")

    st.subheader("Cargar Nuevo Resultado")
    # Filter matches not already in live_results
    if not live_results.empty and "home_team" in live_results.columns:
        played = set(zip(live_results["home_team"], live_results["away_team"]))
        unplayed = fixture_df[~fixture_df.apply(lambda x: (x["home_team"], x["away_team"]) in played, axis=1)]
    else:
        unplayed = fixture_df

    if unplayed.empty:
        st.info("¡Todos los partidos de la fase de grupos ya tienen resultado!")
        return

    match_labels = [f"{r['home_team']} vs {r['away_team']}" for _, r in unplayed.iterrows()]
    selected_match = st.selectbox("Seleccionar Partido", match_labels)

    col1, col2 = st.columns(2)
    home_g = col1.number_input("Goles Local", min_value=0, max_value=20, value=0, step=1)
    away_g = col2.number_input("Goles Visitante", min_value=0, max_value=20, value=0, step=1)

    if st.button("Guardar Resultado"):
        idx = match_labels.index(selected_match)
        match_row = unplayed.iloc[idx]
        
        new_row = pd.DataFrame([{
            "date": match_row["date"],
            "home_team": match_row["home_team"],
            "away_team": match_row["away_team"],
            "home_score": home_g,
            "away_score": away_g,
            "tournament": "FIFA World Cup",
            "city": "Unknown",
            "country": "USA",
            "neutral": True
        }])

        df_to_save = pd.concat([live_results, new_row], ignore_index=True) if not live_results.empty else new_row
        df_to_save.to_csv(live_path, index=False)
        st.success("Resultado guardado con éxito. Actualizá la página para verlo en la tabla.")

    st.markdown("---")
    st.subheader("⚙️ Actualizar Modelos")
    st.markdown("Si ya cargaste resultados, hacé clic acá para reentrenar todo el pipeline. (Se ejecutará en segundo plano).")
    if st.button("Re-entrenar y Simular"):
        import subprocess
        st.info("Iniciando pipeline en segundo plano... Revisá tu terminal.")
        subprocess.Popen(["python", "pipeline.py", "--sims", "100", "--closest-only"])


# ===================================================================
# Main app
# ===================================================================

def main() -> None:
    """Main entry point — renders sidebar navigation and selected page."""

    # ── Sidebar navigation ─────────────────────────────────────────
    st.sidebar.title("⚽ Mundial Predictor")
    st.sidebar.markdown("### Mundial 2026")
    st.sidebar.markdown("---")

    nav_options = [
        "📊 Próximos Partidos",
        "🏆 Ranking Campeones",
        "📈 Backtesting",
        "🔴 Resultados en Vivo"
    ]

    page = st.sidebar.radio(
        "Navegación",
        nav_options,
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Fuentes de Datos**")
    st.sidebar.markdown("- Resultados internacionales (Kaggle)")
    st.sidebar.markdown("- Cuotas (Football-data.co.uk)")
    st.sidebar.markdown("- Fixture Mundial 2026")

    # Load shared data
    fixture_df = load_fixture()

    # ── Page routing ───────────────────────────────────────────────

    if page == "📊 Próximos Partidos":
        xgb_model, poisson, _ = load_models()
        feature_store = load_feature_store()
        elo_ratings = None
        if not feature_store.empty:
            with st.spinner("Calculando ratings ELO …"):
                elo_ratings = get_elo_ratings(feature_store)

        upcoming_matches_view(fixture_df, xgb_model, poisson, feature_store, elo_ratings)

    elif page == "🏆 Ranking Campeones":
        champion_ranking_view()

    elif page == "📈 Backtesting":
        _, _, feature_schema = load_models()
        backtesting_view(feature_schema)

    elif page == "🔴 Resultados en Vivo":
        live_results_view(fixture_df)


if __name__ == "__main__":
    main()
