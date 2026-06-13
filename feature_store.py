"""
feature_store.py — Stage 3: Feature Engineering

Builds the feature store from clean matches + ELO history, computing:

- ELO difference (elo_home - elo_away)
- Rolling form features (avg goals for/against, 5 and 10 match windows)
- Head-to-head average goal differential (last 5 meetings)
- Home advantage indicator
- Rest days since each team's last match (capped at 30)
- Implied probabilities from bookmaker odds (overround removed)
- Target variable (1=home win, 0=draw, 2=away win)

Orchestrated by FeatureStore.build() and exported as feature_store.csv.

Based on spec R3.1-R3.8 and design C3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columns in the final feature store (data contract from spec)
# ---------------------------------------------------------------------------
FEATURE_STORE_COLUMNS = [
    "match_id",
    "date",
    "home_team",
    "away_team",
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
    "target",
]

# Default fill values for NaN features
NAN_FILL_DEFAULTS = {
    "form_home_5f": 0.0,
    "form_home_5a": 0.0,
    "form_away_5f": 0.0,
    "form_away_5a": 0.0,
    "form_home_10f": 0.0,
    "form_home_10a": 0.0,
    "form_away_10f": 0.0,
    "form_away_10a": 0.0,
    "h2h_avg_diff": 0.0,
    "implied_home": 1.0 / 3.0,
    "implied_draw": 1.0 / 3.0,
    "implied_away": 1.0 / 3.0,
}

# Tournaments where the final stage is hosted in a single country,
# giving some context advantage even if the specific match venue is neutral.
CONTINENTAL_CUP_PATTERNS = [
    "uefa euro",
    "copa am",
    "african cup of nations",
    "afc asian cup",
    "gold cup",
    "oceania nations cup",
    "confederations cup",
]


# ===================================================================
# FeatureStore
# ===================================================================
class FeatureStore:
    """Feature engineering pipeline for the mundial-predictor.

    Parameters
    ----------
    raw_dir : str | Path
        Directory containing raw data (including ``odds/`` subdirectory).
    """

    def __init__(self, raw_dir: str | Path = "data/raw") -> None:
        self.raw_dir = Path(raw_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        matches_df: pd.DataFrame,
        elo_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Orchestrate all feature computations sequentially.

        Parameters
        ----------
        matches_df : pd.DataFrame
            Clean match data (from ``clean_matches.csv``).
        elo_df : pd.DataFrame
            ELO history (from ``elo_history.csv``).

        Returns
        -------
        pd.DataFrame
            Complete feature store with all columns and no NaN.
        """
        # Ensure dates are parsed and sorted chronologically
        df = matches_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        logger.info("FeatureStore.build: starting with %d matches", len(df))

        # Step 1 — ELO difference
        df = self._elo_diff(df, elo_df)
        logger.info("  elo_diff ✓")

        # Step 2 — Rolling form features (5 and 10 match windows)
        df = self._rolling_goals(df)
        logger.info("  rolling_goals ✓")

        # Step 3 — Head-to-head average goal differential
        df = self._head_to_head(df)
        logger.info("  head_to_head ✓")

        # Step 4 — Home advantage indicator
        df = self._home_advantage(df)
        logger.info("  home_advantage ✓")

        # Step 5 — Rest days since each team's last match
        df = self._rest_days(df)
        logger.info("  rest_days ✓")

        # Step 6 — Odds integration (load + merge all available years)
        all_odds = []
        for year in (2014, 2018, 2022, 2026):
            odds = self._load_odds_from_football_data(year)
            if odds is not None:
                all_odds.append(odds)

        if all_odds:
            combined_odds = pd.concat(all_odds, ignore_index=True)
            df = self._merge_odds(df, combined_odds)
        else:
            # No odds files found — fill with uniform probabilities
            df["implied_home"] = np.nan
            df["implied_draw"] = np.nan
            df["implied_away"] = np.nan
        logger.info("  odds ✓ (%d files loaded)", len(all_odds))

        # Step 7 — Target variable (1=home, 0=draw, 2=away)
        df = self._add_target(df)
        logger.info("  target ✓")

        # Step 8 — Clean NaN values
        df = self._clean_nan(df)
        logger.info("  NaN cleaned ✓")

        # Select and order columns per data contract
        available = [c for c in FEATURE_STORE_COLUMNS if c in df.columns]
        df = df[available]

        logger.info(
            "FeatureStore.build complete: %d rows × %d cols, NaN=%d",
            len(df),
            len(df.columns),
            df.isna().sum().sum(),
        )
        return df

    @staticmethod
    def export_feature_store(df: pd.DataFrame, path: str | Path) -> Path:
        """Save the feature store to CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Feature store DataFrame.
        path : str | Path
            Output path.

        Returns
        -------
        Path
            The output path that was written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Exported feature store (%d rows) → %s", len(df), path)
        return path

    @staticmethod
    def remove_overround(odds: list[float]) -> list[float]:
        """Normalise implied probabilities so they sum to 1.

        Parameters
        ----------
        odds : list[float]
            Decimal odds ``[home, draw, away]``.

        Returns
        -------
        list[float]
            Implied probabilities summing to 1.
        """
        probs = [1.0 / o for o in odds]
        total = sum(probs)
        if total <= 0:
            return [1.0 / 3.0] * 3
        return [p / total for p in probs]

    # ------------------------------------------------------------------
    # Internal feature methods
    # ------------------------------------------------------------------

    @staticmethod
    def _elo_diff(df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
        """Compute ``elo_diff = elo_home_pre - elo_away_pre``.

        Merges pre-match ELO ratings from ``elo_history`` onto the match
        DataFrame using ``(match_id, team)``.
        """
        # Home team pre-match ELO
        home_elo = elo_df[["match_id", "team", "elo_pre"]].copy()
        home_elo = home_elo.rename(
            columns={"team": "home_team", "elo_pre": "elo_home"}
        )
        df = df.merge(home_elo, on=["match_id", "home_team"], how="left")

        # Away team pre-match ELO
        away_elo = elo_df[["match_id", "team", "elo_pre"]].copy()
        away_elo = away_elo.rename(
            columns={"team": "away_team", "elo_pre": "elo_away"}
        )
        df = df.merge(away_elo, on=["match_id", "away_team"], how="left")

        df["elo_diff"] = df["elo_home"] - df["elo_away"]

        # Drop intermediate columns
        df = df.drop(columns=["elo_home", "elo_away"], errors="ignore")
        return df

    @staticmethod
    def _rolling_goals(df: pd.DataFrame) -> pd.DataFrame:
        """Rolling average goals for/against per team (5 and 10 matches).

        Produces columns ``form_{home,away}_{5,10}{f,a}``.  The current
        match's goals are excluded via ``shift(1)`` so features only use
        historical data (no look-ahead bias).
        """
        # Build per-team game records (one row per team per match)
        home_view = df[["match_id", "date", "home_team", "home_goals", "away_goals"]].copy()
        home_view.columns = ["match_id", "date", "team", "gf", "ga"]

        away_view = df[["match_id", "date", "away_team", "away_goals", "home_goals"]].copy()
        away_view.columns = ["match_id", "date", "team", "gf", "ga"]

        team_df = pd.concat([home_view, away_view], ignore_index=True)
        team_df = team_df.sort_values(["team", "date"]).reset_index(drop=True)

        # Compute rolling averages per team
        #   shift(1): exclude current match from the window
        #   min_periods=1: use whatever is available (first match → NaN)
        for col, window in [("gf", 5), ("ga", 5), ("gf", 10), ("ga", 10)]:
            col_name = f"{col}_{window}"
            team_df[col_name] = (
                team_df.groupby("team")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

        # Merge home form back
        home_form = team_df.rename(
            columns={
                "team": "home_team",
                "gf_5": "form_home_5f",
                "ga_5": "form_home_5a",
                "gf_10": "form_home_10f",
                "ga_10": "form_home_10a",
            }
        )
        home_cols = [
            "match_id", "home_team",
            "form_home_5f", "form_home_5a", "form_home_10f", "form_home_10a",
        ]
        df = df.merge(
            home_form[home_cols], on=["match_id", "home_team"], how="left"
        )

        # Merge away form back
        away_form = team_df.rename(
            columns={
                "team": "away_team",
                "gf_5": "form_away_5f",
                "ga_5": "form_away_5a",
                "gf_10": "form_away_10f",
                "ga_10": "form_away_10a",
            }
        )
        away_cols = [
            "match_id", "away_team",
            "form_away_5f", "form_away_5a", "form_away_10f", "form_away_10a",
        ]
        df = df.merge(
            away_form[away_cols], on=["match_id", "away_team"], how="left"
        )

        return df

    @staticmethod
    def _head_to_head(df: pd.DataFrame) -> pd.DataFrame:
        """Average goal differential in the last 5 head-to-head matches.

        Computed from the current home team's perspective.  Uses the last
        5 matches between the same two teams (regardless of which was
        designated home/away in prior meetings).
        """
        # Build H2H records with a canonical pair key (sorted alphabetically)
        h2h = df[["match_id", "date", "home_team", "away_team", "home_goals", "away_goals"]].copy()

        # Goal differential from the home team's perspective
        h2h["goal_diff"] = h2h["home_goals"] - h2h["away_goals"]

        # Pair key — always (min_team, max_team) so both directions match
        h2h["pair"] = h2h.apply(
            lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
        )

        h2h = h2h.sort_values(["pair", "date"]).reset_index(drop=True)

        # Rolling average of goal_diff per pair (last 5, excluding current)
        h2h["h2h_avg_diff"] = (
            h2h.groupby("pair")["goal_diff"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

        # Merge back on match_id
        df = df.merge(h2h[["match_id", "h2h_avg_diff"]], on="match_id", how="left")
        return df

    @staticmethod
    def _home_advantage(df: pd.DataFrame) -> pd.DataFrame:
        """Determine home advantage factor.

        1.0 — true home match (``neutral_venue == 0``)
        0.5 — continental tournament final-stage match between non-host teams
        0.0 — neutral venue
        """
        # Start with the neutral_venue column
        if "neutral_venue" not in df.columns:
            df["home_advantage"] = 1.0
            return df

        # Detect continental cup final tournaments
        is_continental_final = df["tournament_type"].str.contains(
            "|".join(CONTINENTAL_CUP_PATTERNS), case=False, na=False
        )

        # Assign home_advantage
        #   neutral_venue=0 → 1.0 (true home)
        #   neutral_venue=1 AND continental cup → 0.5
        #   neutral_venue=1 AND NOT continental cup → 0.0
        conditions = [
            df["neutral_venue"] == 0,
            (df["neutral_venue"] == 1) & is_continental_final,
        ]
        choices = [1.0, 0.5]
        df["home_advantage"] = np.select(conditions, choices, default=0.0)

        return df

    @staticmethod
    def _rest_days(df: pd.DataFrame) -> pd.DataFrame:
        """Days since each team's last match, capped at 30.

        Processes matches in chronological order.  Teams appearing for the
        first time receive the cap value (30 days).
        """
        # Ensure chronological order (already sorted by build())
        df = df.reset_index(drop=True)

        last_match: dict[str, pd.Timestamp] = {}
        rest_home: list[int] = []
        rest_away: list[int] = []

        for _, row in df.iterrows():
            home: str = row["home_team"]
            away: str = row["away_team"]
            date: pd.Timestamp = row["date"]

            # Home team rest days
            if home in last_match:
                days = (date - last_match[home]).days
                rest_home.append(min(days, 30))
            else:
                rest_home.append(30)

            # Away team rest days
            if away in last_match:
                days = (date - last_match[away]).days
                rest_away.append(min(days, 30))
            else:
                rest_away.append(30)

            # Update last match dates
            last_match[home] = date
            last_match[away] = date

        df["rest_days_home"] = rest_home
        df["rest_days_away"] = rest_away
        return df

    # ------------------------------------------------------------------
    # Odds integration
    # ------------------------------------------------------------------

    def _load_odds_from_football_data(self, year: int) -> Optional[pd.DataFrame]:
        """Load a football-data.co.uk CSV for a given World Cup year.

        Expected path: ``{raw_dir}/odds/odds_{year}.csv``.

        Returns ``None`` (with a warning) if the file does not exist.
        """
        path = self.raw_dir / "odds" / f"odds_{year}.csv"
        if not path.exists():
            logger.warning(
                "Odds file not found: %s (implied probs from ELO only)", path
            )
            return None

        logger.info("Loading odds: %s", path)
        df = pd.read_csv(path)

        # Normalise column names
        df.columns = [c.strip() for c in df.columns]

        # Parse date — football-data.co.uk uses DD/MM/YY or DD/MM/YYYY
        if "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        else:
            logger.warning("Odds file %s has no 'Date' column — skipping", path)
            return None

        # Normalise team names
        df.rename(
            columns={
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
            },
            inplace=True,
        )

        # Keep only the columns we need: Bet365 odds (most reliable across years)
        # If B365 columns missing, fall back to any available bookmaker.
        odds_cols = [c for c in ["B365H", "B365D", "B365A"] if c in df.columns]
        if not odds_cols:
            # Try first available bookmaker columns
            for prefix in ["B365", "BW", "IW", "LB", "WH", "SJ", "VC", "SB"]:
                cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
                if all(c in df.columns for c in cols):
                    odds_cols = cols
                    break

        if not odds_cols:
            logger.warning(
                "No bookmaker odds columns found in %s — skipping", path
            )
            return None

        result = df[["date", "home_team", "away_team"] + odds_cols].copy()
        result.columns = ["date", "home_team", "away_team", "odds_h", "odds_d", "odds_a"]

        # Drop rows with missing odds
        result = result.dropna(subset=["odds_h", "odds_d", "odds_a"])
        # Remove implausible odds (< 1.01)
        result = result[
            (result["odds_h"] >= 1.01)
            & (result["odds_d"] >= 1.01)
            & (result["odds_a"] >= 1.01)
        ]

        logger.info("  → %d matches with odds from %s", len(result), path)
        return result

    @staticmethod
    def _merge_odds(df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Merge odds onto the match DataFrame and compute implied probs.

        If the feature store already has implied probability columns from a
        previous merge, they are overwritten only when new odds match (so the
        first merge wins for each match).  This lets us load multiple odds
        files and keep the earliest successful merge.

        Timezone tolerance: odds with a ±1 day date offset are also matched
        to handle matches played in late timezones (Americas) where the
        football-data.co.uk date differs from clean_matches by one day.
        """
        # Only merge odds for matches that don't already have odds
        if "implied_home" not in df.columns:
            df["implied_home"] = np.nan
            df["implied_draw"] = np.nan
            df["implied_away"] = np.nan

        unmatched = df[df["implied_home"].isna()]
        if unmatched.empty:
            return df

        # Expand odds with ±1 day to handle timezone-related date shifts.
        # Order: exact date first, then day before, then day after.
        # drop_duplicates(keep='first') prefers the exact date match.
        odds_expanded = pd.concat(
            [
                odds_df.assign(date_merge=odds_df["date"]),
                odds_df.assign(date_merge=odds_df["date"] - pd.Timedelta(days=1)),
                odds_df.assign(date_merge=odds_df["date"] + pd.Timedelta(days=1)),
            ],
            ignore_index=True,
        )
        odds_expanded = odds_expanded.drop_duplicates(
            subset=["date_merge", "home_team", "away_team"], keep="first"
        )

        merged = unmatched.merge(
            odds_expanded,
            left_on=["date", "home_team", "away_team"],
            right_on=["date_merge", "home_team", "away_team"],
            how="left",
            suffixes=("", "_odds"),
        )

        # Compute implied probabilities from matched odds
        has_odds = merged["odds_h"].notna()
        if has_odds.any():
            probs = merged.loc[has_odds].apply(
                lambda r: FeatureStore.remove_overround(
                    [r["odds_h"], r["odds_d"], r["odds_a"]]
                ),
                axis=1,
                result_type="expand",
            )
            merged.loc[has_odds, "implied_home"] = probs.iloc[:, 0].values
            merged.loc[has_odds, "implied_draw"] = probs.iloc[:, 1].values
            merged.loc[has_odds, "implied_away"] = probs.iloc[:, 2].values

        # Write back
        df.update(merged[["implied_home", "implied_draw", "implied_away"]])

        matched_count = has_odds.sum()
        if matched_count:
            logger.info("  → merged odds for %d matches (exact + tz-tolerant)", matched_count)

        return df

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------

    @staticmethod
    def _add_target(df: pd.DataFrame) -> pd.DataFrame:
        """Encode match result as a target variable.

        1 — home win (home_goals > away_goals)
        0 — draw      (home_goals == away_goals)
        2 — away win  (home_goals < away_goals)
        """
        conditions = [
            df["home_goals"] > df["away_goals"],
            df["home_goals"] == df["away_goals"],
        ]
        choices = [1, 0]
        df["target"] = np.select(conditions, choices, default=2).astype(int)
        return df

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_nan(df: pd.DataFrame) -> pd.DataFrame:
        """Fill remaining NaN values with sensible defaults.

        Form features: fill with 0 (first match for a new team).
        H2H: fill with 0 (first meeting between pair).
        Odds: fill with uniform 1/3.
        """
        for col, default in NAN_FILL_DEFAULTS.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

        # Any remaining NaN in numeric columns → 0
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in NAN_FILL_DEFAULTS:
                df[col] = df[col].fillna(0)

        return df


# ===================================================================
# CLI entry point
# ===================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the feature store")
    parser.add_argument(
        "--matches",
        default="data/processed/clean_matches.csv",
        help="Path to clean_matches.csv",
    )
    parser.add_argument(
        "--elo",
        default="data/processed/elo_history.csv",
        help="Path to elo_history.csv",
    )
    parser.add_argument(
        "--output",
        default="data/processed/feature_store.csv",
        help="Output path for feature_store.csv",
    )
    args = parser.parse_args()

    logger.info("Loading matches from %s …", args.matches)
    matches = pd.read_csv(args.matches, parse_dates=["date"])

    logger.info("Loading ELO history from %s …", args.elo)
    elo = pd.read_csv(args.elo)

    store = FeatureStore()
    features = store.build(matches, elo)
    store.export_feature_store(features, args.output)
