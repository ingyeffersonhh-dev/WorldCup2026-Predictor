"""
models/elo.py — Stage 2: ELO Rating Engine

Implements the Elo rating system for international football with:

- Confederation-seeded initial ratings (CONMEBOL=1700, UEFA=1650, …)
- Dynamic K-factors per tournament type (WC=40, qualifiers=30, others=20)
- Provisional K = 2× standard for teams with < 15 matches
- Goal-margin adjustment using log-difference scaling
- Batch processing of matches with append-only history export

Based on the spec (R2.1–R2.8) and the design (C2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confederation base ratings (R2.1 / design C2)
BASE_RATINGS: Dict[str, int] = {
    "CONMEBOL": 1700,
    "UEFA": 1650,
    "CONCACAF": 1550,
    "CAF": 1500,
    "AFC": 1450,
    "OFC": 1400,
}

# K-factor by tournament category (R2.2)
K_FACTORS: Dict[str, int] = {
    "World Cup": 40,
    "Qualifiers": 30,
    "others": 10,
}

# Provisional rating rules (R2.3)
PROVISIONAL_MATCH_THRESHOLD: int = 15
PROVISIONAL_K_MULTIPLIER: int = 2

# Default rating when confederation is unknown
DEFAULT_INITIAL_RATING: float = 1500.0

# ---------------------------------------------------------------------------
# Confederation mapping (re-exported from data_pipeline for convenience)
# ---------------------------------------------------------------------------
# This is the authoritative map used by the ELO engine.  It is duplicated
# here so that elo.py is self-contained and can be imported without loading
# the data pipeline module.
CONFEDERATION_MAP: Dict[str, str] = {
    # ---- CONMEBOL ----
    "Argentina": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # ---- UEFA ----
    "Albania": "UEFA",
    "Andorra": "UEFA",
    "Armenia": "UEFA",
    "Austria": "UEFA",
    "Azerbaijan": "UEFA",
    "Belarus": "UEFA",
    "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Bulgaria": "UEFA",
    "Croatia": "UEFA",
    "Cyprus": "UEFA",
    "Czech Republic": "UEFA",
    "Czechoslovakia": "UEFA",
    "Denmark": "UEFA",
    "England": "UEFA",
    "Estonia": "UEFA",
    "Faroe Islands": "UEFA",
    "Finland": "UEFA",
    "France": "UEFA",
    "Georgia": "UEFA",
    "Germany": "UEFA",
    "East Germany": "UEFA",
    "West Germany": "UEFA",
    "Gibraltar": "UEFA",
    "Greece": "UEFA",
    "Hungary": "UEFA",
    "Iceland": "UEFA",
    "Israel": "UEFA",
    "Italy": "UEFA",
    "Kazakhstan": "UEFA",
    "Kosovo": "UEFA",
    "Latvia": "UEFA",
    "Liechtenstein": "UEFA",
    "Lithuania": "UEFA",
    "Luxembourg": "UEFA",
    "Malta": "UEFA",
    "Moldova": "UEFA",
    "Montenegro": "UEFA",
    "Netherlands": "UEFA",
    "North Macedonia": "UEFA",
    "Macedonia": "UEFA",
    "Northern Ireland": "UEFA",
    "Ireland": "UEFA",
    "Norway": "UEFA",
    "Poland": "UEFA",
    "Portugal": "UEFA",
    "Republic of Ireland": "UEFA",
    "Romania": "UEFA",
    "Russia": "UEFA",
    "Soviet Union": "UEFA",
    "CIS": "UEFA",
    "San Marino": "UEFA",
    "Scotland": "UEFA",
    "Serbia": "UEFA",
    "Serbia and Montenegro": "UEFA",
    "FR Yugoslavia": "UEFA",
    "Slovakia": "UEFA",
    "Slovenia": "UEFA",
    "Spain": "UEFA",
    "Sweden": "UEFA",
    "Switzerland": "UEFA",
    "Turkey": "UEFA",
    "Ukraine": "UEFA",
    "Wales": "UEFA",
    "Yugoslavia": "UEFA",
    "Saarland": "UEFA",
    "Kingdom of Yugoslavia": "UEFA",
    "Bohemia": "UEFA",
    "Bohemia and Moravia": "UEFA",
    # ---- CONCACAF ----
    "Anguilla": "CONCACAF",
    "Antigua and Barbuda": "CONCACAF",
    "Aruba": "CONCACAF",
    "Bahamas": "CONCACAF",
    "Barbados": "CONCACAF",
    "Belize": "CONCACAF",
    "Bermuda": "CONCACAF",
    "Bonaire": "CONCACAF",
    "British Virgin Islands": "CONCACAF",
    "Canada": "CONCACAF",
    "Cayman Islands": "CONCACAF",
    "Costa Rica": "CONCACAF",
    "Cuba": "CONCACAF",
    "Curaçao": "CONCACAF",
    "Netherlands Antilles": "CONCACAF",
    "Dominica": "CONCACAF",
    "Dominican Republic": "CONCACAF",
    "El Salvador": "CONCACAF",
    "French Guiana": "CONCACAF",
    "Grenada": "CONCACAF",
    "Guadeloupe": "CONCACAF",
    "Guatemala": "CONCACAF",
    "Guyana": "CONCACAF",
    "British Guiana": "CONCACAF",
    "Haiti": "CONCACAF",
    "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF",
    "Martinique": "CONCACAF",
    "Mexico": "CONCACAF",
    "Montserrat": "CONCACAF",
    "Nicaragua": "CONCACAF",
    "Panama": "CONCACAF",
    "Puerto Rico": "CONCACAF",
    "Saint Kitts and Nevis": "CONCACAF",
    "Saint Lucia": "CONCACAF",
    "Saint Martin": "CONCACAF",
    "Saint Vincent and the Grenadines": "CONCACAF",
    "Sint Maarten": "CONCACAF",
    "Suriname": "CONCACAF",
    "Dutch Guyana": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF",
    "Turks and Caicos Islands": "CONCACAF",
    "United States": "CONCACAF",
    "US Virgin Islands": "CONCACAF",
    # ---- CAF ----
    "Algeria": "CAF",
    "Angola": "CAF",
    "Benin": "CAF",
    "Dahomey": "CAF",
    "Botswana": "CAF",
    "Burkina Faso": "CAF",
    "Upper Volta": "CAF",
    "Burundi": "CAF",
    "Cameroon": "CAF",
    "Cape Verde": "CAF",
    "Central African Republic": "CAF",
    "Chad": "CAF",
    "Comoros": "CAF",
    "Congo": "CAF",
    "DR Congo": "CAF",
    "Zaire": "CAF",
    "Congo-Léopoldville": "CAF",
    "Congo-Kinshasa": "CAF",
    "Djibouti": "CAF",
    "Egypt": "CAF",
    "United Arab Republic": "CAF",
    "Equatorial Guinea": "CAF",
    "Eritrea": "CAF",
    "Eswatini": "CAF",
    "Swaziland": "CAF",
    "Ethiopia": "CAF",
    "Gabon": "CAF",
    "Gambia": "CAF",
    "Ghana": "CAF",
    "Gold Coast": "CAF",
    "Guinea": "CAF",
    "Guinea-Bissau": "CAF",
    "Portuguese Guinea": "CAF",
    "Ivory Coast": "CAF",
    "Kenya": "CAF",
    "Lesotho": "CAF",
    "Liberia": "CAF",
    "Libya": "CAF",
    "Madagascar": "CAF",
    "Malawi": "CAF",
    "Nyasaland": "CAF",
    "Mali": "CAF",
    "Mauritania": "CAF",
    "Mauritius": "CAF",
    "Morocco": "CAF",
    "Mozambique": "CAF",
    "Namibia": "CAF",
    "Niger": "CAF",
    "Nigeria": "CAF",
    "Rwanda": "CAF",
    "São Tomé and Príncipe": "CAF",
    "Senegal": "CAF",
    "Seychelles": "CAF",
    "Sierra Leone": "CAF",
    "Somalia": "CAF",
    "South Africa": "CAF",
    "South Sudan": "CAF",
    "Sudan": "CAF",
    "Tanzania": "CAF",
    "Tanganyika": "CAF",
    "Togo": "CAF",
    "Tunisia": "CAF",
    "Uganda": "CAF",
    "Zambia": "CAF",
    "Northern Rhodesia": "CAF",
    "Zimbabwe": "CAF",
    "Southern Rhodesia": "CAF",
    "Zanzibar": "CAF",
    # ---- AFC ----
    "Afghanistan": "AFC",
    "Australia": "AFC",
    "Bahrain": "AFC",
    "Bangladesh": "AFC",
    "Bhutan": "AFC",
    "Brunei": "AFC",
    "Cambodia": "AFC",
    "China": "AFC",
    "Chinese Taipei": "AFC",
    "East Timor": "AFC",
    "Guam": "AFC",
    "Hong Kong": "AFC",
    "India": "AFC",
    "Indonesia": "AFC",
    "Dutch East Indies": "AFC",
    "Iran": "AFC",
    "Iraq": "AFC",
    "Japan": "AFC",
    "Jordan": "AFC",
    "Kuwait": "AFC",
    "Kyrgyzstan": "AFC",
    "Laos": "AFC",
    "Lebanon": "AFC",
    "Macau": "AFC",
    "Malaysia": "AFC",
    "Malaya": "AFC",
    "Maldives": "AFC",
    "Mongolia": "AFC",
    "Myanmar": "AFC",
    "Burma": "AFC",
    "Nepal": "AFC",
    "North Korea": "AFC",
    "Northern Mariana Islands": "AFC",
    "Oman": "AFC",
    "Pakistan": "AFC",
    "Palestine": "AFC",
    "Philippines": "AFC",
    "Qatar": "AFC",
    "Saudi Arabia": "AFC",
    "Singapore": "AFC",
    "South Korea": "AFC",
    "Sri Lanka": "AFC",
    "Ceylon": "AFC",
    "Syria": "AFC",
    "Tajikistan": "AFC",
    "Thailand": "AFC",
    "Timor-Leste": "AFC",
    "Turkmenistan": "AFC",
    "United Arab Emirates": "AFC",
    "Uzbekistan": "AFC",
    "Vietnam": "AFC",
    "South Vietnam": "AFC",
    "North Vietnam": "AFC",
    "Yemen": "AFC",
    "South Yemen": "AFC",
    # ---- OFC ----
    "American Samoa": "OFC",
    "Cook Islands": "OFC",
    "Fiji": "OFC",
    "Kiribati": "OFC",
    "New Caledonia": "OFC",
    "New Zealand": "OFC",
    "Papua New Guinea": "OFC",
    "Samoa": "OFC",
    "Western Samoa": "OFC",
    "Solomon Islands": "OFC",
    "Tahiti": "OFC",
    "Tonga": "OFC",
    "Tuvalu": "OFC",
    "Vanuatu": "OFC",
    "New Hebrides": "OFC",
}


# ===================================================================
# EloEngine
# ===================================================================
class EloEngine:
    """ELO rating engine for international football matches.

    Maintains an in-memory state of ratings and match counts per team,
    processes matches sequentially (sorted by date), and produces an
    append-only history CSV.

    Parameters
    ----------
    base_ratings : dict[str, int] | None
        Override for confederation base ratings.
    conf_map : dict[str, str] | None
        Override for team → confederation mapping.
    initial_rating : float
        Default rating for teams not in the confederation map.
    """

    def __init__(
        self,
        base_ratings: Optional[Dict[str, int]] = None,
        conf_map: Optional[Dict[str, str]] = None,
        initial_rating: float = DEFAULT_INITIAL_RATING,
    ) -> None:
        self.base_ratings: Dict[str, int] = base_ratings or BASE_RATINGS.copy()
        self.conf_map: Dict[str, str] = conf_map or CONFEDERATION_MAP.copy()
        self.initial_rating: float = initial_rating

        # Mutable internal state
        self.ratings: Dict[str, float] = {}  # team → current Elo
        self.match_counts: Dict[str, int] = {}  # team → matches played

        # Accumulated history rows (list of dicts)
        self.history_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_rating(self, team: str) -> float:
        """Return the current rating for *team*, initialising if necessary."""
        if team not in self.ratings:
            self.ratings[team] = self._initial_rating(team)
            self.match_counts[team] = 0
        return self.ratings[team]

    def get_match_count(self, team: str) -> int:
        """Return how many matches *team* has processed so far."""
        return self.match_counts.get(team, 0)

    # ------------------------------------------------------------------
    # Core ELO formulas (R2.4, R2.5, R2.6)
    # ------------------------------------------------------------------

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Expected score for team A against team B (R2.4).

        ``E = 1 / (1 + 10^((Elo_B - Elo_A) / 400))``
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def goal_margin_mult(
        goals_a: int, goals_b: int, elo_diff: float
    ) -> float:
        """Goal-margin multiplier (R2.6).

        ``margin_mult = ln(|gd| + 1) × (2.2 / (elo_diff × 0.001 + 2.2))``

        where ``gd = goals_a - goals_b``.

        Notes
        -----
        The multiplier is applied to the score ``S`` for non-draw matches
        only.  For draws it is exactly 1 (no adjustment).  The numerator
        ``2.2`` and denominator ``elo_diff × 0.001 + 2.2`` ensure that the
        multiplier decreases as the rating gap increases — a big win against
        a much weaker team is discounted.
        """
        goal_diff = abs(goals_a - goals_b)
        if goal_diff == 0:
            return 1.0  # draw — no margin adjustment
        return np.log(goal_diff + 1.0) * (2.2 / (elo_diff * 0.001 + 2.2))

    def k_factor(self, tournament_type: str, team_matches: int) -> float:
        """Determine the K-factor for a given match (R2.2, R2.3).

        Parameters
        ----------
        tournament_type : str
            Tournament name from the match row.
        team_matches : int
            Number of matches the team has already played under ELO tracking.

        Returns
        -------
        float
            K value (may be a float if provisional multiplier is applied).
        """
        type_lower = tournament_type.lower()

        if "world cup" in type_lower and "qualification" not in type_lower:
            base_k = float(K_FACTORS["World Cup"])
        elif (
            "qualification" in type_lower
            or "qualifier" in type_lower
            or "qual" in type_lower
        ):
            base_k = float(K_FACTORS["Qualifiers"])
        else:
            base_k = float(K_FACTORS["others"])

        # Provisional K (R2.3)
        if team_matches < PROVISIONAL_MATCH_THRESHOLD:
            base_k *= PROVISIONAL_K_MULTIPLIER

        return base_k

    # ------------------------------------------------------------------
    # Single-match update (R2.5)
    # ------------------------------------------------------------------

    def update_ratings(
        self,
        home_goals: int,
        away_goals: int,
        home_rating: float,
        away_rating: float,
        tournament_type: str,
        home_matches: int,
        away_matches: int,
    ) -> Tuple[float, float]:
        """Compute new ratings for home and away after one match (R2.5).

        Returns
        -------
        Tuple[float, float]
            ``(home_new, away_new)``.
        """
        # Result encoding (S)
        if home_goals > away_goals:
            S_home, S_away = 1.0, 0.0
        elif home_goals == away_goals:
            S_home, S_away = 0.5, 0.5
        else:
            S_home, S_away = 0.0, 1.0

        # Expected scores
        E_home = self.expected_score(home_rating, away_rating)
        E_away = 1.0 - E_home

        # K-factors
        K_home = self.k_factor(tournament_type, home_matches)
        K_away = self.k_factor(tournament_type, away_matches)

        # Goal-margin multiplier (applies only to non-draw matches)
        elo_diff = home_rating - away_rating
        margin_mult = self.goal_margin_mult(home_goals, away_goals, elo_diff)

        # --- Update formula ---
        # new = old + K × (S × margin_mult - E)
        home_new = home_rating + K_home * (S_home * margin_mult - E_home)
        away_new = away_rating + K_away * (S_away * margin_mult - E_away)

        return home_new, away_new

    # ------------------------------------------------------------------
    # Batch processing (R2.7)
    # ------------------------------------------------------------------

    def process_matches(
        self,
        matches_df: pd.DataFrame,
        match_id_col: str = "match_id",
        home_col: str = "home_team",
        away_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        tournament_col: str = "tournament_type",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Process all matches sequentially, maintaining ELO state.

        Teams are seeded with confederation-based initial ratings on first
        appearance.  Match history is accumulated in ``self.history_rows``.

        Parameters
        ----------
        matches_df : pd.DataFrame
            Clean match data (should contain at least 1 row).
        match_id_col : str
            Column with unique match identifier.
        home_col : str
            Column with home team name.
        away_col : str
            Column with away team name.
        home_goals_col : str
            Column with home goals.
        away_goals_col : str
            Column with away goals.
        tournament_col : str
            Column with tournament type/name.
        date_col : str
            Column with match date (used for sorting).

        Returns
        -------
        pd.DataFrame
            ``elo_history`` with columns:
            ``match_id, team, elo_pre, elo_post, tournament_type``.
        """
        df = matches_df.sort_values(date_col).reset_index(drop=True)

        # Initialise state for every team that appears
        all_teams = set(df[home_col].unique()) | set(df[away_col].unique())
        for team in all_teams:
            if team not in self.ratings:
                self.ratings[team] = self._initial_rating(team)
                self.match_counts[team] = 0

        rows: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            home = row[home_col]
            away = row[away_col]
            tourn = row[tournament_col]

            home_rating_pre = self.ratings[home]
            away_rating_pre = self.ratings[away]

            home_new, away_new = self.update_ratings(
                home_goals=int(row[home_goals_col]),
                away_goals=int(row[away_goals_col]),
                home_rating=home_rating_pre,
                away_rating=away_rating_pre,
                tournament_type=tourn,
                home_matches=self.match_counts[home],
                away_matches=self.match_counts[away],
            )

            # Persist updated state
            self.ratings[home] = home_new
            self.ratings[away] = away_new
            self.match_counts[home] += 1
            self.match_counts[away] += 1

            mid = int(row[match_id_col])

            # Record history for both teams (R2.7)
            rows.append(
                {
                    "match_id": mid,
                    "team": home,
                    "elo_pre": round(home_rating_pre, 2),
                    "elo_post": round(home_new, 2),
                    "tournament_type": tourn,
                }
            )
            rows.append(
                {
                    "match_id": mid,
                    "team": away,
                    "elo_pre": round(away_rating_pre, 2),
                    "elo_post": round(away_new, 2),
                    "tournament_type": tourn,
                }
            )

        history = pd.DataFrame(rows)
        self.history_rows.extend(rows)
        return history

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_history(
        self,
        path: str | Path = "data/processed/elo_history.csv",
        history_df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Save ELO history to CSV.

        If *history_df* is ``None``, the most recent call to
        ``process_matches`` return value is used.  If the instance has
        accumulated rows across calls, pass ``self.to_history_df()``.

        Returns
        -------
        Path
            Output file path.
        """
        if history_df is not None:
            df = history_df
        elif self.history_rows:
            df = pd.DataFrame(self.history_rows)
        else:
            raise ValueError(
                "No history to export — call process_matches() first "
                "or pass a history_df."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Exported %d ELO history rows → %s", len(df), path)
        return path

    # ------------------------------------------------------------------
    # State introspection
    # ------------------------------------------------------------------

    def to_history_df(self) -> pd.DataFrame:
        """Return accumulated history as a DataFrame."""
        return pd.DataFrame(self.history_rows)

    def rating_table(self) -> pd.DataFrame:
        """Return current ratings and match counts for all tracked teams."""
        teams = sorted(self.ratings)
        return pd.DataFrame(
            {
                "team": teams,
                "rating": [round(self.ratings[t], 2) for t in teams],
                "matches": [self.match_counts.get(t, 0) for t in teams],
                "confederation": [
                    self.conf_map.get(t, "unknown") for t in teams
                ],
            }
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Validation helpers (R2.8 / Task 1.5)
    # ------------------------------------------------------------------

    def validate_ratings(
        self, known_ratings: Dict[str, Dict[str, float]]
    ) -> bool:
        """Check whether computed ratings match expected values.

        Parameters
        ----------
        known_ratings : dict
            ``{team: {"elo": <expected rating>, "matches": <expected count>}}``

        Returns
        -------
        bool
            ``True`` if all teams match within a ±0.5 tolerance.
        """
        for team, expected in known_ratings.items():
            actual_rating = self.ratings.get(team)
            actual_matches = self.match_counts.get(team)
            if actual_rating is None:
                logger.error("Team '%s' not found in ratings", team)
                return False
            if abs(actual_rating - expected["elo"]) > 0.5:
                logger.error(
                    "Rating mismatch for '%s': expected %.2f, got %.2f",
                    team,
                    expected["elo"],
                    actual_rating,
                )
                return False
            if actual_matches != expected.get("matches"):
                logger.error(
                    "Match count mismatch for '%s': expected %d, got %d",
                    team,
                    expected.get("matches"),
                    actual_matches,
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_rating(self, team: str) -> float:
        """Determine the initial rating for a team (R2.1).

        Uses confederation seed if known, otherwise returns the default
        ``initial_rating`` (1500).
        """
        conf = self.conf_map.get(team)
        if conf is not None and conf in self.base_ratings:
            return float(self.base_ratings[conf])
        logger.debug("Team '%s' has no confederation mapping — using default %.0f",
                      team, self.initial_rating)
        return self.initial_rating
