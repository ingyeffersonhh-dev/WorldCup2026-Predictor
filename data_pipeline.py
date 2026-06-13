"""
data_pipeline.py — Stage 1: Data Ingestion & Cleaning

Downloads the Kaggle international football results dataset, audits columns,
filters to official matches (1993+), removes walkovers/abandons, deduplicates,
normalizes team names, adds match IDs, and marks neutral venues.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Official tournament name patterns — matches the Kaggle dataset's tournament
# column.  We keep only FIFA-recognised competitions and their qualifiers.
# ---------------------------------------------------------------------------
OFFICIAL_TOURNAMENT_PATTERNS: List[str] = [
    # FIFA flagship
    "FIFA World Cup",
    # Confederation cups & Nations Leagues
    "UEFA Euro",
    "Copa Am",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",  # CONCACAF Gold Cup
    "Oceania Nations Cup",
    "Confederations Cup",
    "CONCACAF Nations League",
    "UEFA Nations League",
    "CONCACAF Championship",
    "CCCF Championship",
    "NAFC Championship",
    # Qualifiers for all the above
    "qualification",
    "Qualification",
    # Non-FIFA but sanctioned / historically important
    "Olympic Games",
]

# Tournaments explicitly excluded even if they match a partial pattern above.
# (e.g. "Olympic Games" is in the list above but if we wanted to exclude
# we would add it here — we keep Olympics as they use senior national teams.)
EXCLUDED_TOURNAMENTS: List[str] = [
    "Friendly",
    "Inter-Allied Games",
    "The Other Final",
    "FIFI Wild Cup",
    "Viva World Cup",
    "ELF Cup",
    "Muratti Vase",
    "Island Games",
    "Inter Games",
    "Mundialito",
    "GaNEFo",
]

# ---------------------------------------------------------------------------
# Confederation mapping — every FIFA member plus historical / notable
# non-FIFA teams.  Used for ELO seeding.
# ---------------------------------------------------------------------------
CONFEDERATION_MAP: Dict[str, str] = {
    # ---- CONMEBOL (South America, 10 members) ----
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
    # ---- UEFA (Europe, 55 members) ----
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
    # ---- CONCACAF (North/Central America & Caribbean, 41 members) ----
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
    "United States Virgin Islands": "CONCACAF",
    "US Virgin Islands": "CONCACAF",
    # ---- CAF (Africa, 54 members) ----
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
    "Congo DR": "CAF",
    "Zaire": "CAF",
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
    "Côte d'Ivoire": "CAF",
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
    # ---- AFC (Asia, 47 members) ----
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
    "Korea DPR": "AFC",
    "Northern Mariana Islands": "AFC",
    "Oman": "AFC",
    "Pakistan": "AFC",
    "Palestine": "AFC",
    "Philippines": "AFC",
    "Qatar": "AFC",
    "Saudi Arabia": "AFC",
    "Singapore": "AFC",
    "South Korea": "AFC",
    "Korea Republic": "AFC",
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
    # ---- OFC (Oceania, 11 members) ----
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
    "French Polynesia": "OFC",
    "Tonga": "OFC",
    "Tuvalu": "OFC",
    "Vanuatu": "OFC",
    "New Hebrides": "OFC",
    # ---- Historical / defunct teams (default to 1500) ----
}

# Validate: every team in the mapping that appears in the dataset should have
# a known confederation. We fill 1500 for any unregistered team at runtime.

# Column names required by the pipeline (from R1.6 / R1.7 / clean_matches.csv)
REQUIRED_COLUMNS: List[str] = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
]


# ===================================================================
# DataPipeline
# ===================================================================
class DataPipeline:
    """Ingest, audit, filter, clean, and export match data.

    Parameters
    ----------
    raw_dir : str | Path
        Directory for raw downloaded files.
    processed_dir : str | Path
        Directory for cleaned output files.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # -- Download ----------------------------------------------------------

    def download_kaggle(
        self,
        dataset: str = "martj42/international-football-results-from-1872-to-2017",
        force: bool = False,
    ) -> str:
        """Download the Kaggle international football results dataset.

        Uses ``kagglehub`` to download the dataset.  Copies the main
        ``results.csv`` into ``self.raw_dir / kaggle_results.csv``.

        Parameters
        ----------
        dataset : str
            Kaggle dataset slug.
        force : bool
            If ``True``, re-download even if the file already exists.

        Returns
        -------
        str
            Path to the downloaded CSV.
        """
        import kagglehub

        dest = self.raw_dir / "kaggle_results.csv"
        if dest.exists() and not force:
            logger.info("File already exists: %s (use force=True to re-download)", dest)
            return str(dest)

        logger.info("Downloading dataset '%s' from Kaggle …", dataset)
        download_path = kagglehub.dataset_download(dataset)
        logger.info("Downloaded to cache: %s", download_path)

        # Locate results.csv inside the downloaded tree
        src_candidates = list(Path(download_path).rglob("results.csv"))
        if not src_candidates:
            raise FileNotFoundError(
                "results.csv not found in the downloaded dataset. "
                f"Contents: {list(Path(download_path).iterdir())}"
            )
        shutil.copy2(str(src_candidates[0]), str(dest))
        logger.info("Copied results.csv → %s (%d rows)", dest, self._count_rows(dest))
        return str(dest)

    # -- Audit ------------------------------------------------------------

    @staticmethod
    def audit_columns(df: pd.DataFrame) -> Dict[str, Any]:
        """Audit the raw DataFrame for required columns and basic stats.

        Returns a dictionary with:
        - total_rows
        - columns (full list)
        - missing_columns
        - present_columns
        - dtypes (serialised)
        - date_range (min / max)
        """
        present = [c for c in REQUIRED_COLUMNS if c in df.columns]
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

        result: Dict[str, Any] = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "present_columns": present,
            "missing_columns": missing,
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "date_range": None,
        }

        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            result["date_range"] = {
                "min": str(dates.min().date()) if pd.notna(dates.min()) else None,
                "max": str(dates.max().date()) if pd.notna(dates.max()) else None,
            }

        if missing:
            logger.warning("Missing required columns: %s", missing)
        else:
            logger.info("All required columns present ✓")

        return result

    # -- Load -------------------------------------------------------------

    def load_raw(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load the raw CSV from disk.

        Parameters
        ----------
        path : str | Path | None
            Explicit path, or ``None`` to use ``raw_dir / kaggle_results.csv``.
        """
        if path is None:
            path = self.raw_dir / "kaggle_results.csv"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Raw dataset not found: {path}. Run download_kaggle() first.")

        df = pd.read_csv(path)
        # Parse dates
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    # -- Filter official matches ------------------------------------------

    def filter_official(
        self,
        df: pd.DataFrame,
        min_date: str = "2014-01-01",
    ) -> pd.DataFrame:
        """Keep only official tournament matches on or after *min_date*.

        Steps:
        1. Filter by official tournament name patterns.
        2. Remove explicitly excluded tournaments.
        3. Drop rows where the date is before *min_date*.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame.
        min_date : str
            Earliest allowed match date (ISO format, default ``1993-01-01``).
        """
        if df.empty:
            return df

        min_ts = pd.Timestamp(min_date)

        # Build a single case-insensitive pattern
        official_pat = "|".join(OFFICIAL_TOURNAMENT_PATTERNS)
        # Remove walkovers / abandoned markers early
        exclude_pat = "|".join(EXCLUDED_TOURNAMENTS)

        mask = (
            df["tournament"].str.contains(official_pat, case=False, na=False)
            & ~df["tournament"].str.contains(exclude_pat, case=False, na=False)
            & (df["date"] >= min_ts)
        )

        filtered = df[mask].copy()
        logger.info(
            "filter_official: %d → %d rows (date >= %s, official only)",
            len(df),
            len(filtered),
            min_date,
        )
        return filtered

    # -- Clean matches ----------------------------------------------------

    @staticmethod
    def _normalize_team_names(
        df: pd.DataFrame,
        former_names_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Normalize team names by applying renames from *former_names.csv*.

        The Kaggle dataset ships a ``former_names.csv`` that maps historical
        country names to their current FIFA name.  We apply those mappings
        to both ``home_team`` and ``away_team`` columns.
        """
        if former_names_path is None:
            # Look beside the processed_dir / raw_dir
            candidates = [
                Path(p) / "former_names.csv"
                for p in ["data/raw", "data/processed"]
            ]
            for c in candidates:
                if c.exists():
                    former_names_path = c
                    break

        if former_names_path and Path(former_names_path).exists():
            renames = pd.read_csv(former_names_path)
            rename_map: Dict[str, str] = {}
            for _, row in renames.iterrows():
                rename_map[row["former"]] = row["current"]

            if rename_map:
                n_before = len(df)
                df["home_team"] = df["home_team"].replace(rename_map)
                df["away_team"] = df["away_team"].replace(rename_map)
                changed = (n_before - len(df)) + (  # approximate
                    (df["home_team"] != df["home_team"]).sum()  # noqa
                )
                logger.info(
                    "Normalized team names using %s (%d mappings)",
                    former_names_path,
                    len(rename_map),
                )

        return df

    def clean_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove incomplete, abandoned, or invalid match records.

        Operations:
        - Drop rows where ``home_score`` or ``away_score`` is NaN.
        - Remove negative scores.
        - Drop rows whose tournament field signals abandonment or walkover.
        - Deduplicate on (date, home_team, away_team).
        - Reset index.

        Parameters
        ----------
        df : pd.DataFrame
            Pre-filtered DataFrame (should have run ``filter_official`` first).
        """
        n0 = len(df)

        # Drop NaN scores
        df = df.dropna(subset=["home_score", "away_score"])
        n1 = len(df)

        # Remove negative / nonsensical scores
        df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]

        # Abandoned / walkover tournaments
        abandon_pat = r"abandoned|cancelled|walkover|wo\b|awarded|annulled"
        df = df[~df["tournament"].str.contains(abandon_pat, case=False, na=False)]

        # Deduplicate
        df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")

        df = df.reset_index(drop=True)
        logger.info("clean_matches: %d → %d rows (removed %d)", n0, len(df), n0 - len(df))
        return df

    # -- Add match IDs ----------------------------------------------------

    @staticmethod
    def add_match_ids(df: pd.DataFrame) -> pd.DataFrame:
        """Add sequential integer ``match_id`` (1-based, sorted by date)."""
        df = df.sort_values("date").reset_index(drop=True)
        df["match_id"] = range(1, len(df) + 1)
        return df

    # -- Add venue type ---------------------------------------------------

    @staticmethod
    def add_venue_type(df: pd.DataFrame) -> pd.DataFrame:
        """Add ``neutral_venue`` integer column (1 = neutral, 0 = home).

        If the original ``neutral`` column exists (bool), convert to int.
        Otherwise default to 0.
        """
        if "neutral" in df.columns:
            df["neutral_venue"] = df["neutral"].astype(int)
        else:
            df["neutral_venue"] = 0
        return df

    # -- Rename columns to match data contract ----------------------------

    @staticmethod
    def _rename_to_contract(df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to match the ``clean_matches.csv`` data contract.

        Contract columns::
            match_id, date, home_team, away_team, home_goals, away_goals,
            tournament_type, neutral_venue
        """
        rename: Dict[str, str] = {
            "home_score": "home_goals",
            "away_score": "away_goals",
            "tournament": "tournament_type",
        }
        df = df.rename(columns=rename)

        # Ensure all contract columns exist
        contract_cols = [
            "match_id", "date", "home_team", "away_team",
            "home_goals", "away_goals", "tournament_type", "neutral_venue",
        ]
        for c in contract_cols:
            if c not in df.columns:
                df[c] = None
        return df[contract_cols]

    # -- Export -----------------------------------------------------------

    def export(
        self,
        df: pd.DataFrame,
        filename: str = "clean_matches.csv",
        rename_columns: bool = True,
    ) -> Path:
        """Export a DataFrame to the processed directory as CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Data to export.
        filename : str
            Output file name.
        rename_columns : bool
            If ``True``, apply the standard contract column rename.
        """
        if rename_columns:
            df = self._rename_to_contract(df)

        path = self.processed_dir / filename
        df.to_csv(path, index=False)
        logger.info("Exported %d rows → %s", len(df), path)
        return path

    # -- Convenience: full clean pipeline ---------------------------------

    def run_pipeline(
        self,
        min_date: str = "2014-01-01",
        download: bool = True,
        former_names_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Run the full ingestion → cleaning pipeline end-to-end.

        Returns the clean DataFrame and exports it to
        ``data/processed/clean_matches.csv``.
        """
        if download:
            self.download_kaggle()

        df = self.load_raw()

        # Inject live results if present
        live_path = Path("data/raw/live_results.csv")
        if live_path.exists():
            try:
                live_df = pd.read_csv(live_path)
                df = pd.concat([df, live_df], ignore_index=True)
                logger.info(f"Loaded {len(live_df)} live results from live_results.csv.")
            except Exception as e:
                logger.warning(f"Could not load live_results.csv: {e}")

        audit = self.audit_columns(df)
        if audit["missing_columns"]:
            raise ValueError(
                f"Cannot proceed — missing columns: {audit['missing_columns']}"
            )

        df = self.filter_official(df, min_date=min_date)
        df = self._normalize_team_names(df, former_names_path=former_names_path)
        df = self.clean_matches(df)
        df = self.add_match_ids(df)
        df = self.add_venue_type(df)

        self.export(df)
        return df

    # -- Internal helpers -------------------------------------------------

    @staticmethod
    def _count_rows(path: Path) -> int:
        """Quickly count non-header lines in a CSV."""
        try:
            return sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1
        except Exception:
            return -1
