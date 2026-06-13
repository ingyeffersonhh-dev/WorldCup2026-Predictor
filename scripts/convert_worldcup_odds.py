"""Convert WorldCup2026.xlsx to per-year odds CSV files for feature_store.

Handles WC final tournaments (2014/2018/2022) and 2026 qualifiers.
"""

import pandas as pd
from pathlib import Path

XLSX_PATH = Path.home() / "AppData" / "Local" / "Temp" / "WorldCup2026.xlsx"
RAW_DIR = Path("data/raw/odds")

# Team name mappings between football-data.co.uk and clean_matches.csv
TEAM_NAME_MAP = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Central Africa": "Central African Republic",
    "Chinese Taipei": "Taiwan",
    "Curacao": "Curaçao",
    "D.R. Congo": "DR Congo",
    "Guinea Bissau": "Guinea-Bissau",
    "Ireland": "Republic of Ireland",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "Trinidad & Tobago": "Trinidad and Tobago",
}


def extract_sheet(sheet_name, year, odds_map):
    """Extract a sheet from the XLSX and save as odds_{year}.csv."""
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet_name)
    print(f"Sheet: {sheet_name} ({year}) - {len(df)} rows")

    result = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})

    # Normalize team names to match clean_matches.csv
    result["HomeTeam"] = result["HomeTeam"].replace(TEAM_NAME_MAP)
    result["AwayTeam"] = result["AwayTeam"].replace(TEAM_NAME_MAP)

    for old_col, new_col in odds_map.items():
        if old_col in result.columns:
            # Convert to numeric, coerce errors (handles strings from XLSX)
            result[new_col] = pd.to_numeric(result[old_col], errors="coerce")

    missing = [c for c in ["B365H", "B365D", "B365A"] if c not in result.columns]
    if missing:
        print(f"  ERROR - Missing columns: {missing}")
        return

    # Convert Date to DD/MM/YY format (feature_store.py expects this format with dayfirst=True)
    if "Date" in result.columns:
        result["Date"] = pd.to_datetime(result["Date"]).dt.strftime("%d/%m/%y")

    cols = ["Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
    result = result[[c for c in cols if c in result.columns]]

    before = len(result)
    result = result.dropna(subset=["B365H", "B365D", "B365A"])
    result = result[
        (result["B365H"] >= 1.01)
        & (result["B365D"] >= 1.01)
        & (result["B365A"] >= 1.01)
    ]
    print(f"  Rows: {before} -> {len(result)}")

    out_path = RAW_DIR / f"odds_{year}.csv"
    result.to_csv(out_path, index=False)
    sample = result.iloc[0]
    print(f"  Saved: {out_path}")
    print(f"  Sample: {sample['HomeTeam']} vs {sample['AwayTeam']}")
    print()
    return len(result)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    # WC final tournaments
    total += extract_sheet("WorldCup2014", 2014, {"bet365-H": "B365H", "bet365-D": "B365D", "bet365-A": "B365A"})
    total += extract_sheet("WorldCup2018", 2018, {"H-Avg": "B365H", "D-Avg": "B365D", "A-Avg": "B365A"})
    total += extract_sheet("WorldCup2022", 2022, {"bet365-H": "B365H", "bet365-D": "B365D", "bet365-A": "B365A"})
    # 2026 qualifiers (H_Avg as proxy for bookmaker odds)
    total += extract_sheet("WorldCup2026Qualifiers", 2026, {"H_Avg": "B365H", "D_Avg": "B365D", "A_Avg": "B365A"})

    print(f"Total rows extracted: {total}")
    print("\nFiles:")
    for f in sorted(RAW_DIR.glob("*.csv")):
        print(f"  {f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
