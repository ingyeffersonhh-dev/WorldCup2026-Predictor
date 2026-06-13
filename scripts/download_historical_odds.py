"""
scripts/download_historical_odds.py — Download football-data.co.uk historical odds.

Downloads match data with Bet365 odds for major European leagues from 2005
to the present, saving each year as an odds CSV compatible with the feature
store pipeline.

Usage:
    python scripts/download_historical_odds.py

Output:
    data/raw/odds/historical/odds_{year}.csv  (one per year, 2005-2022)
"""

import logging
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/odds/historical")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Major European leagues with Bet365 coverage going back to 2005
# League codes used by football-data.co.uk
LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "SC0": "Scottish Premiership",
    "D1": "Bundesliga",
    "D2": "Bundesliga 2",
    "I1": "Serie A",
    "SP1": "La Liga",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "B1": "Jupiler League",
    "P1": "Primeira Liga",
    "T1": "Super Lig",
}

# URL template for 1993-2016 seasons (old format)
# Format: https://www.football-data.co.uk/mmz4281/1415/E0.csv
def season_code(year):
    """Convert a year to the 2-digit season code: 2005 → 0506, 2006 → 0607, etc."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"

def download_season_league(season_start, league_code):
    """Download one league-season CSV. Returns (season_start, league_code, df|None)."""
    sc = season_code(season_start)
    url = f"https://www.football-data.co.uk/mmz4281/{sc}/{league_code}.csv"
    alt_url = f"https://www.football-data.co.uk/new-data/{sc}/{league_code}.csv"
    
    for u in [url, alt_url]:
        try:
            df = pd.read_csv(u)
            if df is not None and len(df) > 0:
                return (season_start, league_code, df)
        except Exception:
            continue
    return (season_start, league_code, None)

def process_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Parse football-data.co.uk CSV, extract Bet365 odds, return clean DataFrame."""
    # Normalise columns
    df.columns = [c.strip() for c in df.columns]
    
    if "Date" not in df.columns:
        return pd.DataFrame()
    
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    
    # Try Bet365 first, then other bookmakers
    odds_cols = None
    for prefix in ["B365", "BW", "IW", "LB", "WH", "SJ", "VC", "SB"]:
        cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
        if all(c in df.columns for c in cols):
            odds_cols = cols
            break
    
    if odds_cols is None:
        return pd.DataFrame()
    
    result = df[["date", "HomeTeam", "AwayTeam"] + odds_cols].copy()
    result.columns = ["date", "home_team", "away_team", "odds_h", "odds_d", "odds_a"]
    
    # Clean odds
    result = result.dropna(subset=["odds_h", "odds_d", "odds_a"])
    result = result[
        (result["odds_h"] >= 1.01)
        & (result["odds_d"] >= 1.01)
        & (result["odds_a"] >= 1.01)
    ]
    
    return result

def download_all():
    """Download all historical odds data from 2005 to 2022."""
    years = range(2005, 2023)  # 2005/06 to 2022/23 seasons
    all_data = {}  # season_start_year -> list of DataFrames
    
    tasks = [(year, league) for year in years for league in LEAGUES]
    logger.info(f"Downloading {len(tasks)} league-season files...")
    
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(download_season_league, year, league): (year, league)
            for year, league in tasks
        }
        
        for future in as_completed(futures):
            year, league = futures[future]
            try:
                _, _, df = future.result()
                if df is not None and len(df) > 0:
                    processed = process_odds(df)
                    if len(processed) > 0:
                        all_data.setdefault(year, []).append(processed)
                        logger.info(f"  ✅ {year}/{year+1} {league} ({LEAGUES[league]}): {len(processed)} matches")
                    else:
                        logger.info(f"  ⬜ {year}/{year+1} {league} ({LEAGUES[league]}): no odds data")
                else:
                    logger.info(f"  ❌ {year}/{year+1} {league} ({LEAGUES[league]}): not found")
            except Exception as e:
                logger.warning(f"  ❌ {year}/{year+1} {league}: {e}")
    
    # Save one file per year (combine all leagues for that year)
    total = 0
    for year in sorted(all_data.keys()):
        combined = pd.concat(all_data[year], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"])
        combined = combined.sort_values("date").reset_index(drop=True)
        path = RAW_DIR / f"odds_{year}.csv"
        combined.to_csv(path, index=False)
        logger.info(f"Saved {path}: {len(combined)} matches")
        total += len(combined)
    
    logger.info(f"\nTotal: {total} matches across {len(all_data)} years")
    logger.info(f"Files saved to: {RAW_DIR}")

if __name__ == "__main__":
    download_all()
