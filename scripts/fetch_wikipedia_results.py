"""
fetch_wikipedia_results.py

Auto-fetch live match results from Wikipedia's 2026 FIFA World Cup page.
Updates data/raw/live_results.csv for the Monte Carlo simulation.

Usage: python scripts/fetch_wikipedia_results.py

This script fetches the Wikipedia page, finds score patterns involving
fixture teams, and merges them into live_results.csv.
"""
import json
import re
import csv
import sys
import os
from pathlib import Path
from datetime import datetime
import urllib.request

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/ is one level below project root
FIXTURE_PATH = PROJECT_ROOT / "data" / "raw" / "fixture_2026.csv"
RESULTS_PATH = PROJECT_ROOT / "data" / "raw" / "live_results.csv"

# ── Team name aliases ──────────────────────────────────────────────────────
# Maps Wikipedia naming variations to fixture CSV names
TEAM_ALIASES = {
    "Czechia": "Czech Republic",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia": "Bosnia and Herzegovina",
    "USA": "United States",
    "U.S.": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Holland": "Netherlands",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
}


def normalize(name):
    return TEAM_ALIASES.get(name.strip(), name.strip())


def load_fixture_teams():
    """Return dict of match_id -> (home_team, away_team)."""
    teams = {}
    with open(FIXTURE_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            teams[int(row["match_id"])] = (row["home_team"].strip(), row["away_team"].strip())
    return teams


def fetch_wikipedia_text():
    """Fetch the full 2026 World Cup article as cleaned plain text."""
    url = (
        "https://en.wikipedia.org/w/api.php?"
        "action=parse&page=2026_FIFA_World_Cup&prop=text&format=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read().decode("utf-8"))
    html = data["parse"]["text"]["*"]
    clean = re.sub(r"<[^>]+>", " ", html)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def find_scores_between(text, team_a, team_b):
    """
    Search text for patterns like 'Mexico 2-0 South Africa' where
    both team_a and team_b appear with a numeric score between them.
    Returns (home_score, away_score) or None.
    """
    # Normalize dashes
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    a_positions = [m.start() for m in re.finditer(re.escape(team_a), text)]
    b_positions = [m.start() for m in re.finditer(re.escape(team_b), text)]

    for pa in a_positions:
        for pb in b_positions:
            if pa >= pb:
                continue
            between = text[pa + len(team_a):pb]
            score_match = re.search(r"\b(\d+)[-–—](\d+)\b", between)
            if score_match and len(between) < 80:
                return (int(score_match.group(1)), int(score_match.group(2)))

    return None


def extract_all_scores(text, fixture_teams):
    """
    For every fixture match, try to find its score in the Wikipedia text.
    Returns dict of (home_team, away_team) -> (home_score, away_score).
    """
    results = {}
    for mid, (ht, at) in fixture_teams.items():
        score = find_scores_between(text, ht, at)
        if score:
            results[(ht, at)] = score
    return results


def save_results(results, fixture_teams):
    """Merge results with existing CSV and save."""
    # Load existing
    existing = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[(row["home_team"].strip(), row["away_team"].strip())] = (
                    int(row["home_score"]),
                    int(row["away_score"]),
                )

    merged = {**existing, **results}
    new_count = len(set(results.keys()) - set(existing.keys()))
    updated_count = len(
        set(results.keys()) - set(existing.keys())
    )  # all matched from this run are "found"

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["match_id", "home_team", "away_team", "home_score", "away_score"])
        for mid, (ht, at) in sorted(fixture_teams.items()):
            key = (ht, at)
            if key in merged:
                w.writerow([mid, ht, at, merged[key][0], merged[key][1]])

    print(f"\nSaved: {RESULTS_PATH}")
    print(f"  New results this run: {len(results)}")
    print(f"  Total results stored: {len(merged)}")

    # Print what changed
    for key in results:
        if key not in existing:
            print(f"  NEW: {key[0]} {results[key][0]}-{results[key][1]} {key[1]}")
        elif results[key] != existing[key]:
            print(f"  UPDATED: {key[0]} {existing[key][0]}-{existing[key][1]} {key[1]} -> {results[key][0]}-{results[key][1]} {key[1]}")

    return merged


def main():
    print("=" * 60)
    print(f"Wikipedia Results Fetcher — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. Load fixture
    print("\n[1] Loading fixture...")
    fixture_teams = load_fixture_teams()
    print(f"    {len(fixture_teams)} matches")

    # 2. Fetch Wikipedia
    print("\n[2] Fetching Wikipedia article...")
    try:
        text = fetch_wikipedia_text()
    except Exception as e:
        print(f"    FAILED: {e}")
        print("    (Wikipedia may be rate-limiting. Try again later.)")
        sys.exit(1)
    print(f"    {len(text):,} characters")

    # 3. Extract scores
    print("\n[3] Searching for match scores...")
    results = extract_all_scores(text, fixture_teams)

    if not results:
        print("    No new scores found in Wikipedia text.")
        print("    (Matches may not have been played yet or Wikipedia hasn't updated.)")
        # If there are existing results, tell user they're preserved
        if RESULTS_PATH.exists():
            print(f"    Preserving existing results at {RESULTS_PATH}")
        sys.exit(0)

    for (ht, at), (hs, as_) in sorted(results.items()):
        print(f"    {ht} {hs}-{as_} {at}")

    # 4. Save
    print("\n[4] Saving...")
    save_results(results, fixture_teams)

    # 5. Next steps
    print(f"\n{'=' * 60}")
    print("Done! Run the Monte Carlo simulation to recalculate:")
    print(f"  cd {PROJECT_ROOT}")
    print("  python monte_carlo.py --n-sims 1000 --closest-only")
    print()


if __name__ == "__main__":
    main()
