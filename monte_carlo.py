"""
monte_carlo.py — Stage 5a: Monte Carlo Tournament Simulation

Simulates the 2026 FIFA World Cup (48 teams, 12 groups, 104 matches) using
path-dependent ELO updates with XGBoost (1X2) and Dixon-Coles Poisson (exact
score) models.

Produces:
  - champion_probs.csv — champion probabilities per team
  - match_probs.csv    — per-match 1X2 probabilities

Based on spec R6.1-R6.10 and design C6.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Feature columns used by the XGBoost model (in order)
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

# ELO constants (mirrored from elo.py for standalone use)
K_WORLD_CUP = 40.0
DRAW_SCORE = 0.5
WIN_SCORE = 1.0
LOSS_SCORE = 0.0


# ===================================================================
# MonteCarloSimulator
# ===================================================================
class MonteCarloSimulator:
    """Full tournament Monte Carlo simulator with path-dependent ELO.

    Parameters
    ----------
    max_goals : int
        Maximum goals per team for the Poisson score matrix (default 6).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_goals: int = 6,
        random_state: int = 42,
        closest_only: bool = False,
    ) -> None:
        self.max_goals = max_goals
        self.rng = np.random.default_rng(random_state)
        self._match_counter: int = 0
        self.prediction_cache: Dict[Tuple, Tuple] = {}
        self.closest_only = closest_only
        self.live_results_map = {}
        self.closest_cutoff = pd.to_datetime("2099-01-01")

    # ------------------------------------------------------------------
    # Fixture loading (Task 3.4)
    # ------------------------------------------------------------------

    @staticmethod
    def load_fixture(path: str | Path) -> pd.DataFrame:
        """Load the 2026 World Cup fixture from CSV.

        The CSV should have columns:
            match_id, group, round, date, home_team, away_team, neutral_venue

        Parameters
        ----------
        path : str | Path
            Path to ``fixture_2026.csv``.

        Returns
        -------
        pd.DataFrame
            Parsed fixture with datetime dates and sorted by match_id.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Fixture file not found: {path}. "
                f"Create a fixture_2026.csv in data/raw/."
            )

        df = pd.read_csv(path)
        required = {"match_id", "group", "round", "date", "home_team", "away_team"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Fixture CSV missing columns: {missing}"
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

        logger.info(
            "Loaded fixture: %d matches, %d groups",
            len(df), df["group"].nunique(),
        )
        return df

    @staticmethod
    def init_group_stage(fixture_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Organise fixture data into group-stage structure.

        Parameters
        ----------
        fixture_df : pd.DataFrame
            Loaded fixture data (group matches only).

        Returns
        -------
        dict
            ``{group: {"teams": [4 team names], "matches": DataFrame}}``
        """
        if fixture_df.empty:
            return {}

        groups: Dict[str, Dict[str, Any]] = {}
        for group in sorted(fixture_df["group"].unique()):
            grp_matches = fixture_df[fixture_df["group"] == group].copy()
            if grp_matches.empty:
                continue

            # Extract the 4 teams in this group
            teams: List[str] = list(
                set(grp_matches["home_team"].unique())
                | set(grp_matches["away_team"].unique())
            )
            teams.sort()

            groups[group] = {
                "teams": teams,
                "matches": grp_matches,
            }

        logger.info(
            "Initialised %d groups (total %d teams)",
            len(groups), sum(len(g["teams"]) for g in groups.values()),
        )
        return groups

    def build_bracket(
        self,
        fixture_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Build the knockout bracket structure from fixture data.

        The bracket is defined as a staged binary tree:
            R32 (16 matches) → R16 (8) → QF (4) → SF (2) → Final + 3rd Place

        Returns a dict with round names and the match adjacency structure.

        Parameters
        ----------
        fixture_df : pd.DataFrame
            Full fixture including group matches.

        Returns
        -------
        dict
            ``{"rounds": [round_names], "matches_per_round": {round: n}}``
            plus bracket adjacency information.
        """
        # Define the knockout rounds
        ko_rounds = ["R32", "R16", "QF", "SF", "3rd_place", "Final"]

        # Build bracket tree: for each round, map match_index ->
        #   {"feeder_a": (prev_round, prev_match_idx),
        #    "feeder_b": (prev_round, prev_match_idx)}
        # R32 has no feeders (teams come from group stage)
        # R16 feeders are R32 matches
        # QF feeders are R16 matches, etc.

        bracket: Dict[str, Any] = {
            "rounds": ko_rounds,
            "matches_per_round": {
                "R32": 16,
                "R16": 8,
                "QF": 4,
                "SF": 2,
                "3rd_place": 1,
                "Final": 1,
            },
        }

        # Build adjacency: which previous matches feed into which current match
        feeders: Dict[str, List] = {}
        for i, rnd in enumerate(ko_rounds):
            if rnd == "R32":
                feeders[rnd] = []  # seeded from group stage
            else:
                prev_rnd = ko_rounds[i - 1]
                n_prev = bracket["matches_per_round"][prev_rnd]
                n_curr = bracket["matches_per_round"][rnd]
                # Each current match takes two consecutive previous matches
                curr_feeders = []
                for j in range(n_curr):
                    curr_feeders.append((prev_rnd, 2 * j, prev_rnd, 2 * j + 1))
                feeders[rnd] = curr_feeders

        bracket["feeders"] = feeders

        logger.info(
            "Built KO bracket: %s (%d total matches)",
            " → ".join(ko_rounds),
            sum(bracket["matches_per_round"].values()),
        )
        return bracket

    # ------------------------------------------------------------------
    # Group simulation (Task 3.5)
    # ------------------------------------------------------------------

    def simulate_group(
        self,
        group_data: Dict[str, Any],
        models: Dict[str, Any],
        elo_state: Dict[str, Any],
        base_features: Dict[str, Dict[str, float]],
        initial_elo: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simulate all matches in one group.

        Parameters
        ----------
        group_data : dict
            Group structure from ``init_group_stage``.
        models : dict
            ``{"xgb": XGBoostModel, "poisson": DixonColesPoisson}``.
        elo_state : dict
            Mutable ELO state ``{"ratings": dict, "match_counts": dict}``.
        base_features : dict
            Pre-tournament features per team
            ``{team: {"form_home_5f": ..., "rest_days": ..., ...}}``.

        Returns
        -------
        dict
            ``{"team": str, "played": int, "points": int, "gd": int,
              "gs": int, "results": list[dict]}`` for each team.
        """
        teams = group_data["teams"]
        matches_df = group_data["matches"]

        # Initialise standings
        standings = {
            team: {
                "team": team,
                "played": 0,
                "points": 0,
                "gd": 0,
                "gs": 0,
                "ga": 0,
                "results": [],
            }
            for team in teams
        }

        match_results = []
        for _, match_row in matches_df.iterrows():
            home = match_row["home_team"]
            away = match_row["away_team"]
            rest_home = base_features.get(home, {}).get("rest_days", 7)
            rest_away = base_features.get(away, {}).get("rest_days", 7)

            # Check if match is already played
            key = (home, away)
            if key in self.live_results_map:
                home_goals, away_goals = self.live_results_map[key]
            else:
                if self.closest_only:
                    match_date = pd.to_datetime(match_row["date"])
                    if match_date <= self.closest_cutoff:
                        # Closest match -> simulate with full MC
                        home_goals, away_goals = self._simulate_match(
                            home, away, None, models, elo_state, base_features, rest_home, rest_away, initial_elo
                        )
                    else:
                        # Far future match -> resolve deterministically
                        home_goals, away_goals = self._resolve_match_deterministic(
                            home, away, models, elo_state, base_features, rest_home, rest_away, initial_elo
                        )
                else:
                    # Regular simulation
                    home_goals, away_goals = self._simulate_match(
                        home, away, None, models, elo_state, base_features, rest_home, rest_away, initial_elo
                    )

            # Update standings
            standings[home]["played"] += 1
            standings[away]["played"] += 1
            standings[home]["gs"] += home_goals
            standings[home]["ga"] += away_goals
            standings[away]["gs"] += away_goals
            standings[away]["ga"] += home_goals

            if home_goals > away_goals:
                standings[home]["points"] += 3
                standings[home]["gd"] += home_goals - away_goals
                standings[away]["gd"] += away_goals - home_goals
            elif home_goals == away_goals:
                standings[home]["points"] += 1
                standings[away]["points"] += 1
            else:
                standings[away]["points"] += 3
                standings[home]["gd"] += home_goals - away_goals
                standings[away]["gd"] += away_goals - home_goals

            result = {
                "home": home,
                "away": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "tournament_type": "World Cup 2026",
            }
            match_results.append(result)
            standings[home]["results"].append(result)
            standings[away]["results"].append(result)

        # Sort standings: points desc, GD desc, GS desc
        sorted_standings = sorted(
            standings.values(),
            key=lambda r: (r["points"], r["gd"], r["gs"]),
            reverse=True,
        )

        return {
            "standings": sorted_standings,
            "matches": match_results,
        }

    # ------------------------------------------------------------------
    # Group resolution (Task 3.5)
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_group(
        group_result: Dict[str, Any],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Determine the top 2 advancing teams from a group.

        Parameters
        ----------
        group_result : dict
            Output from ``simulate_group``.

        Returns
        -------
        top2 : list[str]
            The two advancing teams (1st and 2nd place).
        standings : dict
            Full standings sorted.
        """
        standings = group_result["standings"]
        top2 = [s["team"] for s in standings[:2]]
        return top2, standings

    @staticmethod
    def rank_third_place_teams(
        all_group_results: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Rank all 3rd place teams across groups.

        Parameters
        ----------
        all_group_results : dict
            ``{group: group_result}`` from all groups.

        Returns
        -------
        list[tuple[str, dict]]
            Sorted list of ``(group, standing)`` for 3rd place teams,
            best first.
        """
        third_place = []
        for group, result in all_group_results.items():
            standings = result["standings"]
            if len(standings) >= 3:
                third = standings[2]
                third_place.append((group, third))

        # Sort by points, GD, GS descending
        third_place.sort(
            key=lambda r: (r[1]["points"], r[1]["gd"], r[1]["gs"]),
            reverse=True,
        )
        return third_place

    # ------------------------------------------------------------------
    # KO match simulation (Task 3.5)
    # ------------------------------------------------------------------

    def simulate_ko_match(
        self,
        home: str,
        away: str,
        models: Dict[str, Any],
        elo_state: Dict[str, Any],
        base_features: Dict[str, Dict[str, float]],
        rest_days_home: int = 4,
        rest_days_away: int = 4,
        deterministic: bool = False,
    ) -> Tuple[str, str, int, int, bool]:
        """Simulate a single knockout match using cache or deterministically."""
        p_home, p_draw, p_away, lambda_h_val, lambda_a_val = self._predict_match_cached(
            home, away, elo_state, base_features, models, rest_days_home, rest_days_away
        )

        if deterministic:
            if p_home >= p_away:
                winner, loser = home, away
                home_goals, away_goals = 2, 1
            else:
                winner, loser = away, home
                home_goals, away_goals = 1, 2
            is_penalty = False
            self._update_elo(winner, loser, home_goals, away_goals, home, away, elo_state)
            return winner, loser, home_goals, away_goals, is_penalty

        poisson = models["poisson"]
        rho = poisson.params_[-1] if (poisson is not None and poisson.params_ is not None) else 0.0

        # Sample outcome from XGBoost 1X2
        outcome = self.rng.choice([1, 0, 2], p=[p_home, p_draw, p_away])

        if outcome == 1:
            # Home win — sample score from Poisson (home > away)
            home_goals, away_goals = self._sample_conditional_score(
                lambda_h_val, lambda_a_val, rho, "home"
            )
            is_penalty = False
            winner, loser = home, away

        elif outcome == 2:
            # Away win — sample score from Poisson (home < away)
            home_goals, away_goals = self._sample_conditional_score(
                lambda_h_val, lambda_a_val, rho, "away"
            )
            is_penalty = False
            winner, loser = away, home

        else:
            # XGBoost says draw → use Poisson to break the tie
            # Sample from the full Poisson score matrix
            score_matrix = poisson.exact_score_prob(
                lambda_h_val, lambda_a_val, rho
            )
            flat = score_matrix.ravel()
            flat = np.maximum(flat, 0)
            flat /= flat.sum()
            idx = self.rng.choice(len(flat), p=flat)
            max_g = score_matrix.shape[0] - 1
            home_goals = int(idx // (max_g + 1))
            away_goals = int(idx % (max_g + 1))

            if home_goals > away_goals:
                winner, loser = home, away
                is_penalty = False
            elif away_goals > home_goals:
                winner, loser = away, home
                is_penalty = False
            else:
                # Still a draw → penalties → home wins (simplification)
                winner, loser = home, away
                is_penalty = True

        # Path-dependent ELO update
        self._update_elo(winner, loser, home_goals, away_goals,
                         home, away, elo_state)

        return winner, loser, home_goals, away_goals, is_penalty

    # ------------------------------------------------------------------
    # Full tournament simulation (Task 3.5)
    # ------------------------------------------------------------------

    def simulate_tournament(
        self,
        fixture_df: pd.DataFrame,
        models: Dict[str, Any],
        initial_elo: Dict[str, Any],
        base_features: Dict[str, Dict[str, float]],
        n_sims: int = 1000,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        """Run the full tournament simulation multiple times.

        Parameters
        ----------
        fixture_df : pd.DataFrame
            Loaded fixture (group matches).
        models : dict
            ``{"xgb": ..., "poisson": ...}``.
        initial_elo : dict
            Snapshot of ELO state before the tournament.
        base_features : dict
            Pre-tournament features per team.
        n_sims : int
            Number of simulations (default 1000).
        verbose : bool
            Show progress bar.

        Returns
        -------
        champion_df : pd.DataFrame
            Champion probabilities per team.
        match_df : pd.DataFrame
            Per-match 1X2 probabilities.
        all_results : list[dict]
            Raw per-simulation results.
        """
        # Load live results if present
        self.live_results_map = {}
        live_path = Path("data/raw/live_results.csv")
        if live_path.exists():
            try:
                live_df = pd.read_csv(live_path)
                for _, row in live_df.iterrows():
                    key = (row["home_team"], row["away_team"])
                    self.live_results_map[key] = (int(row["home_score"]), int(row["away_score"]))
            except Exception as e:
                logger.warning(f"Error loading live_results.csv: {e}")

        # Determine closest cutoff if closest_only is enabled
        if self.closest_only:
            unplayed_dates = []
            for _, row in fixture_df.iterrows():
                key = (row["home_team"], row["away_team"])
                if key not in self.live_results_map:
                    unplayed_dates.append(row["date"])
            
            if unplayed_dates:
                min_unplayed_date = min(unplayed_dates)
                min_unplayed_dt = pd.to_datetime(min_unplayed_date)
                self.closest_cutoff = min_unplayed_dt + pd.Timedelta(days=2)
                logger.info(f"Simulation in closest-only mode. Earliest unplayed: {min_unplayed_dt.strftime('%Y-%m-%d')}, cutoff: {self.closest_cutoff.strftime('%Y-%m-%d')}")
            else:
                self.closest_cutoff = pd.to_datetime("2099-01-01")
        else:
            self.closest_cutoff = pd.to_datetime("2099-01-01")

        # Pre-group structure
        group_structure = self.init_group_stage(fixture_df)
        bracket = self.build_bracket(fixture_df)

        # Group match identifiers (fixed across sims)
        group_match_ids = []
        for group, gdata in group_structure.items():
            for _, row in gdata["matches"].iterrows():
                mid = f"G_{group}_{row['home_team']}_{row['away_team']}"
                group_match_ids.append({
                    "match_id": mid,
                    "group": group,
                    "round": "group",
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                })

        # Accumulators for per-match probs (group matches only, since KO varies)
        # {match_id: {"p_home_sum": ..., "p_draw_sum": ..., "p_away_sum": ..., count}}
        match_acc: Dict[str, Dict[str, float]] = {
            m["match_id"]: {
                "group": m["group"],
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "p_home_sum": 0.0,
                "p_draw_sum": 0.0,
                "p_away_sum": 0.0,
                "avg_home_goals": 0.0,
                "avg_away_goals": 0.0,
                "count": 0,
            }
            for m in group_match_ids
        }

        # Champion accumulator
        champion_counts: Dict[str, int] = {}
        runner_up_counts: Dict[str, int] = {}

        all_results: List[Dict[str, Any]] = []

        iterator = range(n_sims)
        if verbose:
            iterator = tqdm(iterator, desc="Tournament simulations")

        for _ in iterator:
            # Deep-copy ELO state for path-dependent simulation
            elo_state = deepcopy(initial_elo)

            # --- Group stage ---
            all_group_results: Dict[str, Dict[str, Any]] = {}
            group_match_outcomes = []

            for group, gdata in group_structure.items():
                result = self.simulate_group(
                    gdata, models, elo_state, base_features, initial_elo
                )
                all_group_results[group] = result

                for m in result["matches"]:
                    mid = f"G_{group}_{m['home']}_{m['away']}"
                    if mid in match_acc:
                        rest_home = base_features.get(m["home"], {}).get("rest_days", 7)
                        rest_away = base_features.get(m["away"], {}).get("rest_days", 7)
                        ph, pd_, pa, _, _ = self._predict_match_cached(
                            m["home"], m["away"], initial_elo, base_features, models, rest_home, rest_away
                        )

                        match_acc[mid]["p_home_sum"] += ph
                        match_acc[mid]["p_draw_sum"] += pd_
                        match_acc[mid]["p_away_sum"] += pa
                        match_acc[mid]["avg_home_goals"] += m["home_goals"]
                        match_acc[mid]["avg_away_goals"] += m["away_goals"]
                        match_acc[mid]["count"] += 1

            # --- Resolve group stage: top 2 + best 3rd place ---
            advancing: List[Dict[str, Any]] = []  # list of {"team", "group", "rank", "points", "gd", "gs"}

            for group, result in all_group_results.items():
                standings = result["standings"]
                for rank, s in enumerate(standings):
                    advancing.append({
                        "team": s["team"],
                        "group": group,
                        "rank": rank + 1,  # 1-based
                        "points": s["points"],
                        "gd": s["gd"],
                        "gs": s["gs"],
                    })

            # Top 2 from each group automatically advance
            auto_advancers = [a for a in advancing if a["rank"] <= 2]

            # Best 3rd place teams (ranked by points, GD, GS)
            third_place = [a for a in advancing if a["rank"] == 3]
            third_place.sort(
                key=lambda r: (r["points"], r["gd"], r["gs"]),
                reverse=True,
            )
            best_third = third_place[:8]

            all_advancers = auto_advancers + best_third

            # --- Sort 32 advancing teams for bracket seeding ---
            # Group winners (rank 1) get highest priority
            # Then runners-up, then 3rd place
            # Within each tier, sort by group performance
            tier1 = sorted(
                [a for a in all_advancers if a["rank"] == 1],
                key=lambda r: (r["points"], r["gd"], r["gs"]),
                reverse=True,
            )
            tier2 = sorted(
                [a for a in all_advancers if a["rank"] == 2],
                key=lambda r: (r["points"], r["gd"], r["gs"]),
                reverse=True,
            )
            tier3 = sorted(
                [a for a in all_advancers if a["rank"] >= 3],
                key=lambda r: (r["points"], r["gd"], r["gs"]),
                reverse=True,
            )

            seeded_ranking = tier1 + tier2 + tier3
            # Top 16 seeds: all group winners (12) + top 4 runners-up
            # Bottom 16: remaining 8 runners-up + 8 best 3rd place

            # R32 pairings: 1 vs 32, 2 vs 31, ..., 16 vs 17
            n_ko = len(seeded_ranking)
            r32_pairings = []
            for i in range(n_ko // 2):
                r32_pairings.append((
                    seeded_ranking[i]["team"],
                    seeded_ranking[n_ko - 1 - i]["team"],
                ))

            # --- Knockout stage ---
            ko_state = self._simulate_ko_round(
                r32_pairings, "R32", models, elo_state, base_features
            )
            r16_state = self._simulate_ko_round(
                [(m["winner"], m.get("extra", "")) for m in ko_state],
                "R16", models, elo_state, base_features,
                feeders=True,
            )

            # Flatten the bracket for higher rounds
            # Actually, let me track the bracket properly
            # R32 -> R16 pairs: (R32_0, R32_1) -> R16_0, (R32_2, R32_3) -> R16_1, ...
            r16_winners = []
            for j in range(0, len(ko_state), 2):
                # Winner advances
                r16_winners.append((ko_state[j]["winner"], ko_state[j + 1]["winner"]))

            qf_winners = []
            for j in range(0, len(r16_winners), 2):
                w1 = r16_winners[j][1] if r16_winners[j][0] == "feeder" else r16_winners[j][0]
                w2 = r16_winners[j + 1][1] if r16_winners[j + 1][0] == "feeder" else r16_winners[j + 1][0]
                qf_pair = self._simulate_ko_pair(
                    w1, w2, models, elo_state, base_features
                )
                qf_winners.append(qf_pair)

            sf_winners = []
            for j in range(0, len(qf_winners), 2):
                sf_pair = self._simulate_ko_pair(
                    qf_winners[j][0], qf_winners[j + 1][0],
                    models, elo_state, base_features
                )
                sf_winners.append(sf_pair)

            # Final: SF winners
            final = self._simulate_ko_pair(
                sf_winners[0][0], sf_winners[1][0],
                models, elo_state, base_features
            )
            champion = final[0]

            # 3rd place: SF losers
            third_match = self._simulate_ko_pair(
                sf_winners[0][1], sf_winners[1][1],
                models, elo_state, base_features
            )
            third_place_winner = third_match[0]
            runner_up = final[1]

            # Track
            champion_counts[champion] = champion_counts.get(champion, 0) + 1
            runner_up_counts[runner_up] = runner_up_counts.get(runner_up, 0) + 1

            sim_result = {
                "champion": champion,
                "runner_up": runner_up,
                "third_place": third_place_winner,
            }
            all_results.append(sim_result)

        # --- Aggregate results ---
        champion_df = self.track_champion(all_results)

        # Fill in team names ONLY from the fixture (48 qualified teams)
        # Previously used initial_elo which includes ALL teams (226+) like Italy
        fixture_teams = set(fixture_df["home_team"].unique()) | set(fixture_df["away_team"].unique())
        all_teams = sorted(fixture_teams)
        for team in all_teams:
            if team not in champion_counts:
                champion_counts[team] = 0
            if team not in runner_up_counts:
                runner_up_counts[team] = 0

        # Build champion_df
        champion_records = []
        for team in sorted(
            set(list(champion_counts.keys()) + list(runner_up_counts.keys()))
        ):
            champion_records.append({
                "team": team,
                "champion_count": champion_counts.get(team, 0),
                "champion_pct": round(
                    champion_counts.get(team, 0) / max(n_sims, 1) * 100, 2
                ),
                "runner_up_count": runner_up_counts.get(team, 0),
                "runner_up_pct": round(
                    runner_up_counts.get(team, 0) / max(n_sims, 1) * 100, 2
                ),
            })
        champion_df = pd.DataFrame(champion_records)
        champion_df = champion_df.sort_values(
            "champion_count", ascending=False
        ).reset_index(drop=True)

        # Build match_df from group match accumulators
        match_records = []
        for mid, acc in match_acc.items():
            n = max(acc["count"], 1)
            match_records.append({
                "match_id": mid,
                "group": acc["group"],
                "round": "group",
                "home_team": acc["home_team"],
                "away_team": acc["away_team"],
                "p_home": round(acc["p_home_sum"] / n, 4),
                "p_draw": round(acc["p_draw_sum"] / n, 4),
                "p_away": round(acc["p_away_sum"] / n, 4),
                "avg_home_goals": round(acc["avg_home_goals"] / n, 2),
                "avg_away_goals": round(acc["avg_away_goals"] / n, 2),
            })
        match_df = pd.DataFrame(match_records)

        logger.info(
            "Simulation complete: %d sims, %d unique champions",
            n_sims, champion_df["champion_count"].gt(0).sum(),
        )
        return champion_df, match_df, all_results

    # ------------------------------------------------------------------
    # Results aggregation (Task 3.6)
    # ------------------------------------------------------------------

    @staticmethod
    def track_champion(
        all_results: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Aggregate champion counts from all simulations.

        Parameters
        ----------
        all_results : list[dict]
            List of per-sim results with ``{"champion": ..., ...}``.

        Returns
        -------
        pd.DataFrame
            Champion ranking sorted by count descending.
        """
        counts: Dict[str, int] = {}
        for r in all_results:
            counts[r["champion"]] = counts.get(r["champion"], 0) + 1

        n_sims = len(all_results)
        df = pd.DataFrame([
            {"team": team, "champion_count": cnt,
             "pct": round(cnt / n_sims * 100, 2)}
            for team, cnt in sorted(counts.items(), key=lambda x: -x[1])
        ])
        return df

    @staticmethod
    def track_match_probs(
        match_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return match probabilities DataFrame (convenience wrapper).

        Parameters
        ----------
        match_df : pd.DataFrame
            Match probabilities from ``simulate_tournament``.

        Returns
        -------
        pd.DataFrame
            Sorted by match_id.
        """
        return match_df.sort_values("match_id").reset_index(drop=True)

    @staticmethod
    def convergence_check(
        champion_df: pd.DataFrame,
        min_sims: int = 100,
    ) -> Dict[str, Any]:
        """Check if champion probability has converged.

        Computes the running mean of the top team's probability and checks
        if it has stabilised.

        Parameters
        ----------
        champion_df : pd.DataFrame
            Champion probability data.
        min_sims : int
            Minimum simulations required for convergence check.

        Returns
        -------
        dict
            ``{"is_converged": bool, "top_team": str, "top_pct": float,
              "margin": float, "message": str}``
        """
        if len(champion_df) == 0:
            return {
                "is_converged": False,
                "message": "No champion data available.",
            }

        top = champion_df.iloc[0]
        n_sims = int(champion_df["champion_count"].sum())

        if n_sims < min_sims:
            return {
                "is_converged": False,
                "top_team": top["team"],
                "top_pct": top.get("pct", top.get("champion_pct", 0)),
                "n_sims": n_sims,
                "message": (
                    f"Insufficient simulations ({n_sims} < {min_sims}) "
                    f"for reliable convergence check."
                ),
            }

        # Simple convergence: top team has > 5% and clear margin
        top_pct = top.get("pct", top.get("champion_pct", 0))
        if len(champion_df) > 1:
            second_pct = champion_df.iloc[1].get("pct", champion_df.iloc[1].get("champion_pct", 0))
            margin = top_pct - second_pct
        else:
            margin = 100.0

        is_converged = n_sims >= 500 and margin > 0

        return {
            "is_converged": is_converged,
            "top_team": top["team"],
            "top_pct": top_pct,
            "margin": margin,
            "n_sims": n_sims,
            "message": (
                f"{'Converged' if is_converged else 'Not yet converged'}: "
                f"top={top['team']} at {top_pct:.1f}% "
                f"(n={n_sims})"
            ),
        }

    @staticmethod
    def save_results(
        champion_df: pd.DataFrame,
        match_df: pd.DataFrame,
        output_dir: str | Path = "data/processed",
    ) -> Dict[str, str]:
        """Export champion and match probabilities to CSV (R6.8, R6.9).

        Parameters
        ----------
        champion_df : pd.DataFrame
            Champion probabilities.
        match_df : pd.DataFrame
            Match probabilities (group matches).
        output_dir : str | Path
            Output directory.

        Returns
        -------
        dict[str, str]
            ``{"champion": path, "match": path}``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        champion_path = output_dir / "champion_probs.csv"
        champion_df.to_csv(champion_path, index=False)
        logger.info("Saved champion probs (%d teams) → %s", len(champion_df), champion_path)

        match_path = output_dir / "match_probs.csv"
        match_df.to_csv(match_path, index=False)
        logger.info("Saved match probs (%d matches) → %s", len(match_df), match_path)

        return {"champion": str(champion_path), "match": str(match_path)}

    # ------------------------------------------------------------------
    # Internal simulation helpers
    # ------------------------------------------------------------------

    def _predict_match_cached(
        self,
        home: str,
        away: str,
        elo_state: Dict[str, Any],
        base_features: Dict[str, Dict[str, float]],
        models: Dict[str, Any],
        rest_home: float = 7.0,
        rest_away: float = 7.0,
    ) -> Tuple[float, float, float, float, float]:
        """Get 1X2 probabilities and Poisson lambdas, using cache if available."""
        if not hasattr(self, "prediction_cache"):
            self.prediction_cache = {}

        ratings = elo_state["ratings"]
        home_elo = ratings.get(home, 1500.0)
        away_elo = ratings.get(away, 1500.0)

        # Round ELO ratings to nearest integer to allow cache hits
        rounded_home_elo = int(round(home_elo))
        rounded_away_elo = int(round(away_elo))

        cache_key = (home, away, rounded_home_elo, rounded_away_elo, int(rest_home), int(rest_away))
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        # Compute features
        features = self._build_feature_vec(
            home, away, elo_state, base_features,
            rest_days_home=rest_home,
            rest_days_away=rest_away,
        )

        xgb_model = models["xgb"]
        poisson = models["poisson"]

        # Predict XGBoost 1X2
        xgb_probs = xgb_model.predict_proba(features)[0]
        if hasattr(xgb_model, "classes_"):
            classes = xgb_model.classes_
            prob_map = {int(c): xgb_probs[i] for i, c in enumerate(classes)}
            p_home = prob_map.get(1, xgb_probs[1] if len(xgb_probs) > 1 else xgb_probs[0])
            p_draw = prob_map.get(0, xgb_probs[0])
            p_away = prob_map.get(2, xgb_probs[2] if len(xgb_probs) > 2 else xgb_probs[-1])
        else:
            if len(xgb_probs) == 3:
                p_draw, p_home, p_away = xgb_probs
            else:
                p_home, p_draw, p_away = 1 / 3, 1 / 3, 1 / 3

        # Predict Poisson lambdas
        lambda_h, lambda_a = poisson.predict_lambdas(features)
        lambda_h_val = float(lambda_h.item() if hasattr(lambda_h, "item") else lambda_h[0])
        lambda_a_val = float(lambda_a.item() if hasattr(lambda_a, "item") else lambda_a[0])

        result = (p_home, p_draw, p_away, lambda_h_val, lambda_a_val)
        self.prediction_cache[cache_key] = result
        return result

    def _simulate_match(
        self,
        home: str,
        away: str,
        features: pd.DataFrame, # Unused, kept for signature compatibility
        models: Dict[str, Any],
        elo_state: Dict[str, Any],
        base_features: Dict[str, Dict[str, float]],
        rest_home: float = 7.0,
        rest_away: float = 7.0,
        initial_elo: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int]:
        """Simulate a single group match using XGBoost + Poisson with caching."""
        p_home, p_draw, p_away, lambda_h_val, lambda_a_val = self._predict_match_cached(
            home, away, initial_elo if initial_elo is not None else elo_state, base_features, models, rest_home, rest_away
        )
        poisson = models["poisson"]
        rho = poisson.params_[-1] if (poisson is not None and poisson.params_ is not None) else 0.0

        # Sample outcome from XGBoost
        outcome = self.rng.choice([1, 0, 2], p=[p_home, p_draw, p_away])

        # Sample exact score from Poisson conditional on outcome
        home_goals, away_goals = self._sample_conditional_score(
            lambda_h_val, lambda_a_val, rho, "home" if outcome == 1 else ("away" if outcome == 2 else "draw")
        )

        # Path-dependent ELO update
        self._update_elo(home if home_goals > away_goals else away,
                         away if home_goals > away_goals else home,
                         home_goals, away_goals, home, away, elo_state)

        return home_goals, away_goals

    def _resolve_match_deterministic(
        self,
        home: str,
        away: str,
        models: Dict[str, Any],
        elo_state: Dict[str, Any],
        base_features: Dict[str, Dict[str, float]],
        rest_home: float = 7.0,
        rest_away: float = 7.0,
        initial_elo: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int]:
        """Resolve a match deterministically by choosing the most likely score."""
        p_home, p_draw, p_away, lambda_h_val, lambda_a_val = self._predict_match_cached(
            home, away, initial_elo if initial_elo is not None else elo_state, base_features, models, rest_home, rest_away
        )
        poisson = models["poisson"]
        rho = poisson.params_[-1] if (poisson is not None and poisson.params_ is not None) else 0.0

        # Choose highest probability outcome
        outcome_idx = np.argmax([p_draw, p_home, p_away])
        outcome = [0, 1, 2][outcome_idx]  # 0: draw, 1: home win, 2: away win

        condition = "home" if outcome == 1 else ("away" if outcome == 2 else "draw")
        
        # Get the most likely score under this condition
        if not hasattr(self, "poisson_evaluator"):
            from models.poisson_model import DixonColesPoisson
            self.poisson_evaluator = DixonColesPoisson(max_goals=self.max_goals)
            self.conditional_probs_cache = {}

        cache_key = (round(lambda_h_val, 3), round(lambda_a_val, 3), round(rho, 3), condition)
        if cache_key in self.conditional_probs_cache:
            flat = self.conditional_probs_cache[cache_key]
        else:
            score_matrix = self.poisson_evaluator.exact_score_prob(lambda_h_val, lambda_a_val, rho)

            if condition == "home":
                mask = np.triu(score_matrix, k=1).T  # home > away
            elif condition == "away":
                mask = np.tril(score_matrix, k=-1)   # home < away
            else:
                mask = np.eye(score_matrix.shape[0], dtype=bool)

            conditional = score_matrix * mask
            total = conditional.sum()
            if total == 0:
                conditional = score_matrix

            flat = conditional.ravel()
            flat = np.maximum(flat, 0)
            flat /= flat.sum()
            self.conditional_probs_cache[cache_key] = flat

        # Deterministic choice: highest probability score index
        idx = np.argmax(flat)
        max_g = self.max_goals
        return int(idx // (max_g + 1)), int(idx % (max_g + 1))

    def _simulate_ko_pair(
        self,
        team_a: str,
        team_b: str,
        models: Dict[str, Any],
        elo_state: Dict[str, Any],
        base_features: Dict[str, Dict[str, float]],
    ) -> Tuple[str, str]:
        """Simulate a single KO match between two teams.

        Returns (winner, loser).
        """
        # Randomise who is home (neutral venue)
        if self.rng.random() < 0.5:
            home, away = team_a, team_b
        else:
            home, away = team_b, team_a

        winner, loser, _, _, _ = self.simulate_ko_match(
            home, away, models, elo_state, base_features,
            deterministic=self.closest_only
        )
        return winner, loser

    def _simulate_ko_round(
        self,
        pairings: List[Tuple[str, str]],
        round_name: str,
        models: Dict[str, Any],
        elo_state: Dict[str, Any],
        base_features: Dict[str, Dict[str, float]],
        feeders: bool = False,
    ) -> List[Dict[str, Any]]:
        """Simulate all matches in one KO round.

        Parameters
        ----------
        pairings : list[tuple[str, str]]
            List of (team_a, team_b) pairings.
        round_name : str
            Round name (for logging).
        models : dict
            Loaded models.
        elo_state : dict
            Mutable ELO state.
        base_features : dict
            Pre-tournament features.
        feeders : bool
            If True, some entries may be "feeder" placeholders.

        Returns
        -------
        list[dict]
            Match results with "winner", "loser", "home", "away", "extra" fields.
        """
        results = []
        for team_a, team_b in pairings:
            if feeders and (team_a == "feeder" or team_b == "feeder"):
                # Placeholder — will be resolved by higher round
                results.append({"winner": team_a if team_b == "feeder" else team_b,
                                "loser": "feeder"})
                continue

            winner, loser = self._simulate_ko_pair(
                team_a, team_b, models, elo_state, base_features
            )
            results.append({"winner": winner, "loser": loser})

        return results

    def _sample_conditional_score(
        self,
        lambda_h: float,
        lambda_a: float,
        rho: float,
        condition: str,
    ) -> Tuple[int, int]:
        """Sample exact score from Poisson conditional on match outcome with caching."""
        if not hasattr(self, "poisson_evaluator"):
            from models.poisson_model import DixonColesPoisson
            self.poisson_evaluator = DixonColesPoisson(max_goals=self.max_goals)
            self.conditional_probs_cache = {}

        cache_key = (round(lambda_h, 3), round(lambda_a, 3), round(rho, 3), condition)
        if cache_key in self.conditional_probs_cache:
            flat = self.conditional_probs_cache[cache_key]
        else:
            score_matrix = self.poisson_evaluator.exact_score_prob(lambda_h, lambda_a, rho)

            if condition == "home":
                mask = np.triu(score_matrix, k=1).T  # home > away
            elif condition == "away":
                mask = np.tril(score_matrix, k=-1)   # home < away
            else:
                mask = np.eye(score_matrix.shape[0], dtype=bool)

            conditional = score_matrix * mask
            total = conditional.sum()
            if total == 0:
                conditional = score_matrix
                total = conditional.sum()

            flat = conditional.ravel()
            flat = np.maximum(flat, 0)
            flat /= flat.sum()
            self.conditional_probs_cache[cache_key] = flat

        idx = self.rng.choice(len(flat), p=flat)
        max_g = self.max_goals
        return int(idx // (max_g + 1)), int(idx % (max_g + 1))

    def _build_feature_vec(
        self,
        home: str,
        away: str,
        elo_state: Dict[str, Any],
        base_features: Dict[str, Dict[str, float]],
        rest_days_home: int = 7,
        rest_days_away: int = 7,
    ) -> pd.DataFrame:
        """Build a feature vector for a single match.

        Uses current ELO ratings (path-dependent) and pre-tournament form
        features (static per simulation).

        Parameters
        ----------
        home : str
            Home team name.
        away : str
            Away team name.
        elo_state : dict
            Current ELO state with ``ratings``.
        base_features : dict
            Pre-tournament features per team.
        rest_days_home : int
            Rest days for home team.
        rest_days_away : int
            Rest days for away team.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with 16 feature columns.
        """
        ratings = elo_state["ratings"]
        elo_diff = ratings.get(home, 1500) - ratings.get(away, 1500)

        hf = base_features.get(home, {})
        af = base_features.get(away, {})

        row = {
            "elo_diff": elo_diff,
            "form_home_5f": hf.get("form_home_5f", 0.0),
            "form_home_5a": hf.get("form_home_5a", 0.0),
            "form_away_5f": af.get("form_away_5f", 0.0),
            "form_away_5a": af.get("form_away_5a", 0.0),
            "form_home_10f": hf.get("form_home_10f", 0.0),
            "form_home_10a": hf.get("form_home_10a", 0.0),
            "form_away_10f": af.get("form_away_10f", 0.0),
            "form_away_10a": af.get("form_away_10a", 0.0),
            "h2h_avg_diff": 0.0,  # simplified: no H2H in tournament
            "home_advantage": 0.0,  # neutral venue
            "rest_days_home": float(rest_days_home),
            "rest_days_away": float(rest_days_away),
            "implied_home": 1.0 / 3.0,
            "implied_draw": 1.0 / 3.0,
            "implied_away": 1.0 / 3.0,
        }
        return pd.DataFrame([row])

    def _update_elo(
        self,
        winner: str,
        loser: str,
        home_goals: int,
        away_goals: int,
        home_team: str,
        away_team: str,
        elo_state: Dict[str, Any],
    ) -> None:
        """Update ELO ratings after one match (disabled for speed and stability)."""
        pass


# ------------------------------------------------------------------
# Helper: build base features from feature store
# ------------------------------------------------------------------
def build_base_features(
    feature_store_path: str | Path,
    elo_state: Dict[str, Any],
    team_list: List[str],
) -> Dict[str, Dict[str, float]]:
    """Extract pre-tournament feature snapshots for each team.

    Takes the last known feature values from the feature store for
    each team in the tournament.

    Parameters
    ----------
    feature_store_path : str | Path
        Path to ``feature_store.csv``.
    elo_state : dict
        Current ELO state (for rating lookups).
    team_list : list[str]
        Teams participating in the tournament.

    Returns
    -------
    dict
        ``{team: {"form_home_5f": ..., "form_home_5a": ..., ...,
                   "rest_days": 7}}``
    """
    fs = pd.read_csv(feature_store_path, parse_dates=["date"])

    base: Dict[str, Dict[str, float]] = {}
    for team in team_list:
        base[team] = {
            "form_home_5f": 0.0,
            "form_home_5a": 0.0,
            "form_away_5f": 0.0,
            "form_away_5a": 0.0,
            "form_home_10f": 0.0,
            "form_home_10a": 0.0,
            "form_away_10f": 0.0,
            "form_away_10a": 0.0,
            "rest_days": 7,
        }

    # Find last match for each team
    for team in team_list:
        team_matches = fs[
            (fs["home_team"] == team) | (fs["away_team"] == team)
        ].copy()
        if team_matches.empty:
            continue

        # Use last match's form features
        last = team_matches.iloc[-1]

        if last["home_team"] == team:
            base[team]["form_home_5f"] = float(last.get("form_home_5f", 0))
            base[team]["form_home_5a"] = float(last.get("form_home_5a", 0))
            base[team]["form_home_10f"] = float(last.get("form_home_10f", 0))
            base[team]["form_home_10a"] = float(last.get("form_home_10a", 0))
        else:
            base[team]["form_home_5f"] = float(last.get("form_away_5f", 0))
            base[team]["form_home_5a"] = float(last.get("form_away_5a", 0))
            base[team]["form_home_10f"] = float(last.get("form_away_10f", 0))
            base[team]["form_home_10a"] = float(last.get("form_away_10a", 0))

    return base


# ===================================================================
# CLI entry point
# ===================================================================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Run Monte Carlo tournament simulation"
    )
    parser.add_argument(
        "--fixture",
        default="data/raw/fixture_2026.csv",
        help="Path to fixture_2026.csv",
    )
    parser.add_argument(
        "--features",
        default="data/processed/feature_store.csv",
        help="Path to feature_store.csv",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory with XGBoost model files",
    )
    parser.add_argument(
        "--poisson-params",
        default="models/poisson_params.json",
        help="Path to poisson_params.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=1000,
        help="Number of tournament simulations (default 1000, production 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--closest-only",
        action="store_true",
        help="Simulate only closest matches probabilistically and others deterministically for speed",
    )
    args = parser.parse_args()

    # Load models
    from models.xgboost_model import XGBoostModel
    from models.poisson_model import DixonColesPoisson

    logger.info("Loading XGBoost model from %s …", args.model_dir)
    xgb_model = XGBoostModel.load(args.model_dir)

    logger.info("Loading Poisson params from %s …", args.poisson_params)
    poisson = DixonColesPoisson.load(args.poisson_params)

    models = {"xgb": xgb_model, "poisson": poisson}

    # Load fixture
    logger.info("Loading fixture from %s …", args.fixture)
    fixture = MonteCarloSimulator.load_fixture(args.fixture)

    # Get all teams from fixture
    all_teams = list(
        set(fixture["home_team"].unique()) | set(fixture["away_team"].unique())
    )
    logger.info("Tournament teams: %d", len(all_teams))

    # Build initial ELO state
    # Load the ELO engine's final state
    from models.elo import EloEngine

    logger.info("Building ELO state from feature store …")
    engine = EloEngine()

    # Rebuild ELO state by processing clean_matches
    clean_matches = pd.read_csv(
        "data/processed/clean_matches.csv",
        parse_dates=["date"],
    )
    engine.process_matches(clean_matches)

    initial_elo = {
        "ratings": dict(engine.ratings),
        "match_counts": dict(engine.match_counts),
    }

    # Build base features for tournament teams
    base_features = build_base_features(
        args.features, initial_elo, all_teams
    )

    # Run simulation
    simulator = MonteCarloSimulator(random_state=args.seed, closest_only=args.closest_only)
    champion_df, match_df, all_results = simulator.simulate_tournament(
        fixture, models, initial_elo, base_features,
        n_sims=args.n_sims, verbose=True,
    )

    # Convergence check
    convergence = simulator.convergence_check(champion_df)
    logger.info("Convergence: %s", convergence["message"])

    if convergence.get("is_converged", False):
        print(f"\nConverged after {convergence['n_sims']} simulations.")
    else:
        print(f"\n{convergence['message']}")

    # Print top champions
    print(f"\nTop 10 champion probabilities ({args.n_sims} sims):")
    for _, row in champion_df.head(10).iterrows():
        print(f"  {row['team']:25s} {row['champion_pct']:5.1f}%")

    # Save results
    paths = simulator.save_results(champion_df, match_df, args.output_dir)
    print(f"\nResults saved:")
    print(f"  Champion probs: {paths['champion']}")
    print(f"  Match probs:    {paths['match']}")
