"""
simulation.py
-------------
Run playoff bracket simulation using the current projected playoff field and
head-to-head matchup model.

Outputs:
  - models/trained/series_predictions_current.csv
  - models/trained/simulation_team_odds_current.csv
  - models/trained/simulation_metadata.json
  - DuckDB tables:
      series_predictions_current
      simulation_team_odds_current
      app_series_predictions_current
      app_title_odds_current
      app_playoff_field_current
      app_play_in_current

Run:
  python -m pipeline.models.simulation
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from config.settings import CURRENT_SEASON_STR, DB_PATH, MONTE_CARLO_RUNS
from pipeline.models.matchup_model import load_matchup_artifact, predict_matchup_prob

ARTIFACT_DIR = Path("models/trained")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SERIES_PATH = ARTIFACT_DIR / "series_predictions_current.csv"
ODDS_PATH = ARTIFACT_DIR / "simulation_team_odds_current.csv"
META_PATH = ARTIFACT_DIR / "simulation_metadata.json"

RNG_SEED = 42
PAIRINGS = [(1, 8), (2, 7), (3, 6), (4, 5)]


def _load_current_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(DB_PATH)
    try:
        tables = set(
            con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).df()["table_name"].tolist()
        )

        required = {"projected_playoff_field", "current_feature_snapshot"}
        missing = required - tables
        if missing:
            raise ValueError(
                "Missing required tables for simulation: "
                f"{sorted(missing)}. Run `python -m pipeline.models.predict_current` first."
            )

        field = con.execute(
            """
            SELECT *
            FROM projected_playoff_field
            WHERE season = ?
            ORDER BY conference, playoff_seed
            """,
            [CURRENT_SEASON_STR],
        ).df()

        features = con.execute(
            """
            SELECT *
            FROM current_feature_snapshot
            WHERE SEASON = ?
            """,
            [CURRENT_SEASON_STR],
        ).df()

        playin = con.execute(
            """
            SELECT *
            FROM play_in_simulation_results
            WHERE season = ?
            ORDER BY conference, made_playoffs_prob DESC
            """,
            [CURRENT_SEASON_STR],
        ).df() if "play_in_simulation_results" in tables else pd.DataFrame()
    finally:
        con.close()

    if field.empty:
        raise ValueError(f"No projected playoff field rows found for season {CURRENT_SEASON_STR}.")
    if features.empty:
        raise ValueError(f"No current feature snapshot found for season {CURRENT_SEASON_STR}.")

    return field, features, playin


def _build_pair_prob_map(
    teams: list[str],
    feature_df: pd.DataFrame,
    artifact: dict[str, Any],
) -> dict[tuple[str, str], float]:
    """
    Precompute P(team_a beats team_b) for all ordered team pairs.
    This significantly speeds up Monte Carlo loops.
    """
    pair_probs: dict[tuple[str, str], float] = {}
    for team_a in teams:
        for team_b in teams:
            if team_a == team_b:
                continue
            pair_probs[(team_a, team_b)] = predict_matchup_prob(
                team_a, team_b, feature_df, artifact=artifact
            )
    return pair_probs


def _winner_from_prob(team_a: str, team_b: str, p_a: float, rng: np.random.Generator) -> str:
    return team_a if rng.random() < p_a else team_b


def _build_seed_maps(field_df: pd.DataFrame) -> dict[str, dict[int, str]]:
    conf_map: dict[str, dict[int, str]] = {}
    for conf in ["East", "West"]:
        conf_df = field_df[field_df["conference"] == conf]
        seed_map = {int(r.playoff_seed): str(r.TEAM_ABBR) for r in conf_df.itertuples(index=False)}
        if not all(seed in seed_map for seed in range(1, 9)):
            raise ValueError(f"Projected field for {conf} does not contain seeds 1-8.")
        conf_map[conf] = seed_map
    return conf_map


def _deterministic_series_predictions(
    seed_maps: dict[str, dict[int, str]],
    pair_probs: dict[tuple[str, str], float],
) -> pd.DataFrame:
    rows = []

    def series_row(conf: str, round_name: str, high_seed: int, low_seed: int, high_team: str, low_team: str) -> tuple[str, float]:
        p_high = pair_probs[(high_team, low_team)]
        winner = high_team if p_high >= 0.5 else low_team
        rows.append(
            {
                "season": CURRENT_SEASON_STR,
                "conference": conf,
                "round": round_name,
                "high_seed": high_seed,
                "high_team": high_team,
                "low_seed": low_seed,
                "low_team": low_team,
                "high_team_win_prob": p_high,
                "low_team_win_prob": 1.0 - p_high,
                "predicted_winner": winner,
            }
        )
        return winner, p_high

    east_r1 = {}
    west_r1 = {}
    for conf, out_map in [("East", east_r1), ("West", west_r1)]:
        s = seed_maps[conf]
        for high, low in PAIRINGS:
            w, _ = series_row(conf, "First Round", high, low, s[high], s[low])
            out_map[(high, low)] = w

    for conf, r1_map in [("East", east_r1), ("West", west_r1)]:
        sf1_high, sf1_low = r1_map[(1, 8)], r1_map[(4, 5)]
        sf2_high, sf2_low = r1_map[(2, 7)], r1_map[(3, 6)]
        cf1, _ = series_row(conf, "Conference Semifinals", 0, 0, sf1_high, sf1_low)
        cf2, _ = series_row(conf, "Conference Semifinals", 0, 0, sf2_high, sf2_low)
        conf_winner, _ = series_row(conf, "Conference Finals", 0, 0, cf1, cf2)
        r1_map[("conf_winner",)] = conf_winner

    finals_a = east_r1[("conf_winner",)]
    finals_b = west_r1[("conf_winner",)]
    p_a = pair_probs[(finals_a, finals_b)]
    champ = finals_a if p_a >= 0.5 else finals_b
    rows.append(
        {
            "season": CURRENT_SEASON_STR,
            "conference": "NBA",
            "round": "NBA Finals",
            "high_seed": 0,
            "high_team": finals_a,
            "low_seed": 0,
            "low_team": finals_b,
            "high_team_win_prob": p_a,
            "low_team_win_prob": 1.0 - p_a,
            "predicted_winner": champ,
        }
    )

    return pd.DataFrame(rows)


def run_monte_carlo(
    n_simulations: int = MONTE_CARLO_RUNS,
    rng_seed: int = RNG_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    field_df, feature_df, playin_df = _load_current_inputs()
    artifact = load_matchup_artifact()

    seed_maps = _build_seed_maps(field_df)
    teams = sorted(field_df["TEAM_ABBR"].tolist())
    pair_probs = _build_pair_prob_map(teams, feature_df, artifact)
    deterministic_series = _deterministic_series_predictions(seed_maps, pair_probs)

    make_second_round = Counter()
    make_conf_finals = Counter()
    make_finals = Counter()
    win_title = Counter()

    rng = np.random.default_rng(rng_seed)

    for _ in range(n_simulations):
        conf_winners = {}

        for conf in ["East", "West"]:
            s = seed_maps[conf]

            r1_winners = {}
            for high, low in PAIRINGS:
                t_high = s[high]
                t_low = s[low]
                p_high = pair_probs[(t_high, t_low)]
                winner = _winner_from_prob(t_high, t_low, p_high, rng)
                r1_winners[(high, low)] = winner
                make_second_round[winner] += 1

            sf1_a, sf1_b = r1_winners[(1, 8)], r1_winners[(4, 5)]
            sf2_a, sf2_b = r1_winners[(2, 7)], r1_winners[(3, 6)]

            p_sf1 = pair_probs[(sf1_a, sf1_b)]
            p_sf2 = pair_probs[(sf2_a, sf2_b)]

            cf_a = _winner_from_prob(sf1_a, sf1_b, p_sf1, rng)
            cf_b = _winner_from_prob(sf2_a, sf2_b, p_sf2, rng)
            make_conf_finals[cf_a] += 1
            make_conf_finals[cf_b] += 1

            p_cf = pair_probs[(cf_a, cf_b)]
            conf_champ = _winner_from_prob(cf_a, cf_b, p_cf, rng)
            make_finals[conf_champ] += 1
            conf_winners[conf] = conf_champ

        east = conf_winners["East"]
        west = conf_winners["West"]
        p_finals = pair_probs[(east, west)]
        champ = _winner_from_prob(east, west, p_finals, rng)
        win_title[champ] += 1

    odds_rows = []
    for r in field_df.itertuples(index=False):
        team = r.TEAM_ABBR
        odds_rows.append(
            {
                "season": CURRENT_SEASON_STR,
                "conference": r.conference,
                "playoff_seed": int(r.playoff_seed),
                "TEAM_ABBR": team,
                "Record": r.Record,
                "seed_assignment_prob": float(r.seed_assignment_prob),
                "make_second_round_prob": make_second_round[team] / n_simulations,
                "make_conf_finals_prob": make_conf_finals[team] / n_simulations,
                "make_finals_prob": make_finals[team] / n_simulations,
                "title_prob": win_title[team] / n_simulations,
            }
        )

    odds_df = pd.DataFrame(odds_rows).sort_values(
        ["title_prob", "make_finals_prob"], ascending=[False, False]
    ).reset_index(drop=True)

    meta = {
        "season": CURRENT_SEASON_STR,
        "n_simulations": int(n_simulations),
        "rng_seed": int(rng_seed),
        "source": "matchup_model + projected_playoff_field",
        "matchup_model_path": str(Path("models/trained/matchup_model.joblib")),
        "projected_field_rows": int(len(field_df)),
    }

    return deterministic_series, odds_df, meta, field_df, playin_df


def write_outputs(
    deterministic_series: pd.DataFrame,
    odds_df: pd.DataFrame,
    meta: dict[str, Any],
    field_df: pd.DataFrame,
    playin_df: pd.DataFrame,
) -> None:
    deterministic_series.to_csv(SERIES_PATH, index=False)
    odds_df.to_csv(ODDS_PATH, index=False)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    con = duckdb.connect(DB_PATH)
    try:
        con.execute("DROP TABLE IF EXISTS series_predictions_current")
        con.execute("CREATE TABLE series_predictions_current AS SELECT * FROM deterministic_series")

        con.execute("DROP TABLE IF EXISTS simulation_team_odds_current")
        con.execute("CREATE TABLE simulation_team_odds_current AS SELECT * FROM odds_df")

        # App-ready contract tables
        con.execute("DROP TABLE IF EXISTS app_series_predictions_current")
        con.execute("CREATE TABLE app_series_predictions_current AS SELECT * FROM deterministic_series")

        con.execute("DROP TABLE IF EXISTS app_title_odds_current")
        con.execute("CREATE TABLE app_title_odds_current AS SELECT * FROM odds_df")

        con.execute("DROP TABLE IF EXISTS app_playoff_field_current")
        con.execute("CREATE TABLE app_playoff_field_current AS SELECT * FROM field_df")

        con.execute("DROP TABLE IF EXISTS app_play_in_current")
        if playin_df.empty:
            con.execute(
                """
                CREATE TABLE app_play_in_current (
                    season VARCHAR,
                    conference VARCHAR,
                    play_in_format VARCHAR,
                    n_sims BIGINT,
                    team_abbr VARCHAR,
                    seed7_prob DOUBLE,
                    seed8_prob DOUBLE,
                    made_playoffs_prob DOUBLE
                )
                """
            )
        else:
            con.execute("CREATE TABLE app_play_in_current AS SELECT * FROM playin_df")
    finally:
        con.close()


def main() -> None:
    deterministic_series, odds_df, meta, field_df, playin_df = run_monte_carlo()
    write_outputs(deterministic_series, odds_df, meta, field_df, playin_df)

    print("Monte Carlo simulation complete")
    print(f"Saved series predictions: {SERIES_PATH}")
    print(f"Saved team odds: {ODDS_PATH}")
    print(f"Saved metadata: {META_PATH}")
    print("Top 10 title odds:")
    print(odds_df[["TEAM_ABBR", "playoff_seed", "title_prob", "make_finals_prob"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
