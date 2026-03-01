"""
predict_current.py
------------------
Generate current-season predictions with standings and play-in simulation.

Play-in rule regimes:
  - pre-2019-20: no play-in
  - 2019-20: bubble conditional 8v9 (if within 4 games)
  - 2020-21+: standard 7/8 + 9/10 play-in

Outputs:
  - models/trained/current_season_predictions.csv
  - models/trained/projected_playoff_field.csv
  - models/trained/projected_first_round_matchups.csv
  - models/trained/play_in_simulation_results.csv
  - DuckDB tables:
      current_season_predictions
      projected_playoff_field
      projected_first_round_matchups
      play_in_simulation_results

Run:
  python -m pipeline.models.predict_current
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguestandingsv3

from config.settings import CURRENT_SEASON_STR, DB_PATH
from pipeline.models.survival import MODEL_PATH

OUTPUT_DIR = Path("models/trained")
OUTPUT_PREDICTIONS = OUTPUT_DIR / "current_season_predictions.csv"
OUTPUT_PLAYOFF_FIELD = OUTPUT_DIR / "projected_playoff_field.csv"
OUTPUT_MATCHUPS = OUTPUT_DIR / "projected_first_round_matchups.csv"
OUTPUT_PLAYIN = OUTPUT_DIR / "play_in_simulation_results.csv"
OUTPUT_FEATURE_SNAPSHOT = OUTPUT_DIR / "current_feature_snapshot.csv"

CURRENT_RS_FEATURES = [
    "rs_vs_top_teams_win_pct",
    "rs_net_rating",
    "rs_close_game_win_pct",
    "rs_fta",
    "rs_efg_pct",
]

N_PLAYIN_SIMS = 20000
RNG_SEED = 42
HOME_COURT_EDGE = 0.10


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _season_start_year(season: str) -> int:
    return int(str(season).split("-")[0])


def _play_in_format_for_season(season: str) -> str:
    year = _season_start_year(season)
    if year <= 2018:
        return "none"
    if year == 2019:
        return "bubble_conditional"
    return "standard_7_10"


def _load_model_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python -m pipeline.models.survival` first."
        )
    return joblib.load(MODEL_PATH)


def _load_team_lookup(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    lookup = con.execute("SELECT id AS TEAM_ID, abbreviation AS TEAM_ABBR, full_name FROM teams").df()
    if lookup.empty:
        raise ValueError("teams table is empty; run ingestion.load_db first.")
    return lookup


def _get_current_standings(team_lookup: pd.DataFrame) -> pd.DataFrame:
    standings = leaguestandingsv3.LeagueStandingsV3(season=CURRENT_SEASON_STR).get_data_frames()[0]

    keep = standings[
        ["TeamID", "Conference", "PlayoffRank", "WINS", "LOSSES", "WinPCT", "Record"]
    ].copy()
    keep = keep.rename(columns={"TeamID": "TEAM_ID", "WINS": "wins", "LOSSES": "losses", "WinPCT": "win_pct"})

    keep["playoff_rank"] = pd.to_numeric(keep["PlayoffRank"], errors="coerce")
    keep = keep.drop(columns=["PlayoffRank"])

    merged = keep.merge(team_lookup[["TEAM_ID", "TEAM_ABBR"]], on="TEAM_ID", how="left")
    missing_abbr = merged["TEAM_ABBR"].isna().sum()
    if missing_abbr:
        raise ValueError(f"Could not map TEAM_ID->TEAM_ABBR for {missing_abbr} teams from standings.")

    merged["conference"] = merged["Conference"].str.title()
    merged = merged.drop(columns=["Conference"])

    merged["in_playoff_top6"] = merged["playoff_rank"] <= 6
    merged["in_play_in"] = merged["playoff_rank"].between(7, 10, inclusive="both")
    merged["in_top8_by_standings"] = merged["playoff_rank"] <= 8

    return merged


def _current_regular_season_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rs = con.execute(
        """
        WITH current_rs AS (
            SELECT *
            FROM regular_season
            WHERE SEASON = ?
              AND TRY_CAST(FGA AS DOUBLE) > 0
        ),
        team_records AS (
            SELECT
                TEAM_ABBR,
                AVG(CASE WHEN WL = 'W' THEN 1.0 ELSE 0.0 END) AS win_pct
            FROM current_rs
            GROUP BY TEAM_ABBR
        ),
        top_teams AS (
            SELECT TEAM_ABBR
            FROM team_records
            WHERE win_pct >= 0.600
        )
        SELECT
            g.TEAM_ID,
            g.TEAM_ABBR,
            g.SEASON,
            AVG(CAST(g.FTA AS DOUBLE)) AS rs_fta,
            (AVG(CAST(g.FGM AS DOUBLE)) + 0.5 * AVG(CAST(g.FG3M AS DOUBLE)))
                / NULLIF(AVG(CAST(g.FGA AS DOUBLE)), 0) AS rs_efg_pct,
            AVG(CASE
                WHEN ABS(CAST(g.PLUS_MINUS AS DOUBLE)) <= 5
                THEN CASE WHEN g.WL = 'W' THEN 1.0 ELSE 0.0 END
                ELSE NULL
            END) AS rs_close_game_win_pct,
            AVG(CASE WHEN g.WL = 'W' THEN 1.0 ELSE 0.0 END) AS rs_win_pct,
            (
                SUM(CAST(g.PTS AS DOUBLE)) / NULLIF(
                    SUM(
                        CAST(g.FGA AS DOUBLE)
                        + 0.44 * CAST(g.FTA AS DOUBLE)
                        - CAST(g.OREB AS DOUBLE)
                        + CAST(g.TOV AS DOUBLE)
                    ), 0
                ) * 100
            ) - (
                SUM(CAST(g.PTS AS DOUBLE) - CAST(g.PLUS_MINUS AS DOUBLE)) / NULLIF(
                    SUM(
                        CAST(g.FGA AS DOUBLE)
                        + 0.44 * CAST(g.FTA AS DOUBLE)
                        - CAST(g.OREB AS DOUBLE)
                        + CAST(g.TOV AS DOUBLE)
                    ), 0
                ) * 100
            ) AS rs_net_rating,
            AVG(
                CASE
                    WHEN t.TEAM_ABBR IS NOT NULL
                    THEN CASE WHEN g.WL = 'W' THEN 1.0 ELSE 0.0 END
                    ELSE NULL
                END
            ) AS rs_vs_top_teams_win_pct,
            COUNT(CASE WHEN t.TEAM_ABBR IS NOT NULL THEN 1 ELSE NULL END) AS rs_vs_top_teams_games
        FROM current_rs g
        LEFT JOIN top_teams t
            ON RIGHT(g.MATCHUP, 3) = t.TEAM_ABBR
        GROUP BY g.TEAM_ID, g.TEAM_ABBR, g.SEASON
        ORDER BY g.TEAM_ABBR
        """,
        [CURRENT_SEASON_STR],
    ).df()

    if rs.empty:
        raise ValueError(f"No regular-season rows found for current season {CURRENT_SEASON_STR}.")

    return rs


def _game_win_prob(high_team: str, low_team: str, score_map: dict[str, float]) -> float:
    diff = float(score_map[high_team] - score_map[low_team] + HOME_COURT_EDGE)
    return float(_sigmoid(np.array([diff]))[0])


def _games_back_between(seed_a: pd.Series, seed_b: pd.Series) -> float:
    """Games behind of seed_b relative to seed_a, using W/L records."""
    return ((seed_a["wins"] - seed_b["wins"]) + (seed_b["losses"] - seed_a["losses"])) / 2.0


def _simulate_standard_playin(conf_df: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, dict]:
    seed_map = {int(r.playoff_rank): r for r in conf_df.itertuples(index=False)}
    required = [7, 8, 9, 10]
    if not all(s in seed_map for s in required):
        raise ValueError(f"Missing one of play-in seeds {required} for {conf_df['conference'].iloc[0]}.")

    t7 = seed_map[7].TEAM_ABBR
    t8 = seed_map[8].TEAM_ABBR
    t9 = seed_map[9].TEAM_ABBR
    t10 = seed_map[10].TEAM_ABBR

    score_map = {row.TEAM_ABBR: float(row.pred_survival_score) for row in conf_df.itertuples(index=False)}

    p_7v8 = _game_win_prob(t7, t8, score_map)
    p_9v10 = _game_win_prob(t9, t10, score_map)
    p_8v9 = _game_win_prob(t8, t9, score_map)
    p_8v10 = _game_win_prob(t8, t10, score_map)
    p_7v9 = _game_win_prob(t7, t9, score_map)
    p_7v10 = _game_win_prob(t7, t10, score_map)

    seed7_counts = defaultdict(int)
    seed8_counts = defaultdict(int)
    pair_counts = defaultdict(int)

    for _ in range(N_PLAYIN_SIMS):
        game_a_winner = t7 if rng.random() < p_7v8 else t8
        game_b_winner = t9 if rng.random() < p_9v10 else t10

        if game_a_winner == t7:
            slot7 = t7
            final_high = t8
            final_low = game_b_winner
            p_final = p_8v9 if game_b_winner == t9 else p_8v10
        else:
            slot7 = t8
            final_high = t7
            final_low = game_b_winner
            p_final = p_7v9 if game_b_winner == t9 else p_7v10

        final_winner = final_high if rng.random() < p_final else final_low
        slot8 = final_winner

        seed7_counts[slot7] += 1
        seed8_counts[slot8] += 1
        pair_counts[(slot7, slot8)] += 1

    seed7_probs = {k: v / N_PLAYIN_SIMS for k, v in seed7_counts.items()}
    seed8_probs = {k: v / N_PLAYIN_SIMS for k, v in seed8_counts.items()}
    pair_probs = {k: v / N_PLAYIN_SIMS for k, v in pair_counts.items()}

    best_pair, best_pair_prob = max(pair_probs.items(), key=lambda x: x[1])

    summary = {
        "selected_seed7_team": best_pair[0],
        "selected_seed8_team": best_pair[1],
        "selected_pair_prob": float(best_pair_prob),
        "seed7_probs": seed7_probs,
        "seed8_probs": seed8_probs,
    }

    game_probs = {
        "p_7v8": p_7v8,
        "p_9v10": p_9v10,
        "p_8v9": p_8v9,
        "p_8v10": p_8v10,
        "p_7v9": p_7v9,
        "p_7v10": p_7v10,
    }

    return summary, game_probs


def _simulate_bubble_playin(conf_df: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, dict]:
    seed_map = {int(r.playoff_rank): r for r in conf_df.itertuples(index=False)}
    required = [8, 9]
    if not all(s in seed_map for s in required):
        raise ValueError(f"Missing one of bubble seeds {required} for {conf_df['conference'].iloc[0]}.")

    t8 = seed_map[8].TEAM_ABBR
    t9 = seed_map[9].TEAM_ABBR

    gb_9_to_8 = _games_back_between(pd.Series(seed_map[8]._asdict()), pd.Series(seed_map[9]._asdict()))
    trigger = gb_9_to_8 <= 4.0

    score_map = {row.TEAM_ABBR: float(row.pred_survival_score) for row in conf_df.itertuples(index=False)}
    p_8v9 = _game_win_prob(t8, t9, score_map)

    seed8_counts = defaultdict(int)

    if not trigger:
        seed8_counts[t8] = N_PLAYIN_SIMS
    else:
        for _ in range(N_PLAYIN_SIMS):
            game1_winner = t8 if rng.random() < p_8v9 else t9
            if game1_winner == t8:
                seed8_counts[t8] += 1
            else:
                game2_winner = t8 if rng.random() < p_8v9 else t9
                seed8_counts[game2_winner] += 1

    seed8_probs = {k: v / N_PLAYIN_SIMS for k, v in seed8_counts.items()}
    selected_seed8_team, selected_prob = max(seed8_probs.items(), key=lambda x: x[1])

    summary = {
        "selected_seed7_team": None,
        "selected_seed8_team": selected_seed8_team,
        "selected_pair_prob": float(selected_prob),
        "seed7_probs": {},
        "seed8_probs": seed8_probs,
        "bubble_triggered": bool(trigger),
        "gb_9_to_8": float(gb_9_to_8),
    }
    game_probs = {"p_8v9": p_8v9}

    return summary, game_probs


def _project_playoff_field(predictions: pd.DataFrame, play_in_format: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    projected_rows: list[dict] = []
    playin_rows: list[dict] = []

    rng = np.random.default_rng(RNG_SEED)

    for conf in ["East", "West"]:
        conf_df = predictions[predictions["conference"] == conf].copy()
        conf_df = conf_df.sort_values(["playoff_rank", "pred_rank_all_30"])
        seed_map = {int(r.playoff_rank): r for r in conf_df.itertuples(index=False)}

        if play_in_format == "none":
            for seed in range(1, 9):
                if seed not in seed_map:
                    continue
                team = seed_map[seed]
                projected_rows.append(
                    {
                        "season": CURRENT_SEASON_STR,
                        "conference": conf,
                        "playoff_seed": seed,
                        "TEAM_ID": int(team.TEAM_ID),
                        "TEAM_ABBR": team.TEAM_ABBR,
                        "seed_assignment_prob": 1.0,
                        "selection_method": "direct_top8",
                        "play_in_format": play_in_format,
                    }
                )

        elif play_in_format == "bubble_conditional":
            # Seeds 1-7 lock.
            for seed in range(1, 8):
                if seed not in seed_map:
                    continue
                team = seed_map[seed]
                projected_rows.append(
                    {
                        "season": CURRENT_SEASON_STR,
                        "conference": conf,
                        "playoff_seed": seed,
                        "TEAM_ID": int(team.TEAM_ID),
                        "TEAM_ABBR": team.TEAM_ABBR,
                        "seed_assignment_prob": 1.0,
                        "selection_method": "direct_top7",
                        "play_in_format": play_in_format,
                    }
                )

            summary, game_probs = _simulate_bubble_playin(conf_df, rng)
            selected8 = summary["selected_seed8_team"]
            selected8_row = conf_df[conf_df["TEAM_ABBR"] == selected8].iloc[0]

            projected_rows.append(
                {
                    "season": CURRENT_SEASON_STR,
                    "conference": conf,
                    "playoff_seed": 8,
                    "TEAM_ID": int(selected8_row["TEAM_ID"]),
                    "TEAM_ABBR": selected8,
                    "seed_assignment_prob": float(summary["seed8_probs"].get(selected8, 0.0)),
                    "selection_method": "bubble_playin",
                    "play_in_format": play_in_format,
                }
            )

            for team, prob in summary["seed8_probs"].items():
                playin_rows.append(
                    {
                        "season": CURRENT_SEASON_STR,
                        "conference": conf,
                        "play_in_format": play_in_format,
                        "n_sims": N_PLAYIN_SIMS,
                        "team_abbr": team,
                        "seed7_prob": 0.0,
                        "seed8_prob": float(prob),
                        "made_playoffs_prob": float(prob),
                        "selected_seed7_team": None,
                        "selected_seed8_team": selected8,
                        "selected_pair_prob": float(summary["selected_pair_prob"]),
                        "bubble_triggered": bool(summary["bubble_triggered"]),
                        "gb_9_to_8": float(summary["gb_9_to_8"]),
                        "p_8v9": float(game_probs["p_8v9"]),
                    }
                )

        else:
            # standard_7_10: seeds 1-6 lock.
            for seed in range(1, 7):
                if seed not in seed_map:
                    continue
                team = seed_map[seed]
                projected_rows.append(
                    {
                        "season": CURRENT_SEASON_STR,
                        "conference": conf,
                        "playoff_seed": seed,
                        "TEAM_ID": int(team.TEAM_ID),
                        "TEAM_ABBR": team.TEAM_ABBR,
                        "seed_assignment_prob": 1.0,
                        "selection_method": "direct_top6",
                        "play_in_format": play_in_format,
                    }
                )

            summary, game_probs = _simulate_standard_playin(conf_df, rng)
            selected7 = summary["selected_seed7_team"]
            selected8 = summary["selected_seed8_team"]

            selected7_row = conf_df[conf_df["TEAM_ABBR"] == selected7].iloc[0]
            selected8_row = conf_df[conf_df["TEAM_ABBR"] == selected8].iloc[0]

            projected_rows.append(
                {
                    "season": CURRENT_SEASON_STR,
                    "conference": conf,
                    "playoff_seed": 7,
                    "TEAM_ID": int(selected7_row["TEAM_ID"]),
                    "TEAM_ABBR": selected7,
                    "seed_assignment_prob": float(summary["seed7_probs"].get(selected7, 0.0)),
                    "selection_method": "play_in_simulation",
                    "play_in_format": play_in_format,
                }
            )
            projected_rows.append(
                {
                    "season": CURRENT_SEASON_STR,
                    "conference": conf,
                    "playoff_seed": 8,
                    "TEAM_ID": int(selected8_row["TEAM_ID"]),
                    "TEAM_ABBR": selected8,
                    "seed_assignment_prob": float(summary["seed8_probs"].get(selected8, 0.0)),
                    "selection_method": "play_in_simulation",
                    "play_in_format": play_in_format,
                }
            )

            candidate_teams = sorted(set(summary["seed7_probs"]) | set(summary["seed8_probs"]))
            for team in candidate_teams:
                p7 = float(summary["seed7_probs"].get(team, 0.0))
                p8 = float(summary["seed8_probs"].get(team, 0.0))
                playin_rows.append(
                    {
                        "season": CURRENT_SEASON_STR,
                        "conference": conf,
                        "play_in_format": play_in_format,
                        "n_sims": N_PLAYIN_SIMS,
                        "team_abbr": team,
                        "seed7_prob": p7,
                        "seed8_prob": p8,
                        "made_playoffs_prob": p7 + p8,
                        "selected_seed7_team": selected7,
                        "selected_seed8_team": selected8,
                        "selected_pair_prob": float(summary["selected_pair_prob"]),
                        "bubble_triggered": None,
                        "gb_9_to_8": None,
                        "p_8v9": None,
                        **{k: float(v) for k, v in game_probs.items()},
                    }
                )

    projected = pd.DataFrame(projected_rows)
    playin = pd.DataFrame(playin_rows)

    return projected, playin


def _build_first_round_matchups(projected_field: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairings = [(1, 8), (2, 7), (3, 6), (4, 5)]

    score_map = {r.TEAM_ABBR: float(r.pred_survival_score) for r in predictions.itertuples(index=False)}

    for conf in ["East", "West"]:
        conf_field = projected_field[projected_field["conference"] == conf].copy()
        seed_map = {int(r.playoff_seed): r for r in conf_field.itertuples(index=False)}

        for high_seed, low_seed in pairings:
            if high_seed not in seed_map or low_seed not in seed_map:
                continue

            high = seed_map[high_seed]
            low = seed_map[low_seed]

            p_high = _game_win_prob(high.TEAM_ABBR, low.TEAM_ABBR, score_map)

            rows.append(
                {
                    "season": CURRENT_SEASON_STR,
                    "conference": conf,
                    "round": "First Round",
                    "high_seed": high_seed,
                    "high_seed_team": high.TEAM_ABBR,
                    "high_seed_seed_assignment_prob": float(high.seed_assignment_prob),
                    "low_seed": low_seed,
                    "low_seed_team": low.TEAM_ABBR,
                    "low_seed_seed_assignment_prob": float(low.seed_assignment_prob),
                    "high_seed_series_win_prob": p_high,
                    "low_seed_series_win_prob": 1.0 - p_high,
                }
            )

    return pd.DataFrame(rows)


def build_current_feature_frame() -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    artifact = _load_model_artifact()
    feature_cols: list[str] = artifact["feature_columns"]

    con = duckdb.connect(DB_PATH)
    try:
        team_lookup = _load_team_lookup(con)
        standings = _get_current_standings(team_lookup)
        current = _current_regular_season_features(con)
        for col in CURRENT_RS_FEATURES:
            if col not in current.columns:
                continue
            current[col] = pd.to_numeric(current[col], errors="coerce")
            median_val = current[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            current[col] = current[col].fillna(median_val)

        missing = [c for c in feature_cols if c not in current.columns]
        if missing:
            raise ValueError(f"Current feature frame missing required columns: {missing}")

        model_input = current[["TEAM_ID", "TEAM_ABBR", "SEASON"] + feature_cols].copy()
    finally:
        con.close()

    notes = [
        "features are regular-season only (no current-playoff-derived inputs)",
        "missing regular-season feature values are imputed with current-season median",
        "standings/playoff ranks sourced from nba_api LeagueStandingsV3",
    ]

    return model_input, feature_cols, notes, standings


def predict_current_season() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifact = _load_model_artifact()
    model = artifact["model"]
    feature_cols: list[str] = artifact["feature_columns"]

    features_df, required_cols, notes, standings = build_current_feature_frame()
    if required_cols != feature_cols:
        raise ValueError("Model feature column mismatch in prediction pipeline.")

    risk = model.predict_partial_hazard(features_df[feature_cols]).values.reshape(-1)

    out = features_df[["TEAM_ID", "TEAM_ABBR", "SEASON"]].copy()
    out["pred_elimination_risk"] = risk
    out["pred_survival_score"] = -risk

    exps = np.exp(out["pred_survival_score"] - out["pred_survival_score"].max())
    out["title_prob_proxy_all_30"] = exps / exps.sum()

    out = out.merge(
        standings[
            [
                "TEAM_ID",
                "conference",
                "playoff_rank",
                "wins",
                "losses",
                "win_pct",
                "Record",
                "in_playoff_top6",
                "in_play_in",
                "in_top8_by_standings",
            ]
        ],
        on="TEAM_ID",
        how="left",
    )

    out["pred_rank_all_30"] = out["pred_survival_score"].rank(method="min", ascending=False).astype(int)

    play_in_format = _play_in_format_for_season(CURRENT_SEASON_STR)
    projected_field, playin_results = _project_playoff_field(out, play_in_format)

    out = out.merge(
        projected_field[["conference", "TEAM_ABBR", "playoff_seed", "seed_assignment_prob"]],
        on=["conference", "TEAM_ABBR"],
        how="left",
    )
    out["in_projected_playoffs"] = out["playoff_seed"].notna()

    projected_field = projected_field.merge(
        out[
            [
                "TEAM_ID",
                "TEAM_ABBR",
                "Record",
                "wins",
                "losses",
                "win_pct",
                "pred_survival_score",
                "pred_rank_all_30",
                "title_prob_proxy_all_30",
            ]
        ],
        on=["TEAM_ID", "TEAM_ABBR"],
        how="left",
    )

    playoff_exps = np.exp(
        projected_field["pred_survival_score"] - projected_field["pred_survival_score"].max()
    )
    projected_field["title_prob_proxy_playoff_only"] = playoff_exps / playoff_exps.sum()

    projected_field = projected_field.sort_values(["conference", "playoff_seed"]).reset_index(drop=True)
    out = out.sort_values(["conference", "playoff_rank", "pred_rank_all_30"]).reset_index(drop=True)

    matchups = _build_first_round_matchups(projected_field, out)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_FEATURE_SNAPSHOT, index=False)
    out.to_csv(OUTPUT_PREDICTIONS, index=False)
    projected_field.to_csv(OUTPUT_PLAYOFF_FIELD, index=False)
    matchups.to_csv(OUTPUT_MATCHUPS, index=False)
    playin_results.to_csv(OUTPUT_PLAYIN, index=False)

    con = duckdb.connect(DB_PATH)
    try:
        con.execute("DROP TABLE IF EXISTS current_feature_snapshot")
        con.execute("CREATE TABLE current_feature_snapshot AS SELECT * FROM features_df")

        con.execute("DROP TABLE IF EXISTS current_season_predictions")
        con.execute("CREATE TABLE current_season_predictions AS SELECT * FROM out")

        con.execute("DROP TABLE IF EXISTS projected_playoff_field")
        con.execute("CREATE TABLE projected_playoff_field AS SELECT * FROM projected_field")

        con.execute("DROP TABLE IF EXISTS projected_first_round_matchups")
        con.execute("CREATE TABLE projected_first_round_matchups AS SELECT * FROM matchups")

        con.execute("DROP TABLE IF EXISTS play_in_simulation_results")
        con.execute("CREATE TABLE play_in_simulation_results AS SELECT * FROM playin_results")
    finally:
        con.close()

    print(f"Current season predictions generated for {len(out)} teams ({CURRENT_SEASON_STR}).")
    print(f"Play-in format used: {play_in_format}")
    print(f"Saved CSV: {OUTPUT_PREDICTIONS}")
    print(f"Saved current feature snapshot: {OUTPUT_FEATURE_SNAPSHOT}")
    print(f"Saved projected playoff field: {OUTPUT_PLAYOFF_FIELD}")
    print(f"Saved projected round-1 matchups: {OUTPUT_MATCHUPS}")
    print(f"Saved play-in simulation: {OUTPUT_PLAYIN}")
    print("Assumptions:")
    for note in notes:
        print(f"  - {note}")

    print("\nProjected playoff field (seeded):")
    print(
        projected_field[
            [
                "conference",
                "playoff_seed",
                "TEAM_ABBR",
                "Record",
                "seed_assignment_prob",
                "title_prob_proxy_playoff_only",
            ]
        ].to_string(index=False)
    )

    if not playin_results.empty:
        print("\nPlay-in simulation summary:")
        summary_cols = [
            "conference",
            "team_abbr",
            "seed7_prob",
            "seed8_prob",
            "made_playoffs_prob",
            "selected_seed7_team",
            "selected_seed8_team",
            "selected_pair_prob",
        ]
        keep_cols = [c for c in summary_cols if c in playin_results.columns]
        print(playin_results[keep_cols].sort_values(["conference", "made_playoffs_prob"], ascending=[True, False]).to_string(index=False))

    if not matchups.empty:
        print("\nProjected First-Round Matchups:")
        print(matchups.to_string(index=False))

    return out, projected_field, matchups, playin_results


if __name__ == "__main__":
    predict_current_season()
