"""
run_pipeline.py
---------------
Explicit orchestrator for the end-to-end local data pipeline.

Run full pipeline:
  python -m pipeline.run_pipeline

Run without API fetching (rebuild from existing cached files):
  python -m pipeline.run_pipeline --skip-fetch

Include survival model training:
  python -m pipeline.run_pipeline --with-model
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

from pipeline.features.availability import compute_availability_features
from pipeline.features.build_features import build_model_features
from pipeline.features.physicality import compute_physicality_features
from pipeline.features.player_impact import compute_player_impact
from pipeline.features.team_stats import compute_team_stats_features
from pipeline.ingestion.fetch_games import fetch_all_seasons
from pipeline.ingestion.fetch_players import fetch_all_player_logs
from pipeline.ingestion.fetch_series import fetch_all_series
from pipeline.ingestion.load_db import load_to_duckdb
from pipeline.ingestion.validate import _print_report, run_validation
from pipeline.models.historical_scores import compute_daily_model_scores
from pipeline.models.matchup_model import train_matchup_model
from pipeline.models.predict_current import predict_current_season
from pipeline.models.sanity_report import generate_sanity_report
from pipeline.models.simulation import main as run_simulation
from pipeline.models.survival import train_survival_model
from config.settings import DB_PATH


@dataclass
class Stage:
    name: str
    fn: Callable[[], object]


def _run_stage(stage: Stage) -> None:
    print(f"\n=== Stage: {stage.name} ===")
    start = time.time()
    stage.fn()
    elapsed = time.time() - start
    print(f"=== Done: {stage.name} ({elapsed:.1f}s) ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transparent NBA playoff pipeline stages.")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip API fetch stages and use existing cached files/tables.",
    )
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Train survival and matchup models after feature build.",
    )
    parser.add_argument(
        "--with-current-projections",
        action="store_true",
        help="Generate current-season projections + Monte Carlo simulation outputs for the app.",
    )
    parser.add_argument(
        "--with-remaining-schedule",
        action="store_true",
        help="Compute per-game win probs + Monte Carlo seed simulation for remaining RS games.",
    )
    parser.add_argument(
        "--with-sanity-report",
        action="store_true",
        help="Generate model sanity report artifacts (requires simulation outputs).",
    )
    parser.add_argument(
        "--strict-validate",
        action="store_true",
        help="Fail validation on warnings as well as errors.",
    )
    args = parser.parse_args()

    stages: list[Stage] = []
    if not args.skip_fetch:
        stages.extend(
            [
                Stage("Fetch game logs", fetch_all_seasons),
                Stage("Fetch player logs", fetch_all_player_logs),
            ]
        )

    stages.extend(
        [
            Stage("Load game logs into DuckDB", load_to_duckdb),
            Stage("Build playoff series summary", fetch_all_series),
            Stage("Compute physicality features", compute_physicality_features),
            Stage("Compute team stats features", compute_team_stats_features),
            Stage("Compute availability features", compute_availability_features),
            Stage("Build model features", build_model_features),
        ]
    )

    for stage in stages:
        _run_stage(stage)

    print("\n=== Stage: Validate pipeline outputs ===")
    passed, issues = run_validation(strict=args.strict_validate)
    _print_report(passed, issues, args.strict_validate)
    if not passed:
        raise SystemExit(1)

    if args.with_model:
        _run_stage(Stage("Train survival model", train_survival_model))
        _run_stage(Stage("Train matchup model", train_matchup_model))

    if args.with_current_projections:
        _run_stage(Stage("Predict current season field", predict_current_season))

        # Check for live playoff data — run conditional MC if playoffs have started
        from pipeline.ingestion.fetch_playoff_status import get_playoff_series_status
        live_bracket = get_playoff_series_status()
        if not live_bracket.empty:
            completed = live_bracket[live_bracket["series_status"] == "completed"]
            print(f"  Playoff mode: {len(live_bracket)} series found, {len(completed)} completed")
            from pipeline.models.simulation import (
                build_locked_results_from_live,
                run_conditional_monte_carlo,
                write_outputs as write_sim_outputs,
            )
            locked = build_locked_results_from_live(live_bracket)
            results = run_conditional_monte_carlo(locked_results=locked, live_bracket=live_bracket)
            write_sim_outputs(*results)

            # Also persist live bracket to DuckDB
            import duckdb
            con = duckdb.connect(DB_PATH)
            try:
                con.execute("DROP TABLE IF EXISTS playoff_bracket_live")
                con.execute("CREATE TABLE playoff_bracket_live AS SELECT * FROM live_bracket")
            finally:
                con.close()
        else:
            _run_stage(Stage("Run Monte Carlo simulation", run_simulation))

        _run_stage(Stage("Compute daily historical model scores", compute_daily_model_scores))

        def _run_player_impact() -> None:
            import duckdb
            con = duckdb.connect(DB_PATH)
            try:
                compute_player_impact(con)
            finally:
                con.close()

        _run_stage(Stage("Compute player impact splits", _run_player_impact))

    if args.with_remaining_schedule:
        from pipeline.models.remaining_schedule import build_remaining_schedule
        _run_stage(Stage("Build remaining schedule projections", build_remaining_schedule))

    if args.with_sanity_report:
        _run_stage(Stage("Generate model sanity report", generate_sanity_report))

    print("\nPipeline run complete.")


if __name__ == "__main__":
    main()
