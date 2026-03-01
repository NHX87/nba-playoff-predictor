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
from pipeline.features.team_stats import compute_team_stats_features
from pipeline.ingestion.fetch_games import fetch_all_seasons
from pipeline.ingestion.fetch_players import fetch_all_player_logs
from pipeline.ingestion.fetch_series import fetch_all_series
from pipeline.ingestion.load_db import load_to_duckdb
from pipeline.ingestion.validate import _print_report, run_validation
from pipeline.models.matchup_model import train_matchup_model
from pipeline.models.predict_current import predict_current_season
from pipeline.models.sanity_report import generate_sanity_report
from pipeline.models.simulation import main as run_simulation
from pipeline.models.survival import train_survival_model


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
        _run_stage(Stage("Run Monte Carlo simulation", run_simulation))

    if args.with_sanity_report:
        _run_stage(Stage("Generate model sanity report", generate_sanity_report))

    print("\nPipeline run complete.")


if __name__ == "__main__":
    main()
