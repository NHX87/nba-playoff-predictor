"""
validate.py
-----------
Validate data completeness and key table health in DuckDB.

Run:
  python -m pipeline.ingestion.validate
  python -m pipeline.ingestion.validate --strict
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import duckdb

from config.settings import CURRENT_SEASON_STR, DB_PATH, TRAIN_SEASONS


@dataclass
class ValidationIssue:
    level: str  # ERROR or WARN
    check: str
    detail: str


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    rows = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        LIMIT 1
        """,
        [name],
    ).fetchall()
    return len(rows) > 0


def run_validation(strict: bool = False) -> tuple[bool, list[ValidationIssue]]:
    con = duckdb.connect(DB_PATH, read_only=True)
    issues: list[ValidationIssue] = []

    try:
        required_tables = ["raw_game_logs", "regular_season", "playoffs"]
        missing = [t for t in required_tables if not _table_exists(con, t)]
        if missing:
            for table in missing:
                issues.append(ValidationIssue("ERROR", "required_table", f"Missing table/view: {table}"))
            return False, issues

        rs_cov = con.execute(
            """
            SELECT SEASON, COUNT(DISTINCT TEAM_ID) AS teams
            FROM regular_season
            GROUP BY 1
            ORDER BY 1
            """
        ).df()

        po_cov = con.execute(
            """
            SELECT SEASON, COUNT(DISTINCT TEAM_ID) AS teams
            FROM playoffs
            GROUP BY 1
            ORDER BY 1
            """
        ).df()

        print("Regular season team coverage:")
        print(rs_cov.to_string(index=False))
        print("\nPlayoff team coverage:")
        print(po_cov.to_string(index=False))

        for _, row in rs_cov.iterrows():
            season = row["SEASON"]
            teams = int(row["teams"])
            if teams != 30:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "regular_season_team_count",
                        f"{season}: expected 30 teams, found {teams}",
                    )
                )

        train_set = set(TRAIN_SEASONS)
        for _, row in po_cov.iterrows():
            season = row["SEASON"]
            teams = int(row["teams"])
            if season in train_set and teams != 16:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "playoff_team_count",
                        f"{season}: expected 16 playoff teams, found {teams}",
                    )
                )
            elif season == CURRENT_SEASON_STR and teams not in (0, 16):
                issues.append(
                    ValidationIssue(
                        "WARN",
                        "playoff_team_count_current",
                        f"{season}: current season has {teams} playoff teams (likely in-progress season)",
                    )
                )

        # Optional downstream table checks for feature completeness.
        if _table_exists(con, "model_features"):
            feat_cov = con.execute(
                """
                SELECT SEASON, COUNT(*) AS teams
                FROM model_features
                GROUP BY 1
                ORDER BY 1
                """
            ).df()
            print("\nModel feature coverage:")
            print(feat_cov.to_string(index=False))

            po_map = {row["SEASON"]: int(row["teams"]) for _, row in po_cov.iterrows()}
            for _, row in feat_cov.iterrows():
                season = row["SEASON"]
                teams = int(row["teams"])
                expected = po_map.get(season)
                if expected is not None and teams != expected:
                    issues.append(
                        ValidationIssue(
                            "ERROR",
                            "model_features_team_count",
                            f"{season}: model_features has {teams} rows; playoffs has {expected} teams",
                        )
                    )

    finally:
        con.close()

    has_error = any(i.level == "ERROR" for i in issues)
    if strict:
        passed = not issues
    else:
        passed = not has_error

    return passed, issues


def _print_report(passed: bool, issues: list[ValidationIssue], strict: bool) -> None:
    mode = "strict" if strict else "default"
    print(f"\nValidation mode: {mode}")
    if not issues:
        print("Validation passed: no issues found.")
        return

    for issue in issues:
        print(f"[{issue.level}] {issue.check}: {issue.detail}")

    if passed:
        print("\nValidation passed with warnings.")
    else:
        print("\nValidation failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pipeline data completeness in DuckDB.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings as well as errors.")
    args = parser.parse_args()

    passed, issues = run_validation(strict=args.strict)
    _print_report(passed, issues, args.strict)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
