"""
sanity_report.py
----------------
Generate model sanity checks for current-season playoff outputs.

Checks:
  1) Strongest round-1 upset risks
  2) Biggest seed-vs-title-odds gaps
  3) Sensitivity of title odds to Monte Carlo run count (5k/10k/20k)

Outputs:
  - models/trained/model_sanity_report.md
  - models/trained/sanity_upset_risks.csv
  - models/trained/sanity_seed_vs_odds_gap.csv
  - models/trained/sanity_sensitivity.csv

Run:
  python -m pipeline.models.sanity_report
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from config.settings import CURRENT_SEASON_STR, DB_PATH
from pipeline.models.simulation import run_monte_carlo

ARTIFACT_DIR = Path("models/trained")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = ARTIFACT_DIR / "model_sanity_report.md"
UPSET_PATH = ARTIFACT_DIR / "sanity_upset_risks.csv"
GAP_PATH = ARTIFACT_DIR / "sanity_seed_vs_odds_gap.csv"
SENS_PATH = ARTIFACT_DIR / "sanity_sensitivity.csv"

SENS_RUN_COUNTS = [5000, 10000, 20000]


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        vals = [str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _load_app_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        tables = set(
            con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).df()["table_name"].tolist()
        )

        required = {"app_series_predictions_current", "app_title_odds_current"}
        missing = sorted(required - tables)
        if missing:
            raise ValueError(
                f"Missing app tables: {missing}. Run `python -m pipeline.models.simulation` first."
            )

        series_df = con.execute(
            """
            SELECT *
            FROM app_series_predictions_current
            WHERE season = ?
            """,
            [CURRENT_SEASON_STR],
        ).df()

        odds_df = con.execute(
            """
            SELECT *
            FROM app_title_odds_current
            WHERE season = ?
            """,
            [CURRENT_SEASON_STR],
        ).df()
    finally:
        con.close()

    return series_df, odds_df


def _upset_risks(series_df: pd.DataFrame) -> pd.DataFrame:
    r1 = series_df[series_df["round"] == "First Round"].copy()
    if r1.empty:
        return pd.DataFrame()

    r1["upset_risk"] = r1["low_team_win_prob"]
    r1["favorite"] = np.where(
        r1["high_team_win_prob"] >= r1["low_team_win_prob"], r1["high_team"], r1["low_team"]
    )
    r1["underdog"] = np.where(r1["favorite"] == r1["high_team"], r1["low_team"], r1["high_team"])
    r1["favorite_win_prob"] = np.where(
        r1["favorite"] == r1["high_team"], r1["high_team_win_prob"], r1["low_team_win_prob"]
    )

    out = r1[
        [
            "conference",
            "high_seed",
            "high_team",
            "low_seed",
            "low_team",
            "upset_risk",
            "favorite",
            "favorite_win_prob",
            "predicted_winner",
        ]
    ].sort_values("upset_risk", ascending=False)

    out["upset_risk"] = out["upset_risk"].round(4)
    out["favorite_win_prob"] = out["favorite_win_prob"].round(4)
    return out.reset_index(drop=True)


def _seed_vs_odds_gap(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        return pd.DataFrame()

    x = odds_df.copy()
    x["title_rank_conf"] = x.groupby("conference")["title_prob"].rank(method="min", ascending=False)
    x["title_rank_overall"] = x["title_prob"].rank(method="min", ascending=False)

    # Positive = stronger title-odds rank than standings seed implies.
    x["seed_vs_title_gap_conf"] = x["playoff_seed"] - x["title_rank_conf"]
    x["abs_gap_conf"] = x["seed_vs_title_gap_conf"].abs()

    out = x[
        [
            "TEAM_ABBR",
            "conference",
            "playoff_seed",
            "title_rank_conf",
            "title_rank_overall",
            "title_prob",
            "seed_vs_title_gap_conf",
            "abs_gap_conf",
        ]
    ].sort_values(["abs_gap_conf", "title_prob"], ascending=[False, False])

    out["title_prob"] = out["title_prob"].round(4)
    return out.reset_index(drop=True)


def _sensitivity_sweep() -> pd.DataFrame:
    run_frames = []

    for n in SENS_RUN_COUNTS:
        _, odds_df, _, _, _ = run_monte_carlo(n_simulations=n, rng_seed=42)
        tmp = odds_df[["TEAM_ABBR", "playoff_seed", "title_prob"]].copy()
        tmp = tmp.rename(columns={"title_prob": f"title_prob_{n}"})
        tmp[f"rank_{n}"] = tmp[f"title_prob_{n}"].rank(method="min", ascending=False)
        run_frames.append(tmp)

    merged = run_frames[0]
    for nxt in run_frames[1:]:
        merged = merged.merge(nxt, on=["TEAM_ABBR", "playoff_seed"], how="outer")

    prob_cols = [f"title_prob_{n}" for n in SENS_RUN_COUNTS]
    rank_cols = [f"rank_{n}" for n in SENS_RUN_COUNTS]

    merged["title_prob_mean"] = merged[prob_cols].mean(axis=1)
    merged["title_prob_max_min_spread"] = merged[prob_cols].max(axis=1) - merged[prob_cols].min(axis=1)
    merged["rank_std"] = merged[rank_cols].std(axis=1)

    merged = merged.sort_values("title_prob_10000", ascending=False).reset_index(drop=True)

    for c in prob_cols + ["title_prob_mean", "title_prob_max_min_spread"]:
        merged[c] = merged[c].round(4)
    merged["rank_std"] = merged["rank_std"].round(3)

    return merged


def generate_sanity_report() -> None:
    series_df, odds_df = _load_app_tables()

    upset_df = _upset_risks(series_df)
    gap_df = _seed_vs_odds_gap(odds_df)
    sens_df = _sensitivity_sweep()

    upset_df.to_csv(UPSET_PATH, index=False)
    gap_df.to_csv(GAP_PATH, index=False)
    sens_df.to_csv(SENS_PATH, index=False)

    top_upsets = upset_df.head(8).copy()
    top_gap = gap_df.head(10).copy()
    top_sens = sens_df[
        [
            "TEAM_ABBR",
            "playoff_seed",
            "title_prob_5000",
            "title_prob_10000",
            "title_prob_20000",
            "title_prob_max_min_spread",
            "rank_5000",
            "rank_10000",
            "rank_20000",
            "rank_std",
        ]
    ].head(12)

    report = f"""# Model Sanity Report ({CURRENT_SEASON_STR})

## 1) Strongest Round-1 Upset Risks

Upset risk is defined as `P(low seed beats high seed)`.

{_md_table(top_upsets)}

## 2) Biggest Seed vs Title-Odds Gaps

`seed_vs_title_gap_conf = playoff_seed - conference_title_rank`

- Positive gap: team is outperforming seed in model odds.
- Negative gap: team is underperforming seed in model odds.

{_md_table(top_gap)}

## 3) Monte Carlo Sensitivity (5k vs 10k vs 20k)

This checks stability of title odds and ranking against simulation count.

{_md_table(top_sens)}

## Artifact Files

- `{UPSET_PATH}`
- `{GAP_PATH}`
- `{SENS_PATH}`
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Model sanity report generated")
    print(f"Saved markdown: {REPORT_PATH}")
    print(f"Saved upset risks: {UPSET_PATH}")
    print(f"Saved seed-vs-odds gap: {GAP_PATH}")
    print(f"Saved sensitivity sweep: {SENS_PATH}")


if __name__ == "__main__":
    generate_sanity_report()
