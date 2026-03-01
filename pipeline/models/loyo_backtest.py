"""
loyo_backtest.py
----------------
Leave-One-Year-Out (LOYO) cross-validation for survival and matchup models.

For each season S in the historical dataset:
  - Survival model: train on all other seasons, evaluate on S
      → C-index for season S, champion's predicted rank
  - Matchup model: train on all other seasons' series, evaluate on S's series
      → per-series accuracy, Brier score, probability assigned to actual winner

This produces true out-of-sample accuracy estimates across all 15 seasons —
a much more reliable signal than the single 3-season holdout used in training.

Outputs:
  - models/trained/loyo_backtest_results.json     (aggregate + per-season metrics)
  - models/trained/loyo_season_summary.csv        (per-season survival metrics)
  - models/trained/loyo_series_predictions.csv    (per-series matchup predictions)
  - DuckDB tables: loyo_season_summary, loyo_series_predictions

Run:
  python -m pipeline.models.loyo_backtest
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config.settings import DB_PATH
from pipeline.models.matchup_model import (
    _build_pipeline,
    _build_training_frame,
    _load_series_outcomes,
    _load_team_features,
)
from pipeline.models.survival import (
    build_validation_predictions,
    evaluate_concordance,
    fit_cox_model,
    load_training_frame,
    season_sanity_checks,
)

ARTIFACT_DIR = Path("models/trained")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = ARTIFACT_DIR / "loyo_backtest_results.json"
SEASON_SUMMARY_PATH = ARTIFACT_DIR / "loyo_season_summary.csv"
SERIES_PATH = ARTIFACT_DIR / "loyo_series_predictions.csv"

ROUND_LABELS = {
    1: "First Round",
    2: "Conference Semifinals",
    3: "Conference Finals",
    4: "NBA Finals",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load all tables needed for LOYO folds in one DB connection."""
    survival_df, _, feature_cols = load_training_frame()

    con = duckdb.connect(DB_PATH)
    try:
        series_df = _load_series_outcomes(con)
        team_feats, _ = _load_team_features(con)
    finally:
        con.close()

    return survival_df, series_df, team_feats, feature_cols


# ---------------------------------------------------------------------------
# Per-fold logic
# ---------------------------------------------------------------------------

def _survival_fold(
    test_season: str,
    all_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Train CoxPH on all seasons except test_season; evaluate on test_season."""
    train_df = all_df[all_df["SEASON"] != test_season].copy()
    test_df = all_df[all_df["SEASON"] == test_season].copy()

    if train_df.empty or test_df.empty:
        return {}

    model = fit_cox_model(train_df, feature_cols)
    c_index = evaluate_concordance(model, test_df, feature_cols)
    preds = build_validation_predictions(model, test_df, feature_cols)
    checks = season_sanity_checks(preds)
    check = checks[0] if checks else {}

    return {
        "season": test_season,
        "n_teams": len(test_df),
        "c_index": round(c_index, 4),
        "champion": check.get("champion"),
        "champion_pred_rank": check.get("champion_pred_rank"),
        "champion_in_top_3": check.get("champion_in_top_3"),
        "champion_in_top_5": check.get("champion_in_top_5"),
    }


def _matchup_fold(
    test_season: str,
    all_series: pd.DataFrame,
    all_team_feats: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    """
    Train logistic matchup model on all seasons except test_season.
    Evaluate on every series in test_season.
    Always orient: team_a = actual winner (so correct prediction ↔ p_winner ≥ 0.5).
    """
    train_series = all_series[all_series["SEASON"] != test_season].copy()
    test_series = all_series[all_series["SEASON"] == test_season].copy()
    train_feats = all_team_feats[all_team_feats["SEASON"] != test_season].copy()
    test_feats = all_team_feats[all_team_feats["SEASON"] == test_season].copy()

    if train_series.empty or test_series.empty or train_feats.empty or test_feats.empty:
        return []

    train_frame = _build_training_frame(train_series, train_feats, feature_cols)
    if train_frame.empty:
        return []

    delta_cols = [f"delta_{c}" for c in feature_cols]
    pipe = _build_pipeline()
    pipe.fit(train_frame[delta_cols], train_frame["target_team_a_wins"])

    feat_lookup = test_feats.set_index("TEAM_ABBR")
    rows: list[dict[str, Any]] = []

    for r in test_series.itertuples(index=False):
        winner = str(r.winner_team_abbr)
        loser = str(r.loser_team_abbr)

        if winner not in feat_lookup.index or loser not in feat_lookup.index:
            continue

        # delta = winner_features - loser_features → target=1 means winner beats loser
        delta = pd.DataFrame(
            [{f"delta_{c}": float(feat_lookup.loc[winner, c] - feat_lookup.loc[loser, c])
              for c in feature_cols}],
            columns=delta_cols,
        )
        p_winner = float(pipe.predict_proba(delta)[0, 1])

        rows.append({
            "season": str(r.SEASON),
            "round_code": int(r.round_code),
            "round_label": ROUND_LABELS.get(int(r.round_code), f"Round {r.round_code}"),
            "winner": winner,
            "loser": loser,
            "p_winner": round(p_winner, 4),
            "p_loser": round(1.0 - p_winner, 4),
            "correct": bool(p_winner >= 0.5),
            "brier": round((p_winner - 1.0) ** 2, 4),  # actual=1
        })

    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_survival(season_rows: list[dict]) -> dict[str, Any]:
    c_indices = [r["c_index"] for r in season_rows if r.get("c_index") is not None]
    ranks = [r["champion_pred_rank"] for r in season_rows
             if r.get("champion_pred_rank") is not None]

    if not c_indices:
        return {}

    return {
        "n_seasons_evaluated": len(season_rows),
        "mean_c_index": round(float(np.mean(c_indices)), 4),
        "std_c_index": round(float(np.std(c_indices)), 4),
        "min_c_index": round(float(np.min(c_indices)), 4),
        "max_c_index": round(float(np.max(c_indices)), 4),
        "champion_top1_rate": round(sum(r == 1 for r in ranks) / len(ranks), 4) if ranks else None,
        "champion_top3_rate": round(sum(r <= 3 for r in ranks) / len(ranks), 4) if ranks else None,
        "champion_top5_rate": round(sum(r <= 5 for r in ranks) / len(ranks), 4) if ranks else None,
        "mean_champion_rank": round(float(np.mean(ranks)), 2) if ranks else None,
    }


def _aggregate_matchup(series_rows: list[dict]) -> dict[str, Any]:
    if not series_rows:
        return {}

    correct = [r["correct"] for r in series_rows]
    briers = [r["brier"] for r in series_rows]
    p_winners = [r["p_winner"] for r in series_rows]

    # ROC-AUC: build symmetric dataset (each series contributes winner+loser row)
    probs_sym = p_winners + [1.0 - p for p in p_winners]
    actuals_sym = [1] * len(p_winners) + [0] * len(p_winners)
    try:
        roc = round(float(roc_auc_score(actuals_sym, probs_sym)), 4)
    except Exception:
        roc = None

    by_round: dict[str, dict] = {}
    for rc, label in ROUND_LABELS.items():
        rd = [r for r in series_rows if r["round_code"] == rc]
        if rd:
            by_round[label] = {
                "n_series": len(rd),
                "accuracy": round(float(np.mean([r["correct"] for r in rd])), 4),
                "mean_brier": round(float(np.mean([r["brier"] for r in rd])), 4),
            }

    return {
        "n_series_evaluated": len(series_rows),
        "overall_accuracy": round(float(np.mean(correct)), 4),
        "mean_brier_score": round(float(np.mean(briers)), 4),
        "roc_auc": roc,
        "by_round": by_round,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_loyo_backtest() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    print("Loading data for LOYO backtest...")
    survival_df, series_df, team_feats, feature_cols = _load_all_data()

    all_seasons = sorted(survival_df["SEASON"].unique())
    n = len(all_seasons)
    print(f"Running {n} LOYO folds  ({all_seasons[0]} → {all_seasons[-1]})")
    print(f"Features: {feature_cols}\n")

    season_rows: list[dict] = []
    all_series_rows: list[dict] = []

    for i, season in enumerate(all_seasons, 1):
        print(f"  [{i:>2}/{n}] test={season} ", end="", flush=True)

        surv = _survival_fold(season, survival_df, feature_cols)
        matchup_rows = _matchup_fold(season, series_df, team_feats, feature_cols)

        if surv:
            season_rows.append(surv)
        all_series_rows.extend(matchup_rows)

        rank = surv.get("champion_pred_rank", "?")
        ci = surv.get("c_index", 0.0)
        top3 = "✓" if surv.get("champion_in_top_3") else "✗"
        n_series = len(matchup_rows)
        matchup_acc = np.mean([r["correct"] for r in matchup_rows]) if matchup_rows else float("nan")
        print(f"c-idx={ci:.3f}  champ={surv.get('champion','?')} rank={rank} top3={top3}  "
              f"series={n_series} acc={matchup_acc:.0%}")

    results = {
        "features_used": feature_cols,
        "n_seasons": n,
        "seasons_range": f"{all_seasons[0]} → {all_seasons[-1]}",
        "survival": _aggregate_survival(season_rows),
        "matchup": _aggregate_matchup(all_series_rows),
        "per_season": season_rows,
    }

    season_summary_df = pd.DataFrame(season_rows)
    series_summary_df = pd.DataFrame(all_series_rows)

    return results, season_summary_df, series_summary_df


def save_outputs(
    results: dict[str, Any],
    season_df: pd.DataFrame,
    series_df: pd.DataFrame,
) -> None:
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    season_df.to_csv(SEASON_SUMMARY_PATH, index=False)
    series_df.to_csv(SERIES_PATH, index=False)

    con = duckdb.connect(DB_PATH)
    try:
        con.execute("DROP TABLE IF EXISTS loyo_season_summary")
        con.execute("CREATE TABLE loyo_season_summary AS SELECT * FROM season_df")

        con.execute("DROP TABLE IF EXISTS loyo_series_predictions")
        con.execute("CREATE TABLE loyo_series_predictions AS SELECT * FROM series_df")
    finally:
        con.close()

    print(f"\nSaved: {RESULTS_PATH}")
    print(f"Saved: {SEASON_SUMMARY_PATH}")
    print(f"Saved: {SERIES_PATH}")


def _print_report(results: dict[str, Any]) -> None:
    s = results["survival"]
    m = results["matchup"]

    print("\n" + "=" * 60)
    print("LOYO BACKTEST RESULTS")
    print("=" * 60)
    print(f"Features : {', '.join(results['features_used'])}")
    print(f"Seasons  : {results['seasons_range']}  (n={results['n_seasons']})")

    print("\n── Survival Model (CoxPH) ──────────────────────────────")
    print(f"  Mean C-index       {s['mean_c_index']:.4f}  ±{s['std_c_index']:.4f}")
    print(f"  Range              {s['min_c_index']:.4f} – {s['max_c_index']:.4f}")
    print(f"  Champion top-1     {s['champion_top1_rate']:.1%}")
    print(f"  Champion top-3     {s['champion_top3_rate']:.1%}")
    print(f"  Champion top-5     {s['champion_top5_rate']:.1%}")
    print(f"  Mean champ rank    {s['mean_champion_rank']:.1f} / 16")

    print("\n── Matchup Model (Logistic Regression) ─────────────────")
    if m:
        print(f"  Series accuracy    {m['overall_accuracy']:.1%}  ({m['n_series_evaluated']} series)")
        print(f"  Mean Brier score   {m['mean_brier_score']:.4f}  (random baseline = 0.25)")
        if m.get("roc_auc"):
            print(f"  ROC-AUC            {m['roc_auc']:.4f}")
        print("\n  By round:")
        for label, rd in m.get("by_round", {}).items():
            print(f"    {label:<28}  acc={rd['accuracy']:.1%}  "
                  f"brier={rd['mean_brier']:.4f}  n={rd['n_series']}")

    print("\n── Per-Season Survival Summary ─────────────────────────")
    print(f"  {'Season':<10} {'Champion':<6} {'Rank':>4}  {'Top3':>4}  {'C-index':>7}")
    print(f"  {'-'*9} {'-'*5} {'-'*4}  {'-'*4}  {'-'*7}")
    for r in results["per_season"]:
        top3 = " ✓" if r.get("champion_in_top_3") else " ✗"
        rank = r.get("champion_pred_rank", "?")
        ci = r.get("c_index", 0.0)
        print(f"  {r['season']:<10} {r.get('champion','?'):<6} {rank:>4} {top3:>5}  {ci:>7.3f}")


if __name__ == "__main__":
    results, season_df, series_df = run_loyo_backtest()
    save_outputs(results, season_df, series_df)
    _print_report(results)
