"""
survival.py
-----------
Train and evaluate a Cox proportional hazards model for playoff elimination risk.

Inputs:
  - DuckDB table: model_features

Outputs:
  - models/trained/survival_coxph.joblib
  - models/trained/survival_metrics.json
  - models/trained/survival_validation_predictions.csv
  - DuckDB table: survival_validation_predictions

Run:
  python -m pipeline.models.survival
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from config.settings import DB_PATH, FINAL_FEATURES, TARGET

ARTIFACT_DIR = Path("models/trained")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "survival_coxph.joblib"
METRICS_PATH = ARTIFACT_DIR / "survival_metrics.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "survival_validation_predictions.csv"

DURATION_COL = "rounds_reached"
EVENT_COL = "event_eliminated"
SEASON_NUM_COL = "season_start_year"


def _season_start_year(season: str) -> int:
    """Convert season string '2024-25' to integer 2024 for sorting/splitting."""
    return int(str(season).split("-")[0])


def _resolve_target_column(df: pd.DataFrame) -> str:
    """Resolve target column from config name or known fallback names."""
    if TARGET in df.columns:
        return TARGET

    fallback = ["playoff_rounds_reached"]
    for col in fallback:
        if col in df.columns:
            return col

    available = ", ".join(df.columns.tolist())
    raise ValueError(
        f"Could not find target column '{TARGET}' or fallback target in model_features. "
        f"Available columns: {available}"
    )


def load_training_frame() -> tuple[pd.DataFrame, str, list[str]]:
    """Load model_features and produce a clean frame for Cox training."""
    con = duckdb.connect(DB_PATH)
    try:
        df = con.execute("SELECT * FROM model_features").df()
    finally:
        con.close()

    if df.empty:
        raise ValueError("model_features is empty. Build feature tables before training.")

    target_col = _resolve_target_column(df)

    available_features = [f for f in FINAL_FEATURES if f in df.columns]
    if not available_features:
        raise ValueError(
            "None of FINAL_FEATURES are present in model_features. "
            "Run feature engineering and build_features first."
        )

    # Keep identifiers for reporting and sanity checks.
    keep_cols = ["TEAM_ID", "TEAM_ABBR", "SEASON", target_col] + available_features
    model_df = df[keep_cols].copy()

    # Standardize target name internally.
    model_df[DURATION_COL] = pd.to_numeric(model_df[target_col], errors="coerce")

    # Champions (round=5) are treated as right-censored, others are elimination events.
    model_df[EVENT_COL] = (model_df[DURATION_COL] < 5).astype(int)

    model_df[SEASON_NUM_COL] = model_df["SEASON"].apply(_season_start_year)

    # Coerce numeric features and drop invalid rows.
    for col in available_features:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    required_cols = [DURATION_COL, EVENT_COL] + available_features
    model_df = model_df.dropna(subset=required_cols).copy()

    # CoxPH requires strictly positive durations.
    model_df = model_df[model_df[DURATION_COL] > 0].copy()

    if len(model_df) < 40:
        raise ValueError(
            f"Not enough clean rows for survival training ({len(model_df)})."
        )

    return model_df, target_col, available_features


def split_train_validation(df: pd.DataFrame, holdout_seasons: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: last N seasons for validation."""
    seasons = sorted(df[SEASON_NUM_COL].unique())
    if len(seasons) <= holdout_seasons:
        raise ValueError(
            f"Need more than {holdout_seasons} seasons to create a validation split; found {len(seasons)}"
        )

    val_seasons = set(seasons[-holdout_seasons:])
    train_df = df[~df[SEASON_NUM_COL].isin(val_seasons)].copy()
    val_df = df[df[SEASON_NUM_COL].isin(val_seasons)].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split produced an empty partition.")

    return train_df, val_df


def fit_cox_model(train_df: pd.DataFrame, feature_cols: list[str]) -> CoxPHFitter:
    """Fit Cox proportional hazards model with light regularization for stability."""
    fit_df = train_df[[DURATION_COL, EVENT_COL] + feature_cols].copy()

    model = CoxPHFitter(penalizer=0.05)
    model.fit(
        fit_df,
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        show_progress=False,
    )
    return model


def evaluate_concordance(model: CoxPHFitter, df: pd.DataFrame, feature_cols: list[str]) -> float:
    """Compute concordance index where higher survival score predicts deeper run."""
    eval_df = df[[DURATION_COL, EVENT_COL] + feature_cols].copy()
    risk = model.predict_partial_hazard(eval_df[feature_cols]).values.reshape(-1)

    # Higher risk => earlier elimination. Negate so higher score means longer survival.
    survival_score = -risk
    c_index = concordance_index(
        eval_df[DURATION_COL].values,
        survival_score,
        eval_df[EVENT_COL].values,
    )
    return float(c_index)


def build_validation_predictions(
    model: CoxPHFitter, val_df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Build season-wise validation ranking outputs for sanity checks."""
    out = val_df[["TEAM_ID", "TEAM_ABBR", "SEASON", DURATION_COL, EVENT_COL]].copy()
    risk = model.predict_partial_hazard(val_df[feature_cols]).values.reshape(-1)
    out["pred_elimination_risk"] = risk
    out["pred_survival_score"] = -risk

    out["pred_rank_in_season"] = (
        out.groupby("SEASON")["pred_survival_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    out["actual_rank_in_season"] = (
        out.groupby("SEASON")[DURATION_COL]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    out["is_champion"] = (out[DURATION_COL] == 5).astype(int)

    return out.sort_values(["SEASON", "pred_rank_in_season", "TEAM_ABBR"]).reset_index(drop=True)


def season_sanity_checks(pred_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-season checks: champion rank and top-k hit flags."""
    checks: list[dict[str, Any]] = []

    for season, grp in pred_df.groupby("SEASON"):
        champs = grp[grp["is_champion"] == 1]
        champion_name = champs.iloc[0]["TEAM_ABBR"] if not champs.empty else None
        champion_rank = int(champs.iloc[0]["pred_rank_in_season"]) if not champs.empty else None

        checks.append(
            {
                "season": season,
                "champion": champion_name,
                "champion_pred_rank": champion_rank,
                "champion_in_top_3": bool(champion_rank is not None and champion_rank <= 3),
                "champion_in_top_5": bool(champion_rank is not None and champion_rank <= 5),
                "teams_in_season": int(len(grp)),
            }
        )

    return sorted(checks, key=lambda x: x["season"])


def save_artifacts(
    model: CoxPHFitter,
    feature_cols: list[str],
    metrics: dict[str, Any],
    validation_predictions: pd.DataFrame,
) -> None:
    """Persist model and evaluation outputs for downstream app/model usage."""
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "duration_col": DURATION_COL,
            "event_col": EVENT_COL,
        },
        MODEL_PATH,
    )

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    validation_predictions.to_csv(PREDICTIONS_PATH, index=False)

    con = duckdb.connect(DB_PATH)
    try:
        con.execute("DROP TABLE IF EXISTS survival_validation_predictions")
        con.execute(
            "CREATE TABLE survival_validation_predictions AS SELECT * FROM validation_predictions"
        )
    finally:
        con.close()


def train_survival_model() -> dict[str, Any]:
    """Train CoxPH model, evaluate, and persist artifacts."""
    df, target_col, feature_cols = load_training_frame()
    train_df, val_df = split_train_validation(df, holdout_seasons=3)

    model = fit_cox_model(train_df, feature_cols)

    train_c_index = evaluate_concordance(model, train_df, feature_cols)
    val_c_index = evaluate_concordance(model, val_df, feature_cols)

    val_preds = build_validation_predictions(model, val_df, feature_cols)
    season_checks = season_sanity_checks(val_preds)

    metrics = {
        "target_column_used": target_col,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train_df)),
        "n_rows_validation": int(len(val_df)),
        "n_features": int(len(feature_cols)),
        "features_used": feature_cols,
        "train_c_index": train_c_index,
        "validation_c_index": val_c_index,
        "holdout_seasons": sorted(val_df["SEASON"].unique().tolist()),
        "season_sanity_checks": season_checks,
        "coefficients": {
            k: float(v) for k, v in model.params_.to_dict().items()
        },
    }

    save_artifacts(model, feature_cols, metrics, val_preds)
    return metrics


def _print_summary(metrics: dict[str, Any]) -> None:
    print("\nSurvival model training complete")
    print(f"Target used: {metrics['target_column_used']}")
    print(
        f"Rows: total={metrics['n_rows_total']} train={metrics['n_rows_train']} "
        f"validation={metrics['n_rows_validation']}"
    )
    print(f"Features used ({metrics['n_features']}): {', '.join(metrics['features_used'])}")
    print(f"Train c-index: {metrics['train_c_index']:.3f}")
    print(f"Validation c-index: {metrics['validation_c_index']:.3f}")
    print("\nSeason sanity checks (validation):")
    for row in metrics["season_sanity_checks"]:
        print(
            f"  {row['season']}: champion={row['champion']} "
            f"pred_rank={row['champion_pred_rank']} "
            f"top3={row['champion_in_top_3']} top5={row['champion_in_top_5']}"
        )
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved validation predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    summary_metrics = train_survival_model()
    _print_summary(summary_metrics)
