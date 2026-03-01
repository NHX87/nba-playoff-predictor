"""
matchup_model.py
----------------
Train a head-to-head series winner model from historical playoff series.

Inputs:
  - DuckDB table: playoffs
  - DuckDB table: model_features

Outputs:
  - models/trained/matchup_model.joblib
  - models/trained/matchup_metrics.json
  - models/trained/matchup_validation_predictions.csv
  - DuckDB table: matchup_validation_predictions

Run:
  python -m pipeline.models.matchup_model
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import DB_PATH, FINAL_FEATURES, RANDOM_STATE

ARTIFACT_DIR = Path("models/trained")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "matchup_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "matchup_metrics.json"
VAL_PRED_PATH = ARTIFACT_DIR / "matchup_validation_predictions.csv"


def _season_start_year(season: str) -> int:
    return int(str(season).split("-")[0])


def _load_series_outcomes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Derive series winner/loser pairs from playoff game logs."""
    df = con.execute(
        """
        WITH po AS (
            SELECT
                SEASON,
                GAME_ID,
                TEAM_ID,
                TEAM_ABBR,
                WL,
                CAST(SUBSTR(CAST(GAME_ID AS VARCHAR), 4, 2) AS INT) AS round_code,
                SUBSTR(CAST(GAME_ID AS VARCHAR), 6, 3) AS series_num
            FROM playoffs
            WHERE GAME_ID IS NOT NULL
              AND CAST(GAME_ID AS VARCHAR) LIKE '004%'
        ),
        wins AS (
            SELECT
                SEASON,
                round_code,
                series_num,
                TEAM_ID,
                TEAM_ABBR,
                SUM(CASE WHEN WL = 'W' THEN 1 ELSE 0 END) AS wins
            FROM po
            GROUP BY 1, 2, 3, 4, 5
        )
        SELECT
            SEASON,
            round_code,
            series_num,
            MAX(CASE WHEN wins = 4 THEN TEAM_ID END) AS winner_team_id,
            MAX(CASE WHEN wins = 4 THEN TEAM_ABBR END) AS winner_team_abbr,
            MAX(CASE WHEN wins < 4 THEN TEAM_ID END) AS loser_team_id,
            MAX(CASE WHEN wins < 4 THEN TEAM_ABBR END) AS loser_team_abbr
        FROM wins
        GROUP BY 1, 2, 3
        HAVING COUNT(*) = 2
           AND MAX(wins) = 4
        ORDER BY SEASON, round_code, series_num
        """
    ).df()

    if df.empty:
        raise ValueError("No historical playoff series outcomes found in playoffs table.")

    return df


def _load_team_features(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, list[str]]:
    mf = con.execute("SELECT * FROM model_features").df()
    if mf.empty:
        raise ValueError("model_features table is empty. Build features first.")

    feature_cols = [c for c in FINAL_FEATURES if c in mf.columns]
    if not feature_cols:
        raise ValueError("No FINAL_FEATURES found in model_features.")

    keep = ["TEAM_ID", "TEAM_ABBR", "SEASON"] + feature_cols
    out = mf[keep].copy()
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out, feature_cols


def _build_training_frame(series_df: pd.DataFrame, team_feats: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Create symmetric matchup rows: (winner, loser)=1 and (loser, winner)=0."""
    # Positive orientation
    pos = series_df.merge(
        team_feats,
        left_on=["winner_team_id", "SEASON"],
        right_on=["TEAM_ID", "SEASON"],
        how="left",
    ).rename(columns={"TEAM_ABBR": "team_a_abbr"})

    pos = pos.merge(
        team_feats,
        left_on=["loser_team_id", "SEASON"],
        right_on=["TEAM_ID", "SEASON"],
        how="left",
        suffixes=("_a", "_b"),
    ).rename(columns={"TEAM_ABBR": "team_b_abbr"})

    rows = []
    for r in pos.itertuples(index=False):
        base = {
            "SEASON": r.SEASON,
            "round_code": int(r.round_code),
            "series_num": str(r.series_num),
        }

        delta = {}
        for c in feature_cols:
            va = getattr(r, f"{c}_a", None)
            vb = getattr(r, f"{c}_b", None)
            delta[f"delta_{c}"] = (va - vb) if pd.notna(va) and pd.notna(vb) else None

        row_pos = {
            **base,
            "team_a": r.winner_team_abbr,
            "team_b": r.loser_team_abbr,
            "target_team_a_wins": 1,
            **delta,
        }

        row_neg = {
            **base,
            "team_a": r.loser_team_abbr,
            "team_b": r.winner_team_abbr,
            "target_team_a_wins": 0,
            **{k: (-v if v is not None else None) for k, v in delta.items()},
        }

        rows.append(row_pos)
        rows.append(row_neg)

    df = pd.DataFrame(rows)
    delta_cols = [f"delta_{c}" for c in feature_cols]
    df = df.dropna(subset=delta_cols).copy()
    if df.empty:
        raise ValueError("Matchup training frame is empty after feature joins.")

    df["season_start_year"] = df["SEASON"].apply(_season_start_year)
    return df


def _split_train_val(df: pd.DataFrame, holdout_seasons: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df["season_start_year"].unique())
    if len(seasons) <= holdout_seasons:
        raise ValueError(
            f"Need > {holdout_seasons} seasons for validation split, found {len(seasons)}"
        )
    val_years = set(seasons[-holdout_seasons:])
    train = df[~df["season_start_year"].isin(val_years)].copy()
    val = df[df["season_start_year"].isin(val_years)].copy()
    return train, val


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_matchup_model() -> dict[str, Any]:
    con = duckdb.connect(DB_PATH)
    try:
        series_df = _load_series_outcomes(con)
        team_feats, feature_cols = _load_team_features(con)
    finally:
        con.close()

    frame = _build_training_frame(series_df, team_feats, feature_cols)
    train_df, val_df = _split_train_val(frame, holdout_seasons=3)

    delta_cols = [f"delta_{c}" for c in feature_cols]

    X_train = train_df[delta_cols]
    y_train = train_df["target_team_a_wins"]
    X_val = val_df[delta_cols]
    y_val = val_df["target_team_a_wins"]

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_val = pipe.predict_proba(X_val)[:, 1]

    train_metrics = {
        "accuracy": float(accuracy_score(y_train, (p_train >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y_train, p_train)),
        "log_loss": float(log_loss(y_train, p_train)),
        "brier": float(brier_score_loss(y_train, p_train)),
    }
    val_metrics = {
        "accuracy": float(accuracy_score(y_val, (p_val >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y_val, p_val)),
        "log_loss": float(log_loss(y_val, p_val)),
        "brier": float(brier_score_loss(y_val, p_val)),
    }

    val_out = val_df[["SEASON", "round_code", "series_num", "team_a", "team_b", "target_team_a_wins"]].copy()
    val_out["p_team_a_wins"] = p_val
    val_out["pred_team_a_wins"] = (val_out["p_team_a_wins"] >= 0.5).astype(int)
    val_out["predicted_winner"] = val_out.apply(
        lambda r: r["team_a"] if r["pred_team_a_wins"] == 1 else r["team_b"], axis=1
    )

    artifact = {
        "model": pipe,
        "team_feature_columns": feature_cols,
        "delta_feature_columns": delta_cols,
    }
    joblib.dump(artifact, MODEL_PATH)

    metrics = {
        "n_series": int(len(series_df)),
        "n_rows_total": int(len(frame)),
        "n_rows_train": int(len(train_df)),
        "n_rows_validation": int(len(val_df)),
        "holdout_seasons": sorted(val_df["SEASON"].unique().tolist()),
        "team_feature_columns": feature_cols,
        "delta_feature_columns": delta_cols,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    val_out.to_csv(VAL_PRED_PATH, index=False)

    con = duckdb.connect(DB_PATH)
    try:
        con.execute("DROP TABLE IF EXISTS matchup_validation_predictions")
        con.execute("CREATE TABLE matchup_validation_predictions AS SELECT * FROM val_out")
    finally:
        con.close()

    print("Matchup model training complete")
    print(f"Train ROC-AUC: {train_metrics['roc_auc']:.3f}")
    print(f"Val ROC-AUC: {val_metrics['roc_auc']:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved validation predictions: {VAL_PRED_PATH}")

    return metrics


def load_matchup_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Matchup artifact not found at {MODEL_PATH}. "
            "Run `python -m pipeline.models.matchup_model` first."
        )
    return joblib.load(MODEL_PATH)


def predict_matchup_prob(
    team_a: str,
    team_b: str,
    current_features: pd.DataFrame,
    artifact: dict[str, Any] | None = None,
) -> float:
    """Predict probability that team_a wins a best-of-7 series vs team_b."""
    if artifact is None:
        artifact = load_matchup_artifact()

    model = artifact["model"]
    feat_cols: list[str] = artifact["team_feature_columns"]
    delta_cols: list[str] = artifact["delta_feature_columns"]

    lookup = current_features.set_index("TEAM_ABBR")
    if team_a not in lookup.index or team_b not in lookup.index:
        raise ValueError(f"Missing team in current feature frame: {team_a} vs {team_b}")

    row_a = lookup.loc[team_a]
    row_b = lookup.loc[team_b]

    x = pd.DataFrame(
        [{f"delta_{c}": float(row_a[c] - row_b[c]) for c in feat_cols}],
        columns=delta_cols,
    )
    return float(model.predict_proba(x)[0, 1])


if __name__ == "__main__":
    train_matchup_model()
