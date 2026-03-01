"""
evaluate.py
-----------
Evaluate saved survival model artifact against current model_features data.

Run:
  python -m pipeline.models.evaluate
"""

from __future__ import annotations

import joblib

from pipeline.models.survival import (
    MODEL_PATH,
    build_validation_predictions,
    evaluate_concordance,
    load_training_frame,
    season_sanity_checks,
    split_train_validation,
)


def evaluate_saved_survival_model(holdout_seasons: int = 3) -> dict:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    df, target_col, _ = load_training_frame()
    train_df, val_df = split_train_validation(df, holdout_seasons=holdout_seasons)

    train_c_index = evaluate_concordance(model, train_df, feature_cols)
    val_c_index = evaluate_concordance(model, val_df, feature_cols)

    val_preds = build_validation_predictions(model, val_df, feature_cols)
    checks = season_sanity_checks(val_preds)

    return {
        "target_column_used": target_col,
        "train_c_index": train_c_index,
        "validation_c_index": val_c_index,
        "holdout_seasons": sorted(val_df["SEASON"].unique().tolist()),
        "season_sanity_checks": checks,
    }


def _print_report(metrics: dict) -> None:
    print("Saved survival model evaluation")
    print(f"Target used: {metrics['target_column_used']}")
    print(f"Train c-index: {metrics['train_c_index']:.3f}")
    print(f"Validation c-index: {metrics['validation_c_index']:.3f}")
    print(f"Holdout seasons: {', '.join(metrics['holdout_seasons'])}")
    print("Season sanity checks:")
    for row in metrics["season_sanity_checks"]:
        print(
            f"  {row['season']}: champion={row['champion']} "
            f"pred_rank={row['champion_pred_rank']} "
            f"top3={row['champion_in_top_3']} top5={row['champion_in_top_5']}"
        )


if __name__ == "__main__":
    results = evaluate_saved_survival_model()
    _print_report(results)
