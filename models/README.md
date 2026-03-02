# Model Artifacts Directory

This folder stores trained model artifacts and evaluation outputs.

## Current Files

Core artifacts (non-exhaustive):
- `trained/survival_coxph.joblib`
- `trained/survival_metrics.json`
- `trained/survival_validation_predictions.csv`
- `trained/matchup_model.joblib`
- `trained/matchup_metrics.json`
- `trained/matchup_validation_predictions.csv`
- `trained/current_season_predictions.csv`
- `trained/projected_playoff_field.csv`
- `trained/projected_first_round_matchups.csv`
- `trained/play_in_simulation_results.csv`
- `trained/series_predictions_current.csv`
- `trained/simulation_team_odds_current.csv`
- `trained/simulation_metadata.json`
- `trained/model_sanity_report.md`

## Regenerate

```bash
python -m pipeline.models.survival
python -m pipeline.models.matchup_model
python -m pipeline.models.predict_current
python -m pipeline.models.simulation
python -m pipeline.models.sanity_report
```

## Notes

- Artifacts are generated outputs, not source code.
- Treat metrics files as snapshots tied to current `model_features` state.
