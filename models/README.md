# Model Artifacts Directory

This folder stores trained model artifacts and evaluation outputs.

## Current Files

- `trained/survival_coxph.joblib`
- `trained/survival_metrics.json`
- `trained/survival_validation_predictions.csv`

## Regenerate

```bash
python -m pipeline.models.survival
```

## Notes

- Artifacts are generated outputs, not source code.
- Treat metrics files as snapshots tied to current `model_features` state.
