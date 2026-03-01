# App Module

Streamlit frontend for viewing predictions and interacting with the AI analyst.

## File

- [main.py](/Users/nick/Downloads/nba_playoff_predictor/app/main.py)

## Current State

- UI reads app-ready analytics tables from DuckDB when available.
- Chat wiring to `pipeline.agent.analyst` is in place.
- Falls back to placeholder visuals if model outputs are missing.

## Run

```bash
streamlit run app/main.py
```

## Expected App Inputs (when fully wired)

- `app_title_odds_current`
- `app_series_predictions_current`
- `app_playoff_field_current`
- `app_play_in_current`

Generate these inputs by running:

```bash
python -m pipeline.models.predict_current
python -m pipeline.models.matchup_model
python -m pipeline.models.simulation
```
