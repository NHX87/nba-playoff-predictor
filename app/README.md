# App Module

Streamlit frontend for viewing predictions and interacting with the AI analyst.

## File

- [main.py](/Users/nick/Downloads/nba_playoff_predictor/app/main.py)

## Current State

- UI reads app-ready analytics tables from DuckDB and can also pull live scoreboard data from `nba_api.live`.
- Current navigation is tab-first:
  - `Play-In Predictor / Race`
  - `Playoff Predictor`
  - `Team-by-Team Breakdown`
- Sidebar is dedicated to `Live / Recent Scores` with mode toggle:
  - `Compact Scores`
  - `Scores + Stats`
- Each major section includes data/metric disclosure chips (`Metric`, `Source`, and context notes).
- Team `Odds Over Time (Current Season)` is model-implied and calibrated so the latest point matches current title odds.

## Run

```bash
streamlit run app/main.py
```

## Companion Portfolio Site

A separate portfolio website lives in [../portfolio](/Users/nick/Downloads/nba_playoff_predictor/portfolio/README.md) and is intended to host this project publicly.

## UX Contract

Reference UI behavior and metric semantics:

- [docs/UX_SPEC.md](/Users/nick/Downloads/nba_playoff_predictor/docs/UX_SPEC.md)

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

Optional live score feed (runtime fetch from NBA live endpoint):

```bash
python -m pipeline.ingestion.fetch_live_scores
```
