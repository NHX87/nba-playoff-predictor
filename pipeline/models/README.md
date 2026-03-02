# Models Module

Model training/evaluation lives in `pipeline/models/`.

## Files

- [survival.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/models/survival.py)
- [evaluate.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/models/evaluate.py)
- [matchup_model.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/models/matchup_model.py)
- [predict_current.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/models/predict_current.py)
- [simulation.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/models/simulation.py)
- [sanity_report.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/models/sanity_report.py)

## Current Model

Implemented models:
- Cox proportional hazards survival model for playoff elimination risk.
- Logistic regression head-to-head model for playoff series outcomes.

Training input:
- `model_features` table from DuckDB
- Feature list from `config.settings.FINAL_FEATURES`

Target behavior:
- Duration: rounds reached
- Event: eliminated before championship (`rounds_reached < 5`)

## Commands

```bash
# Train and save artifacts
python -m pipeline.models.survival

# Evaluate saved artifact
python -m pipeline.models.evaluate

# Train matchup model
python -m pipeline.models.matchup_model

# Generate current-season prediction ranking
python -m pipeline.models.predict_current

# Run Monte Carlo bracket simulation + app outputs
python -m pipeline.models.simulation

# Generate model sanity report (upset risks, seed-vs-odds gap, sim-count sensitivity)
python -m pipeline.models.sanity_report
```

## Artifacts

Written to `models/trained/`:
- `survival_coxph.joblib`
- `survival_metrics.json`
- `survival_validation_predictions.csv`
- `matchup_model.joblib`
- `matchup_metrics.json`
- `matchup_validation_predictions.csv`
- `current_feature_snapshot.csv`
- `series_predictions_current.csv`
- `simulation_team_odds_current.csv`
- `simulation_metadata.json`
- `model_sanity_report.md`
- `sanity_upset_risks.csv`
- `sanity_seed_vs_odds_gap.csv`
- `sanity_sensitivity.csv`

Also written to DuckDB:
- `survival_validation_predictions`
- `matchup_validation_predictions`
- `current_season_predictions`
- `current_feature_snapshot`
- `series_predictions_current`
- `simulation_team_odds_current`

## Current-Season Prediction Notes

`predict_current.py` produces a pre-playoff proxy ranking for `CURRENT_SEASON_STR`.

Play-in regime logic is season-aware:
- `<= 2018-19`: no play-in (direct 1-8 seeds)
- `2019-20`: bubble conditional 8v9 (only if 9th is within 4 games of 8th)
- `>= 2020-21`: standard 7/8 + 9/10 play-in tournament

Because several trained features are playoff-dependent, the script:
- computes regular-season-available features from the current season,
- fills playoff-dependent features from each team's historical median,
- falls back to league medians when team history is missing.

Outputs:
- `models/trained/current_season_predictions.csv`
- `models/trained/projected_playoff_field.csv`
- `models/trained/projected_first_round_matchups.csv`
- `models/trained/play_in_simulation_results.csv`
- `current_season_predictions` table in DuckDB
- `projected_playoff_field` table in DuckDB
- `projected_first_round_matchups` table in DuckDB
- `play_in_simulation_results` table in DuckDB

`simulation.py` consumes:
- `projected_playoff_field` + `current_feature_snapshot` + `matchup_model`

and publishes app-ready tables:
- `app_title_odds_current`
- `app_series_predictions_current`
- `app_playoff_field_current`
- `app_play_in_current`

## Next Planned Models

- Historical rolling backtest module

## App Metric Mapping

This mapping is the UI contract between modeling outputs and app surfaces.

- `app_title_odds_current.title_prob`
  - used as current title odds in Team tab KPI
  - used as endpoint calibration target for team odds-over-time chart
- `app_series_predictions_current`
  - drives bracket shelves in Playoff tab
  - winner probability shown under each shelf
  - `p_4_games..p_7_games` / `expected_games` converted to projected series scoreline
- `app_play_in_current`
  - drives play-in team cards (projected seed + make-playoffs probability)
