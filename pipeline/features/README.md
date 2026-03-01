# Feature Engineering Module

Transforms raw team/player game logs into model features by team-season.

## Files

- [physicality.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/features/physicality.py)
- [team_stats.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/features/team_stats.py)
- [availability.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/features/availability.py)
- [build_features.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/features/build_features.py)

## Feature Tables

- `physicality_features`
- `team_stats_features`
- `availability_features`
- `game_availability`
- `model_features` (joined output)

## Build Commands

```bash
python -m pipeline.features.physicality
python -m pipeline.features.team_stats
python -m pipeline.features.availability
python -m pipeline.features.build_features
```

## Design Notes

- `team_stats` computes regular-season to playoff deltas and prior-round depth.
- `availability` handles traded players by mapping to playoff team and using full-season profile stats.
- `build_features` joins available feature tables and creates `playoff_readiness_score`.

## Required Upstream Tables

Before running features, ensure ingestion has produced:
- `regular_season`
- `playoffs`
- `team_series_summary`
- `raw_player_logs_rs` and `raw_player_logs_po` (for availability)
