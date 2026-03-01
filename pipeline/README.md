# Pipeline Overview

`pipeline/` is the execution backbone of the project.

## Submodules

- [ingestion](/Users/nick/Downloads/nba_playoff_predictor/pipeline/ingestion/README.md): Pull/cache/load/validate raw data.
- [features](/Users/nick/Downloads/nba_playoff_predictor/pipeline/features/README.md): Build feature tables for modeling.
- [models](/Users/nick/Downloads/nba_playoff_predictor/pipeline/models/README.md): Train and evaluate predictive models.
- [agent](/Users/nick/Downloads/nba_playoff_predictor/pipeline/agent/README.md): LLM analyst functions.

## Orchestrator

Main entrypoint: [run_pipeline.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/run_pipeline.py)

```bash
# Full run including API fetches
python -m pipeline.run_pipeline

# Build from cache only
python -m pipeline.run_pipeline --skip-fetch

# Build + train survival model
python -m pipeline.run_pipeline --skip-fetch --with-model
```

## Stage Order

1. Fetch game logs (`fetch_games`)
2. Fetch player logs (`fetch_players`)
3. Load games into DuckDB (`load_db`)
4. Build series summary (`fetch_series`)
5. Build features (`physicality`, `team_stats`, `availability`, `build_features`)
6. Validate coverage (`validate`)
7. Optional model training (`models.survival`, `models.matchup_model`)
8. Optional current-season projections (`models.predict_current`, `models.simulation`)

## Output Contract

Successful pipeline run should produce:
- Valid game coverage (30 RS teams/season; 16 playoff teams/season in train window)
- Non-empty `model_features`
- Passing validation report
