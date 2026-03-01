# Ingestion Module

Ingestion is responsible for moving source API data into local durable storage.

## Files

- [fetch_games.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/ingestion/fetch_games.py)
- [fetch_players.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/ingestion/fetch_players.py)
- [load_db.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/ingestion/load_db.py)
- [fetch_series.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/ingestion/fetch_series.py)
- [validate.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/ingestion/validate.py)

## Responsibilities

- Pull team game logs and player logs from `nba_api`.
- Cache raw responses to parquet in `data/raw/`.
- Load game logs into DuckDB and create core views.
- Derive playoff series outcomes by season/team.
- Validate season/team completeness so missing data is visible.

## Common Commands

```bash
# Pull all configured seasons (cached)
python -m pipeline.ingestion.fetch_games

# Pull all player logs (cached)
python -m pipeline.ingestion.fetch_players

# Load cached game logs into DuckDB
python -m pipeline.ingestion.load_db

# Build team-level playoff rounds reached
python -m pipeline.ingestion.fetch_series

# Validate data coverage
python -m pipeline.ingestion.validate
```

## Input/Output

Inputs:
- `nba_api`
- `data/raw/games_*.parquet`
- `data/raw/player_logs_*.parquet`

Outputs:
- `raw_game_logs`, `regular_season`, `playoffs`
- `raw_player_logs_rs`, `raw_player_logs_po`
- `rotation_players_rs`, `rotation_players_po`
- `team_series_summary`

## Debugging Notes

- If a team is missing from a season, rerun that season in `fetch_games` and then rerun `load_db` + `validate`.
- `validate` is the fastest way to find hidden ingestion gaps.
