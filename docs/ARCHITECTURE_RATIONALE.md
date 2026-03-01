# Architecture Rationale

This document explains why the codebase is structured the way it is, what each boundary protects, and how data flows end-to-end.

## System Goal

Build a reproducible, inspectable NBA playoff prediction system where:
- data ingestion is robust to API instability,
- feature engineering is explicit and testable,
- model training is repeatable,
- app behavior is traceable back to data/model artifacts.

## Design Principles

- Separation of concerns: ingestion, features, models, app, and agent are independent modules.
- Re-runnable stages: each stage can be run without re-running the whole stack.
- Durable intermediate state: parquet caches + DuckDB tables preserve progress and enable debugging.
- Fail-loud data quality: validation checks surface hidden data gaps (missing teams/seasons).
- Explainability-first: each prediction should be traceable to concrete feature values.

## Module Rationale

## `config/`

Purpose:
- Centralize environment-driven settings and project constants.

Why this boundary exists:
- Prevents hard-coded values from being scattered across modules.
- Makes behavior changes (season window, DB path, feature list) predictable.

Inputs:
- `.env` variables.

Outputs:
- Shared constants like `DB_PATH`, `FINAL_FEATURES`, `TARGET`.

Invariants:
- Config should load without side effects beyond env parsing.

Failure modes:
- Missing/incorrect env var causes downstream path/model issues.

Tradeoff:
- Tight central config is simple; per-module config would be more flexible but harder to govern.

## `pipeline/ingestion/`

Purpose:
- Pull source data from `nba_api`, cache to parquet, and materialize canonical raw tables/views in DuckDB.

Why this boundary exists:
- External API behavior is the least reliable part of the system.
- Isolating retries/caching/coverage checks keeps downstream logic deterministic.

Inputs:
- `nba_api` endpoints
- Existing parquet caches in `data/raw/`

Outputs:
- `raw_game_logs`, `regular_season`, `playoffs`
- `raw_player_logs_rs`, `raw_player_logs_po`
- `rotation_players_rs`, `rotation_players_po`
- `team_series_summary`

Invariants:
- Training seasons should have 30 RS teams and 16 playoff teams.
- `regular_season`/`playoffs` views must match `raw_game_logs` content.

Failure modes:
- API returns malformed/empty payloads for a team.
- Cached season missing a team creates silent model skew unless validated.

Tradeoff:
- Full refresh is expensive but simple.
- Incremental updates are faster but add complexity around idempotency and versioning.

## `pipeline/features/`

Purpose:
- Convert raw team/player logs into model-ready team-season features.

Why this boundary exists:
- Feature logic is domain-heavy and should evolve independently of ingestion and model code.
- Keeps statistical transformations explicit and debuggable.

Inputs:
- `regular_season`, `playoffs`, `team_series_summary`
- player log tables for availability features

Outputs:
- `physicality_features`
- `team_stats_features`
- `availability_features`
- `game_availability`
- `model_features`

Invariants:
- One row per playoff team-season in `model_features`.
- Required model columns are present and numeric.

Failure modes:
- Missing upstream team data drops rows in join outputs.
- Null-heavy feature columns degrade model trainability.

Tradeoff:
- Separate scripts are easier to reason about; a single monolith might be faster to run but harder to debug.

## `pipeline/models/`

Purpose:
- Train and evaluate predictive models using stable feature tables.

Why this boundary exists:
- Modeling assumptions (target construction, censoring, splits, metrics) should be isolated and auditable.

Inputs:
- `model_features` table in DuckDB
- `config.settings.FINAL_FEATURES`

Outputs:
- model artifacts in `models/trained/`
- evaluation outputs (`survival_metrics.json`, prediction CSV, DuckDB validation table)

Invariants:
- Time-based split must be deterministic.
- Artifact schema must remain stable for downstream app use.

Failure modes:
- Target/feature schema mismatch.
- Data leakage via incorrect split strategy.

Tradeoff:
- Current approach favors clarity over maximal model complexity.

## `pipeline/agent/`

Purpose:
- Translate model outputs into natural-language analysis.

Why this boundary exists:
- Interpretation concerns are separate from prediction concerns.
- Keeps LLM variability from contaminating core model pipeline.

Inputs:
- Model context (odds/features/current settings)
- Anthropic API key + model config

Outputs:
- Analyst answers/explanations

Invariants:
- Agent responses are advisory; source-of-truth remains DB/model artifacts.

Failure modes:
- Missing API key.
- Hallucinated explanations if context payload is weak.

Tradeoff:
- Strong UX value, but introduces external dependency and prompt-maintenance overhead.

## `app/`

Purpose:
- Present model state and allow interactive exploration.

Why this boundary exists:
- UI should consume stable outputs, not own data transformation logic.

Inputs:
- DuckDB outputs and model artifacts
- user interaction state

Outputs:
- visualizations, controls, and analyst chat UX

Invariants:
- App should degrade gracefully when artifacts/data are missing.

Failure modes:
- Placeholder data diverges from model outputs.
- Agent errors surface if key/config missing.

Tradeoff:
- Fast prototyping with Streamlit vs finer control in a full frontend stack.

## One Row Journey (Traceability Walkthrough)

Example: `DEN`, season `2022-23`.

1. Ingestion:
- `fetch_games.py` pulls Denver regular-season and playoff game logs from `nba_api`.
- Cached in `data/raw/games_2022-23_Regular_Season.parquet` and `..._Playoffs.parquet`.

2. Warehouse load:
- `load_db.py` loads all `games_*.parquet` into `raw_game_logs`.
- Views split rows into `regular_season` and `playoffs`.

3. Series outcome derivation:
- `fetch_series.py` parses playoff `GAME_ID` groups to infer series wins.
- Writes `team_series_summary` with `rounds_reached`.

4. Feature engineering:
- `team_stats.py` computes Denver RS->PO deltas (e.g., `ppg_delta`, `ts_pct_delta`) and playoff depth fields.
- `availability.py` computes rotation availability/lineup quality features.
- `physicality.py` computes physicality-related deltas.

5. Model feature assembly:
- `build_features.py` joins feature tables into one Denver 2022-23 row in `model_features`.
- Derived composite `playoff_readiness_score` is added.

6. Validation:
- `validate.py` confirms season/team coverage and model-feature row completeness.

7. Model training/eval:
- `models/survival.py` uses Denver’s row with peers in time-based split.
- Produces elimination risk and season ranking in validation outputs.

8. App + agent consumption:
- `app/main.py` reads outputs for visualization.
- `agent/analyst.py` turns Denver’s numerical profile into narrative explanation.

## Why This Structure Works for Portfolio Review

- You can explain any number in the app back to source rows.
- Data reliability problems are caught by explicit validation.
- Model runs are reproducible from stable intermediate tables.
- Boundaries show production thinking, not notebook-only experimentation.

## Known Gaps and Planned Evolution

- Add per-stage unit/integration tests for contracts.
- Add rolling backtests and calibration reports.
- Replace remaining app placeholders with live artifact/DB reads.
- Add automated run metadata logging (`pipeline_run_log`) for observability.
