# UX Spec

Defines the UI contract for the Streamlit app.

## Scope

File:
- [app/main.py](/Users/nick/Downloads/nba_playoff_predictor/app/main.py)

Tabs:
- `Play-In Predictor / Race`
- `Playoff Predictor`
- `Team-by-Team Breakdown`

Sidebar:
- `Live / Recent Scores`

## Global UX Rules

- App is tab-first under the title hero.
- Every major section should expose metric disclosure chips:
  - `Metric`
  - `Source`
  - Context (for interpretation/smoothing assumptions)
- Fallback behavior must be explicit in text.

## Data Freshness Strip

Top metadata chips should show:
- season
- model outputs timestamp
- games snapshot timestamp
- score source (`live feed` or `fallback (local db)`)

## Sidebar Scores

Primary source:
- `pipeline.ingestion.fetch_live_scores.get_live_scoreboard()`

Fallback source:
- local `regular_season` game rows via DuckDB

Sections (grouped):
- `Live`
- `Final`
- `Upcoming`

Modes:
- `Compact Scores` (score rows only)
- `Scores + Stats` (add reb/ast/eFG + top scorer lines for started games)

## Play-In Tab

Source:
- `app_play_in_current`

Per-team card must include:
- logo + team abbreviation
- projected play-in seed (7 or 8)
- make playoffs probability

## Playoff Tab (Bracket)

Source:
- `app_series_predictions_current`

Board layout:
- western conference side
- center finals shelf
- eastern conference side

Series shelf must include:
- winner and loser with logos
- projected series scoreline (`WINNER 4 - LOSER x`)
- winner probability

Finals area:
- projected champion text
- projected champion logo under the text

## Team Tab

Selector:
- all 30 teams from `teams` table (fallback: distinct teams in current season)

Key metrics:
- title odds rank
- current title odds

Odds chart:
- current-season only
- model-implied daily title odds path
- calibrated so final point matches `app_title_odds_current.title_prob`
- smoothed with endpoint-preserving EWM
- team-color line and transparent plot background

Additional sections:
- last 10 table
- next 10 projected W-L
- player per-game leaders
