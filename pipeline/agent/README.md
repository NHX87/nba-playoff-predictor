# Agent Module

LLM analyst layer for translating model outputs into basketball explanations.

## File

- [analyst.py](/Users/nick/Downloads/nba_playoff_predictor/pipeline/agent/analyst.py)

## What It Does

- Team scouting report generation.
- Conversational Q&A over current model context.
- Explanations for odds shifts when slider/context changes.

## Inputs

- `ANTHROPIC_API_KEY` from `.env`
- Runtime model context from app/pipeline (favorites, features, settings)

## Usage Notes

- Keep context payloads explicit and current.
- Treat this module as interpretation, not source-of-truth modeling.
