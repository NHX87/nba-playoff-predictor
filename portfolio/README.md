# Portfolio Website

Public-facing project site for the NBA Playoff Predictor.

## Files

- [index.html](/Users/nick/Downloads/nba_playoff_predictor/portfolio/index.html)
- [styles.css](/Users/nick/Downloads/nba_playoff_predictor/portfolio/styles.css)
- [main.js](/Users/nick/Downloads/nba_playoff_predictor/portfolio/main.js)

## Run Locally

From repo root:

```bash
python -m http.server
```

Then open:

- `http://localhost:8000/portfolio/index.html`

## Data Hydration

`main.js` attempts to load:

- `models/trained/simulation_team_odds_current.csv`
- `models/trained/simulation_metadata.json`

If unavailable, the UI keeps safe placeholders.
