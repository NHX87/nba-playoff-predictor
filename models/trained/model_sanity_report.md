# Model Sanity Report (2025-26)

## 1) Strongest Round-1 Upset Risks

Upset risk is defined as `P(low seed beats high seed)`.


| conference | high_seed | high_team | low_seed | low_team | upset_risk | favorite | favorite_win_prob | predicted_winner |
| ---------- | --------- | --------- | -------- | -------- | ---------- | -------- | ----------------- | ---------------- |
| West       | 4         | DEN       | 5        | MIN      | 0.7895     | MIN      | 0.7895            | MIN              |
| East       | 4         | CLE       | 5        | TOR      | 0.7407     | TOR      | 0.7407            | TOR              |
| West       | 2         | SAS       | 7        | GSW      | 0.6093     | GSW      | 0.6093            | GSW              |
| West       | 1         | OKC       | 8        | LAC      | 0.5879     | LAC      | 0.5879            | LAC              |
| East       | 2         | BOS       | 7        | MIA      | 0.4915     | BOS      | 0.5085            | BOS              |
| East       | 3         | NYK       | 6        | PHI      | 0.3461     | NYK      | 0.6539            | NYK              |
| West       | 3         | HOU       | 6        | LAL      | 0.1426     | HOU      | 0.8574            | HOU              |
| East       | 1         | DET       | 8        | ORL      | 0.0859     | DET      | 0.9141            | DET              |


## 2) Biggest Seed vs Title-Odds Gaps

`seed_vs_title_gap_conf = playoff_seed - conference_title_rank`

- Positive gap: team is outperforming seed in model odds.
- Negative gap: team is underperforming seed in model odds.


| TEAM_ABBR | conference | playoff_seed | title_rank_conf | title_rank_overall | title_prob | seed_vs_title_gap_conf | abs_gap_conf |
| --------- | ---------- | ------------ | --------------- | ------------------ | ---------- | ---------------------- | ------------ |
| OKC       | West       | 1            | 7.0             | 14.0               | 0.002      | -6.0                   | 6.0          |
| GSW       | West       | 7            | 2.0             | 5.0                | 0.0334     | 5.0                    | 5.0          |
| TOR       | East       | 5            | 1.0             | 2.0                | 0.176      | 4.0                    | 4.0          |
| DEN       | West       | 4            | 8.0             | 15.0               | 0.0003     | -4.0                   | 4.0          |
| LAL       | West       | 6            | 3.0             | 10.0               | 0.0129     | 3.0                    | 3.0          |
| HOU       | West       | 3            | 1.0             | 1.0                | 0.4577     | 2.0                    | 2.0          |
| BOS       | East       | 2            | 4.0             | 5.0                | 0.0334     | -2.0                   | 2.0          |
| MIA       | East       | 7            | 5.0             | 7.0                | 0.0284     | 2.0                    | 2.0          |
| CLE       | East       | 4            | 6.0             | 8.0                | 0.0144     | -2.0                   | 2.0          |
| SAS       | West       | 2            | 4.0             | 11.0               | 0.0092     | -2.0                   | 2.0          |


## 3) Monte Carlo Sensitivity (5k vs 10k vs 20k)

This checks stability of title odds and ranking against simulation count.


| TEAM_ABBR | playoff_seed | title_prob_5000 | title_prob_10000 | title_prob_20000 | title_prob_max_min_spread | rank_5000 | rank_10000 | rank_20000 | rank_std |
| --------- | ------------ | --------------- | ---------------- | ---------------- | ------------------------- | --------- | ---------- | ---------- | -------- |
| HOU       | 3            | 0.456           | 0.4577           | 0.4584           | 0.0025                    | 1.0       | 1.0        | 1.0        | 0.0      |
| TOR       | 5            | 0.1744          | 0.176            | 0.1755           | 0.0016                    | 2.0       | 2.0        | 2.0        | 0.0      |
| DET       | 1            | 0.1386          | 0.1383           | 0.1353           | 0.0033                    | 3.0       | 3.0        | 3.0        | 0.0      |
| NYK       | 3            | 0.0688          | 0.0679           | 0.0684           | 0.0009                    | 4.0       | 4.0        | 4.0        | 0.0      |
| BOS       | 2            | 0.0352          | 0.0334           | 0.0332           | 0.002                     | 5.0       | 5.0        | 5.0        | 0.0      |
| GSW       | 7            | 0.0326          | 0.0334           | 0.0328           | 0.0008                    | 6.0       | 5.0        | 6.0        | 0.577    |
| MIA       | 7            | 0.03            | 0.0284           | 0.0292           | 0.0016                    | 7.0       | 7.0        | 7.0        | 0.0      |
| CLE       | 4            | 0.014           | 0.0144           | 0.0147           | 0.0007                    | 8.0       | 8.0        | 8.0        | 0.0      |
| PHI       | 6            | 0.0136          | 0.0131           | 0.0133           | 0.0005                    | 9.0       | 9.0        | 9.0        | 0.0      |
| LAL       | 6            | 0.0132          | 0.0129           | 0.013            | 0.0003                    | 10.0      | 10.0       | 10.0       | 0.0      |
| SAS       | 2            | 0.0086          | 0.0092           | 0.0094           | 0.0008                    | 11.0      | 11.0       | 11.0       | 0.0      |
| MIN       | 5            | 0.0074          | 0.0082           | 0.0087           | 0.0013                    | 12.0      | 12.0       | 12.0       | 0.0      |


## Artifact Files

- `models/trained/sanity_upset_risks.csv`
- `models/trained/sanity_seed_vs_odds_gap.csv`
- `models/trained/sanity_sensitivity.csv`

