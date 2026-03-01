# Model Sanity Report (2025-26)

## 1) Strongest Round-1 Upset Risks

Upset risk is defined as `P(low seed beats high seed)`.

| conference | high_seed | high_team | low_seed | low_team | upset_risk | favorite | favorite_win_prob | predicted_winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| West | 4 | DEN | 5 | MIN | 0.9653 | MIN | 0.9653 | MIN |
| West | 1 | OKC | 8 | PHX | 0.9598 | PHX | 0.9598 | PHX |
| East | 4 | CLE | 5 | TOR | 0.8694 | TOR | 0.8694 | TOR |
| East | 2 | BOS | 7 | MIA | 0.4861 | BOS | 0.5139 | BOS |
| West | 2 | SAS | 7 | GSW | 0.4551 | SAS | 0.5449 | SAS |
| East | 3 | NYK | 6 | PHI | 0.0966 | NYK | 0.9034 | NYK |
| East | 1 | DET | 8 | ORL | 0.0296 | DET | 0.9704 | DET |
| West | 3 | HOU | 6 | LAL | 0.0127 | HOU | 0.9873 | HOU |

## 2) Biggest Seed vs Title-Odds Gaps

`seed_vs_title_gap_conf = playoff_seed - conference_title_rank`

- Positive gap: team is outperforming seed in model odds.
- Negative gap: team is underperforming seed in model odds.

| TEAM_ABBR | conference | playoff_seed | title_rank_conf | title_rank_overall | title_prob | seed_vs_title_gap_conf | abs_gap_conf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PHX | West | 8 | 2.0 | 3.0 | 0.0935 | 6.0 | 6.0 |
| OKC | West | 1 | 4.0 | 10.0 | 0.0 | -3.0 | 3.0 |
| GSW | West | 7 | 4.0 | 10.0 | 0.0 | 3.0 | 3.0 |
| CLE | East | 4 | 7.0 | 10.0 | 0.0 | -3.0 | 3.0 |
| HOU | West | 3 | 1.0 | 2.0 | 0.2626 | 2.0 | 2.0 |
| TOR | East | 5 | 3.0 | 5.0 | 0.0224 | 2.0 | 2.0 |
| BOS | East | 2 | 4.0 | 6.0 | 0.0012 | -2.0 | 2.0 |
| MIA | East | 7 | 5.0 | 7.0 | 0.0008 | 2.0 | 2.0 |
| ORL | East | 8 | 6.0 | 8.0 | 0.0002 | 2.0 | 2.0 |
| LAL | West | 6 | 4.0 | 10.0 | 0.0 | 2.0 | 2.0 |

## 3) Monte Carlo Sensitivity (5k vs 10k vs 20k)

This checks stability of title odds and ranking against simulation count.

| TEAM_ABBR | playoff_seed | title_prob_5000 | title_prob_10000 | title_prob_20000 | title_prob_max_min_spread | rank_5000 | rank_10000 | rank_20000 | rank_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DET | 1 | 0.5946 | 0.5961 | 0.5952 | 0.0015 | 1.0 | 1.0 | 1.0 | 0.0 |
| HOU | 3 | 0.2628 | 0.2626 | 0.2666 | 0.004 | 2.0 | 2.0 | 2.0 | 0.0 |
| PHX | 8 | 0.0954 | 0.0935 | 0.0922 | 0.0033 | 3.0 | 3.0 | 3.0 | 0.0 |
| NYK | 3 | 0.0212 | 0.0231 | 0.0213 | 0.0019 | 5.0 | 4.0 | 5.0 | 0.577 |
| TOR | 5 | 0.024 | 0.0224 | 0.0222 | 0.0018 | 4.0 | 5.0 | 4.0 | 0.577 |
| BOS | 2 | 0.001 | 0.0012 | 0.001 | 0.0002 | 6.0 | 6.0 | 7.0 | 0.577 |
| MIA | 7 | 0.0008 | 0.0008 | 0.0012 | 0.0003 | 7.0 | 7.0 | 6.0 | 0.577 |
| ORL | 8 | 0.0 | 0.0002 | 0.0002 | 0.0002 | 9.0 | 8.0 | 8.0 | 0.577 |
| SAS | 2 | 0.0002 | 0.0001 | 0.0002 | 0.0002 | 8.0 | 9.0 | 8.0 | 0.577 |
| CLE | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 9.0 | 10.0 | 11.0 | 1.0 |
| DEN | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 9.0 | 10.0 | 11.0 | 1.0 |
| GSW | 7 | 0.0 | 0.0 | 0.0 | 0.0 | 9.0 | 10.0 | 10.0 | 0.577 |

## Artifact Files

- `models/trained/sanity_upset_risks.csv`
- `models/trained/sanity_seed_vs_odds_gap.csv`
- `models/trained/sanity_sensitivity.csv`
