# Model Sanity Report (2025-26)

## 0) LOYO Backtest — True Out-of-Sample Accuracy

Leave-one-year-out cross-validation across all historical seasons.
Each fold trains on 14 seasons and evaluates on the held-out season.

**Seasons evaluated:** 15 (2010-11 → 2024-25)  
**Features:** rs_vs_top_teams_win_pct, rs_net_rating, rs_close_game_win_pct, rs_fta, rs_efg_pct

### Survival Model (CoxPH)
| Metric | Value |
| --- | --- |
| Mean C-index | 0.7961 ±0.0737 |
| C-index range | 0.6353 – 0.9059 |
| Champion top-1 rate | 46.7% |
| Champion top-3 rate | 80.0% |
| Champion top-5 rate | 100.0% |
| Mean champion rank | 2.27 / 16 |

### Matchup Model (Logistic Regression)
| Metric | Value |
| --- | --- |
| Series accuracy | 72.9% (225 series) |
| Mean Brier score | 0.1826 (random baseline = 0.25) |
| ROC-AUC | 0.8022 |

**By round:**
| Round | Accuracy | Brier | n |
| --- | --- | --- | --- |
| First Round | 76.7% | 0.1547 | 120 |
| Conference Semifinals | 65.0% | 0.2346 | 60 |
| Conference Finals | 70.0% | 0.2021 | 30 |
| NBA Finals | 80.0% | 0.1587 | 15 |

**Per-season champion ranks:**
| Season | Champion | Rank | Top-3 | C-index |
| --- | --- | --- | --- | --- |
| 2010-11 | DAL | 5 | ✗ | 0.682 |
| 2011-12 | MIA | 3 | ✓ | 0.812 |
| 2012-13 | MIA | 1 | ✓ | 0.706 |
| 2013-14 | SAS | 1 | ✓ | 0.812 |
| 2014-15 | GSW | 1 | ✓ | 0.824 |
| 2015-16 | CLE | 3 | ✓ | 0.824 |
| 2016-17 | GSW | 1 | ✓ | 0.906 |
| 2017-18 | GSW | 1 | ✓ | 0.835 |
| 2018-19 | TOR | 3 | ✓ | 0.894 |
| 2019-20 | LAL | 2 | ✓ | 0.812 |
| 2020-21 | MIL | 5 | ✗ | 0.824 |
| 2021-22 | GSW | 4 | ✗ | 0.824 |
| 2022-23 | DEN | 2 | ✓ | 0.635 |
| 2023-24 | BOS | 1 | ✓ | 0.718 |
| 2024-25 | OKC | 1 | ✓ | 0.835 |

## 1) Strongest Round-1 Upset Risks

Upset risk is defined as `P(low seed beats high seed)`.

| conference | high_seed | high_team | low_seed | low_team | upset_risk | favorite | favorite_win_prob | predicted_winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| West | 4 | DEN | 5 | MIN | 0.4717 | DEN | 0.5283 | DEN |
| West | 3 | HOU | 6 | LAL | 0.3274 | HOU | 0.6726 | HOU |
| East | 4 | CLE | 5 | TOR | 0.2662 | CLE | 0.7338 | CLE |
| West | 2 | SAS | 7 | PHX | 0.1804 | SAS | 0.8196 | SAS |
| East | 2 | BOS | 7 | MIA | 0.1639 | BOS | 0.8361 | BOS |
| West | 1 | OKC | 8 | GSW | 0.1369 | OKC | 0.8631 | OKC |
| East | 3 | NYK | 6 | PHI | 0.0847 | NYK | 0.9153 | NYK |
| East | 1 | DET | 8 | ORL | 0.0618 | DET | 0.9382 | DET |

## 2) Biggest Seed vs Title-Odds Gaps

`seed_vs_title_gap_conf = playoff_seed - conference_title_rank`

- Positive gap: team is outperforming seed in model odds.
- Negative gap: team is underperforming seed in model odds.

| TEAM_ABBR | conference | playoff_seed | title_rank_conf | title_rank_overall | title_prob | seed_vs_title_gap_conf | abs_gap_conf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MIA | East | 7 | 5.0 | 13.0 | 0.0014 | 2.0 | 2.0 |
| NYK | East | 3 | 2.0 | 4.0 | 0.1299 | 1.0 | 1.0 |
| BOS | East | 2 | 3.0 | 5.0 | 0.1158 | -1.0 | 1.0 |
| GSW | West | 8 | 7.0 | 11.0 | 0.0024 | 1.0 | 1.0 |
| PHX | West | 7 | 8.0 | 12.0 | 0.0018 | -1.0 | 1.0 |
| TOR | East | 5 | 6.0 | 14.0 | 0.0007 | -1.0 | 1.0 |
| PHI | East | 6 | 7.0 | 15.0 | 0.0001 | -1.0 | 1.0 |
| OKC | West | 1 | 1.0 | 1.0 | 0.281 | 0.0 | 0.0 |
| DET | East | 1 | 1.0 | 2.0 | 0.2394 | 0.0 | 0.0 |
| SAS | West | 2 | 2.0 | 3.0 | 0.1361 | 0.0 | 0.0 |

## 3) Monte Carlo Sensitivity (5k vs 10k vs 20k)

This checks stability of title odds and ranking against simulation count.

| TEAM_ABBR | playoff_seed | title_prob_5000 | title_prob_10000 | title_prob_20000 | title_prob_max_min_spread | rank_5000 | rank_10000 | rank_20000 | rank_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OKC | 1 | 0.2774 | 0.281 | 0.2796 | 0.0036 | 1.0 | 1.0 | 1.0 | 0.0 |
| DET | 1 | 0.2388 | 0.2394 | 0.237 | 0.0023 | 2.0 | 2.0 | 2.0 | 0.0 |
| SAS | 2 | 0.1358 | 0.1361 | 0.1412 | 0.0053 | 3.0 | 3.0 | 3.0 | 0.0 |
| NYK | 3 | 0.1312 | 0.1299 | 0.1308 | 0.0013 | 4.0 | 4.0 | 4.0 | 0.0 |
| BOS | 2 | 0.1154 | 0.1158 | 0.1148 | 0.001 | 5.0 | 5.0 | 5.0 | 0.0 |
| HOU | 3 | 0.027 | 0.0258 | 0.0251 | 0.0019 | 6.0 | 6.0 | 6.0 | 0.0 |
| DEN | 4 | 0.0232 | 0.0238 | 0.023 | 0.0008 | 8.0 | 7.0 | 7.0 | 0.577 |
| CLE | 4 | 0.0244 | 0.0222 | 0.022 | 0.0024 | 7.0 | 8.0 | 8.0 | 0.577 |
| MIN | 5 | 0.0168 | 0.017 | 0.0169 | 0.0002 | 9.0 | 9.0 | 9.0 | 0.0 |
| LAL | 6 | 0.0032 | 0.0026 | 0.0026 | 0.0006 | 10.0 | 10.0 | 10.0 | 0.0 |
| GSW | 8 | 0.002 | 0.0024 | 0.0026 | 0.0006 | 12.0 | 11.0 | 11.0 | 0.577 |
| PHX | 7 | 0.0014 | 0.0018 | 0.0018 | 0.0005 | 13.0 | 12.0 | 12.0 | 0.577 |

## Artifact Files

- `models/trained/sanity_upset_risks.csv`
- `models/trained/sanity_seed_vs_odds_gap.csv`
- `models/trained/sanity_sensitivity.csv`
