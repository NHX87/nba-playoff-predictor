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
| West | 4 | DEN | 5 | HOU | 0.4747 | DEN | 0.5253 | DEN |
| West | 3 | MIN | 6 | LAL | 0.311 | MIN | 0.689 | MIN |
| East | 4 | CLE | 5 | TOR | 0.2131 | CLE | 0.7869 | CLE |
| East | 1 | DET | 8 | CHA | 0.2087 | DET | 0.7913 | DET |
| East | 2 | BOS | 7 | MIA | 0.1917 | BOS | 0.8083 | BOS |
| West | 2 | SAS | 7 | PHX | 0.1703 | SAS | 0.8297 | SAS |
| West | 1 | OKC | 8 | GSW | 0.1359 | OKC | 0.8641 | OKC |
| East | 3 | NYK | 6 | PHI | 0.0684 | NYK | 0.9316 | NYK |

## 2) Biggest Seed vs Title-Odds Gaps

`seed_vs_title_gap_conf = playoff_seed - conference_title_rank`

- Positive gap: team is outperforming seed in model odds.
- Negative gap: team is underperforming seed in model odds.

| TEAM_ABBR | conference | playoff_seed | title_rank_conf | title_rank_overall | title_prob | seed_vs_title_gap_conf | abs_gap_conf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CHA | East | 8 | 5.0 | 11.0 | 0.0037 | 3.0 | 3.0 |
| NYK | East | 3 | 1.0 | 2.0 | 0.1533 | 2.0 | 2.0 |
| DET | East | 1 | 3.0 | 5.0 | 0.1183 | -2.0 | 2.0 |
| GSW | West | 8 | 6.0 | 10.0 | 0.0038 | 2.0 | 2.0 |
| TOR | East | 5 | 7.0 | 15.0 | 0.0006 | -2.0 | 2.0 |
| PHI | East | 6 | 8.0 | 16.0 | 0.0 | -2.0 | 2.0 |
| LAL | West | 6 | 7.0 | 12.0 | 0.0034 | -1.0 | 1.0 |
| MIA | East | 7 | 6.0 | 13.0 | 0.0023 | 1.0 | 1.0 |
| PHX | West | 7 | 8.0 | 14.0 | 0.0013 | -1.0 | 1.0 |
| OKC | West | 1 | 1.0 | 1.0 | 0.3324 | 0.0 | 0.0 |

## 3) Monte Carlo Sensitivity (5k vs 10k vs 20k)

This checks stability of title odds and ranking against simulation count.

| TEAM_ABBR | playoff_seed | title_prob_5000 | title_prob_10000 | title_prob_20000 | title_prob_max_min_spread | rank_5000 | rank_10000 | rank_20000 | rank_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OKC | 1 | 0.3298 | 0.3324 | 0.3304 | 0.0026 | 1.0 | 1.0 | 1.0 | 0.0 |
| NYK | 3 | 0.1526 | 0.1533 | 0.1536 | 0.001 | 2.0 | 2.0 | 2.0 | 0.0 |
| SAS | 2 | 0.142 | 0.1448 | 0.1496 | 0.0077 | 3.0 | 3.0 | 3.0 | 0.0 |
| BOS | 2 | 0.1246 | 0.1203 | 0.117 | 0.0076 | 4.0 | 4.0 | 5.0 | 0.577 |
| DET | 1 | 0.1172 | 0.1183 | 0.1182 | 0.0011 | 5.0 | 5.0 | 4.0 | 0.577 |
| CLE | 4 | 0.0422 | 0.041 | 0.0419 | 0.0012 | 6.0 | 6.0 | 6.0 | 0.0 |
| MIN | 3 | 0.0376 | 0.0364 | 0.0362 | 0.0014 | 7.0 | 7.0 | 7.0 | 0.0 |
| DEN | 4 | 0.0208 | 0.0215 | 0.0215 | 0.0007 | 8.0 | 8.0 | 8.0 | 0.0 |
| HOU | 5 | 0.0164 | 0.0169 | 0.0167 | 0.0005 | 9.0 | 9.0 | 9.0 | 0.0 |
| GSW | 8 | 0.003 | 0.0038 | 0.0036 | 0.0008 | 13.0 | 10.0 | 10.0 | 1.732 |
| CHA | 8 | 0.0046 | 0.0037 | 0.0032 | 0.0014 | 10.0 | 11.0 | 11.0 | 0.577 |
| LAL | 6 | 0.0042 | 0.0034 | 0.0032 | 0.001 | 11.0 | 12.0 | 11.0 | 0.577 |

## Artifact Files

- `models/trained/sanity_upset_risks.csv`
- `models/trained/sanity_seed_vs_odds_gap.csv`
- `models/trained/sanity_sensitivity.csv`
