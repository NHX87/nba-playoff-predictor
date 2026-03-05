"""
series_length.py
----------------
Compute per-series game-length probabilities from a series win probability.

Given p_series = P(higher-seeded team wins the series), back-calculates
the implied per-game win probability p_game via numerical root-finding,
then derives P(series goes exactly N games) for N ∈ {4, 5, 6, 7}.

Mathematical basis:
  P(team A wins series) = Σ_{k=4}^{7} C(k-1, 3) * p^4 * (1-p)^(k-4)
  P(series goes N games) = C(N-1, 3) * [p^4*(1-p)^(N-4) + (1-p)^4*p^(N-4)]

where p = p_game (per-game win probability of the higher-seeded team).
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def _series_win_prob(p_game: float) -> float:
    """P(team with per-game win probability p wins a best-of-7 series)."""
    q = 1.0 - p_game
    return p_game**4 * (1.0 + 4.0 * q + 10.0 * q**2 + 20.0 * q**3)


def _series_length_probs(p_game: float) -> dict[int, float]:
    """P(series lasts exactly N games) for N in {4, 5, 6, 7}."""
    q = 1.0 - p_game
    return {
        n: comb(n - 1, 3) * (p_game**4 * q ** (n - 4) + q**4 * p_game ** (n - 4))
        for n in range(4, 8)
    }


def _back_calculate_p_game(p_series: float) -> float:
    """
    Given P(team A wins the series), find the implied per-game win probability.
    Uses Brent's root-finding. p_series is clamped to [0.001, 0.999].
    At p_series = 0.5, p_game = 0.5 exactly (symmetric series).
    """
    p_series = float(np.clip(p_series, 0.001, 0.999))
    if abs(p_series - 0.5) < 1e-9:
        return 0.5
    lo, hi = (0.5, 0.9999) if p_series > 0.5 else (0.0001, 0.5)
    return float(brentq(lambda p: _series_win_prob(p) - p_series, lo, hi))


def add_series_length_cols(
    df: pd.DataFrame,
    win_prob_col: str = "high_team_win_prob",
) -> pd.DataFrame:
    """
    Enrich a series-predictions DataFrame with game-count probability columns.

    Adds columns:
      p_4_games, p_5_games, p_6_games, p_7_games — P(series lasts N games)
      expected_games                               — E[series length]

    Args:
        df: DataFrame with a series win probability column.
        win_prob_col: Column containing P(higher-seeded team wins series).
    """
    p4s, p5s, p6s, p7s, exps = [], [], [], [], []

    for p_series in df[win_prob_col]:
        p_game = _back_calculate_p_game(float(p_series))
        lp = _series_length_probs(p_game)
        p4s.append(round(lp[4], 4))
        p5s.append(round(lp[5], 4))
        p6s.append(round(lp[6], 4))
        p7s.append(round(lp[7], 4))
        exps.append(round(sum(n * lp[n] for n in range(4, 8)), 2))

    out = df.copy()
    out["p_4_games"] = p4s
    out["p_5_games"] = p5s
    out["p_6_games"] = p6s
    out["p_7_games"] = p7s
    out["expected_games"] = exps
    out["most_likely_games"] = (
        out[["p_4_games", "p_5_games", "p_6_games", "p_7_games"]]
        .idxmax(axis=1)
        .str.extract(r"(\d)_games")[0]
        .astype(int)
    )
    return out
