"""Compute CoxPH model scores for every team at every game date in the current season.

Produces the ``daily_model_scores`` DuckDB table, which powers the
"Historical Model Odds" chart on the team drilldown page.

Algorithm
---------
1. From the ``regular_season`` table, compute the 5 FINAL_FEATURES cumulatively
   after each game for every team.
2. At each game date, call ``model.predict_partial_hazard()`` on all 30 teams and
   convert to implied title probabilities via softmax of (1 / hazard).
3. Blend with vig-removed 2025-26 preseason Vegas odds (Basketball-Reference):
   - At game 0  → 100 % preseason prior
   - At game 30+ → 100 % in-season model output
   - Between: linear interpolation
4. Write ``daily_model_scores`` to the main DuckDB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from config.settings import DB_PATH, CURRENT_SEASON_STR, FINAL_FEATURES

logger = logging.getLogger(__name__)

SURVIVAL_MODEL_PATH = Path("models/trained/survival_coxph.joblib")

# Vig-removed implied title probabilities from 2025-26 preseason Vegas odds.
# Source: Basketball-Reference preseason odds (Oct 2025), normalized to sum = 1.
PRESEASON_PROBS: dict[str, float] = {
    "OKC": 0.2433,
    "DEN": 0.1272,
    "CLE": 0.0973,
    "NYK": 0.0827,
    "MIN": 0.0591,
    "HOU": 0.0551,
    "LAL": 0.0487,
    "LAC": 0.0435,
    "ORL": 0.0435,
    "GSW": 0.0318,
    "DET": 0.0243,
    "DAL": 0.0230,
    "PHI": 0.0202,
    "ATL": 0.0202,
    "MIL": 0.0148,
    "BOS": 0.0136,
    "SAS": 0.0123,
    "IND": 0.0082,
    "TOR": 0.0082,
    "MEM": 0.0066,
    "MIA": 0.0041,
    "NOP": 0.0027,
    "PHX": 0.0017,
    "POR": 0.0017,
    "SAC": 0.0017,
    "CHI": 0.0017,
    "CHA": 0.0008,
    "BKN": 0.0008,
    "UTA": 0.0008,
    "WAS": 0.0008,
}

# Games played until the model output is trusted 100% (preseason weight → 0).
# 40 games ≈ halfway through the season; enough data for stable feature estimates.
_BLEND_GAMES = 40


def _extract_opp(matchup: str) -> str:
    """'ATL @ MIL' → 'MIL',  'ATL vs. MIL' → 'MIL'."""
    return matchup.strip().split()[-1]


def _cumulative_features(rs_df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Return one row per (GAME_DATE, TEAM_ABBR) with cumulative FINAL_FEATURES."""
    df = rs_df[
        (rs_df["SEASON"] == season) & (rs_df["SEASON_TYPE"] == "Regular Season")
    ].copy()
    if df.empty:
        return pd.DataFrame()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    df["win"] = (df["WL"] == "W").astype(int)
    df["margin"] = df["PLUS_MINUS"].fillna(0).astype(float)
    df["efg_num"] = df["FGM"].fillna(0) + 0.5 * df["FG3M"].fillna(0)
    df["is_close"] = (df["margin"].abs() <= 5).astype(int)
    df["close_win"] = df["win"] * df["is_close"]
    df["opp_abbr"] = df["MATCHUP"].apply(_extract_opp)

    # Games played counter per team
    df["gp"] = df.groupby("TEAM_ABBR").cumcount() + 1

    # Cumulative wins — used for top-team labelling
    df["cum_wins"] = df.groupby("TEAM_ABBR")["win"].cumsum()
    df["cum_win_pct"] = df["cum_wins"] / df["gp"]

    # Define "top teams" from current (end-of-data) win%
    # This is retrospective but consistent with how the model was trained
    # (full-season win% used to label top-team games).
    current_wpct = df.groupby("TEAM_ABBR")["cum_win_pct"].last()
    top_teams: set[str] = set(current_wpct[current_wpct >= 0.60].index)
    df["vs_top"] = df["opp_abbr"].isin(top_teams).astype(int)
    df["top_win"] = df["win"] * df["vs_top"]

    # Cumulative sums
    grp = df.groupby("TEAM_ABBR")
    df["cum_close_games"] = grp["is_close"].cumsum()
    df["cum_close_wins"] = grp["close_win"].cumsum()
    df["cum_fta"] = grp["FTA"].cumsum()
    df["cum_efg_num"] = grp["efg_num"].cumsum()
    df["cum_fga"] = grp["FGA"].cumsum()
    df["cum_vs_top"] = grp["vs_top"].cumsum()
    df["cum_top_wins"] = grp["top_win"].cumsum()
    df["cum_pm"] = grp["margin"].cumsum()

    # FINAL_FEATURES — with Bayesian shrinkage on binary-rate features.
    # Without shrinkage, a 1-0 or 0-1 record makes rate features = 1.0 or 0.0,
    # which feeds extreme inputs to a model trained on full-season averages.
    _CLOSE_PRIOR = 15  # equivalent prior games at 50% close-game win%
    _TOP_PRIOR   = 10  # equivalent prior games at 50% vs-top win%
    _NET_PRIOR   = 20  # equivalent prior games at 0 net rating

    # eFG% and FTA are high-volume per game (~90 FGA, ~25 FTA) so they stabilise
    # quickly and need no prior.
    df["rs_efg_pct"] = df["cum_efg_num"] / df["cum_fga"].clip(lower=1)
    df["rs_fta"] = df["cum_fta"] / df["gp"]

    # Shrink rate features toward the league-average prior
    df["rs_close_game_win_pct"] = (
        (df["cum_close_wins"] + _CLOSE_PRIOR * 0.5)
        / (df["cum_close_games"] + _CLOSE_PRIOR)
    )
    df["rs_vs_top_teams_win_pct"] = (
        (df["cum_top_wins"] + _TOP_PRIOR * 0.5)
        / (df["cum_vs_top"] + _TOP_PRIOR)
    )
    # Shrink net rating toward 0: treat _NET_PRIOR phantom games at 0 point diff
    df["rs_net_rating"] = df["cum_pm"] / (df["gp"] + _NET_PRIOR)

    return df[["GAME_DATE", "TEAM_ABBR", "gp"] + FINAL_FEATURES]


def compute_daily_model_scores(
    con: duckdb.DuckDBPyConnection | None = None,
    season: str = CURRENT_SEASON_STR,
) -> pd.DataFrame:
    """
    Run CoxPH at each game date; write ``daily_model_scores`` to DuckDB.

    Returns the resulting DataFrame (rows: one per game-date × team).
    """
    if not SURVIVAL_MODEL_PATH.exists():
        logger.warning("survival_coxph.joblib not found — skipping daily model scores")
        return pd.DataFrame()

    artifact = joblib.load(SURVIVAL_MODEL_PATH)
    model = artifact["model"]
    feature_cols: list[str] = artifact["feature_columns"]

    close_after = con is None
    if con is None:
        con = duckdb.connect(str(DB_PATH))

    try:
        rs_df = con.execute(
            "SELECT TEAM_ABBR, SEASON, SEASON_TYPE, GAME_DATE, MATCHUP, WL, "
            "PLUS_MINUS, FGM, FGA, FG3M, FTA FROM regular_season"
        ).df()
    finally:
        if close_after:
            pass  # we'll close at the end

    feat_df = _cumulative_features(rs_df, season)
    if feat_df.empty:
        logger.warning("No regular-season data found for %s", season)
        return pd.DataFrame()

    dates = np.sort(feat_df["GAME_DATE"].unique())
    logger.info("Computing daily model scores for %d dates …", len(dates))

    records: list[dict] = []
    for date in dates:
        # Latest cumulative features per team up to this date
        snap = (
            feat_df[feat_df["GAME_DATE"] <= date]
            .sort_values("GAME_DATE")
            .groupby("TEAM_ABBR")
            .last()
            .reset_index()
        )
        if snap.empty or not all(c in snap.columns for c in feature_cols):
            continue

        hazards = model.predict_partial_hazard(snap[feature_cols])
        snap["hazard"] = hazards.values

        # Implied title prob: softmax of 1/hazard
        inv_h = 1.0 / snap["hazard"].clip(lower=1e-10)
        snap["model_title_prob"] = inv_h / inv_h.sum()

        for _, row in snap.iterrows():
            abbr = row["TEAM_ABBR"]
            gp = int(row.get("gp", 1))
            model_prob = float(row["model_title_prob"])
            preseason_prob = PRESEASON_PROBS.get(abbr, 1 / 30)

            # Blend: 100% preseason at game 0, 100% model at game _BLEND_GAMES
            w = min(1.0, gp / _BLEND_GAMES)
            blended = (1 - w) * preseason_prob + w * model_prob

            records.append(
                {
                    "SEASON": season,
                    "GAME_DATE": date,
                    "TEAM_ABBR": abbr,
                    "games_played": gp,
                    "model_title_prob": model_prob,
                    "preseason_prob": preseason_prob,
                    "blend_weight": w,
                    "title_prob_blended": blended,
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # Write to DuckDB
    con.execute("DROP TABLE IF EXISTS daily_model_scores")
    con.execute("CREATE TABLE daily_model_scores AS SELECT * FROM result")
    logger.info(
        "daily_model_scores written: %d rows (%d dates × ~30 teams)",
        len(result),
        len(dates),
    )

    if close_after:
        con.close()

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = compute_daily_model_scores()
    print(df.tail(10).to_string())
