"""
remaining_schedule.py
---------------------
Compute per-game win probabilities for all remaining regular-season games,
then run a Monte Carlo simulation to project final records and seed distributions.

Outputs two DuckDB tables:
  team_remaining_games   — one row per (team, remaining game) with win probability
  team_projected_record  — per-team projected final record + seed probability buckets

Algorithm:
  Per-game win probability (logistic):
    p_home_win = expit(HOME_ADV + (home_net_rating - away_net_rating) * K)
    HOME_ADV = 0.10  →  equal-strength teams: 52.5% home win rate
    K = 0.05         →  10-pt net rating gap ≈ 62% win prob for home team

  Monte Carlo (N_SIMS = 5,000):
    Sample all remaining games simultaneously (vectorized Bernoulli draws)
    Accumulate wins per team per simulation
    Rank within conference → seed distribution
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit  # logistic function

from config.settings import CURRENT_SEASON_STR, DB_PATH
from pipeline.ingestion.fetch_schedule import fetch_remaining_schedule

N_SIMS = 5_000
HOME_ADV = 0.10
K = 0.05
RANDOM_STATE = 42


def _load_current_records(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load current W/L records and conference from current_season_predictions."""
    return con.execute(
        f"""
        SELECT TEAM_ABBR, conference, wins, losses, win_pct
        FROM current_season_predictions
        WHERE SEASON = '{CURRENT_SEASON_STR}'
        ORDER BY TEAM_ABBR
        """
    ).df()


def _load_net_ratings(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """Load current-season net ratings from current_feature_snapshot."""
    df = con.execute(
        f"""
        SELECT TEAM_ABBR, rs_net_rating
        FROM current_feature_snapshot
        WHERE SEASON = '{CURRENT_SEASON_STR}'
        """
    ).df()
    return dict(zip(df["TEAM_ABBR"], df["rs_net_rating"]))


def build_remaining_schedule() -> None:
    """Main entry point: build team_remaining_games and team_projected_record."""
    print("\nBuilding remaining schedule projections…")

    # --- Load schedule ---
    sched = fetch_remaining_schedule(CURRENT_SEASON_STR)
    remaining = sched[~sched["IS_PLAYED"]].copy().reset_index(drop=True)
    if remaining.empty:
        print("  No remaining games found — skipping remaining schedule step.")
        return
    print(f"  {len(remaining)} games remaining in regular season")

    # --- Load team data ---
    con = duckdb.connect(DB_PATH)
    records_df = _load_current_records(con)
    net_ratings = _load_net_ratings(con)

    if records_df.empty:
        print("  No current_season_predictions found — run predict_current_season first.")
        con.close()
        return

    # Fill missing net ratings with league average (0.0)
    all_teams = set(remaining["HOME_TEAM_ABBR"]) | set(remaining["AWAY_TEAM_ABBR"])
    for t in all_teams:
        if t not in net_ratings:
            net_ratings[t] = 0.0

    # --- Compute per-game win probabilities ---
    remaining["HOME_NET_RATING"] = remaining["HOME_TEAM_ABBR"].map(net_ratings).fillna(0.0)
    remaining["AWAY_NET_RATING"] = remaining["AWAY_TEAM_ABBR"].map(net_ratings).fillna(0.0)
    remaining["HOME_WIN_PROB"] = expit(
        HOME_ADV + (remaining["HOME_NET_RATING"] - remaining["AWAY_NET_RATING"]) * K
    )
    remaining["AWAY_WIN_PROB"] = 1.0 - remaining["HOME_WIN_PROB"]

    # --- Build team_remaining_games (two rows per game: home + away perspective) ---
    home_view = remaining[["GAME_ID", "GAME_DATE", "HOME_TEAM_ABBR", "AWAY_TEAM_ABBR", "HOME_WIN_PROB", "AWAY_NET_RATING"]].copy()
    home_view.columns = ["GAME_ID", "GAME_DATE", "TEAM_ABBR", "OPP_ABBR", "GAME_WIN_PROB", "OPP_NET_RATING"]
    home_view["IS_HOME"] = True

    away_view = remaining[["GAME_ID", "GAME_DATE", "AWAY_TEAM_ABBR", "HOME_TEAM_ABBR", "AWAY_WIN_PROB", "HOME_NET_RATING"]].copy()
    away_view.columns = ["GAME_ID", "GAME_DATE", "TEAM_ABBR", "OPP_ABBR", "GAME_WIN_PROB", "OPP_NET_RATING"]
    away_view["IS_HOME"] = False

    team_remaining_games = pd.concat([home_view, away_view], ignore_index=True)
    team_remaining_games["SEASON"] = CURRENT_SEASON_STR
    team_remaining_games = team_remaining_games.sort_values(["TEAM_ABBR", "GAME_DATE"])

    print(f"  Built team_remaining_games: {len(team_remaining_games)} team-game rows")

    # --- Monte Carlo simulation ---
    team_list = sorted(records_df["TEAM_ABBR"].tolist())
    team_idx = {t: i for i, t in enumerate(team_list)}
    n_teams = len(team_list)
    n_games = len(remaining)

    # p_home_win array for all remaining games
    p_home = remaining["HOME_WIN_PROB"].values  # shape (n_games,)

    # home_team_idx and away_team_idx for each game
    home_idxs = np.array([team_idx.get(t, -1) for t in remaining["HOME_TEAM_ABBR"]])
    away_idxs = np.array([team_idx.get(t, -1) for t in remaining["AWAY_TEAM_ABBR"]])

    # Mask valid games (both teams in our 30-team universe)
    valid = (home_idxs >= 0) & (away_idxs >= 0)
    p_home = p_home[valid]
    home_idxs = home_idxs[valid]
    away_idxs = away_idxs[valid]
    n_valid_games = valid.sum()
    print(f"  Running {N_SIMS:,} simulations over {n_valid_games} valid games…")

    rng = np.random.default_rng(RANDOM_STATE)

    # (N_SIMS, n_valid_games): True = home team wins
    outcomes = rng.random((N_SIMS, n_valid_games)) < p_home[np.newaxis, :]

    # Accumulate additional wins: (N_SIMS, n_teams)
    additional_wins = np.zeros((N_SIMS, n_teams), dtype=np.int16)
    for g in range(n_valid_games):
        additional_wins[:, home_idxs[g]] += outcomes[:, g].astype(np.int16)
        additional_wins[:, away_idxs[g]] += (~outcomes[:, g]).astype(np.int16)

    # Current wins for each team (matching team_list order)
    current_wins_arr = np.array(
        [records_df.loc[records_df["TEAM_ABBR"] == t, "wins"].iloc[0]
         if t in records_df["TEAM_ABBR"].values else 0
         for t in team_list],
        dtype=np.int16,
    )

    final_wins = current_wins_arr[np.newaxis, :] + additional_wins  # (N_SIMS, n_teams)

    # Conference membership: assign conference to each team index
    conf_map = dict(zip(records_df["TEAM_ABBR"], records_df["conference"]))
    west_mask = np.array([conf_map.get(t, "West") == "West" for t in team_list])
    east_mask = ~west_mask

    # Win-pct tiebreaker (static, for consistent ranking within sims)
    win_pct_arr = np.array(
        [records_df.loc[records_df["TEAM_ABBR"] == t, "win_pct"].iloc[0]
         if t in records_df["TEAM_ABBR"].values else 0.5
         for t in team_list]
    )

    # For each sim: rank within conference (higher wins = lower seed number)
    # Use (-final_wins * 1000 - win_pct_tiebreak) for consistent argsort
    tiebreak = win_pct_arr[np.newaxis, :] * 0.001  # tiny, won't override win count
    sort_key = -(final_wins.astype(float) + tiebreak)  # (N_SIMS, n_teams)

    # Seeds within West (15 teams)
    west_indices = np.where(west_mask)[0]
    east_indices = np.where(east_mask)[0]

    final_seeds = np.zeros((N_SIMS, n_teams), dtype=np.int8)

    # Rank West teams
    west_sort_key = sort_key[:, west_indices]  # (N_SIMS, 15)
    west_ranks = np.argsort(np.argsort(west_sort_key, axis=1), axis=1) + 1  # 1-indexed
    final_seeds[:, west_indices] = west_ranks

    # Rank East teams
    east_sort_key = sort_key[:, east_indices]  # (N_SIMS, 15)
    east_ranks = np.argsort(np.argsort(east_sort_key, axis=1), axis=1) + 1
    final_seeds[:, east_indices] = east_ranks

    print(f"  Simulation complete. Aggregating results…")

    # --- Build team_projected_record ---
    rows = []
    current_losses_map = dict(zip(records_df["TEAM_ABBR"], records_df["losses"]))
    current_wins_map = dict(zip(records_df["TEAM_ABBR"], records_df["wins"]))
    games_played_map = {t: current_wins_map.get(t, 0) + current_losses_map.get(t, 0) for t in team_list}

    total_games = 82
    for i, team in enumerate(team_list):
        cur_w = int(current_wins_map.get(team, 0))
        cur_l = int(current_losses_map.get(team, 0))
        games_rem = total_games - cur_w - cur_l

        team_final_wins = final_wins[:, i].astype(float)
        team_seeds = final_seeds[:, i]

        exp_add_w = float(additional_wins[:, i].mean())
        exp_final_w = float(team_final_wins.mean())
        exp_final_l = float(total_games - exp_final_w)

        rows.append({
            "TEAM_ABBR": team,
            "SEASON": CURRENT_SEASON_STR,
            "CONFERENCE": conf_map.get(team, "Unknown"),
            "current_wins": cur_w,
            "current_losses": cur_l,
            "games_remaining": games_rem,
            "expected_additional_wins": round(exp_add_w, 2),
            "expected_final_wins": round(exp_final_w, 1),
            "expected_final_losses": round(exp_final_l, 1),
            "p10_final_wins": int(np.percentile(team_final_wins, 10)),
            "p90_final_wins": int(np.percentile(team_final_wins, 90)),
            "prob_make_top6": float((team_seeds <= 6).mean()),
            "prob_make_playin": float(((team_seeds >= 7) & (team_seeds <= 10)).mean()),
            "prob_miss_playoffs": float((team_seeds > 10).mean()),
            "projected_seed_median": int(np.median(team_seeds)),
        })

    team_projected_record = pd.DataFrame(rows)

    # --- Write to DuckDB ---
    con.execute("DROP TABLE IF EXISTS team_remaining_games")
    con.execute("CREATE TABLE team_remaining_games AS SELECT * FROM team_remaining_games")

    con.execute("DROP TABLE IF EXISTS team_projected_record")
    con.execute("CREATE TABLE team_projected_record AS SELECT * FROM team_projected_record")

    con.close()
    print(f"  Wrote team_remaining_games ({len(team_remaining_games)} rows)")
    print(f"  Wrote team_projected_record ({len(team_projected_record)} rows)")

    # Quick sanity check
    prob_sum = team_projected_record[["prob_make_top6", "prob_make_playin", "prob_miss_playoffs"]].sum(axis=1)
    if not (prob_sum.between(0.99, 1.01)).all():
        print(f"  WARNING: probability sums not ≈ 1.0: {prob_sum.describe()}")
    else:
        print("  Sanity check passed: prob_make_top6 + prob_make_playin + prob_miss_playoffs ≈ 1.0")

    # Print preview
    preview = team_projected_record.sort_values("expected_final_wins", ascending=False).head(10)
    print("\n  Top 10 by projected final wins:")
    print(
        preview[["TEAM_ABBR", "CONFERENCE", "current_wins", "games_remaining",
                  "expected_final_wins", "p10_final_wins", "p90_final_wins",
                  "prob_make_top6", "prob_make_playin"]].to_string(index=False)
    )


if __name__ == "__main__":
    build_remaining_schedule()
