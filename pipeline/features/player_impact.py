"""
player_impact.py
----------------
Compute player impact metrics using with/without game splits.

For each rotation player on each team (current season), compares team
performance (win%, net rating) in games the player appeared vs. missed.

Output table: `player_impact` with columns:
  SEASON, TEAM_ABBR, PLAYER_NAME, games_played, games_missed,
  team_win_pct_with, team_win_pct_without, team_net_with, team_net_without,
  win_pct_delta, net_rating_delta, mpg, ppg, rpg, apg
"""

from __future__ import annotations

import duckdb
import pandas as pd

from config.settings import CURRENT_SEASON, CURRENT_SEASON_STR, season_str as make_season_str


def compute_player_impact(con: duckdb.DuckDBPyConnection, season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Compute with/without splits for every rotation player this season."""
    season_str = make_season_str(season)

    # All team games this season with W/L and plus_minus (proxy for net rating)
    team_games = con.execute(
        """
        SELECT TEAM_ABBREVIATION AS TEAM_ABBR, GAME_ID, GAME_DATE, WL, PLUS_MINUS
        FROM regular_season
        WHERE SEASON = ?
        """,
        [season_str],
    ).fetchdf()

    if team_games.empty:
        return pd.DataFrame()

    # All player game logs: who played in which game
    player_games = con.execute(
        """
        SELECT PLAYER_NAME, TEAM_ABBREVIATION AS TEAM_ABBR, GAME_ID,
               MIN AS MINUTES, PTS, REB, AST
        FROM raw_player_logs_rs
        WHERE SEASON = ?
          AND MIN > 0
        """,
        [season_str],
    ).fetchdf()

    if player_games.empty:
        return pd.DataFrame()

    # Get set of all GAME_IDs per team
    team_game_ids = team_games.groupby("TEAM_ABBR")["GAME_ID"].apply(set).to_dict()

    # Team-level stats per game for quick lookup
    team_games["WIN"] = (team_games["WL"] == "W").astype(int)
    game_stats = team_games.set_index(["TEAM_ABBR", "GAME_ID"])[["WIN", "PLUS_MINUS"]].to_dict("index")

    # Player season averages
    player_avgs = (
        player_games.groupby(["TEAM_ABBR", "PLAYER_NAME"])
        .agg(
            games_played=("GAME_ID", "nunique"),
            mpg=("MINUTES", "mean"),
            ppg=("PTS", "mean"),
            rpg=("REB", "mean"),
            apg=("AST", "mean"),
        )
        .reset_index()
    )

    # Filter to rotation players: 15+ games and 12+ mpg
    rotation = player_avgs[(player_avgs["games_played"] >= 15) & (player_avgs["mpg"] >= 12)].copy()

    if rotation.empty:
        return pd.DataFrame()

    # Games each player appeared in
    player_game_sets = (
        player_games.groupby(["TEAM_ABBR", "PLAYER_NAME"])["GAME_ID"]
        .apply(set)
        .to_dict()
    )

    rows = []
    for _, p in rotation.iterrows():
        team = p["TEAM_ABBR"]
        name = p["PLAYER_NAME"]
        all_games = team_game_ids.get(team, set())
        played_games = player_game_sets.get((team, name), set())
        missed_games = all_games - played_games

        n_played = len(played_games)
        n_missed = len(missed_games)

        # Need at least 3 missed games to compute meaningful split
        if n_missed < 3:
            # Still include but mark as insufficient data
            rows.append({
                "SEASON": season_str,
                "TEAM_ABBR": team,
                "PLAYER_NAME": name,
                "games_played": n_played,
                "games_missed": n_missed,
                "team_win_pct_with": None,
                "team_win_pct_without": None,
                "team_net_with": None,
                "team_net_without": None,
                "win_pct_delta": None,
                "net_rating_delta": None,
                "mpg": round(p["mpg"], 1),
                "ppg": round(p["ppg"], 1),
                "rpg": round(p["rpg"], 1),
                "apg": round(p["apg"], 1),
            })
            continue

        # Compute splits
        with_wins = sum(game_stats.get((team, g), {}).get("WIN", 0) for g in played_games)
        with_pm = [game_stats.get((team, g), {}).get("PLUS_MINUS", 0) for g in played_games]
        without_wins = sum(game_stats.get((team, g), {}).get("WIN", 0) for g in missed_games)
        without_pm = [game_stats.get((team, g), {}).get("PLUS_MINUS", 0) for g in missed_games]

        wpct_with = with_wins / n_played if n_played else 0
        wpct_without = without_wins / n_missed if n_missed else 0
        net_with = sum(with_pm) / n_played if n_played else 0
        net_without = sum(without_pm) / n_missed if n_missed else 0

        rows.append({
            "SEASON": season_str,
            "TEAM_ABBR": team,
            "PLAYER_NAME": name,
            "games_played": n_played,
            "games_missed": n_missed,
            "team_win_pct_with": round(wpct_with, 3),
            "team_win_pct_without": round(wpct_without, 3),
            "team_net_with": round(net_with, 1),
            "team_net_without": round(net_without, 1),
            "win_pct_delta": round(wpct_with - wpct_without, 3),
            "net_rating_delta": round(net_with - net_without, 1),
            "mpg": round(p["mpg"], 1),
            "ppg": round(p["ppg"], 1),
            "rpg": round(p["rpg"], 1),
            "apg": round(p["apg"], 1),
        })

    df = pd.DataFrame(rows)

    # Write to DuckDB
    con.execute("DROP TABLE IF EXISTS player_impact")
    con.execute("CREATE TABLE player_impact AS SELECT * FROM df")

    n = len(df)
    n_with_splits = df["win_pct_delta"].notna().sum()
    print(f"[player_impact] {n} rotation players, {n_with_splits} with meaningful with/without splits")

    return df


if __name__ == "__main__":
    con = duckdb.connect("data/processed/nba.duckdb")
    df = compute_player_impact(con)
    if not df.empty:
        # Show top impact players
        splits = df.dropna(subset=["net_rating_delta"]).sort_values("net_rating_delta", ascending=False)
        print("\nTop 10 players by net rating impact (team WITH minus WITHOUT):")
        print(splits.head(10)[["TEAM_ABBR", "PLAYER_NAME", "games_played", "games_missed",
                                "team_net_with", "team_net_without", "net_rating_delta",
                                "ppg"]].to_string(index=False))
        print("\nBottom 10:")
        print(splits.tail(10)[["TEAM_ABBR", "PLAYER_NAME", "games_played", "games_missed",
                                "team_net_with", "team_net_without", "net_rating_delta",
                                "ppg"]].to_string(index=False))
    con.close()
