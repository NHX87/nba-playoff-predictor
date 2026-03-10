"""
fetch_playoff_status.py
-----------------------
Fetch the current state of all playoff series for a given season.

Combines:
  1. CommonPlayoffSeries endpoint — actual matchups + game schedule
  2. playoffs DuckDB table — W/L outcomes per game (from nightly fetch)

Output DataFrame (one row per series):
  series_id, conference, round, round_num,
  high_team, low_team, high_seed, low_seed,
  high_team_wins, low_team_wins, total_games,
  series_status (completed / in_progress / not_started),
  actual_winner

Run:
  python -m pipeline.ingestion.fetch_playoff_status
"""

from __future__ import annotations

import duckdb
import pandas as pd
from nba_api.stats.static import teams as nba_teams

from config.settings import CURRENT_SEASON_STR, DB_PATH

# Team ID → abbreviation mapping
_TEAM_ID_TO_ABBR: dict[int, str] = {
    t["id"]: t["abbreviation"] for t in nba_teams.get_teams()
}

# Round number from SERIES_ID character at position 7 (0-indexed)
# SERIES_ID format: 004YY00RS where R=round(1-4), S=series_index
_ROUND_NAMES = {
    1: "First Round",
    2: "Conference Semifinals",
    3: "Conference Finals",
    4: "NBA Finals",
}


def _parse_series_id(series_id: str) -> tuple[int, int]:
    """Extract (round_num, series_index) from SERIES_ID string."""
    # e.g. "004240010" -> round=1, series_idx=0
    round_num = int(series_id[7])
    series_idx = int(series_id[8])
    return round_num, series_idx


def _load_seed_conf_map(con: duckdb.DuckDBPyConnection) -> dict[str, dict]:
    """Load team → {conference, seed} from app_playoff_field_current."""
    try:
        df = con.execute("""
            SELECT TEAM_ABBR, conference, playoff_seed
            FROM app_playoff_field_current
        """).fetchdf()
        return {
            row["TEAM_ABBR"]: {"conference": row["conference"], "seed": int(row["playoff_seed"])}
            for _, row in df.iterrows()
        }
    except Exception:
        return {}


def get_playoff_series_status(season: str = CURRENT_SEASON_STR) -> pd.DataFrame:
    """
    Fetch current playoff bracket state.

    Returns DataFrame with one row per series, or empty DataFrame if no
    playoff data exists for the given season.
    """
    # 1. Fetch series schedule from NBA API
    try:
        from nba_api.stats.endpoints import commonplayoffseries
        cps = commonplayoffseries.CommonPlayoffSeries(season=season, timeout=30)
        schedule = cps.get_data_frames()[0]
    except Exception as exc:
        print(f"[fetch_playoff_status] CommonPlayoffSeries failed: {exc}")
        return pd.DataFrame()

    if schedule.empty:
        return pd.DataFrame()

    # 2. Load playoff game outcomes from DuckDB
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        seed_conf = _load_seed_conf_map(con)

        try:
            game_results = con.execute("""
                SELECT TEAM_ABBREVIATION AS TEAM_ABBR, GAME_ID, WL
                FROM playoffs
                WHERE SEASON = ?
            """, [season]).fetchdf()
        except Exception:
            game_results = pd.DataFrame()
    finally:
        con.close()

    # Build W/L lookup: game_id → {team_abbr: 'W'/'L'}
    wl_lookup: dict[str, dict[str, str]] = {}
    if not game_results.empty:
        for _, row in game_results.iterrows():
            gid = str(row["GAME_ID"])
            if gid not in wl_lookup:
                wl_lookup[gid] = {}
            wl_lookup[gid][str(row["TEAM_ABBR"])] = str(row["WL"])

    # 3. Aggregate per series
    rows = []
    for series_id in sorted(schedule["SERIES_ID"].unique()):
        s_games = schedule[schedule["SERIES_ID"] == series_id].sort_values("GAME_NUM")
        round_num, series_idx = _parse_series_id(str(series_id))
        round_name = _ROUND_NAMES.get(round_num, f"Round {round_num}")

        # Identify the two teams
        all_team_ids = set(s_games["HOME_TEAM_ID"]) | set(s_games["VISITOR_TEAM_ID"])
        team_abbrs = [_TEAM_ID_TO_ABBR.get(tid, str(tid)) for tid in sorted(all_team_ids)]

        if len(team_abbrs) != 2:
            continue

        team_a, team_b = team_abbrs[0], team_abbrs[1]

        # Determine high/low seed from playoff field
        info_a = seed_conf.get(team_a, {})
        info_b = seed_conf.get(team_b, {})
        seed_a = info_a.get("seed", 99)
        seed_b = info_b.get("seed", 99)

        if seed_a <= seed_b:
            high_team, low_team = team_a, team_b
            high_seed, low_seed = seed_a, seed_b
        else:
            high_team, low_team = team_b, team_a
            high_seed, low_seed = seed_b, seed_a

        # Determine conference
        if round_num == 4:
            conference = "NBA"
        else:
            conf_a = info_a.get("conference", "")
            conf_b = info_b.get("conference", "")
            conference = conf_a or conf_b or "Unknown"

        # Count wins per team from DuckDB game results
        high_wins = 0
        low_wins = 0
        games_played = 0

        for _, g in s_games.iterrows():
            gid = str(g["GAME_ID"])
            result = wl_lookup.get(gid, {})
            if high_team in result:
                games_played += 1
                if result[high_team] == "W":
                    high_wins += 1
                else:
                    low_wins += 1

        # Determine status
        total_scheduled = len(s_games)
        if high_wins == 4 or low_wins == 4:
            status = "completed"
            actual_winner = high_team if high_wins == 4 else low_team
        elif games_played > 0:
            status = "in_progress"
            actual_winner = None
        else:
            status = "not_started"
            actual_winner = None

        rows.append({
            "series_id": str(series_id),
            "conference": conference,
            "round": round_name,
            "round_num": round_num,
            "high_team": high_team,
            "low_team": low_team,
            "high_seed": high_seed,
            "low_seed": low_seed,
            "high_team_wins": high_wins,
            "low_team_wins": low_wins,
            "total_games": games_played,
            "series_status": status,
            "actual_winner": actual_winner,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["round_num", "conference", "high_seed"]).reset_index(drop=True)


if __name__ == "__main__":
    df = get_playoff_series_status("2024-25")
    if df.empty:
        print("No playoff data found.")
    else:
        print(df.to_string())
        print(f"\n{len(df)} series total")
        for status in ["completed", "in_progress", "not_started"]:
            n = (df["series_status"] == status).sum()
            if n:
                print(f"  {status}: {n}")
