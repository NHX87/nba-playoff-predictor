"""
fetch_players.py
----------------
Pulls individual player game logs for all seasons via nba_api.
Used to compute availability features and lineup quality scores.

Tables written to DuckDB:
  - raw_player_game_logs   : one row per player per game
  - rotation_players       : identified rotation players per team per season

Run after fetch_games.py and load_db.py.
"""

import time
import pandas as pd
import duckdb
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.library import http as _nba_http
from config.settings import DB_PATH, TRAIN_SEASONS, CURRENT_SEASON_STR

# stats.nba.com requires these headers; inject before any requests are made
_nba_http.STATS_HEADERS.update({
    "x-nba-stats-token": "true",
    "x-nba-stats-origin": "stats",
})

RAW_PATH = Path("data/raw")
RAW_PATH.mkdir(parents=True, exist_ok=True)

SLEEP_BETWEEN_CALLS = 1.5
MIN_ROTATION_MINUTES = 12.0   # avg minutes to qualify as rotation player
MIN_GAMES_PLAYED_PCT = 0.20   # must have played 20%+ of team games


def fetch_player_logs_by_season(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Pull all player game logs for a given season and season type.
    """
    cache_file = RAW_PATH / f"player_logs_{season}_{season_type.replace(' ', '_')}.parquet"

    if cache_file.exists():
        print(f"  Cache hit: {cache_file.name}")
        return pd.read_parquet(cache_file)

    print(f"  Fetching player logs {season_type} {season}...")

    try:
        logs = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
            timeout=60,
        )
        df = logs.get_data_frames()[0]

        if df.empty:
            print(f"  No data for {season} {season_type}")
            return pd.DataFrame()

        df["SEASON"] = season
        df["SEASON_TYPE"] = season_type
        df.to_parquet(cache_file, index=False)
        print(f"  Saved {len(df)} player-game rows → {cache_file.name}")
        time.sleep(SLEEP_BETWEEN_CALLS)
        return df

    except Exception as e:
        print(f"  Error fetching {season} {season_type}: {e}")
        time.sleep(2.0)
        return pd.DataFrame()


def identify_rotation_players(player_logs: pd.DataFrame) -> pd.DataFrame:
    """
    From player game logs, identify rotation players per team per season.
    
    Rotation player criteria:
    - Averaged MIN_ROTATION_MINUTES+ minutes per game
    - Played in MIN_GAMES_PLAYED_PCT+ of team games
    """
    if player_logs.empty:
        return pd.DataFrame()

    # Ensure numeric
    player_logs = player_logs.copy()
    player_logs['MIN'] = pd.to_numeric(player_logs['MIN'], errors='coerce').fillna(0)

    # Aggregate per player per team per season
    agg = player_logs.groupby(
        ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON', 'SEASON_TYPE']
    ).agg(
        games_played=('GAME_ID', 'count'),
        avg_minutes=('MIN', 'mean'),
        total_minutes=('MIN', 'sum'),
        avg_pts=('PTS', 'mean'),
        avg_reb=('REB', 'mean'),
        avg_ast=('AST', 'mean'),
    ).reset_index()

    # Get team game counts
    team_games = player_logs.groupby(
        ['TEAM_ID', 'SEASON', 'SEASON_TYPE']
    )['GAME_ID'].nunique().reset_index(name='team_games')

    agg = agg.merge(team_games, on=['TEAM_ID', 'SEASON', 'SEASON_TYPE'], how='left')
    agg['games_played_pct'] = agg['games_played'] / agg['team_games'].clip(lower=1)

    # Apply rotation player filter
    rotation = agg[
        (agg['avg_minutes'] >= MIN_ROTATION_MINUTES) &
        (agg['games_played_pct'] >= MIN_GAMES_PLAYED_PCT)
    ].copy()

    # Rank players by minutes (kept for reference) and by PPG (used for star flags)
    rotation['minutes_rank'] = rotation.groupby(
        ['TEAM_ID', 'SEASON', 'SEASON_TYPE']
    )['avg_minutes'].rank(ascending=False, method='dense')

    rotation['ppg_rank'] = rotation.groupby(
        ['TEAM_ID', 'SEASON', 'SEASON_TYPE']
    )['avg_pts'].rank(ascending=False, method='dense')

    # Flag top 2 players by PPG
    rotation['is_star'] = rotation['ppg_rank'] == 1
    rotation['is_second_star'] = rotation['ppg_rank'] == 2

    return rotation


def fetch_all_player_logs():
    """
    Pull player game logs for all seasons, load into DuckDB.
    """
    all_seasons = TRAIN_SEASONS + (
        [CURRENT_SEASON_STR] if CURRENT_SEASON_STR not in TRAIN_SEASONS else []
    )

    print(f"\nFetching player logs for {len(all_seasons)} seasons × 2 season types...")
    print("This will take a while on first run — all data is cached.\n")

    reg_dfs = []
    po_dfs = []

    for season in all_seasons:
        print(f"\n--- {season} ---")
        reg = fetch_player_logs_by_season(season, "Regular Season")
        po = fetch_player_logs_by_season(season, "Playoffs")

        if not reg.empty:
            reg_dfs.append(reg)
        if not po.empty:
            po_dfs.append(po)

    if not reg_dfs and not po_dfs:
        print("No player data fetched.")
        return

    con = duckdb.connect(DB_PATH)

    # Load regular season player logs
    if reg_dfs:
        reg_combined = pd.concat(reg_dfs, ignore_index=True)
        con.execute("DROP TABLE IF EXISTS raw_player_logs_rs")
        con.execute("CREATE TABLE raw_player_logs_rs AS SELECT * FROM reg_combined")
        print(f"\nLoaded {len(reg_combined):,} rows → raw_player_logs_rs")

        # Identify rotation players
        rotation_rs = identify_rotation_players(reg_combined)
        con.execute("DROP TABLE IF EXISTS rotation_players_rs")
        con.execute("CREATE TABLE rotation_players_rs AS SELECT * FROM rotation_rs")
        print(f"Loaded {len(rotation_rs):,} rows → rotation_players_rs")

    # Load playoff player logs
    if po_dfs:
        po_combined = pd.concat(po_dfs, ignore_index=True)
        con.execute("DROP TABLE IF EXISTS raw_player_logs_po")
        con.execute("CREATE TABLE raw_player_logs_po AS SELECT * FROM po_combined")
        print(f"Loaded {len(po_combined):,} rows → raw_player_logs_po")

        rotation_po = identify_rotation_players(po_combined)
        con.execute("DROP TABLE IF EXISTS rotation_players_po")
        con.execute("CREATE TABLE rotation_players_po AS SELECT * FROM rotation_po")
        print(f"Loaded {len(rotation_po):,} rows → rotation_players_po")

    # Preview rotation sizes
    print("\nAvg rotation size per team (regular season):")
    preview = con.execute("""
        SELECT SEASON, 
               AVG(rotation_size) AS avg_rotation_size,
               MIN(rotation_size) AS min_rotation,
               MAX(rotation_size) AS max_rotation
        FROM (
            SELECT SEASON, TEAM_ID, COUNT(*) AS rotation_size
            FROM rotation_players_rs
            GROUP BY SEASON, TEAM_ID
        )
        GROUP BY SEASON
        ORDER BY SEASON DESC
        LIMIT 5
    """).df()
    print(preview.to_string())

    con.close()
    print("\nPlayer logs ready.")


if __name__ == "__main__":
    fetch_all_player_logs()