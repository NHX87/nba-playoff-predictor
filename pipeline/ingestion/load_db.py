"""
load_db.py
----------
Load cached game parquet files from data/raw into DuckDB.

Tables created:
  - raw_game_logs
  - teams
Views created:
  - regular_season
  - playoffs
"""

from pathlib import Path

import duckdb
import pandas as pd

from config.settings import DB_PATH

RAW_PATH = Path("data/raw")


def load_to_duckdb() -> None:
    print(f"\nConnecting to DuckDB at {DB_PATH}...")
    con = duckdb.connect(DB_PATH)

    parquet_files = sorted(RAW_PATH.glob("games_*.parquet"))
    if not parquet_files:
        print("No game parquet files found. Run fetch_games.py first.")
        con.close()
        return

    print(f"Loading {len(parquet_files)} parquet files into DuckDB...")

    dfs = []
    skipped_empty = 0
    for fpath in parquet_files:
        df = pd.read_parquet(fpath)
        df = df.dropna(axis=1, how="all")
        if df.empty:
            skipped_empty += 1
            continue
        dfs.append(df)

    if not dfs:
        print("All game parquet files were empty; nothing to load.")
        con.close()
        return

    combined = pd.concat(dfs, ignore_index=True)

    con.execute("DROP TABLE IF EXISTS raw_game_logs")
    con.execute("CREATE TABLE raw_game_logs AS SELECT * FROM combined")

    row_count = con.execute("SELECT COUNT(*) FROM raw_game_logs").fetchone()[0]
    season_summary = con.execute(
        """
        SELECT SEASON, SEASON_TYPE, COUNT(*) AS rows, COUNT(DISTINCT TEAM_ID) AS teams
        FROM raw_game_logs
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()

    print(f"Loaded {row_count:,} rows into raw_game_logs")
    if skipped_empty:
        print(f"Skipped {skipped_empty} empty parquet file(s)")
    print("Raw game coverage:")
    print(season_summary.to_string(index=False))

    from nba_api.stats.static import teams

    teams_df = pd.DataFrame(teams.get_teams())
    con.execute("DROP TABLE IF EXISTS teams")
    con.execute("CREATE TABLE teams AS SELECT * FROM teams_df")
    print(f"Loaded {len(teams_df)} teams into teams")

    for name, condition in [
        ("regular_season", "SEASON_TYPE = 'Regular Season'"),
        ("playoffs",       "SEASON_TYPE = 'Playoffs'"),
    ]:
        row = con.execute(
            "SELECT table_type FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()
        if row:
            obj_type = "VIEW" if row[0] == "VIEW" else "TABLE"
            con.execute(f"DROP {obj_type} IF EXISTS {name}")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM raw_game_logs WHERE {condition}")

    print("Created/updated views: regular_season, playoffs")
    con.close()
    print("DuckDB load complete.")


if __name__ == "__main__":
    load_to_duckdb()
