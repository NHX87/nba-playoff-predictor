"""
fetch_games.py
--------------
Pull team game logs for regular season and playoffs from nba_api
and cache each season/type to parquet under data/raw/.

Design goals:
- Transparent progress (per season + summary)
- Retry transient API failures per team
- Explicitly report incomplete pulls (missing teams)
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import teamgamelogs
from nba_api.stats.library import http as _nba_http
from nba_api.stats.static import teams
from tqdm import tqdm

from config.settings import CURRENT_SEASON_STR, TRAIN_SEASONS

# stats.nba.com requires these headers; inject before any requests are made
_nba_http.STATS_HEADERS.update({
    "x-nba-stats-token": "true",
    "x-nba-stats-origin": "stats",
})

RAW_PATH = Path("data/raw")
RAW_PATH.mkdir(parents=True, exist_ok=True)

SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES_PER_TEAM = 3
RETRY_BASE_SLEEP = 5.0
REQUEST_TIMEOUT = 45


def _fetch_single_team_logs(team: dict, season: str, season_type: str) -> pd.DataFrame:
    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team["id"],
        season_nullable=season,
        season_type_nullable=season_type,
        timeout=REQUEST_TIMEOUT,
    )
    df = logs.get_data_frames()[0]
    df["TEAM_ID"] = team["id"]
    df["TEAM_NAME"] = team["full_name"]
    df["TEAM_ABBR"] = team["abbreviation"]
    return df


def fetch_team_game_logs(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Pull game logs for all teams for a given season and season type.
    season_type: "Regular Season" or "Playoffs"
    """
    cache_file = RAW_PATH / f"games_{season}_{season_type.replace(' ', '_')}.parquet"

    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        print(
            f"  Cache hit: {cache_file.name} "
            f"({len(cached):,} rows, {cached['TEAM_ID'].nunique()} teams)"
        )
        return cached

    print(f"  Fetching {season_type} {season}...")

    all_teams = teams.get_teams()
    dfs = []
    failed_teams = []

    for team in tqdm(all_teams, desc=f"{season} {season_type}", leave=False):
        team_df = None
        last_error = None

        for attempt in range(1, MAX_RETRIES_PER_TEAM + 1):
            try:
                team_df = _fetch_single_team_logs(team, season, season_type)
                break
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES_PER_TEAM:
                    sleep_s = RETRY_BASE_SLEEP * attempt
                    time.sleep(sleep_s)

        if team_df is None:
            failed_teams.append(team["abbreviation"])
            print(
                f"  Warning: Failed {team['abbreviation']} {season} {season_type} "
                f"after {MAX_RETRIES_PER_TEAM} attempts — {last_error}"
            )
            continue

        if not team_df.empty:
            dfs.append(team_df)

        time.sleep(SLEEP_BETWEEN_CALLS)

    if not dfs:
        print(f"  No data returned for {season} {season_type}")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["SEASON"] = season
    combined["SEASON_TYPE"] = season_type

    combined.to_parquet(cache_file, index=False)

    teams_fetched = combined["TEAM_ID"].nunique()
    missing = sorted(set(t["abbreviation"] for t in all_teams) - set(combined["TEAM_ABBR"].unique()))

    print(
        f"  Saved {len(combined):,} rows -> {cache_file.name} "
        f"({teams_fetched}/30 teams)"
    )
    if missing:
        print(f"  Missing teams in cache: {', '.join(missing)}")
    if failed_teams:
        print(f"  Teams that failed all retries: {', '.join(sorted(set(failed_teams)))}")

    return combined


def fetch_all_seasons() -> None:
    """Pull regular season and playoff logs for training seasons plus current season."""
    all_seasons = TRAIN_SEASONS + (
        [CURRENT_SEASON_STR] if CURRENT_SEASON_STR not in TRAIN_SEASONS else []
    )

    print(f"\nFetching {len(all_seasons)} seasons x 2 season types...")
    print("This can take a while on first run. Cached files are reused.\n")

    for season in all_seasons:
        print(f"\n--- {season} ---")
        fetch_team_game_logs(season, "Regular Season")
        fetch_team_game_logs(season, "Playoffs")

    print("\nAll requested seasons fetched/cached.")


if __name__ == "__main__":
    fetch_all_seasons()
