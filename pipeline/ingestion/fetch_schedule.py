"""
fetch_schedule.py
-----------------
Fetch the current season's remaining regular-season schedule from the NBA CDN.

Uses https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json
  - No IP blocking (CDN, not stats.nba.com)
  - Single HTTP call, fast
  - Cached daily in data/raw/schedule_{season}.parquet

Output schema:
  GAME_ID         str     NBA game ID
  GAME_DATE       date    Scheduled date (EST)
  HOME_TEAM_ABBR  str     Home team tricode
  AWAY_TEAM_ABBR  str     Away team tricode
  IS_PLAYED       bool    True = already completed (gameStatus == 3)
  SEASON          str     e.g. "2025-26"
"""

import json
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from config.settings import CURRENT_SEASON_STR

RAW_PATH = Path("data/raw")
RAW_PATH.mkdir(parents=True, exist_ok=True)

CDN_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"

# Regular-season game IDs start with "0022" (002 = NBA, 2 = Regular Season)
RS_GAME_ID_PREFIX = "0022"


def fetch_remaining_schedule(season: str = CURRENT_SEASON_STR) -> pd.DataFrame:
    """
    Fetch full regular-season schedule for *season* and return as a DataFrame.

    Caches to data/raw/schedule_{season}.parquet, refreshed once per calendar day.
    Returns both played and unplayed games (use IS_PLAYED to filter).
    """
    cache_file = RAW_PATH / f"schedule_{season}.parquet"

    # Return today's cache if it exists
    if cache_file.exists():
        mtime_date = datetime.fromtimestamp(cache_file.stat().st_mtime).date()
        if mtime_date == date.today():
            print(f"  Cache hit: {cache_file.name}")
            return pd.read_parquet(cache_file)

    print(f"  Fetching schedule from NBA CDN for season {season}…")
    for attempt in range(3):
        try:
            req = urllib.request.Request(CDN_URL, headers={"User-Agent": "python/nba-playoff-predictor"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            break
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"Failed to fetch schedule after 3 attempts: {exc}") from exc
            print(f"  Retry {attempt + 1}/3 after error: {exc}")
            time.sleep(3)

    league = data.get("leagueSchedule", {})
    cdn_season = league.get("seasonYear", "")
    if cdn_season != season:
        print(f"  Warning: CDN season {cdn_season!r} != requested {season!r} — using anyway")

    rows = []
    for game_date_entry in league.get("gameDates", []):
        for game in game_date_entry.get("games", []):
            game_id = game.get("gameId", "")
            # Regular season only (gameId starts with 0022)
            if not game_id.startswith(RS_GAME_ID_PREFIX):
                continue

            game_date_raw = game.get("gameDateEst", "")[:10]  # "2025-10-21"
            home = game.get("homeTeam", {}).get("teamTricode", "")
            away = game.get("awayTeam", {}).get("teamTricode", "")
            status = game.get("gameStatus", 1)  # 1=scheduled, 2=in_progress, 3=final
            if not home or not away:
                continue

            rows.append({
                "GAME_ID": game_id,
                "GAME_DATE": game_date_raw,
                "HOME_TEAM_ABBR": home,
                "AWAY_TEAM_ABBR": away,
                "IS_PLAYED": status == 3,
                "SEASON": season,
            })

    if not rows:
        print("  Warning: no regular-season rows found in CDN schedule.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    df.to_parquet(cache_file, index=False)

    played = df["IS_PLAYED"].sum()
    remaining = (~df["IS_PLAYED"]).sum()
    print(f"  Schedule cached: {len(df)} regular-season games ({played} played, {remaining} remaining)")
    return df


if __name__ == "__main__":
    df = fetch_remaining_schedule()
    remaining = df[~df["IS_PLAYED"]]
    print(f"\nNext 5 remaining games:\n{remaining.head()}")
