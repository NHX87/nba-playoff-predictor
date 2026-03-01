"""
fetch_series.py
---------------
Pulls actual playoff series results from nba_api.
Uses PlayoffSeries endpoint to get exact series outcomes.

This fixes the rounds_reached calculation in team_stats.py
by giving us actual series wins per team per season rather
than estimating from games played.

Tables written to DuckDB:
  - raw_series_results   : one row per series per season
  - team_series_summary  : one row per team per season with exact rounds reached

Run after fetch_games.py and load_db.py.
"""

import pandas as pd
import duckdb
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import DB_PATH

RAW_PATH = Path("data/raw")
RAW_PATH.mkdir(parents=True, exist_ok=True)

def build_series_from_game_logs() -> pd.DataFrame:
    """
    Derive playoff series results directly from game logs in DuckDB.
    Much more reliable than PlayoffPicture endpoint for historical data.
    """
    con = duckdb.connect(DB_PATH)
    
    print("Building series results from game logs...")
    
    # Each playoff game has a GAME_ID that encodes the series
    # Game ID format: 004YYRRSSS where YY=year, RR=round, SSS=series
    series_data = con.execute("""
        SELECT 
            SEASON,
            GAME_ID,
            TEAM_ID,
            TEAM_ABBR,
            WL,
            GAME_DATE,
            -- Extract round from game ID (positions 4-5)
            CAST(SUBSTR(CAST(GAME_ID AS VARCHAR), 4, 2) AS INT) AS round,
            -- Extract series number
            SUBSTR(CAST(GAME_ID AS VARCHAR), 6, 3) AS series_num
        FROM playoffs
        WHERE GAME_ID IS NOT NULL
        ORDER BY SEASON, GAME_ID, GAME_DATE
    """).df()
    
    con.close()
    
    print(f"Found {len(series_data)} playoff game records")
    print(f"Sample game IDs: {series_data['GAME_ID'].head(5).tolist()}")
    print(f"Rounds found: {sorted(series_data['round'].unique())}")
    
    return series_data


def build_team_series_summary(all_series: pd.DataFrame) -> pd.DataFrame:
    if all_series.empty:
        print("No series data to summarize.")
        return pd.DataFrame()

    print(f"Building summary from {len(all_series)} series records...")

    # Derive winners from win counts since cached data has None for winner_team_id
    def get_winner_loser(row):
        try:
            h = int(row['high_seed_wins'] or 0)
            l = int(row['low_seed_wins'] or 0)
            if h == 4:
                return row['high_seed_team_id'], row['low_seed_team_id']
            elif l == 4:
                return row['low_seed_team_id'], row['high_seed_team_id']
        except:
            pass
        return None, None

    all_series = all_series.copy()
    all_series[['winner_team_id', 'loser_team_id']] = all_series.apply(
        lambda row: pd.Series(get_winner_loser(row)), axis=1
    )

    print(f"Winners found: {all_series['winner_team_id'].notna().sum()}")
    print(f"Sample:\n{all_series[['season','high_seed_team','high_seed_wins','low_seed_team','low_seed_wins','winner_team_id']].head(5)}")

    records = []
    winners = all_series[all_series['winner_team_id'].notna()].copy()
    
    for season in winners['season'].unique():
        season_data = winners[winners['season'] == season]
        win_counts = season_data.groupby('winner_team_id').size().reset_index()
        win_counts.columns = ['team_id', 'series_wins']
        for _, row in win_counts.iterrows():
            records.append({
                'team_id': row['team_id'],
                'season': season,
                'series_wins': int(row['series_wins']),
                'rounds_reached': int(row['series_wins']) + 1
            })

    winner_keys = {(r['team_id'], r['season']) for r in records}
    losers = all_series[all_series['loser_team_id'].notna()].copy()
    for _, row in losers.iterrows():
        key = (row['loser_team_id'], row['season'])
        if key not in winner_keys:
            records.append({
                'team_id': row['loser_team_id'],
                'season': row['season'],
                'series_wins': 0,
                'rounds_reached': 1
            })
            winner_keys.add(key)

    if not records:
        print("Still no records — printing full sample:")
        print(all_series.head(10).to_string())
        return pd.DataFrame()

    summary = pd.DataFrame(records)
    round_labels = {1: 'First Round', 2: 'Conf Semis',
                    3: 'Conf Finals', 4: 'Finals', 5: 'Champion'}
    summary['round_label'] = summary['rounds_reached'].map(round_labels)
    return summary.sort_values(['season', 'rounds_reached'], ascending=[True, False])


def fetch_all_series():
    """
    Build series results from game logs already in DuckDB.
    """
    series_data = build_series_from_game_logs()
    
    if series_data.empty:
        print("No series data found. Make sure game logs are loaded.")
        return

    # Count wins per team per series
    series_wins = series_data[series_data['WL'] == 'W'].groupby(
        ['SEASON', 'TEAM_ID', 'TEAM_ABBR', 'round', 'series_num']
    ).size().reset_index(name='wins')

    # A team won a series if they got 4 wins
    series_winners = series_wins[series_wins['wins'] == 4].copy()
    series_winners = series_winners.rename(columns={'round': 'round_won'})

    # Count total series wins per team per season = rounds reached
    team_summary = series_winners.groupby(
        ['SEASON', 'TEAM_ID', 'TEAM_ABBR']
    )['round_won'].count().reset_index(name='series_wins')

    team_summary['rounds_reached'] = team_summary['series_wins'] + 1

    # Add first round exits (teams with 0 series wins)
    all_teams = series_data[['SEASON', 'TEAM_ID', 'TEAM_ABBR']].drop_duplicates()
    team_summary_keys = set(zip(team_summary['SEASON'], team_summary['TEAM_ID']))

    first_round_exits = []
    for _, row in all_teams.iterrows():
        if (row['SEASON'], row['TEAM_ID']) not in team_summary_keys:
            first_round_exits.append({
                'SEASON': row['SEASON'],
                'TEAM_ID': row['TEAM_ID'],
                'TEAM_ABBR': row['TEAM_ABBR'],
                'series_wins': 0,
                'rounds_reached': 1
            })

    if first_round_exits:
        team_summary = pd.concat(
            [team_summary, pd.DataFrame(first_round_exits)],
            ignore_index=True
        )

    round_labels = {1: 'First Round', 2: 'Conf Semis',
                    3: 'Conf Finals', 4: 'Finals', 5: 'Champion'}
    team_summary['round_label'] = team_summary['rounds_reached'].map(round_labels)
    team_summary = team_summary.rename(columns={
        'SEASON': 'season', 'TEAM_ID': 'team_id', 'TEAM_ABBR': 'team_abbr'
    })

    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS team_series_summary")
    con.execute("CREATE TABLE team_series_summary AS SELECT * FROM team_summary")
    print(f"\nLoaded {len(team_summary)} team-season records → team_series_summary")

    print("\nSample — recent champions:")
    champs = con.execute("""
        SELECT season, team_abbr, series_wins, round_label
        FROM team_series_summary
        WHERE rounds_reached = 5
        ORDER BY season DESC
        LIMIT 10
    """).df()
    print(champs.to_string())
    con.close()


if __name__ == "__main__":
    fetch_all_series()
