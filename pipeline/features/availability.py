"""
availability.py
---------------
Computes availability features for all rotation players (8+ minutes).

For each team per playoff season:
  - rotation_size              : number of rotation players
  - avg_availability_rate      : % of games rotation players were available
  - star_availability_rate     : availability rate of top player by minutes
  - second_star_availability   : availability rate of second player by minutes
  - lineup_quality_score       : weighted sum of available players' minute shares
  - depth_score                : how evenly distributed minutes are (bench depth)
  - injury_games               : total player-games lost to injury/DNP

Binary available/unavailable threshold:
  A player is "available" in a game if MIN >= MIN_AVAILABLE_THRESHOLD (5 minutes)
  Below 5 minutes = injury, DNP, or garbage time appearance — not a real contributor

Tables written to DuckDB:
  - availability_features     : one row per team per season (for model_features join)
  - game_availability         : one row per team per game (for live tracking)
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import DB_PATH

MIN_AVAILABLE_THRESHOLD = 5.0    # minutes to count as available
MIN_ROTATION_MINUTES = 15.0      # minutes to qualify as rotation player
MIN_GAMES_PLAYED_PCT = 0.20      # must have played 20%+ of team games


def compute_availability_features() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    print("Computing availability features...")

    # Check required tables exist
    tables = con.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'main'
    """).df()["table_name"].tolist()

    required = ['raw_player_logs_rs', 'raw_player_logs_po']
    missing = [t for t in required if t not in tables]
    if missing:
        raise ValueError(f"Missing tables: {missing}. Run fetch_players.py first.")

    # --- STEP 1: Identify rotation players from regular season ---
    # Trade handling: use playoff team as source of truth for team assignment.
    # For traded players, use FULL season stats across all teams for quality measure
    # so a player traded mid-season isn't penalized for split sample sizes.
    print("  Identifying rotation players from regular season...")

    rs_logs = con.execute("""
        SELECT
            PLAYER_ID,
            PLAYER_NAME,
            TEAM_ID,
            TEAM_ABBREVIATION,
            SEASON,
            GAME_ID,
            CAST(MIN AS FLOAT) AS minutes,
            CAST(PTS AS FLOAT) AS pts,
            CAST(REB AS FLOAT) AS reb,
            CAST(AST AS FLOAT) AS ast
        FROM raw_player_logs_rs
        WHERE MIN IS NOT NULL
    """).df()

    po_logs_raw = con.execute("""
        SELECT
            PLAYER_ID,
            TEAM_ID,
            TEAM_ABBREVIATION,
            SEASON,
            GAME_ID,
            CAST(MIN AS FLOAT) AS minutes
        FROM raw_player_logs_po
        WHERE MIN IS NOT NULL
    """).df()

    # --- TRADE HANDLING ---
    # Step A: Identify each player's playoff team
    # Use the team they played the most playoff games for
    playoff_team_map = (
        po_logs_raw[po_logs_raw['minutes'] >= MIN_AVAILABLE_THRESHOLD]
        .groupby(['PLAYER_ID', 'SEASON', 'TEAM_ID'])
        .size()
        .reset_index(name='po_games')
        .sort_values('po_games', ascending=False)
        .groupby(['PLAYER_ID', 'SEASON'])
        .first()
        .reset_index()[['PLAYER_ID', 'SEASON', 'TEAM_ID', 'po_games']]
        .rename(columns={'TEAM_ID': 'playoff_team_id'})
    )

    # Step B: Compute FULL season averages per player regardless of team
    # This prevents split-season sample size issues for traded players
    player_full_season = rs_logs.groupby(
        ['PLAYER_ID', 'PLAYER_NAME', 'SEASON']
    ).agg(
        games_played=('GAME_ID', 'count'),
        avg_minutes=('minutes', 'mean'),
        avg_pts=('pts', 'mean'),
        avg_reb=('reb', 'mean'),
        avg_ast=('ast', 'mean'),
    ).reset_index()

    # Step C: Attach playoff team to full-season stats
    player_full_season = player_full_season.merge(
        playoff_team_map, on=['PLAYER_ID', 'SEASON'], how='inner'
    )
    # Only keep players who actually played meaningful playoff minutes
    player_full_season = player_full_season[player_full_season['po_games'] >= 2]

    # Step D: Total RS games in season (use 82 as denominator, not team-specific)
    total_rs_games = rs_logs.groupby('SEASON')['GAME_ID'].nunique().reset_index(name='season_games')
    # Approx: each team plays ~82, use player's actual games vs 82
    player_full_season['games_played_pct'] = (
        player_full_season['games_played'] / 82.0
    ).clip(upper=1.0)

    # Apply rotation player filter using full-season stats
    rotation = player_full_season[
        (player_full_season['avg_minutes'] >= MIN_ROTATION_MINUTES) &
        (player_full_season['games_played_pct'] >= MIN_GAMES_PLAYED_PCT)
    ].copy()

    # Rename for consistency downstream
    rotation = rotation.rename(columns={'playoff_team_id': 'TEAM_ID'})

    # Rank by minutes within playoff team — identifies star players
    rotation['minutes_rank'] = rotation.groupby(
        ['TEAM_ID', 'SEASON']
    )['avg_minutes'].rank(ascending=False, method='dense')

    # Minutes share — weight for lineup quality score
    team_total_minutes = rotation.groupby(
        ['TEAM_ID', 'SEASON']
    )['avg_minutes'].sum().reset_index(name='team_total_minutes')

    rotation = rotation.merge(team_total_minutes, on=['TEAM_ID', 'SEASON'], how='left')
    rotation['minutes_share'] = (
        rotation['avg_minutes'] / rotation['team_total_minutes'].clip(lower=1)
    )

    # Flag traded players for transparency
    # A traded player appears under multiple TEAM_IDs in RS logs
    traded_players = (
        rs_logs.groupby(['PLAYER_ID', 'SEASON'])['TEAM_ID']
        .nunique()
        .reset_index(name='teams_count')
    )
    rotation = rotation.merge(
        traded_players, on=['PLAYER_ID', 'SEASON'], how='left'
    )
    rotation['was_traded'] = rotation['teams_count'] > 1

    traded_count = rotation['was_traded'].sum()
    print(f"  Identified {len(rotation)} rotation player-seasons")
    print(f"  Traded players handled: {traded_count} ({traded_count/len(rotation)*100:.1f}%)")
    print(f"  Avg rotation size: {rotation.groupby(['TEAM_ID','SEASON']).size().mean():.1f} players")

    # --- STEP 2: Measure playoff availability ---
    print("  Computing playoff availability...")

    # Reuse po_logs_raw already loaded in step 1
    po_logs = po_logs_raw.copy()

    # Binary available/unavailable per game
    po_logs['available'] = (po_logs['minutes'] >= MIN_AVAILABLE_THRESHOLD).astype(int)

    # Team playoff game counts
    team_games_po = po_logs.groupby(
        ['TEAM_ID', 'SEASON']
    )['GAME_ID'].nunique().reset_index(name='playoff_games')

    # Per-player playoff availability
    player_po_avail = po_logs.groupby(
        ['PLAYER_ID', 'TEAM_ID', 'SEASON']
    ).agg(
        games_available=('available', 'sum'),
        games_appeared=('GAME_ID', 'count'),
    ).reset_index()

    player_po_avail = player_po_avail.merge(
        team_games_po, on=['TEAM_ID', 'SEASON'], how='left'
    )
    player_po_avail['availability_rate'] = (
        player_po_avail['games_available'] /
        player_po_avail['playoff_games'].clip(lower=1)
    )

    # --- STEP 3: Join rotation players with playoff availability ---
    merged = rotation.merge(
        player_po_avail[['PLAYER_ID', 'TEAM_ID', 'SEASON',
                          'games_available', 'availability_rate', 'playoff_games']],
        on=['PLAYER_ID', 'TEAM_ID', 'SEASON'],
        how='left'
    )

    # Players not in playoffs get availability_rate = NaN → fill 0
    # (they didn't make playoffs with this team)
    merged = merged[merged['availability_rate'].notna()].copy()

    # --- STEP 4: Aggregate to team level ---
    print("  Aggregating to team level...")

    team_avail = []

    for (team_id, season), group in merged.groupby(['TEAM_ID', 'SEASON']):
        group = group.sort_values('minutes_rank')
        n_rotation = len(group)

        # Star players
        star = group[group['minutes_rank'] == 1]
        second_star = group[group['minutes_rank'] == 2]

        star_avail = star['availability_rate'].values[0] if len(star) > 0 else 0.0
        second_star_avail = second_star['availability_rate'].values[0] if len(second_star) > 0 else 0.0

        # Lineup quality score: sum of (minutes_share × availability_rate)
        # Higher = more of your rotation was available weighted by their importance
        lineup_quality = (group['minutes_share'] * group['availability_rate']).sum()

        # Depth score: std deviation of availability rates
        # Low std = evenly distributed availability (deep team)
        # High std = some players healthy, some not (uneven)
        depth_score = 1.0 - group['availability_rate'].std() if n_rotation > 1 else 1.0

        # Avg availability across all rotation players
        avg_avail = group['availability_rate'].mean()

        # Injury games lost
        po_games = group['playoff_games'].max() if len(group) > 0 else 0
        expected_player_games = n_rotation * po_games
        actual_player_games = group['games_available'].sum()
        injury_games_lost = expected_player_games - actual_player_games

        team_avail.append({
            'TEAM_ID': team_id,
            'SEASON': season,
            'rotation_size': n_rotation,
            'avg_availability_rate': avg_avail,
            'star_availability_rate': star_avail,
            'second_star_availability_rate': second_star_avail,
            'lineup_quality_score': lineup_quality,
            'depth_score': max(depth_score, 0.0),
            'injury_games_lost': injury_games_lost,
            'playoff_games': po_games,
        })

    availability_df = pd.DataFrame(team_avail)
    print(f"  Built availability features for {len(availability_df)} team-seasons")

    # --- STEP 5: Build game-level availability for live tracking ---
    print("  Building game-level availability table...")

    # Join rotation players with each playoff game
    game_level = po_logs.merge(
        rotation[['PLAYER_ID', 'TEAM_ID', 'SEASON',
                   'PLAYER_NAME', 'avg_minutes', 'minutes_rank', 'minutes_share']],
        on=['PLAYER_ID', 'TEAM_ID', 'SEASON'],
        how='inner'  # only rotation players
    )

    game_avail_agg = game_level.groupby(
        ['TEAM_ID', 'SEASON', 'GAME_ID']
    ).agg(
        rotation_available=('available', 'sum'),
        rotation_total=('PLAYER_ID', 'count'),
        lineup_quality=('minutes_share', lambda x: (x * game_level.loc[x.index, 'available']).sum()),
        star_available=('available', lambda x: int(
            game_level.loc[x.index[game_level.loc[x.index, 'minutes_rank'] == 1], 'available'].sum() > 0
        ) if (game_level.loc[x.index, 'minutes_rank'] == 1).any() else 0),
    ).reset_index()

    game_avail_agg['game_availability_rate'] = (
        game_avail_agg['rotation_available'] /
        game_avail_agg['rotation_total'].clip(lower=1)
    )

    # --- STEP 6: Save to DuckDB ---
    con.execute("DROP TABLE IF EXISTS availability_features")
    con.execute("CREATE TABLE availability_features AS SELECT * FROM availability_df")
    print(f"\nSaved {len(availability_df)} rows → availability_features")

    con.execute("DROP TABLE IF EXISTS game_availability")
    con.execute("CREATE TABLE game_availability AS SELECT * FROM game_avail_agg")
    print(f"Saved {len(game_avail_agg)} rows → game_availability")

    # Preview
    print("\nAvailability feature summary:")
    print(availability_df[[
        'SEASON', 'TEAM_ID', 'rotation_size',
        'avg_availability_rate', 'star_availability_rate',
        'lineup_quality_score', 'depth_score'
    ]].describe().round(3).to_string())

    con.close()
    return availability_df


if __name__ == "__main__":
    df = compute_availability_features()
    print("\nSample:")
    print(df[[
        'TEAM_ID', 'SEASON', 'rotation_size',
        'avg_availability_rate', 'star_availability_rate',
        'lineup_quality_score', 'depth_score', 'injury_games_lost'
    ]].head(20).to_string())