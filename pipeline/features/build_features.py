"""
build_features.py
-----------------
Joins all individual feature tables into a single model_features table
that the survival model and matchup model consume.

Run this after all individual feature modules have been executed:
  - physicality.py
  - team_stats.py
  - pace.py
  - home_court.py
  - availability.py

Output table: model_features (one row per team per playoff season)
"""

import duckdb
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import DB_PATH


def build_model_features() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    print("Building model features table...")

    # Check which feature tables exist
    tables = con.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'main'
    """).df()["table_name"].tolist()

    print(f"Available tables: {tables}")

    # --- BASE: physicality features (required) ---
    if "physicality_features" not in tables:
        raise ValueError("physicality_features table not found. Run physicality.py first.")

    query = """
        SELECT
            p.TEAM_ID,
            p.TEAM_ABBR,
            p.SEASON,
            p.games_played,
            p.playoff_games,

            -- Physicality features
            p.foul_rate,
            p.foul_rate_playoff,
            p.foul_rate_delta,
            p.ft_rate,
            p.ft_rate_playoff,
            p.ft_rate_delta,
            p.dreb_pct,
            p.dreb_pct_playoff,
            p.dreb_pct_delta,
            p.physicality_score
    """

    from_clause = "FROM physicality_features p"
    joins = []

    # --- JOIN: team stats features ---
    if "team_stats_features" in tables:
        query += """,
            -- Team stats features
            ts.three_pt_rate_delta,
            ts.ts_pct_delta,
            ts.efg_pct_delta,
            ts.tov_rate_delta,
            ts.blk_rate_delta,
            ts.stl_rate_delta,
            ts.ppg_delta,
            ts.rs_close_game_win_pct,
            ts.rs_close_game_count,
            ts.playoff_rounds_reached,
            ts.playoff_rounds_prior,
            ts.offensive_adaptability_score,
            ts.defensive_intensity_score,
            ts.ball_security_score,
            ts.po_games_played,
            ts.rs_ppg,
            ts.rs_win_pct,
            ts.rs_fta,
            ts.rs_efg_pct,
            ts.rs_sos_win_pct_avg,
            ts.rs_opp_net_rating_avg,
            ts.rs_rest_days_avg,
            ts.rs_b2b_rate,
            ts.rs_rest_travel_burden,
            ts.rs_home_win_pct,
            ts.rs_away_win_pct,
            ts.rs_home_away_win_pct_gap,
            ts.rs_tov_forced_rate,
            ts.rs_reb_pct,
            ts.po_ppg,
            ts.rs_off_rating,
            ts.rs_def_rating,
            ts.rs_net_rating,
            ts.po_off_rating,
            ts.po_def_rating,
            ts.po_net_rating,
            ts.off_rating_delta,
            ts.def_rating_delta,
            ts.net_rating_delta,
            ts.rs_vs_top_teams_win_pct,
            ts.rs_vs_top_teams_games
        """
        joins.append("""
            LEFT JOIN team_stats_features ts
            ON p.TEAM_ID = ts.TEAM_ID AND p.SEASON = ts.SEASON
        """)

    # --- JOIN: pace features (when built) ---
    if "pace_features" in tables:
        query += """,
            pc.pace_delta,
            pc.pace_reg,
            pc.pace_playoff,
            pc.pace_tolerance_score
        """
        joins.append("""
            LEFT JOIN pace_features pc
            ON p.TEAM_ID = pc.TEAM_ID AND p.SEASON = pc.SEASON
        """)

    # --- JOIN: availability features (when built) ---
    if "availability_features" in tables:
        query += """,
            av.lineup_quality_score,
            av.depth_score,
            av.star_availability_rate,
            av.second_star_availability_rate,
            av.injury_games_lost,
            av.rotation_size,
            av.avg_availability_rate
        """
        joins.append("""
            LEFT JOIN availability_features av
            ON p.TEAM_ID = av.TEAM_ID AND p.SEASON = av.SEASON
        """)

    # Build full query
    full_query = query + "\n" + from_clause + "\n" + "\n".join(joins)
    print(f"\nExecuting join query...")

    model_features = con.execute(full_query).df()

    # --- ADD DERIVED FEATURES ---

    # Composite playoff readiness score
    available_composite_cols = []

    if "physicality_score" in model_features.columns:
        available_composite_cols.append("physicality_score")

    if "offensive_adaptability_score" in model_features.columns:
        available_composite_cols.append("offensive_adaptability_score")

    if "defensive_intensity_score" in model_features.columns:
        available_composite_cols.append("defensive_intensity_score")

    if "ball_security_score" in model_features.columns:
        available_composite_cols.append("ball_security_score")

    if available_composite_cols:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        model_features["playoff_readiness_score"] = scaler.fit_transform(
            model_features[available_composite_cols].fillna(0)
        ).mean(axis=1)
        print(f"Playoff readiness score built from: {available_composite_cols}")

    # --- SAVE ---
    con.execute("DROP TABLE IF EXISTS model_features")
    con.execute("CREATE TABLE model_features AS SELECT * FROM model_features")
    print(f"\nmodel_features table built: {len(model_features)} rows × {len(model_features.columns)} columns")

    # Summary stats
    print("\nFeature summary:")
    print(model_features.describe().round(3).to_string())

    con.close()
    return model_features


def get_feature_list() -> list:
    """Returns the list of features available for modeling."""
    con = duckdb.connect(DB_PATH)
    try:
        cols = con.execute("DESCRIBE model_features").df()["column_name"].tolist()
        # Exclude ID and metadata columns
        exclude = ["TEAM_ID", "TEAM_ABBR", "SEASON", "games_played",
                   "playoff_games", "po_games_played"]
        return [c for c in cols if c not in exclude]
    except Exception:
        return []
    finally:
        con.close()


if __name__ == "__main__":
    df = build_model_features()
    print(f"\nFeatures available for modeling:")
    for col in get_feature_list():
        print(f"  - {col}")
