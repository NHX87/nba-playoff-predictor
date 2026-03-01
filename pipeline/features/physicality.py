"""
physicality.py
--------------
Engineers physicality delta features — the core hypothesis of this model.

For each team and season, computes:
  - foul_rate_delta: how much foul rate changes regular season → playoffs
  - paint_touches_delta: paint touch rate change
  - defensive_rating_delta: defensive rating improvement in playoffs
  - contested_shots_delta: contested shot rate change

Positive delta = team gets MORE physical in playoffs (good signal)
Negative delta = team gets LESS physical (bad signal for this model)
"""

import duckdb
import pandas as pd
from config.settings import DB_PATH


def compute_physicality_features() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)

    print("Computing physicality features...")

    # Aggregate regular season stats per team per season
    reg_season = con.execute("""
        SELECT
            TEAM_ID,
            TEAM_ABBR,
            SEASON,
            COUNT(*) AS games_played,
            AVG(CAST(PF AS FLOAT)) AS avg_fouls,
            AVG(CAST(PFD AS FLOAT)) AS avg_fouls_drawn,
            AVG(CAST(FGA AS FLOAT)) AS avg_fga,
            AVG(CAST(FTA AS FLOAT)) AS avg_fta,
            -- Foul rate = fouls per 100 possessions (proxy)
            AVG(CAST(PF AS FLOAT)) / NULLIF(AVG(CAST(FGA AS FLOAT)), 0) * 100 AS foul_rate,
            -- Free throw rate as physicality proxy
            AVG(CAST(FTA AS FLOAT)) / NULLIF(AVG(CAST(FGA AS FLOAT)), 0) AS ft_rate,
            -- Defensive proxy: opponent FG% (not directly available, use DREB as proxy)
            AVG(CAST(DREB AS FLOAT)) AS avg_dreb,
            AVG(CAST(DREB AS FLOAT)) / NULLIF(AVG(CAST(REB AS FLOAT)), 0) AS dreb_pct
        FROM regular_season
        WHERE CAST(FGA AS FLOAT) > 0
        GROUP BY TEAM_ID, TEAM_ABBR, SEASON
    """).df()

    # Aggregate playoff stats per team per season
    playoff = con.execute("""
        SELECT
            TEAM_ID,
            TEAM_ABBR,
            SEASON,
            COUNT(*) AS playoff_games,
            AVG(CAST(PF AS FLOAT)) AS avg_fouls_playoff,
            AVG(CAST(FGA AS FLOAT)) AS avg_fga_playoff,
            AVG(CAST(FTA AS FLOAT)) AS avg_fta_playoff,
            AVG(CAST(PF AS FLOAT)) / NULLIF(AVG(CAST(FGA AS FLOAT)), 0) * 100 AS foul_rate_playoff,
            AVG(CAST(FTA AS FLOAT)) / NULLIF(AVG(CAST(FGA AS FLOAT)), 0) AS ft_rate_playoff,
            AVG(CAST(DREB AS FLOAT)) AS avg_dreb_playoff,
            AVG(CAST(DREB AS FLOAT)) / NULLIF(AVG(CAST(REB AS FLOAT)), 0) AS dreb_pct_playoff
        FROM playoffs
        WHERE CAST(FGA AS FLOAT) > 0
        GROUP BY TEAM_ID, TEAM_ABBR, SEASON
    """).df()

    # Merge and compute deltas
    merged = reg_season.merge(
        playoff,
        on=["TEAM_ID", "TEAM_ABBR", "SEASON"],
        how="inner"  # Only teams that made playoffs
    )

    merged["foul_rate_delta"] = merged["foul_rate_playoff"] - merged["foul_rate"]
    merged["ft_rate_delta"] = merged["ft_rate_playoff"] - merged["ft_rate"]
    merged["dreb_pct_delta"] = merged["dreb_pct_playoff"] - merged["dreb_pct"]

    # Physicality composite score (normalized)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    phys_cols = ["foul_rate_delta", "ft_rate_delta", "dreb_pct_delta"]
    merged["physicality_score"] = scaler.fit_transform(
        merged[phys_cols].fillna(0)
    ).mean(axis=1)

    # Save back to DuckDB
    con.execute("DROP TABLE IF EXISTS physicality_features")
    con.execute("CREATE TABLE physicality_features AS SELECT * FROM merged")
    print(f"Saved physicality features for {len(merged)} team-seasons")

    con.close()
    return merged


if __name__ == "__main__":
    df = compute_physicality_features()
    print(df[["TEAM_ABBR", "SEASON", "foul_rate_delta", "physicality_score"]].head(20))
