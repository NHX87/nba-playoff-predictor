"""
team_stats.py
-------------
Engineers team performance features that capture how a team's
fundamental stats and efficiency shift from regular season to playoffs.

Tables written to DuckDB:
  - team_stats_features

Features:
  Offensive Identity
    - three_pt_rate_delta       : 3PA/FGA change RS → playoffs
    - paint_scoring_pct_delta   : paint FGA% change
    - tov_rate_delta            : turnover rate change

  Defensive Identity
    - opp_paint_score_delta     : opponent paint scoring allowed delta
    - blk_rate_delta            : block rate delta
    - stl_rate_delta            : steal rate delta

  Shooting Efficiency
    - ts_pct_delta              : true shooting % delta
    - efg_pct_delta             : effective FG% delta
    - ft_rate_delta             : free throw rate delta (FTA/FGA)

  Clutch & Close Games
    - close_game_win_pct        : win % in games decided by 5 or fewer points
    - close_game_count          : sample size for close game metric

  Playoff Experience
    - avg_playoff_games         : avg playoff games played by roster (proxy: prior seasons)
    - playoff_rounds_prior      : how deep did this team go last postseason
"""

import duckdb
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import DB_PATH


def compute_team_stats_features() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    print("Computing team stats features...")

    # --- REGULAR SEASON STATS ---
    reg = con.execute("""
        SELECT
            TEAM_ID,
            TEAM_ABBR,
            SEASON,
            COUNT(*) AS rs_games,
            AVG(CASE WHEN WL = 'W' THEN 1.0 ELSE 0.0 END) AS rs_win_pct,

            -- Shooting efficiency
            AVG(CAST(PTS AS FLOAT)) AS rs_ppg,
            AVG(CAST(FGM AS FLOAT)) AS rs_fgm,
            AVG(CAST(FGA AS FLOAT)) AS rs_fga,
            AVG(CAST(FG_PCT AS FLOAT)) AS rs_fg_pct,
            AVG(CAST(FG3M AS FLOAT)) AS rs_fg3m,
            AVG(CAST(FG3A AS FLOAT)) AS rs_fg3a,
            AVG(CAST(FTA AS FLOAT)) AS rs_fta,
            AVG(CAST(FTM AS FLOAT)) AS rs_ftm,

            -- Three point rate
            AVG(CAST(FG3A AS FLOAT)) /
                NULLIF(AVG(CAST(FGA AS FLOAT)), 0) AS rs_three_pt_rate,

            -- Free throw rate
            AVG(CAST(FTA AS FLOAT)) /
                NULLIF(AVG(CAST(FGA AS FLOAT)), 0) AS rs_ft_rate,

            -- True shooting %
            AVG(CAST(PTS AS FLOAT)) / NULLIF(
                2 * (AVG(CAST(FGA AS FLOAT)) +
                     0.44 * AVG(CAST(FTA AS FLOAT))), 0
            ) AS rs_ts_pct,

            -- eFG%
            (AVG(CAST(FGM AS FLOAT)) + 0.5 * AVG(CAST(FG3M AS FLOAT))) /
                NULLIF(AVG(CAST(FGA AS FLOAT)), 0) AS rs_efg_pct,

            -- Defensive proxies
            AVG(CAST(STL AS FLOAT)) AS rs_stl,
            AVG(CAST(BLK AS FLOAT)) AS rs_blk,
            AVG(CAST(TOV AS FLOAT)) AS rs_tov,
            AVG(CAST(DREB AS FLOAT)) AS rs_dreb,

            -- Possessions and ratings
            SUM(CAST(PTS AS FLOAT)) AS rs_pts_total,
            SUM(CAST(PTS AS FLOAT) - CAST(PLUS_MINUS AS FLOAT)) AS rs_opp_pts_total,
            SUM(
                CAST(FGA AS FLOAT)
                + 0.44 * CAST(FTA AS FLOAT)
                - CAST(OREB AS FLOAT)
                + CAST(TOV AS FLOAT)
            ) AS rs_possessions_total,
            SUM(CAST(PTS AS FLOAT)) / NULLIF(
                SUM(
                    CAST(FGA AS FLOAT)
                    + 0.44 * CAST(FTA AS FLOAT)
                    - CAST(OREB AS FLOAT)
                    + CAST(TOV AS FLOAT)
                ), 0
            ) * 100 AS rs_off_rating,
            SUM(CAST(PTS AS FLOAT) - CAST(PLUS_MINUS AS FLOAT)) / NULLIF(
                SUM(
                    CAST(FGA AS FLOAT)
                    + 0.44 * CAST(FTA AS FLOAT)
                    - CAST(OREB AS FLOAT)
                    + CAST(TOV AS FLOAT)
                ), 0
            ) * 100 AS rs_def_rating,
            (
                SUM(CAST(PTS AS FLOAT)) / NULLIF(
                    SUM(
                        CAST(FGA AS FLOAT)
                        + 0.44 * CAST(FTA AS FLOAT)
                        - CAST(OREB AS FLOAT)
                        + CAST(TOV AS FLOAT)
                    ), 0
                ) * 100
            ) - (
                SUM(CAST(PTS AS FLOAT) - CAST(PLUS_MINUS AS FLOAT)) / NULLIF(
                    SUM(
                        CAST(FGA AS FLOAT)
                        + 0.44 * CAST(FTA AS FLOAT)
                        - CAST(OREB AS FLOAT)
                        + CAST(TOV AS FLOAT)
                    ), 0
                ) * 100
            ) AS rs_net_rating,

            -- Turnover rate (per 100 possessions proxy)
            AVG(CAST(TOV AS FLOAT)) /
                NULLIF(AVG(CAST(FGA AS FLOAT)) +
                       0.44 * AVG(CAST(FTA AS FLOAT)) +
                       AVG(CAST(TOV AS FLOAT)), 0) * 100 AS rs_tov_rate,

            -- Block rate
            AVG(CAST(BLK AS FLOAT)) /
                NULLIF(AVG(CAST(FGA AS FLOAT)), 0) * 100 AS rs_blk_rate,

            -- Steal rate
            AVG(CAST(STL AS FLOAT)) /
                NULLIF(AVG(CAST(FGA AS FLOAT)), 0) * 100 AS rs_stl_rate,

            -- Close game win % (margin proxy using PLUS_MINUS)
            AVG(CASE
                WHEN ABS(CAST(PLUS_MINUS AS FLOAT)) <= 5
                THEN CASE WHEN WL = 'W' THEN 1.0 ELSE 0.0 END
                ELSE NULL
            END) AS rs_close_game_win_pct,

            COUNT(CASE
                WHEN ABS(CAST(PLUS_MINUS AS FLOAT)) <= 5
                THEN 1 ELSE NULL
            END) AS rs_close_game_count

        FROM regular_season
        WHERE CAST(FGA AS FLOAT) > 0
        GROUP BY TEAM_ID, TEAM_ABBR, SEASON
    """).df()

    # --- PLAYOFF STATS ---
    po = con.execute("""
        SELECT
            TEAM_ID,
            TEAM_ABBR,
            SEASON,
            COUNT(*) AS po_games,

            AVG(CAST(PTS AS FLOAT)) AS po_ppg,
            AVG(CAST(FGM AS FLOAT)) AS po_fgm,
            AVG(CAST(FGA AS FLOAT)) AS po_fga,
            AVG(CAST(FG_PCT AS FLOAT)) AS po_fg_pct,
            AVG(CAST(FG3M AS FLOAT)) AS po_fg3m,
            AVG(CAST(FG3A AS FLOAT)) AS po_fg3a,
            AVG(CAST(FTA AS FLOAT)) AS po_fta,
            AVG(CAST(FTM AS FLOAT)) AS po_ftm,

            AVG(CAST(FG3A AS FLOAT)) /
                NULLIF(AVG(CAST(FGA AS FLOAT)), 0) AS po_three_pt_rate,

            AVG(CAST(FTA AS FLOAT)) /
                NULLIF(AVG(CAST(FGA AS FLOAT)), 0) AS po_ft_rate,

            AVG(CAST(PTS AS FLOAT)) / NULLIF(
                2 * (AVG(CAST(FGA AS FLOAT)) +
                     0.44 * AVG(CAST(FTA AS FLOAT))), 0
            ) AS po_ts_pct,

            (AVG(CAST(FGM AS FLOAT)) + 0.5 * AVG(CAST(FG3M AS FLOAT))) /
                NULLIF(AVG(CAST(FGA AS FLOAT)), 0) AS po_efg_pct,

            AVG(CAST(STL AS FLOAT)) AS po_stl,
            AVG(CAST(BLK AS FLOAT)) AS po_blk,
            AVG(CAST(TOV AS FLOAT)) AS po_tov,
            AVG(CAST(DREB AS FLOAT)) AS po_dreb,

            -- Possessions and ratings
            SUM(CAST(PTS AS FLOAT)) AS po_pts_total,
            SUM(CAST(PTS AS FLOAT) - CAST(PLUS_MINUS AS FLOAT)) AS po_opp_pts_total,
            SUM(
                CAST(FGA AS FLOAT)
                + 0.44 * CAST(FTA AS FLOAT)
                - CAST(OREB AS FLOAT)
                + CAST(TOV AS FLOAT)
            ) AS po_possessions_total,
            SUM(CAST(PTS AS FLOAT)) / NULLIF(
                SUM(
                    CAST(FGA AS FLOAT)
                    + 0.44 * CAST(FTA AS FLOAT)
                    - CAST(OREB AS FLOAT)
                    + CAST(TOV AS FLOAT)
                ), 0
            ) * 100 AS po_off_rating,
            SUM(CAST(PTS AS FLOAT) - CAST(PLUS_MINUS AS FLOAT)) / NULLIF(
                SUM(
                    CAST(FGA AS FLOAT)
                    + 0.44 * CAST(FTA AS FLOAT)
                    - CAST(OREB AS FLOAT)
                    + CAST(TOV AS FLOAT)
                ), 0
            ) * 100 AS po_def_rating,
            (
                SUM(CAST(PTS AS FLOAT)) / NULLIF(
                    SUM(
                        CAST(FGA AS FLOAT)
                        + 0.44 * CAST(FTA AS FLOAT)
                        - CAST(OREB AS FLOAT)
                        + CAST(TOV AS FLOAT)
                    ), 0
                ) * 100
            ) - (
                SUM(CAST(PTS AS FLOAT) - CAST(PLUS_MINUS AS FLOAT)) / NULLIF(
                    SUM(
                        CAST(FGA AS FLOAT)
                        + 0.44 * CAST(FTA AS FLOAT)
                        - CAST(OREB AS FLOAT)
                        + CAST(TOV AS FLOAT)
                    ), 0
                ) * 100
            ) AS po_net_rating,

            AVG(CAST(TOV AS FLOAT)) /
                NULLIF(AVG(CAST(FGA AS FLOAT)) +
                       0.44 * AVG(CAST(FTA AS FLOAT)) +
                       AVG(CAST(TOV AS FLOAT)), 0) * 100 AS po_tov_rate,

            AVG(CAST(BLK AS FLOAT)) /
                NULLIF(AVG(CAST(FGA AS FLOAT)), 0) * 100 AS po_blk_rate,

            AVG(CAST(STL AS FLOAT)) /
                NULLIF(AVG(CAST(FGA AS FLOAT)), 0) * 100 AS po_stl_rate,

            -- Rounds reached (proxy: playoff games played)
            COUNT(*) AS po_games_played

        FROM playoffs
        WHERE CAST(FGA AS FLOAT) > 0
        GROUP BY TEAM_ID, TEAM_ABBR, SEASON
    """).df()

    # --- RECORD VS TOP TEAMS (regular season) ---
    # Top teams are season peers with regular-season win pct >= 0.600.
    rs_vs_top = con.execute("""
        WITH team_records AS (
            SELECT
                SEASON,
                TEAM_ABBR,
                AVG(CASE WHEN WL = 'W' THEN 1.0 ELSE 0.0 END) AS win_pct
            FROM regular_season
            GROUP BY SEASON, TEAM_ABBR
        ),
        top_teams AS (
            SELECT SEASON, TEAM_ABBR
            FROM team_records
            WHERE win_pct >= 0.600
        ),
        games AS (
            SELECT
                TEAM_ID,
                TEAM_ABBR,
                SEASON,
                WL,
                RIGHT(MATCHUP, 3) AS opp_abbr
            FROM regular_season
        )
        SELECT
            g.TEAM_ID,
            g.TEAM_ABBR,
            g.SEASON,
            AVG(
                CASE
                    WHEN t.TEAM_ABBR IS NOT NULL
                    THEN CASE WHEN g.WL = 'W' THEN 1.0 ELSE 0.0 END
                    ELSE NULL
                END
            ) AS rs_vs_top_teams_win_pct,
            COUNT(CASE WHEN t.TEAM_ABBR IS NOT NULL THEN 1 ELSE NULL END) AS rs_vs_top_teams_games
        FROM games g
        LEFT JOIN top_teams t
            ON g.SEASON = t.SEASON
           AND g.opp_abbr = t.TEAM_ABBR
        GROUP BY g.TEAM_ID, g.TEAM_ABBR, g.SEASON
    """).df()

    # --- MERGE AND COMPUTE DELTAS ---
    merged = reg.merge(
        po,
        on=["TEAM_ID", "TEAM_ABBR", "SEASON"],
        how="inner"  # Only teams that made playoffs
    )

    merged = merged.merge(
        rs_vs_top,
        on=["TEAM_ID", "TEAM_ABBR", "SEASON"],
        how="left"
    )

    merged["rs_vs_top_teams_win_pct"] = merged["rs_vs_top_teams_win_pct"].fillna(0.0)
    merged["rs_vs_top_teams_games"] = merged["rs_vs_top_teams_games"].fillna(0)

    # Compute deltas
    merged["three_pt_rate_delta"] = merged["po_three_pt_rate"] - merged["rs_three_pt_rate"]
    merged["ft_rate_delta"] = merged["po_ft_rate"] - merged["rs_ft_rate"]
    merged["ts_pct_delta"] = merged["po_ts_pct"] - merged["rs_ts_pct"]
    merged["efg_pct_delta"] = merged["po_efg_pct"] - merged["rs_efg_pct"]
    merged["tov_rate_delta"] = merged["po_tov_rate"] - merged["rs_tov_rate"]
    merged["blk_rate_delta"] = merged["po_blk_rate"] - merged["rs_blk_rate"]
    merged["stl_rate_delta"] = merged["po_stl_rate"] - merged["rs_stl_rate"]
    merged["ppg_delta"] = merged["po_ppg"] - merged["rs_ppg"]
    merged["off_rating_delta"] = merged["po_off_rating"] - merged["rs_off_rating"]
    merged["def_rating_delta"] = merged["po_def_rating"] - merged["rs_def_rating"]
    merged["net_rating_delta"] = merged["po_net_rating"] - merged["rs_net_rating"]

    # Playoff rounds proxy (games played → approximate rounds)
    # Exact rounds reached from series data
    series_summary = con.execute("""
        SELECT team_id, season, rounds_reached, series_wins
        FROM team_series_summary
    """).df()

    merged = merged.merge(
        series_summary,
        left_on=['TEAM_ID', 'SEASON'],
        right_on=['team_id', 'season'],
        how='left'
    )

    # Drop duplicate columns from merge
    merged = merged.drop(columns=['team_id', 'season'], errors='ignore')
    merged = merged.rename(columns={'rounds_reached': 'playoff_rounds_reached'})

    # Prior season playoff depth (shift by season)
    merged = merged.sort_values(["TEAM_ID", "SEASON"])
    merged["playoff_rounds_prior"] = merged.groupby("TEAM_ID")["playoff_rounds_reached"].shift(1).fillna(0)

    # --- COMPOSITE SCORES ---
    scaler = StandardScaler()

    # Offensive adaptability score
    off_cols = ["three_pt_rate_delta", "ts_pct_delta", "efg_pct_delta"]
    merged["offensive_adaptability_score"] = scaler.fit_transform(
        merged[off_cols].fillna(0)
    ).mean(axis=1)

    # Defensive intensity score
    def_cols = ["blk_rate_delta", "stl_rate_delta"]
    merged["defensive_intensity_score"] = scaler.fit_transform(
        merged[def_cols].fillna(0)
    ).mean(axis=1)

    # Ball security score (negative tov_rate_delta is good)
    merged["ball_security_score"] = -merged["tov_rate_delta"].fillna(0)

    # --- SAVE TO DUCKDB ---
    con.execute("DROP TABLE IF EXISTS team_stats_features")
    con.execute("CREATE TABLE team_stats_features AS SELECT * FROM merged")
    print(f"Saved team stats features for {len(merged)} team-seasons")

    con.close()
    return merged


if __name__ == "__main__":
    df = compute_team_stats_features()
    print("\nSample output:")
    print(df[[
        "TEAM_ABBR", "SEASON",
        "rs_win_pct",
        "rs_off_rating", "rs_def_rating", "rs_net_rating",
        "rs_vs_top_teams_win_pct", "rs_vs_top_teams_games",
        "off_rating_delta", "def_rating_delta", "net_rating_delta",
        "playoff_rounds_reached"
    ]].head(20).to_string())
