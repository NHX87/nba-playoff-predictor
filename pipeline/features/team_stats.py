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

    # --- RS CONTEXT FEATURES ---
    # Adds schedule/context metrics requested for RS-only modeling:
    # - opponent strength (SOS proxies)
    # - rest/travel stress
    # - home/away splits
    # - turnover forced rate
    # - rebound rate
    rs_context = con.execute("""
        WITH games AS (
            SELECT
                TEAM_ID,
                TEAM_ABBR,
                SEASON,
                GAME_ID,
                WL,
                CASE WHEN WL = 'W' THEN 1.0 ELSE 0.0 END AS wl_win,
                RIGHT(MATCHUP, 3) AS opp_abbr,
                CASE WHEN MATCHUP LIKE '% vs. %' THEN 1 ELSE 0 END AS is_home,
                CASE WHEN MATCHUP LIKE '% @ %' THEN 1 ELSE 0 END AS is_away,
                COALESCE(
                    TRY_CAST(GAME_DATE AS DATE),
                    TRY_STRPTIME(GAME_DATE, '%Y-%m-%dT%H:%M:%S'),
                    TRY_STRPTIME(GAME_DATE, '%b %d, %Y'),
                    TRY_STRPTIME(GAME_DATE, '%Y-%m-%d')
                )::DATE AS game_dt,
                CAST(FGA AS FLOAT) AS fga,
                CAST(FTA AS FLOAT) AS fta,
                CAST(OREB AS FLOAT) AS oreb,
                CAST(TOV AS FLOAT) AS tov,
                CAST(REB AS FLOAT) AS reb
            FROM regular_season
            WHERE CAST(FGA AS FLOAT) > 0
        ),
        games_rest AS (
            SELECT
                g.*,
                DATE_DIFF(
                    'day',
                    LAG(game_dt) OVER (PARTITION BY TEAM_ID, SEASON ORDER BY game_dt, GAME_ID),
                    game_dt
                ) AS rest_days
            FROM games g
        ),
        team_records AS (
            SELECT
                SEASON,
                TEAM_ABBR,
                AVG(CASE WHEN WL = 'W' THEN 1.0 ELSE 0.0 END) AS team_win_pct,
                SUM(CAST(PTS AS FLOAT)) / NULLIF(
                    SUM(
                        CAST(FGA AS FLOAT)
                        + 0.44 * CAST(FTA AS FLOAT)
                        - CAST(OREB AS FLOAT)
                        + CAST(TOV AS FLOAT)
                    ), 0
                ) * 100 AS team_off_rating,
                SUM(CAST(PTS AS FLOAT) - CAST(PLUS_MINUS AS FLOAT)) / NULLIF(
                    SUM(
                        CAST(FGA AS FLOAT)
                        + 0.44 * CAST(FTA AS FLOAT)
                        - CAST(OREB AS FLOAT)
                        + CAST(TOV AS FLOAT)
                    ), 0
                ) * 100 AS team_def_rating
            FROM regular_season
            WHERE CAST(FGA AS FLOAT) > 0
            GROUP BY SEASON, TEAM_ABBR
        ),
        game_pairs AS (
            SELECT
                a.TEAM_ID,
                a.TEAM_ABBR,
                a.SEASON,
                a.GAME_ID,
                a.wl_win,
                a.is_home,
                a.is_away,
                a.rest_days,
                a.opp_abbr,
                a.reb AS team_reb,
                b.reb AS opp_reb,
                b.tov AS opp_tov,
                (
                    b.fga
                    + 0.44 * b.fta
                    - b.oreb
                    + b.tov
                ) AS opp_possessions
            FROM games_rest a
            JOIN games_rest b
              ON a.SEASON = b.SEASON
             AND a.GAME_ID = b.GAME_ID
             AND a.TEAM_ID <> b.TEAM_ID
        )
        SELECT
            gp.TEAM_ID,
            gp.TEAM_ABBR,
            gp.SEASON,
            AVG(tr.team_win_pct) AS rs_sos_win_pct_avg,
            AVG(tr.team_off_rating - tr.team_def_rating) AS rs_opp_net_rating_avg,
            AVG(CASE WHEN gp.rest_days IS NOT NULL THEN gp.rest_days ELSE NULL END) AS rs_rest_days_avg,
            AVG(CASE WHEN gp.rest_days <= 1 THEN 1.0 ELSE 0.0 END) AS rs_b2b_rate,
            AVG(CASE WHEN gp.is_away = 1 AND gp.rest_days <= 1 THEN 1.0 ELSE 0.0 END) AS rs_rest_travel_burden,
            AVG(CASE WHEN gp.is_home = 1 THEN gp.wl_win ELSE NULL END) AS rs_home_win_pct,
            AVG(CASE WHEN gp.is_away = 1 THEN gp.wl_win ELSE NULL END) AS rs_away_win_pct,
            AVG(CASE WHEN gp.is_home = 1 THEN gp.wl_win ELSE NULL END)
                - AVG(CASE WHEN gp.is_away = 1 THEN gp.wl_win ELSE NULL END) AS rs_home_away_win_pct_gap,
            SUM(gp.opp_tov) / NULLIF(SUM(gp.opp_possessions), 0) * 100 AS rs_tov_forced_rate,
            SUM(gp.team_reb) / NULLIF(SUM(gp.team_reb + gp.opp_reb), 0) AS rs_reb_pct
        FROM game_pairs gp
        LEFT JOIN team_records tr
          ON gp.SEASON = tr.SEASON
         AND gp.opp_abbr = tr.TEAM_ABBR
        GROUP BY gp.TEAM_ID, gp.TEAM_ABBR, gp.SEASON
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
    merged = merged.merge(
        rs_context,
        on=["TEAM_ID", "TEAM_ABBR", "SEASON"],
        how="left"
    )

    merged["rs_vs_top_teams_win_pct"] = merged["rs_vs_top_teams_win_pct"].fillna(0.0)
    merged["rs_vs_top_teams_games"] = merged["rs_vs_top_teams_games"].fillna(0)
    merged["rs_sos_win_pct_avg"] = merged["rs_sos_win_pct_avg"].fillna(merged["rs_win_pct"])
    merged["rs_opp_net_rating_avg"] = merged["rs_opp_net_rating_avg"].fillna(0.0)
    merged["rs_rest_days_avg"] = merged["rs_rest_days_avg"].fillna(2.0)
    merged["rs_b2b_rate"] = merged["rs_b2b_rate"].fillna(0.0)
    merged["rs_rest_travel_burden"] = merged["rs_rest_travel_burden"].fillna(0.0)
    merged["rs_home_win_pct"] = merged["rs_home_win_pct"].fillna(merged["rs_win_pct"])
    merged["rs_away_win_pct"] = merged["rs_away_win_pct"].fillna(merged["rs_win_pct"])
    merged["rs_home_away_win_pct_gap"] = merged["rs_home_away_win_pct_gap"].fillna(0.0)
    merged["rs_tov_forced_rate"] = merged["rs_tov_forced_rate"].fillna(0.0)
    merged["rs_reb_pct"] = merged["rs_reb_pct"].fillna(0.5)

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
        "rs_sos_win_pct_avg", "rs_opp_net_rating_avg",
        "rs_rest_days_avg", "rs_b2b_rate", "rs_rest_travel_burden",
        "rs_home_win_pct", "rs_away_win_pct", "rs_home_away_win_pct_gap",
        "rs_tov_forced_rate", "rs_reb_pct",
        "rs_off_rating", "rs_def_rating", "rs_net_rating",
        "rs_vs_top_teams_win_pct", "rs_vs_top_teams_games",
        "off_rating_delta", "def_rating_delta", "net_rating_delta",
        "playoff_rounds_reached"
    ]].head(20).to_string())
