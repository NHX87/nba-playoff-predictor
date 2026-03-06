"""
main.py
-------
Streamlit entry point for NBA Playoff Predictor analytics app.

Run with: streamlit run app/main.py
"""

import math
import sys
from html import escape
from pathlib import Path
from textwrap import dedent

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (  # noqa: E402
    CURRENT_SEASON_STR,
    DB_PATH,
)


def pct(val: float) -> str:
    return f"{val:.1%}"


# NBA API abbreviations that differ from ESPN CDN slugs
_ESPN_SLUG: dict[str, str] = {
    "NOP": "no", "UTA": "utah", "GSW": "gs", "SAS": "sa", "NYK": "ny",
}


def logo_url(team_abbr: str) -> str:
    slug = _ESPN_SLUG.get(team_abbr.upper(), team_abbr.lower())
    return f"https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{slug}.png"


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _fmt_dt(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "n/a"
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


TEAM_COLORS = {
    "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#FFFFFF", "CHA": "#1D1160", "CHI": "#CE1141",
    "CLE": "#860038", "DAL": "#00538C", "DEN": "#FEC524", "DET": "#C8102E", "GSW": "#1D428A",
    "HOU": "#CE1141", "IND": "#FDBB30", "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9",
    "MIA": "#98002E", "MIL": "#00471B", "MIN": "#236192", "NOP": "#0C2340", "NYK": "#F58426",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#E56020", "POR": "#E03A3E",
    "SAC": "#5A2D81", "SAS": "#C4CED4", "TOR": "#CE1141", "UTA": "#002B5C", "WAS": "#002B5C",
}


@st.cache_data(ttl=300)
def load_base_tables() -> dict[str, pd.DataFrame]:
    """Load required app and modeling tables from DuckDB."""
    out: dict[str, pd.DataFrame] = {}
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
    except Exception:
        return out

    queries = {
        "title": """
            SELECT TEAM_ABBR, conference, playoff_seed, Record,
                   title_prob, make_finals_prob, make_conf_finals_prob, make_second_round_prob
            FROM app_title_odds_current
            ORDER BY title_prob DESC
        """,
        "series": """
            SELECT conference, round, high_seed, high_team, low_seed, low_team,
                   high_team_win_prob, low_team_win_prob, predicted_winner,
                   p_4_games, p_5_games, p_6_games, p_7_games, expected_games
            FROM app_series_predictions_current
            ORDER BY
                CASE round
                    WHEN 'First Round' THEN 1
                    WHEN 'Conference Semifinals' THEN 2
                    WHEN 'Conference Finals' THEN 3
                    WHEN 'NBA Finals' THEN 4
                    ELSE 5
                END,
                conference,
                high_seed
        """,
        "play_in": """
            SELECT conference, team_abbr, seed7_prob, seed8_prob, made_playoffs_prob
            FROM app_play_in_current
            ORDER BY conference, made_playoffs_prob DESC
        """,
        "teams": "SELECT abbreviation AS TEAM_ABBR, full_name AS TEAM_NAME, city FROM teams ORDER BY abbreviation",
        "rs": f"""
            SELECT TEAM_ABBR, TEAM_NAME, GAME_ID, GAME_DATE, MATCHUP, WL, PTS, REB, AST, TOV, PLUS_MINUS
            FROM regular_season
            WHERE SEASON = '{CURRENT_SEASON_STR}'
        """,
        "player_rs": f"""
            SELECT TEAM_ABBREVIATION AS TEAM_ABBR, PLAYER_NAME, GAME_DATE, WL, MIN, PTS, REB, AST, STL, BLK, TOV
            FROM raw_player_logs_rs
            WHERE SEASON = '{CURRENT_SEASON_STR}'
        """,
        "current_preds": f"""
            SELECT TEAM_ABBR, conference, wins, losses, win_pct, pred_rank_all_30,
                   playoff_rank, playoff_seed, title_prob_proxy_all_30
            FROM current_season_predictions
            WHERE SEASON = '{CURRENT_SEASON_STR}'
            ORDER BY pred_rank_all_30
        """,
        "features": f"""
            SELECT TEAM_ABBR, rs_net_rating, rs_off_rating, rs_def_rating, rs_vs_top_teams_win_pct
            FROM model_features
            WHERE SEASON = '{CURRENT_SEASON_STR}'
        """,
        "remaining_games": f"""
            SELECT TEAM_ABBR, GAME_DATE, OPP_ABBR, IS_HOME, OPP_NET_RATING, GAME_WIN_PROB
            FROM team_remaining_games
            WHERE SEASON = '{CURRENT_SEASON_STR}'
            ORDER BY TEAM_ABBR, GAME_DATE
        """,
        "projected_records": f"""
            SELECT TEAM_ABBR, CONFERENCE, current_wins, current_losses, games_remaining,
                   expected_final_wins, expected_final_losses,
                   p10_final_wins, p90_final_wins,
                   prob_make_top6, prob_make_playin, prob_miss_playoffs, projected_seed_median
            FROM team_projected_record
            WHERE SEASON = '{CURRENT_SEASON_STR}'
        """,
    }

    try:
        tables = set(
            con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").df()[
                "table_name"
            ]
        )
        mapping = {
            "title": "app_title_odds_current",
            "series": "app_series_predictions_current",
            "play_in": "app_play_in_current",
            "teams": "teams",
            "rs": "regular_season",
            "player_rs": "raw_player_logs_rs",
            "current_preds": "current_season_predictions",
            "features": "model_features",
            "remaining_games": "team_remaining_games",
            "projected_records": "team_projected_record",
        }
        for key, sql in queries.items():
            out[key] = con.execute(sql).df() if mapping[key] in tables else pd.DataFrame()
    finally:
        con.close()

    if not out.get("rs", pd.DataFrame()).empty:
        out["rs"]["GAME_DATE"] = _to_dt(out["rs"]["GAME_DATE"])

    if not out.get("player_rs", pd.DataFrame()).empty:
        out["player_rs"]["GAME_DATE"] = _to_dt(out["player_rs"]["GAME_DATE"])

    if not out.get("remaining_games", pd.DataFrame()).empty:
        out["remaining_games"]["GAME_DATE"] = _to_dt(out["remaining_games"]["GAME_DATE"])

    return out


def build_team_title_odds_series(
    rs_df: pd.DataFrame, title_df: pd.DataFrame, team_abbr: str
) -> pd.DataFrame:
    """Build implied title odds trajectory from RS win% + net rating softmax.

    Shows the raw relative strength of this team vs all 30 teams on each game
    date — no endpoint scaling so historical values reflect what the model
    would have said at that point in the season.
    """
    if rs_df.empty:
        return pd.DataFrame()

    season = rs_df.copy().dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE")
    if season.empty:
        return pd.DataFrame()

    season["win"] = (season["WL"] == "W").astype(int)
    season["pm"] = season["PLUS_MINUS"].fillna(0).astype(float)

    season["gp"] = season.groupby("TEAM_ABBR").cumcount() + 1
    season["cum_wins"] = season.groupby("TEAM_ABBR")["win"].cumsum()
    season["cum_win_pct"] = season["cum_wins"] / season["gp"]
    season["cum_pm"] = season.groupby("TEAM_ABBR")["pm"].cumsum() / season["gp"]

    season["strength"] = (season["cum_win_pct"] - 0.5) * 7.0 + (season["cum_pm"] * 0.07)
    season["strength"] = season["strength"].clip(-6, 6)

    daily = (
        season.sort_values(["GAME_DATE", "TEAM_ABBR"])
        .groupby(["GAME_DATE", "TEAM_ABBR"], as_index=False)
        .tail(1)
    )
    if daily.empty:
        return pd.DataFrame()

    def normalize_day(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        exps = np.exp(g["strength"] - g["strength"].max())
        g["implied_title_prob"] = exps / exps.sum()
        return g

    daily = daily.groupby("GAME_DATE", group_keys=False).apply(normalize_day)
    team_daily = daily[daily["TEAM_ABBR"] == team_abbr].copy().sort_values("GAME_DATE")
    if team_daily.empty:
        return pd.DataFrame()

    # Smooth for readability without distorting historical shape.
    smoothed = team_daily["implied_title_prob"].ewm(span=6, adjust=False).mean()
    team_daily["title_odds_smoothed"] = np.clip(smoothed, 0, 1)

    team_daily["title_odds_pct"] = team_daily["implied_title_prob"] * 100.0
    team_daily["title_odds_smoothed_pct"] = team_daily["title_odds_smoothed"] * 100.0
    return team_daily[["GAME_DATE", "title_odds_pct", "title_odds_smoothed_pct"]]


def get_daily_scoreboard(rs_df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Build game-level scores for a single date from regular-season team rows."""
    if rs_df.empty:
        return pd.DataFrame()

    d = pd.Timestamp(day).normalize()
    day_rows = rs_df[rs_df["GAME_DATE"].dt.normalize() == d].copy()
    if day_rows.empty:
        return pd.DataFrame()

    day_rows["is_home"] = day_rows["MATCHUP"].fillna("").str.contains("vs.")

    games: list[dict] = []
    for game_id, grp in day_rows.groupby("GAME_ID"):
        home = grp[grp["is_home"]]
        away = grp[~grp["is_home"]]

        if home.empty or away.empty:
            grp = grp.sort_values("TEAM_ABBR")
            if len(grp) < 2:
                continue
            away_row = grp.iloc[0]
            home_row = grp.iloc[1]
        else:
            home_row = home.iloc[0]
            away_row = away.iloc[0]

        games.append(
            {
                "GAME_ID": game_id,
                "GAME_DATE": d,
                "away_team": away_row["TEAM_ABBR"],
                "home_team": home_row["TEAM_ABBR"],
                "away_pts": int(away_row["PTS"]),
                "home_pts": int(home_row["PTS"]),
                "score": f"{away_row['TEAM_ABBR']} {int(away_row['PTS'])} - {int(home_row['PTS'])} {home_row['TEAM_ABBR']}",
            }
        )

    out = pd.DataFrame(games)
    if out.empty:
        return out
    return out.sort_values(["GAME_DATE", "GAME_ID"], ascending=[False, True])


@st.cache_data(ttl=60)
def load_live_scoreboard_cached() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load today's live NBA scoreboard + leaders from live endpoint."""
    try:
        from pipeline.ingestion.fetch_live_scores import get_live_scoreboard

        games_df, leaders_df = get_live_scoreboard()
        return games_df, leaders_df
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=60)
def load_game_player_stats_cached(game_id: str) -> pd.DataFrame:
    """Load full player boxscore lines for a single live-score game id."""
    if not game_id:
        return pd.DataFrame()
    try:
        from nba_api.live.nba.endpoints.boxscore import BoxScore

        bs = BoxScore(game_id)
        game = bs.nba_response.get_dict().get("game", {})
    except Exception:
        return pd.DataFrame()

    rows: list[dict] = []
    for side in ("awayTeam", "homeTeam"):
        team = game.get(side, {})
        tri = team.get("teamTricode", "")
        for p in team.get("players", []):
            stats = p.get("statistics", {}) or {}
            rows.append(
                {
                    "Team": tri,
                    "Player": p.get("name", ""),
                    "Pos": p.get("position", ""),
                    "MIN": stats.get("minutes", 0),
                    "PTS": stats.get("points", 0),
                    "REB": stats.get("reboundsTotal", 0),
                    "AST": stats.get("assists", 0),
                    "STL": stats.get("steals", 0),
                    "BLK": stats.get("blocks", 0),
                    "TOV": stats.get("turnovers", 0),
                    "+/-": stats.get("plusMinusPoints", 0),
                    "FGM": stats.get("fieldGoalsMade", 0),
                    "FGA": stats.get("fieldGoalsAttempted", 0),
                    "FG3M": stats.get("threePointersMade", 0),
                    "FG3A": stats.get("threePointersAttempted", 0),
                    "FTM": stats.get("freeThrowsMade", 0),
                    "FTA": stats.get("freeThrowsAttempted", 0),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Keep active contributors visible first.
    return df.sort_values(["Team", "PTS", "MIN"], ascending=[True, False, False]).reset_index(drop=True)


def model_outputs_last_updated() -> pd.Timestamp | None:
    candidates = [
        Path("models/trained/simulation_team_odds_current.csv"),
        Path("models/trained/series_predictions_current.csv"),
        Path("models/trained/current_season_predictions.csv"),
        Path("models/trained/play_in_simulation_results.csv"),
    ]
    latest = None
    for path in candidates:
        if path.exists():
            ts = pd.Timestamp(path.stat().st_mtime, unit="s")
            latest = ts if latest is None else max(latest, ts)
    return latest


def render_meta_chips(items: list[tuple[str, str]]) -> None:
    chips = "".join([f"<span class='meta-chip'><b>{escape(k)}:</b> {escape(v)}</span>" for k, v in items])
    st.markdown(f"<div class='meta-strip'>{chips}</div>", unsafe_allow_html=True)


def _leader_map(leaders_df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    if leaders_df.empty:
        return {}
    return leaders_df.set_index(["game_id", "team_tricode"]).to_dict("index")


def render_sidebar_live_scores(games_df: pd.DataFrame, leaders_df: pd.DataFrame) -> None:
    """Render compact grouped Live / Final / Upcoming games with logo-first clickable rows."""
    if games_df.empty:
        st.caption("No live scoreboard data returned.")
        return

    games = games_df.copy()
    leaders = _leader_map(leaders_df)
    game_lookup = games.set_index("game_id").to_dict("index")

    if "selected_sidebar_game_id" not in st.session_state:
        default = games[games["game_status"] >= 2]
        st.session_state.selected_sidebar_game_id = (
            str(default.iloc[0]["game_id"]) if not default.empty else str(games.iloc[0]["game_id"])
        )

    live = games[games["game_status"] == 2].copy()
    final = games[games["game_status"] == 3].copy()
    upcoming = games[games["game_status"] == 1].copy()

    sections = [
        ("Live", live),
        ("Final", final),
        ("Upcoming", upcoming),
    ]

    for label, df in sections:
        st.markdown(f"**{label} ({len(df)})**")
        if df.empty:
            st.caption("None")
            continue

        for row in df.itertuples(index=False):
            c1, c2, c3 = st.columns([0.55, 3.1, 0.55])
            with c1:
                st.image(logo_url(row.away_tricode), width=16)
            with c2:
                game_label = f"{row.away_tricode} {int(row.away_score)} @ {int(row.home_score)} {row.home_tricode}"
                if st.button(game_label, key=f"sidebar_game_{row.game_id}", use_container_width=True):
                    st.session_state.selected_sidebar_game_id = str(row.game_id)
                st.caption(f"{row.game_status_text} • {row.period_label} {row.clock}".strip())
            with c3:
                st.image(logo_url(row.home_tricode), width=16)
            st.markdown(
                "<div style='border-bottom:1px solid #E4E7EE; margin:0.2rem 0 0.35rem 0;'></div>",
                unsafe_allow_html=True,
            )


def render_selected_game_info(games_df: pd.DataFrame, leaders_df: pd.DataFrame) -> None:
    """Render full selected game detail panel in main content area."""
    if games_df.empty:
        return
    selected_id = st.session_state.get("selected_sidebar_game_id")
    if not selected_id:
        return

    game_lookup = games_df.set_index("game_id").to_dict("index")
    if selected_id not in game_lookup:
        return

    g = game_lookup[selected_id]
    leaders = _leader_map(leaders_df)
    away_tri = g["away_tricode"]
    home_tri = g["home_tricode"]
    away = leaders.get((selected_id, away_tri), {})
    home = leaders.get((selected_id, home_tri), {})

    with st.expander("Game Info", expanded=True):
        st.markdown(f"**{away_tri} @ {home_tri}**")
        st.caption(f"{g['game_status_text']} • {g['period_label']} {g['clock']}".strip())

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Team": away_tri,
                        "Score": int(g["away_score"]),
                        "Q1": int(g.get("away_q1", 0)),
                        "Q2": int(g.get("away_q2", 0)),
                        "Q3": int(g.get("away_q3", 0)),
                        "Q4": int(g.get("away_q4", 0)),
                        "REB": away.get("team_reb", "-"),
                        "AST": away.get("team_ast", "-"),
                        "eFG": f"{away.get('team_efg', np.nan):.1%}" if away else "-",
                        "Top Scorer": (
                            f"{away.get('top_scorer_name', '-')} {away.get('top_scorer_pts', '-')}"
                            if away else "-"
                        ),
                        "Top Reb": (
                            f"{away.get('top_rebounder_name', '-')} {away.get('top_rebounder_reb', '-')}"
                            if away else "-"
                        ),
                        "Top Ast": (
                            f"{away.get('top_assister_name', '-')} {away.get('top_assister_ast', '-')}"
                            if away else "-"
                        ),
                    },
                    {
                        "Team": home_tri,
                        "Score": int(g["home_score"]),
                        "Q1": int(g.get("home_q1", 0)),
                        "Q2": int(g.get("home_q2", 0)),
                        "Q3": int(g.get("home_q3", 0)),
                        "Q4": int(g.get("home_q4", 0)),
                        "REB": home.get("team_reb", "-"),
                        "AST": home.get("team_ast", "-"),
                        "eFG": f"{home.get('team_efg', np.nan):.1%}" if home else "-",
                        "Top Scorer": (
                            f"{home.get('top_scorer_name', '-')} {home.get('top_scorer_pts', '-')}"
                            if home else "-"
                        ),
                        "Top Reb": (
                            f"{home.get('top_rebounder_name', '-')} {home.get('top_rebounder_reb', '-')}"
                            if home else "-"
                        ),
                        "Top Ast": (
                            f"{home.get('top_assister_name', '-')} {home.get('top_assister_ast', '-')}"
                            if home else "-"
                        ),
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        hdr_col, filter_col = st.columns([1.5, 1.0])
        with hdr_col:
            st.markdown("**Player Box Score**")
        with filter_col:
            team_filter = st.selectbox(
                "Team",
                ["All", away_tri, home_tri],
                index=0,
                key=f"player_box_team_filter_{selected_id}",
                label_visibility="collapsed",
            )
        players_df = load_game_player_stats_cached(str(selected_id))
        if players_df.empty:
            st.caption("Player stats unavailable for this game.")
        else:
            if team_filter != "All":
                players_df = players_df[players_df["Team"] == team_filter].copy()
            st.dataframe(
                players_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "PTS": st.column_config.NumberColumn("PTS", format="%d"),
                    "REB": st.column_config.NumberColumn("REB", format="%d"),
                    "AST": st.column_config.NumberColumn("AST", format="%d"),
                    "STL": st.column_config.NumberColumn("STL", format="%d"),
                    "BLK": st.column_config.NumberColumn("BLK", format="%d"),
                    "TOV": st.column_config.NumberColumn("TOV", format="%d"),
                    "FGM": st.column_config.NumberColumn("FGM", format="%d"),
                    "FGA": st.column_config.NumberColumn("FGA", format="%d"),
                    "FG3M": st.column_config.NumberColumn("FG3M", format="%d"),
                    "FG3A": st.column_config.NumberColumn("FG3A", format="%d"),
                    "FTM": st.column_config.NumberColumn("FTM", format="%d"),
                    "FTA": st.column_config.NumberColumn("FTA", format="%d"),
                },
            )


def team_last10(rs_df: pd.DataFrame, team_abbr: str) -> pd.DataFrame:
    if rs_df.empty:
        return pd.DataFrame()
    team = rs_df[rs_df["TEAM_ABBR"] == team_abbr].copy().sort_values("GAME_DATE")
    if team.empty:
        return pd.DataFrame()
    return team.tail(10).copy()


def team_next10_projection(rs_df: pd.DataFrame, features_df: pd.DataFrame, team_abbr: str) -> dict:
    """Schedule-agnostic 10-game projection from current form + net rating."""
    default = {
        "games_available": 0,
        "projected_wins": 0.0,
        "projected_losses": 0.0,
        "win_prob_per_game": 0.5,
        "range_low": 0,
        "range_high": 0,
    }

    if rs_df.empty:
        return default

    team = rs_df[rs_df["TEAM_ABBR"] == team_abbr].copy()
    if team.empty:
        return default

    gp = int(len(team))
    wins = float((team["WL"] == "W").sum())
    current_wpct = wins / gp if gp > 0 else 0.5

    net = 0.0
    if not features_df.empty and team_abbr in set(features_df["TEAM_ABBR"]):
        row = features_df[features_df["TEAM_ABBR"] == team_abbr].iloc[0]
        net = float(row.get("rs_net_rating") or 0.0)

    strength_prob = _sigmoid(net * 0.11)
    p = max(0.05, min(0.95, 0.55 * current_wpct + 0.45 * strength_prob))

    remaining = max(0, 82 - gp)
    n = min(10, remaining)
    exp_w = n * p
    std = math.sqrt(max(0.0, n * p * (1 - p)))

    return {
        "games_available": n,
        "projected_wins": exp_w,
        "projected_losses": max(0.0, n - exp_w),
        "win_prob_per_game": p,
        "range_low": max(0, int(round(exp_w - std))),
        "range_high": min(n, int(round(exp_w + std))),
    }


def player_summary(player_df: pd.DataFrame, team_abbr: str) -> pd.DataFrame:
    if player_df.empty:
        return pd.DataFrame()

    team = player_df[player_df["TEAM_ABBR"] == team_abbr].copy()
    if team.empty:
        return pd.DataFrame()

    grouped = (
        team.groupby("PLAYER_NAME", as_index=False)
        .agg(
            GP=("GAME_DATE", "count"),
            MIN=("MIN", "mean"),
            PTS=("PTS", "mean"),
            REB=("REB", "mean"),
            AST=("AST", "mean"),
            STL=("STL", "mean"),
            BLK=("BLK", "mean"),
        )
        .sort_values(["PTS", "MIN"], ascending=False)
    )

    grouped = grouped[grouped["GP"] >= 10]
    return grouped.head(12)


def projected_series_games(row: pd.Series) -> int:
    """Return projected total games in series as integer in [4, 7]."""
    expected = row.get("expected_games")
    if pd.notna(expected):
        return int(max(4, min(7, round(float(expected)))))

    probs = {
        4: row.get("p_4_games"),
        5: row.get("p_5_games"),
        6: row.get("p_6_games"),
        7: row.get("p_7_games"),
    }
    probs = {k: float(v) for k, v in probs.items() if pd.notna(v)}
    if probs:
        return max(probs, key=probs.get)
    return 6


def projected_series_summary(row: pd.Series) -> dict:
    """Build winner/loser projection summary for bracket cards."""
    winner = str(row["predicted_winner"])
    high_team = str(row["high_team"])
    low_team = str(row["low_team"])
    loser = low_team if winner == high_team else high_team

    win_prob = float(row["high_team_win_prob"]) if winner == high_team else float(row["low_team_win_prob"])
    games = projected_series_games(row)
    loser_wins = max(0, min(3, games - 4))

    return {
        "winner": winner,
        "loser": loser,
        "win_prob": win_prob,
        "games": games,
        "scoreline": f"{winner} 4 - {loser} {loser_wins}",
    }


def bracket_series_card_html(row: pd.Series | None, connector: str) -> str:
    """Return HTML card for bracket shelf."""
    if row is None:
        return f"<div class='br-card placeholder {connector}'></div>"

    proj = projected_series_summary(row)
    return dedent(
        f"""
        <div class="br-card {connector}">
            <div class="br-row winner">
                <img src="{logo_url(proj['winner'])}" class="br-logo"/>
                <span>{escape(proj['winner'])}</span>
            </div>
            <div class="br-row loser">
                <img src="{logo_url(proj['loser'])}" class="br-logo small"/>
                <span>{escape(proj['loser'])}</span>
            </div>
            <div class="br-score">{escape(proj['scoreline'])}</div>
            <div class="br-odds">Win odds <span>{proj['win_prob']:.1%}</span></div>
        </div>
    """
    ).strip()


def build_round_cards(series_df: pd.DataFrame, conference: str, rnd: str, expected_n: int, connector: str) -> str:
    conf = series_df[(series_df["conference"] == conference) & (series_df["round"] == rnd)].copy()
    conf = conf.sort_values(["high_seed", "low_seed"])
    rows = [r for _, r in conf.iterrows()][:expected_n]
    while len(rows) < expected_n:
        rows.append(None)
    return "".join(bracket_series_card_html(row, connector) for row in rows)


def render_playoff_bracket_board(series_df: pd.DataFrame) -> None:
    """Render a classic playoff board style bracket with connectors."""
    rounds = {
        "First Round": 4,
        "Conference Semifinals": 2,
        "Conference Finals": 1,
    }

    west_r1 = build_round_cards(series_df, "West", "First Round", rounds["First Round"], "to-right")
    west_r2 = build_round_cards(series_df, "West", "Conference Semifinals", rounds["Conference Semifinals"], "to-right")
    west_r3 = build_round_cards(series_df, "West", "Conference Finals", rounds["Conference Finals"], "to-right")

    east_r3 = build_round_cards(series_df, "East", "Conference Finals", rounds["Conference Finals"], "to-left")
    east_r2 = build_round_cards(series_df, "East", "Conference Semifinals", rounds["Conference Semifinals"], "to-left")
    east_r1 = build_round_cards(series_df, "East", "First Round", rounds["First Round"], "to-left")

    finals_df = series_df[series_df["round"] == "NBA Finals"].copy().sort_values(["conference", "high_seed"])
    finals_row = finals_df.iloc[0] if not finals_df.empty else None
    finals_card = bracket_series_card_html(finals_row, "center")

    champion_note = ""
    if finals_row is not None:
        champ = projected_series_summary(finals_row)
        champion_note = (
            f"<div class='champ-note'>Projected Champion: {escape(champ['winner'])}</div>"
            f"<div class='champ-logo-wrap'><img src='{logo_url(champ['winner'])}' class='champ-logo' /></div>"
        )

    bracket_html = (
        '<div class="bracket-board">'
        '<div class="bracket-title-row">'
        '<div class="bracket-title west">WESTERN CONFERENCE</div>'
        '<div class="bracket-title center">NBA PLAYOFFS</div>'
        '<div class="bracket-title east">EASTERN CONFERENCE</div>'
        "</div>"
        '<div class="bracket-main">'
        '<div class="bracket-side west">'
        f'<div class="br-col r1">{west_r1}</div>'
        f'<div class="br-col r2">{west_r2}</div>'
        f'<div class="br-col r3">{west_r3}</div>'
        "</div>"
        '<div class="bracket-finals">'
        f"{finals_card}"
        f"{champion_note}"
        "</div>"
        '<div class="bracket-side east">'
        f'<div class="br-col r3">{east_r3}</div>'
        f'<div class="br-col r2">{east_r2}</div>'
        f'<div class="br-col r1">{east_r1}</div>'
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(bracket_html, unsafe_allow_html=True)


def add_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Manrope:wght@400;500;600;700&display=swap');

        :root {
            --bg: #F7F8FC;
            --surface: #FFFFFF;
            --surface2: #F1F3F9;
            --ink: #111827;
            --muted: #6B7280;
            --line: #E4E7EE;
            --brand: #2563EB;
            --accent: #059669;
        }
        .stApp {
            font-family: 'Manrope', sans-serif;
            color: var(--ink);
            background: var(--bg);
        }
        h1, h2, h3, h4 { font-family: 'Sora', sans-serif; color: var(--ink); }
        .block-container { padding-top: 1.2rem; }

        .hero {
            border: none;
            background: transparent;
            border-radius: 0;
            padding: 0.2rem 0 0.3rem 0;
            box-shadow: none;
            margin-bottom: 0.45rem;
        }
        .hero-title { font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em; color: var(--ink); }
        .hero-sub { color: var(--muted); margin-top: 0.3rem; }

        .summary-line { color: var(--ink); font-size: 1rem; font-weight: 600; }
        .summary-line .muted { color: var(--muted); font-weight: 500; }

        .section-label { color: var(--ink); font-weight: 700; margin: 0.2rem 0 0.55rem 0; }
        .meta-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.2rem 0 0.5rem 0;
        }
        .meta-chip {
            display: inline-block;
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 5px;
            color: var(--muted);
            font-size: 0.76rem;
            padding: 0.2rem 0.45rem;
            line-height: 1.2;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .champ-note {
            margin-top: 0.45rem;
            color: var(--ink);
            font-weight: 700;
            text-align: center;
            font-size: 0.92rem;
        }
        .champ-logo-wrap {
            margin-top: 0.3rem;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .champ-logo {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border: 1px solid var(--line);
            background: var(--surface);
            padding: 2px;
        }
        .bracket-board {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: var(--surface);
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
            padding: 0.9rem;
        }
        .bracket-title {
            font-family: 'Sora', sans-serif;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            color: var(--muted);
            text-align: center;
        }
        .bracket-title-row {
            display: grid;
            grid-template-columns: 1.45fr 1.2fr 1.45fr;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 0.45rem;
        }
        .bracket-title.center { color: var(--ink); }
        .bracket-main {
            display: grid;
            grid-template-columns: 1.45fr 1.2fr 1.45fr;
            gap: 0.55rem;
            align-items: start;
        }
        .bracket-side {
            width: 100%;
            display: flex;
            gap: 0.48rem;
        }
        .bracket-finals {
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding-top: 5.6rem;
            min-height: 22rem;
        }
        .br-col {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.42rem;
            position: relative;
        }
        .br-col.r2 { padding-top: 1.75rem; }
        .br-col.r3 { padding-top: 4.6rem; }
        .br-card {
            position: relative;
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 8px;
            padding: 0.34rem 0.4rem;
            min-height: 4.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .br-card.placeholder {
            opacity: 0.25;
            min-height: 4.2rem;
        }
        .br-row {
            display: flex;
            align-items: center;
            gap: 0.36rem;
            line-height: 1.1;
        }
        .br-row.winner span { color: var(--ink); font-weight: 700; font-size: 0.82rem; }
        .br-row.loser span { color: var(--muted); font-size: 0.76rem; }
        .br-logo { width: 18px; height: 18px; border-radius: 50%; }
        .br-logo.small { width: 14px; height: 14px; opacity: 0.75; }
        .br-score {
            margin-top: 0.26rem;
            color: var(--ink);
            font-size: 0.75rem;
            font-weight: 600;
        }
        .br-odds {
            color: var(--muted);
            font-size: 0.72rem;
        }
        .br-odds span { color: var(--brand); font-weight: 700; }
        .br-card.to-right::after {
            content: "";
            position: absolute;
            right: -9px;
            top: 50%;
            width: 9px;
            border-top: 1px solid var(--line);
        }
        .br-card.to-left::before {
            content: "";
            position: absolute;
            left: -9px;
            top: 50%;
            width: 9px;
            border-top: 1px solid var(--line);
        }
        @media (max-width: 1200px) {
            .bracket-title-row,
            .bracket-main {
                grid-template-columns: 1fr;
                gap: 0.4rem;
            }
            .bracket-finals {
                padding-top: 0.6rem;
                min-height: auto;
            }
            .br-col.r2, .br-col.r3 { padding-top: 0; }
            .br-card.to-right::after, .br-card.to-left::before { display: none; }
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1.15rem;
            border-bottom: 1px solid var(--line);
            padding-bottom: 0.2rem;
        }
        .stTabs [data-baseweb="tab"] {
            border: none;
            border-radius: 0;
            background: transparent;
            color: var(--muted);
            font-weight: 600;
            padding: 0.25rem 0.1rem 0.4rem 0.1rem;
            min-height: auto;
        }
        .stTabs [aria-selected="true"] {
            background: transparent !important;
            color: var(--ink) !important;
            border-bottom: 2px solid var(--brand) !important;
        }

        [data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--line);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="NBA Playoff Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)
add_theme()

tables = load_base_tables()
title_df = tables.get("title", pd.DataFrame())
series_df = tables.get("series", pd.DataFrame())
play_in_df = tables.get("play_in", pd.DataFrame())
teams_df = tables.get("teams", pd.DataFrame())
rs_df = tables.get("rs", pd.DataFrame())
player_rs_df = tables.get("player_rs", pd.DataFrame())
current_preds_df = tables.get("current_preds", pd.DataFrame())
features_df = tables.get("features", pd.DataFrame())
remaining_games_df = tables.get("remaining_games", pd.DataFrame())
projected_records_df = tables.get("projected_records", pd.DataFrame())

st.markdown(
    f"""
    <div class='hero'>
      <div class='hero-title'>NBA Playoff Predictor Dashboard</div>
      <div class='hero-sub'>Analytics dashboard for {CURRENT_SEASON_STR}: play-in race, playoff prediction, and full team-level scouting.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

rs_last_dt = rs_df["GAME_DATE"].max() if (not rs_df.empty and "GAME_DATE" in rs_df.columns) else None
model_last_dt = model_outputs_last_updated()
live_games_df, live_leaders_df = load_live_scoreboard_cached()
live_status = "live feed" if not live_games_df.empty else "fallback (local db)"

render_meta_chips(
    [
        ("Season", CURRENT_SEASON_STR),
        ("Model outputs", _fmt_dt(model_last_dt)),
        ("Games snapshot", _fmt_dt(rs_last_dt)),
        ("Score source", live_status),
    ]
)

render_selected_game_info(live_games_df, live_leaders_df)

with st.sidebar:
    st.header("Live / Recent Scores")
    st.caption(f"Season: {CURRENT_SEASON_STR}")
    if not live_games_df.empty:
        render_sidebar_live_scores(live_games_df, live_leaders_df)
    else:
        st.caption("Live endpoint unavailable. Showing local recent finals.")
        if rs_df.empty or rs_df["GAME_DATE"].isna().all():
            st.caption("No regular-season game logs available.")
        else:
            today_date = pd.Timestamp.now().normalize()
            today_scores = get_daily_scoreboard(rs_df, today_date)
            if today_scores.empty:
                latest_date = rs_df["GAME_DATE"].max().normalize()
                final_scores = get_daily_scoreboard(rs_df, latest_date)
                st.caption(f"No games today ({today_date.strftime('%Y-%m-%d')}). Showing {latest_date.strftime('%Y-%m-%d')}.")
            else:
                final_scores = today_scores
                st.caption("Today's completed games in local data:")
            for row in final_scores.head(12).itertuples(index=False):
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:0.35rem;margin:0.15rem 0 0.35rem 0;">
                        <img src="{logo_url(row.away_team)}" width="16" height="16" style="border-radius:50%;" />
                        <span style="font-weight:700;">{row.away_team} {row.away_pts} - {row.home_pts} {row.home_team}</span>
                        <img src="{logo_url(row.home_team)}" width="16" height="16" style="border-radius:50%;" />
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

playin_tab, playoff_tab, team_tab, analyst_tab = st.tabs(
    ["Play-In Predictor / Race", "Playoff Predictor", "Team-by-Team Breakdown", "🤖 AI Analyst"]
)

with playin_tab:
    st.markdown("<div class='section-label'>Play-In Race & Standings Bubble</div>", unsafe_allow_html=True)
    render_meta_chips(
        [
            ("Metric", "Make playoffs probability"),
            ("Source", "app_play_in_current + current_season_predictions"),
            ("Interpretation", "Seeds 5-12 bubble context + play-in odds"),
        ]
    )
    if play_in_df.empty:
        st.info("`app_play_in_current` is missing. Run simulation pipeline first.")
    else:
        p = play_in_df.copy()
        p["projected_seed"] = p.apply(
            lambda r: 7 if float(r["seed7_prob"]) >= float(r["seed8_prob"]) else 8,
            axis=1,
        )
        # Build a lookup of play-in probs by team
        playin_lookup = {row.team_abbr: row for row in p.itertuples(index=False)}

        # Hot streak: last 10 W-L per team from regular_season
        def _hot_streak(team: str) -> str:
            if rs_df.empty:
                return ""
            t = rs_df[rs_df["TEAM_ABBR"] == team].sort_values("GAME_DATE")
            if t.empty:
                return ""
            last10 = t.tail(10)
            w = int((last10["WL"] == "W").sum())
            l = int((last10["WL"] == "L").sum())
            color = "#059669" if w >= 7 else ("#DC2626" if w <= 3 else "#6B7280")
            return f'<span style="color:{color};font-weight:700;font-size:0.82rem;">{w}-{l} L10</span>'

        conf_left, conf_right = st.columns(2)
        for conf_name, mount in [("East", conf_left), ("West", conf_right)]:
            with mount:
                st.markdown(f"### {conf_name}")

                # ── Standings bubble (seeds 5-12) ──────────────────────────
                if not current_preds_df.empty:
                    conf_preds = (
                        current_preds_df[current_preds_df["conference"] == conf_name]
                        .copy()
                        .sort_values("playoff_rank")
                    )
                    # seeds 5-12 by conference standing rank
                    bubble = conf_preds[
                        (conf_preds["playoff_rank"] >= 5) & (conf_preds["playoff_rank"] <= 12)
                    ].copy()

                    if not bubble.empty:
                        # Max wins for GB calculation
                        max_wins = int(bubble.iloc[0]["wins"])
                        bubble["gb"] = (max_wins - bubble["wins"]) / 1.0

                        # Build projected-record lookup for this conference
                        proj_lookup: dict = {}
                        if not projected_records_df.empty:
                            conf_proj = projected_records_df[
                                projected_records_df["CONFERENCE"] == conf_name
                            ]
                            for _, pr in conf_proj.iterrows():
                                proj_lookup[pr["TEAM_ABBR"]] = pr

                        st.markdown("**Bubble Standings (Seeds 5–12)**")
                        rows_html = ""
                        for br in bubble.itertuples(index=False):
                            seed = int(br.playoff_rank)
                            abbr = br.TEAM_ABBR
                            record = f"{int(br.wins)}-{int(br.losses)}"
                            gb = float(br.gb)
                            gb_str = "—" if gb == 0 else f"{gb:.1f}"

                            # Projected final record from Monte Carlo
                            proj = proj_lookup.get(abbr)
                            proj_str = ""
                            seed_risk_badge = ""
                            if proj is not None:
                                ef_w = proj["expected_final_wins"]
                                p10_w = proj["p10_final_wins"]
                                p90_w = proj["p90_final_wins"]
                                proj_str = f'<span style="color:#6B7280;font-size:0.78rem;margin-left:0.3rem;">proj {ef_w:.0f}W ({p10_w}–{p90_w})</span>'
                                # Seed-specific risk/opportunity badge
                                if seed == 6:
                                    fall_p = proj["prob_make_playin"]
                                    if fall_p > 0.15:
                                        seed_risk_badge = f'<span style="background:#FEF3C7;color:#92400E;border-radius:4px;padding:1px 5px;font-size:0.72rem;">FALL RISK {fall_p:.0%}</span>'
                                elif seed == 11:
                                    climb_p = proj["prob_make_playin"]
                                    if climb_p > 0.10:
                                        seed_risk_badge = f'<span style="background:#D1FAE5;color:#065F46;border-radius:4px;padding:1px 5px;font-size:0.72rem;">CLIMB {climb_p:.0%}</span>'

                            # Zone badge
                            if seed <= 6:
                                zone = '<span style="background:#D1FAE5;color:#065F46;border-radius:4px;padding:1px 6px;font-size:0.75rem;font-weight:700;">AUTO</span>'
                                # flag if gap to seed 7 is small
                                seed7_wins = bubble[bubble["playoff_rank"] == 7]["wins"].values
                                if len(seed7_wins) and (int(br.wins) - int(seed7_wins[0])) <= 2:
                                    zone += ' <span style="background:#FEF3C7;color:#92400E;border-radius:4px;padding:1px 5px;font-size:0.72rem;">WATCH</span>'
                            elif seed <= 10:
                                zone = '<span style="background:#DBEAFE;color:#1E40AF;border-radius:4px;padding:1px 6px;font-size:0.75rem;font-weight:700;">PLAY-IN</span>'
                                if proj is not None:
                                    zone += f' <span style="color:#6B7280;font-size:0.78rem;">{proj["prob_make_playin"]:.0%} in</span>'
                            else:
                                zone = '<span style="background:#F3F4F6;color:#6B7280;border-radius:4px;padding:1px 6px;font-size:0.75rem;font-weight:700;">OUT</span>'

                            hot = _hot_streak(abbr)
                            rows_html += (
                                '<div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;border-bottom:1px solid #F3F4F6;flex-wrap:wrap;">'
                                f'<span style="color:#9CA3AF;font-size:0.8rem;width:1.4rem;text-align:right;">{seed}</span>'
                                f'<img src="{logo_url(abbr)}" width="20" height="20" style="border-radius:50%;" />'
                                f'<span style="font-weight:700;font-size:0.9rem;width:2.5rem;">{abbr}</span>'
                                f'<span style="color:#374151;font-size:0.85rem;width:4rem;">{record}</span>'
                                f'{proj_str}'
                                f'<span style="color:#9CA3AF;font-size:0.8rem;width:3rem;">{gb_str} GB</span>'
                                f'{zone}'
                                f'{seed_risk_badge}'
                                f'<span style="margin-left:auto;">{hot}</span>'
                                '</div>'
                            )
                        st.markdown(
                            f'<div style="background:#FAFAFA;border:1px solid #E4E7EE;border-radius:8px;padding:0.5rem 0.75rem;margin-bottom:1rem;">{rows_html}</div>',
                            unsafe_allow_html=True,
                        )

                # ── Play-in cards (seeds 7-10) ──────────────────────────
                st.markdown("**Play-In Teams**")
                conf_playin = p[p["conference"] == conf_name].sort_values(
                    "made_playoffs_prob", ascending=False
                )
                if conf_playin.empty:
                    st.caption("No teams found.")
                    continue

                for row in conf_playin.itertuples(index=False):
                    hot = _hot_streak(row.team_abbr)
                    # Seed from current_preds if available
                    seed_num = ""
                    if not current_preds_df.empty:
                        match = current_preds_df[current_preds_df["TEAM_ABBR"] == row.team_abbr]
                        if not match.empty:
                            raw_seed = match.iloc[0]['playoff_seed']
                            if raw_seed == raw_seed:  # not NaN
                                seed_num = f"#{int(raw_seed)} · "
                    st.markdown(
                        f"""
                        <div style="
                            border:1px solid #E4E7EE;
                            background:#FFFFFF;
                            border-radius:8px;
                            padding:0.55rem 0.7rem;
                            margin-bottom:0.5rem;
                            display:flex;
                            align-items:center;
                            justify-content:space-between;
                            gap:0.6rem;
                            box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                            <div style="display:flex;align-items:center;gap:0.55rem;">
                                <img src="{logo_url(row.team_abbr)}" width="30" height="30" style="border-radius:50%;" />
                                <div>
                                    <div style="font-weight:700;color:#111827;">{seed_num}{row.team_abbr}</div>
                                    <div style="font-size:0.78rem;margin-top:2px;">{hot}</div>
                                </div>
                            </div>
                            <div style="text-align:center;">
                                <div style="color:#6B7280;font-size:0.75rem;">Seed 7</div>
                                <div style="font-weight:700;color:#111827;">{float(row.seed7_prob):.1%}</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="color:#6B7280;font-size:0.75rem;">Seed 8</div>
                                <div style="font-weight:700;color:#111827;">{float(row.seed8_prob):.1%}</div>
                            </div>
                            <div style="text-align:center;">
                                <div style="color:#6B7280;font-size:0.75rem;">Make Playoffs</div>
                                <div style="font-weight:700;color:#059669;">{float(row.made_playoffs_prob):.1%}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

with playoff_tab:
    st.markdown("<div class='section-label'>Projected Playoff Bracket</div>", unsafe_allow_html=True)
    render_meta_chips(
        [
            ("Metric", "Series winner probability"),
            ("Source", "app_series_predictions_current"),
            ("Interpretation", "Projected winners by bracket round"),
        ]
    )
    if series_df.empty:
        st.info("`app_series_predictions_current` is missing.")
    else:
        render_playoff_bracket_board(series_df)

        st.markdown("---")
        st.markdown("<div class='section-label'>Series Detail</div>", unsafe_allow_html=True)

        round_order = ["First Round", "Conference Semifinals", "Conference Finals", "NBA Finals"]
        def _series_label(r: pd.Series) -> str:
            conf = r["conference"] if r["round"] != "NBA Finals" else "Finals"
            rnd = r["round"].replace("Conference ", "")
            return f"{conf} {rnd}: {r['high_team']} vs {r['low_team']}"

        sorted_series = series_df.copy()
        sorted_series["_ro"] = sorted_series["round"].map(lambda x: round_order.index(x) if x in round_order else 9)
        sorted_series = sorted_series.sort_values(["_ro", "conference", "high_seed"])
        series_labels = [_series_label(r) for _, r in sorted_series.iterrows()]
        series_keys = [f"{r['round']}|{r['conference']}|{r['high_team']}" for _, r in sorted_series.iterrows()]

        selected_label = st.selectbox("Select a series to inspect:", series_labels, key="selected_series")
        if selected_label:
            sel_idx = series_labels.index(selected_label)
            sel_key = series_keys[sel_idx]
            sel_row = sorted_series.iloc[sel_idx]

            high_team = str(sel_row["high_team"])
            low_team = str(sel_row["low_team"])
            high_prob = float(sel_row["high_team_win_prob"])
            low_prob = float(sel_row["low_team_win_prob"])
            winner = str(sel_row["predicted_winner"])
            loser = low_team if winner == high_team else high_team
            win_prob = high_prob if winner == high_team else low_prob

            # Header: logos + names
            col_h, col_vs, col_l = st.columns([2, 1, 2])
            with col_h:
                st.image(logo_url(high_team), width=72)
                st.markdown(f"**{high_team}** (Seed {int(sel_row['high_seed'])})")
                st.markdown(f"Win prob: **{high_prob:.1%}**")
            with col_vs:
                st.markdown("<div style='text-align:center;font-size:1.4rem;font-weight:700;padding-top:1.5rem;'>vs</div>", unsafe_allow_html=True)
            with col_l:
                st.image(logo_url(low_team), width=72)
                st.markdown(f"**{low_team}** (Seed {int(sel_row['low_seed'])})")
                st.markdown(f"Win prob: **{low_prob:.1%}**")

            # Win probability bar
            st.markdown(f"**Projected winner: {winner}** ({win_prob:.1%})")
            st.progress(win_prob if winner == high_team else 1 - win_prob)

            # Series length distribution
            p_cols = ["p_4_games", "p_5_games", "p_6_games", "p_7_games"]
            if all(c in sel_row.index for c in p_cols):
                length_vals = [float(sel_row[c]) for c in p_cols]
                length_labels = ["4 Games", "5 Games", "6 Games", "7 Games"]
                fig_len = go.Figure(go.Bar(
                    x=length_labels,
                    y=[v * 100 for v in length_vals],
                    marker_color=[TEAM_COLORS.get(winner, "#35a7ff")] * 4,
                    text=[f"{v:.1%}" for v in length_vals],
                    textposition="outside",
                ))
                expected = float(sel_row.get("expected_games", 0) or 0)
                fig_len.update_layout(
                    title=f"Series Length Distribution (Expected: {expected:.1f} games)",
                    yaxis_title="Probability (%)",
                    height=280,
                    margin=dict(t=40, b=20, l=20, r=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_len, use_container_width=True, key=f"len_chart_{sel_key}")

            # Model features comparison
            if not features_df.empty:
                feat_cols = [c for c in ["rs_net_rating", "rs_vs_top_teams_win_pct", "rs_off_rating", "rs_def_rating"] if c in features_df.columns]
                feat_labels = {
                    "rs_net_rating": "Net Rating",
                    "rs_vs_top_teams_win_pct": "vs Top Teams W%",
                    "rs_off_rating": "Off Rating",
                    "rs_def_rating": "Def Rating",
                }
                high_feat = features_df[features_df["TEAM_ABBR"] == high_team]
                low_feat = features_df[features_df["TEAM_ABBR"] == low_team]
                if not high_feat.empty or not low_feat.empty:
                    st.markdown("**Model Feature Comparison**")
                    feat_header = st.columns([2, 1, 1])
                    feat_header[0].markdown("**Feature**")
                    feat_header[1].markdown(f"**{high_team}**")
                    feat_header[2].markdown(f"**{low_team}**")
                    for fc in feat_cols:
                        hv = float(high_feat.iloc[0][fc]) if not high_feat.empty else float("nan")
                        lv = float(low_feat.iloc[0][fc]) if not low_feat.empty else float("nan")
                        row_cols = st.columns([2, 1, 1])
                        row_cols[0].markdown(feat_labels.get(fc, fc))
                        h_bold = "**" if hv >= lv else ""
                        l_bold = "**" if lv >= hv else ""
                        fmt = ".3f" if "win_pct" in fc else ".1f"
                        row_cols[1].markdown(f"{h_bold}{hv:{fmt}}{h_bold}" if not pd.isna(hv) else "—")
                        row_cols[2].markdown(f"{l_bold}{lv:{fmt}}{l_bold}" if not pd.isna(lv) else "—")

with team_tab:
    st.markdown("<div class='section-label'>All 30 Teams Drilldown</div>", unsafe_allow_html=True)

    all_teams = teams_df["TEAM_ABBR"].dropna().tolist() if not teams_df.empty else []
    if not all_teams and not rs_df.empty:
        all_teams = sorted(rs_df["TEAM_ABBR"].dropna().unique().tolist())

    if not all_teams:
        st.info("No team universe found in local DB.")
    else:
        default_team = "BOS" if "BOS" in all_teams else all_teams[0]
        team_abbr = st.selectbox("Team", all_teams, index=all_teams.index(default_team))

        row_a, row_b, row_c = st.columns(3)
        with row_a:
            st.image(logo_url(team_abbr), width=72)
        with row_b:
            team_title = title_df[title_df["TEAM_ABBR"] == team_abbr] if not title_df.empty else pd.DataFrame()
            team_rank = (
                int(team_title.index[0] + 1)
                if not team_title.empty
                else None
            )
            st.metric("Title Odds Rank", value=str(team_rank) if team_rank else "-")
        with row_c:
            if not team_title.empty:
                st.metric("Current Title Odds", value=pct(float(team_title.iloc[0]["title_prob"])))
            else:
                st.metric("Current Title Odds", value="-")

        # Round-by-round odds row
        if not team_title.empty:
            r2, rcf, rf, rt = st.columns(4)
            tr = team_title.iloc[0]
            r2.metric("Make Round 2", pct(float(tr.get("make_second_round_prob", 0) or 0)))
            rcf.metric("Conf Finals", pct(float(tr.get("make_conf_finals_prob", 0) or 0)))
            rf.metric("Finals", pct(float(tr.get("make_finals_prob", 0) or 0)))
            rt.metric("Title", pct(float(tr.get("title_prob", 0) or 0)))

        c1, c2 = st.columns([1.45, 1.0])
        with c1:
            st.markdown("<div class='section-label'>Odds Over Time (Current Season)</div>", unsafe_allow_html=True)
            render_meta_chips(
                [
                    ("Metric", "Implied title odds path"),
                    ("Source", "regular_season (win% + net rating softmax)"),
                    ("Smoothing", "EWM span=6"),
                ]
            )
            traj = build_team_title_odds_series(rs_df, title_df, team_abbr)
            line_color = TEAM_COLORS.get(team_abbr, "#35a7ff")

            fig = go.Figure()
            if not traj.empty:
                fig.add_trace(
                    go.Scatter(
                        x=traj["GAME_DATE"],
                        y=traj["title_odds_smoothed_pct"],
                        mode="lines",
                        name=f"{team_abbr} title odds",
                        line=dict(color=line_color, width=3, shape="spline", smoothing=1.0),
                    )
                )

            fig.update_layout(
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#111827",
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                yaxis=dict(title="Title Odds (%)", color=line_color, ticksuffix="%", gridcolor="#E4E7EE"),
                xaxis=dict(title="Date", gridcolor="#E4E7EE"),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Implied title odds based on RS win% + net rating softmax vs all 30 teams on each game date. Reflects actual relative strength at each point in the season."
            )

        with c2:
            st.markdown("<div class='section-label'>Last 10 + Next 10</div>", unsafe_allow_html=True)
            last10 = team_last10(rs_df, team_abbr)
            if last10.empty:
                st.info("No regular-season rows available for selected team.")
            else:
                w10 = int((last10["WL"] == "W").sum())
                l10 = int((last10["WL"] == "L").sum())
                st.metric("Last 10", value=f"{w10}-{l10}")
                st.dataframe(
                    last10[["GAME_DATE", "MATCHUP", "WL", "PTS", "REB", "AST", "PLUS_MINUS"]]
                    .sort_values("GAME_DATE", ascending=False)
                    .assign(GAME_DATE=lambda d: d["GAME_DATE"].dt.strftime("%Y-%m-%d")),
                    width="stretch",
                    hide_index=True,
                )

            # ── Upcoming games + projected final record ──────────────────
            team_upcoming = (
                remaining_games_df[remaining_games_df["TEAM_ABBR"] == team_abbr]
                .sort_values("GAME_DATE")
                .head(10)
                if not remaining_games_df.empty else pd.DataFrame()
            )

            team_proj_rec = (
                projected_records_df[projected_records_df["TEAM_ABBR"] == team_abbr].iloc[0]
                if not projected_records_df.empty and team_abbr in projected_records_df["TEAM_ABBR"].values
                else None
            )

            if team_proj_rec is not None:
                ef_w = team_proj_rec["expected_final_wins"]
                ef_l = team_proj_rec["expected_final_losses"]
                p10 = team_proj_rec["p10_final_wins"]
                p90 = team_proj_rec["p90_final_wins"]
                g_rem = int(team_proj_rec["games_remaining"])
                st.metric(
                    "Projected Final Record",
                    value=f"{ef_w:.0f}W – {ef_l:.0f}L",
                )
                st.caption(
                    f"Range: {p10}–{p90} wins | {g_rem} games remaining (5,000-sim Monte Carlo)"
                )

            if not team_upcoming.empty:
                st.markdown("**Upcoming Games**")
                up_display = team_upcoming.copy()
                up_display["Date"] = up_display["GAME_DATE"].apply(
                    lambda d: d.strftime("%b %d") if hasattr(d, "strftime") else str(d)
                )
                up_display["H/A"] = up_display["IS_HOME"].map({True: "vs", False: "@"})
                up_display["Opp Net Rtg"] = up_display["OPP_NET_RATING"].map("{:+.1f}".format)
                up_display["Win Prob"] = up_display["GAME_WIN_PROB"].map("{:.0%}".format)
                st.dataframe(
                    up_display[["Date", "H/A", "OPP_ABBR", "Opp Net Rtg", "Win Prob"]].rename(
                        columns={"OPP_ABBR": "Opp"}
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                # Fallback to schedule-agnostic projection when schedule not in DB
                proj = team_next10_projection(rs_df, features_df, team_abbr)
                n_available = proj["games_available"]
                st.metric(
                    f"Next {n_available} Projected W-L",
                    value=f"{proj['projected_wins']:.1f}W – {proj['projected_losses']:.1f}L",
                )
                st.caption(
                    f"Range: {proj['range_low']}–{proj['range_high']} wins. "
                    "Based on team strength vs league average — schedule not yet in DB."
                )

        st.markdown("<div class='section-label'>Player Stats (Current Season)</div>", unsafe_allow_html=True)
        players = player_summary(player_rs_df, team_abbr)
        if players.empty:
            st.info("No player-level data for selected team.")
        else:
            st.dataframe(
                players,
                width="stretch",
                hide_index=True,
                column_config={
                    "MIN": st.column_config.NumberColumn("MIN", format="%.1f"),
                    "PTS": st.column_config.NumberColumn("PTS", format="%.1f"),
                    "REB": st.column_config.NumberColumn("REB", format="%.1f"),
                    "AST": st.column_config.NumberColumn("AST", format="%.1f"),
                    "STL": st.column_config.NumberColumn("STL", format="%.1f"),
                    "BLK": st.column_config.NumberColumn("BLK", format="%.1f"),
                },
            )

with analyst_tab:
    from config.settings import ANTHROPIC_API_KEY, CURRENT_SEASON_STR as _cur_season

    if not ANTHROPIC_API_KEY:
        st.warning(
            "AI Analyst requires an **ANTHROPIC_API_KEY**. "
            "Add it under Space Settings → Variables and secrets."
        )
    else:
        try:
            from pipeline.agent.analyst import answer_question, get_team_scouting_report
            _analyst_ok = True
        except Exception as _e:
            st.error(f"Could not load analyst module: {_e}")
            _analyst_ok = False

        if _analyst_ok:
            # Session state
            if "analyst_messages" not in st.session_state:
                st.session_state.analyst_messages = []
            if "analyst_history" not in st.session_state:
                st.session_state.analyst_history = []

            # Build model context from loaded tables
            _top3 = []
            if not title_df.empty:
                _top3 = (
                    title_df.head(3)
                    .apply(lambda r: f"{r['TEAM_ABBR']} ({float(r['title_prob']):.1%})", axis=1)
                    .tolist()
                )
            _model_context = {
                "physicality_weight": 1.0,
                "top_3": ", ".join(_top3) if _top3 else "N/A",
                "most_physical": "N/A",
                "pace_teams": "N/A",
                "season": _cur_season,
            }

            st.markdown(
                "<div class='section-label'>Ask about predictions, teams, or the model</div>",
                unsafe_allow_html=True,
            )

            # Quick scouting report
            if not title_df.empty:
                _col1, _col2 = st.columns([2, 1])
                with _col1:
                    _scout_team = st.selectbox(
                        "Quick scouting report",
                        title_df["TEAM_ABBR"].tolist(),
                        key="analyst_scout_team",
                        label_visibility="collapsed",
                    )
                with _col2:
                    if st.button("Generate report", key="analyst_report_btn", use_container_width=True):
                        _row = title_df[title_df["TEAM_ABBR"] == _scout_team].iloc[0]
                        _prompt_text = f"Give me a playoff scouting report for {_scout_team}."
                        with st.spinner("Generating scouting report…"):
                            try:
                                _report = get_team_scouting_report(
                                    _scout_team,
                                    {},
                                    float(_row["title_prob"]),
                                )
                                st.session_state.analyst_messages.append(
                                    {"role": "user", "content": _prompt_text}
                                )
                                st.session_state.analyst_messages.append(
                                    {"role": "assistant", "content": _report}
                                )
                                st.session_state.analyst_history.append(
                                    {"role": "user", "content": _prompt_text}
                                )
                                st.session_state.analyst_history.append(
                                    {"role": "assistant", "content": _report}
                                )
                                st.rerun()
                            except Exception as _err:
                                st.error(f"Analyst error: {_err}")

            st.divider()

            # Chat history
            for _msg in st.session_state.analyst_messages:
                with st.chat_message(_msg["role"]):
                    st.markdown(_msg["content"])

            # Chat input
            if _user_input := st.chat_input("Ask about teams, model features, predictions…"):
                st.session_state.analyst_messages.append(
                    {"role": "user", "content": _user_input}
                )
                with st.chat_message("user"):
                    st.markdown(_user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            _answer, _updated_hist = answer_question(
                                _user_input,
                                _model_context,
                                st.session_state.analyst_history,
                            )
                            st.markdown(_answer)
                            st.session_state.analyst_messages.append(
                                {"role": "assistant", "content": _answer}
                            )
                            st.session_state.analyst_history = _updated_hist
                        except Exception as _err:
                            _err_msg = f"Error: {_err}"
                            st.error(_err_msg)
                            st.session_state.analyst_messages.append(
                                {"role": "assistant", "content": _err_msg}
                            )
