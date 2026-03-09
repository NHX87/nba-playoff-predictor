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


def _playin_game_card(
    seed_a: int,
    abbr_a: str,
    score_a: float,
    record_a: str,
    seed_b: int,
    abbr_b: str,
    score_b: float,
    record_b: str,
    outcome_label: str,
    home_court_team: str,  # abbr of team with home court
) -> str:
    """Build a single-line HTML string for one play-in game matchup card."""
    HOME_COURT = 0.10
    # Higher seed (lower number) gets home court
    p_a = _sigmoid((score_a - score_b) + (HOME_COURT if seed_a < seed_b else -HOME_COURT))
    p_b = 1.0 - p_a
    color_a = "#059669" if p_a > 0.55 else ("#DC2626" if p_a < 0.45 else "#374151")
    color_b = "#059669" if p_b > 0.55 else ("#DC2626" if p_b < 0.45 else "#374151")
    bar_a = int(p_a * 100)
    bar_b = 100 - bar_a
    return (
        '<div style="border:1px solid #E4E7EE;border-radius:8px;padding:0.5rem 0.7rem;margin-bottom:0.5rem;background:#FFFFFF;">'
        f'<div style="font-size:0.7rem;color:#9CA3AF;font-weight:600;letter-spacing:0.05em;margin-bottom:0.35rem;">{outcome_label}</div>'
        '<div style="display:flex;align-items:center;gap:0.4rem;">'
        f'<img src="{logo_url(abbr_a)}" width="22" height="22" style="border-radius:50%;" />'
        f'<span style="font-weight:700;font-size:0.88rem;width:2.4rem;">#{seed_a} {abbr_a}</span>'
        f'<span style="color:#6B7280;font-size:0.8rem;width:3.5rem;">{record_a}</span>'
        f'<span style="font-weight:700;font-size:0.88rem;color:{color_a};margin-left:auto;">{p_a:.0%}</span>'
        '<span style="color:#9CA3AF;font-size:0.8rem;padding:0 0.3rem;">vs</span>'
        f'<span style="font-weight:700;font-size:0.88rem;color:{color_b};">{p_b:.0%}</span>'
        f'<span style="color:#6B7280;font-size:0.8rem;width:3.5rem;text-align:right;">{record_b}</span>'
        f'<span style="font-weight:700;font-size:0.88rem;width:2.4rem;text-align:right;">#{seed_b} {abbr_b}</span>'
        f'<img src="{logo_url(abbr_b)}" width="22" height="22" style="border-radius:50%;" />'
        '</div>'
        f'<div style="display:flex;gap:0;border-radius:4px;overflow:hidden;margin-top:0.3rem;height:5px;">'
        f'<div style="width:{bar_a}%;background:{color_a};"></div>'
        f'<div style="width:{bar_b}%;background:{color_b};"></div>'
        '</div>'
        '</div>'
    )


def _render_playin_bracket(
    conf_name: str,
    current_preds_df: pd.DataFrame,
    play_in_df: pd.DataFrame,
) -> None:
    """Render the 3-game play-in tournament bracket for one conference."""
    st.markdown("**Play-In Bracket**")
    if current_preds_df.empty:
        st.caption("No standings data.")
        return

    conf_preds = (
        current_preds_df[current_preds_df["conference"] == conf_name]
        .copy()
        .sort_values("playoff_rank")
    )

    def _get(rank: int) -> tuple[str, float, str]:
        row = conf_preds[conf_preds["playoff_rank"] == rank]
        if row.empty:
            return ("?", 0.0, "0-0")
        r = row.iloc[0]
        abbr = str(r["TEAM_ABBR"])
        score = float(r["pred_survival_score"]) if not pd.isna(r["pred_survival_score"]) else 0.0
        record = f"{int(r['wins'])}-{int(r['losses'])}"
        return (abbr, score, record)

    a7, s7, r7 = _get(7)
    a8, s8, r8 = _get(8)
    a9, s9, r9 = _get(9)
    a10, s10, r10 = _get(10)

    # Simulate probabilities for Game 3 from play-in data
    pi = play_in_df[play_in_df["conference"] == conf_name] if not play_in_df.empty else pd.DataFrame()

    def _made_prob(abbr: str) -> str:
        if pi.empty:
            return ""
        row = pi[pi["team_abbr"] == abbr]
        if row.empty:
            return ""
        p = float(row.iloc[0]["made_playoffs_prob"])
        c = "#059669" if p > 0.55 else ("#DC2626" if p < 0.35 else "#92400E")
        return f'<span style="font-size:0.72rem;color:{c};font-weight:600;">{p:.0%} make playoffs</span>'

    # Game 1: seed 7 vs 8, home court to seed 7
    st.markdown(
        _playin_game_card(7, a7, s7, r7, 8, a8, s8, r8, "GAME 1 · Winner → Seed 7 · Loser plays again", a7),
        unsafe_allow_html=True,
    )
    # Game 2: seed 9 vs 10, home court to seed 9
    st.markdown(
        _playin_game_card(9, a9, s9, r9, 10, a10, s10, r10, "GAME 2 · Winner advances · Loser eliminated", a9),
        unsafe_allow_html=True,
    )
    # Game 3: likely Seed 8 vs likely Seed 9 (loser G1 vs winner G2)
    g3_a_abbr, g3_a_score, g3_a_rec = a8, s8, r8  # likely loser of G1
    g3_b_abbr, g3_b_score, g3_b_rec = a9, s9, r9  # likely winner of G2
    g3_card = _playin_game_card(8, g3_a_abbr, g3_a_score, g3_a_rec, 9, g3_b_abbr, g3_b_score, g3_b_rec, "GAME 3 · Winner → Seed 8 · Loser eliminated", g3_a_abbr)
    # Append made-playoffs probability annotation
    prob_line = _made_prob(g3_a_abbr) + ('&nbsp;&nbsp;' if _made_prob(g3_a_abbr) and _made_prob(g3_b_abbr) else '') + _made_prob(g3_b_abbr)
    if prob_line.strip():
        g3_card = g3_card[:-6] + f'<div style="margin-top:0.25rem;display:flex;justify-content:space-between;">{prob_line}</div>' + '</div>'
    st.markdown(g3_card, unsafe_allow_html=True)


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
def fetch_live_standings() -> pd.DataFrame:
    """Fetch current W/L from stats.nba.com (cached 5 min). Returns empty df on failure."""
    try:
        from nba_api.stats.endpoints import leaguestandingsv3
        raw = leaguestandingsv3.LeagueStandingsV3(
            season=CURRENT_SEASON_STR, timeout=30
        ).get_data_frames()[0]
        df = raw[["TeamSlug", "Conference", "WINS", "LOSSES", "WinPCT", "Record"]].copy()
        # Map TeamSlug to TEAM_ABBR via a quick lookup
        slug_to_abbr = {
            "hawks": "ATL", "celtics": "BOS", "nets": "BKN", "hornets": "CHA", "bulls": "CHI",
            "cavaliers": "CLE", "mavericks": "DAL", "nuggets": "DEN", "pistons": "DET", "warriors": "GSW",
            "rockets": "HOU", "pacers": "IND", "clippers": "LAC", "lakers": "LAL", "grizzlies": "MEM",
            "heat": "MIA", "bucks": "MIL", "timberwolves": "MIN", "pelicans": "NOP", "knicks": "NYK",
            "thunder": "OKC", "magic": "ORL", "76ers": "PHI", "suns": "PHX", "trail-blazers": "POR",
            "kings": "SAC", "spurs": "SAS", "raptors": "TOR", "jazz": "UTA", "wizards": "WAS",
        }
        df["TEAM_ABBR"] = df["TeamSlug"].map(slug_to_abbr)
        df = df.rename(columns={"WINS": "wins", "LOSSES": "losses", "WinPCT": "win_pct", "Conference": "conference"})
        df["conference"] = df["conference"].str.title()
        return df[["TEAM_ABBR", "conference", "wins", "losses", "win_pct", "Record"]].dropna(subset=["TEAM_ABBR"])
    except Exception:
        return pd.DataFrame()


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
                   p_4_games, p_5_games, p_6_games, p_7_games, expected_games, most_likely_games
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
                   playoff_rank, playoff_seed, title_prob_proxy_all_30, pred_survival_score
            FROM current_season_predictions
            WHERE SEASON = '{CURRENT_SEASON_STR}'
            ORDER BY pred_rank_all_30
        """,
        "features": f"""
            SELECT TEAM_ABBR, rs_net_rating, rs_off_rating, rs_def_rating,
                   rs_vs_top_teams_win_pct, rs_close_game_win_pct, rs_fta, rs_efg_pct
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
        "daily_model_scores": f"""
            SELECT GAME_DATE, TEAM_ABBR, games_played, preseason_prob,
                   model_title_prob, title_prob_blended
            FROM daily_model_scores
            WHERE SEASON = '{CURRENT_SEASON_STR}'
            ORDER BY TEAM_ABBR, GAME_DATE
        """,
        "player_impact": f"""
            SELECT TEAM_ABBR, PLAYER_NAME, games_played, games_missed,
                   team_win_pct_with, team_win_pct_without, team_net_with, team_net_without,
                   win_pct_delta, net_rating_delta, mpg, ppg, rpg, apg
            FROM player_impact
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
            "daily_model_scores": "daily_model_scores",
            "player_impact": "player_impact",
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

    if not out.get("daily_model_scores", pd.DataFrame()).empty:
        out["daily_model_scores"]["GAME_DATE"] = _to_dt(out["daily_model_scores"]["GAME_DATE"])

    return out


def build_team_title_odds_series(
    rs_df: pd.DataFrame, title_df: pd.DataFrame, team_abbr: str
) -> pd.DataFrame:
    """Build implied title odds trajectory anchored to current Monte Carlo odds.

    Shape comes from RS win% + net rating softmax with Bayesian shrinkage
    (20-game prior toward .500) to damp early-season small-sample spikes.
    The entire series is then scaled so the final point equals the team's
    current Monte Carlo title probability — preventing October 3-0 starts
    from inflating odds to 40% when actual odds were ~2-3%.
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
    # Bayesian shrinkage: blend cumulative win% toward .500 with a 20-game prior.
    # At gp=3, a 3-0 team shows 0.565 not 1.000, eliminating early spikes.
    _prior = 20
    season["shrunk_win_pct"] = (season["cum_wins"] + _prior * 0.5) / (season["gp"] + _prior)
    season["cum_pm"] = season.groupby("TEAM_ABBR")["pm"].cumsum() / season["gp"]

    season["strength"] = (season["shrunk_win_pct"] - 0.5) * 7.0 + (season["cum_pm"] * 0.07)
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

    # Smooth for readability.
    smoothed = team_daily["implied_title_prob"].ewm(span=6, adjust=False).mean()
    team_daily["title_odds_smoothed"] = np.clip(smoothed, 0, 1)

    # Look up current Monte Carlo title odds (used as reference line in chart, not for scaling).
    # Endpoint scaling distorts early values for teams whose standing has shifted over the season;
    # it's more honest to show shrinkage-dampened relative strength + a separate MC reference.
    mc_prob = None
    if title_df is not None and not title_df.empty and "title_prob" in title_df.columns:
        mc_row = title_df[title_df["TEAM_ABBR"] == team_abbr]
        if not mc_row.empty:
            mc_prob = float(mc_row.iloc[0]["title_prob"])

    team_daily["title_odds_pct"] = team_daily["implied_title_prob"] * 100.0
    team_daily["title_odds_smoothed_pct"] = team_daily["title_odds_smoothed"] * 100.0
    team_daily["mc_title_prob_pct"] = (mc_prob * 100.0) if mc_prob is not None else None
    return team_daily[["GAME_DATE", "title_odds_pct", "title_odds_smoothed_pct", "mc_title_prob_pct"]]


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
            use_container_width=True,
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
                use_container_width=True,
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
    """Return projected total games in series as integer in [4, 7].
    Uses most_likely_games (mode of discrete distribution) for better variance."""
    mlg = row.get("most_likely_games")
    if pd.notna(mlg):
        return int(mlg)

    # Fallback: mode from individual probabilities
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
        .block-container { padding-top: 3.5rem; }

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

        @media (max-width: 768px) {
            /* Reduce main container padding */
            .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-top: 3.5rem !important;
                max-width: 100vw !important;
                overflow-x: hidden !important;
            }

            /* Shrink hero */
            .hero-title { font-size: 1.25rem !important; }
            .hero-sub { font-size: 0.85rem !important; }

            /* Tab bar: horizontal scroll instead of wrapping */
            .stTabs [data-baseweb="tab-list"] {
                overflow-x: auto !important;
                flex-wrap: nowrap !important;
                scrollbar-width: none !important;
                -webkit-overflow-scrolling: touch !important;
                padding-bottom: 0 !important;
            }
            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }
            .stTabs [data-baseweb="tab"] {
                white-space: nowrap !important;
                font-size: 0.8rem !important;
                padding: 0.2rem 0.05rem 0.35rem !important;
            }

            /* Stack all st.columns() on mobile */
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.25rem 0 !important;
            }
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 0 !important;
            }

            /* Bracket inner sides: stack R1/R2/R3 vertically */
            .bracket-side { flex-direction: column !important; }
            .br-col.r1, .br-col.r2, .br-col.r3 { padding-top: 0 !important; }

            /* Sidebar button: keep it reachable */
            [data-testid="stSidebarCollapseButton"] { display: block !important; }

            /* Meta chips smaller on mobile */
            .meta-chip { font-size: 0.72rem !important; padding: 0.15rem 0.35rem !important; }
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
    initial_sidebar_state="auto",
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
daily_model_scores_df = tables.get("daily_model_scores", pd.DataFrame())
player_impact_df = tables.get("player_impact", pd.DataFrame())

# Overlay live standings on current_preds_df so records stay up to date between pipeline runs
live_standings = fetch_live_standings()
if not live_standings.empty and not current_preds_df.empty:
    live_map = live_standings.set_index("TEAM_ABBR")[["wins", "losses", "win_pct", "Record"]].to_dict("index")
    for col in ["wins", "losses", "win_pct"]:
        current_preds_df[col] = current_preds_df["TEAM_ABBR"].map(lambda a, c=col: live_map.get(a, {}).get(c, None)).fillna(current_preds_df[col])
    if not title_df.empty and "Record" in title_df.columns:
        title_df["Record"] = title_df["TEAM_ABBR"].map(lambda a: live_map.get(a, {}).get("Record")).fillna(title_df["Record"])
    if not projected_records_df.empty:
        for col in ["current_wins", "current_losses"]:
            src_col = "wins" if "wins" in col else "losses"
            projected_records_df[col] = projected_records_df["TEAM_ABBR"].map(
                lambda a, c=src_col: live_map.get(a, {}).get(c, None)
            ).fillna(projected_records_df[col])

# ── Preseason Vegas odds (BBRef, vig-removed) — used for badge logic in Tab 1 ──
_PRESEASON_PROBS: dict[str, float] = {
    "OKC": 0.2433, "DEN": 0.1272, "CLE": 0.0973, "NYK": 0.0827,
    "MIN": 0.0591, "HOU": 0.0551, "LAL": 0.0487, "LAC": 0.0435,
    "ORL": 0.0435, "GSW": 0.0318, "DET": 0.0243, "DAL": 0.0230,
    "PHI": 0.0202, "ATL": 0.0202, "MIL": 0.0148, "BOS": 0.0136,
    "SAS": 0.0123, "IND": 0.0082, "TOR": 0.0082, "MEM": 0.0066,
    "MIA": 0.0041, "NOP": 0.0027, "PHX": 0.0017, "POR": 0.0017,
    "SAC": 0.0017, "CHI": 0.0017, "CHA": 0.0008, "BKN": 0.0008,
    "UTA": 0.0008, "WAS": 0.0008,
}


def team_full_name(abbr: str, city_only: bool = False) -> str:
    """'BOS' → 'Boston Celtics' (or 'Boston' if city_only=True)."""
    if teams_df.empty:
        return abbr
    row = teams_df[teams_df["TEAM_ABBR"] == abbr]
    if row.empty:
        return abbr
    city = str(row.iloc[0].get("city", abbr))
    name = str(row.iloc[0].get("TEAM_NAME", abbr))
    return city if city_only else name


def build_hero_narrative(title_df: pd.DataFrame) -> str:
    """Generate a plain-English championship summary paragraph."""
    if title_df.empty:
        return "Run the simulation pipeline to generate championship predictions."
    sorted_df = title_df.sort_values("title_prob", ascending=False)
    leader = sorted_df.iloc[0]
    leader_name = team_full_name(str(leader["TEAM_ABBR"]))
    leader_pct = float(leader["title_prob"])
    n_sims = 10_000

    surprises = []
    for _, row in sorted_df.iterrows():
        abbr = str(row["TEAM_ABBR"])
        current = float(row["title_prob"])
        preseason = _PRESEASON_PROBS.get(abbr, 1 / 30)
        if current > preseason * 2.5 and current > 0.05:
            surprises.append(team_full_name(abbr))

    lead_sentence = (
        f"The model simulated the 2025–26 NBA playoffs {n_sims:,} times. "
        f"**{leader_name} are the strongest favorite ({leader_pct:.0%})** — "
        f"they win the championship roughly 1 in {round(1/leader_pct):.0f} times the bracket plays out."
    )
    if surprises:
        surprise_str = " and ".join(surprises[:2])
        lead_sentence += (
            f" {surprise_str} are the biggest surprises, "
            f"both far exceeding their preseason expectations."
        )
    return lead_sentence


def _title_badge(abbr: str, prob: float, rank: int) -> str:
    """Return plain-English badge for a team's title outlook."""
    if rank == 1:
        return "★ Favorite"
    preseason = _PRESEASON_PROBS.get(abbr, 1 / 30)
    if prob > preseason * 2.5 and prob > 0.05:
        return "↑ Surprise"
    if prob >= 0.05:
        return "→ Contender"
    return "⚠ Longshot"


def build_analyst_context() -> str:
    """Build a rich, LLM-readable context block from all loaded tables."""
    from config.settings import CURRENT_SEASON_STR as _cs
    lines: list[str] = [f"=== {_cs} NBA SEASON — MODEL CONTEXT ===\n"]

    # Title odds
    if not title_df.empty:
        lines.append("TITLE ODDS (Monte Carlo, 10,000 simulations):")
        for _, r in title_df.iterrows():
            seed_str = f"Seed {int(r['playoff_seed'])}" if not pd.isna(r.get("playoff_seed")) else "play-in"
            lines.append(
                f"  {r['TEAM_ABBR']} ({r['conference']}, {seed_str}): "
                f"{float(r['title_prob']):.1%} title | "
                f"{float(r['make_finals_prob']):.1%} Finals | "
                f"{float(r['make_conf_finals_prob']):.1%} Conf Finals | "
                f"{float(r['make_second_round_prob']):.1%} 2nd Round"
            )
        lines.append("")

    # Standings + features + projected records merged
    if not current_preds_df.empty:
        merged = current_preds_df.copy()
        if not features_df.empty:
            merged = merged.merge(features_df, on="TEAM_ABBR", how="left")
        if not projected_records_df.empty:
            proj_cols = ["TEAM_ABBR", "expected_final_wins", "p10_final_wins", "p90_final_wins",
                         "prob_make_top6", "prob_make_playin", "prob_miss_playoffs"]
            proj_sub = projected_records_df[[c for c in proj_cols if c in projected_records_df.columns]]
            merged = merged.merge(proj_sub, on="TEAM_ABBR", how="left")
        merged = merged.sort_values("pred_rank_all_30")

        lines.append("TEAM STANDINGS + MODEL METRICS (sorted by model rank):")
        for _, r in merged.iterrows():
            def _f(col, fmt):
                v = r.get(col)
                return f"{float(v):{fmt}}" if col in merged.columns and v == v and v is not None else "N/A"
            nr = _f("rs_net_rating", "+.1f")
            off = _f("rs_off_rating", ".1f")
            def_ = _f("rs_def_rating", ".1f")
            vt = _f("rs_vs_top_teams_win_pct", ".1%")
            cg = _f("rs_close_game_win_pct", ".1%")
            fta = _f("rs_fta", ".1f")
            efg = _f("rs_efg_pct", ".1%")
            score = _f("pred_survival_score", ".3f")
            proj = ""
            if "expected_final_wins" in merged.columns and not pd.isna(r.get("expected_final_wins")):
                proj = f" | proj {r['expected_final_wins']:.0f}W ({int(r['p10_final_wins'])}–{int(r['p90_final_wins'])})"
            lines.append(
                f"  #{int(r['pred_rank_all_30'])}/30 {r['TEAM_ABBR']} "
                f"({r['conference']} #{int(r['playoff_rank'])}): "
                f"{int(r['wins'])}-{int(r['losses'])} | "
                f"Net {nr} Off {off} Def {def_} | "
                f"vs-top {vt} close-game {cg} FTA {fta} eFG {efg} | "
                f"model-score {score}{proj}"
            )
        lines.append("")

    # Play-in odds
    if not play_in_df.empty:
        lines.append("PLAY-IN TOURNAMENT ODDS:")
        for conf in ["East", "West"]:
            conf_pi = play_in_df[play_in_df["conference"] == conf].sort_values("made_playoffs_prob", ascending=False)
            if not conf_pi.empty:
                parts = [
                    f"{r['team_abbr']} (S7:{float(r['seed7_prob']):.0%} S8:{float(r['seed8_prob']):.0%} overall:{float(r['made_playoffs_prob']):.0%})"
                    for _, r in conf_pi.iterrows()
                ]
                lines.append(f"  {conf}: {' | '.join(parts)}")
        lines.append("")

    # Series predictions
    if not series_df.empty:
        lines.append("PROJECTED BRACKET:")
        for _, r in series_df.iterrows():
            eg = f"{float(r['expected_games']):.1f}g" if not pd.isna(r.get("expected_games")) else ""
            lines.append(
                f"  {r['round']} ({r['conference']}): "
                f"#{r['high_seed']} {r['high_team']} {float(r['high_team_win_prob']):.0%} vs "
                f"#{r['low_seed']} {r['low_team']} {float(r['low_team_win_prob']):.0%} "
                f"→ pick: {r['predicted_winner']} {eg}"
            )
        lines.append("")

    # Player impact — top impact players league-wide for context
    if not player_impact_df.empty:
        splits = player_impact_df.dropna(subset=["net_rating_delta"]).copy()
        if not splits.empty:
            lines.append("KEY PLAYER IMPACT (with/without splits — top 15 by net rating delta):")
            top = splits.sort_values("net_rating_delta", ascending=False).head(15)
            for _, r in top.iterrows():
                lines.append(
                    f"  {r['TEAM_ABBR']} {r['PLAYER_NAME']}: "
                    f"net {r['net_rating_delta']:+.1f} (with {r['team_net_with']:+.1f}, without {r['team_net_without']:+.1f}) | "
                    f"win% {r['win_pct_delta']:+.0%} | "
                    f"{r['ppg']:.1f}ppg, played {int(r['games_played'])} missed {int(r['games_missed'])}"
                )
            lines.append("")

    return "\n".join(lines)


def build_team_context(team_abbr: str) -> dict:
    """Build a per-team stats dict for analyst scouting reports."""
    stats: dict = {}

    if not current_preds_df.empty:
        row = current_preds_df[current_preds_df["TEAM_ABBR"] == team_abbr]
        if not row.empty:
            r = row.iloc[0]
            stats["current_record"] = f"{int(r['wins'])}-{int(r['losses'])}"
            stats["win_pct"] = float(r["win_pct"])
            stats["conf_seed"] = int(r["playoff_rank"])
            stats["overall_rank"] = int(r["pred_rank_all_30"])
            if not pd.isna(r.get("pred_survival_score")):
                stats["survival_score"] = float(r["pred_survival_score"])
            raw_seed = r.get("playoff_seed")
            if raw_seed == raw_seed:  # not NaN
                stats["playoff_seed"] = int(raw_seed)

    if not features_df.empty:
        row = features_df[features_df["TEAM_ABBR"] == team_abbr]
        if not row.empty:
            r = row.iloc[0]
            stats["net_rating"] = float(r["rs_net_rating"])
            stats["off_rating"] = float(r["rs_off_rating"])
            stats["def_rating"] = float(r["rs_def_rating"])
            stats["vs_top_win_pct"] = float(r["rs_vs_top_teams_win_pct"])
            if "rs_close_game_win_pct" in features_df.columns and not pd.isna(r.get("rs_close_game_win_pct")):
                stats["close_game_win_pct"] = float(r["rs_close_game_win_pct"])
            if "rs_fta" in features_df.columns and not pd.isna(r.get("rs_fta")):
                stats["fta_per_game"] = float(r["rs_fta"])
            if "rs_efg_pct" in features_df.columns and not pd.isna(r.get("rs_efg_pct")):
                stats["efg_pct"] = float(r["rs_efg_pct"])

    if not projected_records_df.empty:
        row = projected_records_df[projected_records_df["TEAM_ABBR"] == team_abbr]
        if not row.empty:
            r = row.iloc[0]
            stats["projected_wins"] = float(r["expected_final_wins"])
            stats["proj_range"] = f"{int(r['p10_final_wins'])}–{int(r['p90_final_wins'])}"
            stats["prob_auto"] = float(r["prob_make_top6"])
            stats["prob_playin"] = float(r["prob_make_playin"])
            stats["prob_miss"] = float(r["prob_miss_playoffs"])

    if not title_df.empty:
        row = title_df[title_df["TEAM_ABBR"] == team_abbr]
        if not row.empty:
            r = row.iloc[0]
            stats["title_prob"] = float(r["title_prob"])
            stats["make_finals_prob"] = float(r["make_finals_prob"])
            stats["make_conf_finals_prob"] = float(r["make_conf_finals_prob"])
            stats["make_second_round_prob"] = float(r["make_second_round_prob"])

    if not play_in_df.empty:
        row = play_in_df[play_in_df["team_abbr"] == team_abbr]
        if not row.empty:
            r = row.iloc[0]
            stats["seed7_prob"] = float(r["seed7_prob"])
            stats["seed8_prob"] = float(r["seed8_prob"])
            stats["made_playoffs_prob"] = float(r["made_playoffs_prob"])

    if not player_rs_df.empty:
        team_players = player_rs_df[player_rs_df["TEAM_ABBR"] == team_abbr]
        if not team_players.empty:
            recent = team_players.sort_values("GAME_DATE").groupby("PLAYER_NAME").tail(10)
            avg_stats = (
                recent.groupby("PLAYER_NAME")
                .agg(ppg=("PTS", "mean"), rpg=("REB", "mean"), apg=("AST", "mean"), gp=("PTS", "count"))
                .reset_index()
            )
            avg_stats = avg_stats[avg_stats["gp"] >= 5].sort_values("ppg", ascending=False).head(5)
            stats["top_players"] = [
                f"{r['PLAYER_NAME']}: {r['ppg']:.1f}pts/{r['rpg']:.1f}reb/{r['apg']:.1f}ast"
                for _, r in avg_stats.iterrows()
            ]

    if not remaining_games_df.empty:
        team_sched = (
            remaining_games_df[remaining_games_df["TEAM_ABBR"] == team_abbr]
            .sort_values("GAME_DATE")
            .head(5)
        )
        if not team_sched.empty:
            stats["next_games"] = [
                f"vs {r['OPP_ABBR']} ({'Home' if r['IS_HOME'] else 'Away'}) {float(r['GAME_WIN_PROB']):.0%} win"
                for _, r in team_sched.iterrows()
            ]

    if not player_impact_df.empty:
        team_impact = player_impact_df[player_impact_df["TEAM_ABBR"] == team_abbr].copy()
        if not team_impact.empty:
            # Players with meaningful splits (3+ missed games), sorted by impact
            with_splits = team_impact.dropna(subset=["net_rating_delta"]).sort_values("net_rating_delta", ascending=False)
            stats["player_impact"] = []
            for _, r in with_splits.iterrows():
                stats["player_impact"].append(
                    f"{r['PLAYER_NAME']}: {r['ppg']:.1f}ppg {r['mpg']:.1f}mpg | "
                    f"played {int(r['games_played'])} missed {int(r['games_missed'])} | "
                    f"team net WITH {r['team_net_with']:+.1f} WITHOUT {r['team_net_without']:+.1f} "
                    f"(delta {r['net_rating_delta']:+.1f}) | "
                    f"win% WITH {r['team_win_pct_with']:.0%} WITHOUT {r['team_win_pct_without']:.0%} "
                    f"(delta {r['win_pct_delta']:+.0%})"
                )
            # Also include players without splits (haven't missed enough games)
            no_splits = team_impact[team_impact["net_rating_delta"].isna()]
            for _, r in no_splits.sort_values("ppg", ascending=False).iterrows():
                stats["player_impact"].append(
                    f"{r['PLAYER_NAME']}: {r['ppg']:.1f}ppg {r['mpg']:.1f}mpg | "
                    f"played {int(r['games_played'])} missed {int(r['games_missed'])} (insufficient missed games for split)"
                )

    return stats


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

tab_who, tab_ask, tab_bracket, tab_details = st.tabs(
    ["🏆 Who Wins?", "🤖 Ask the AI", "📊 The Bracket", "🔍 Details"]
)

# ── Tab 1: Who Wins? ────────────────────────────────────────────────────────
with tab_who:
    # Plain-English hero narrative
    narrative = build_hero_narrative(title_df)
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.82);border:1px solid #d6deea;border-radius:14px;'
        f'padding:1rem 1.2rem;margin-bottom:1.2rem;font-size:1.05rem;line-height:1.6;">'
        f'{narrative}</div>',
        unsafe_allow_html=True,
    )

    # Championship odds leaderboard — clickable tiles with inline detail
    st.markdown("<div class='section-label'>Championship Odds — All Playoff Contenders</div>", unsafe_allow_html=True)
    st.caption("Based on 10,000 simulated playoff brackets. Tap a team to see details.")

    if title_df.empty:
        st.info("Run the simulation pipeline to generate championship odds.")
    else:
        sorted_title = title_df.sort_values("title_prob", ascending=False).reset_index(drop=True)
        max_prob = float(sorted_title["title_prob"].iloc[0])

        # Pre-build quick-fact lookups
        _preds_map = {}
        if not current_preds_df.empty:
            _preds_map = current_preds_df.set_index("TEAM_ABBR").to_dict("index")
        _feats_map = {}
        if not features_df.empty:
            _feats_map = features_df.set_index("TEAM_ABBR").to_dict("index")
        _impact_map = {}
        if not player_impact_df.empty:
            for team, grp in player_impact_df.groupby("TEAM_ABBR"):
                _impact_map[team] = grp.sort_values("ppg", ascending=False).head(3)

        # Track which team is selected
        if "who_selected" not in st.session_state:
            st.session_state.who_selected = None

        for rank, row in sorted_title.iterrows():
            abbr = str(row["TEAM_ABBR"])
            prob = float(row["title_prob"])
            full_name = team_full_name(abbr)
            badge = _title_badge(abbr, prob, rank + 1)
            bar_pct = int(prob / max_prob * 100) if max_prob > 0 else 0
            is_selected = st.session_state.who_selected == abbr

            badge_color = {"★ Favorite": "#059669", "↑ Surprise": "#2563EB", "→ Contender": "#6B7280", "⚠ Longshot": "#9CA3AF"}.get(badge, "#9CA3AF")
            bar_color = TEAM_COLORS.get(abbr, "#0f4fd8")
            border_style = f"2px solid {bar_color}" if is_selected else "1px solid #e5e7eb"
            bg = "rgba(240,245,255,0.95)" if is_selected else "rgba(255,255,255,0.8)"

            # Tile card — clicking logo toggles detail panel
            tile_cols = st.columns([0.06, 0.79, 0.15])
            with tile_cols[0]:
                if st.button(
                    f"![{abbr}]({logo_url(abbr)})",
                    key=f"who_{abbr}",
                    type="tertiary",
                    help=f"Click for {full_name} details",
                ):
                    st.session_state.who_selected = None if is_selected else abbr
                    st.rerun()
            with tile_cols[1]:
                st.markdown(
                    f'<div style="font-weight:700;font-size:0.95rem;line-height:1.2;">{full_name}</div>'
                    f'<div style="background:#e5e7eb;border-radius:4px;height:6px;margin-top:4px;">'
                    f'  <div style="background:{bar_color};border-radius:4px;height:6px;width:{bar_pct}%;"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with tile_cols[2]:
                st.markdown(
                    f'<div style="text-align:right;padding-top:2px;">'
                    f'  <div style="font-weight:700;font-size:1.05rem;">{prob:.0%}</div>'
                    f'  <div style="font-size:0.7rem;color:{badge_color};font-weight:600;">{badge}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Inline detail panel — renders right below the selected tile
            if is_selected:
                sel_row = row
                pr = _preds_map.get(abbr, {})
                fr = _feats_map.get(abbr, {})

                rec = f"{int(pr['wins'])}-{int(pr['losses'])}" if pr else ""
                seed_str = ""
                if pr:
                    seed = pr.get("playoff_seed")
                    conf = pr.get("conference", "")
                    if seed == seed and seed is not None:
                        seed_str = f"#{int(seed)} seed · {conf}"

                detail_l, detail_r = st.columns([1.2, 1])
                with detail_l:
                    st.caption(f"{rec} · {seed_str}" if seed_str else rec)

                    # Playoff path funnel
                    r2_p = float(sel_row.get("make_second_round_prob", 0))
                    cf_p = float(sel_row.get("make_conf_finals_prob", 0))
                    fin_p = float(sel_row.get("make_finals_prob", 0))
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:0.3rem;flex-wrap:wrap;margin:0.3rem 0 0.5rem 0;">'
                        f'<span style="background:#dbeafe;border-radius:6px;padding:3px 8px;font-weight:600;font-size:0.8rem;">2nd Rd {r2_p:.0%}</span>'
                        f'<span style="color:#94a3b8;">→</span>'
                        f'<span style="background:#c7d2fe;border-radius:6px;padding:3px 8px;font-weight:600;font-size:0.8rem;">Conf Finals {cf_p:.0%}</span>'
                        f'<span style="color:#94a3b8;">→</span>'
                        f'<span style="background:#a5b4fc;border-radius:6px;padding:3px 8px;font-weight:600;font-size:0.8rem;color:#312e81;">Finals {fin_p:.0%}</span>'
                        f'<span style="color:#94a3b8;">→</span>'
                        f'<span style="background:#6366f1;border-radius:6px;padding:3px 8px;font-weight:700;font-size:0.8rem;color:white;">Title {prob:.0%}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if fr:
                        metric_parts = []
                        nr = fr.get("rs_net_rating")
                        if nr == nr and nr is not None:
                            metric_parts.append(f"Net Rating **{float(nr):+.1f}**")
                        efg = fr.get("rs_efg_pct")
                        if efg == efg and efg is not None:
                            metric_parts.append(f"eFG% **{float(efg):.1%}**")
                        cg = fr.get("rs_close_game_win_pct")
                        if cg == cg and cg is not None:
                            metric_parts.append(f"Close games **{float(cg):.0%}**")
                        vt = fr.get("rs_vs_top_teams_win_pct")
                        if vt == vt and vt is not None:
                            metric_parts.append(f"vs Top teams **{float(vt):.0%}**")
                        if metric_parts:
                            st.markdown(" · ".join(metric_parts))

                with detail_r:
                    st.markdown("**Key Players**")
                    impact_rows = _impact_map.get(abbr)
                    if impact_rows is not None and not impact_rows.empty:
                        for _, p in impact_rows.iterrows():
                            delta_html = ""
                            if not pd.isna(p.get("net_rating_delta")):
                                d = float(p["net_rating_delta"])
                                color = "#059669" if d > 0 else "#DC2626"
                                delta_html = f' <span style="color:{color};font-weight:600;">{d:+.1f} net</span>'
                            st.markdown(
                                f"**{p['PLAYER_NAME']}** — {p['ppg']:.1f}pts / {p['rpg']:.1f}reb / {p['apg']:.1f}ast{delta_html}",
                                unsafe_allow_html=True,
                            )
                    else:
                        if not player_rs_df.empty:
                            tp = player_rs_df[player_rs_df["TEAM_ABBR"] == abbr]
                            if not tp.empty:
                                recent = tp.sort_values("GAME_DATE").groupby("PLAYER_NAME").tail(10)
                                avgs = recent.groupby("PLAYER_NAME").agg(ppg=("PTS", "mean"), rpg=("REB", "mean"), apg=("AST", "mean"), gp=("PTS", "count")).reset_index()
                                avgs = avgs[avgs["gp"] >= 5].sort_values("ppg", ascending=False).head(3)
                                for _, p in avgs.iterrows():
                                    st.markdown(f"**{p['PLAYER_NAME']}** — {p['ppg']:.1f}pts / {p['rpg']:.1f}reb / {p['apg']:.1f}ast")

                st.markdown("---")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬 Show model details & methodology"):
        st.caption(
            "**How this works:** The model runs a Cox Proportional Hazards survival model trained on 15 seasons "
            "of NBA playoff data (2010–2024). It learns which regular-season metrics predict how far a team goes. "
            "Those scores feed a Logistic Regression that predicts each individual series, then a Monte Carlo "
            "simulation plays out the full bracket 10,000 times to generate these championship probabilities."
        )
        st.caption("**Key metrics:** Net rating, eFG%, close-game win%, vs-top-teams win%, free throw rate.")
        if not title_df.empty:
            st.dataframe(
                title_df.sort_values("title_prob", ascending=False)
                .assign(title_pct=lambda d: (d["title_prob"] * 100).round(1))
                [["TEAM_ABBR", "title_pct"]]
                .rename(columns={"TEAM_ABBR": "Team", "title_pct": "Title Odds (%)"}),
                hide_index=True,
                use_container_width=True,
            )

# ── Tab 2: Ask the AI ───────────────────────────────────────────────────────
with tab_ask:
    from config.settings import ANTHROPIC_API_KEY, CURRENT_SEASON_STR as _cur_season

    if not ANTHROPIC_API_KEY:
        st.info(
            "The AI analyst isn't available in this environment. "
            "Explore the other tabs to see the predictions!"
        )
    else:
        try:
            from pipeline.agent.analyst import answer_question, get_team_scouting_report
            _analyst_ok = True
        except Exception as _e:
            st.error(f"Could not load analyst module: {_e}")
            _analyst_ok = False

        if _analyst_ok:
            if "analyst_messages" not in st.session_state:
                st.session_state.analyst_messages = []
            if "analyst_history" not in st.session_state:
                st.session_state.analyst_history = []
            if "analyst_full_context" not in st.session_state:
                st.session_state.analyst_full_context = build_analyst_context()
            if "ai_prefill" not in st.session_state:
                st.session_state.ai_prefill = None

            _model_context = {
                "season": _cur_season,
                "full_context": st.session_state.analyst_full_context,
            }

            st.markdown(
                "<div style='font-size:1.5rem;font-weight:700;margin-bottom:0.3rem;'>Ask me anything about the NBA playoffs</div>",
                unsafe_allow_html=True,
            )
            st.caption("Powered by Claude AI — ask about teams, matchups, predictions, or how the model works.")

            # Starter question buttons
            _starters = [
                "Who's going to win the championship? 🏆",
                "Any surprise teams this year? 📈",
                "How does the model work? 🧠",
                "Who are the biggest upsets waiting to happen? ⚡",
            ]
            _sq_cols = st.columns(len(_starters))
            for _sq_col, _sq in zip(_sq_cols, _starters):
                if _sq_col.button(_sq, use_container_width=True, key=f"starter_{_sq[:15]}"):
                    st.session_state.ai_prefill = _sq

            st.markdown("<br>", unsafe_allow_html=True)

            # Scouting report in expander
            with st.expander("📋 Generate a team scouting report"):
                _all_teams = sorted(current_preds_df["TEAM_ABBR"].tolist()) if not current_preds_df.empty else (title_df["TEAM_ABBR"].tolist() if not title_df.empty else [])
                if _all_teams:
                    _col1, _col2 = st.columns([2, 1])
                    with _col1:
                        _scout_team = st.selectbox("Choose a team:", _all_teams, key="analyst_scout_team", label_visibility="collapsed")
                    with _col2:
                        if st.button("Generate", key="analyst_report_btn", use_container_width=True):
                            _team_stats = build_team_context(_scout_team)
                            _title_prob = _team_stats.get("title_prob", 0.0)
                            _prompt_text = f"Give me a playoff scouting report for {_scout_team}."
                            with st.spinner("Generating scouting report…"):
                                try:
                                    _report = get_team_scouting_report(_scout_team, _team_stats, _title_prob)
                                    st.session_state.analyst_messages.append({"role": "user", "content": _prompt_text})
                                    st.session_state.analyst_messages.append({"role": "assistant", "content": _report})
                                    st.session_state.analyst_history.append({"role": "user", "content": _prompt_text})
                                    st.session_state.analyst_history.append({"role": "assistant", "content": _report})
                                    st.rerun()
                                except Exception as _err:
                                    st.error(f"Analyst error: {_err}")

            # Chat history display
            for _msg in st.session_state.analyst_messages:
                with st.chat_message(_msg["role"]):
                    st.markdown(_msg["content"])

            # Handle prefill from starter buttons (pop clears it after one use)
            _prefill = st.session_state.pop("ai_prefill", None)

            # Chat input — prefill wins if set (button was just clicked)
            _user_input = _prefill or st.chat_input("Ask about teams, matchups, predictions…")
            if _user_input:
                st.session_state.analyst_messages.append({"role": "user", "content": _user_input})
                with st.chat_message("user"):
                    st.markdown(_user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            _answer, _updated_hist = answer_question(_user_input, _model_context, st.session_state.analyst_history)
                            st.markdown(_answer)
                            st.session_state.analyst_messages.append({"role": "assistant", "content": _answer})
                            st.session_state.analyst_history = _updated_hist
                        except Exception as _err:
                            _err_msg = f"Error: {_err}"
                            st.error(_err_msg)
                            st.session_state.analyst_messages.append({"role": "assistant", "content": _err_msg})

# ── Tab 3: The Bracket ──────────────────────────────────────────────────────
with tab_bracket:
    if series_df.empty:
        st.info("Run the simulation pipeline to generate bracket predictions.")
    else:
        st.markdown("<div class='section-label'>Projected First-Round Matchups</div>", unsafe_allow_html=True)
        st.caption("Plain-English series previews based on 10,000 simulated brackets.")

        round_order = ["First Round", "Conference Semifinals", "Conference Finals", "NBA Finals"]
        first_round = series_df[series_df["round"] == "First Round"].copy()
        first_round["_conf_sort"] = first_round["conference"].map({"East": 0, "West": 1})
        first_round = first_round.sort_values(["_conf_sort", "high_seed"])

        for _, sr in first_round.iterrows():
            high = str(sr["high_team"])
            low = str(sr["low_team"])
            high_prob = float(sr["high_team_win_prob"])
            low_prob = float(sr["low_team_win_prob"])
            winner = str(sr["predicted_winner"])
            win_prob = high_prob if winner == high else low_prob
            win_city = team_full_name(winner, city_only=True)
            lose_city = team_full_name(low if winner == high else high, city_only=True)
            mlg = int(sr.get("most_likely_games", 6) or 6)
            heavy = "heavy " if win_prob > 0.70 else ""
            conf = str(sr.get("conference", ""))

            plain = (
                f"**{team_full_name(high, city_only=True)} vs {team_full_name(low, city_only=True)}** "
                f"({conf}) — {win_city} are {heavy}favored with a **{win_prob:.0%} chance** to advance. "
                f"Series likely goes **{mlg} games**."
            )
            c_logo, c_text = st.columns([1, 10])
            with c_logo:
                st.image(logo_url(winner), width=36)
            with c_text:
                st.markdown(plain)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Show full bracket & technical series detail"):
            render_playoff_bracket_board(series_df)
            st.markdown("---")
            st.markdown("<div class='section-label'>Series Detail</div>", unsafe_allow_html=True)
            sorted_series = series_df.copy()
            sorted_series["_ro"] = sorted_series["round"].map(lambda x: round_order.index(x) if x in round_order else 9)
            sorted_series = sorted_series.sort_values(["_ro", "conference", "high_seed"])
            series_labels_b = [
                (f"{r['conference'] if r['round'] != 'NBA Finals' else 'Finals'} "
                 f"{r['round'].replace('Conference ', '')}: {r['high_team']} vs {r['low_team']}")
                for _, r in sorted_series.iterrows()
            ]
            series_keys_b = [f"{r['round']}|{r['conference']}|{r['high_team']}" for _, r in sorted_series.iterrows()]
            selected_label_b = st.selectbox("Select a series:", series_labels_b, key="selected_series_b")
            if selected_label_b:
                sel_idx_b = series_labels_b.index(selected_label_b)
                sel_key_b = series_keys_b[sel_idx_b]
                sel_row_b = sorted_series.iloc[sel_idx_b]
                high_team_b = str(sel_row_b["high_team"])
                low_team_b = str(sel_row_b["low_team"])
                high_prob_b = float(sel_row_b["high_team_win_prob"])
                low_prob_b = float(sel_row_b["low_team_win_prob"])
                winner_b = str(sel_row_b["predicted_winner"])
                win_prob_b = high_prob_b if winner_b == high_team_b else low_prob_b
                col_h_b, col_vs_b, col_l_b = st.columns([2, 1, 2])
                with col_h_b:
                    st.image(logo_url(high_team_b), width=72)
                    st.markdown(f"**{high_team_b}** (Seed {int(sel_row_b['high_seed'])})")
                    st.markdown(f"Win prob: **{high_prob_b:.1%}**")
                with col_vs_b:
                    st.markdown("<div style='text-align:center;font-size:1.4rem;font-weight:700;padding-top:1.5rem;'>vs</div>", unsafe_allow_html=True)
                with col_l_b:
                    st.image(logo_url(low_team_b), width=72)
                    st.markdown(f"**{low_team_b}** (Seed {int(sel_row_b['low_seed'])})")
                    st.markdown(f"Win prob: **{low_prob_b:.1%}**")
                st.markdown(f"**Projected winner: {winner_b}** ({win_prob_b:.1%})")
                st.progress(win_prob_b if winner_b == high_team_b else 1 - win_prob_b)
                p_cols_b = ["p_4_games", "p_5_games", "p_6_games", "p_7_games"]
                if all(c in sel_row_b.index for c in p_cols_b):
                    length_vals_b = [float(sel_row_b[c]) for c in p_cols_b]
                    mlg_val_b = int(sel_row_b.get("most_likely_games", 6) or 6)
                    fig_len_b = go.Figure(go.Bar(
                        x=["4 Games", "5 Games", "6 Games", "7 Games"],
                        y=[v * 100 for v in length_vals_b],
                        marker_color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"],
                        text=[f"{v:.1%}" for v in length_vals_b],
                        textposition="outside",
                    ))
                    fig_len_b.update_layout(
                        title=f"Series Length (Most likely: {mlg_val_b} games)",
                        yaxis_title="Probability (%)", height=280,
                        margin=dict(t=40, b=20, l=20, r=20), showlegend=False,
                    )
                    st.plotly_chart(fig_len_b, use_container_width=True, key=f"len_chart_b_{sel_key_b}")

# ── Tab 4: Details ──────────────────────────────────────────────────────────
with tab_details:
    st.info("📊 Technical detail view — the same data powering the other tabs, with model internals exposed.")

with tab_details:
    st.markdown("<div class='section-label'>Projected Final Standings → Play-In</div>", unsafe_allow_html=True)
    render_meta_chips(
        [
            ("Metric", "Monte Carlo projected final record + make-playoffs probability"),
            ("Source", "team_projected_record + app_play_in_current + current_season_predictions"),
            ("Interpretation", "Full 15-team projected standings feed into the play-in bracket"),
        ]
    )
    if play_in_df.empty:
        st.info("`app_play_in_current` is missing. Run simulation pipeline first.")
    else:
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

                # ── Projected Final Standings (all 15 teams) ──────────────────────────
                if not current_preds_df.empty:
                    conf_preds = (
                        current_preds_df[current_preds_df["conference"] == conf_name]
                        .copy()
                        .sort_values("playoff_rank")
                    )

                    # Build projected-record lookup for this conference
                    proj_lookup: dict = {}
                    if not projected_records_df.empty:
                        conf_proj = projected_records_df[
                            projected_records_df["CONFERENCE"] == conf_name
                        ]
                        for _, pr in conf_proj.iterrows():
                            proj_lookup[pr["TEAM_ABBR"]] = pr

                    # Sort by projected wins if Monte Carlo data available, else current rank
                    if proj_lookup:
                        conf_preds["_proj_wins"] = conf_preds["TEAM_ABBR"].map(
                            lambda a: proj_lookup[a]["expected_final_wins"] if a in proj_lookup else float(conf_preds[conf_preds["TEAM_ABBR"] == a]["wins"].iloc[0])
                        )
                        conf_preds = conf_preds.sort_values("_proj_wins", ascending=False).reset_index(drop=True)
                    conf_preds["proj_rank"] = range(1, len(conf_preds) + 1)

                    st.markdown("**Projected Final Standings**")
                    rows_html = ""
                    for _, br in conf_preds.iterrows():
                        proj_rank = int(br["proj_rank"])
                        abbr = str(br["TEAM_ABBR"])
                        record = f"{int(br['wins'])}-{int(br['losses'])}"

                        # Zone dividers
                        if proj_rank == 7:
                            rows_html += '<div style="text-align:center;color:#1E40AF;background:#EFF6FF;font-size:0.68rem;font-weight:600;padding:0.18rem 0;margin:0.15rem 0;border-radius:4px;letter-spacing:0.06em;">── PLAY-IN LINE ──</div>'
                        elif proj_rank == 11:
                            rows_html += '<div style="text-align:center;color:#9CA3AF;background:#F9FAFB;font-size:0.68rem;font-weight:600;padding:0.18rem 0;margin:0.15rem 0;border-radius:4px;letter-spacing:0.06em;">── ELIMINATED ──</div>'

                        # Projected record from Monte Carlo
                        proj = proj_lookup.get(abbr)
                        proj_str = ""
                        prob_badge = ""
                        if proj is not None:
                            ef_w = proj["expected_final_wins"]
                            p10_w = proj["p10_final_wins"]
                            p90_w = proj["p90_final_wins"]
                            proj_str = (
                                f'<span style="color:#374151;font-size:0.8rem;font-weight:600;">→ {ef_w:.0f}W</span>'
                                f'<span style="color:#9CA3AF;font-size:0.74rem;"> ({p10_w}–{p90_w})</span>'
                            )
                            if proj_rank <= 6:
                                p_auto = float(proj["prob_make_top6"])
                                if p_auto < 0.85:
                                    prob_badge = f'<span style="background:#FEF3C7;color:#92400E;border-radius:4px;padding:1px 5px;font-size:0.68rem;font-weight:600;">{p_auto:.0%} lock</span>'
                            elif proj_rank <= 10:
                                p_in = float(proj["prob_make_playin"])
                                prob_badge = f'<span style="background:#EFF6FF;color:#1E40AF;border-radius:4px;padding:1px 5px;font-size:0.68rem;font-weight:600;">{p_in:.0%} in</span>'
                            else:
                                p_climb = float(proj["prob_make_playin"])
                                if p_climb > 0.05:
                                    prob_badge = f'<span style="background:#F3F4F6;color:#6B7280;border-radius:4px;padding:1px 5px;font-size:0.68rem;">{p_climb:.0%} climb</span>'

                        # Zone badge
                        if proj_rank <= 6:
                            zone = '<span style="background:#D1FAE5;color:#065F46;border-radius:4px;padding:1px 5px;font-size:0.68rem;font-weight:700;">AUTO</span>'
                        elif proj_rank <= 10:
                            zone = '<span style="background:#DBEAFE;color:#1E40AF;border-radius:4px;padding:1px 5px;font-size:0.68rem;font-weight:700;">PLAY-IN</span>'
                        else:
                            zone = '<span style="background:#F3F4F6;color:#9CA3AF;border-radius:4px;padding:1px 5px;font-size:0.68rem;font-weight:700;">OUT</span>'

                        hot = _hot_streak(abbr)
                        rows_html += (
                            '<div style="display:flex;align-items:center;gap:0.35rem;padding:0.22rem 0;border-bottom:1px solid #F3F4F6;flex-wrap:wrap;">'
                            f'<span style="color:#9CA3AF;font-size:0.75rem;width:1.2rem;text-align:right;">{proj_rank}</span>'
                            f'<img src="{logo_url(abbr)}" width="18" height="18" style="border-radius:50%;" />'
                            f'<span style="font-weight:700;font-size:0.85rem;width:2.4rem;">{abbr}</span>'
                            f'<span style="color:#6B7280;font-size:0.78rem;width:3.4rem;">{record}</span>'
                            f'<span style="flex:1;">{proj_str}</span>'
                            f'{zone}'
                            f'{prob_badge}'
                            f'<span style="margin-left:auto;">{hot}</span>'
                            '</div>'
                        )
                    st.markdown(
                        f'<div style="background:#FAFAFA;border:1px solid #E4E7EE;border-radius:8px;padding:0.5rem 0.75rem;margin-bottom:0.75rem;">{rows_html}</div>',
                        unsafe_allow_html=True,
                    )

                # ── Play-in bracket (seeds 7-10) ──────────────────────────
                st.caption("Seeds 7–10 above enter the play-in. Winners claim Seed 7 and Seed 8.")
                _render_playin_bracket(conf_name, current_preds_df, play_in_df)

        st.caption("Play-in winners join Seeds 1–6 in the full 16-team field → see the ② Playoff Predictor tab.")


    st.markdown("---")
    st.markdown("<div class='section-label'>Team Breakdown</div>", unsafe_allow_html=True)

with tab_details:
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

        # ── Model Signal Breakdown ──────────────────────────────────────────
        if not features_df.empty:
            team_feat = features_df[features_df["TEAM_ABBR"] == team_abbr]
            if not team_feat.empty:
                tf = team_feat.iloc[0]
                st.markdown("<div class='section-label'>Model Signal Breakdown</div>", unsafe_allow_html=True)
                render_meta_chips([
                    ("Source", "model_features (5 FINAL_FEATURES)"),
                    ("Model", "CoxPH survival — rounds_reached target"),
                    ("Percentile", "vs all 30 teams this season"),
                ])

                def _pct_rank(col: str) -> int:
                    """League percentile for this team on col (higher = better)."""
                    if col not in features_df.columns:
                        return 50
                    vals = features_df[col].dropna()
                    if vals.empty:
                        return 50
                    return int((vals <= float(tf[col])).mean() * 100)

                def _bar(pct: int, color: str) -> str:
                    empty = "#E4E7EE"
                    return (
                        f'<div style="background:{empty};border-radius:3px;height:6px;width:100%;margin-top:3px;">'
                        f'<div style="background:{color};border-radius:3px;height:6px;width:{pct}%;"></div>'
                        '</div>'
                    )

                FEATURES_META = [
                    ("rs_net_rating",           "Net Rating",          "+.1f", "pts/100 possessions (overall team quality)"),
                    ("rs_vs_top_teams_win_pct", "vs Top Teams Win%",   ".1%",  "win% against ≥.600 teams this season"),
                    ("rs_close_game_win_pct",   "Close Game Win%",     ".1%",  "win% in games decided by ≤5 pts (clutch)"),
                    ("rs_efg_pct",              "eFG%",                ".1%",  "shot-quality-adjusted field goal efficiency"),
                    ("rs_fta",                  "Free Throw Att/g",    ".1f",  "physical pressure / foul-drawing volume"),
                ]

                feat_cols = st.columns(len(FEATURES_META))
                for col_ui, (feat, label, fmt, desc) in zip(feat_cols, FEATURES_META):
                    with col_ui:
                        val = tf.get(feat)
                        if val is None or (hasattr(val, '__class__') and val != val):
                            col_ui.caption(label)
                            col_ui.markdown("—")
                            continue
                        val = float(val)
                        pct_r = _pct_rank(feat)
                        bar_color = "#059669" if pct_r >= 67 else ("#F59E0B" if pct_r >= 33 else "#DC2626")
                        formatted = f"{val:{fmt}}"
                        st.markdown(
                            f'<div style="border:1px solid #E4E7EE;border-radius:8px;padding:0.5rem 0.6rem;background:#FAFAFA;">'
                            f'<div style="font-size:0.72rem;color:#9CA3AF;font-weight:600;">{label}</div>'
                            f'<div style="font-size:1.05rem;font-weight:700;color:#111827;">{formatted}</div>'
                            f'<div style="font-size:0.68rem;color:{bar_color};font-weight:600;">{pct_r}th pct</div>'
                            f'{_bar(pct_r, bar_color)}'
                            f'<div style="font-size:0.65rem;color:#9CA3AF;margin-top:3px;">{desc}</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                # Survival score summary
                team_pred = current_preds_df[current_preds_df["TEAM_ABBR"] == team_abbr]
                if not team_pred.empty:
                    score = team_pred.iloc[0].get("pred_survival_score")
                    rank_30 = team_pred.iloc[0].get("pred_rank_all_30")
                    if score == score and score is not None:  # not NaN
                        score_pct = _pct_rank("rs_net_rating")  # proxy; survival score not in features_df
                        all_scores = current_preds_df["pred_survival_score"].dropna().sort_values(ascending=False)
                        score_rank = int((all_scores > float(score)).sum()) + 1
                        st.caption(
                            f"**Model Score: {float(score):.3f}** — ranked #{score_rank}/30 by CoxPH survival output "
                            f"(higher = model projects deeper playoff run)"
                        )

        c1, c2 = st.columns([1.45, 1.0])
        with c1:
            st.markdown("<div class='section-label'>Title Odds Over Time (Current Season)</div>", unsafe_allow_html=True)
            line_color = TEAM_COLORS.get(team_abbr, "#35a7ff")

            # Use pre-computed daily CoxPH model scores from the pipeline.
            # Falls back to the simple win%+softmax function if table isn't loaded yet.
            team_scores = (
                daily_model_scores_df[daily_model_scores_df["TEAM_ABBR"] == team_abbr]
                .sort_values("GAME_DATE")
                if not daily_model_scores_df.empty else pd.DataFrame()
            )

            if not team_scores.empty:
                render_meta_chips(
                    [
                        ("Metric", "CoxPH model title odds"),
                        ("Preseason", f"{team_scores['preseason_prob'].iloc[0]*100:.1f}% Vegas prior"),
                        ("Blend", "prior fades out over first 40 games"),
                    ]
                )
                # Smooth for display
                raw = team_scores["title_prob_blended"] * 100
                smoothed = raw.ewm(span=6, adjust=False).mean()
                team_scores = team_scores.copy()
                team_scores["smoothed_pct"] = smoothed.values

                # Current MC title odds for reference line
                mc_val = None
                if title_df is not None and not title_df.empty and "title_prob" in title_df.columns:
                    mc_row = title_df[title_df["TEAM_ABBR"] == team_abbr]
                    if not mc_row.empty:
                        mc_val = float(mc_row.iloc[0]["title_prob"]) * 100

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=team_scores["GAME_DATE"],
                        y=team_scores["smoothed_pct"],
                        mode="lines",
                        name=f"{team_abbr} title odds",
                        line=dict(color=line_color, width=3, shape="spline", smoothing=1.0),
                    )
                )
                if mc_val is not None:
                    fig.add_hline(
                        y=mc_val,
                        line=dict(color=line_color, width=1.5, dash="dash"),
                        annotation_text=f"MC sim: {mc_val:.1f}%",
                        annotation_position="top right",
                        annotation_font_color=line_color,
                    )
                fig.update_layout(
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#111827",
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    yaxis=dict(title="Title Probability (%)", color=line_color, ticksuffix="%", gridcolor="#E4E7EE"),
                    xaxis=dict(title="Date", gridcolor="#E4E7EE"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "CoxPH model title probability estimated at each game date using cumulative season features. "
                    "Starts at preseason Vegas odds (Bayesian prior) and transitions to model output over the first 40 games. "
                    "Dashed line = current Monte Carlo simulation title probability."
                )
            else:
                # Fallback: simple softmax trajectory (used before daily_model_scores is available)
                render_meta_chips(
                    [
                        ("Metric", "Relative title likelihood path"),
                        ("Source", "RS win% + net rating softmax (Bayesian-shrunk)"),
                        ("Reference", "dashed line = current MC title odds"),
                    ]
                )
                traj = build_team_title_odds_series(rs_df, title_df, team_abbr)
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
                    mc_val = traj["mc_title_prob_pct"].iloc[-1] if "mc_title_prob_pct" in traj.columns else None
                    if mc_val is not None and not pd.isna(mc_val):
                        fig.add_hline(
                            y=float(mc_val),
                            line=dict(color=line_color, width=1.5, dash="dash"),
                            annotation_text=f"MC: {mc_val:.1f}%",
                            annotation_position="top right",
                            annotation_font_color=line_color,
                        )
                fig.update_layout(
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#111827",
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    yaxis=dict(title="Title Probability (%)", color=line_color, ticksuffix="%", gridcolor="#E4E7EE"),
                    xaxis=dict(title="Date", gridcolor="#E4E7EE"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Relative title likelihood from RS win% + net rating softmax. Dashed line = current MC title odds.")

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
                    use_container_width=True,
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
                use_container_width=True,
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

