"""
fetch_live_scores.py
--------------------
Fetch today's live and completed NBA game scores + per-team stat leaders
from nba_api.live endpoints.

Two DataFrames returned:
  games_df   — one row per game (scores, status, period, clock, quarter splits)
  leaders_df — one row per team per game (top scorer / rebounder / assister)

Usage (standalone):
  python -m pipeline.ingestion.fetch_live_scores

Usage (in Streamlit with caching):
  from pipeline.ingestion.fetch_live_scores import get_live_scoreboard

  @st.cache_data(ttl=60)
  def live_scores():
      return get_live_scoreboard()

  games_df, leaders_df = live_scores()
"""

from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Clock helpers
# ---------------------------------------------------------------------------

def _parse_clock(clock_str: str) -> str:
    """Convert 'PT02M31.00S' → '2:31'. Returns empty string for blanks."""
    if not clock_str:
        return ""
    m = re.search(r"PT(\d+)M([\d.]+)S", clock_str)
    if not m:
        return clock_str
    mins = int(m.group(1))
    secs = int(float(m.group(2)))
    return f"{mins}:{secs:02d}"


def _period_label(period: int) -> str:
    if period <= 4:
        return f"Q{period}"
    return f"OT{period - 4}" if period > 5 else "OT"


# ---------------------------------------------------------------------------
# BoxScore leaders
# ---------------------------------------------------------------------------

def _top_player(players: list[dict], stat: str) -> dict[str, Any]:
    """Return the player dict with the highest value for `stat`."""
    eligible = [p for p in players if p.get("played") == "1"]
    if not eligible:
        eligible = players
    if not eligible:
        return {}
    return max(eligible, key=lambda p: p.get("statistics", {}).get(stat, 0))


def _leader_row(player: dict, team_tricode: str) -> dict[str, Any]:
    """Flatten a player dict into display-ready fields."""
    if not player:
        return {"name": "—", "pts": 0, "reb": 0, "ast": 0, "team_tricode": team_tricode}
    s = player.get("statistics", {})
    return {
        "name": player.get("name", "—"),
        "pts": int(s.get("points", 0)),
        "reb": int(s.get("reboundsTotal", 0)),
        "ast": int(s.get("assists", 0)),
        "team_tricode": team_tricode,
    }


def _fetch_game_leaders(game_id: str) -> dict[str, Any] | None:
    """
    Call BoxScore for a single game and extract top scorer/rebounder/assister
    for each team. Returns None on any error.
    """
    try:
        from nba_api.live.nba.endpoints.boxscore import BoxScore
        bs = BoxScore(game_id)
        game = bs.nba_response.get_dict().get("game", {})
    except Exception:
        return None

    result: dict[str, Any] = {}
    for side in ("homeTeam", "awayTeam"):
        team = game.get(side, {})
        tricode = team.get("teamTricode", "")
        players = team.get("players", [])
        team_stats = team.get("statistics", {})

        top_scorer   = _top_player(players, "points")
        top_rebounder = _top_player(players, "reboundsTotal")
        top_assister  = _top_player(players, "assists")

        result[side] = {
            "team_tricode": tricode,
            "team_pts": int(team.get("score", team_stats.get("points", 0))),
            "team_reb": int(team_stats.get("reboundsTotal", 0)),
            "team_ast": int(team_stats.get("assists", 0)),
            "team_efg": round(float(team_stats.get("fieldGoalsEffectiveAdjusted", 0)), 3),
            "top_scorer":    _leader_row(top_scorer,    tricode),
            "top_rebounder": _leader_row(top_rebounder, tricode),
            "top_assister":  _leader_row(top_assister,  tricode),
        }

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SLEEP_BETWEEN_BOXSCORE = 0.35  # seconds; live endpoint is lenient but be polite


def get_live_scoreboard() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch today's scoreboard + per-game stat leaders.

    Returns
    -------
    games_df : pd.DataFrame
        One row per game. Columns:
        game_id, game_status, game_status_text, period, period_label,
        clock, series_text,
        away_tricode, away_score, away_q1..q4,
        home_tricode, home_score, home_q1..q4
    leaders_df : pd.DataFrame
        Two rows per game (one per team). Columns:
        game_id, side (home/away), team_tricode, team_pts, team_reb, team_ast, team_efg,
        top_scorer_name, top_scorer_pts, top_scorer_reb, top_scorer_ast,
        top_rebounder_name, top_rebounder_reb,
        top_assister_name, top_assister_ast
    """
    from nba_api.live.nba.endpoints.scoreboard import ScoreBoard

    try:
        sb = ScoreBoard()
        raw_games = sb.nba_response.get_dict().get("scoreboard", {}).get("games", [])
    except Exception as exc:
        print(f"[fetch_live_scores] ScoreBoard fetch failed: {exc}")
        return pd.DataFrame(), pd.DataFrame()

    game_rows: list[dict] = []
    leader_rows: list[dict] = []

    for g in raw_games:
        game_id = g.get("gameId", "")
        status  = int(g.get("gameStatus", 0))   # 1=sched 2=live 3=final
        period  = int(g.get("period", 0))

        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})

        def _q_scores(team: dict) -> dict[str, int]:
            out: dict[str, int] = {f"q{i}": 0 for i in range(1, 5)}
            for p in team.get("periods", []):
                n = int(p.get("period", 0))
                if 1 <= n <= 4:
                    out[f"q{n}"] = int(p.get("score", 0))
            return out

        home_q = _q_scores(home)
        away_q = _q_scores(away)

        game_rows.append({
            "game_id":         game_id,
            "game_status":     status,
            "game_status_text": g.get("gameStatusText", ""),
            "period":          period,
            "period_label":    _period_label(period) if period > 0 else "—",
            "clock":           _parse_clock(g.get("gameClock", "")),
            "series_text":     g.get("seriesText", ""),
            "away_tricode":    away.get("teamTricode", ""),
            "away_score":      int(away.get("score", 0)),
            "away_q1": away_q["q1"], "away_q2": away_q["q2"],
            "away_q3": away_q["q3"], "away_q4": away_q["q4"],
            "home_tricode":    home.get("teamTricode", ""),
            "home_score":      int(home.get("score", 0)),
            "home_q1": home_q["q1"], "home_q2": home_q["q2"],
            "home_q3": home_q["q3"], "home_q4": home_q["q4"],
        })

        # Only fetch BoxScore for games that have started
        if status < 2:
            continue

        detail = _fetch_game_leaders(game_id)
        time.sleep(SLEEP_BETWEEN_BOXSCORE)

        if detail is None:
            continue

        for side_key, side_label in [("homeTeam", "home"), ("awayTeam", "away")]:
            d = detail.get(side_key, {})
            if not d:
                continue

            sc = d["top_scorer"]
            rb = d["top_rebounder"]
            ast = d["top_assister"]

            leader_rows.append({
                "game_id":           game_id,
                "side":              side_label,
                "team_tricode":      d["team_tricode"],
                "team_pts":          d["team_pts"],
                "team_reb":          d["team_reb"],
                "team_ast":          d["team_ast"],
                "team_efg":          d["team_efg"],
                # top scorer (with full pts/reb/ast line)
                "top_scorer_name":   sc["name"],
                "top_scorer_pts":    sc["pts"],
                "top_scorer_reb":    sc["reb"],
                "top_scorer_ast":    sc["ast"],
                # top rebounder (name + stat)
                "top_rebounder_name": rb["name"],
                "top_rebounder_reb":  rb["reb"],
                # top assister (name + stat)
                "top_assister_name":  ast["name"],
                "top_assister_ast":   ast["ast"],
            })

    games_df   = pd.DataFrame(game_rows)
    leaders_df = pd.DataFrame(leader_rows)

    return games_df, leaders_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_scoreboard(games_df: pd.DataFrame, leaders_df: pd.DataFrame) -> None:
    if games_df.empty:
        print("No games today.")
        return

    lkp = (
        leaders_df.set_index(["game_id", "team_tricode"]).to_dict("index")
        if not leaders_df.empty else {}
    )

    for _, g in games_df.iterrows():
        status = g["game_status_text"]
        series = f"  [{g['series_text']}]" if g["series_text"] else ""
        print(f"\n{'─'*55}")
        print(f"  {g['away_tricode']:>4}  {g['away_score']:>3}   @   {g['home_score']:<3}  {g['home_tricode']:<4}    {status}{series}")
        print(f"  Q1 {g['away_q1']}-{g['home_q1']}  Q2 {g['away_q2']}-{g['home_q2']}  "
              f"Q3 {g['away_q3']}-{g['home_q3']}  Q4 {g['away_q4']}-{g['home_q4']}")

        for tri in (g["away_tricode"], g["home_tricode"]):
            d = lkp.get((g["game_id"], tri))
            if not d:
                continue
            print(f"\n  {tri}  (eFG {d['team_efg']:.1%}  {d['team_reb']}reb  {d['team_ast']}ast)")
            print(f"    Scorer:    {d['top_scorer_name']:<24}  {d['top_scorer_pts']}pts / {d['top_scorer_reb']}reb / {d['top_scorer_ast']}ast")
            print(f"    Rebounder: {d['top_rebounder_name']:<24}  {d['top_rebounder_reb']}reb")
            print(f"    Assister:  {d['top_assister_name']:<24}  {d['top_assister_ast']}ast")


if __name__ == "__main__":
    print("Fetching live scoreboard...")
    games, leaders = get_live_scoreboard()
    _print_scoreboard(games, leaders)
    print(f"\ngames_df:   {len(games)} rows")
    print(f"leaders_df: {len(leaders)} rows")
