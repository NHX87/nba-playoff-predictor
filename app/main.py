"""
main.py
-------
Streamlit entry point for NBA Playoff Physicality Predictor.

Run with: streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    PHYSICALITY_WEIGHT_MIN,
    PHYSICALITY_WEIGHT_MAX,
    PHYSICALITY_WEIGHT_DEFAULT,
    CURRENT_SEASON_STR,
    DB_PATH,
)


@st.cache_data(ttl=300)
def load_app_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load app-ready analytics tables from DuckDB if available.
    Returns (title_odds, series_predictions, play_in_summary).
    """
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        tables = set(
            con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).df()["table_name"].tolist()
        )

        title_df = pd.DataFrame()
        series_df = pd.DataFrame()
        play_in_df = pd.DataFrame()

        if "app_title_odds_current" in tables:
            title_df = con.execute(
                """
                SELECT
                    TEAM_ABBR,
                    conference,
                    playoff_seed,
                    Record,
                    title_prob,
                    make_finals_prob,
                    make_conf_finals_prob,
                    make_second_round_prob
                FROM app_title_odds_current
                ORDER BY title_prob DESC
                """
            ).df()

        if "app_series_predictions_current" in tables:
            series_df = con.execute(
                """
                SELECT
                    conference,
                    round,
                    high_seed,
                    high_team,
                    low_seed,
                    low_team,
                    high_team_win_prob,
                    low_team_win_prob,
                    predicted_winner
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
                """
            ).df()

        if "app_play_in_current" in tables:
            play_in_df = con.execute(
                """
                SELECT
                    conference,
                    team_abbr,
                    seed7_prob,
                    seed8_prob,
                    made_playoffs_prob
                FROM app_play_in_current
                ORDER BY conference, made_playoffs_prob DESC
                """
            ).df()

        return title_df, series_df, play_in_df
    finally:
        con.close()

st.set_page_config(
    page_title="NBA Playoff Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLES ---
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }
    h1, h2, h3 { font-family: 'Georgia', serif; }
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .team-prob {
        font-size: 1.4rem;
        font-weight: bold;
        color: #4a9eff;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🏀 NBA Playoff Physicality Predictor")
st.markdown(f"*{CURRENT_SEASON_STR} Season — Title Probability Model*")
st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Model Controls")

    physicality_weight = st.slider(
        "Physicality Weight",
        min_value=PHYSICALITY_WEIGHT_MIN,
        max_value=PHYSICALITY_WEIGHT_MAX,
        value=PHYSICALITY_WEIGHT_DEFAULT,
        step=0.1,
        help="How much does playoff physicality matter? Low = pace-and-space rules. High = physical defense rewarded."
    )

    if physicality_weight < 0.8:
        st.info("📊 Low physicality: Pace teams and free throw drawers are favored.")
    elif physicality_weight > 1.4:
        st.warning("💪 High physicality: Paint presence and defensive teams are favored.")
    else:
        st.success("⚖️ Balanced: Historical average playoff environment.")

    st.divider()
    st.subheader("Injury Adjustments")
    st.caption("Toggle key player availability")
    # Placeholder — will be populated with actual current roster data
    st.info("Load current season data to enable injury toggles.")

    st.divider()
    st.caption("Built by Nicholas A. Harris")
    st.caption("[Portfolio](https://nicholasharris.dev) | [LinkedIn](https://linkedin.com/in/nicholas-harris-3166b719b)")

# --- MAIN CONTENT ---
col1, col2 = st.columns([2, 1])
title_odds_df, series_preds_df, play_in_df = load_app_outputs()

with col1:
    st.subheader("Title Probability")
    st.caption("Based on matchup model + Monte Carlo bracket simulation")

    if title_odds_df.empty:
        st.info("⚙️ No app-ready model outputs found. Run `python -m pipeline.models.simulation`.")
        sample_data = pd.DataFrame({
            "Team": ["Team A", "Team B", "Team C", "Team D", "Team E"],
            "Title Probability": [0.22, 0.18, 0.15, 0.12, 0.10],
        })
        fig = px.bar(
            sample_data,
            x="Title Probability",
            y="Team",
            orientation="h",
            title="Sample Output (run simulation to load real predictions)",
            template="plotly_dark"
        )
    else:
        plot_df = title_odds_df.copy()
        plot_df["Title Probability"] = plot_df["title_prob"]
        plot_df["Team"] = plot_df["TEAM_ABBR"] + " (" + plot_df["playoff_seed"].astype(str) + ")"
        fig = px.bar(
            plot_df.head(12),
            x="Title Probability",
            y="Team",
            orientation="h",
            color="make_finals_prob",
            color_continuous_scale="YlOrRd",
            title=f"{CURRENT_SEASON_STR} Simulated Title Odds",
            template="plotly_dark"
        )

    fig.update_layout(
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#0f0f0f",
        font_color="#f0f0f0"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Series Picks")
    st.caption("Head-to-head model predicted winner by series")

    if series_preds_df.empty:
        st.info("Run `python -m pipeline.models.simulation` to load series predictions.")
    else:
        first_round = series_preds_df[series_preds_df["round"] == "First Round"].copy()
        if first_round.empty:
            st.info("No first-round predictions available yet.")
        else:
            first_round["matchup"] = (
                first_round["high_seed"].astype(str)
                + " "
                + first_round["high_team"]
                + " vs "
                + first_round["low_seed"].astype(str)
                + " "
                + first_round["low_team"]
            )
            first_round["win_prob"] = first_round.apply(
                lambda r: r["high_team_win_prob"]
                if r["predicted_winner"] == r["high_team"]
                else r["low_team_win_prob"],
                axis=1,
            )
            st.dataframe(
                first_round[["conference", "matchup", "predicted_winner", "win_prob"]]
                .rename(columns={"win_prob": "predicted_win_prob"}),
                use_container_width=True,
                hide_index=True,
            )

    if not play_in_df.empty:
        st.divider()
        st.caption("Play-In Make-Playoffs Probabilities")
        st.dataframe(
            play_in_df.rename(
                columns={
                    "team_abbr": "team",
                    "made_playoffs_prob": "make_playoffs_prob",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# --- AI ANALYST CHAT ---
st.subheader("🤖 AI Analyst")
st.caption("Ask about predictions, team profiles, or why the model thinks what it thinks.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "I'm your NBA playoff analyst. Ask me about any team's physicality profile, why the model favors certain teams, or how the physicality slider affects title odds."
    })

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about the model or any team..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                from pipeline.agent.analyst import answer_question
                if not title_odds_df.empty:
                    top3 = ", ".join(
                        [
                            f"{r.TEAM_ABBR} ({r.title_prob:.1%})"
                            for r in title_odds_df.head(3).itertuples(index=False)
                        ]
                    )
                else:
                    top3 = "Load data to see current favorites"
                model_context = {
                    "physicality_weight": physicality_weight,
                    "season": CURRENT_SEASON_STR,
                    "top_3": top3,
                }
                answer, st.session_state.conversation_history = answer_question(
                    prompt,
                    model_context,
                    st.session_state.conversation_history
                )
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Agent error: {e}. Check your ANTHROPIC_API_KEY in .env")
