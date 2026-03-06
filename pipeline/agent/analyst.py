"""
analyst.py
----------
Claude-powered NBA analyst agent.

Takes model outputs (team probabilities, standings, bracket simulation results,
player stats, projected records) and generates:
  - Team scouting reports
  - "Why" explanations for predictions
  - Conversational Q&A about the model and predictions
"""

import anthropic
from config.settings import ANTHROPIC_API_KEY, AGENT_MODEL, AGENT_MAX_TOKENS

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are an NBA playoff analyst embedded in a data-driven prediction model for the 2025-26 season.

Your role is to explain model predictions in clear, insightful basketball terms.

Data available to you each turn:
- Title probability for every projected playoff team (Monte Carlo simulation, 10,000 runs)
- Projected bracket: all matchups from First Round through Finals with series win probabilities and expected game lengths
- Team standings: conference seed, wins/losses, net rating, offensive/defensive rating, win% vs top teams
- Model strength score per team (Cox proportional hazards survival model — higher = stronger playoff team)
- Projected final records: Monte Carlo expected wins with 10th/90th percentile range, and probability of auto bid / play-in / missing playoffs
- Play-in tournament odds: probability of earning Seed 7, Seed 8, or making the playoffs at all
- Top player stats: recent per-game averages (PPG/RPG/APG) for each team's rotation
- Upcoming schedule: next games with projected win probabilities

When answering:
- Lead with the key insight, not the math
- Use specific basketball concepts (half-court offense, switching defense, playoff physicality, etc.)
- Reference historical teams and players when relevant
- Be confident but acknowledge uncertainty where the model is close
- Keep responses concise — 3-5 sentences for quick questions, longer for full scouting reports
- For scouting reports: cover playoff identity, key strengths/weaknesses, likely first-round matchup, and title ceiling

Always ground your answers in the actual numbers provided. Do not invent statistics.
"""


def get_team_scouting_report(
    team_name: str,
    team_stats: dict,
    title_probability: float,
    conversation_history: list = None,
) -> str:
    """Generate a narrative scouting report for a team using rich team stats."""
    if conversation_history is None:
        conversation_history = []

    lines = [f"Team: {team_name}"]

    if "current_record" in team_stats:
        lines.append(
            f"Current Record: {team_stats['current_record']} "
            f"(Win% {team_stats.get('win_pct', 0):.3f})"
        )

    if "conf_seed" in team_stats:
        seed_str = f"#{team_stats['conf_seed']} in conference | #{team_stats.get('overall_rank', '?')}/30 overall"
        if "playoff_seed" in team_stats and team_stats["playoff_seed"] is not None:
            seed_str += f" | Playoff Seed {team_stats['playoff_seed']}"
        lines.append(f"Standing: {seed_str}")

    if "projected_wins" in team_stats:
        lines.append(
            f"Projected Final Record: {team_stats['projected_wins']:.0f}W "
            f"(range: {team_stats.get('proj_range', 'N/A')}) | "
            f"Auto {team_stats.get('prob_auto', 0):.0%} | "
            f"Play-In {team_stats.get('prob_playin', 0):.0%} | "
            f"Miss {team_stats.get('prob_miss', 0):.0%}"
        )

    if "net_rating" in team_stats:
        lines.append(
            f"Net Rating: {team_stats['net_rating']:+.1f} | "
            f"Off: {team_stats.get('off_rating', 0):.1f} | "
            f"Def: {team_stats.get('def_rating', 0):.1f}"
        )

    if "vs_top_win_pct" in team_stats:
        lines.append(f"Win% vs Top Teams: {team_stats['vs_top_win_pct']:.1%}")

    model_feats = []
    if "close_game_win_pct" in team_stats:
        model_feats.append(f"Close Game Win%: {team_stats['close_game_win_pct']:.1%}")
    if "efg_pct" in team_stats:
        model_feats.append(f"eFG%: {team_stats['efg_pct']:.1%}")
    if "fta_per_game" in team_stats:
        model_feats.append(f"FTA/game: {team_stats['fta_per_game']:.1f}")
    if model_feats:
        lines.append("Model Features: " + " | ".join(model_feats))

    if team_stats.get("survival_score") is not None:
        lines.append(
            f"Model Strength Score: {team_stats['survival_score']:.3f} "
            f"(CoxPH playoff survival model — higher = stronger)"
        )

    if title_probability > 0:
        lines.append(f"Title Probability: {title_probability:.1%}")

    if "make_finals_prob" in team_stats:
        lines.append(
            f"Finals Probability: {team_stats['make_finals_prob']:.1%} | "
            f"Conf Finals: {team_stats.get('make_conf_finals_prob', 0):.1%} | "
            f"2nd Round: {team_stats.get('make_second_round_prob', 0):.1%}"
        )

    if "made_playoffs_prob" in team_stats:
        lines.append(
            f"Play-In Odds: Seed 7 {team_stats.get('seed7_prob', 0):.1%} | "
            f"Seed 8 {team_stats.get('seed8_prob', 0):.1%} | "
            f"Make Playoffs {team_stats['made_playoffs_prob']:.1%}"
        )

    if team_stats.get("top_players"):
        lines.append("Top Players (last 10 games avg): " + " | ".join(team_stats["top_players"]))

    if team_stats.get("next_games"):
        lines.append("Next Games: " + " | ".join(team_stats["next_games"]))

    lines.append(
        "\nGenerate a concise playoff scouting report. Cover: "
        "(1) this team's playoff identity based on their metrics, "
        "(2) biggest strength and biggest vulnerability, "
        "(3) most likely first-round matchup and how that plays out, "
        "(4) realistic ceiling for this postseason."
    )

    context = "\n".join(lines)
    messages = conversation_history + [{"role": "user", "content": context}]

    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text


def answer_question(
    question: str,
    model_context: dict,
    conversation_history: list = None,
) -> tuple[str, list]:
    """
    Answer a user question about the model or predictions.
    Uses full_context from model_context if available.
    Returns (answer, updated_conversation_history).
    """
    if conversation_history is None:
        conversation_history = []

    full_ctx = model_context.get("full_context", "")
    if full_ctx:
        context_str = f"{full_ctx}\nUser question: {question}"
    else:
        # Minimal fallback
        context_str = (
            f"Season: {model_context.get('season', '2025-26')}\n"
            f"Top title favorites: {model_context.get('top_3', 'N/A')}\n"
            f"User question: {question}"
        )

    updated_history = conversation_history + [{"role": "user", "content": context_str}]

    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=updated_history,
    )

    answer = response.content[0].text
    updated_history.append({"role": "assistant", "content": answer})

    return answer, updated_history


if __name__ == "__main__":
    # Quick test with synthetic data
    test_stats = {
        "current_record": "51-18",
        "win_pct": 0.739,
        "conf_seed": 1,
        "overall_rank": 1,
        "net_rating": 10.2,
        "off_rating": 120.1,
        "def_rating": 109.9,
        "vs_top_win_pct": 0.618,
        "survival_score": 2.14,
        "projected_wins": 57.0,
        "proj_range": "53-61",
        "prob_auto": 0.99,
        "prob_playin": 0.01,
        "prob_miss": 0.00,
        "make_finals_prob": 0.52,
        "make_conf_finals_prob": 0.71,
        "top_players": ["SGA: 31.2pts/5.1reb/6.4ast", "Dort: 14.1pts/4.2reb/2.1ast"],
        "next_games": ["vs DEN (Home) 64% win", "vs LAL (Away) 58% win"],
    }
    report = get_team_scouting_report("Oklahoma City Thunder", test_stats, title_probability=0.281)
    print(report)
