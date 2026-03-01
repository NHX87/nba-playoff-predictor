"""
analyst.py
----------
Claude-powered NBA analyst agent.

Takes model outputs (team probabilities, physicality scores, 
bracket simulation results) and generates:
  - Team scouting reports
  - "Why" explanations for predictions  
  - Scenario analysis based on physicality slider setting
  - Historical comparisons
  - Conversational Q&A about the model and predictions
"""

import anthropic
from config.settings import ANTHROPIC_API_KEY, AGENT_MODEL, AGENT_MAX_TOKENS

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are an NBA playoff analyst embedded in a data-driven prediction model.

Your role is to explain model predictions in clear, insightful basketball terms. You have access to:
- Team physicality scores (how much more physical each team gets in playoffs vs regular season)
- Pace differential scores (how much teams slow down in playoffs)
- Title probability distributions from Monte Carlo simulation
- Historical comparisons to past playoff teams

When explaining predictions:
- Lead with the key insight, not the math
- Use specific basketball concepts (half-court offense, switching defense, playoff officiating, etc.)
- Reference historical teams and players when relevant
- Be confident but acknowledge uncertainty
- Keep responses concise — 3-5 sentences for most explanations, longer for full scouting reports

Physicality slider context:
- Low physicality (0.5): Refs are calling it tight. Pace teams, free throw drawers, and perimeter play thrives.
- Default (1.0): Historical average playoff environment
- High physicality (2.0): Physical defense rewarded, paint presence dominant, pace teams punished.

Always ground explanations in the actual data and model outputs provided.
"""


def get_team_scouting_report(
    team_name: str,
    team_stats: dict,
    title_probability: float,
    physicality_weight: float = 1.0,
    conversation_history: list = None
) -> str:
    """
    Generate a narrative scouting report for a team.
    """
    if conversation_history is None:
        conversation_history = []

    context = f"""
Team: {team_name}
Title Probability: {title_probability:.1%}
Physicality Score: {team_stats.get('physicality_score', 'N/A'):.3f} (higher = more physical in playoffs)
Foul Rate Delta: {team_stats.get('foul_rate_delta', 'N/A'):.2f} (regular season → playoffs)
Pace Delta: {team_stats.get('pace_delta', 'N/A'):.2f} (negative = slows down in playoffs)
Defensive Rebound Delta: {team_stats.get('dreb_pct_delta', 'N/A'):.3f}
Physicality Weight Setting: {physicality_weight:.1f}x

Generate a concise playoff scouting report for this team based on their physicality profile and title probability.
Include: what their metrics say about their playoff identity, how the current physicality setting affects their odds, and one historical comparison if relevant.
"""

    messages = conversation_history + [{"role": "user", "content": context}]

    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    return response.content[0].text


def answer_question(
    question: str,
    model_context: dict,
    conversation_history: list = None
) -> tuple[str, list]:
    """
    Answer a user question about the model or predictions.
    Returns (answer, updated_conversation_history).
    """
    if conversation_history is None:
        conversation_history = []

    # Build context string from model outputs
    context_str = f"""
Current model context:
- Physicality weight: {model_context.get('physicality_weight', 1.0):.1f}x
- Top 3 title favorites: {model_context.get('top_3', 'N/A')}
- Biggest physicality advantage teams: {model_context.get('most_physical', 'N/A')}
- Biggest pace-and-space teams: {model_context.get('pace_teams', 'N/A')}
- Current season: {model_context.get('season', '2024-25')}

User question: {question}
"""

    updated_history = conversation_history + [{"role": "user", "content": context_str}]

    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=updated_history
    )

    answer = response.content[0].text
    updated_history.append({"role": "assistant", "content": answer})

    return answer, updated_history


def explain_physicality_shift(
    physicality_weight: float,
    odds_before: dict,
    odds_after: dict
) -> str:
    """
    Explain how changing the physicality slider shifted title odds.
    """
    # Find biggest movers
    movers = []
    for team in odds_before:
        if team in odds_after:
            delta = odds_after[team] - odds_before[team]
            movers.append((team, delta))

    movers.sort(key=lambda x: abs(x[1]), reverse=True)
    top_movers = movers[:4]

    context = f"""
The physicality weight was adjusted to {physicality_weight:.1f}x.

Biggest odds changes:
{chr(10).join([f"- {t}: {d:+.1%}" for t, d in top_movers])}

Explain in 3-4 sentences why these teams were most affected by this physicality adjustment.
Be specific about what their underlying metrics tell us.
"""

    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}]
    )

    return response.content[0].text


if __name__ == "__main__":
    # Quick test
    test_stats = {
        "physicality_score": 0.82,
        "foul_rate_delta": -1.2,
        "pace_delta": -2.8,
        "dreb_pct_delta": 0.04
    }
    report = get_team_scouting_report(
        "San Antonio Spurs",
        test_stats,
        title_probability=0.08,
        physicality_weight=1.5
    )
    print(report)
