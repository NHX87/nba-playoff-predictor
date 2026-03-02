import os
from dotenv import load_dotenv

load_dotenv()

# --- DATABASE ---
# Streamlit Community Cloud sets secrets via st.secrets, which are also
# injected as environment variables — so os.getenv picks them up automatically.
# Locally: uses full nba.duckdb. On Streamlit Cloud: uses slim nba_app.duckdb.
DB_PATH = os.getenv("DB_PATH", "data/processed/nba.duckdb")

# --- SEASONS ---
TRAIN_SEASON_START = int(os.getenv("TRAIN_SEASON_START", 2010))
TRAIN_SEASON_END = int(os.getenv("TRAIN_SEASON_END", 2024))
CURRENT_SEASON = int(os.getenv("CURRENT_SEASON", 2025))

def season_str(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"

TRAIN_SEASONS = [season_str(y) for y in range(TRAIN_SEASON_START, TRAIN_SEASON_END + 1)]
CURRENT_SEASON_STR = season_str(CURRENT_SEASON)

# --- FINAL MODEL FEATURES ---
# RS-only feature policy:
# Use only regular-season features that are available during the season.
# Avoid playoff-derived features and RS->PO delta features.

FINAL_FEATURES = [
    'rs_vs_top_teams_win_pct',    # Performance vs regular-season top teams (>=0.600 win pct)
    'rs_net_rating',              # Team point differential per 100 possessions
    'rs_close_game_win_pct',      # Clutch regular-season performance
    'rs_fta',                     # Free throw attempts volume (physical pressure proxy)
    'rs_efg_pct',                 # Shot-quality-adjusted efficiency
]

TARGET = 'rounds_reached'

# Features dropped — documented for reproducibility
DROPPED_FEATURES = [
    'rs_win_pct',                 # Redundant with rs_net_rating (high collinearity, weaker joint validation)
    'rs_off_rating',              # Redundant with shooting efficiency features
    'rs_def_rating',              # Added noise in current validation split
    'rs_close_game_count',        # Weak/unstable incremental value
    'rs_ppg',                     # Largely subsumed by rs_efg_pct + rs_fta
    'playoff_readiness_score',       # Composite includes playoff-derived components
    'ppg_delta',                     # Requires playoff points
    'playoff_rounds_prior',          # Prior-playoff signal removed for strict RS-only policy
    'defensive_intensity_score',     # Playoff-derived delta signal
    'ball_security_score',           # Playoff-derived delta signal
    'ts_pct_delta',                  # RS->PO delta
    'efg_pct_delta',                 # RS->PO delta
    'offensive_adaptability_score',  # Playoff-derived composite
    'tov_rate_delta',                # RS->PO delta
    'stl_rate_delta',                # RS->PO delta
    'physicality_score',             # Contains playoff adjustment effects
    'foul_rate_delta',               # RS->PO delta
    'three_pt_rate_delta',           # RS->PO delta
    'blk_rate_delta',                # RS->PO delta
    'ft_rate_delta',                 # RS->PO delta
]

# --- MODEL ---
MONTE_CARLO_RUNS = 10_000
RANDOM_STATE = 42

# Slider range kept for current Streamlit UI controls.
# This slider is presentation-only and does not alter RS-only model training.
PHYSICALITY_WEIGHT_MIN = 0.5
PHYSICALITY_WEIGHT_MAX = 2.0
PHYSICALITY_WEIGHT_DEFAULT = 1.0

# --- AGENT ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AGENT_MODEL = "claude-sonnet-4-20250514"
AGENT_MAX_TOKENS = 1000
