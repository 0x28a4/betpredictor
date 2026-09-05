"""Central configuration and tunable model parameters.

Values here are deliberately explicit rather than hidden magic numbers so the
model's behaviour is auditable. Anything a bettor might reasonably want to
tune (home advantage, the Dixon-Coles low-score correction, how much weight
the machine-learning blender gets, calibration decay) lives here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Filesystem layout ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "betpredictor.db"

# The three English leagues this project targets, with their
# football-data.org competition codes for live fetching.
LEAGUES = {
    "premier-league": {"name": "Premier League", "fd_code": "PL", "tier": 1},
    "championship": {"name": "EFL Championship", "fd_code": "ELC", "tier": 2},
    "league-one": {"name": "EFL League One", "fd_code": None, "tier": 3},
}


@dataclass(frozen=True)
class ModelConfig:
    """Tunable parameters for the statistical + ML ensemble."""

    # Maximum goals-per-side considered when building the scoreline matrix.
    # 10 covers >99.99% of probability mass for realistic expected-goal values.
    max_goals: int = 10

    # Global home advantage multiplier applied to the home side's expected
    # goals. ~1.25 reflects the long-run English league home edge.
    home_advantage: float = 1.25

    # Dixon-Coles low-score dependence parameter (rho). Negative values lift
    # 0-0/1-1 and deflate 1-0/0-1 relative to independent Poisson, matching
    # observed football scorelines.
    dixon_coles_rho: float = -0.08

    # League-average total goals per match, used as the scoring baseline when
    # a team's own rate is unknown. Split per side (base / 2 each before
    # strength adjustments).
    league_avg_goals: float = 2.7

    # Ensemble weight on the ML blender vs. the analytical Poisson market
    # probability. 0 = pure Poisson, 1 = pure ML. Re-tuned by the feedback
    # loop as realised results accumulate.
    ml_weight: float = 0.35

    # Exponential decay for weighting historical matches by recency when
    # (re)estimating team strengths. Higher = forget the past faster.
    recency_decay: float = 0.0065  # per day

    # Recommendation threshold: only surface a "Draw or Over 2.5" pick when
    # the model probability clears this. The market's natural floor for an
    # even game is ~0.65 (it loses only on 1-0/2-0/0-1/0-2), so 0.75 keeps the
    # screen genuinely selective rather than flagging every fixture.
    recommend_threshold: float = 0.75

    # Elo parameters for the results-driven ratings updater.
    elo_k: float = 20.0
    elo_home_field: float = 65.0
    elo_start: float = 1500.0


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    # football-data.org API token, read from the environment so secrets never
    # live in source control. Free tier covers PL + Championship.
    football_data_token: str = field(
        default_factory=lambda: os.environ.get("FOOTBALL_DATA_TOKEN", "")
    )
    # Optional API-Football (RapidAPI) key for League One and richer data.
    api_football_key: str = field(
        default_factory=lambda: os.environ.get("API_FOOTBALL_KEY", "")
    )

    def ensure_dirs(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)


CONFIG = AppConfig()
