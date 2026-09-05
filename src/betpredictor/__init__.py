"""BetPredictor - AI-driven football match outcome prediction.

A hybrid statistical + machine-learning engine for estimating football match
markets (1X2, Over/Under, BTTS and combination markets such as
"Draw or Over 2.5 goals"), with a feedback loop that learns from realised
results and user insight.

The core (models, markets, engine, feedback) depends only on the Python
standard library so it runs anywhere. The web UI and live-data fetching add
optional dependencies (see requirements.txt).
"""

from .engine import PredictionEngine, Prediction
from .models.poisson import DixonColesModel
from .markets import derive_markets, draw_or_over_2_5

__version__ = "0.1.0"

__all__ = [
    "PredictionEngine",
    "Prediction",
    "DixonColesModel",
    "derive_markets",
    "draw_or_over_2_5",
    "__version__",
]
