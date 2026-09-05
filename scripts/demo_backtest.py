"""Seed the feedback loop with a simulated backtest.

This makes ``evaluate`` and ``retrain`` meaningful out of the box by generating
synthetic-but-plausible results (scorelines sampled from each fixture's own
model) and folding them through the real feedback loop. It exercises the exact
production code path -- record predictions, settle results, score, retrain.

    PYTHONPATH=src python scripts/demo_backtest.py

Note: results are SIMULATED for demonstration. Replace with real results via
``betpredictor.cli result ...`` for genuine performance tracking.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betpredictor.data.sample_data import FIXTURES_2026_09_05
from betpredictor.engine import PredictionEngine
from betpredictor.feedback import FeedbackLoop


def _poisson_sample(lmbda: float, rng: random.Random) -> int:
    """Knuth's Poisson sampler (no numpy dependency)."""
    import math
    l = math.exp(-lmbda)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l:
            return k - 1


def main(n_matches: int = 120, seed: int = 7) -> None:
    rng = random.Random(seed)
    engine = PredictionEngine()
    loop = FeedbackLoop(engine)

    print(f"Simulating {n_matches} historical matches through the feedback loop...")
    for i in range(n_matches):
        league, home, away = rng.choice(FIXTURES_2026_09_05)
        pred = engine.predict(league, home, away)
        hg = _poisson_sample(pred.exp_home_goals, rng)
        ag = _poisson_sample(pred.exp_away_goals, rng)
        date = f"2025-{(i % 9) + 1:02d}-{(i % 27) + 1:02d}"
        loop.record_predictions(date, [pred])
        loop.record_result(date, home, away, hg, ag)

    perf = loop.performance(avg_odds=1.30)
    print("\nBefore retrain:", perf.as_dict())
    print("Retrain:", loop.retrain(min_samples=30))
    print("After retrain (future picks use tuned weight):",
          loop.performance(avg_odds=1.30).as_dict())
    print("\nDone. Try: PYTHONPATH=src python -m betpredictor.cli evaluate --odds 1.30")


if __name__ == "__main__":
    main()
