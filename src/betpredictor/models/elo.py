"""Elo ratings that learn from realised results.

Elo gives the engine a second, self-correcting view of team strength that is
updated every time a result comes in through the feedback loop. The rating
*difference* between two teams is also fed to the ML blender as a feature, and
is used to nudge the Poisson expected-goals when a team has little match
history.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple


class EloRatings:
    def __init__(self, k: float = 20.0, home_field: float = 65.0, start: float = 1500.0):
        self.k = k
        self.home_field = home_field
        self.start = start
        self._ratings: Dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self._ratings.get(team, self.start)

    def set_rating(self, team: str, value: float) -> None:
        self._ratings[team] = value

    def expected_home_score(self, home: str, away: str) -> float:
        """Expected result in [0,1] for the home team (1=win, .5=draw)."""
        diff = (self.rating(home) + self.home_field) - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        """Update both ratings after a result, with a goal-margin multiplier.

        The margin-of-victory multiplier (as used in the World Football Elo
        ratings) makes bigger wins move ratings more, while damping the
        rating-difference feedback that would otherwise inflate favourites.
        """
        exp_home = self.expected_home_score(home, away)
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals < away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5

        margin = abs(home_goals - away_goals)
        rating_diff = abs((self.rating(home) + self.home_field) - self.rating(away))
        mov = math.log(margin + 1.0) * (2.2 / (rating_diff * 0.001 + 2.2))

        delta = self.k * mov * (actual_home - exp_home)
        self._ratings[home] = self.rating(home) + delta
        self._ratings[away] = self.rating(away) - delta

    def as_dict(self) -> Dict[str, float]:
        return dict(self._ratings)

    def load(self, ratings: Dict[str, float]) -> None:
        self._ratings.update(ratings)

    def strength_multipliers(self, home: str, away: str) -> Tuple[float, float]:
        """Convert the Elo gap into gentle attack multipliers around 1.0.

        Returns (home_mult, away_mult) used to tilt Poisson expected goals when
        blending Elo with the strength table. Bounded so Elo never dominates.
        """
        diff = (self.rating(home) + self.home_field) - self.rating(away)
        tilt = max(-0.35, min(0.35, diff / 1000.0))
        return 1.0 + tilt, 1.0 - tilt
