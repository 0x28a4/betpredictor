"""Derive betting-market probabilities from a scoreline matrix.

Everything here is a deterministic sum over cells of the Dixon-Coles matrix,
so the outputs are internally consistent (e.g. p_home + p_draw + p_away == 1).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

Matrix = List[List[float]]


@dataclass
class Markets:
    """Probabilities for the markets the app cares about."""

    p_home: float
    p_draw: float
    p_away: float
    p_over_2_5: float
    p_under_2_5: float
    p_btts: float
    p_draw_or_over_2_5: float
    exp_home_goals: float
    exp_away_goals: float
    exp_total_goals: float

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def _cell(matrix: Matrix, i: int, j: int) -> float:
    if 0 <= i < len(matrix) and 0 <= j < len(matrix):
        return matrix[i][j]
    return 0.0


def draw_or_over_2_5(matrix: Matrix) -> float:
    """Probability that a match is a draw OR has 3+ total goals.

    Key insight used both here and in the tests: this combination bet loses on
    *exactly four* scorelines -- a decisive result with two or fewer goals:
    1-0, 2-0, 0-1, 0-2. So::

        P(draw or over 2.5) = 1 - P(1-0) - P(2-0) - P(0-1) - P(0-2)

    which is far less error-prone than inclusion-exclusion over two events.
    """
    losing = (
        _cell(matrix, 1, 0)
        + _cell(matrix, 2, 0)
        + _cell(matrix, 0, 1)
        + _cell(matrix, 0, 2)
    )
    return 1.0 - losing


def prob_over(matrix: Matrix, line: float = 2.5) -> float:
    """Probability total goals strictly exceed ``line`` (line is a .5 value)."""
    n = len(matrix)
    return sum(
        matrix[i][j] for i in range(n) for j in range(n) if (i + j) > line
    )


def prob_btts(matrix: Matrix) -> float:
    """Probability both teams score at least one goal."""
    n = len(matrix)
    return sum(matrix[i][j] for i in range(1, n) for j in range(1, n))


def derive_markets(matrix: Matrix) -> Markets:
    n = len(matrix)
    p_home = sum(matrix[i][j] for i in range(n) for j in range(n) if i > j)
    p_draw = sum(matrix[i][i] for i in range(n))
    p_away = sum(matrix[i][j] for i in range(n) for j in range(n) if i < j)

    p_over = prob_over(matrix, 2.5)
    eh = sum(i * sum(matrix[i][j] for j in range(n)) for i in range(n))
    ea = sum(j * sum(matrix[i][j] for i in range(n)) for j in range(n))

    return Markets(
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        p_over_2_5=p_over,
        p_under_2_5=1.0 - p_over,
        p_btts=prob_btts(matrix),
        p_draw_or_over_2_5=draw_or_over_2_5(matrix),
        exp_home_goals=eh,
        exp_away_goals=ea,
        exp_total_goals=eh + ea,
    )
