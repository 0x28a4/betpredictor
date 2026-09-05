"""Dixon-Coles bivariate Poisson goal model.

This is the mathematical backbone of the predictor. Given each side's
expected goals (lambda for home, mu for away), it builds the full matrix of
scoreline probabilities P(home=i, away=j). Every market -- 1X2, Over/Under,
BTTS, correct score, and the "Draw or Over 2.5" combination -- is then just a
sum over the relevant cells of that matrix.

The Dixon & Coles (1997) correction adjusts the four lowest scorelines
(0-0, 1-0, 0-1, 1-1) because real football exhibits dependence between the two
teams' goal counts that an independent double-Poisson misses.

Pure standard library: no numpy required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

Matrix = List[List[float]]


def _poisson_pmf(k: int, lam: float) -> float:
    """Probability of exactly k events for a Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _dc_tau(i: int, j: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score dependence correction factor tau(i, j)."""
    if i == 0 and j == 0:
        return 1.0 - lam * mu * rho
    if i == 0 and j == 1:
        return 1.0 + lam * rho
    if i == 1 and j == 0:
        return 1.0 + mu * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


@dataclass
class DixonColesModel:
    """Builds scoreline-probability matrices from expected goals.

    Parameters mirror ``ModelConfig`` so the model is independent of the wider
    app and trivially unit-testable.
    """

    max_goals: int = 10
    rho: float = -0.08

    def score_matrix(self, home_xg: float, away_xg: float) -> Matrix:
        """Return a (max_goals+1) x (max_goals+1) matrix of P(i home, j away).

        The matrix is renormalised so it sums to exactly 1, absorbing the tiny
        truncated tail beyond ``max_goals`` and any distortion introduced by
        the low-score correction.
        """
        home_xg = max(home_xg, 1e-6)
        away_xg = max(away_xg, 1e-6)
        n = self.max_goals + 1

        home_pmf = [_poisson_pmf(i, home_xg) for i in range(n)]
        away_pmf = [_poisson_pmf(j, away_xg) for j in range(n)]

        matrix = [[0.0] * n for _ in range(n)]
        total = 0.0
        for i in range(n):
            for j in range(n):
                p = home_pmf[i] * away_pmf[j] * _dc_tau(i, j, home_xg, away_xg, self.rho)
                # The correction can in principle push a cell marginally
                # negative for extreme rho; clamp to keep a valid distribution.
                p = max(p, 0.0)
                matrix[i][j] = p
                total += p

        if total > 0:
            inv = 1.0 / total
            for i in range(n):
                for j in range(n):
                    matrix[i][j] *= inv
        return matrix

    @staticmethod
    def expected_goals_from_matrix(matrix: Matrix) -> tuple[float, float]:
        """Recover (E[home goals], E[away goals]) from a scoreline matrix."""
        n = len(matrix)
        eh = sum(i * sum(matrix[i][j] for j in range(n)) for i in range(n))
        ea = sum(j * sum(matrix[i][j] for i in range(n)) for j in range(n))
        return eh, ea
