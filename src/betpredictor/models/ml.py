"""A lightweight logistic-regression blender (pure standard library).

The analytical Dixon-Coles model gives a principled probability for the
"Draw or Over 2.5" market. The machine-learning layer learns systematic
corrections to that probability from historical data and from realised results
fed back through the feedback loop -- capturing signal the parametric model
misses (e.g. league-specific goal inflation, venue effects, form momentum).

Implemented from scratch with batch gradient descent + L2 regularisation and
feature standardisation, so there is no third-party ML dependency. The API
(``fit`` / ``predict_proba``) intentionally mirrors scikit-learn so it can be
swapped for ``sklearn.linear_model.LogisticRegression`` without touching the
rest of the codebase.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Sequence

# Feature order is fixed and documented so persisted models stay interpretable.
FEATURE_NAMES = [
    "poisson_combo_prob",   # analytical P(draw or over 2.5)
    "exp_total_goals",      # Dixon-Coles expected total goals
    "abs_elo_diff",         # |elo gap| incl. home field (closeness -> draws)
    "attack_sum",           # combined attacking strength
    "defense_sum",          # combined (inverse) defensive solidity
    "home_form",            # recent home points-per-game, 0..3
    "away_form",            # recent away points-per-game, 0..3
]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class LogisticBlender:
    l2: float = 1.0
    lr: float = 0.1
    epochs: int = 400
    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    mean: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    fitted: bool = False

    # --- standardisation -------------------------------------------------
    def _standardise(self, x: Sequence[float]) -> List[float]:
        return [
            (x[i] - self.mean[i]) / self.std[i] if self.std[i] > 1e-9 else 0.0
            for i in range(len(x))
        ]

    def _compute_norm(self, X: Sequence[Sequence[float]]) -> None:
        n_feat = len(X[0])
        self.mean = [0.0] * n_feat
        self.std = [1.0] * n_feat
        m = len(X)
        for j in range(n_feat):
            col = [row[j] for row in X]
            mu = sum(col) / m
            var = sum((v - mu) ** 2 for v in col) / m
            self.mean[j] = mu
            self.std[j] = math.sqrt(var) if var > 0 else 1.0

    # --- training --------------------------------------------------------
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "LogisticBlender":
        if not X:
            raise ValueError("cannot fit on empty data")
        n_feat = len(X[0])
        self._compute_norm(X)
        Xs = [self._standardise(row) for row in X]
        self.weights = [0.0] * n_feat
        self.bias = 0.0
        m = len(Xs)

        for _ in range(self.epochs):
            grad_w = [0.0] * n_feat
            grad_b = 0.0
            for row, target in zip(Xs, y):
                pred = _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, row)))
                err = pred - target
                for j in range(n_feat):
                    grad_w[j] += err * row[j]
                grad_b += err
            for j in range(n_feat):
                grad_w[j] = grad_w[j] / m + self.l2 * self.weights[j] / m
                self.weights[j] -= self.lr * grad_w[j]
            self.bias -= self.lr * (grad_b / m)

        self.fitted = True
        return self

    # --- inference -------------------------------------------------------
    def predict_proba(self, x: Sequence[float]) -> float:
        if not self.fitted:
            # Untrained: defer entirely to the analytical probability, which is
            # feature 0. This makes an unfit blender a safe no-op.
            return float(x[0])
        xs = self._standardise(x)
        z = self.bias + sum(w * v for w, v in zip(self.weights, xs))
        return _sigmoid(z)

    # --- persistence -----------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            {
                "l2": self.l2,
                "lr": self.lr,
                "epochs": self.epochs,
                "weights": self.weights,
                "bias": self.bias,
                "mean": self.mean,
                "std": self.std,
                "fitted": self.fitted,
                "feature_names": FEATURE_NAMES,
            }
        )

    @classmethod
    def from_json(cls, blob: str) -> "LogisticBlender":
        d = json.loads(blob)
        m = cls(l2=d["l2"], lr=d["lr"], epochs=d["epochs"])
        m.weights = d["weights"]
        m.bias = d["bias"]
        m.mean = d["mean"]
        m.std = d["std"]
        m.fitted = d["fitted"]
        return m
