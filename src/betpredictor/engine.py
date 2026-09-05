"""The prediction engine: fixtures in, market probabilities and picks out.

Pipeline for each fixture:

    strengths + Elo  ->  expected goals (lambda, mu)
                     ->  Dixon-Coles scoreline matrix
                     ->  analytical market probabilities
                     ->  ML blender correction
                     ->  final probability + confidence + recommendation
                     ->  (optional) value/edge vs. bookmaker odds
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ModelConfig
from .data.sample_data import (
    LEAGUE_BASE_GOALS,
    TEAM_STRENGTHS,
    default_strength,
    normalize_team_name,
)
from .markets import Markets, derive_markets
from .models.elo import EloRatings
from .models.ml import FEATURE_NAMES, LogisticBlender
from .models.poisson import DixonColesModel


@dataclass
class Prediction:
    league: str
    home: str
    away: str
    exp_home_goals: float
    exp_away_goals: float
    p_home: float
    p_draw: float
    p_away: float
    p_over_2_5: float
    p_btts: float
    # Headline market for this project:
    p_draw_or_over_2_5: float
    confidence: str
    recommended: bool
    # Optional value analysis when bookmaker odds are supplied:
    odds: Optional[float] = None
    implied_prob: Optional[float] = None
    edge: Optional[float] = None
    fair_odds: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def _confidence_label(p: float) -> str:
    if p >= 0.72:
        return "High"
    if p >= 0.62:
        return "Medium"
    if p >= 0.52:
        return "Low"
    return "Avoid"


class PredictionEngine:
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        strengths: Optional[Dict[str, Dict[str, object]]] = None,
        elo: Optional[EloRatings] = None,
        blender: Optional[LogisticBlender] = None,
    ):
        self.config = config or ModelConfig()
        self.strengths = strengths if strengths is not None else dict(TEAM_STRENGTHS)
        self.dc = DixonColesModel(max_goals=self.config.max_goals, rho=self.config.dixon_coles_rho)
        self.elo = elo or EloRatings(
            k=self.config.elo_k,
            home_field=self.config.elo_home_field,
            start=self.config.elo_start,
        )
        self.blender = blender or LogisticBlender()

    # --- team attributes -------------------------------------------------
    def _attrs(self, team: str, league: str) -> Dict[str, object]:
        if team in self.strengths:
            return self.strengths[team]
        canonical = normalize_team_name(team)
        return self.strengths.get(canonical, default_strength(league))

    # --- expected goals --------------------------------------------------
    def expected_goals(self, league: str, home: str, away: str) -> Tuple[float, float]:
        base = LEAGUE_BASE_GOALS.get(league, 1.35)
        h = self._attrs(home, league)
        a = self._attrs(away, league)

        home_xg = base * float(h["attack"]) * float(a["defense"]) * self.config.home_advantage
        away_xg = base * float(a["attack"]) * float(h["defense"])

        # Gently tilt by Elo when ratings have diverged from the priors through
        # the feedback loop. Bounded inside EloRatings so it can't dominate.
        h_mult, a_mult = self.elo.strength_multipliers(home, away)
        home_xg *= h_mult
        away_xg *= a_mult
        return max(home_xg, 0.05), max(away_xg, 0.05)

    # --- feature vector for the ML blender -------------------------------
    def _features(self, league: str, home: str, away: str, markets: Markets) -> List[float]:
        h = self._attrs(home, league)
        a = self._attrs(away, league)
        abs_elo_diff = abs(
            (self.elo.rating(home) + self.elo.home_field) - self.elo.rating(away)
        )
        return [
            markets.p_draw_or_over_2_5,
            markets.exp_total_goals,
            abs_elo_diff,
            float(h["attack"]) + float(a["attack"]),
            float(h["defense"]) + float(a["defense"]),
            float(h["form"]),
            float(a["form"]),
        ]

    def build_features(
        self, league: str, home: str, away: str
    ) -> Tuple[List[float], Markets]:
        """Public: (ML feature vector, analytical markets) for a fixture.

        Used by the feedback loop when (re)training the blender on realised
        results.
        """
        home_xg, away_xg = self.expected_goals(league, home, away)
        matrix = self.dc.score_matrix(home_xg, away_xg)
        markets = derive_markets(matrix)
        return self._features(league, home, away, markets), markets

    # --- single-fixture prediction ---------------------------------------
    def predict(
        self,
        league: str,
        home: str,
        away: str,
        odds: Optional[float] = None,
    ) -> Prediction:
        home_xg, away_xg = self.expected_goals(league, home, away)
        matrix = self.dc.score_matrix(home_xg, away_xg)
        markets = derive_markets(matrix)

        # Blend the analytical combo probability with the ML correction.
        features = self._features(league, home, away, markets)
        ml_p = self.blender.predict_proba(features)
        w = self.config.ml_weight if self.blender.fitted else 0.0
        combo = (1 - w) * markets.p_draw_or_over_2_5 + w * ml_p
        combo = min(max(combo, 0.0), 1.0)

        pred = Prediction(
            league=league,
            home=home,
            away=away,
            exp_home_goals=markets.exp_home_goals,
            exp_away_goals=markets.exp_away_goals,
            p_home=markets.p_home,
            p_draw=markets.p_draw,
            p_away=markets.p_away,
            p_over_2_5=markets.p_over_2_5,
            p_btts=markets.p_btts,
            p_draw_or_over_2_5=combo,
            confidence=_confidence_label(combo),
            recommended=combo >= self.config.recommend_threshold,
        )

        if odds is not None and odds > 1.0:
            pred.odds = odds
            pred.implied_prob = 1.0 / odds
            pred.fair_odds = (1.0 / combo) if combo > 0 else None
            # Expected value per unit staked: p*odds - 1. Positive = value.
            pred.edge = combo * odds - 1.0

        return pred

    # --- screen a slate of fixtures --------------------------------------
    def screen(
        self,
        fixtures: Sequence[Tuple[str, str, str]],
        threshold: Optional[float] = None,
        odds: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> List[Prediction]:
        """Predict every fixture and return those clearing ``threshold`` for the
        Draw-or-Over-2.5 market, sorted most-confident first."""
        thr = self.config.recommend_threshold if threshold is None else threshold
        preds: List[Prediction] = []
        for league, home, away in fixtures:
            o = odds.get((home, away)) if odds else None
            preds.append(self.predict(league, home, away, odds=o))
        picks = [p for p in preds if p.p_draw_or_over_2_5 >= thr]
        picks.sort(key=lambda p: p.p_draw_or_over_2_5, reverse=True)
        return picks

    def predict_all(
        self, fixtures: Sequence[Tuple[str, str, str]]
    ) -> List[Prediction]:
        """Predict every fixture (no filtering), sorted most-confident first."""
        preds = [self.predict(lg, h, a) for lg, h, a in fixtures]
        preds.sort(key=lambda p: p.p_draw_or_over_2_5, reverse=True)
        return preds


# Expose feature order for introspection / documentation.
ML_FEATURES = FEATURE_NAMES
