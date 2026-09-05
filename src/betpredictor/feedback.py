"""The feedback loop -- what makes the system improve over time.

Three sources of feedback are folded back into the model:

1. Realised results     -> Elo ratings update; ML blender retrains; scoring.
2. Scoring metrics       -> Brier score, log-loss, hit-rate and ROI tell us how
                            calibrated we are and re-tune the ensemble weight.
3. User insight          -> 1..5 ratings + comments stored alongside picks, and
                            surfaced so humans can flag model blind spots.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .engine import Prediction, PredictionEngine
from .models.ml import LogisticBlender
from .storage import Storage


def combo_hit(home_goals: int, away_goals: int) -> bool:
    """Did 'Draw or Over 2.5 goals' land? True unless a <=2 goal decisive game."""
    return (home_goals == away_goals) or (home_goals + away_goals >= 3)


@dataclass
class Performance:
    n: int
    hit_rate: float           # fraction of settled picks that landed
    brier: float              # mean squared calibration error (lower better)
    log_loss: float           # lower better
    roi_flat: Optional[float] # ROI at supplied average odds, flat staking
    recommended_hit_rate: Optional[float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "hit_rate": round(self.hit_rate, 4),
            "brier": round(self.brier, 4),
            "log_loss": round(self.log_loss, 4),
            "roi_flat": round(self.roi_flat, 4) if self.roi_flat is not None else None,
            "recommended_hit_rate": (
                round(self.recommended_hit_rate, 4)
                if self.recommended_hit_rate is not None
                else None
            ),
        }


class FeedbackLoop:
    ELO_STATE_KEY = "elo_ratings"
    ML_STATE_KEY = "ml_blender"
    WEIGHT_STATE_KEY = "ml_weight"
    STRENGTHS_STATE_KEY = "team_strengths"

    def __init__(self, engine: PredictionEngine, storage: Optional[Storage] = None):
        self.engine = engine
        self.storage = storage or Storage()
        self._restore()

    # --- persistence glue ------------------------------------------------
    def _restore(self) -> None:
        elo_blob = self.storage.get_state(self.ELO_STATE_KEY)
        if elo_blob:
            self.engine.elo.load(json.loads(elo_blob))
        ml_blob = self.storage.get_state(self.ML_STATE_KEY)
        if ml_blob:
            self.engine.blender = LogisticBlender.from_json(ml_blob)
        w = self.storage.get_state(self.WEIGHT_STATE_KEY)
        if w:
            # ModelConfig is frozen; rebuild with the tuned weight.
            from dataclasses import replace
            self.engine.config = replace(self.engine.config, ml_weight=float(w))
        strengths = self.storage.get_state(self.STRENGTHS_STATE_KEY)
        if strengths:
            # Calibrated strengths override the seed priors for known teams.
            self.engine.strengths.update(json.loads(strengths))

    def save_strengths(self, strengths: Dict[str, Dict[str, object]]) -> None:
        """Persist calibrated team strengths and apply them to the engine."""
        self.engine.strengths.update(strengths)
        self.storage.set_state(self.STRENGTHS_STATE_KEY, json.dumps(strengths))

    def _persist_elo(self) -> None:
        self.storage.set_state(self.ELO_STATE_KEY, json.dumps(self.engine.elo.as_dict()))

    # --- recording -------------------------------------------------------
    def record_predictions(self, match_date: str, predictions: Sequence[Prediction]) -> List[int]:
        ids = []
        for p in predictions:
            ids.append(
                self.storage.save_prediction(
                    match_date=match_date,
                    league=p.league,
                    home=p.home,
                    away=p.away,
                    p_combo=p.p_draw_or_over_2_5,
                    p_over=p.p_over_2_5,
                    p_draw=p.p_draw,
                    exp_total=p.exp_home_goals + p.exp_away_goals,
                    recommended=p.recommended,
                )
            )
        return ids

    def record_result(
        self, match_date: str, home: str, away: str, home_goals: int, away_goals: int
    ) -> bool:
        """Ingest a realised result: store it, update Elo, persist. Returns the
        combo hit/miss. Retraining is a separate explicit step."""
        pid = self.storage.get_prediction_id(match_date, home, away)
        hit = combo_hit(home_goals, away_goals)
        if pid is not None:
            self.storage.save_result(pid, home_goals, away_goals, hit)
        # Elo learns from every result we see, predicted or not.
        self.engine.elo.update(home, away, home_goals, away_goals)
        self._persist_elo()
        return hit

    def record_user_feedback(
        self,
        rating: Optional[int] = None,
        comment: str = "",
        match_date: Optional[str] = None,
        home: Optional[str] = None,
        away: Optional[str] = None,
    ) -> int:
        pid = None
        if match_date and home and away:
            pid = self.storage.get_prediction_id(match_date, home, away)
        return self.storage.save_feedback(pid, rating, comment)

    # --- scoring ---------------------------------------------------------
    def performance(self, avg_odds: Optional[float] = None) -> Performance:
        rows = self.storage.settled_rows()
        if not rows:
            return Performance(0, 0.0, 0.0, 0.0, None, None)

        n = len(rows)
        hits = 0
        brier = 0.0
        ll = 0.0
        rec_hits = 0
        rec_n = 0
        roi_sum = 0.0

        eps = 1e-12
        for r in rows:
            p = float(r["p_draw_or_over_2_5"])
            y = int(r["combo_hit"])
            hits += y
            brier += (p - y) ** 2
            ll += -(y * math.log(p + eps) + (1 - y) * math.log(1 - p + eps))
            if int(r["recommended"]):
                rec_n += 1
                rec_hits += y
                if avg_odds:
                    roi_sum += (avg_odds - 1.0) if y else -1.0

        return Performance(
            n=n,
            hit_rate=hits / n,
            brier=brier / n,
            log_loss=ll / n,
            roi_flat=(roi_sum / rec_n) if (avg_odds and rec_n) else None,
            recommended_hit_rate=(rec_hits / rec_n) if rec_n else None,
        )

    # --- retraining ------------------------------------------------------
    def retrain(self, min_samples: int = 30) -> Dict[str, object]:
        """Refit the ML blender on realised results and re-tune the ensemble
        weight by comparing calibrated error. No-op below ``min_samples`` so we
        don't overfit a handful of games."""
        rows = self.storage.settled_rows()
        if len(rows) < min_samples:
            return {
                "retrained": False,
                "reason": f"need >= {min_samples} settled matches, have {len(rows)}",
                "n": len(rows),
            }

        X: List[List[float]] = []
        y: List[int] = []
        poisson_probs: List[float] = []
        for r in rows:
            feats, markets = self.engine.build_features(r["league"], r["home"], r["away"])
            X.append(feats)
            y.append(int(r["combo_hit"]))
            poisson_probs.append(markets.p_draw_or_over_2_5)

        blender = LogisticBlender()
        blender.fit(X, y)
        self.engine.blender = blender
        self.storage.set_state(self.ML_STATE_KEY, blender.to_json())

        # Re-tune ensemble weight: pick the weight (0..1 grid) minimising Brier
        # of the blend on the training set. Simple, transparent, and bounded.
        best_w, best_brier = 0.0, float("inf")
        for step in range(0, 11):
            w = step / 10.0
            brier = 0.0
            for feats, pois, target in zip(X, poisson_probs, y):
                ml_p = blender.predict_proba(feats)
                blend = (1 - w) * pois + w * ml_p
                brier += (blend - target) ** 2
            brier /= len(y)
            if brier < best_brier:
                best_brier, best_w = brier, w

        from dataclasses import replace
        self.engine.config = replace(self.engine.config, ml_weight=best_w)
        self.storage.set_state(self.WEIGHT_STATE_KEY, str(best_w))

        return {
            "retrained": True,
            "n": len(rows),
            "tuned_ml_weight": best_w,
            "train_brier": round(best_brier, 4),
        }
