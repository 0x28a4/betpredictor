"""FastAPI web interface for BetPredictor.

Run with:
    pip install -r requirements.txt
    PYTHONPATH=src uvicorn web.app:app --reload

Endpoints
    GET  /                       -> single-page UI
    GET  /api/fixtures           -> predictions for the (sample or live) slate
    POST /api/predict            -> predict one custom fixture
    POST /api/feedback           -> submit user feedback / a realised result
    GET  /api/performance        -> feedback-loop metrics
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys

# Make the src/ package importable when run as `uvicorn web.app:app`.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from betpredictor.config import LEAGUES  # noqa: E402
from betpredictor.data.sample_data import FIXTURES_2026_09_05  # noqa: E402
from betpredictor.engine import PredictionEngine  # noqa: E402
from betpredictor.feedback import FeedbackLoop  # noqa: E402

app = FastAPI(title="BetPredictor", version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"

engine = PredictionEngine()
loop = FeedbackLoop(engine)  # loads any learned state from the DB


# --- schemas -------------------------------------------------------------
class PredictRequest(BaseModel):
    home: str
    away: str
    league: str = "premier-league"
    odds: Optional[float] = None


class FeedbackRequest(BaseModel):
    match_date: Optional[str] = None
    home: Optional[str] = None
    away: Optional[str] = None
    rating: Optional[int] = None
    comment: str = ""
    # Optional realised result -> folds into the learning loop.
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None


# --- routes --------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/leagues")
def leagues() -> JSONResponse:
    return JSONResponse({k: v["name"] for k, v in LEAGUES.items()})


@app.get("/api/fixtures")
def fixtures(date: str = "2026-09-05", live: bool = False, threshold: float = 0.75) -> JSONResponse:
    slate = list(FIXTURES_2026_09_05)
    source = "sample"
    if live:
        try:
            from betpredictor.data.sources import fetch_all_fixtures
            fetched = fetch_all_fixtures(date)
            if fetched:
                slate = [(f.league, f.home, f.away) for f in fetched]
                source = "live"
        except Exception as exc:  # noqa: BLE001 - report, then fall back
            source = f"sample (live failed: {exc})"

    preds = engine.predict_all(slate)
    # Persist so the feedback loop can later settle these picks.
    loop.record_predictions(date, preds)
    return JSONResponse(
        {
            "date": date,
            "source": source,
            "threshold": threshold,
            "count": len(preds),
            "predictions": [p.as_dict() for p in preds],
        }
    )


@app.post("/api/predict")
def predict(req: PredictRequest) -> JSONResponse:
    p = engine.predict(req.league, req.home, req.away, odds=req.odds)
    return JSONResponse(p.as_dict())


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> JSONResponse:
    result_hit = None
    if req.home_goals is not None and req.away_goals is not None and req.match_date and req.home and req.away:
        result_hit = loop.record_result(
            req.match_date, req.home, req.away, req.home_goals, req.away_goals
        )
    fid = loop.record_user_feedback(
        rating=req.rating, comment=req.comment,
        match_date=req.match_date, home=req.home, away=req.away,
    )
    return JSONResponse({"feedback_id": fid, "combo_hit": result_hit})


@app.get("/api/performance")
def performance(avg_odds: Optional[float] = None) -> JSONResponse:
    return JSONResponse(loop.performance(avg_odds=avg_odds).as_dict())


@app.post("/api/retrain")
def retrain() -> JSONResponse:
    return JSONResponse(loop.retrain())


# Serve static assets (JS/CSS) if present.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
