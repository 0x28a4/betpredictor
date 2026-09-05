"""SQLite persistence for predictions, results, user feedback and model state.

Standard-library ``sqlite3`` only. A single file DB (see ``config.DB_PATH``)
keeps the whole history of what the model predicted, what actually happened,
and what users told us -- which is exactly what the feedback loop needs to
score itself and retrain.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .config import CONFIG, DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now')),
    match_date TEXT,
    league TEXT,
    home TEXT,
    away TEXT,
    p_draw_or_over_2_5 REAL,
    p_over_2_5 REAL,
    p_draw REAL,
    exp_total_goals REAL,
    recommended INTEGER,
    UNIQUE(match_date, home, away)
);

CREATE TABLE IF NOT EXISTS results (
    prediction_id INTEGER PRIMARY KEY,
    home_goals INTEGER,
    away_goals INTEGER,
    combo_hit INTEGER,          -- 1 if draw-or-over-2.5 landed
    settled_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    rating INTEGER,             -- user's 1..5 confidence in the pick
    comment TEXT,
    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
);

CREATE TABLE IF NOT EXISTS model_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);
"""


class Storage:
    def __init__(self, db_path: Optional[Path] = None):
        CONFIG.ensure_dirs()
        self.db_path = str(db_path) if db_path is not None else str(DB_PATH)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # --- predictions -----------------------------------------------------
    def save_prediction(
        self,
        match_date: str,
        league: str,
        home: str,
        away: str,
        p_combo: float,
        p_over: float,
        p_draw: float,
        exp_total: float,
        recommended: bool,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (match_date, league, home, away, p_draw_or_over_2_5,
                    p_over_2_5, p_draw, exp_total_goals, recommended)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(match_date, home, away) DO UPDATE SET
                     p_draw_or_over_2_5=excluded.p_draw_or_over_2_5,
                     p_over_2_5=excluded.p_over_2_5,
                     p_draw=excluded.p_draw,
                     exp_total_goals=excluded.exp_total_goals,
                     recommended=excluded.recommended
                """,
                (match_date, league, home, away, p_combo, p_over, p_draw,
                 exp_total, int(recommended)),
            )
            if cur.lastrowid:
                row = conn.execute(
                    "SELECT id FROM predictions WHERE match_date=? AND home=? AND away=?",
                    (match_date, home, away),
                ).fetchone()
                return int(row["id"])
            return int(cur.lastrowid)

    def get_prediction_id(self, match_date: str, home: str, away: str) -> Optional[int]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM predictions WHERE match_date=? AND home=? AND away=?",
                (match_date, home, away),
            ).fetchone()
            return int(row["id"]) if row else None

    # --- results ---------------------------------------------------------
    def save_result(self, prediction_id: int, home_goals: int, away_goals: int, combo_hit: bool) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO results (prediction_id, home_goals, away_goals, combo_hit)
                   VALUES (?,?,?,?)
                   ON CONFLICT(prediction_id) DO UPDATE SET
                     home_goals=excluded.home_goals,
                     away_goals=excluded.away_goals,
                     combo_hit=excluded.combo_hit,
                     settled_at=datetime('now')
                """,
                (prediction_id, home_goals, away_goals, int(combo_hit)),
            )

    def settled_rows(self) -> List[sqlite3.Row]:
        """Predictions that have a realised result, for scoring/retraining."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT p.*, r.home_goals, r.away_goals, r.combo_hit
                   FROM predictions p JOIN results r ON r.prediction_id = p.id
                   ORDER BY p.match_date"""
            ).fetchall()

    # --- feedback --------------------------------------------------------
    def save_feedback(self, prediction_id: Optional[int], rating: Optional[int], comment: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO feedback (prediction_id, rating, comment) VALUES (?,?,?)",
                (prediction_id, rating, comment),
            )
            return int(cur.lastrowid)

    def all_feedback(self) -> List[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC"
            ).fetchall()

    # --- model state (elo, ml, tuned weights) ----------------------------
    def set_state(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO model_state (key, value, updated_at)
                   VALUES (?,?,datetime('now'))
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                     updated_at=datetime('now')""",
                (key, value),
            )

    def get_state(self, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM model_state WHERE key=?", (key,)
            ).fetchone()
            return row["value"] if row else None
