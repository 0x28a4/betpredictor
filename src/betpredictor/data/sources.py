"""Live data fetching from football-data.org (optional).

Kept deliberately thin and dependency-isolated: ``requests`` is imported lazily
so the core engine still works with zero third-party packages installed. If no
API token is configured, callers fall back to the offline sample dataset.

football-data.org free tier covers the Premier League (``PL``) and the EFL
Championship (``ELC``). League One is not on the free tier; supply an
API-Football key and extend ``fetch_fixtures_api_football`` for full coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..config import CONFIG, LEAGUES


@dataclass
class Fixture:
    league: str          # internal league key
    home: str
    away: str
    kickoff: str = ""    # ISO8601 if known
    status: str = "SCHEDULED"
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None


class DataUnavailable(RuntimeError):
    """Raised when live data can't be fetched (no token, network, etc.)."""


def _require_requests():
    try:
        import requests  # noqa: WPS433 (lazy, optional dependency)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataUnavailable(
            "The 'requests' package is required for live fetching. "
            "Install it with: pip install requests"
        ) from exc
    return requests


def fetch_fixtures_football_data(league_key: str, date: str) -> List[Fixture]:
    """Fetch fixtures for one league on a given ``YYYY-MM-DD`` date.

    Raises ``DataUnavailable`` if the league is not on football-data.org or no
    token is configured, so the caller can fall back gracefully.
    """
    token = CONFIG.football_data_token
    if not token:
        raise DataUnavailable(
            "No FOOTBALL_DATA_TOKEN set. Get a free key at "
            "https://www.football-data.org/ and export it, or run in "
            "offline/sample mode."
        )

    fd_code = LEAGUES.get(league_key, {}).get("fd_code")
    if not fd_code:
        raise DataUnavailable(
            f"League '{league_key}' is not available on football-data.org's "
            "free tier (League One needs API-Football)."
        )

    requests = _require_requests()
    url = f"https://api.football-data.org/v4/competitions/{fd_code}/matches"
    params = {"dateFrom": date, "dateTo": date}
    headers = {"X-Auth-Token": token}
    resp = requests.get(url, params=params, headers=headers, timeout=20)
    if resp.status_code != 200:
        raise DataUnavailable(
            f"football-data.org returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    fixtures: List[Fixture] = []
    for m in resp.json().get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        fixtures.append(
            Fixture(
                league=league_key,
                home=m["homeTeam"]["name"],
                away=m["awayTeam"]["name"],
                kickoff=m.get("utcDate", ""),
                status=m.get("status", "SCHEDULED"),
                home_goals=score.get("home"),
                away_goals=score.get("away"),
            )
        )
    return fixtures


def fetch_all_fixtures(date: str) -> List[Fixture]:
    """Best-effort fetch across all three target leagues for ``date``.

    Leagues that can't be fetched (no token, not on free tier) are skipped
    rather than failing the whole call. Returns whatever was retrievable.
    """
    out: List[Fixture] = []
    for league_key in LEAGUES:
        try:
            out.extend(fetch_fixtures_football_data(league_key, date))
        except DataUnavailable:
            continue
    return out
