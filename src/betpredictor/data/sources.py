"""Live fixture/result fetching (optional dependency: ``requests``).

Two real providers are wired in, tried in order of convenience:

1. **ESPN** (default) -- ESPN's public JSON scoreboard API. No API key, and it
   covers all three target leagues (``eng.1`` Premier League, ``eng.2``
   Championship, ``eng.3`` League One). This is the recommended source.

2. **football-data.org** -- used when a ``FOOTBALL_DATA_TOKEN`` is set. Higher
   data quality but the free tier only covers PL + Championship.

Both return the same ``Fixture`` shape. ``requests`` is imported lazily so the
core engine keeps working with zero third-party packages installed. If nothing
is reachable, callers fall back to the offline sample dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..config import CONFIG, LEAGUES

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
FOOTBALL_DATA_MATCHES = "https://api.football-data.org/v4/competitions/{code}/matches"


@dataclass
class Fixture:
    league: str          # internal league key (e.g. "premier-league")
    home: str
    away: str
    kickoff: str = ""    # ISO8601 if known
    status: str = "SCHEDULED"
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    source: str = ""


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


# ---------------------------------------------------------------------------
# ESPN provider (no API key)
# ---------------------------------------------------------------------------
def _parse_espn_scoreboard(payload: dict, league_key: str) -> List[Fixture]:
    """Turn an ESPN scoreboard JSON payload into ``Fixture`` objects.

    Pulled out as a pure function so it is unit-testable against a captured
    payload without any network access.
    """
    fixtures: List[Fixture] = []
    for event in payload.get("events", []):
        comps = event.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        home = away = None
        home_score = away_score = None
        for c in comp.get("competitors", []):
            team_name = (c.get("team") or {}).get("displayName") or (c.get("team") or {}).get("name")
            score_raw = c.get("score")
            try:
                score = int(score_raw) if score_raw not in (None, "") else None
            except (TypeError, ValueError):
                score = None
            if c.get("homeAway") == "home":
                home, home_score = team_name, score
            elif c.get("homeAway") == "away":
                away, away_score = team_name, score
        if not home or not away:
            continue

        state = ((event.get("status") or {}).get("type") or {}).get("state", "pre")
        status = {"pre": "SCHEDULED", "in": "IN_PLAY", "post": "FINISHED"}.get(state, "SCHEDULED")
        # Scores are only meaningful once a game has started.
        if status == "SCHEDULED":
            home_score = away_score = None

        fixtures.append(
            Fixture(
                league=league_key,
                home=home,
                away=away,
                kickoff=event.get("date", ""),
                status=status,
                home_goals=home_score,
                away_goals=away_score,
                source="espn",
            )
        )
    return fixtures


def fetch_fixtures_espn(league_key: str, date: str, timeout: int = 20) -> List[Fixture]:
    """Fetch fixtures for one league on ``date`` (YYYY-MM-DD) from ESPN."""
    slug = LEAGUES.get(league_key, {}).get("espn_slug")
    if not slug:
        raise DataUnavailable(f"No ESPN slug configured for league '{league_key}'.")

    requests = _require_requests()
    url = ESPN_SCOREBOARD.format(slug=slug)
    params = {"dates": date.replace("-", "")}  # ESPN wants YYYYMMDD
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - network errors are expected
        raise DataUnavailable(f"ESPN request failed: {exc}") from exc
    if resp.status_code != 200:
        raise DataUnavailable(f"ESPN returned HTTP {resp.status_code}")
    return _parse_espn_scoreboard(resp.json(), league_key)


# ---------------------------------------------------------------------------
# football-data.org provider (needs a free API token)
# ---------------------------------------------------------------------------
def _parse_football_data(payload: dict, league_key: str) -> List[Fixture]:
    fixtures: List[Fixture] = []
    for m in payload.get("matches", []):
        score = (m.get("score", {}) or {}).get("fullTime", {}) or {}
        fixtures.append(
            Fixture(
                league=league_key,
                home=m["homeTeam"]["name"],
                away=m["awayTeam"]["name"],
                kickoff=m.get("utcDate", ""),
                status=m.get("status", "SCHEDULED"),
                home_goals=score.get("home"),
                away_goals=score.get("away"),
                source="football-data",
            )
        )
    return fixtures


def fetch_fixtures_football_data(league_key: str, date: str, timeout: int = 20) -> List[Fixture]:
    token = CONFIG.football_data_token
    if not token:
        raise DataUnavailable("No FOOTBALL_DATA_TOKEN set.")
    fd_code = LEAGUES.get(league_key, {}).get("fd_code")
    if not fd_code:
        raise DataUnavailable(
            f"League '{league_key}' is not on football-data.org's free tier."
        )
    requests = _require_requests()
    url = FOOTBALL_DATA_MATCHES.format(code=fd_code)
    try:
        resp = requests.get(
            url,
            params={"dateFrom": date, "dateTo": date},
            headers={"X-Auth-Token": token},
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        raise DataUnavailable(f"football-data.org request failed: {exc}") from exc
    if resp.status_code != 200:
        raise DataUnavailable(
            f"football-data.org returned HTTP {resp.status_code}: {resp.text[:160]}"
        )
    return _parse_football_data(resp.json(), league_key)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def fetch_fixtures(league_key: str, date: str, provider: str = "auto") -> List[Fixture]:
    """Fetch one league's fixtures for a date.

    ``provider`` is "espn", "football-data", or "auto" (prefer football-data
    when a token is set for its higher quality, else ESPN).
    """
    if provider == "espn":
        return fetch_fixtures_espn(league_key, date)
    if provider == "football-data":
        return fetch_fixtures_football_data(league_key, date)

    # auto: football-data first if a token exists AND the league is covered,
    # otherwise ESPN.
    if CONFIG.football_data_token and LEAGUES.get(league_key, {}).get("fd_code"):
        try:
            return fetch_fixtures_football_data(league_key, date)
        except DataUnavailable:
            pass
    return fetch_fixtures_espn(league_key, date)


def fetch_all_fixtures(date: str, provider: str = "auto") -> List[Fixture]:
    """Best-effort fetch across all three target leagues for ``date``.

    Leagues that can't be fetched are skipped rather than failing the whole
    call. Raises ``DataUnavailable`` only if *every* league failed, so the
    caller can distinguish "no games today" from "no data source reachable".
    """
    out: List[Fixture] = []
    errors: List[str] = []
    for league_key in LEAGUES:
        try:
            out.extend(fetch_fixtures(league_key, date, provider=provider))
        except DataUnavailable as exc:
            errors.append(f"{league_key}: {exc}")
    if not out and errors:
        raise DataUnavailable("; ".join(errors))
    return out
