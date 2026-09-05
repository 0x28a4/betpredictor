"""Offline sample dataset so the engine runs with no API key or network.

IMPORTANT / HONESTY NOTE
------------------------
The team strengths and the 5 Sep 2026 fixture list below are *illustrative
priors*, not verified live data. They let the engine, tests, web UI and demo
run deterministically offline. For real predictions, fetch live fixtures and
recent results via ``betpredictor.data.sources`` with an API token; the engine
will then estimate strengths from actual match data instead of these seeds.

Strength convention
-------------------
- ``attack``  : goals-scored multiplier vs. league average (1.0 = average).
- ``defense`` : goals-CONCEDED multiplier (LOWER is a better defence).
- ``form``    : recent points-per-game, 0..3.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Average goals per side, per league (home+away halves before adjustments).
LEAGUE_BASE_GOALS: Dict[str, float] = {
    "premier-league": 1.40,
    "championship": 1.30,
    "league-one": 1.30,
    "league-two": 1.30,
}

# team -> attributes. ``league`` links a team to its base scoring rate.
TEAM_STRENGTHS: Dict[str, Dict[str, object]] = {
    # --- Premier League ---
    "Manchester City": {"attack": 1.55, "defense": 0.62, "form": 2.4, "league": "premier-league"},
    "Tottenham Hotspur": {"attack": 1.30, "defense": 0.95, "form": 1.8, "league": "premier-league"},
    "Newcastle United": {"attack": 1.20, "defense": 0.85, "form": 1.8, "league": "premier-league"},
    "Brighton & Hove Albion": {"attack": 1.18, "defense": 0.95, "form": 1.7, "league": "premier-league"},
    "Nottingham Forest": {"attack": 1.05, "defense": 0.95, "form": 1.6, "league": "premier-league"},
    "AFC Bournemouth": {"attack": 1.05, "defense": 1.05, "form": 1.5, "league": "premier-league"},
    "Brentford": {"attack": 1.05, "defense": 1.02, "form": 1.4, "league": "premier-league"},
    "Fulham": {"attack": 1.02, "defense": 1.00, "form": 1.5, "league": "premier-league"},
    "Crystal Palace": {"attack": 1.00, "defense": 0.90, "form": 1.5, "league": "premier-league"},
    "Leeds United": {"attack": 0.95, "defense": 1.12, "form": 1.3, "league": "premier-league"},
    "Sunderland": {"attack": 0.85, "defense": 1.10, "form": 1.2, "league": "premier-league"},
    "Coventry City": {"attack": 0.85, "defense": 1.15, "form": 1.2, "league": "premier-league"},

    # --- EFL Championship ---
    "Birmingham City": {"attack": 1.15, "defense": 0.90, "form": 1.8, "league": "championship"},
    "Burnley": {"attack": 1.10, "defense": 0.85, "form": 1.7, "league": "championship"},
    "Wolverhampton Wanderers": {"attack": 1.05, "defense": 1.00, "form": 1.4, "league": "championship"},
    "Bristol City": {"attack": 1.00, "defense": 1.00, "form": 1.4, "league": "championship"},
    "Blackburn Rovers": {"attack": 1.00, "defense": 1.02, "form": 1.4, "league": "championship"},
    "Swansea City": {"attack": 1.00, "defense": 1.05, "form": 1.3, "league": "championship"},
    "Watford": {"attack": 1.00, "defense": 1.05, "form": 1.3, "league": "championship"},
    "Millwall": {"attack": 0.95, "defense": 0.90, "form": 1.4, "league": "championship"},
    "Stoke City": {"attack": 0.95, "defense": 1.00, "form": 1.3, "league": "championship"},
    "Preston North End": {"attack": 0.90, "defense": 0.95, "form": 1.3, "league": "championship"},

    # --- EFL League One ---
    "Bolton Wanderers": {"attack": 1.15, "defense": 0.90, "form": 1.7, "league": "league-one"},
    "Peterborough United": {"attack": 1.20, "defense": 1.00, "form": 1.5, "league": "league-one"},
    "Huddersfield Town": {"attack": 1.10, "defense": 0.95, "form": 1.5, "league": "league-one"},
    "Stockport County": {"attack": 1.05, "defense": 0.95, "form": 1.5, "league": "league-one"},
    "Barnsley": {"attack": 1.05, "defense": 1.00, "form": 1.4, "league": "league-one"},
    "Wigan Athletic": {"attack": 1.00, "defense": 1.00, "form": 1.4, "league": "league-one"},
    "Reading": {"attack": 1.00, "defense": 1.00, "form": 1.4, "league": "league-one"},
    "Lincoln City": {"attack": 1.00, "defense": 1.00, "form": 1.4, "league": "league-one"},
    "Blackpool": {"attack": 1.00, "defense": 1.02, "form": 1.3, "league": "league-one"},
    "Sheffield Wednesday": {"attack": 0.95, "defense": 1.05, "form": 1.2, "league": "league-one"},
}

# Illustrative fixtures for Sat 5 Sep 2026 (league, home, away). NOT verified
# live data -- see the module docstring. Replace via a live fetch for real use.
FIXTURES_2026_09_05: List[Tuple[str, str, str]] = [
    # Premier League
    ("premier-league", "Newcastle United", "AFC Bournemouth"),
    ("premier-league", "Brentford", "Sunderland"),
    ("premier-league", "Brighton & Hove Albion", "Leeds United"),
    ("premier-league", "Fulham", "Crystal Palace"),
    ("premier-league", "Manchester City", "Coventry City"),
    ("premier-league", "Nottingham Forest", "Tottenham Hotspur"),
    # Championship
    ("championship", "Birmingham City", "Wolverhampton Wanderers"),
    ("championship", "Burnley", "Bristol City"),
    ("championship", "Watford", "Stoke City"),
    ("championship", "Preston North End", "Blackburn Rovers"),
    ("championship", "Millwall", "Swansea City"),
    # League One
    ("league-one", "Peterborough United", "Sheffield Wednesday"),
    ("league-one", "Wigan Athletic", "Stockport County"),
    ("league-one", "Bolton Wanderers", "Barnsley"),
    ("league-one", "Reading", "Blackpool"),
    ("league-one", "Huddersfield Town", "Lincoln City"),
]


def default_strength(league: str) -> Dict[str, object]:
    """Fallback attributes for an unknown team (league-average everything)."""
    return {"attack": 1.0, "defense": 1.0, "form": 1.4, "league": league}


# Map the short/alternate names live providers (ESPN in particular) use onto the
# canonical names in TEAM_STRENGTHS, so live fixtures still hit seed strengths.
TEAM_ALIASES: Dict[str, str] = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton & Hove Albion",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "Man City": "Manchester City",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Weds": "Sheffield Wednesday",
    "Sheffield Wed": "Sheffield Wednesday",
    "Preston": "Preston North End",
    "Blackburn": "Blackburn Rovers",
    "Bolton": "Bolton Wanderers",
    "Wigan": "Wigan Athletic",
    "Stockport": "Stockport County",
    "Huddersfield": "Huddersfield Town",
    "Lincoln": "Lincoln City",
    "Peterborough": "Peterborough United",
    "Newcastle": "Newcastle United",
    "Leeds": "Leeds United",
    "Birmingham": "Birmingham City",
    "Bristol City FC": "Bristol City",
    "Coventry": "Coventry City",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
}


def normalize_team_name(name: str) -> str:
    """Canonicalise a team name; trims a trailing ' FC' and applies aliases."""
    if not name:
        return name
    n = name.strip()
    if n in TEAM_ALIASES:
        return TEAM_ALIASES[n]
    if n.endswith(" FC"):
        base = n[:-3].strip()
        return TEAM_ALIASES.get(base, base)
    return n
