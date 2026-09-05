"""Estimate team attack/defence strengths from historical results.

This is the "integrate historical data" half of the engine. Given a set of
finished matches (from any provider), it computes each team's attacking and
defensive strength relative to its league average, with **shrinkage** toward
1.0 so teams with few games aren't over-fit. The result plugs straight into
``PredictionEngine(strengths=...)``.

Pure function over ``Fixture``-like objects, so it's fully unit-testable
without network access.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from .sample_data import normalize_team_name

# Prior sample size for shrinkage: with SHRINKAGE "virtual" league-average
# games mixed in, a team needs a few real games before its strength moves much.
SHRINKAGE = 5.0
# How many recent games count toward the form (points-per-game) figure.
FORM_WINDOW = 6


def estimate_strengths(
    finished: Sequence[object],
    shrinkage: float = SHRINKAGE,
) -> Dict[str, Dict[str, object]]:
    """Return {team: {attack, defense, form, league}} from finished fixtures.

    ``finished`` items need ``.league, .home, .away, .home_goals, .away_goals``
    with integer goals (unfinished games are ignored).
    """
    # Group goals by league to compute per-league averages.
    league_games: Dict[str, List[object]] = defaultdict(list)
    for f in finished:
        if getattr(f, "home_goals", None) is None or getattr(f, "away_goals", None) is None:
            continue
        league_games[f.league].append(f)

    strengths: Dict[str, Dict[str, object]] = {}

    for league, games in league_games.items():
        # League-average goals scored per team per game.
        total_goals = sum(g.home_goals + g.away_goals for g in games)
        league_avg = total_goals / (2 * len(games)) if games else 1.35
        if league_avg <= 0:
            league_avg = 1.35

        scored: Dict[str, float] = defaultdict(float)
        conceded: Dict[str, float] = defaultdict(float)
        played: Dict[str, int] = defaultdict(int)
        # (match_index, team) -> points, to derive recent form in date order.
        history: Dict[str, List[int]] = defaultdict(list)

        for g in games:
            h = normalize_team_name(g.home)
            a = normalize_team_name(g.away)
            scored[h] += g.home_goals
            conceded[h] += g.away_goals
            scored[a] += g.away_goals
            conceded[a] += g.home_goals
            played[h] += 1
            played[a] += 1
            if g.home_goals > g.away_goals:
                history[h].append(3); history[a].append(0)
            elif g.home_goals < g.away_goals:
                history[h].append(0); history[a].append(3)
            else:
                history[h].append(1); history[a].append(1)

        for team, n in played.items():
            raw_attack = (scored[team] / n) / league_avg if n else 1.0
            raw_defense = (conceded[team] / n) / league_avg if n else 1.0
            # Shrink toward league average (1.0) by the virtual sample size.
            attack = (n * raw_attack + shrinkage * 1.0) / (n + shrinkage)
            defense = (n * raw_defense + shrinkage * 1.0) / (n + shrinkage)
            recent = history[team][-FORM_WINDOW:]
            form = (sum(recent) / len(recent)) if recent else 1.4
            strengths[team] = {
                "attack": round(attack, 4),
                "defense": round(defense, 4),
                "form": round(form, 3),
                "league": league,
                "games": n,
            }

    return strengths
