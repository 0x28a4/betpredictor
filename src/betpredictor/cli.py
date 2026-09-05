"""Command-line interface for BetPredictor.

Examples
--------
    python -m betpredictor.cli screen                 # today's sample slate
    python -m betpredictor.cli screen --threshold 0.6
    python -m betpredictor.cli predict "Man City" "Coventry City" --league premier-league
    python -m betpredictor.cli fetch --date 2026-09-05    # real fixtures (ESPN, no key)
    python -m betpredictor.cli calibrate --start 2026-08-08 --end 2026-09-01
    python -m betpredictor.cli result 2026-09-05 "Burnley" "Bristol City" 2 1
    python -m betpredictor.cli evaluate
    python -m betpredictor.cli retrain
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

from .config import LEAGUES
from .data.sample_data import FIXTURES_2026_09_05
from .engine import Prediction, PredictionEngine
from .feedback import FeedbackLoop

DISCLAIMER = (
    "Model estimates only -- not a guarantee. No bet is 'safe'. "
    "Gamble responsibly. 18+. BeGambleAware.org / GamCare 0808 8020 133."
)


def _fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _print_predictions(preds: List[Prediction], title: str) -> None:
    print(f"\n{title}")
    print("-" * 88)
    print(
        f"{'Fixture':<44}{'D or O2.5':>10}{'Over2.5':>9}{'Draw':>7}{'xG':>7}  Conf"
    )
    print("-" * 88)
    for p in preds:
        fixture = f"{p.home} v {p.away}"
        xg = f"{p.exp_home_goals + p.exp_away_goals:.2f}"
        star = " *" if p.recommended else "  "
        print(
            f"{fixture:<44}{_fmt_pct(p.p_draw_or_over_2_5):>10}"
            f"{_fmt_pct(p.p_over_2_5):>9}{_fmt_pct(p.p_draw):>7}{xg:>7}  "
            f"{p.confidence}{star}"
        )
    print("-" * 88)
    print("* = meets recommendation threshold")


def _get_engine() -> PredictionEngine:
    engine = PredictionEngine()
    # Load any learned state (Elo/ML/weight) from the feedback DB if present.
    FeedbackLoop(engine)
    return engine


def cmd_screen(args: argparse.Namespace) -> int:
    engine = _get_engine()
    fixtures = FIXTURES_2026_09_05
    if args.league != "all":
        fixtures = [f for f in fixtures if f[0] == args.league]
    picks = engine.screen(fixtures, threshold=args.threshold)
    _print_predictions(picks, f"Draw-or-Over-2.5 screen (threshold {args.threshold:.0%})")
    print(f"\n{len(picks)} fixture(s) clear the threshold.")
    print(DISCLAIMER)
    return 0


def cmd_today(args: argparse.Namespace) -> int:
    engine = _get_engine()
    preds = engine.predict_all(FIXTURES_2026_09_05)
    _print_predictions(preds, "All fixtures (sample slate), ranked by Draw-or-Over-2.5")
    print(DISCLAIMER)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    engine = _get_engine()
    p = engine.predict(args.league, args.home, args.away, odds=args.odds)
    print(f"\n{p.home} v {p.away}  ({LEAGUES.get(p.league, {}).get('name', p.league)})")
    print(f"  Expected goals : {p.exp_home_goals:.2f} - {p.exp_away_goals:.2f} "
          f"(total {p.exp_home_goals + p.exp_away_goals:.2f})")
    print(f"  1X2            : Home {_fmt_pct(p.p_home)}  Draw {_fmt_pct(p.p_draw)}  Away {_fmt_pct(p.p_away)}")
    print(f"  Over 2.5       : {_fmt_pct(p.p_over_2_5)}   BTTS {_fmt_pct(p.p_btts)}")
    print(f"  DRAW or OVER2.5: {_fmt_pct(p.p_draw_or_over_2_5)}  ->  {p.confidence}"
          f"{'  [RECOMMENDED]' if p.recommended else ''}")
    if p.odds:
        edge_txt = f"{p.edge * 100:+.1f}%" if p.edge is not None else "n/a"
        fair = f"  fair odds {p.fair_odds:.2f}" if p.fair_odds else ""
        value = "  <- VALUE" if (p.edge is not None and p.edge > 0) else ""
        print(f"  Odds {p.odds:.2f} (implied {_fmt_pct(p.implied_prob)})  ->  "
              f"edge {edge_txt}{fair}{value}")
    print(DISCLAIMER)
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .data.sources import fetch_all_fixtures, DataUnavailable
    try:
        fixtures = fetch_all_fixtures(args.date, provider=args.provider)
    except DataUnavailable as exc:
        print(f"Live fetch unavailable: {exc}", file=sys.stderr)
        print("Tip: ESPN (default) needs no key. Check network access and the date.",
              file=sys.stderr)
        return 2
    if not fixtures:
        print(f"No fixtures found for {args.date} in these leagues (rest day?).")
        return 0
    srcs = sorted({f.source for f in fixtures})
    print(f"Fetched {len(fixtures)} fixture(s) for {args.date} via {', '.join(srcs)}.")
    engine = _get_engine()
    slate: List[Tuple[str, str, str]] = [(f.league, f.home, f.away) for f in fixtures]
    picks = engine.screen(slate, threshold=args.threshold)
    _print_predictions(picks, f"Live Draw-or-Over-2.5 screen for {args.date}")
    print(f"\n{len(picks)} of {len(fixtures)} fixture(s) clear the threshold.")
    print(DISCLAIMER)
    return 0


def _date_range(start: str, end: str):
    from datetime import date, timedelta
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    step = timedelta(days=1)
    cur = d0
    while cur <= d1:
        yield cur.isoformat()
        cur += step


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Estimate team strengths from finished matches over a date range."""
    from .data.sources import fetch_all_fixtures, DataUnavailable
    from .data.calibrate import estimate_strengths

    finished = []
    n_days = 0
    for day in _date_range(args.start, args.end):
        n_days += 1
        try:
            for f in fetch_all_fixtures(day, provider=args.provider):
                if f.home_goals is not None and f.away_goals is not None:
                    finished.append(f)
        except DataUnavailable:
            continue
    if not finished:
        print("No finished matches fetched for that range — nothing to calibrate.",
              file=sys.stderr)
        return 2

    strengths = estimate_strengths(finished)
    engine = _get_engine()
    loop = FeedbackLoop(engine)
    loop.save_strengths(strengths)
    print(f"Calibrated {len(strengths)} teams from {len(finished)} finished "
          f"matches over {n_days} day(s). Strengths saved and will be used for "
          f"future predictions.")
    # Show a few extremes as a sanity check.
    ranked = sorted(strengths.items(), key=lambda kv: kv[1]["attack"], reverse=True)
    print("\nTop attacks:")
    for team, s in ranked[:5]:
        print(f"  {team:<28} attack {s['attack']:.2f}  defense {s['defense']:.2f}  ({s['games']} games)")
    return 0


def cmd_result(args: argparse.Namespace) -> int:
    engine = _get_engine()
    loop = FeedbackLoop(engine)
    hit = loop.record_result(args.date, args.home, args.away, args.home_goals, args.away_goals)
    print(f"Recorded {args.home} {args.home_goals}-{args.away_goals} {args.away}: "
          f"Draw-or-Over-2.5 {'HIT' if hit else 'MISS'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    engine = _get_engine()
    loop = FeedbackLoop(engine)
    perf = loop.performance(avg_odds=args.odds)
    d = perf.as_dict()
    print("\nModel performance on settled predictions")
    print("-" * 44)
    for k, v in d.items():
        print(f"  {k:<22}: {v}")
    if perf.n == 0:
        print("  (no settled results yet -- record some with `result`)")
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    engine = _get_engine()
    loop = FeedbackLoop(engine)
    out = loop.retrain(min_samples=args.min_samples)
    print("Retrain:", out)
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    engine = _get_engine()
    loop = FeedbackLoop(engine)
    fid = loop.record_user_feedback(
        rating=args.rating, comment=args.comment,
        match_date=args.date, home=args.home, away=args.away,
    )
    print(f"Thanks -- feedback #{fid} recorded.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="betpredictor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("screen", help="screen the sample slate for Draw-or-Over-2.5")
    p.add_argument("--league", default="all", choices=["all", *LEAGUES.keys()])
    p.add_argument("--threshold", type=float, default=0.75)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("today", help="rank all sample fixtures")
    p.set_defaults(func=cmd_today)

    p = sub.add_parser("predict", help="predict a single fixture")
    p.add_argument("home")
    p.add_argument("away")
    p.add_argument("--league", default="premier-league", choices=list(LEAGUES.keys()))
    p.add_argument("--odds", type=float, default=None, help="bookmaker decimal odds for value calc")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("fetch", help="fetch live fixtures and screen (ESPN, no key)")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--provider", default="auto", choices=["auto", "espn", "football-data"])
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("calibrate", help="estimate team strengths from finished results")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--provider", default="auto", choices=["auto", "espn", "football-data"])
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("result", help="record a realised result (feedback loop)")
    p.add_argument("date")
    p.add_argument("home")
    p.add_argument("away")
    p.add_argument("home_goals", type=int)
    p.add_argument("away_goals", type=int)
    p.set_defaults(func=cmd_result)

    p = sub.add_parser("evaluate", help="show performance on settled predictions")
    p.add_argument("--odds", type=float, default=None, help="avg decimal odds for ROI")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("retrain", help="retrain ML blender + re-tune weight")
    p.add_argument("--min-samples", type=int, default=30)
    p.set_defaults(func=cmd_retrain)

    p = sub.add_parser("feedback", help="submit user feedback on a pick")
    p.add_argument("--rating", type=int, choices=[1, 2, 3, 4, 5], default=None)
    p.add_argument("--comment", default="")
    p.add_argument("--date", default=None)
    p.add_argument("--home", default=None)
    p.add_argument("--away", default=None)
    p.set_defaults(func=cmd_feedback)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
