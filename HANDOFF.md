# Handoff — run BetPredictor locally

Copy the prompt below into Claude Code (or any AI coding assistant) running on
your own machine. It carries everything needed to pick the project up, verify
live data fetching (which never ran in the cloud sandbox), and finish the value
tooling.

> **Why this exists:** the project was built in a cloud environment whose network
> blocked all sports sites and odds APIs, so live fetching was only unit-tested,
> never actually run. On a normal network it should work for real — that's the
> first thing to verify locally.

---

## Prompt

```
I have a football betting-prediction project on GitHub I want to run locally and
finish. Repo: 0x28a4/betpredictor, branch: claude/football-betting-predictions-pcgor6.

WHAT IT IS
A "Draw or Over 2.5 goals" predictor for English leagues (Premier League,
Championship, League One, League Two). It uses a Dixon-Coles bivariate-Poisson
model plus a logistic-regression blender and a feedback loop. The core
(src/betpredictor: models/, markets.py, engine.py, feedback.py, storage.py,
cli.py) is pure Python standard library — no third-party deps. FastAPI + requests
are only for the optional web UI and live data fetching. There's a standalone
browser page at web/artifact/index.html and a FastAPI app at web/app.py.

IMPORTANT CONTEXT
The project was built in a cloud sandbox whose network blocked all sports sites
and APIs, so live fetching was never actually exercised there — only unit-tested
against a captured payload. On my local machine the network is open, so the live
fetch should work for real. That's the main thing I need you to verify.

WHAT I WANT YOU TO DO
1. Clone the repo and check out that branch. Read the README.
2. Create a venv and `pip install -r requirements.txt`.
3. Run the tests: `python -m unittest discover -s tests` (should be ~33 passing).
4. Verify LIVE fetching works (this is the key step that never ran in the cloud):
   `PYTHONPATH=src python -m betpredictor.cli fetch --date <TODAY>`
   It should pull real fixtures from ESPN's public API (no key needed:
   eng.1=PL, eng.2=Championship, eng.3=League One, eng.4=League Two) and print
   the Draw-or-Over-2.5 screen. If it errors, diagnose and fix the fetch in
   src/betpredictor/data/sources.py.
5. Calibrate team strengths from real recent results so predictions aren't just
   priors: `PYTHONPATH=src python -m betpredictor.cli calibrate --start <~4 weeks ago> --end <yesterday>`
   then re-run the fetch/screen.
6. Start the web UI (`PYTHONPATH=src uvicorn web.app:app --reload`) and confirm it
   shows today's real fetched fixtures.

THEN, TWO IMPROVEMENTS I WANT
A. A "value" mode: accept a football-data.co.uk odds CSV (columns like AvgH/AvgD/
   AvgA, Avg>2.5/Avg<2.5, plus closing "C" columns), calibrate the Dixon-Coles
   model to the de-vigged market odds, and report each fixture's fair
   P(Draw or Over 2.5) and fair odds. Flag value only where a real best price
   beats fair — and REJECT outlier prices (any "best" >10% above the average is a
   data error, not value). Add unit tests for the odds parsing and the de-vig.
B. Wire the web UI so I can type a bookmaker's price for a fixture and it tells me
   instantly whether it beats fair value.

CONSTRAINTS
- Keep the core stdlib-only; keep everything tested before committing.
- This is for personal use. Add/keep clear responsible-gambling notices; the tool
  estimates probabilities, it does not promise winners, and value flags must be
  honest (no dressing up market-efficient bets as edges).
- Commit to the same branch with clear messages; don't open a PR unless I ask.
```

---

## Quick reference (once you're local)

```bash
git clone https://github.com/0x28a4/betpredictor.git
cd betpredictor
git checkout claude/football-betting-predictions-pcgor6

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m unittest discover -s tests                     # ~33 tests
PYTHONPATH=src python -m betpredictor.cli fetch --date 2026-09-08   # real card via ESPN
PYTHONPATH=src uvicorn web.app:app --reload              # web UI at :8000
```

## Honest caveats to keep in mind
- Any fixtures produced in the cloud session were from unreliable web search —
  trust the live `fetch` (or your bookmaker's own card) instead.
- The model estimates probabilities; it does not promise winners. On real market
  prices the "Draw or Over 2.5" / Over 2.5 markets showed **no positive value** —
  don't expect an edge the market doesn't leave. Bet only what you can lose.
  18+ · BeGambleAware.org · GamCare 0808 8020 133.
