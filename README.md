# BetPredictor ⚽📈

An AI-driven system for predicting English football match outcomes, built
around the **"Draw or Over 2.5 goals"** market for the Premier League, EFL
Championship and EFL League One.

It combines a **mathematical model** (Dixon-Coles bivariate Poisson) with a
**machine-learning blender** and a **feedback loop** that learns from realised
results and user insight — served through a **CLI** and a **web UI**.

> ⚠️ **Responsible gambling.** These are statistical estimates, **not
> guarantees**. Football is high-variance; any prediction can be wrong and you
> can lose money. Only stake what you can afford to lose. **18+** ·
> [BeGambleAware.org](https://www.begambleaware.org) · GamCare 0808 8020 133.

---

## Why "Draw or Over 2.5 goals"?

It's a combination bet that wins if **either** the match is a draw **or** it has
**3+ goals**. Usefully, it loses on **exactly four scorelines**:

```
1-0   2-0   0-1   0-2      (a decisive result with ≤ 2 goals)
```

So `P(draw or over 2.5) = 1 − P(1-0) − P(2-0) − P(0-1) − P(0-2)`. The engine
computes every one of those cells exactly from the scoreline matrix.

---

## How it works

```
 team strengths + Elo ratings
          │
          ▼
   expected goals  λ (home), μ (away)        ← home advantage, form, Elo tilt
          │
          ▼
   Dixon-Coles bivariate Poisson             ← low-score dependence correction
   → full scoreline probability matrix
          │
          ▼
   market probabilities                      ← 1X2, Over/Under 2.5, BTTS,
   (incl. Draw-or-Over-2.5)                     Draw-or-Over-2.5
          │
          ▼
   ML blender (logistic regression)          ← learns systematic corrections
   → final probability + confidence          ← ensemble weight auto-tuned
          │
          ▼
   recommendation + value vs. book odds      ← edge = p·odds − 1
          │
          ▼
   feedback loop  ◄── realised results + user ratings/comments
   (Elo update · ML retrain · Brier/log-loss/ROI scoring)
```

**Design choice:** the entire core (`models`, `markets`, `engine`, `feedback`,
`storage`) uses **only the Python standard library** — no numpy, no
scikit-learn — so it runs anywhere and every piece is unit-tested. FastAPI and
`requests` are needed only for the optional web UI and live-data fetching.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # only needed for web UI + live fetch
```

The CLI and tests work with **no dependencies at all**.

## Quick start (offline, no API key)

```bash
# Screen today's sample slate for Draw-or-Over-2.5 picks
PYTHONPATH=src python -m betpredictor.cli screen

# Predict one fixture, with optional book odds for a value/edge read
PYTHONPATH=src python -m betpredictor.cli predict "Burnley" "Bristol City" \
    --league championship --odds 1.30

# Seed the feedback loop with a simulated backtest, then score it
PYTHONPATH=src python scripts/demo_backtest.py
PYTHONPATH=src python -m betpredictor.cli evaluate --odds 1.30
```

## Web UI

Two front-ends ship with the project:

**1. Standalone page (zero setup).** `web/artifact/index.html` is a single,
self-contained page that runs the whole Dixon-Coles engine in the browser with
today's slate loaded in — league filters, a probability slider, per-fixture
market breakdowns, a "match lab" custom predictor with value/edge, and a
feedback note pad (saved to `localStorage`). Just open the file, or view the
hosted copy:
<https://claude.ai/code/artifact/72ae1294-9ad1-405e-8ef6-c2f10e75e870>

**2. Full app (with the learning loop).** The FastAPI backend serves the
ranked slate, live fetch, a performance panel and a feedback form wired to the
SQLite feedback loop:

```bash
PYTHONPATH=src uvicorn web.app:app --reload
# open http://127.0.0.1:8000
```

## Live data (real fixtures) — no API key needed

The offline dataset is only for demos. For **real** fixtures the engine uses
**ESPN's public JSON API by default — no key, and it covers all three target
leagues** (`eng.1` Premier League, `eng.2` Championship, `eng.3` League One):

```bash
# Real fixtures for a date, screened for Draw-or-Over-2.5 (no key required)
PYTHONPATH=src python -m betpredictor.cli fetch --date 2026-09-05
```

Optionally add football-data.org for higher-quality data on PL + Championship;
when a token is present it's preferred automatically for those leagues:

```bash
export FOOTBALL_DATA_TOKEN=your_free_key   # https://www.football-data.org/
PYTHONPATH=src python -m betpredictor.cli fetch --date 2026-09-05 --provider auto
```

Providers live in `data/sources.py` behind one `Fixture` shape; the JSON
parsers are pure functions with unit tests, so provider changes never touch the
engine. (Note: the parser is exercised by tests offline; a sandbox with locked
egress will 403 on the live call — run it where outbound HTTPS is allowed.)

### Calibrate strengths from real history

Instead of the seed priors, estimate each team's attack/defence from **actual
finished results** over a date range (Dixon-Coles strengths with shrinkage).
The calibrated strengths are saved and used for all future predictions:

```bash
PYTHONPATH=src python -m betpredictor.cli calibrate --start 2026-08-08 --end 2026-09-01
PYTHONPATH=src python -m betpredictor.cli fetch --date 2026-09-05   # now data-driven
```

## Feedback loop in practice

```bash
# Record a real result — settles the pick and updates Elo
PYTHONPATH=src python -m betpredictor.cli result 2026-09-05 "Burnley" "Bristol City" 2 1

# Add user insight
PYTHONPATH=src python -m betpredictor.cli feedback --rating 4 \
    --comment "both sides missing first-choice keeper" \
    --date 2026-09-05 --home "Burnley" --away "Bristol City"

# Once enough results accumulate, refit the ML blender + re-tune the weight
PYTHONPATH=src python -m betpredictor.cli retrain
```

Metrics tracked: **hit rate**, **Brier score**, **log-loss** (calibration) and
**flat-stake ROI** at your average odds.

## Tests

```bash
python -m unittest discover -s tests      # 23 tests, stdlib only
# or, if you have pytest:  pytest -q
```

---

## Project layout

```
src/betpredictor/
  config.py            tunable parameters (home adv, rho, ML weight, thresholds)
  markets.py           market probabilities from a scoreline matrix
  engine.py            PredictionEngine: fixtures → predictions → picks
  feedback.py          learning loop: results, scoring, retraining
  storage.py           SQLite persistence (predictions/results/feedback/state)
  cli.py               command-line interface
  models/
    poisson.py         Dixon-Coles bivariate Poisson
    elo.py             results-driven Elo ratings
    ml.py              from-scratch logistic-regression blender
  data/
    sources.py         ESPN (no-key) + football-data.org live fetch
    calibrate.py       estimate team strengths from finished results
    sample_data.py     offline strengths, fixtures, name normalisation
web/                   FastAPI app + single-page UI
scripts/demo_backtest.py   simulated backtest to seed the loop
tests/                 unittest suite
```

## Honesty & limitations

- The bundled team strengths and the 5 Sep 2026 fixture list are **illustrative
  priors**, not verified live data — they make the demo deterministic. Use
  `fetch` (real fixtures via ESPN) and `calibrate` (real strengths) for genuine
  predictions.
- No model beats the closing line reliably. Treat outputs as **one input** to
  your own judgement, compare against real market odds (the value/edge tool
  helps), and never chase losses.
- Predicted probabilities are only as good as the input data and calibration;
  watch the Brier/log-loss in `evaluate` before trusting the picks.
