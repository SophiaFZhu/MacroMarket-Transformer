# sophia-macro-ai-spy

AI Macro-to-Market Research System — does market-implied expectations from
Polymarket, combined with U.S. macro data and market data, add any
information for forecasting SPY? Built as the capstone project from
`../Github Project 1/Sophia_AI_Foundations_GitHub_Roadmap_Updated.pdf`
(Part VI).

**This is a learning project, not investment advice.** The goal is not to
produce a winning strategy — it's to run a full, honest research pipeline
(data → features → baselines → Transformer → backtest → review) and report
whatever the data actually shows, including a negative result.

## Research question

> Does Polymarket expectation data (Fed cuts/hikes, inflation, recession
> odds) add forecasting information for SPY beyond realized macro data
> alone — and does a Transformer add value beyond simpler baselines?

## Status

Phase 1 (data collection) in progress. Building the project end-to-end
first, with the PDF roadmap as the explanatory syllabus alongside each
piece of real code — see "Current phase" below. This README will be
updated as each phase lands, and the final version should report actual
baseline-vs-Transformer results per the roadmap's completion checklist.

## Current phase

**Phase 1 — data collection.** Working, real (not placeholder) clients so
far:

- `src/data/market_client.py` — SPY + VIX daily OHLCV via yfinance. Tested,
  pulls ~2,900 days of history, no API key needed.
- `src/data/polymarket_client.py` — searches Polymarket events (e.g. "Fed
  Decision in September?") and pulls daily probability history for a
  market's outcome token via the CLOB API. Tested and working, but only
  finds *currently active* events so far — pulling the full historical
  series back to whenever Polymarket had liquid Fed/CPI/recession markets
  is still open (see Limitations).
- `src/data/fred_client.py` — CPI, PPI, unemployment, payrolls, jobless
  claims, Fed funds rate, 2Y/10Y Treasury yields, all via one free FRED API
  key (FRED mirrors the BLS series too, so a separate BLS key isn't needed
  yet). Code is done; blocked on a `FRED_API_KEY` in `.env`.

`week1_basics/` still has the NumPy warm-up (now fixed and runs cleanly) as
a reference, but isn't the main path anymore — the plan is to build the
real pipeline and learn the underlying math/CS concepts from that working
code as we go, phase by phase, rather than solving exercises first.

**Known limitation to fix before Phase 2:** Polymarket search above only
surfaces open/active markets. Need to pull closed historical events too
(paginating `public-search` with `events_status` unset, or Polymarket's
older markets by tag) to get a real historical time series instead of just
the last few weeks.

Everything else below is still a `# TODO` placeholder file, filled in as
we reach that phase.

| Phase | Roadmap week(s) | What it covers |
|---|---|---|
| 1. Define question & data sources | 1-3 | Four data groups, prediction target, horizons |
| 2. Time alignment & vintage data | 3 | Release-date alignment, no look-ahead bias |
| 3. SQL + feature engineering | 3 | `sql/schema.sql`, daily feature vector `X_t` |
| 4. Baselines → LSTM → Transformer | 4, 6 | Model ladder in `src/models/` |
| 5. Multi-agent workflow & validation | 8 | Agents in `src/agents/`, walk-forward backtest |

## Data sources (planned)

- **Polymarket** — Fed cuts/hikes, inflation, recession, unemployment, GDP
  expectation markets
- **FRED** — Fed Funds Rate, 2Y/10Y Treasury, CPI, PPI
- **BLS** — Unemployment rate, payrolls, jobless claims, JOLTS, wage growth
- **SPY market data** — OHLCV, realized volatility, VIX

## Key principle: no look-ahead bias

The feature table for date `t` may only use information that was actually
public by date `t`. Macro series have an `observation_date` (the period the
number describes) and a separate `release_date` (when it became public) —
`sql/schema.sql` keeps these distinct on purpose. A model trained on leaked
future information is scientifically invalid, not just "optimistic."

## Repository structure

```
data/raw/{polymarket,fred,bls,spy}/   # untouched source data
data/processed/                       # aligned, leakage-safe feature tables
src/data/                             # API clients (fred, bls, polymarket, market)
src/features/                         # macro / market / prediction-market features
src/models/                           # baseline, logistic, lstm, transformer
src/validation/                       # walk-forward split, metrics, backtest
src/agents/                           # orchestrator + specialized research agents
src/database/                         # SQLite access layer
sql/schema.sql                        # raw observations vs. processed features
notebooks/                            # exploration notebooks, one per phase
tests/                                # unit tests for data/tool logic
reports/                              # backtest results, findings write-ups
week1_basics/                         # roadmap Week 1 warm-up exercises
```

## Setup

```bash
cd sophia-macro-ai-spy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in FRED_API_KEY etc.
```

No API keys are required yet — nothing calls out to FRED/BLS/Polymarket
until Phase 1 data collection starts.

## What I learned

(To be filled in once there's something to report — data leakage findings,
baseline vs. Transformer comparison, whether Polymarket data added
incremental information, and research limitations.)
