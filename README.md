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

**Phase 1 — data collection: done.** All three sources pulling real data
into `data/raw/`:

- `src/data/market_client.py` — SPY + VIX daily OHLCV via yfinance.
  ~2,900 days, back to 2015, no API key needed.
- `src/data/polymarket_client.py` — stitches every FOMC meeting's "no rate
  change" market into one time series (`fetch_fed_decision_history`).
  2,066 rows across 15 meetings, back to the September 2024 meeting.
- `src/data/fred_client.py` — CPI, PPI, unemployment, payrolls, jobless
  claims, Fed funds rate, 2Y/10Y Treasury yields.

**Phase 2 — release-date / vintage alignment: done, for FRED.** FRED
series get revised after their first release (CPI, PPI, unemployment,
payrolls, jobless claims); pulling today's value and treating it as "what
the market knew back then" is look-ahead bias. `fred_client.py` now
queries FRED's full vintage history and keeps only each observation
period's *first* published value, reporting that vintage's publish date as
`release_date` — matching `sql/schema.sql`'s `macro_observations` table
exactly. Example: January 2026 CPI = 326.588, first published 2026-02-13;
later revisions to that same month exist in FRED but are correctly
ignored. Real lags recovered: CPI/jobless claims ≈ 2 weeks, PPI/payrolls/
unemployment up to ~3 months for the oldest (2015-era) observations —
current releases run faster (~2 weeks for CPI, per the tail of
`cpi.csv`). Fed funds rate and Treasury yields (`DFF`, `DGS2`, `DGS10`)
aren't meaningfully revised, so those just use `release_date ==
observation_date`.

Polymarket and SPY/VIX data need no such fix — a market price *is* the
market's information as of that timestamp, there's no "revision" concept.

`week1_basics/` still has the NumPy warm-up (fixed, runs cleanly) as a
reference, but isn't the main path — we're learning the underlying
math/CS/Python from the real pipeline as it's built, phase by phase.

## Known limitations (so far)

- **Polymarket history is short.** Liquid Fed-decision markets only go
  back to ~August 2024 (15 meetings), vs. 10+ years of SPY/FRED data.
  Any "does Polymarket add value" comparison is only valid over that
  overlapping window — this will need to be stated explicitly in any
  final results, not glossed over.
- **Earliest FRED vintages may overstate release lag.** ALFRED (FRED's
  vintage archive) only tracks revisions from whenever it started
  recording a series; for the oldest (2015) observations in some series,
  the "first tracked vintage" may not be the true original release date,
  which is why max release lag looks larger for old data than recent data.
  Worth double-checking against official BLS release calendars before
  trusting lag numbers for anything older than a few years.
- Only the "no rate change" market is tracked per meeting so far — the
  25bps/50bps cut and hike markets exist too and may be worth pulling for
  a richer signal than a single probability.

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
