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

**Results dashboard**: `docs/index.html` — open it directly in a browser,
or once this repo is pushed and GitHub Pages is enabled (Settings → Pages
→ Deploy from branch → `main` / `docs`), it's live at
`https://<your-username>.github.io/sophia-macro-ai-spy/`.

## Research question

> Does Polymarket expectation data (Fed cuts/hikes, inflation, recession
> odds) add forecasting information for SPY beyond realized macro data
> alone — and does a Transformer add value beyond simpler baselines?

## Status

All 5 roadmap phases done (data → vintage alignment → daily feature
matrix → model ladder → Polymarket comparison + backtest + agents). **The
project's actual research question is answered**: no model — with or
without Polymarket expectation data — beats a naive baseline in a way
that isn't noise, and every model-based backtest badly underperforms
buy-and-hold SPY over this period. See `reports/PHASE4_FINDINGS.md` and
`reports/PHASE5_FINDINGS.md`. Built end-to-end with the PDF roadmap as the
explanatory syllabus alongside each piece of real code — see "Current
phase" below for what exists and `reports/REVIEWER_REPORT.md` for the
automated review of it all.

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

**Phase 3 — SQL + daily feature matrix: done.** `src/database/database.py`
loads all raw data into `data/processed/research.db` per `sql/schema.sql`
(now with a `vix_prices` table and a richer `polymarket_prices` that keeps
`meeting_end_date`/`outcome` alongside each price, since feature
engineering needs to know which FOMC meeting a probability belongs to).
`src/features/{macro,market,prediction_market}_features.py` build each
feature group; `src/features/build_daily_features.py` as-of-joins them all
onto the SPY trading-day calendar into one `daily_features` table /
`data/processed/daily_features.csv` (2,939 rows, 2015-01-02 → today).

The leakage guard is `pd.merge_asof(..., direction="backward")` on
`release_date`, not a promise or a comment — verified empirically: July
2026 CPI (released 2026-08-12) is invisible in `cpi_yoy`/`cpi_surprise` on
2026-08-11 and appears exactly on 2026-08-12, nowhere earlier. Covered by
`tests/test_macro_features.py` (2 passing tests) so this can't silently
break later. Fed-cut probability sums the `cut_25` + `cut_50` outcome
buckets for whichever FOMC meeting is next as of that trading day (not
just "no change" anymore — extended `polymarket_client.py` to pull all 5
outcome buckets per meeting).

**Phase 4 — model ladder: done.** `src/models/{baseline,logistic,lstm,
transformer}.py` implement all 5 rungs (naive, persistence, logistic
regression, random forest, LSTM, small Transformer — random forest and
naive share `baseline.py`). `src/validation/walk_forward.py` is
expanding-window walk-forward with a purge gap (no training label's
future-return window overlaps the test period — covered by
`tests/test_walk_forward.py`, 3 passing tests). `src/validation/
run_model_ladder.py` runs every model through the same 5 folds and reports
accuracy/AUC/log loss.

**Result: no edge.** On the macro-only feature set (Polymarket excluded —
see below), every model is statistically indistinguishable from a
majority-class baseline; AUCs cluster at ~0.50. Full write-up:
`reports/PHASE4_FINDINGS.md`. This is treated as a real, reportable
finding, not a bug to chase away — matches this user's prior project
([`spy-garch-vwap`](../spy-garch-vwap/results/FINDINGS.md)), which found
the same kind of "no real edge under honest validation" result.

`week1_basics/` still has the NumPy warm-up (fixed, runs cleanly) as a
reference, but isn't the main path — we're learning the underlying
math/CS/Python from the real pipeline as it's built, phase by phase.

## Known limitations (so far)

- **Polymarket history is short.** Liquid Fed-decision markets only go
  back to ~August 2024 (15 meetings), vs. 10+ years of SPY/FRED data —
  `fed_cut_probability` is `NaN` for ~82% of rows (2,414 / 2,939) before
  that. Any "does Polymarket add value" comparison is only valid over the
  overlapping window — this will need to be stated explicitly in any final
  results, not glossed over.
- **Earliest FRED vintages may overstate release lag.** ALFRED (FRED's
  vintage archive) only tracks revisions from whenever it started
  recording a series; for the oldest (2015) observations in some series,
  the "first tracked vintage" may not be the true original release date,
  which is why max release lag looks larger for old data than recent data.
  Worth double-checking against official BLS release calendars before
  trusting lag numbers for anything older than a few years.
- **`cpi_surprise` is a proxy, not a real surprise metric.** A true
  "surprise" is actual-vs-consensus-forecast; we don't have consensus
  forecast data, so `cpi_surprise`/`cpi_yoy` are computed from
  month-over-month / year-over-year change in the first-released value
  itself. Documented in `macro_features.py`'s docstrings — don't present
  this as a real economic surprise metric without fixing this first.
  Similarly, `cpi_yoy`/`ppi_yoy` compare two first-release values, not two
  final-revision values — a defensible but non-standard convention.
- One 2025 FOMC meeting had an outsized "75+ bps" cut bucket instead of the
  usual "50+ bps" one; `polymarket_client.py`'s outcome classifier doesn't
  match it, so that meeting's cut probability slightly understates the
  true total for that one window.
- **No hyperparameter tuning was done against the evaluation folds**, in
  either Phase 4 or Phase 5 — model settings (RF depth, LSTM/Transformer
  size, epoch counts) are fixed reasonable defaults, not the result of a
  search. Tuning against the same folds used for the final reported
  comparison would itself be a form of look-ahead bias into the
  validation set (the roadmap's Reviewer Agent role exists partly to catch
  exactly this kind of thing — see `src/agents/reviewer_agent.py`).
- **The Phase 5 backtest includes no transaction costs.** Real trading
  would be worse than the reported `strategy_total_return` numbers,
  especially given ~64 trades over the ~2-year Polymarket-covered window.
  `spy-garch-vwap`'s 1% round-trip assumption is a reasonable precedent to
  apply here before treating any of this as remotely investable.

**Phase 5 — the actual research question + backtest + agents: done.**
`src/validation/run_polymarket_comparison.py` runs the macro-only vs.
macro+Polymarket feature comparison on the ~524-row Aug-2024-onward
overlap window (3 walk-forward folds; LSTM/Transformer skipped here —
too small a sample to trust a deep sequence model's result, and Phase 4
already showed complexity wasn't the bottleneck). `src/validation/
backtest.py` turns out-of-sample predictions into a simple long/flat
strategy vs. buy-and-hold, on a non-overlapping trade sequence (stride =
the 5-day horizon, so consecutive trades' return windows don't overlap).
`src/agents/{data_quality,reviewer}_agent.py` are real, runnable
automated-check modules (not LLM calls — see their docstrings for why),
and `src/agents/{macro,polymarket,market,modelling,backtest}_agent.py` +
`orchestrator.py` wire the whole pipeline into one entry point. Full
write-up: `reports/PHASE5_FINDINGS.md`.

**Result: still no evidence Polymarket data helps** — AUC for both
logistic regression and random forest was *slightly lower* with
`fed_cut_probability`/`_delta` added than without, though the sample here
(3 folds, ~106 rows each) is too small to call that a real negative
effect rather than noise. The backtest tells a consistent story: both
feature sets' long/flat strategies badly underperform buy-and-hold SPY
over this window (macro-only: 4.4% strategy return vs. 31.0%
buy-and-hold; macro+Polymarket: 6.0% vs. 31.0%) — being flat ~45-47% of
the time forfeits most of a strong bull run's upside. See
`reports/PHASE5_FINDINGS.md` for the full picture, including why this
doesn't contradict Phase 4's finding.

| Phase | Roadmap week(s) | What it covers |
|---|---|---|
| 1. Define question & data sources | 1-3 | Four data groups, prediction target, horizons |
| 2. Time alignment & vintage data | 3 | Release-date alignment, no look-ahead bias |
| 3. SQL + feature engineering | 3 | `sql/schema.sql`, daily feature vector `X_t` |
| 4. Baselines → LSTM → Transformer | 4, 6 | Model ladder in `src/models/` |
| 5. Multi-agent workflow & validation | 8 | Agents in `src/agents/`, walk-forward backtest |

## Data sources

- **Polymarket** — Fed cuts/hikes/no-change expectation markets, all
  meetings back to Sept 2024. *Not yet pulled*: inflation, recession,
  unemployment, GDP expectation markets from the roadmap's original scope.
- **FRED** — Fed Funds Rate, 2Y/10Y Treasury, CPI, PPI, unemployment rate,
  payrolls, jobless claims (FRED mirrors the BLS series, so no separate
  BLS client was needed)
- **SPY market data** — OHLCV, realized volatility, VIX

## Key principle: no look-ahead bias

The feature table for date `t` may only use information that was actually
public by date `t`. Macro series have an `observation_date` (the period the
number describes) and a separate `release_date` (when it became public) —
`sql/schema.sql` keeps these distinct on purpose. A model trained on leaked
future information is scientifically invalid, not just "optimistic."

## Repository structure

```
data/raw/{polymarket,fred,spy}/       # untouched source data (no BLS client -- FRED mirrors what's needed)
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
cp .env.example .env   # then add your free FRED_API_KEY (fred.stlouisfed.org/docs/api/api_key.html)
```

Polymarket and SPY/VIX need no API key. FRED does, but it's free and
instant.

## Running it

```bash
python3 src/agents/orchestrator.py   # the whole pipeline: data -> SQL -> quality check -> models -> backtest -> review
```

Or run any stage independently — each agent module and script is
standalone runnable (`python3 src/agents/<name>_agent.py`,
`python3 src/validation/run_model_ladder.py`, etc.) — useful when only a
downstream stage needs to rerun (e.g. after a model code change with no
new data). Run `python3 -m pytest tests/` for the unit tests.

## What I learned

- **Look-ahead bias is easy to introduce by accident, even when you're
  specifically trying to avoid it.** Two separate bugs made it into this
  project despite Phase 2 being entirely about preventing exactly this
  class of error: FRED's daily rate series (`DFF`/`DGS2`/`DGS10`) blew
  past the vintage-query API's row cap because they get re-stamped daily
  without being revised (fixed by treating them differently, not with the
  vintage-history approach used for CPI/PPI/etc.); and `NaN > 0` evaluates
  to `False` in pandas, not `NaN`, which silently mislabeled the last 5
  rows of every dataset as "down" instead of dropping them, and broke the
  Phase 5 backtest's cumulative return outright (`NaN` poisons a
  `cumprod()` from the first bad row onward) before it was caught and
  fixed. The `merge_asof(direction="backward")` mechanism itself worked
  correctly throughout and is now covered by tests — it was the code
  *around* it that had bugs.
- **Model complexity did not help, anywhere, on this problem.** Across
  both the 10-year macro-only ladder (Phase 4) and the 2-year
  Polymarket-inclusive comparison (Phase 5), logistic regression and
  random forest were competitive with or better than LSTM and a small
  Transformer, and nothing beat a naive majority-class baseline by more
  than noise. This matches the roadmap's own expectation (section 26)
  and this user's prior project ([`spy-garch-vwap`](../spy-garch-vwap/results/FINDINGS.md)):
  short-horizon SPY direction from daily macro/market features doesn't
  contain much learnable signal, and a fancier architecture can't
  manufacture signal that isn't in the data.
- **Polymarket's Fed-cut expectations did not add forecasting value**
  over the ~2-year window where the data exists — if anything AUC was
  slightly lower with it included, though the sample (519 rows, 3 folds)
  is too small to call that a real effect rather than noise. This is the
  project's central research question, and the honest answer, on the
  slice of Polymarket data collected so far (Fed-decision markets only —
  inflation/recession/GDP expectation markets from the roadmap's original
  scope were never pulled), is: no evidence it helps.
- **A backtest number and a forecast-accuracy number can tell different
  parts of the same story.** Neither model beat buy-and-hold SPY (4-6%
  strategy return vs. 31% buy-and-hold over 2 years) even though their
  AUCs were only mildly worse than random — being wrong (or flat) on the
  wrong days in a strong bull market is expensive in a way accuracy alone
  doesn't capture.
- **Full pipeline reproducibility (`orchestrator.py`) surfaces bugs that
  a one-off script run won't.** Re-running the whole thing end-to-end
  caught a real bug immediately: `database.py`'s schema creation wasn't
  idempotent (`CREATE TABLE` without `IF NOT EXISTS`), so a second run
  against an existing database crashed. Fixed by making every `CREATE
  TABLE` idempotent — an easy thing to miss when you only ever test
  against a freshly deleted database file.
