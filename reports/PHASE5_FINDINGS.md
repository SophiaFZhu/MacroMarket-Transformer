# Phase 5: does Polymarket data help, and what happens if you trade on it?

This is the project's actual research question, finally answerable:
*does adding Polymarket's Fed-cut expectation data improve on macro-only
forecasts of SPY direction, and does trading on any of this beat just
holding SPY?*

## Setup

- **Window**: restricted to the rows where both feature sets are
  fully populated — 2024-08-08 → 2026-09-10 (524 rows as of the
  2026-09-17 refresh; Polymarket's real coverage start, see README Known
  Limitations). Both feature sets use the identical rows and identical
  folds — a difference in score can only be the feature, not which days
  got tested.
- **Feature sets**: `macro_only` (the same 11 features as Phase 4) vs.
  `macro_plus_polymarket` (+ `fed_cut_probability`, `fed_cut_probability_delta`).
- **Models**: naive, persistence, logistic, random forest. LSTM/Transformer
  deliberately skipped — 519 rows split into 3 folds leaves ~300 rows per
  training set, too little to trust a deep sequence model's result, and
  Phase 4 already found model complexity wasn't the bottleneck on 5x more
  data.
- **Validation**: same walk-forward + purge-gap methodology as Phase 4,
  3 folds instead of 5 (smaller sample).
- **Backtest**: logistic model's out-of-sample predictions from each
  feature set, converted to a long/flat strategy (long when P(up) > 0.5),
  evaluated on a non-overlapping trade sequence (every 5th day, matching
  the 5-day prediction horizon) against buy-and-hold SPY. No transaction
  costs modeled.

## Results

### Model comparison (mean across 3 folds)

| Feature set | Model | Accuracy | AUC |
|---|---|---|---|
| macro_only | naive | 0.611 | 0.500 |
| macro_only | persistence | 0.466 | 0.407 |
| macro_only | logistic | 0.454 | **0.548** |
| macro_only | random forest | 0.546 | 0.558 |
| macro_plus_polymarket | naive | 0.611 | 0.500 |
| macro_plus_polymarket | persistence | 0.466 | 0.407 |
| macro_plus_polymarket | logistic | 0.441 | 0.528 |
| macro_plus_polymarket | random forest | 0.549 | **0.561** |

Full numbers: `phase5_polymarket_comparison_summary.csv` /
`_folds.csv`.

### Backtest (logistic model, 65 non-overlapping trades, ~2 years)

| | Total return | Sharpe | Max drawdown | % time long |
|---|---|---|---|---|
| macro_only strategy | 10.0% | 0.85 | -8.7% | 53.8% |
| macro_plus_polymarket strategy | 9.0% | 0.75 | -8.7% | 61.5% |
| Buy-and-hold SPY | **29.5%** | **1.76** | -8.7% | 100% |

*(Refreshed 2026-09-17, see `PHASE4_FINDINGS.md` for the same-day data
refresh note re: the FOMC's 0.25pp move not yet appearing in FRED's
published series. Numbers above replace the 2026-09-10 run: dataset grew
from 519 to 524 rows and the window's end moved from 2026-09-02 to
2026-09-10. Direction of the finding is unchanged, though which model
"wins" on AUC when Polymarket is added flipped for random forest — 0.571
→ 0.558 in the prior run, 0.558 → 0.561 here — underscoring the
Interpretation section's point below that this comparison is noisy at 3
folds and shouldn't be read as a stable ranking either way.)*

## Interpretation

**Adding Polymarket data did not clearly help — results are mixed and
within noise.** Logistic regression's AUC got slightly *worse* with
Polymarket added (0.548 → 0.528); random forest's got slightly *better*
(0.558 → 0.561). Given the small sample (3 folds, ~106 rows each), neither
move should be read as a real effect in either direction — both are well
within what 3 folds of noise could produce. What the results do support
is the negative: **no consistent evidence Polymarket's Fed-cut
expectations add forecasting value for SPY direction over this window**,
consistent with Phase 4's broader finding that nothing in this pipeline
beats a simple baseline by a meaningful margin.

Interesting aside: `macro_only`'s AUC here (0.548, 0.558) is close to
Phase 4's full-history run (0.518, 0.500) — but this window is also much
shorter (2 years vs. 10) and a different market regime (mostly a strong,
low-volatility bull run), so any small difference isn't a contradiction,
just a reminder that a 3-fold, 2-year result is noisier and less
trustworthy than the 5-fold, 10-year one. The full-history result is the
one to trust more.

**The backtest makes the practical stakes concrete.** Both model-based
strategies dramatically underperform simply holding SPY — 9-10% total
return vs. ~30% buy-and-hold, over a period when SPY happened to go up a
lot. Being flat ~38-46% of the time (whenever the model predicted "down")
forfeited most of that upside, and the model wasn't accurate enough at
picking *which* days to be flat on to make up for it. This is exactly
what "no real edge" costs in practice, not just an abstract AUC number.

**What would change this conclusion**: more Polymarket history (the
single biggest constraint — ~500 rows is a small sample for this kind of
question), the missing CPI/inflation/recession Polymarket markets (this
project only pulled Fed-decision markets — see README structure, the PDF
also lists inflation/recession/GDP expectation markets as available),
consensus-forecast data for a real `cpi_surprise` metric, and
transaction-cost-adjusted backtesting before any of this could inform a
real decision.

## Automated review

`src/agents/reviewer_agent.py` re-checks this project's own claims against
fixed rules every time the pipeline runs (sample sizes, AUC-vs-noise,
whether known caveats are actually documented) — see
`reports/REVIEWER_REPORT.md` for the latest run. As of this writeup: 0
FLAG-level findings (nothing here would need more scrutiny than it's
already gotten), 1 standing WARN (no transaction costs in the backtest,
which is already disclosed above and in the README, not a surprise).
