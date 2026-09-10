# Phase 4: model ladder findings

## Setup

- **Feature set**: macro-only (`spy_return_1d`, `spy_return_5d`,
  `realized_vol`, `cpi_yoy`, `cpi_surprise`, `ppi_yoy`,
  `unemployment_rate`, `yield_2y`, `yield_10y`, `yield_spread_10y_2y`,
  `vix`) — 11 features. `fed_cut_probability` deliberately excluded here:
  Polymarket coverage only starts ~Aug 2024, and this run uses the full
  2016-2026 history. Whether Polymarket data adds anything is a separate,
  narrower Phase 5 comparison over just the overlapping window.
- **Target**: `1` if SPY closes higher 5 trading days later, else `0`.
  2,655 rows (2016-02-19 → 2026-09-10) after dropping feature warm-up rows
  and the 5 rows with no future price yet. Base rate: 61.1% up (5-day SPY
  returns are positive more often than not over this sample — mostly a
  long secular bull market).
- **Validation**: walk-forward, 5 expanding-window folds, 500-row minimum
  initial train, 5-day purge gap at each train/test boundary (see
  `src/validation/walk_forward.py`) so no training label's realized return
  window overlaps the test period.
- No hyperparameter tuning against these folds — model settings
  (RF depth, LSTM/Transformer size, epoch counts) were fixed upfront from
  reasonable defaults / the roadmap's suggested Transformer size, not
  searched. Tuning against the same folds used for the final comparison
  would be a form of the "cherry-picking" the roadmap's Reviewer Agent
  role (Phase 5) is explicitly supposed to catch.

## Results (mean ± std across 5 folds)

| Model | Accuracy | AUC | Log loss |
|---|---|---|---|
| Naive (majority class) | 0.604 ± 0.050 | 0.500 ± 0.000 | 0.674 ± 0.027 |
| Persistence (sign of last 5d return) | 0.513 ± 0.036 | 0.486 ± 0.023 | 6.731 ± 0.495 |
| Logistic regression | 0.603 ± 0.051 | 0.519 ± 0.040 | 0.865 ± 0.240 |
| Random forest | 0.582 ± 0.039 | 0.501 ± 0.044 | 0.697 ± 0.015 |
| LSTM | 0.586 ± 0.048 | 0.500 ± 0.056 | 0.724 ± 0.063 |
| Small Transformer | 0.556 ± 0.051 | 0.528 ± 0.062 | 1.025 ± 0.247 |

Full per-fold numbers: `phase4_model_ladder_folds.csv`.

## Interpretation

**No model beats the naive baseline in a way that looks like real signal.**
The naive majority-class predictor — which uses zero information about
any feature, it just always predicts "up" — already gets 60.4% accuracy,
because SPY goes up more often than not over 5-day windows in this
sample. Every other model's accuracy is at or below that. AUC (which,
unlike accuracy, actually measures discrimination rather than exploiting
class imbalance) sits at essentially 0.50 ± noise for every model except
the Transformer's 0.528 ± 0.062 — and that gap is well within one
standard deviation of pure noise, not a reliable edge.

The persistence baseline doing *worse* than coin-flip AUC (0.486) is
itself informative: 5-day SPY returns don't meaningfully autocorrelate in
this data — the market doesn't obviously trend at this horizon.

**This is a legitimate negative result, not a broken pipeline.** It's
consistent with a large body of finance literature on short-horizon
equity return predictability, and with this user's own prior project
([`spy-garch-vwap`](../../spy-garch-vwap/results/FINDINGS.md)), which
found no real edge for a published GARCH+VWAP strategy on SPY either.
Monthly/weekly macro releases (CPI, PPI, unemployment) simply don't carry
enough new information on a random Tuesday to move the needle on a 5-day
SPY forecast — most of the variation across days in this feature set
actually comes from the market-derived features (VIX, yields, recent SPY
returns), not the macro ones.

**Model complexity didn't help, as the roadmap predicted it might not**
(section 26: "If Logistic Regression matches or beats the Transformer,
that is a useful finding"). Logistic regression is statistically tied
with the naive baseline on accuracy and has the second-best AUC. The
Transformer has the best AUC but also the worst log loss (most
overconfident-and-wrong), on a dataset (~2,100 training rows per fold)
that's small relative to what Transformers are usually built for.

## What this does and doesn't say about Phase 5

This result does **not** yet answer the project's actual research
question (does Polymarket data help?) — that requires the narrower,
Polymarket-window-only comparison planned for Phase 5. What it does
establish: the macro-only baseline these models need to beat, measured
honestly, is "logistic regression, ~52% AUC" — not "nothing." If a
Polymarket-inclusive model in Phase 5 can't clear that bar on the same
kind of walk-forward evaluation, that's meaningful evidence Polymarket
data isn't adding value here, not just an artifact of a weak baseline.
