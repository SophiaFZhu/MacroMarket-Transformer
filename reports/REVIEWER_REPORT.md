# Reviewer Agent report

## Sample sizes

- **[PASS]** Phase 4 (macro-only ladder): smallest fold has n=430 test rows (flag threshold: 100)
- **[PASS]** Phase 5 (Polymarket comparison): smallest fold has n=106 test rows (flag threshold: 100)

## AUC vs. noise

- **[PASS]** naive_majority: AUC 0.500 ± 0.000 -- not distinguishable from random, correctly reported as no edge
- **[PASS]** persistence: AUC 0.486 ± 0.024 -- not distinguishable from random, correctly reported as no edge
- **[PASS]** logistic: AUC 0.518 ± 0.037 -- not distinguishable from random, correctly reported as no edge
- **[PASS]** random_forest: AUC 0.499 ± 0.049 -- not distinguishable from random, correctly reported as no edge
- **[PASS]** lstm: AUC 0.489 ± 0.039 -- not distinguishable from random, correctly reported as no edge
- **[PASS]** transformer: AUC 0.514 ± 0.065 -- not distinguishable from random, correctly reported as no edge

## Known caveats documented in README

- **[PASS]** README documents: backtest has no transaction costs modeled
- **[PASS]** README documents: Polymarket's short history window is disclosed
- **[PASS]** README documents: no hyperparameter search against the eval folds

## Backtest cost assumptions

- **[WARN]** Backtest results include NO transaction costs -- real total returns would be lower, especially at ~64 trades over 2 years. Do not treat backtest returns as investable without adding costs (see spy-garch-vwap's 1% round-trip assumption for a precedent).
