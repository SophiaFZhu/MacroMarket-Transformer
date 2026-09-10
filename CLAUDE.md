# sophia-macro-ai-spy

AI Macro-to-Market Research System. Full context and architecture are in
`README.md`; the source roadmap is
`../Github Project 1/Sophia_AI_Foundations_GitHub_Roadmap_Updated.pdf`
(Part VI, §21-27, and the 8-week plan in §30).

## Conventions

- Beginner-paced: Sophia is learning Python from scratch through this
  project (started 2026-09-10). Prefer explaining concepts explicitly over
  assuming familiarity, and keep her actually writing the code rather than
  having it appear fully formed — match the roadmap week she's on.
- Every macro/market table keeps `observation_date` separate from
  `release_date` (see `sql/schema.sql`). Never join macro data into a
  day's feature row using a date the data wasn't yet public on — that's
  look-ahead bias and invalidates the whole experiment.
- Model ladder must be respected: naive → logistic → random forest/XGBoost
  → LSTM → small Transformer. Never report a Transformer result without the
  simpler baselines run on the same split.
- Evaluation is walk-forward / out-of-sample only. No metric gets reported
  from in-sample fit.
- See sibling project `../spy-garch-vwap` for this user's established
  conventions on honest reporting of negative/weak results (README +
  `results/FINDINGS.md` pattern) — worth reusing here once there's a
  backtest to report.
