# MacroMarket-Transformer

AI Macro-to-Market Research System. Full context and architecture are in
`README.md`; the source roadmap is
`../Github Project 1/Sophia_AI_Foundations_GitHub_Roadmap_Updated.pdf`
(Part VI, §21-27, and the 8-week plan in §30).

## Conventions

- Project-first (as of 2026-09-10, superseding the original beginner-paced
  plan): Sophia wants the project actually finished, learning the
  math/CS/Python concepts from the real working code as it's built rather
  than from standalone exercises first. Write real, tested code for each
  phase and explain it (in comments + chat, referencing the PDF's section
  numbers) rather than leaving fill-in-the-blank stubs. She's still a
  programming beginner, so explain concepts explicitly — just don't gate
  the actual pipeline behind exercises anymore.
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
