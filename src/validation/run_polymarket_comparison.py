"""
The project's actual research question, finally answerable: within the
window where Polymarket data exists (~Aug 2024 onward, 524 rows), does
adding fed_cut_probability / fed_cut_probability_delta to the macro-only
feature set improve out-of-sample forecasts of SPY direction?

Both feature sets are evaluated on the EXACT SAME rows and the SAME
walk-forward folds -- otherwise a difference in score could just be a
difference in which days got tested, not the feature itself. LSTM/
Transformer are deliberately skipped here (see README): 524 rows is too
small to trust a deep sequence model's walk-forward result, and Phase 4
already established that model complexity wasn't the bottleneck on 5x
more data.

Also runs the walk-forward backtest (roadmap section 27's "Walk-Forward
Backtest" -> "Reviewer Agent" step) on the logistic model's out-of-sample
predictions from both feature sets, vs. buy-and-hold, over the same
non-overlapping trade sequence.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

sys.path.insert(0, str(PROJECT_ROOT / "src" / "database"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest  # noqa: E402
import database  # noqa: E402
import metrics  # noqa: E402
from baseline import NaiveMajorityModel, PersistenceModel, RandomForestModel  # noqa: E402
from logistic import LogisticModel  # noqa: E402
from walk_forward import walk_forward_splits  # noqa: E402

MACRO_ONLY_FEATURES = [
    "spy_return_1d", "spy_return_5d", "realized_vol",
    "cpi_yoy", "cpi_surprise", "ppi_yoy", "unemployment_rate",
    "yield_2y", "yield_10y", "yield_spread_10y_2y", "vix",
]
POLYMARKET_FEATURES = ["fed_cut_probability", "fed_cut_probability_delta"]
FULL_FEATURES = MACRO_ONLY_FEATURES + POLYMARKET_FEATURES
HORIZON = 5


def build_dataset(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM daily_features ORDER BY date", conn, parse_dates=["date"])
    spy = pd.read_sql("SELECT date, close FROM spy_prices ORDER BY date", conn, parse_dates=["date"])
    df = df.merge(spy, on="date", how="left")
    df["future_return"] = df["close"].shift(-HORIZON) / df["close"] - 1.0
    # NaN > 0 is False, not NaN -- drop rows with no future price BEFORE
    # deriving target, or the last HORIZON rows silently mislabel as "down".
    df = df.dropna(subset=FULL_FEATURES + ["future_return"]).reset_index(drop=True)
    df["target"] = (df["future_return"] > 0).astype(int)
    return df


def run_feature_set(name, feature_cols, df, y, folds):
    rows = []
    oos_predictions = {}  # model_name -> (dates, y_prob, future_return), concatenated across folds
    X_full = df[feature_cols]
    for model_name, model_factory in [
        ("naive_majority", NaiveMajorityModel),
        ("persistence", PersistenceModel),
        ("logistic", LogisticModel),
        ("random_forest", RandomForestModel),
    ]:
        fold_dates, fold_probs, fold_returns = [], [], []
        for k, fold in enumerate(folds):
            model = model_factory()
            model.fit(X_full.iloc[fold.train_idx], y.iloc[fold.train_idx])
            y_prob = model.predict_proba(X_full.iloc[fold.test_idx])
            m = metrics.summarize(y.iloc[fold.test_idx].to_numpy(), y_prob)
            rows.append({"feature_set": name, "model": model_name, "fold": k, **m})

            fold_dates.append(df["date"].iloc[fold.test_idx])
            fold_probs.append(y_prob)
            fold_returns.append(df["future_return"].iloc[fold.test_idx].to_numpy())

        oos_predictions[model_name] = (
            pd.concat(fold_dates).reset_index(drop=True),
            np.concatenate(fold_probs),
            np.concatenate(fold_returns),
        )
    return rows, oos_predictions


def non_overlapping_backtest(dates, y_prob, future_return, horizon=HORIZON):
    """Subsample every `horizon`-th out-of-sample row so consecutive
    trades' forward-return windows don't overlap in time."""
    idx = np.arange(0, len(dates), horizon)
    return backtest.run_backtest(dates.iloc[idx], y_prob[idx], future_return[idx])


if __name__ == "__main__":
    conn = database.get_connection()
    df = build_dataset(conn)
    y = df["target"]
    print(f"Dataset: {len(df)} rows, {df['date'].min().date()} -> {df['date'].max().date()}, "
          f"base rate P(up)={y.mean():.3f}")

    folds = walk_forward_splits(len(df), n_splits=3, min_train_size=200, purge=HORIZON)
    for k, fold in enumerate(folds):
        print(f"  fold {k}: n_train={len(fold.train_idx)} n_test={len(fold.test_idx)}")

    all_rows = []
    all_oos = {}
    for name, feature_cols in [("macro_only", MACRO_ONLY_FEATURES), ("macro_plus_polymarket", FULL_FEATURES)]:
        t0 = time.time()
        rows, oos = run_feature_set(name, feature_cols, df, y, folds)
        all_rows += rows
        all_oos[name] = oos
        print(f"{name:25s} done in {time.time() - t0:.1f}s")

    results = pd.DataFrame(all_rows)
    summary = results.groupby(["feature_set", "model"])[["accuracy", "auc", "log_loss"]].mean()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "phase5_polymarket_comparison_folds.csv", index=False)
    summary.to_csv(REPORTS_DIR / "phase5_polymarket_comparison_summary.csv")

    print("\n=== macro_only vs macro_plus_polymarket (mean across 3 folds) ===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== Walk-forward backtest (logistic model, non-overlapping trades) ===")
    backtest_summaries = {}
    for name in ["macro_only", "macro_plus_polymarket"]:
        dates, y_prob, future_return = all_oos[name]["logistic"]
        result = non_overlapping_backtest(dates, y_prob, future_return)
        backtest_summaries[name] = result["summary"]
        result["equity"].to_csv(REPORTS_DIR / f"phase5_backtest_equity_{name}.csv", index=False)
        print(f"\n{name}:")
        for k, v in result["summary"].items():
            print(f"  {k:25s} {v:.4f}" if isinstance(v, float) else f"  {k:25s} {v}")

    pd.DataFrame(backtest_summaries).to_csv(REPORTS_DIR / "phase5_backtest_summary.csv")
    print(f"\n-> {REPORTS_DIR}/phase5_*.csv")
    conn.close()
