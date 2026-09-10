"""
Runs the full model ladder (naive -> persistence -> logistic -> random
forest -> LSTM -> small Transformer) through walk-forward validation on
the same data and reports the comparison. This is the actual deliverable
of Phase 4 -- roadmap section 26's point is that the comparison itself
*is* the result, not any single model's number in isolation.

Feature set for this run deliberately excludes fed_cut_probability /
fed_cut_probability_delta: Polymarket coverage only starts ~Aug 2024 (see
README Known Limitations), so including it here would force dropping most
of the 2015-2024 history this ladder is meant to run over. Whether
Polymarket data adds anything beyond this macro-only ladder is a separate,
narrower comparison for Phase 5, over just the overlapping window.

Not in the PDF's file list -- added as the natural place to wire the
model ladder + walk-forward validation together, same pattern as
features/build_daily_features.py.
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

import database  # noqa: E402
import metrics  # noqa: E402
from baseline import NaiveMajorityModel, PersistenceModel, RandomForestModel  # noqa: E402
from logistic import LogisticModel  # noqa: E402
from lstm import LSTMModel  # noqa: E402
from transformer import TransformerModel  # noqa: E402
from walk_forward import walk_forward_splits  # noqa: E402

MACRO_ONLY_FEATURES = [
    "spy_return_1d", "spy_return_5d", "realized_vol",
    "cpi_yoy", "cpi_surprise", "ppi_yoy", "unemployment_rate",
    "yield_2y", "yield_10y", "yield_spread_10y_2y", "vix",
]
HORIZON = 5  # predict direction of the SPY return over the next 5 trading days
SEQUENCE_WINDOW = 60


def build_dataset(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM daily_features ORDER BY date", conn, parse_dates=["date"])
    spy = pd.read_sql("SELECT date, close FROM spy_prices ORDER BY date", conn, parse_dates=["date"])
    df = df.merge(spy, on="date", how="left")

    # Label: does SPY close HORIZON trading days from now end up higher?
    # This is a forward-looking label by design (that's what "prediction"
    # means) -- the leakage rule from Phase 2/3 is about FEATURES, not
    # about the label depending on the future.
    df["future_return"] = df["close"].shift(-HORIZON) / df["close"] - 1.0
    # NaN > 0 is False, not NaN -- drop rows with no future price BEFORE
    # deriving target, or the last HORIZON rows silently mislabel as "down".
    df = df.dropna(subset=MACRO_ONLY_FEATURES + ["future_return"]).reset_index(drop=True)
    df["target"] = (df["future_return"] > 0).astype(int)
    return df


def run_flat_model(name, model, X, y, folds):
    rows = []
    for k, fold in enumerate(folds):
        model.fit(X.iloc[fold.train_idx], y.iloc[fold.train_idx])
        y_prob = model.predict_proba(X.iloc[fold.test_idx])
        m = metrics.summarize(y.iloc[fold.test_idx].to_numpy(), y_prob)
        rows.append({"model": name, "fold": k, **m})
    return rows


def run_sequence_model(name, model, X, y, folds):
    rows = []
    for k, fold in enumerate(folds):
        y_prob = model.fit_predict(X, y, fold.train_idx, fold.test_idx)
        m = metrics.summarize(y.iloc[fold.test_idx].to_numpy(), y_prob)
        rows.append({"model": name, "fold": k, **m})
    return rows


if __name__ == "__main__":
    conn = database.get_connection()
    df = build_dataset(conn)
    X = df[MACRO_ONLY_FEATURES]
    y = df["target"]
    print(f"Dataset: {len(df)} rows, {df['date'].min().date()} -> {df['date'].max().date()}, "
          f"base rate P(up)={y.mean():.3f}")

    folds = walk_forward_splits(len(df), n_splits=5, min_train_size=500, purge=HORIZON)
    for k, fold in enumerate(folds):
        print(f"  fold {k}: train=[0:{len(fold.train_idx)}) "
              f"test=[{fold.test_idx[0]}:{fold.test_idx[-1]+1}) n_test={len(fold.test_idx)}")

    all_rows = []
    for name, model in [
        ("naive_majority", NaiveMajorityModel()),
        ("persistence", PersistenceModel()),
        ("logistic", LogisticModel()),
        ("random_forest", RandomForestModel()),
    ]:
        t0 = time.time()
        all_rows += run_flat_model(name, model, X, y, folds)
        print(f"{name:20s} done in {time.time() - t0:.1f}s")

    for name, model in [
        ("lstm", LSTMModel(window=SEQUENCE_WINDOW)),
        ("transformer", TransformerModel(window=SEQUENCE_WINDOW)),
    ]:
        t0 = time.time()
        all_rows += run_sequence_model(name, model, X, y, folds)
        print(f"{name:20s} done in {time.time() - t0:.1f}s")

    results = pd.DataFrame(all_rows)
    summary = results.groupby("model")[["accuracy", "auc", "log_loss"]].agg(["mean", "std"])
    summary = summary.reindex([
        "naive_majority", "persistence", "logistic", "random_forest", "lstm", "transformer"
    ])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "phase4_model_ladder_folds.csv", index=False)
    summary.to_csv(REPORTS_DIR / "phase4_model_ladder_summary.csv")

    print("\n=== Walk-forward summary (mean +/- std across 5 folds) ===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {REPORTS_DIR / 'phase4_model_ladder_folds.csv'}")
    print(f"-> {REPORTS_DIR / 'phase4_model_ladder_summary.csv'}")
    conn.close()
