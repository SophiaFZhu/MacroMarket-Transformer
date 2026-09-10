"""
Data Quality Agent (roadmap section 27): "checks missing values, release
dates, revisions, time zones and leakage."

This and the other agents/ modules are plain Python, not LLM calls. The
roadmap's own framing (section 18) is "Agent = LLM + Memory + Tools +
Workflow," but section 27's actual job descriptions for these 5 agents
("checks missing values," "runs walk-forward evaluation," "searches for
overfitting") are deterministic checks and pipeline steps -- exactly the
kind of thing you write as code, not something that benefits from an
LLM's judgment at runtime. Roadmap section 18 itself says the real skill
is task decomposition, not the word "Agent" -- so decomposing the pipeline
into named, single-responsibility stages (this file being one) is the
actual point, whether or not an LLM sits inside any of them.

Run standalone: `python3 src/agents/data_quality_agent.py`
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "database"))
import database  # noqa: E402


def check_release_before_observation(conn: sqlite3.Connection) -> list:
    """The one check that would catch a Phase-2-style bug immediately:
    no macro row should ever claim to be released before the period it
    describes even started."""
    bad = pd.read_sql(
        "SELECT series_id, observation_date, release_date FROM macro_observations "
        "WHERE release_date < observation_date",
        conn,
    )
    if len(bad):
        return [("FAIL", f"{len(bad)} macro_observations rows have release_date < observation_date", bad)]
    return [("PASS", "All macro_observations: release_date >= observation_date", None)]


def check_primary_key_uniqueness(conn: sqlite3.Connection) -> list:
    findings = []
    tables_keys = {
        "macro_observations": ["series_id", "observation_date"],
        "spy_prices": ["date"],
        "vix_prices": ["date"],
        "polymarket_prices": ["market_id", "timestamp"],
        "daily_features": ["date"],
    }
    for table, keys in tables_keys.items():
        df = pd.read_sql(f"SELECT {', '.join(keys)} FROM {table}", conn)
        n_dupes = df.duplicated().sum()
        if n_dupes:
            findings.append(("FAIL", f"{table}: {n_dupes} duplicate rows on {keys}", None))
        else:
            findings.append(("PASS", f"{table}: no duplicate keys", None))
    return findings


def check_date_continuity(conn: sqlite3.Connection) -> list:
    """Not every calendar day should be present (weekends/holidays aren't
    trading days) -- but there shouldn't be multi-week gaps either, which
    would indicate a broken data pull."""
    spy = pd.read_sql("SELECT date FROM spy_prices ORDER BY date", conn, parse_dates=["date"])
    gaps = spy["date"].diff().dt.days
    max_gap = gaps.max()
    if max_gap > 10:
        worst = spy["date"].iloc[gaps.idxmax()]
        return [("WARN", f"Largest gap in spy_prices is {int(max_gap)} days, ending {worst.date()}", None)]
    return [("PASS", f"spy_prices: largest gap is {int(max_gap)} days (holidays only)", None)]


def check_daily_features_coverage(conn: sqlite3.Connection) -> list:
    df = pd.read_sql("SELECT * FROM daily_features", conn)
    findings = []
    always_populated = ["yield_2y", "yield_10y", "vix", "spy_return_1d"]
    for col in always_populated:
        n_na = df[col].isna().sum()
        if n_na > 5:  # a handful at the very start is expected (rolling-window warm-up)
            findings.append(("WARN", f"daily_features.{col}: {n_na} NaN rows (expected ~0-5 at warm-up)", None))
        else:
            findings.append(("PASS", f"daily_features.{col}: {n_na} NaN rows", None))
    poly_coverage = df["fed_cut_probability"].notna().mean()
    findings.append((
        "INFO",
        f"daily_features.fed_cut_probability: {poly_coverage:.1%} coverage "
        f"(short Polymarket history is a known, documented limitation, not a bug)",
        None,
    ))
    return findings


def run() -> bool:
    conn = database.get_connection()
    checks = [
        check_release_before_observation,
        check_primary_key_uniqueness,
        check_date_continuity,
        check_daily_features_coverage,
    ]
    all_findings = []
    for check in checks:
        all_findings += check(conn)
    conn.close()

    print("=== Data Quality Agent ===")
    for status, message, detail in all_findings:
        print(f"[{status}] {message}")
        if detail is not None and not detail.empty:
            print(detail.head().to_string(index=False))

    n_fail = sum(1 for status, _, _ in all_findings if status == "FAIL")
    print(f"\n{len(all_findings)} checks, {n_fail} failed.")
    return n_fail == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
