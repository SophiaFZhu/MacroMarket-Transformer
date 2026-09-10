"""
As-of joins: for each trading day t, what was the latest FRED value known
by t? This is the actual mechanism that enforces "day t may only use
information released by day t" (roadmap section 25's "core principle").

pd.merge_asof(..., direction="backward") does exactly this: for each row
in `left` it finds the closest `right` row whose key is <= the left key.
Used here with left_on=trading date, right_on=release_date, it can never
pick a macro row released after that trading day -- that's the leakage
guard, enforced by the join itself rather than by convention.
"""

import sqlite3

import pandas as pd


def load_series(conn: sqlite3.Connection, series_id: str) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT observation_date, release_date, value FROM macro_observations "
        "WHERE series_id = ? ORDER BY release_date",
        conn, params=(series_id,), parse_dates=["observation_date", "release_date"],
    )
    return df


def asof_join(trading_days: pd.DataFrame, series: pd.DataFrame, value_col: str) -> pd.Series:
    """trading_days: DataFrame with a sorted 'date' column.
    series: DataFrame with 'release_date' (sorted) and 'value'.
    Returns the as-of value for each trading day, aligned to trading_days.index."""
    merged = pd.merge_asof(
        trading_days[["date"]], series[["release_date", "value"]],
        left_on="date", right_on="release_date", direction="backward",
    )
    return merged["value"].rename(value_col)


def yoy_series(series: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year % change, computed on first-release values for both
    the current and year-ago observation (see fred_client.py docstring --
    we only keep first-release values, so this is a "first-release YoY,"
    not "final-revision YoY." A defensible proxy, not the official number."""
    s = series.sort_values("observation_date").copy()
    s["yoy"] = s["value"] / s["value"].shift(12) - 1.0
    return s[["release_date", "yoy"]].dropna()


def mom_change_series(series: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month change in the raw value. Used as a "surprise" proxy
    in daily_features.cpi_surprise since we don't have consensus/forecast
    data to compute a true actual-vs-expected surprise -- see README Known
    Limitations. Not the same thing as a real surprise metric."""
    s = series.sort_values("observation_date").copy()
    s["mom_change"] = s["value"] - s["value"].shift(1)
    return s[["release_date", "mom_change"]].dropna()
