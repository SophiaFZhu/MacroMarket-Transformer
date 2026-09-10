"""
SPY/VIX-derived features. These need no as-of join for leakage safety --
each day's close is known by that day's close, full stop -- but the
*rolling* windows below (.pct_change, .rolling) only ever look backward
from each row, which is what keeps them leakage-safe too.
"""

import sqlite3

import numpy as np
import pandas as pd


def load_spy(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT date, close FROM spy_prices ORDER BY date",
        conn, parse_dates=["date"],
    )


def load_vix(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT date, close AS vix FROM vix_prices ORDER BY date",
        conn, parse_dates=["date"],
    )


def spy_features(spy: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    df = spy.sort_values("date").copy()
    log_return = np.log(df["close"] / df["close"].shift(1))
    df["spy_return_1d"] = df["close"].pct_change(1)
    df["spy_return_5d"] = df["close"].pct_change(5)
    # Annualized realized vol from a trailing window of daily log returns.
    df["realized_vol"] = log_return.rolling(vol_window).std() * np.sqrt(252)
    return df[["date", "spy_return_1d", "spy_return_5d", "realized_vol"]]
