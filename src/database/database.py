"""
SQLite access layer: init the schema, load raw CSVs into it.

Roadmap ref: PDF section 25 "SQL, Feature Engineering, and the Daily
Feature Vector".
"""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "research.db"

FRED_SERIES = [
    "cpi", "core_cpi", "ppi", "unemployment_rate", "payrolls",
    "jobless_claims", "fed_funds_rate", "yield_2y", "yield_10y",
]


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())


def load_macro_observations(conn: sqlite3.Connection) -> int:
    """One row per (series_id, observation_date), each already resolved
    to its first-released value and release_date by fred_client.py."""
    conn.execute("DELETE FROM macro_observations")
    total = 0
    for series_id in FRED_SERIES:
        path = RAW_DIR / "fred" / f"{series_id}.csv"
        df = pd.read_csv(path)
        df["series_id"] = series_id
        df.to_sql("macro_observations", conn, if_exists="append", index=False)
        total += len(df)
    conn.commit()
    return total


def load_spy_prices(conn: sqlite3.Connection) -> int:
    conn.execute("DELETE FROM spy_prices")
    df = pd.read_csv(RAW_DIR / "spy" / "spy_ohlcv.csv")
    df.to_sql("spy_prices", conn, if_exists="append", index=False)
    conn.commit()
    return len(df)


def load_vix_prices(conn: sqlite3.Connection) -> int:
    conn.execute("DELETE FROM vix_prices")
    df = pd.read_csv(RAW_DIR / "spy" / "vix_ohlcv.csv")[["date", "close"]]
    df.to_sql("vix_prices", conn, if_exists="append", index=False)
    conn.commit()
    return len(df)


def load_polymarket_prices(conn: sqlite3.Connection) -> int:
    """market_id = the full market question, e.g. "Will the Fed decrease
    interest rates by 25 bps after the January 2025 meeting?" -- unique
    per meeting per outcome bucket. NOTE: event *titles* like "Fed decision
    in January?" repeat every year, so meeting_event alone is NOT a safe
    key (caused duplicate-key inserts the first time this was written)."""
    conn.execute("DELETE FROM polymarket_prices")
    df = pd.read_csv(RAW_DIR / "polymarket" / "fed_decision_probability_history.csv")
    out = pd.DataFrame({
        "market_id": df["market_question"],
        "meeting_end_date": pd.to_datetime(df["meeting_end_date"]).dt.date.astype(str),
        "outcome": df["outcome"],
        "timestamp": df["timestamp"],
        "probability": df["probability"],
    }).drop_duplicates(subset=["market_id", "timestamp"])
    out.to_sql("polymarket_prices", conn, if_exists="append", index=False)
    conn.commit()
    return len(out)


if __name__ == "__main__":
    conn = get_connection()
    init_schema(conn)
    n_macro = load_macro_observations(conn)
    n_spy = load_spy_prices(conn)
    n_vix = load_vix_prices(conn)
    n_poly = load_polymarket_prices(conn)
    print(f"macro_observations: {n_macro} rows")
    print(f"spy_prices:         {n_spy} rows")
    print(f"vix_prices:         {n_vix} rows")
    print(f"polymarket_prices:  {n_poly} rows")
    print(f"-> {DB_PATH}")
    conn.close()
