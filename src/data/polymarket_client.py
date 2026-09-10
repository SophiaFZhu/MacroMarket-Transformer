"""
Polymarket expectation markets: find markets by keyword, pull their daily
probability history.

Roadmap ref: PDF section 23 "Polymarket Expectations" (Fed cuts/hikes,
inflation, recession, unemployment, GDP).

Two Polymarket APIs are involved:
- gamma-api.polymarket.com  -> market metadata + search (which market am I
  looking for, what are its outcome token ids)
- clob.polymarket.com       -> prices-history for one outcome token (the
  actual probability time series)
"""

from __future__ import annotations  # this repo's venv is Python 3.9; defers `bool | None` etc.

import json
from pathlib import Path

import pandas as pd
import requests

SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "polymarket"


def search_events(query: str, limit_per_type: int = 50, active_only: bool = False) -> list[dict]:
    """Free-text search over Polymarket events, closed + active by default.
    Each event bundles the related markets (e.g. one "Fed Decision in
    September?" event contains the 25bps/50bps/no-change markets for that
    meeting). Coverage is whatever Polymarket had -- for Fed-decision
    events that's meetings from roughly August 2024 onward (older meetings
    didn't have a liquid market), which is shorter than the SPY/FRED
    history. Any backtest comparing "with vs. without Polymarket" is only
    valid over the overlapping window."""
    params = {"q": query, "limit_per_type": limit_per_type}
    if active_only:
        params["events_status"] = "active"
    resp = requests.get(SEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("events", [])


def fetch_fed_decision_history() -> pd.DataFrame:
    """Stitch every FOMC meeting's "no rate change" market into one raw
    time series: one row per (meeting, day) with that meeting's implied
    probability of no change. Phase 3 will turn this into a single rolling
    "Fed cut probability" feature -- here we just collect the raw pieces,
    tagged by which meeting they belong to, so no information is lost."""
    events = search_events("Fed interest rate decision")
    rows = []
    for event in events:
        no_change = next(
            (m for m in event["markets"] if "no change" in m["question"].lower()),
            None,
        )
        if no_change is None or not no_change.get("clobTokenIds"):
            continue
        token_ids = json.loads(no_change["clobTokenIds"])
        history = fetch_price_history(token_ids[0])
        if history.empty:
            continue
        history = history.reset_index()
        history["meeting_event"] = event["title"]
        history["meeting_end_date"] = event["endDate"]
        history["market_question"] = no_change["question"]
        rows.append(history)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fetch_price_history(clob_token_id: str, fidelity_minutes: int = 1440) -> pd.DataFrame:
    """Daily ("fidelity=1440" minutes) probability history for one outcome token."""
    params = {"market": clob_token_id, "interval": "max", "fidelity": fidelity_minutes}
    resp = requests.get(CLOB_PRICES_URL, params=params, timeout=15)
    resp.raise_for_status()
    history = resp.json().get("history", [])
    df = pd.DataFrame(history)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["t"], unit="s")
    df = df.rename(columns={"p": "probability"}).set_index("timestamp")[["probability"]]
    return df


def save_raw(df: pd.DataFrame, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{name}.csv"
    df.to_csv(out_path)
    return out_path


if __name__ == "__main__":
    history = fetch_fed_decision_history()
    n_meetings = history["meeting_event"].nunique()
    path = save_raw(history, "fed_no_change_probability_history")
    print(f"{len(history)} rows across {n_meetings} FOMC meetings -> {path}")
    print(history[["timestamp", "probability", "meeting_event"]].tail(10))
