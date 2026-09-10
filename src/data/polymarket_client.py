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


def _classify_outcome(question: str) -> str | None:
    """Map a market question to one of 5 outcome buckets, or None if it
    doesn't match the expected "Fed Decision in <Month>?" event shape."""
    q = question.lower()
    if "no change" in q:
        return "no_change"
    if "decrease" in q and "50" in q:
        return "cut_50"
    if "decrease" in q and "25" in q:
        return "cut_25"
    if "increase" in q and "50" in q:
        return "hike_50"
    if "increase" in q and "25" in q:
        return "hike_25"
    return None


def fetch_fed_decision_history() -> pd.DataFrame:
    """Stitch every FOMC meeting's 5 outcome-bucket markets (no change,
    cut 25bps, cut 50+bps, hike 25bps, hike 50+bps) into one raw time
    series: one row per (meeting, outcome, day). Phase 3 sums the cut
    buckets into a single "probability of any cut" feature -- here we just
    collect all 5 raw pieces per meeting, tagged by outcome, so no
    information is lost and hikes are still available if needed later."""
    events = search_events("Fed interest rate decision")
    rows = []
    for event in events:
        for market in event["markets"]:
            outcome = _classify_outcome(market["question"])
            if outcome is None or not market.get("clobTokenIds"):
                continue
            token_ids = json.loads(market["clobTokenIds"])
            history = fetch_price_history(token_ids[0])
            if history.empty:
                continue
            history = history.reset_index()
            history["meeting_event"] = event["title"]
            history["meeting_end_date"] = event["endDate"]
            history["outcome"] = outcome
            history["market_question"] = market["question"]
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
    path = save_raw(history, "fed_decision_probability_history")
    print(f"{len(history)} rows across {n_meetings} FOMC meetings, "
          f"outcomes: {sorted(history['outcome'].unique())} -> {path}")
    print(history[["timestamp", "probability", "meeting_event", "outcome"]].tail(10))
