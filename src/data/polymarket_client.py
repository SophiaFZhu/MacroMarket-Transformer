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


def search_events(query: str, limit_per_type: int = 10, active_only: bool = True) -> list[dict]:
    """Free-text search over Polymarket events. Each event bundles the
    related markets (e.g. one "Fed Decision in September?" event contains
    the 25bps/50bps/no-change markets for that meeting)."""
    params = {"q": query, "limit_per_type": limit_per_type}
    if active_only:
        params["events_status"] = "active"
    resp = requests.get(SEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("events", [])


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
    events = search_events("Fed interest rate decision")
    print(f"Found {len(events)} events matching 'Fed interest rate decision':")
    for e in events[:5]:
        print(" -", e["title"], "| markets:", len(e.get("markets", [])))

    # Nearest upcoming FOMC event's "no change" market as a worked example.
    top_event = events[0]
    no_change_market = next(
        m for m in top_event["markets"] if "no change" in m["question"].lower()
        or "25 bps" in m["question"].lower()
    )
    token_ids = json.loads(no_change_market["clobTokenIds"])  # [Yes_token_id, No_token_id]
    history = fetch_price_history(token_ids[0])
    path = save_raw(history, "fed_decision_example")
    print(f"\nSaved {len(history)} rows of '{no_change_market['question']}' -> {path}")
    print(history.tail())
