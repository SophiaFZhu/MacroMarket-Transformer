"""
U.S. macro data via FRED (Federal Reserve Economic Data) -- point-in-time,
not just-the-latest-revision.

Roadmap ref: PDF section 23 "Fed & Rates" + "Inflation & Employment", and
section 24 "Time Alignment, Release Dates, and Vintage Data". FRED mirrors
the BLS series we need (CPI, PPI, unemployment, payrolls, jobless claims)
in addition to its own rate series, so one free API key covers all the
"macro" data groups.

Requires FRED_API_KEY in .env (free, instant signup:
https://fred.stlouisfed.org/docs/api/api_key.html).

## Why "first release" and not "today's value"

FRED (technically its ALFRED archive) keeps every historical revision of a
series. Query the observations endpoint with a wide realtime_start/
realtime_end window and you get one row *per vintage*: e.g. January 2024
CPI shows up three times below, once for the value as first published
(2024-02-13), once for a later revision (2025-02-12), and once for the
most recent revision (2026-02-13):

    realtime_start   date        value
    2024-02-13       2024-01-01  309.685   <- what the market actually saw first
    2025-02-12       2024-01-01  309.794   <- revised a year later
    2026-02-13       2024-01-01  309.698   <- revised again

A model predicting "what happens after January 2024 CPI comes out" may
only use 309.685, released 2024-02-13 -- not the number FRED shows today.
This module keeps only the first (smallest realtime_start) vintage per
observation period and reports that vintage's realtime_start as the
release_date, matching sql/schema.sql's macro_observations table exactly.
Daily series (Fed funds rate, Treasury yields) aren't revised in practice,
so this just recovers the plain value with release_date == observation
date -- same function, no special-casing needed.
"""

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "fred"

# friendly name -> FRED series ID
SERIES = {
    "cpi": "CPIAUCSL",           # CPI, all urban consumers, seasonally adjusted
    "core_cpi": "CPILFESL",      # CPI less food & energy
    "ppi": "PPIACO",             # Producer Price Index, all commodities
    "unemployment_rate": "UNRATE",
    "payrolls": "PAYEMS",        # nonfarm payrolls
    "jobless_claims": "ICSA",    # initial jobless claims, weekly
    "fed_funds_rate": "DFF",     # effective federal funds rate, daily
    "yield_2y": "DGS2",          # 2-year Treasury yield, daily
    "yield_10y": "DGS10",        # 10-year Treasury yield, daily
}

# Daily market/policy-rate series get re-stamped with a new "vintage" every
# single day even though the value itself is never revised -- FRED caps
# vintage requests at 2000 dates, so the wide realtime-range trick above
# blows that limit for 10 years of daily data. These don't need revision
# handling anyway (real-time rate, not a survey estimate), so they're
# fetched as plain current values with release_date == observation_date.
NOT_REVISED = {"fed_funds_rate", "yield_2y", "yield_10y"}


def get_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and add it "
            "to .env as FRED_API_KEY=..."
        )
    return api_key


def fetch_first_release_series(
    api_key: str, series_id: str, start: str = "2015-01-01"
) -> pd.DataFrame:
    """One row per observation period: the value as first published, and
    the date it was actually published (release_date)."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": "1776-07-04",  # FRED's documented "beginning of time"
        "realtime_end": "9999-12-31",    # and "end of time" -- returns all vintages
        "observation_start": start,
    }
    resp = requests.get(OBSERVATIONS_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observations"]

    df = pd.DataFrame(obs)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    df = df.dropna(subset=["value"])
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])

    first_release = (
        df.sort_values("realtime_start")
        .groupby("date", as_index=False)
        .first()
        .rename(columns={"date": "observation_date", "realtime_start": "release_date"})
        [["observation_date", "release_date", "value"]]
        .sort_values("observation_date")
        .reset_index(drop=True)
    )
    return first_release


def fetch_current_series(api_key: str, series_id: str, start: str = "2015-01-01") -> pd.DataFrame:
    """Plain current values for a series with no meaningful revisions
    (see NOT_REVISED) -- release_date is just set equal to observation_date."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    resp = requests.get(OBSERVATIONS_URL, params=params, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["observations"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["observation_date"] = pd.to_datetime(df["date"])
    df["release_date"] = df["observation_date"]
    return df[["observation_date", "release_date", "value"]].reset_index(drop=True)


def save_raw(df: pd.DataFrame, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{name}.csv"
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    api_key = get_api_key()
    for name, series_id in SERIES.items():
        if name in NOT_REVISED:
            df = fetch_current_series(api_key, series_id)
        else:
            df = fetch_first_release_series(api_key, series_id)
        path = save_raw(df, name)
        max_lag = (df["release_date"] - df["observation_date"]).dt.days.max()
        print(f"{name:20s} ({series_id:10s}) {len(df):5d} rows, "
              f"max release lag {max_lag:3d}d -> {path}")
