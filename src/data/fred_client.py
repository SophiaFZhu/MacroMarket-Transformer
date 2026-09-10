"""
U.S. macro data via FRED (Federal Reserve Economic Data).

Roadmap ref: PDF section 23 "Fed & Rates" + "Inflation & Employment".
FRED mirrors the BLS series we need (CPI, PPI, unemployment, payrolls,
jobless claims) in addition to its own rate series, so one free API key
covers all the "macro" data groups -- no separate BLS key needed for now.

Requires FRED_API_KEY in .env (free, instant signup:
https://fred.stlouisfed.org/docs/api/api_key.html).

Note: this pulls the *current* (most-revised) value of each series. Real
macro data gets revised after first release (see roadmap section 24,
"Vintage Data" / "Revisions") -- using today's revised CPI value as if it
were known on the original release date is look-ahead bias. Phase 2 will
replace this with FRED's realtime/vintage API before anything gets used
to train a model. For now this just gets raw data flowing.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

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


def get_client() -> Fred:
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and add it "
            "to .env as FRED_API_KEY=..."
        )
    return Fred(api_key=api_key)


def fetch_series(fred: Fred, series_id: str, start: str = "2015-01-01") -> pd.DataFrame:
    s = fred.get_series(series_id, observation_start=start)
    return s.rename("value").rename_axis("observation_date").to_frame()


def save_raw(df: pd.DataFrame, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{name}.csv"
    df.to_csv(out_path)
    return out_path


if __name__ == "__main__":
    fred = get_client()
    for name, series_id in SERIES.items():
        df = fetch_series(fred, series_id)
        path = save_raw(df, name)
        print(f"{name:20s} ({series_id:10s}) {len(df):5d} rows -> {path}")
