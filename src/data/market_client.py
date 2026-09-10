"""
SPY + VIX daily OHLCV via yfinance (free, no API key).

Roadmap ref: PDF section 23 "SPY Market Data" (Open/High/Low/Close/Volume,
Return, Realized Volatility, VIX).
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "spy"


def fetch_ohlcv(ticker: str, start: str = "2015-01-01") -> pd.DataFrame:
    """Daily OHLCV for one ticker, columns flattened to lowercase strings."""
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    df.columns = [c[0].lower() for c in df.columns]  # yfinance returns MultiIndex columns
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def save_raw(df: pd.DataFrame, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{name}.csv"
    df.to_csv(out_path)
    return out_path


if __name__ == "__main__":
    spy = fetch_ohlcv("SPY")
    vix = fetch_ohlcv("^VIX")
    spy_path = save_raw(spy, "spy_ohlcv")
    vix_path = save_raw(vix, "vix_ohlcv")
    print(f"SPY:  {len(spy)} rows -> {spy_path}")
    print(f"VIX:  {len(vix)} rows -> {vix_path}")
    print(spy.tail())
