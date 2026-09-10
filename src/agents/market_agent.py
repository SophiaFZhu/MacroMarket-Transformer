"""
Market Data Agent (roadmap section 27): "collects SPY/VIX/yields." Thin
wrapper around src/data/market_client.py -- see macro_agent.py for why
this is a plain subprocess call, not an LLM call. Treasury yields
themselves are collected by macro_agent.py (they're a FRED series), not
here -- this agent covers the two tickers that come from yfinance.
Run standalone: `python3 src/agents/market_agent.py`
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run() -> bool:
    print("=== Market Data Agent (SPY, VIX) ===")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "data" / "market_client.py")],
    )
    return result.returncode == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
