"""
Polymarket Agent (roadmap section 27): "tracks expectation probabilities
and changes." Thin wrapper around src/data/polymarket_client.py -- see
macro_agent.py for why this is a plain subprocess call, not an LLM call.
Run standalone: `python3 src/agents/polymarket_agent.py`
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run() -> bool:
    print("=== Polymarket Agent ===")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "data" / "polymarket_client.py")],
    )
    return result.returncode == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
