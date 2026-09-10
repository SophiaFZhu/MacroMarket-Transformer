"""
Macro Data Agent (roadmap section 27): "collects CPI/PPI/jobs/rates."
Thin wrapper around src/data/fred_client.py -- the real work already lives
there and is independently tested; this just gives the orchestrator one
consistent thing to call. Run standalone: `python3 src/agents/macro_agent.py`
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run() -> bool:
    print("=== Macro Data Agent (FRED) ===")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "data" / "fred_client.py")],
    )
    return result.returncode == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
