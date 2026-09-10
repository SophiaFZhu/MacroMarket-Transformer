"""
Backtest Agent (roadmap section 27): "runs walk-forward evaluation." Wraps
src/validation/run_polymarket_comparison.py, which produces both the
macro-only-vs-Polymarket model comparison AND the walk-forward backtest
in one run -- the backtest needs that comparison's out-of-sample
predictions as its input, so splitting them into two subprocess calls
would just mean re-running the same walk-forward folds twice.
Run standalone: `python3 src/agents/backtest_agent.py`
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run() -> bool:
    print("=== Backtest Agent ===")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "validation" / "run_polymarket_comparison.py")],
    )
    return result.returncode == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
