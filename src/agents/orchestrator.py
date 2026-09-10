"""
Orchestrator (roadmap section 27, Figure 21): runs the full pipeline
end to end, stage by stage, stopping at the first failure.

  Macro / Polymarket / Market agents (data collection)
        |
        v
  Load into SQL (database.py)
        |
        v
  Data Quality Agent
        |
        v
  Modelling Agent (feature engineering + model ladder)
        |
        v
  Backtest Agent (Polymarket comparison + walk-forward backtest)
        |
        v
  Reviewer Agent

Run standalone: `python3 src/agents/orchestrator.py`. Each stage is also
independently runnable (see each agent module's own docstring) -- useful
when only the downstream stages need to rerun, e.g. after a code change to
a model with no new data.
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest_agent  # noqa: E402
import data_quality_agent  # noqa: E402
import macro_agent  # noqa: E402
import market_agent  # noqa: E402
import modelling_agent  # noqa: E402
import polymarket_agent  # noqa: E402
import reviewer_agent  # noqa: E402


def load_into_sql() -> bool:
    print("=== Load into SQL (database.py) ===")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "database" / "database.py")],
    )
    return result.returncode == 0


if __name__ == "__main__":
    stages = [
        ("Macro Data Agent", macro_agent.run),
        ("Polymarket Agent", polymarket_agent.run),
        ("Market Data Agent", market_agent.run),
        ("Load into SQL", load_into_sql),
        ("Data Quality Agent", data_quality_agent.run),
        ("Modelling Agent", modelling_agent.run),
        ("Backtest Agent", backtest_agent.run),
        ("Reviewer Agent", reviewer_agent.run),
    ]

    print(f"Orchestrator: running {len(stages)} stages\n")
    for name, stage_fn in stages:
        t0 = time.time()
        ok = stage_fn()
        elapsed = time.time() - t0
        status = "OK" if ok else "FAILED"
        print(f"\n>>> [{status}] {name} ({elapsed:.1f}s)\n")
        if not ok:
            print(f"Orchestrator stopped: {name} failed.")
            sys.exit(1)

    print("Orchestrator: all stages completed successfully.")
