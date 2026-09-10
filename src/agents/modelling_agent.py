"""
Modelling Agent (roadmap section 27): "trains baselines and sequence
models." Wraps the Feature Engineer step (src/features/
build_daily_features.py) followed by the model ladder
(src/validation/run_model_ladder.py) -- feature engineering has to run
first every time the raw data changes, so bundling it here keeps the
orchestrator from having to know that ordering dependency itself.
Run standalone: `python3 src/agents/modelling_agent.py`
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run() -> bool:
    print("=== Modelling Agent ===")
    steps = [
        PROJECT_ROOT / "src" / "features" / "build_daily_features.py",
        PROJECT_ROOT / "src" / "validation" / "run_model_ladder.py",
    ]
    for step in steps:
        result = subprocess.run([sys.executable, str(step)])
        if result.returncode != 0:
            return False
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
