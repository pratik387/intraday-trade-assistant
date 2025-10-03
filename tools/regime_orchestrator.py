#!/usr/bin/env python3
"""
Regime-Based Structure Evaluation Orchestrator

Runs existing engine.py 6 times with different date ranges for distinct market regimes.
Leverages existing analytics and comprehensive_run_analyzer.py.
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REGIME_CONFIGS = [
    {"name": "Strong_Uptrend", "start": "2023-12-01", "end": "2023-12-31"},
    {"name": "Shock_Down", "start": "2024-01-01", "end": "2024-01-31"},
    {"name": "Event_Driven_HighVol", "start": "2024-06-01", "end": "2024-06-30"},
    {"name": "Correction_RiskOff", "start": "2024-10-01", "end": "2024-10-31"},
    {"name": "Prolonged_Drawdown", "start": "2025-02-01", "end": "2025-02-28"},
    {"name": "Low_Vol_Range", "start": "2025-07-01", "end": "2025-07-31"}
]

def run_regime_backtest(regime):
    """Run engine.py for a single regime with modified dates"""

    engine_script = f"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.engine import run
import tools.engine as engine_module

engine_module.START_DATE = "{regime['start']}"
engine_module.END_DATE = "{regime['end']}"

if __name__ == "__main__":
    sys.exit(run())
"""

    temp_path = ROOT / f"temp_{regime['name']}.py"
    with open(temp_path, 'w') as f:
        f.write(engine_script)

    try:
        print(f"Running {regime['name']} ({regime['start']} to {regime['end']})")
        result = subprocess.run([sys.executable, str(temp_path)], cwd=str(ROOT))
        return result.returncode == 0
    finally:
        temp_path.unlink()

def main():
    """Run all regime backtests sequentially"""

    successful = 0
    for regime in REGIME_CONFIGS:
        if run_regime_backtest(regime):
            successful += 1
        else:
            print(f"FAILED: {regime['name']}")

    print(f"Completed: {successful}/{len(REGIME_CONFIGS)} regimes successful")
    return 0 if successful == len(REGIME_CONFIGS) else 1

if __name__ == "__main__":
    sys.exit(main())