#!/usr/bin/env python3
"""
Regime-Based Structure Evaluation Orchestrator

Runs existing engine.py 6 times with different date ranges for distinct market regimes.
Leverages existing analytics and comprehensive_run_analyzer.py.
"""

import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Configuration
COOLDOWN_SECONDS = 0  # Pause between regimes (0 = no pause, set to 30-60 for cooldown)
PAUSE_FOR_INPUT = False  # Set True to wait for Enter key between regimes

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
        print(f"\n{'='*80}")
        print(f"Running {regime['name']} ({regime['start']} to {regime['end']})")
        print(f"{'='*80}\n")
        result = subprocess.run([sys.executable, str(temp_path)], cwd=str(ROOT))
        return result.returncode == 0
    finally:
        temp_path.unlink()

def main():
    """Run all regime backtests sequentially"""

    successful = 0
    total = len(REGIME_CONFIGS)

    for idx, regime in enumerate(REGIME_CONFIGS, 1):
        if run_regime_backtest(regime):
            successful += 1
            print(f"\n[OK] {regime['name']} completed successfully ({idx}/{total})")
        else:
            print(f"\n[FAILED] {regime['name']} ({idx}/{total})")

        # Pause between regimes if configured
        if idx < total:  # Don't pause after the last regime
            if PAUSE_FOR_INPUT:
                print(f"\nCompleted {idx}/{total} regimes. Press Enter to continue to next regime...")
                input()
            elif COOLDOWN_SECONDS > 0:
                print(f"\nCooldown: Waiting {COOLDOWN_SECONDS} seconds before next regime...")
                time.sleep(COOLDOWN_SECONDS)

    print(f"\n{'='*80}")
    print(f"Final Results: {successful}/{total} regimes successful")
    print(f"{'='*80}\n")
    return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())
