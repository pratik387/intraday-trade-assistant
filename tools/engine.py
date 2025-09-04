#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.util import is_trading_day

# ====== HARD-CODED SETTINGS ======
START_DATE = "2025-08-01"   # YYYY-MM-DD
END_DATE   = "2025-08-02"   # YYYY-MM-DD (inclusive)
FROM_HHMM  = "09:10"        # intraday window start
TO_HHMM    = "15:30"        # intraday window end
MAIN_PATH  = Path(__file__).resolve().parents[1] / "main.py"  # adjust if needed
# =================================

def _daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def run() -> int:
    if not MAIN_PATH.exists():
        print(f"ERROR: main.py not found at {MAIN_PATH}", file=sys.stderr)
        return 2

    try:
        start = date.fromisoformat(START_DATE)
        end   = date.fromisoformat(END_DATE)
    except Exception:
        print("ERROR: START_DATE/END_DATE must be YYYY-MM-DD", file=sys.stderr)
        return 2

    if end < start:
        print("ERROR: END_DATE must be >= START_DATE", file=sys.stderr)
        return 2

    py = sys.executable

    for day in _daterange(start, end):
        # Skip non-trading days (weekends + official NSE holidays)
        if not is_trading_day(day):
            print(f"[skip] {day} (non-trading day)")
            continue

        print(f"\n=== DRY RUN {day} {FROM_HHMM}-{TO_HHMM} ===")
        cmd = [
            py,
            str(MAIN_PATH),
            "--dry-run",
            "--session-date", day.isoformat(),
            "--from-hhmm", FROM_HHMM,
            "--to-hhmm",   TO_HHMM,
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[stop] {day} exited with code {rc}", file=sys.stderr)
            return rc

    print("\nAll requested days completed successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
