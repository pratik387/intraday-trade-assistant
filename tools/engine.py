#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

# ----- repo root on sys.path -----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.util import is_trading_day  # noqa: E402

# ====== SETTINGS ======
START_DATE = "2025-08-06"   # YYYY-MM-DD
END_DATE   = "2025-08-21"   # YYYY-MM-DD (inclusive)
FROM_HHMM  = "09:10"
TO_HHMM    = "15:30"
MAIN_PATH  = ROOT / "main.py"

# parallelism: 2–4 is usually safe
MAX_WORKERS = 4
# per-task start stagger (sec) to prevent same-second logger run_id collisions
STAGGER_SEC = 2
# ======================

def _daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def _build_cmd(py_exe: str, day: date) -> List[str]:
    return [
        py_exe,
        str(MAIN_PATH),
        "--dry-run",
        "--session-date", day.isoformat(),
        "--from-hhmm", FROM_HHMM,
        "--to-hhmm",   TO_HHMM,
    ]

def _run_one_day(day: date) -> Tuple[date, int, str]:
    """Run a single day in a separate process, from repo root (CWD=ROOT)."""
    py = sys.executable
    cmd = _build_cmd(py, day)
    try:
        # launch with cwd=ROOT so relative paths (e.g., nse_all.json) work exactly like prod
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        return (day, rc, "ok" if rc == 0 else f"non-zero exit {rc}")
    except Exception as e:
        return (day, 999, f"exception: {e!r}")

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

    # Build trading-day list and report skips
    days_all = list(_daterange(start, end))
    days = [d for d in days_all if is_trading_day(d)]
    for d in days_all:
        if d not in days:
            print(f"[skip] {d} (non-trading day)")

    if not days:
        print("No trading days in the requested range.")
        return 0

    print(f"\n=== Parallel DRY RUN for {len(days)} day(s), {FROM_HHMM}-{TO_HHMM}, "
          f"max_workers={MAX_WORKERS} ===")

    results: List[Tuple[date, int, str]] = []

    # Submit with a tiny stagger so logger run_id (second precision) won't collide
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = []
        for i, day in enumerate(days):
            def _task(d=day, k=i):
                time.sleep(STAGGER_SEC * k)
                return _run_one_day(d)
            futs.append(pool.submit(_task))

        for fut in as_completed(futs):
            day, rc, msg = fut.result()
            status = "OK" if rc == 0 else "FAIL"
            print(f"[{status}] {day} -> {msg}")
            results.append((day, rc, msg))

    failures = [(d, rc, m) for d, rc, m in results if rc != 0]
    print("\n=== Summary ===")
    print(f"Completed: {len(results) - len(failures)}/{len(results)} succeeded.")
    if failures:
        for d, rc, m in failures:
            print(f"- {d}: rc={rc}, note={m}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
