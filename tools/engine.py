#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import time
import json
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
from diagnostics.diagnostics_report_builder import build_csv_from_events  # noqa: E402
from config.filters_setup import load_filters  # noqa: E402

# ====== SETTINGS ======
START_DATE = "2025-07-01"   # YYYY-MM-DD
END_DATE   = "2025-07-02"   # YYYY-MM-DD (inclusive)
MAIN_PATH  = ROOT / "main.py"

# Load time windows from config to respect time_window_block
config = load_filters()
time_windows = config.get("time_windows", {})
FROM_HHMM = time_windows.get("morning_start", "09:10")  # Start at first trading window
TO_HHMM = time_windows.get("afternoon_end", "15:30")    # End at last trading window

# parallelism: 2–4 is usually safe
MAX_WORKERS = 5
# per-task start stagger (sec) to prevent same-second logger run_id collisions
STAGGER_SEC = 2
# ======================

def _daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def _build_cmd(py_exe: str, day: date, run_prefix: str = "") -> List[str]:
    # Use forward slashes and raw strings to avoid Windows path issues
    root_path = str(ROOT).replace('\\', '/')
    main_path = str(MAIN_PATH).replace('\\', '/')

    cmd = [
        py_exe, "-c",
        f"import sys; sys.path.insert(0, r'{root_path}'); "
        f"from config.logging_config import set_global_run_prefix; "
        f"set_global_run_prefix('{run_prefix}'); "
        f"exec(open(r'{main_path}').read())",
        "--dry-run",
        "--session-date", day.isoformat(),
        "--from-hhmm", FROM_HHMM,
        "--to-hhmm", TO_HHMM,
    ]
    if run_prefix:
        cmd.extend(["--run-prefix", run_prefix])
    return cmd

def _run_one_day(day: date, run_prefix: str = "") -> Tuple[date, int, str]:
    """Run a single day in a separate process, from repo root (CWD=ROOT)."""
    py = sys.executable
    cmd = _build_cmd(py, day, run_prefix)
    try:
        # Get timestamp before subprocess to find new session
        import time
        start_time = time.time()

        # launch with cwd=ROOT so relative paths (e.g., nse_all.json) work exactly like prod
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode

        # Session registration and analytics will be handled post-run via prefix discovery
        if rc == 0:
            print(f"[engine] Subprocess completed successfully for {day}")
        else:
            print(f"[engine] Subprocess failed for {day} with return code {rc}")

        return (day, rc, "ok" if rc == 0 else f"non-zero exit {rc}")
    except Exception as e:
        return (day, 999, f"exception: {e!r}")

def _discover_sessions_by_prefix(run_prefix: str) -> list[str]:
    """Discover all session directories that match the given run prefix"""
    try:
        logs_dir = ROOT / "logs"
        if not logs_dir.exists():
            return []

        sessions = []
        for item in logs_dir.iterdir():
            if item.is_dir() and item.name.startswith(run_prefix):
                sessions.append(item.name)

        # Sort by session timestamp for consistent processing order
        sessions.sort()
        print(f"[engine] Discovered {len(sessions)} sessions with prefix '{run_prefix}': {sessions}")
        return sessions

    except Exception as e:
        print(f"[engine] ERROR discovering sessions by prefix: {e}")
        return []

def _register_session_with_run(session_id: str) -> None:
    """Log session discovery (run registration no longer needed with prefix approach)"""
    print(f"[discovery] Found session: {session_id}")

def _generate_analytics_for_session(session_id: str, log_dir: str) -> None:
    """Generate analytics for the specified session"""
    try:
        from services.logging.trading_logger import TradingLogger

        if not log_dir or not session_id:
            print("[analytics] No session info provided")
            return

        print(f"[analytics] Generating analytics for session {session_id}")
        print(f"[analytics] Processing {log_dir}")

        # Populate analytics.jsonl and performance.json from events.jsonl
        try:
            logger = TradingLogger(session_id, log_dir)
            logger.populate_analytics_from_events()
            print(f"[analytics] Enhanced analytics populated")
        except Exception as e:
            print(f"[analytics] Failed to populate enhanced analytics: {e}")

        # Generate CSV report
        csv_path = build_csv_from_events(log_dir=log_dir)
        print(f"[analytics] Diagnostics CSV written: {csv_path}")

    except Exception as e:
        print(f"[analytics] Failed to generate analytics for session {session_id}: {e}")


def _process_run_sessions(run_prefix: str) -> int:
    """Discover and process all sessions for the completed run"""
    print(f"\n=== Post-Run Processing ===")
    print(f"[+] Discovering sessions with prefix: {run_prefix}")

    # Discover all sessions created with this run prefix
    discovered_sessions = _discover_sessions_by_prefix(run_prefix)

    if not discovered_sessions:
        print("[!] No sessions found for this run")
        return 0

    # Log all discovered sessions
    print(f"[+] Processing {len(discovered_sessions)} discovered sessions...")
    for session_id in discovered_sessions:
        _register_session_with_run(session_id)

    # Process analytics for all sessions with events
    print(f"[+] Processing analytics for discovered sessions...")
    processed_count = 0

    for session_id in discovered_sessions:
        log_dir = str(ROOT / "logs" / session_id)
        events_file = ROOT / "logs" / session_id / "events.jsonl"

        if events_file.exists() and events_file.stat().st_size > 0:
            print(f"[+] Processing analytics for session {session_id}")
            _generate_analytics_for_session(session_id, log_dir)
            processed_count += 1
        else:
            print(f"[~] Skipping session {session_id} (no events)")

    print(f"[+] Processed analytics for {processed_count}/{len(discovered_sessions)} sessions")

    # Run comprehensive analysis if we have processed sessions
    if processed_count > 0:
        print(f"[+] Running comprehensive analysis for run prefix: {run_prefix}")
        try:
            import subprocess
            import sys
            result = subprocess.run([
                sys.executable, "comprehensive_run_analyzer.py", run_prefix
            ], capture_output=True, text=True, cwd=str(ROOT))

            if result.returncode == 0:
                print(f"[+] Comprehensive analysis completed successfully")
                if result.stdout:
                    print(f"[+] Analysis output: {result.stdout.strip()}")
            else:
                print(f"[!] Comprehensive analysis failed: {result.stderr.strip()}")
        except Exception as e:
            print(f"[!] Failed to run comprehensive analysis: {e}")

    return processed_count


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

    # START NEW TRADING RUN with unique ID and prefix
    import datetime
    import uuid
    unique_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"run_{uuid.uuid4().hex[:8]}_"
    run_description = f"Engine Backtest {START_DATE} to {END_DATE} [{unique_id}]"

    print(f"[+] Run prefix: {run_prefix}")
    print(f"[+] Starting new trading run: {run_description}")

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
            def _task(d=day, k=i, prefix=run_prefix):
                time.sleep(STAGGER_SEC * k)
                return _run_one_day(d, prefix)
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

    # POST-RUN SESSION PROCESSING
    processed_sessions = _process_run_sessions(run_prefix)

    # END TRADING RUN
    print(f"[+] Completed trading run: {run_description}")
    print(f"[+] Processed {processed_sessions} sessions with run prefix: {run_prefix}")

    if processed_sessions > 0:
        print(f"[+] Analytics and CSV reports generated for sessions with trading events")

    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(run())
