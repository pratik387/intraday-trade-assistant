#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess extracted backtest - generate analytics.jsonl from events.jsonl
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from services.logging.trading_logger import TradingLogger
from diagnostics.diagnostics_report_builder import build_csv_from_events

def postprocess_backtest(backtest_dir):
    """
    Postprocess a backtest directory - generate analytics for all sessions

    Args:
        backtest_dir: Path to extracted backtest directory (should contain session subdirs)
    """
    backtest_path = Path(backtest_dir)

    if not backtest_path.exists():
        print(f"ERROR: Backtest directory not found: {backtest_dir}")
        return 1

    # Find all session directories (YYYY-MM-DD format)
    session_dirs = sorted([d for d in backtest_path.iterdir() if d.is_dir() and len(d.name) == 10 and d.name[4] == '-' and d.name[7] == '-'])

    if not session_dirs:
        print(f"ERROR: No session directories found in {backtest_dir}")
        return 1

    print(f"Found {len(session_dirs)} session directories")
    print(f"Processing analytics...")

    processed = 0
    skipped = 0

    for session_dir in session_dirs:
        session_id = session_dir.name
        events_file = session_dir / "events.jsonl"

        if not events_file.exists() or events_file.stat().st_size == 0:
            print(f"[SKIP] {session_id} - no events")
            skipped += 1
            continue

        try:
            # Generate analytics.jsonl from events.jsonl
            logger = TradingLogger(session_id, str(session_dir))
            logger.populate_analytics_from_events()

            # Generate CSV report
            build_csv_from_events(log_dir=str(session_dir))

            print(f"[OK] {session_id}")
            processed += 1

        except Exception as e:
            print(f"[ERROR] {session_id}: {e}")

    print(f"\nProcessing complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(session_dirs)}")

    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python postprocess_extracted_backtest.py <backtest_dir>")
        print("Example: python postprocess_extracted_backtest.py backtest_20251107-083559_extracted/20251107-083559_full/20251107-083559")
        sys.exit(1)

    backtest_dir = sys.argv[1]
    sys.exit(postprocess_backtest(backtest_dir))
