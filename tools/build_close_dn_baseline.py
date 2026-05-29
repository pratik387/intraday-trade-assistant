#!/usr/bin/env python3
"""Manual rebuild of close_dn_overnight_long baseline + candidate snapshots.

For day-to-day operation the build runs INSIDE run_verify_exit() at 09:30
IST as a piggyback on the existing daily cron (services/execution/
overnight_handlers.py:run_verify_exit). This standalone script remains for:
  - Manual rebuilds when the verify-exit hook is skipped (e.g. weekend
    backfills, post-failure recovery).
  - One-off testing.

Both paths share services.execution.close_dn_baseline_build so the file
formats are guaranteed identical.

Usage:
    python tools/build_close_dn_baseline.py                       # today
    python tools/build_close_dn_baseline.py --session-date 2026-05-29
    python tools/build_close_dn_baseline.py --rolling-days 20 --rps 20
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--session-date", type=str, default=None,
                        help="The cron date this baseline will serve (YYYY-MM-DD). "
                             "Defaults to today.")
    parser.add_argument("--rolling-days", type=int, default=20)
    parser.add_argument("--rps", type=float, default=20.0)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--cell-min-prior-ret-pct", type=float, default=3.0,
                        help="Candidate filter — keep symbols whose prior-day "
                             "return is at least this percent (default 3.0).")
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args()

    session_date = (
        _date.fromisoformat(args.session_date) if args.session_date else _date.today()
    )

    from broker.upstox.upstox_data_client import UpstoxDataClient
    from services.execution.close_dn_baseline_build import build_baseline_and_candidates

    sdk = UpstoxDataClient()
    stats = build_baseline_and_candidates(
        sdk, session_date,
        rolling_days=args.rolling_days,
        rps=args.rps,
        concurrency=args.concurrency,
        cell_min_prior_ret_pct=args.cell_min_prior_ret_pct,
        max_symbols=args.max_symbols,
    )
    print(f"baseline_path:   {stats['baseline_path']}")
    print(f"candidates_path: {stats['candidates_path']}")
    print(f"n_symbols_with_baseline: {stats['n_symbols_with_baseline']}")
    print(f"n_candidates:            {stats['n_candidates']}")
    print(f"fetch_seconds:           {stats['fetch_seconds']}")


if __name__ == "__main__":
    main()
