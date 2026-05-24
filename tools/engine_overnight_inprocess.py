#!/usr/bin/env python
"""In-process variant of tools/engine_overnight.py — 10x+ faster.

Instead of forking a subprocess per (day, action), keep ONE MockBroker + ONE
detector instance alive across the entire date range. Saves ~25s per day in
Python startup + module init + cache rebuild.

Functionally equivalent to engine_overnight.py — same state files, same
trades.csv schema, same summary.json shape. Use when the date range exceeds
~30 days (subprocess overhead dominates).

USAGE
=====
    .venv/Scripts/python tools/engine_overnight_inprocess.py \\
        --from 2025-07-01 --to 2026-04-30

ISOLATION
=========
Uses a per-run state directory under reports/engine_overnight/<run_id>/state/
so concurrent backtests don't collide on state/overnight_slots.json. Override
the config's state_file paths via env vars before calling overnight_handlers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.util import is_trading_day


# ====== SETTINGS (overridable via --from/--to) ======
DEFAULT_START_DATE = "2025-07-01"
DEFAULT_END_DATE   = "2026-04-30"
# ====================================================


def _load_cfg() -> dict:
    with open(ROOT / "config" / "configuration.json", encoding="utf-8") as f:
        return json.load(f)


def _next_trading_day(d: date, max_lookahead: int = 7) -> date:
    cur = d + timedelta(days=1)
    for _ in range(max_lookahead):
        if is_trading_day(cur):
            return cur
        cur += timedelta(days=1)
    raise RuntimeError(f"no trading day found within {max_lookahead} days after {d}")


def _state_files_from_config(cfg: dict, state_dir: Path) -> dict:
    """Rewrite state file paths to point into state_dir (per-run isolation)."""
    overrides = {}
    setups = cfg.get("setups", {})
    for name, raw in setups.items():
        if not isinstance(raw, dict) or raw.get("mode") != "overnight":
            continue
        ca = raw.get("capital_allocation", {})
        if "state_file" in ca:
            old = ca["state_file"]
            new = str((state_dir / Path(old).name).resolve())
            ca["state_file"] = new
            overrides[old] = new
        dt = raw.get("decay_tripwire", {})
        if "state_file" in dt:
            old = dt["state_file"]
            new = str((state_dir / Path(old).name).resolve())
            dt["state_file"] = new
            overrides[old] = new
    return overrides


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="In-process overnight backtest driver")
    ap.add_argument("--from", dest="from_date", default=DEFAULT_START_DATE)
    ap.add_argument("--to", dest="to_date", default=DEFAULT_END_DATE)
    return ap.parse_args()


def run() -> int:
    args = _parse_args()
    try:
        start = date.fromisoformat(args.from_date)
        end   = date.fromisoformat(args.to_date)
    except Exception:
        print("ERROR: --from/--to must be YYYY-MM-DD", file=sys.stderr)
        return 2
    if end < start:
        print("ERROR: --to must be >= --from", file=sys.stderr)
        return 2

    # Include PID + microseconds to avoid collisions when multiple parallel
    # workers spawn simultaneously (parent engine_overnight_parallel.py).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = ROOT / "reports" / "engine_overnight" / f"inproc_{ts}_pid{os.getpid()}"
    state_dir = out_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Run output dir: {out_dir}")

    cfg = _load_cfg()
    overrides = _state_files_from_config(cfg, state_dir)
    print(f"[+] State files redirected to {state_dir.relative_to(ROOT)}: {list(overrides.values())}")

    # Build trading-day list
    days_all = []
    d = start
    while d <= end:
        days_all.append(d)
        d += timedelta(days=1)
    days = [d for d in days_all if is_trading_day(d)]
    print(f"[+] {len(days)} trading days in [{start} .. {end}]")
    if not days:
        return 0

    # Build single MockBroker with full date range (warmup back 45 cal days)
    from broker.mock.mock_broker import MockBroker
    warmup_start = days[0] - timedelta(days=45)
    broker_from = pd.Timestamp.combine(warmup_start, time(9, 15))
    broker_to   = pd.Timestamp.combine(days[-1], time(15, 30))
    slip_bps = float(cfg.get("fees_slippage_bps", 0.0))
    print(f"[+] Building MockBroker for {broker_from} .. {broker_to} (slippage_bps={slip_bps})")
    broker = MockBroker(
        path_json="nse_all.json",
        from_date=broker_from,
        to_date=broker_to,
        slippage_bps=slip_bps,
    )
    # Pre-load enriched 5m once (memoized; later calls O(1))
    print(f"[+] Pre-loading enriched 5m cache...")
    broker._load_enriched_5m()

    # In-process driver doesn't need a DryRunSDK wrapper for place_order
    # because we never call place_order; we'll override paper-mode helpers.
    # Actually run_entry calls broker.place_order — we need that to no-op.
    # Solution: monkey-patch broker.place_order.
    broker.place_order = lambda **kw: "inproc-order-id"

    # Import overnight handlers (uses the cfg dict we built with state overrides)
    from services.execution.overnight_handlers import run_entry, run_verify_exit

    print(f"\n=== Sequential overnight DRY RUN ({len(days)} days, in-process) ===\n")

    per_day_rows = []
    for i, d in enumerate(days, start=1):
        broker.set_session_date(d)
        # verify-exit at 09:30
        verify_ts = pd.Timestamp.combine(d, time(9, 30))
        s_v = run_verify_exit(cfg, broker, now_ist=verify_ts, paper_mode=True)
        # entry at 15:25
        entry_ts = pd.Timestamp.combine(d, time(15, 25))
        s_e = run_entry(cfg, broker, now_ist=entry_ts, paper_mode=True)
        per_day_rows.append({
            "day": d.isoformat(),
            "verify_settled": s_v["settled_count"],
            "verify_released": s_v["released_count"],
            "verify_orphan_t0": s_v["orphan_t0_count"],
            "entry_fired": s_e["fired_count"],
            "entry_skipped": s_e["skipped_count"],
            "entry_rejected": s_e["rejected_count"],
        })
        if i % 10 == 0 or i == len(days):
            cum_fired = sum(r["entry_fired"] for r in per_day_rows)
            cum_settled = sum(r["verify_settled"] for r in per_day_rows)
            cum_released = sum(r["verify_released"] for r in per_day_rows)
            print(f"  [{i:4d}/{len(days)}] {d}: "
                  f"fired={s_e['fired_count']} settled={s_v['settled_count']} | "
                  f"cum_fired={cum_fired} cum_settled={cum_settled} cum_released={cum_released}")

    # Final settle: next trading day after end
    final_d = _next_trading_day(days[-1])
    broker.set_session_date(final_d)
    verify_ts = pd.Timestamp.combine(final_d, time(9, 30))
    s_f = run_verify_exit(cfg, broker, now_ist=verify_ts, paper_mode=True)
    per_day_rows.append({
        "day": final_d.isoformat(),
        "verify_settled": s_f["settled_count"],
        "verify_released": s_f["released_count"],
        "verify_orphan_t0": s_f["orphan_t0_count"],
        "entry_fired": None, "entry_skipped": None, "entry_rejected": None,
    })
    print(f"\n[+] Final settle on {final_d}: settled={s_f['settled_count']} released={s_f['released_count']}")

    # Write per_day
    import csv
    pd_csv = out_dir / "per_day.csv"
    with open(pd_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_day_rows[0].keys()))
        w.writeheader()
        w.writerows(per_day_rows)

    # Aggregate trades from state files (reuse engine_overnight aggregator logic)
    state_paths = []
    for v in overrides.values():
        state_paths.append(Path(v))

    from tools.engine_overnight import _aggregate_trades
    summary = _aggregate_trades(state_paths, out_dir)

    print(f"\n=== Summary ===")
    print(f"  output: {out_dir.relative_to(ROOT)}")
    print(f"  n_trades:        {summary['n_trades']}")
    print(f"  sum_realized:    Rs {summary['sum_realized_pnl_inr']:+,.2f}")
    print(f"  wins/losses:     {summary['wins']} / {summary['losses']}")
    print(f"  gross_PF:        {summary['gross_PF']}")
    cs = summary["cohort_split"]
    print(f"  Variant B:       n={cs['variant_b_true_n']} PF={cs['variant_b_true_PF']} sum={cs['variant_b_true_sum']:+,.2f}")
    print(f"  Baseline-only:   n={cs['baseline_only_n']} PF={cs['baseline_only_PF']} sum={cs['baseline_only_sum']:+,.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
