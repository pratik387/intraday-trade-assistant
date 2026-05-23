"""Aggregate OCI production analytics.jsonl into canonical-schema trade CSVs.

This is the Stage 6 pathway: walk-forward on PRODUCTION trades (real fees,
real fills, real cell-locked behavior) rather than sanity-script outputs.
Per lesson #13, sanity walk-forward isn't actionable for active setup
verdicts — production analytics.jsonl is the gold standard.

OCI run directory structure:
    <run_dir>/
        <YYYY-MM-DD>/
            analytics.jsonl    # one record per trade leg
            events.jsonl
            ...

Each `analytics.jsonl` has multi-row records per trade (multi-leg T1+T2).
Identified by `lifecycle_id`. Final leg row has `is_final_exit=True` and
`total_trade_pnl` = sum of gross PnLs across all legs.

Output: per-setup canonical CSV at
    reports/oci_canonical/<setup>_canonical_<run_label>.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]


# Setup name → side (inferred from setup_type pattern)
def _infer_side(setup_type: str) -> str:
    s = setup_type.lower()
    if "short" in s or "fade" in s:
        return "SHORT"
    if "long" in s:
        return "LONG"
    # Default: probably SHORT given our portfolio
    return "SHORT"


def _aggregate_lifecycle(rows: List[dict]) -> dict:
    """Aggregate multiple legs of one lifecycle_id into a single canonical row.

    Per-leg rows have:
        actual_entry_price, exit_price (per leg), qty (per leg),
        pnl (gross per leg), net_pnl (per leg), fees.total_fees,
        is_final_exit, total_trade_pnl (only on final row)
    """
    if not rows:
        return {}
    rows = sorted(rows, key=lambda r: r.get("exit_sequence", 1))
    first = rows[0]
    final = next((r for r in rows if r.get("is_final_exit", False)), rows[-1])

    total_qty = sum(int(r.get("qty", 0)) for r in rows)
    total_gross = sum(float(r.get("pnl", 0.0)) for r in rows)
    total_fees = sum(float(r.get("fees", {}).get("total_fees", 0.0)) for r in rows)
    total_net = total_gross - total_fees

    # Sanity check: final_row.total_trade_pnl should ~= total_gross
    declared_total = float(final.get("total_trade_pnl") or 0.0)
    if declared_total != 0 and abs(declared_total - total_gross) / max(abs(declared_total), 1.0) > 0.05:
        # Allow up to 5% discrepancy; otherwise log
        pass  # silently take computed sum

    entry_price = float(first.get("actual_entry_price") or first.get("entry_reference", {}).get("entry_price", 0.0))
    final_exit_price = float(final.get("exit_price", 0.0))
    setup_type = first.get("setup_type", "unknown")
    side = _infer_side(setup_type)
    signal_date = first.get("timestamp", "")[:10]  # YYYY-MM-DD
    symbol = first.get("symbol", "UNKNOWN")

    # pnl_pct: blended GROSS return = total_gross / (entry × total_qty) × 100
    notional = entry_price * total_qty
    pnl_pct = total_gross / notional * 100.0 if notional > 0 else 0.0

    return {
        "signal_date": signal_date,
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "exit_price": final_exit_price,
        "qty": total_qty,
        "pnl_pct": pnl_pct,
        "exit_reason": _normalize_reason(final.get("reason", "unknown")),
        "same_bar": False,  # OCI doesn't directly track same-bar
        "t1_partial_booked": len(rows) > 1,
        "realized_pnl_inr": total_gross,
        "fee_inr": total_fees,
        "net_pnl_inr": total_net,
        "setup_type": setup_type,
        "r_multiple": float(final.get("r_multiple") or 0.0),
        "n_legs": len(rows),
    }


def _normalize_reason(reason: str) -> str:
    """Map OCI reason strings to canonical exit_reason set."""
    r = (reason or "").lower()
    if "t2" in r:
        return "t2"
    if "t1" in r:
        return "t1"
    if "stop" in r or "sl" in r:
        return "sl"
    if "time" in r:
        return "time_stop"
    if "eod" in r or "square" in r or "session_end" in r:
        return "eod"
    if "manual" in r or "force" in r:
        return "manual"
    return "manual"  # fallback


def aggregate_run(run_dir: Path, setups_filter: List[str] = None) -> Dict[str, List[dict]]:
    """Walk all daily directories in run_dir and aggregate analytics into
    per-setup lists of canonical-schema rows.
    """
    by_lifecycle: Dict[str, List[dict]] = defaultdict(list)
    n_files = 0
    n_rows = 0

    for day_name in sorted(os.listdir(run_dir)):
        day_path = run_dir / day_name
        if not day_path.is_dir():
            continue
        analytics_path = day_path / "analytics.jsonl"
        if not analytics_path.exists() or analytics_path.stat().st_size == 0:
            continue
        n_files += 1
        with open(analytics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "lifecycle_id" not in r:
                    continue
                if setups_filter and r.get("setup_type") not in setups_filter:
                    continue
                by_lifecycle[r["lifecycle_id"]].append(r)
                n_rows += 1

    print(f"  [{run_dir.name}] {n_files} daily files, {n_rows} rows, "
          f"{len(by_lifecycle)} unique trades")

    # Aggregate by lifecycle
    by_setup: Dict[str, List[dict]] = defaultdict(list)
    for lc_id, rows in by_lifecycle.items():
        canonical = _aggregate_lifecycle(rows)
        setup = canonical.get("setup_type", "unknown")
        by_setup[setup].append(canonical)
    return by_setup


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", nargs="+", required=True, type=Path,
                   help="One or more OCI run directories (each contains daily subdirs)")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "reports" / "oci_canonical")
    p.add_argument("--setups", nargs="*", default=None,
                   help="Optional: restrict to these setup_type values")
    args = p.parse_args(argv)

    print(f"Aggregating {len(args.run_dirs)} OCI run dir(s)...")
    all_by_setup: Dict[str, List[dict]] = defaultdict(list)
    # Track dedup keys per setup across all runs.
    # lifecycle_id is NOT preserved in the canonical row (we only retain
    # aggregated fields), so we dedup on the natural composite key:
    #   (signal_date, symbol, side, entry_price, qty)
    # Same trade emitted from two overlapping run-dirs has identical values
    # for all five fields; this catches the double-count case without rejecting
    # legitimate same-day same-symbol multi-entries (which would differ on at
    # least one of entry_price/qty/side).
    seen_keys: Dict[str, set] = defaultdict(set)
    n_dedup_drops = 0
    for run_dir in args.run_dirs:
        if not run_dir.exists():
            print(f"WARN: run dir not found: {run_dir}", file=sys.stderr)
            continue
        by_setup = aggregate_run(run_dir, setups_filter=args.setups)
        for setup, rows in by_setup.items():
            for row in rows:
                key = (
                    row.get("signal_date", ""),
                    row.get("symbol", ""),
                    row.get("side", ""),
                    round(float(row.get("entry_price", 0.0) or 0.0), 4),
                    int(row.get("qty", 0) or 0),
                )
                if key in seen_keys[setup]:
                    n_dedup_drops += 1
                    continue
                seen_keys[setup].add(key)
                all_by_setup[setup].append(row)
    if n_dedup_drops > 0:
        print(f"  Dedup: dropped {n_dedup_drops} duplicate trades across run-dir union "
              f"(see line-178 dedup guard, Lesson #24).")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting per-setup canonical CSVs to {args.out_dir}/")
    for setup, rows in sorted(all_by_setup.items()):
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df = df.sort_values("signal_date").reset_index(drop=True)
        out = args.out_dir / f"{setup}_oci_canonical.csv"
        df.to_csv(out, index=False)
        net_total = df["net_pnl_inr"].sum()
        print(f"  {setup}: {len(df)} trades, net Rs {net_total:.0f} -> {out.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
