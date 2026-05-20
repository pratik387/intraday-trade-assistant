"""Phase 5: Cell sweep for extreme_vol_revert_long on Discovery.

Uses tools/methodology/cell_sweep.py v2 with pre-registered dim_pool
from brief section 5.

Workflow:
  1. Load sanity output for Discovery
  2. Configure CellSweepConfig with locked dim_pool + r_grid
  3. Run sweep, get top cells by PF
  4. Lock the winning cell via cell_sweep.lock_cell()
  5. Print cell-locked stats for OOS one-shot test (separate run)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.methodology.cell_sweep import (  # noqa: E402
    CellSweepConfig, GridEntry, run_cell_sweep, select_best_cell, lock_cell,
)


# Pre-registered dim_pool (brief section 5)
DIM_POOL = [
    "cap_segment",
    "vol_ratio_bin",
    "hhmm_bucket",
    "bar_return_bin",
]

# R-grid: T1/T2 sweep (brief section 5)
GRID: list = []
for t1 in (0.5, 1.0, 1.5):
    for t2 in (1.0, 1.5, 2.0):
        if t2 <= t1:
            continue
        for ts in (1430, 1500):
            for pm in ("partial_50_no_trail", "partial_50_be_trail"):
                GRID.append(GridEntry(
                    label=f"t1={t1}_t2={t2}_ts={ts}_pm={pm}",
                    ts_hhmm=ts,
                    partial_mode=pm,
                    t1=t1, t2=t2, sl=1.0,
                ))


def main() -> int:
    in_path = _REPO_ROOT / "reports" / "sub9_sanity" / "_extreme_vol_revert_long_trades_discovery.csv"
    if not in_path.exists():
        print(f"ABORT: {in_path} not found. Run sanity_extreme_vol_revert_long.py first.")
        return 1

    print(f"Loading Discovery trades from {in_path}...")
    df = pd.read_csv(in_path, parse_dates=["signal_date"])
    df["signal_date"] = df["signal_date"].dt.date
    print(f"  {len(df):,} trades loaded")
    if df.empty:
        print("ABORT: no trades")
        return 1

    print(f"\nCohort summary:")
    print(f"  n_total = {len(df):,}")
    print(f"  cap_segments: {df['cap_segment'].value_counts().to_dict()}")
    print(f"  vol_ratio_bin: {df['vol_ratio_bin'].value_counts().to_dict()}")
    print(f"  hhmm_bucket: {df['hhmm_bucket'].value_counts().to_dict()}")
    print(f"  bar_return_bin: {df['bar_return_bin'].value_counts().to_dict()}")
    print(f"  exit_reason: {df['exit_reason'].value_counts().to_dict()}")

    # Aggregate stats (sanity check)
    gross_total = df["realized_pnl_inr"].sum()
    fee_total = df["fee_inr"].sum()
    net_total = df["net_pnl_inr"].sum()
    print(f"\nAggregate Discovery economics (NO leverage, NO cell-lock):")
    print(f"  Gross:   Rs {gross_total:+,.0f}")
    print(f"  Fees:    Rs {fee_total:+,.0f}")
    print(f"  Net:     Rs {net_total:+,.0f}")
    print(f"  Per-trade avg net: Rs {net_total/len(df):+,.2f}")

    cfg = CellSweepConfig(
        side="LONG",
        target_unit="R",
        grid=GRID,
        dim_pool=DIM_POOL,
        k_max=2,
        n_min_floor=200,    # Phase 5 spec: n>=200 floor
        pf_min_floor=1.10,
        n_min_ship=500,
        pf_min_ship=1.20,
    )

    print(f"\nRunning cell sweep ({len(GRID)} grid entries x dim 1D+2D)...")
    results = run_cell_sweep(df, cfg)
    if results.empty:
        print("\nVERDICT: no cell passed floor (n>=200, PF>=1.10). KILL.")
        return 0

    print(f"\n{len(results):,} cells passed floor (n>=200, PF>=1.10)")
    print("\nTop 20 cells by PF:")
    print(results.head(20).to_string(index=False))

    best = select_best_cell(results, cfg, require_ship_eligible=True)
    if best is None:
        print("\nVERDICT: no cell meets ship-eligibility (n>=500, PF>=1.20). KILL.")
        return 0

    print(f"\n*** Best ship-eligible cell ***")
    for k, v in best.items():
        print(f"  {k}: {v}")

    lock_path = _REPO_ROOT / "tools" / "sub9_research" / "extreme_vol_revert_long_cell_lock.json"
    lock_cell(
        best, setup_name="extreme_vol_revert_long",
        window_label="Discovery",
        output_path=lock_path,
        extra_metadata={
            "discovery_window": "2023-01-02 to 2024-06-30",
            "brief": "specs/2026-05-20-brief-extreme_vol_revert_long.md",
            "n_discovery_trades": len(df),
            "n_passing_cells": len(results),
        },
    )
    print(f"\nLock written: {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
