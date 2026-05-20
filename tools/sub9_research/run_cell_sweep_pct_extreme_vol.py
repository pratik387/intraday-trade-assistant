"""Cell sweep over (SL_pct, T1_pct, T2_pct, time_stop, partial_mode) using
target_unit='pct'. Tests whether a different SL distance rescues
extreme_vol_revert_long from Phase 5 R-mode kill.

Derives mfe_pct + mae_pct from existing discovery sanity output:
  mfe_pct = mfe_r * R_per_share / entry_price * 100
  mae_pct = mae_r * R_per_share / entry_price * 100

Sweeps:
  SL_pct in {0.3, 0.5, 0.7, 1.0, 1.5}    (stop distance as % of entry)
  T1_pct in {0.2, 0.5, 0.7, 1.0}         (T1 target as % favorable)
  T2_pct in {0.5, 1.0, 1.5, 2.0}         (T2 target, with T2 > T1)
  time_stop in {14:30, 15:00}
  partial_mode in {partial_50_no_trail, partial_50_be_trail, all_in}
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.methodology.cell_sweep import (  # noqa: E402
    CellSweepConfig, GridEntry, run_cell_sweep, select_best_cell,
)


DIM_POOL = ["cap_segment", "vol_ratio_bin", "hhmm_bucket", "bar_return_bin"]


def build_grid():
    grid = []
    sl_pcts = [0.3, 0.5, 0.7, 1.0, 1.5]
    t1_pcts = [0.2, 0.5, 0.7, 1.0]
    t2_pcts = [0.5, 1.0, 1.5, 2.0]
    ts_list = [1430, 1500]
    pm_list = ["partial_50_no_trail", "partial_50_be_trail", "all_in"]
    for sl in sl_pcts:
        for t1 in t1_pcts:
            for t2 in t2_pcts:
                if t2 <= t1:
                    continue
                for ts in ts_list:
                    for pm in pm_list:
                        grid.append(GridEntry(
                            label=f"sl={sl}_t1={t1}_t2={t2}_ts={ts}_pm={pm}",
                            ts_hhmm=ts,
                            partial_mode=pm,
                            t1=t1, t2=t2, sl=sl,
                        ))
    return grid


def main() -> int:
    in_path = _REPO_ROOT / "reports" / "sub9_sanity" / "_extreme_vol_revert_long_trades_discovery.csv"
    if not in_path.exists():
        print(f"ABORT: {in_path} not found.")
        return 1

    print(f"Loading Discovery trades from {in_path}...")
    df = pd.read_csv(in_path, parse_dates=["signal_date"])
    df["signal_date"] = df["signal_date"].dt.date
    print(f"  {len(df):,} trades loaded")

    # Derive mfe_pct + mae_pct from existing fields
    print("  Deriving mfe_pct + mae_pct from mfe_r * R_per_share / entry...")
    df["mfe_pct"] = (df["mfe_r"] * df["R_per_share"] / df["entry_price"] * 100.0).clip(lower=0.0)
    df["mae_pct"] = (df["mae_r"] * df["R_per_share"] / df["entry_price"] * 100.0).clip(lower=0.0)

    grid = build_grid()
    print(f"\nGrid size: {len(grid)} entries")

    cfg = CellSweepConfig(
        side="LONG",
        target_unit="pct",
        grid=grid,
        dim_pool=DIM_POOL,
        k_max=2,
        n_min_floor=200,
        pf_min_floor=1.10,
        n_min_ship=500,
        pf_min_ship=1.20,
    )

    print(f"\nRunning cell sweep ({len(grid)} grid x dim 1D+2D)...")
    print("  (This may take 5-15 minutes — each grid entry runs the full dim sweep)\n")
    results = run_cell_sweep(df, cfg)
    if results.empty:
        print("\nVERDICT: no cell passed floor (n>=200, PF>=1.10) across pct sweep. KILL CONFIRMED.")
        return 0

    print(f"\n{len(results):,} cells passed floor (n>=200, PF>=1.10)")
    print("\nTop 30 cells by PF (with pct-mode SL/T1/T2 sweep):")
    cols_to_show = ["cell_label", "grid_label", "t1", "t2", "sl", "ts_hhmm", "partial_mode",
                    "n", "pf", "wr_pct", "net_pnl_inr", "expectancy_inr"]
    cols_avail = [c for c in cols_to_show if c in results.columns]
    print(results.head(30)[cols_avail].to_string(index=False))

    best = select_best_cell(results, cfg, require_ship_eligible=True)
    if best is None:
        print("\nVERDICT: no cell meets ship-eligibility (n>=500, PF>=1.20). KILL CONFIRMED.")
        # But also report the strongest cell that passed floor for reference
        top1 = results.iloc[0]
        print(f"\nStrongest floor-passing cell:")
        for k in cols_avail:
            print(f"  {k}: {top1.get(k)}")
        return 0

    print(f"\n*** Best ship-eligible cell (pct sweep) ***")
    for k, v in best.items():
        print(f"  {k}: {v}")

    out_path = _REPO_ROOT / "reports" / "sub9_sanity" / "_extreme_vol_revert_long_pct_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
