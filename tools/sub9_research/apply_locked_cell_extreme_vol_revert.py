"""Apply the Discovery-locked cell to OOS + Holdout trade ledgers.

Workflow:
  1. Read tools/sub9_research/extreme_vol_revert_long_cell_lock.json
  2. Load OOS + Holdout trade CSVs
  3. Apply the locked filter cell + R-tuple
  4. Compute PF / win-rate / aggregate net per window
  5. Report Disc / OOS / HO stationarity check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _profit_factor(pnls: pd.Series) -> float:
    gross = float(pnls[pnls > 0].sum())
    loss = float(-pnls[pnls < 0].sum())
    if loss <= 0:
        return float("inf") if gross > 0 else 1.0
    return gross / loss


def _apply_cell_filter(trades: pd.DataFrame, cell: dict) -> pd.DataFrame:
    """Filter trades to the locked cell's dim values."""
    dims = cell.get("dims", [])
    cell_label = cell.get("cell_label", "")
    # cell_label is like "cap_segment=mid_cap | hhmm_bucket=morning_0930_1100"
    # parse it back
    filter_dict = {}
    for part in cell_label.split("|"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            filter_dict[k.strip()] = v.strip()
    out = trades.copy()
    for k, v in filter_dict.items():
        if k not in out.columns:
            print(f"  WARN: filter dim '{k}' not in trades df; skipping")
            continue
        out = out[out[k].astype(str) == v]
    return out


def _evaluate_window(trades: pd.DataFrame, label: str) -> Dict[str, float]:
    if trades.empty:
        return {"window": label, "n": 0, "pf_gross": 0, "pf_net": 0, "wr": 0, "net": 0}
    n = len(trades)
    pf_gross = _profit_factor(trades["realized_pnl_inr"])
    pf_net = _profit_factor(trades["net_pnl_inr"])
    wr = float((trades["net_pnl_inr"] > 0).mean() * 100.0)
    net = float(trades["net_pnl_inr"].sum())
    return {
        "window": label, "n": n,
        "pf_gross": round(pf_gross, 3),
        "pf_net": round(pf_net, 3),
        "wr": round(wr, 1),
        "net": round(net, 0),
        "mean_per_trade": round(net / n, 2) if n else 0,
    }


def main() -> int:
    lock_path = _REPO_ROOT / "tools" / "sub9_research" / "extreme_vol_revert_long_cell_lock.json"
    if not lock_path.exists():
        print(f"ABORT: lock file not found: {lock_path}")
        print("Run run_cell_sweep_extreme_vol_revert.py first.")
        return 1

    with lock_path.open(encoding="utf-8") as f:
        lock = json.load(f)
    cell = lock["selected_cell"]
    print(f"=== Locked Cell ===")
    for k, v in cell.items():
        print(f"  {k}: {v}")

    # Filter each window
    sanity_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    rows = []
    for window_label in ("discovery", "oos", "holdout"):
        csv_path = sanity_dir / f"_extreme_vol_revert_long_trades_{window_label}.csv"
        if not csv_path.exists():
            print(f"\n[WARN] Missing: {csv_path}")
            continue
        all_trades = pd.read_csv(csv_path, parse_dates=["signal_date"])
        all_trades["signal_date"] = all_trades["signal_date"].dt.date
        filtered = _apply_cell_filter(all_trades, cell)
        # Also filter by R-grid: this is the trickier part — for now, accept
        # any T1/T2 combo (the sanity emits fixed T1=1R/T2=2R; cell-locked
        # R-tuple would require re-walking, deferred)
        print(f"\n[{window_label}] all={len(all_trades):,}  cell-filtered={len(filtered):,}")
        stats = _evaluate_window(filtered, window_label)
        print(f"  n={stats['n']}  PF_gross={stats['pf_gross']}  PF_net={stats['pf_net']}  "
              f"WR={stats['wr']}%  NET=Rs {stats['net']:+,.0f}  "
              f"per-trade=Rs {stats['mean_per_trade']:+,.2f}")
        rows.append(stats)

    # Stationarity check
    print(f"\n=== Stationarity Check ===")
    pf_nets = [r["pf_net"] for r in rows if r["n"] >= 30]
    if len(pf_nets) >= 2:
        spread = max(pf_nets) - min(pf_nets)
        print(f"  PF_net range across windows: {min(pf_nets):.3f} to {max(pf_nets):.3f}  spread={spread:.3f}")
        if spread <= 0.30:
            print(f"  -> PASS stationarity (spread <= 0.30)")
        else:
            print(f"  -> FAIL stationarity (spread > 0.30 = regime-conditional)")

    # Phase 5 ship gate
    print(f"\n=== Phase 5 Ship Gate ===")
    for r in rows:
        passes = (r["n"] >= 200 and r["pf_net"] >= 1.10)
        print(f"  {r['window']:<10} n={r['n']:>4}  PF_net={r['pf_net']}  "
              f"{'PASS' if passes else 'FAIL'} (need n>=200, PF_net>=1.10)")

    # Save summary
    summary_df = pd.DataFrame(rows)
    out_path = sanity_dir / "_extreme_vol_revert_long_cell_locked_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSummary saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
