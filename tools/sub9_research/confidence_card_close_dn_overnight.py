"""Lesson #15 confidence framework on top 5 close_dn_overnight_long cells.

Per cell:
  - Component 1: Bootstrap BCa CI on aggregate PF / expectancy / WR
  - Component 2: 7-regime breakdown
  - Component 3: Harvey-Liu haircut with M = 58 (ship-eligible cells with per-day <= 10)

The 5 cells were selected from the run_cell_sweep_close_dn_overnight.py output
on Discovery, then filtered by Disc per-day median <= 10.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.methodology.confidence.bootstrap_ci import compute_aggregate_ci
from tools.methodology.confidence.regime_breakdown import (
    compute_per_regime_stats, format_regime_table,
)
from tools.methodology.confidence.selection_bias import (
    build_daily_equity_curve, harvey_liu_haircut,
)


M_TESTED = 58  # ship-eligible cells with per-day median <= 10

CANDIDATES = [
    {
        "id": 1,
        "label": "neg0.9_to_neg1.0 × volume_z=high",
        "filters": {
            "signed_vol_ratio_bin": "neg0.9_to_neg1.0",
            "closing_30m_volume_z_bin": "high",
        },
    },
    {
        "id": 2,
        "label": "neg0.9_to_neg1.0 (1D)",
        "filters": {"signed_vol_ratio_bin": "neg0.9_to_neg1.0"},
    },
    {
        "id": 3,
        "label": "neg0.9_to_neg1.0 × volume_z=extreme",
        "filters": {
            "signed_vol_ratio_bin": "neg0.9_to_neg1.0",
            "closing_30m_volume_z_bin": "extreme",
        },
    },
    {
        "id": 4,
        "label": "neg0.9_to_neg1.0 × prior_ret=flat",
        "filters": {
            "signed_vol_ratio_bin": "neg0.9_to_neg1.0",
            "prior_day_return_bin": "flat",
        },
    },
    {
        "id": 5,
        "label": "volume_z=extreme × prior_ret=up_gt_3pct",
        "filters": {
            "closing_30m_volume_z_bin": "extreme",
            "prior_day_return_bin": "up_gt_3pct",
        },
    },
]


def _apply(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for k, v in filters.items():
        df = df[df[k] == v]
    return df.reset_index(drop=True)


def _pf(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    return (g / l) if l > 0 else float("inf")


def render_card(cand: dict):
    print("\n" + "=" * 100)
    print(f"CANDIDATE #{cand['id']}: {cand['label']}")
    print(f"Filters: {cand['filters']}")
    print(f"M (Harvey-Liu): {M_TESTED}  (ship-eligible cells with Disc per-day median <= 10)")
    print("=" * 100)

    # Load + filter each window
    parts = []
    for window in ("discovery", "oos", "holdout"):
        p = _REPO_ROOT / "reports" / "sub9_sanity" / f"_close_dn_overnight_long_trades_{window}.csv"
        df = pd.read_csv(p)
        sub = _apply(df, cand["filters"])
        if sub.empty:
            print(f"  [{window:>9}] empty after filter")
            continue
        sub["window"] = window
        net = sub["net_pnl_inr"].sum()
        wins = (sub["net_pnl_inr"] > 0).sum()
        losses = (sub["net_pnl_inr"] < 0).sum()
        pf = _pf(sub["net_pnl_inr"])
        print(f"  [{window:>9}] n={len(sub):>5}  PF={pf:.3f}  WR={wins/max(1,len(sub))*100:.2f}%  "
              f"NET=Rs{net:+,.0f}  exp=Rs{net/max(1,len(sub)):+.2f}/trade")
        parts.append(sub)

    if not parts:
        return

    pooled = pd.concat(parts, ignore_index=True)
    pooled["signal_date"] = pd.to_datetime(pooled["signal_date"])
    print(f"  Pooled n={len(pooled):,} ({pooled['signal_date'].min().date()} to {pooled['signal_date'].max().date()})")

    # Component 1: aggregate CI
    print()
    print("  --- Component 1: Aggregate BCa CI ---")
    agg = compute_aggregate_ci(pooled)
    for name, info in agg.items():
        print(f"    {name}: point={info.point_estimate:.4f}  "
              f"[CI {info.ci_lower:.4f}, {info.ci_upper:.4f}]")

    # Component 2: regime
    p_for_r = pooled.copy()
    p_for_r["signal_date"] = p_for_r["signal_date"].dt.date
    print()
    print("  --- Component 2: Per-regime breakdown ---")
    rstats = compute_per_regime_stats(p_for_r)
    print(format_regime_table(rstats))

    # Component 3: Harvey-Liu
    daily = build_daily_equity_curve(p_for_r)
    print()
    print("  --- Component 3: Harvey-Liu haircut ---")
    print(f"    Daily observations: {len(daily)}")
    for method in ("Bonferroni", "BHY"):
        h = harvey_liu_haircut(f"cand_{cand['id']}", daily, M=M_TESTED, method=method)
        verdict = "POSITIVE" if h.adjusted_sharpe > 0 else "NEGATIVE"
        print(f"    [{method:>10}] raw_SR={h.raw_sharpe:+.3f}  adj_SR={h.adjusted_sharpe:+.3f}  "
              f"haircut={h.haircut_pct:.1f}%  -> {verdict}")


def main() -> int:
    for cand in CANDIDATES:
        render_card(cand)
    print("\n" + "=" * 100)
    print("DONE — Inspect: PF CI lower > 1.0, regime breadth, adj SR > 0 after Bonferroni haircut.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
