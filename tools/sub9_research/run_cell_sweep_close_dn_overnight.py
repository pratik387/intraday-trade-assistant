"""Phase 5 cell sweep: close_dn_overnight_long.

Brief: specs/2026-05-21-brief-close_dn_overnight_long.md
Sanity: tools/sub9_research/sanity_close_dn_overnight_long.py

This setup is single-entry-single-exit (15:25 close → next-day 09:15 open),
so there's no R-grid/SL/T1/T2 to sweep. The cell sweep is purely about
filter-dimension combinations on the sanity output.

Pre-registered dim_pool (brief section 5):
  - cap_segment
  - signed_vol_ratio_bin
  - closing_30m_volume_z_bin
  - prior_day_return_bin

(news_proximity was tested in Phase 4 preview — PF for 'clear' (1.59) vs
'within_1day_earnings' (1.66) on HO showed no edge from the filter. Dropped
from sweep to reduce dim count.)

Acceptance gates (brief section 10):
  1. Disc cell n>=500, PF>=1.20
  2. OOS PF>=1.10 with WR within 10pp of Disc
  3. HO PF>=1.10
  4. Stationarity max-min PF<=0.30 across (Disc, OOS, HO)

Output: cross-window scan CSV ranked by stationarity.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]


DIM_POOL = [
    "cap_segment",
    "signed_vol_ratio_bin",
    "closing_30m_volume_z_bin",
    "prior_day_return_bin",
]
N_MIN_FLOOR = 200
PF_MIN_FLOOR = 1.10
N_MIN_SHIP = 500
PF_MIN_SHIP = 1.20

# Acceptance gates
OOS_PF_MIN = 1.10
HO_PF_MIN = 1.10
WR_DRIFT_MAX_PP = 10.0
STATIONARITY_MAX = 0.30


def _profit_factor(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    if l <= 0:
        return float("inf") if g > 0 else 1.0
    return g / l


def _cell_stats(sub: pd.DataFrame) -> dict:
    s = sub["net_pnl_inr"]
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    pf = (g / l) if l > 0 else float("inf")
    return {
        "n": int(len(s)),
        "pf": float(pf),
        "wr": float((s > 0).mean() * 100.0),
        "net": float(s.sum()),
        "exp": float(s.mean() if len(s) else 0.0),
    }


def _sweep_disc(disc: pd.DataFrame, k_max: int = 3) -> pd.DataFrame:
    """Mine cells on Discovery — only ship-eligible ones survive."""
    rows = []
    for k in range(1, k_max + 1):
        for combo in combinations(DIM_POOL, k):
            agg = disc.groupby(list(combo), observed=True, dropna=False)["net_pnl_inr"]
            for key, s in agg:
                key_tuple = key if isinstance(key, tuple) else (key,)
                n = len(s)
                if n < N_MIN_FLOOR:
                    continue
                pf = _profit_factor(s)
                if pf < PF_MIN_FLOOR:
                    continue
                cell_label = " | ".join(f"{c}={v}" for c, v in zip(combo, key_tuple))
                rows.append({
                    "dims": str(list(combo)),
                    "cell_label": cell_label,
                    "k_dims": k,
                    "d_n": n,
                    "d_pf": pf,
                    "d_wr": float((s > 0).mean() * 100.0),
                    "d_net": float(s.sum()),
                    "d_exp": float(s.mean()),
                    "_dim_keys": dict(zip(combo, key_tuple)),  # internal use
                })
    return pd.DataFrame(rows)


def _apply_cell(df: pd.DataFrame, dim_keys: dict) -> pd.DataFrame:
    sub = df
    for k, v in dim_keys.items():
        sub = sub[sub[k] == v]
    return sub


def main() -> int:
    sanity_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    disc = pd.read_csv(sanity_dir / "_close_dn_overnight_long_trades_discovery.csv")
    oos = pd.read_csv(sanity_dir / "_close_dn_overnight_long_trades_oos.csv")
    ho = pd.read_csv(sanity_dir / "_close_dn_overnight_long_trades_holdout.csv")
    print(f"Loaded sanity: Disc={len(disc):,}  OOS={len(oos):,}  HO={len(ho):,}")
    print(f"Discovery aggregate PF: {_profit_factor(disc['net_pnl_inr']):.3f}")
    print(f"OOS aggregate PF:       {_profit_factor(oos['net_pnl_inr']):.3f}")
    print(f"HO aggregate PF:        {_profit_factor(ho['net_pnl_inr']):.3f}")
    print()

    print(f"Sweeping Discovery cells (1D + 2D + 3D combos of {DIM_POOL})...")
    sweep = _sweep_disc(disc, k_max=3)
    print(f"  Total cells passing floor (n>={N_MIN_FLOOR}, PF>={PF_MIN_FLOOR}): {len(sweep):,}")
    ship = sweep[(sweep["d_n"] >= N_MIN_SHIP) & (sweep["d_pf"] >= PF_MIN_SHIP)].reset_index(drop=True)
    print(f"  Ship-eligible (n>={N_MIN_SHIP}, PF>={PF_MIN_SHIP}): {len(ship):,}")
    print()

    print("Cross-applying ship-eligible Disc cells to OOS + HO...")
    rows = []
    for _, row in ship.iterrows():
        dim_keys = row["_dim_keys"]
        o_sub = _apply_cell(oos, dim_keys)
        h_sub = _apply_cell(ho, dim_keys)
        o_st = _cell_stats(o_sub)
        h_st = _cell_stats(h_sub)
        pfs = [row["d_pf"], o_st["pf"], h_st["pf"]]
        wr_drift = abs(o_st["wr"] - row["d_wr"])
        stationarity = max(pfs) - min(pfs)
        rows.append({
            "cell_label": row["cell_label"],
            "dims": row["dims"],
            "k_dims": row["k_dims"],
            "d_n": row["d_n"], "d_pf": row["d_pf"], "d_wr": row["d_wr"], "d_exp": row["d_exp"],
            "o_n": o_st["n"], "o_pf": o_st["pf"], "o_wr": o_st["wr"], "o_exp": o_st["exp"],
            "h_n": h_st["n"], "h_pf": h_st["pf"], "h_wr": h_st["wr"], "h_exp": h_st["exp"],
            "min_pf": min(pfs),
            "stationarity_dpf": stationarity,
            "wr_drift_dpp": wr_drift,
            "gate_oos_pf": o_st["pf"] >= OOS_PF_MIN,
            "gate_oos_wr": wr_drift <= WR_DRIFT_MAX_PP,
            "gate_ho_pf": h_st["pf"] >= HO_PF_MIN,
            "gate_stationarity": stationarity <= STATIONARITY_MAX,
        })

    cross = pd.DataFrame(rows)
    cross["all_gates_pass"] = (
        cross["gate_oos_pf"] & cross["gate_oos_wr"]
        & cross["gate_ho_pf"] & cross["gate_stationarity"]
    )

    out_path = _REPO_ROOT / "reports" / "sub9_sanity" / "_close_dn_overnight_long_cross_window_scan.csv"
    cross.sort_values(["all_gates_pass", "min_pf"], ascending=[False, False]).to_csv(out_path, index=False)
    print(f"Full cross-window scan saved: {out_path}")
    print()

    survivors = cross[cross["all_gates_pass"]].sort_values("min_pf", ascending=False)
    print("=" * 100)
    print(f"SURVIVORS: cells passing ALL gates (Disc n>={N_MIN_SHIP}, PF>={PF_MIN_SHIP}; "
          f"OOS PF>={OOS_PF_MIN}; HO PF>={HO_PF_MIN}; stationarity<={STATIONARITY_MAX}; "
          f"WR drift<={WR_DRIFT_MAX_PP}pp)")
    print("=" * 100)
    if survivors.empty:
        print("  NONE.")
    else:
        print(f"  {len(survivors):,} survivors")
        print()
        print("Top 20 by min_pf:")
        cols = ["cell_label", "k_dims", "d_n", "d_pf", "o_n", "o_pf", "h_n", "h_pf",
                "min_pf", "stationarity_dpf", "wr_drift_dpp"]
        with pd.option_context("display.max_colwidth", 120, "display.width", 240):
            print(survivors.head(20)[cols].to_string(index=False))

    print()
    print("Top 15 by D_PF (regardless of all_gates_pass) for diagnostic:")
    cols = ["cell_label", "k_dims", "d_n", "d_pf", "o_pf", "h_pf",
            "stationarity_dpf", "all_gates_pass"]
    top_dpf = cross.sort_values("d_pf", ascending=False).head(15)
    with pd.option_context("display.max_colwidth", 120, "display.width", 240):
        print(top_dpf[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
