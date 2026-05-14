"""Cell-mine the C4a/C4b gap-down intraday trades using the existing
_cell_mine_tier_a infrastructure. Aggregate sanity FAILED — proper next step
is to scan 1D/2D/3D cell combinations and find sub-cells with edge.

Pre-registered thresholds (locked from existing gauntlet methodology):
  SURVIVOR: n >= 100, PF >= 1.20
  SHIP:     n >= 200, PF >= 1.40

If survivors found in post_rule that did NOT exist in pre_rule, that
confirms the regime-shift hypothesis at the cell level. Then OOS-validate
those cells on a held-out slice.

Usage:
    python -m tools.research.post_sebi.cellmine_gap_down
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sub9_research._cell_mine_tier_a import (  # noqa: E402
    scan_cells, pf_of, N_MIN_SURVIVOR, PF_MIN_SURVIVOR, N_MIN_SHIP, PF_MIN_SHIP,
)

_TRADES = _REPO / "reports" / "research" / "post_sebi" / "gap_down_intraday" / "gap_down_intraday_trades.parquet"
_OUT_DIR = _REPO / "reports" / "research" / "post_sebi" / "gap_down_intraday"

RULE_DATE = pd.Timestamp("2025-02-01").date()
WAR_START = pd.Timestamp("2026-01-01").date()


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Cell dimensions relevant to C4a/C4b gap-down setups."""
    df = df.copy()
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date

    # Gap magnitude bucket (|gap_pct| in pct, NOT fraction)
    df["gap_abs_pct"] = df["gap_pct"].abs() * 100.0
    df["gap_bucket"] = pd.cut(
        df["gap_abs_pct"],
        bins=[0.5, 0.75, 1.0, 1.5, 2.0, 100],
        labels=["0.5-0.75", "0.75-1.0", "1.0-1.5", "1.5-2.0", "2.0+"],
    )

    # First bar strength relative to R (how strong was the signal bar?)
    df["bar1_strength_pct"] = (df["close_920"] - df["open_915"]).abs() / df["open_915"] * 100.0
    df["bar1_strength_bucket"] = pd.cut(
        df["bar1_strength_pct"],
        bins=[0, 0.1, 0.25, 0.5, 1.0, 100],
        labels=["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1+"],
    )

    # Stop distance bucket (R size relative to entry, in pct)
    df["stop_distance_pct"] = df["stop_distance"] / df["entry_price"] * 100.0
    df["R_size_bucket"] = pd.cut(
        df["stop_distance_pct"],
        bins=[0, 0.5, 1.0, 1.5, 2.5, 100],
        labels=["<0.5%", "0.5-1%", "1-1.5%", "1.5-2.5%", "2.5%+"],
    )

    # Regime + sub-period
    df["regime"] = df["session_date"].apply(
        lambda d: "pre_rule" if d < RULE_DATE else "post_rule"
    )
    df["sub_period"] = df.apply(
        lambda r: (
            "pre_rule" if r["regime"] == "pre_rule"
            else ("war" if r["session_date"] >= WAR_START else "pre_war")
        ),
        axis=1,
    )

    # Calendar
    sd = pd.to_datetime(df["session_date"])
    df["dow"] = sd.dt.day_name().str[:3]
    df["month"] = sd.dt.strftime("%Y-%m")
    df["year"] = sd.dt.year.astype(str)

    # T1 hit flag (already in data)
    df["t1_hit_str"] = df["t1_hit"].astype(str)

    return df


def report_setup_per_regime(setup_name: str, df: pd.DataFrame, dims: list):
    print(f"\n{'='*82}")
    print(f"{setup_name}")
    print('='*82)
    for regime in ["pre_rule", "post_rule"]:
        sub = df[df["regime"] == regime]
        if sub.empty:
            continue
        print(f"\n--- regime = {regime} ---")
        agg_n = len(sub)
        agg_pf = pf_of(sub["net_pnl"])
        agg_wr = 100 * (sub["net_pnl"] > 0).mean()
        print(f"  AGGREGATE: n={agg_n:,}  PF={agg_pf:.3f}  WR={agg_wr:.1f}%")

        # Filter dims to those present + non-degenerate
        usable_dims = [d for d in dims if d in sub.columns and sub[d].nunique() > 1]
        if not usable_dims:
            print("  (no usable dims)")
            continue

        cells = scan_cells(sub, usable_dims, "net_pnl", max_combo=2)
        if cells.empty:
            print("  (no cells)")
            continue

        survivors = cells[(cells["n"] >= N_MIN_SURVIVOR) & (cells["pf"] >= PF_MIN_SURVIVOR)]
        ship = survivors[(survivors["n"] >= N_MIN_SHIP) & (survivors["pf"] >= PF_MIN_SHIP)]
        survivors = survivors.sort_values(["pf", "n"], ascending=[False, False])

        print(f"  SURVIVORS (n>={N_MIN_SURVIVOR}, PF>={PF_MIN_SURVIVOR}): {len(survivors)}")
        for _, r in survivors.head(15).iterrows():
            print(f"    [{r['dims']}] {r['cell']}  "
                  f"n={r['n']:,} PF={r['pf']:.3f} WR={r['wr']:.1f}% NET={r['net']:,.0f}")

        print(f"  SHIP-ELIGIBLE (n>={N_MIN_SHIP}, PF>={PF_MIN_SHIP}): {len(ship)}")
        for _, r in ship.head(10).iterrows():
            print(f"    [{r['dims']}] {r['cell']}  "
                  f"n={r['n']:,} PF={r['pf']:.3f} WR={r['wr']:.1f}% NET={r['net']:,.0f}")

        # Save survivors to CSV for follow-up
        out_path = _OUT_DIR / f"cellmine_{setup_name}_{regime}.csv"
        survivors.to_csv(out_path, index=False)
        print(f"  saved: {out_path.name}")


def main():
    if not _TRADES.exists():
        print(f"ERROR: trades file missing: {_TRADES}")
        print("Run sanity_gap_down_intraday first.")
        sys.exit(1)

    trades = pd.read_parquet(_TRADES)
    print(f"Loaded {len(trades):,} trades from {_TRADES.name}")
    trades = add_buckets(trades)

    dims_common = [
        "gap_bucket", "bar1_strength_bucket", "R_size_bucket",
        "dow", "month", "sub_period",
    ]

    for setup in ["c4a_gap_down_reversal_long", "c4b_gap_down_continuation_short"]:
        sub = trades[trades["setup"] == setup]
        report_setup_per_regime(setup, sub, dims_common)


if __name__ == "__main__":
    main()
