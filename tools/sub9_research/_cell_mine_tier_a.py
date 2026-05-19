"""Multi-dimensional cell mining across Tier A sub9 sanity trades.

Aggregate verdicts on Tier A all retired (PF < 1.10), but trade counts are
huge (20K-89K per direction). This scans all 1D/2D/3D combinations of
available cell dimensions and reports any cells that pass Stage 3
thresholds.

Same methodology that surfaced capitulation_long_morning's cell-locked ship
from a sanity that had borderline aggregate PF.

UPSTREAM DEPENDENCY (CRITICAL):
  This tool consumes per-trade CSVs that ALREADY have exit mechanics locked.
  Run SL/T1/T2/partial sweep BEFORE cell mining:
    - `_gap_fade_short_sl_target_sweep.py`
    - `_circuit_t1_sl_target_sweep.py`
    - `_fhm_resim_sweep.py`
    - `_target_sweep_delivery_pct.py`
    - `_structural_sweep_delivery_pct.py`
    - `_chrs_resim_r_sweep.py`
  These vary SL_BUFFER × ATR_MULT × T1_R × T2_R × T1_QTY_PCT × TIME_STOP
  and lock the optimal combo per setup. Cell mining then finds which
  (cell × locked-exit-mechanic) combos have edge.

GAUNTLET-V2 ALIGNED THRESHOLDS (added 2026-05-14):
  Survivor (worth investigating): n >= 100, PF >= 1.20, Sharpe > 0
  Ship-eligible (gauntlet ready): n >= 125 (power-derived), PF >= 1.30,
                                  Sharpe >= 0.5 (Holdout floor),
                                  losing_months_pct <= 40%,
                                  top_month_concentration < 40%
  Multi-test caveat: 1D/2D/3D scan tests dozens of cells; expect 2-3 false
  positives at alpha=0.05. Survivors require independent OOS/Holdout
  validation before shipping (one-shot OOS discipline per gauntlet v2).

Usage:
    python tools/sub9_research/_cell_mine_tier_a.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]

# Survivor thresholds (cells worth investigating further)
N_MIN_SURVIVOR = 100
PF_MIN_SURVIVOR = 1.20
SHARPE_MIN_SURVIVOR = 0.0

# Ship-eligible thresholds (matches gauntlet v2 Phase 4 holdout + power-derived n)
N_MIN_SHIP = 125  # was 200 — see tools/research/post_sebi/power_calc_cellmine_threshold.py
PF_MIN_SHIP = 1.30  # was 1.40 — aligned with sub-9 Stage 3 brief default
SHARPE_MIN_SHIP = 0.5  # gauntlet v2 Phase 4 holdout floor
LOSING_MONTHS_PCT_MAX_SHIP = 40.0  # gauntlet v2 Phase 4 losing_days <= 40% (at monthly granularity)
TOP_MONTH_CONCENTRATION_MAX = 40.0  # no single month can carry > 40% of NET


def pf_of(s: pd.Series) -> float:
    g = s[s > 0].sum()
    l = -s[s < 0].sum()
    return g / l if l > 0 else float("inf")


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add common bucket columns from raw trade fields."""
    df = df.copy()

    # Time bucket
    if "hour_bucket" not in df.columns:
        if "entry_ts" in df.columns:
            ts = pd.to_datetime(df["entry_ts"])
            h = ts.dt.hour
            df["hour_bucket"] = pd.cut(
                h, bins=[8, 10, 12, 14, 16],
                labels=["09-10", "10-12", "12-14", "14-15"], right=False)

    # Day-of-week
    if "T0_signal_date" in df.columns:
        d = pd.to_datetime(df["T0_signal_date"])
        df["dow"] = d.dt.day_name().str[:3]

    # Month
    if "_month" not in df.columns and "T0_signal_date" in df.columns:
        df["_month"] = pd.to_datetime(df["T0_signal_date"]).dt.strftime("%Y-%m")

    # Setup-specific buckets
    if "gap_pct" in df.columns:
        df["gap_bucket"] = pd.cut(
            df["gap_pct"].abs(),
            bins=[0, 1, 2, 3, 5, 100],
            labels=["<1%", "1-2%", "2-3%", "3-5%", "5%+"])
    if "cross_day_rvol" in df.columns:
        df["rvol_bucket"] = pd.cut(
            df["cross_day_rvol"],
            bins=[0, 2, 3, 5, 10, 1000],
            labels=["<2", "2-3", "3-5", "5-10", "10+"])
    if "or_range_pct" in df.columns:
        df["or_size_bucket"] = pd.cut(
            df["or_range_pct"],
            bins=[0, 0.5, 1.0, 1.5, 2.5, 100],
            labels=["<0.5%", "0.5-1%", "1-1.5%", "1.5-2.5%", "2.5%+"])
    if "fade_strength" in df.columns:
        df["fade_bucket"] = pd.cut(
            df["fade_strength"],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["<0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"])
    if "pierce_pct" in df.columns:
        df["pierce_bucket"] = pd.cut(
            df["pierce_pct"],
            bins=[0, 0.1, 0.3, 0.5, 1.0, 10],
            labels=["<0.1%", "0.1-0.3%", "0.3-0.5%", "0.5-1%", "1%+"])
    if "adv_20d_cr" in df.columns:
        df["adv_bucket"] = pd.cut(
            df["adv_20d_cr"],
            bins=[0, 5, 10, 30, 100, 1000, 100000],
            labels=["<5cr", "5-10cr", "10-30cr", "30-100cr", "100-1000cr", "1000+"])
    if "vol_persist_ratio_mean4" in df.columns:
        df["vp_bucket"] = pd.cut(
            df["vol_persist_ratio_mean4"],
            bins=[0, 0.5, 0.8, 1.0, 1.3, 100],
            labels=["<0.5", "0.5-0.8", "0.8-1.0", "1.0-1.3", "1.3+"])
    # vwap-specific
    if "vol_ratio" in df.columns:
        df["vol_ratio_bucket"] = pd.cut(
            df["vol_ratio"],
            bins=[0, 1.0, 1.3, 1.6, 2.0, 5.0, 1000],
            labels=["<1", "1-1.3", "1.3-1.6", "1.6-2", "2-5", "5+"])
    if "below_count_5bar" in df.columns:
        df["below5_bucket"] = df["below_count_5bar"].astype("Int64").astype(str)
    if "above_count_5bar" in df.columns:
        df["above5_bucket"] = df["above_count_5bar"].astype("Int64").astype(str)
    return df


def _sharpe_of(daily_pnl: pd.Series) -> float:
    """Daily Sharpe = mean(daily PnL) / std(daily PnL). 0.0 on degenerate input."""
    if daily_pnl.empty or daily_pnl.std() == 0 or daily_pnl.size < 2:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std())


def _monthly_stats(sub: pd.DataFrame, pnl_col: str, month_col: str) -> dict:
    """Per-month aggregation. Returns:
        - n_months
        - winning_months_pct
        - losing_months_pct
        - top_month_concentration_pct (|abs(monthly NET)| max / |abs(total NET)|)
    """
    monthly = sub.groupby(month_col)[pnl_col].sum()
    n_months = int(monthly.size)
    if n_months == 0:
        return dict(n_months=0, win_mo_pct=0.0, lose_mo_pct=0.0, top_mo_pct=0.0)
    win_pct = 100.0 * float((monthly > 0).mean())
    lose_pct = 100.0 * float((monthly < 0).mean())
    total_net = float(sub[pnl_col].sum())
    if abs(total_net) > 1e-6:
        top_pct = 100.0 * float(monthly.abs().max()) / abs(total_net)
    else:
        top_pct = 0.0
    return dict(
        n_months=n_months,
        win_mo_pct=round(win_pct, 1),
        lose_mo_pct=round(lose_pct, 1),
        top_mo_pct=round(top_pct, 1),
    )


def scan_cells(df: pd.DataFrame, dims: List[str], pnl_col: str,
               max_combo: int = 3,
               date_col: Optional[str] = None,
               month_col: Optional[str] = None) -> pd.DataFrame:
    """Scan all 1D, 2D, ... up to max_combo combinations of dims.

    Per-combo NaN filtering (not global) so a single sparse dim doesn't
    drop the whole frame.

    If `date_col` is provided, computes daily Sharpe per cell.
    If `month_col` is provided, computes monthly stability stats per cell.
    Both are optional for backward compat; missing → cell row gets 0.0 / 0
    for those columns and downstream code can skip gauntlet-v2 gates.
    """
    rows = []
    has_date = date_col is not None and date_col in df.columns
    has_month = month_col is not None and month_col in df.columns

    for k in range(1, max_combo + 1):
        for combo in combinations(dims, k):
            needed = list(combo) + [pnl_col]
            if has_date: needed.append(date_col)
            if has_month: needed.append(month_col)
            sub = df[needed].dropna(subset=list(combo) + [pnl_col])
            if sub.empty:
                continue
            for cell_values, cell_sub in sub.groupby(list(combo), observed=True):
                # Aggregate metrics
                pnl = cell_sub[pnl_col]
                n = int(len(pnl))
                wins = pnl[pnl > 0]
                losses = pnl[pnl <= 0]
                gw = float(wins.sum())
                gl = float(-losses.sum())
                pf = gw / gl if gl > 0 else float("inf")
                wr = 100.0 * float((pnl > 0).mean()) if n > 0 else 0.0
                net = float(pnl.sum())

                # Sharpe (daily) — requires date_col
                if has_date:
                    daily = cell_sub.groupby(date_col)[pnl_col].sum()
                    sharpe = _sharpe_of(daily)
                else:
                    sharpe = 0.0

                # Monthly stats — requires month_col
                if has_month:
                    m = _monthly_stats(cell_sub, pnl_col, month_col)
                else:
                    m = dict(n_months=0, win_mo_pct=0.0,
                             lose_mo_pct=0.0, top_mo_pct=0.0)

                # Cell label
                if not isinstance(cell_values, tuple):
                    cell_values = (cell_values,)
                cell = " | ".join(f"{c}={v}" for c, v in zip(combo, cell_values))

                rows.append({
                    "dims": ",".join(combo),
                    "k": k,
                    "cell": cell,
                    "n": n,
                    "pf": float(pf),
                    "wr": float(wr),
                    "net": net,
                    "sharpe": float(sharpe),
                    "n_months": m["n_months"],
                    "win_mo_pct": m["win_mo_pct"],
                    "lose_mo_pct": m["lose_mo_pct"],
                    "top_mo_pct": m["top_mo_pct"],
                })
    return pd.DataFrame(rows)


def report_setup(name: str, csv: Path, dims: List[str]):
    print(f"\n{'='*78}\n{name.upper()}\n{'='*78}")
    if not csv.exists():
        print(f"  MISSING: {csv}")
        return
    df = pd.read_csv(csv)
    print(f"  rows: {len(df):,}")
    df = add_buckets(df)

    # Add session_date + month columns if derivable (enables Sharpe + monthly stats)
    date_col = None
    month_col = None
    for candidate in ("T0_signal_date", "T1_entry_date", "session_date", "entry_date"):
        if candidate in df.columns:
            df["_session_date"] = pd.to_datetime(df[candidate]).dt.date
            df["_month"] = pd.to_datetime(df[candidate]).dt.strftime("%Y-%m")
            date_col = "_session_date"
            month_col = "_month"
            break

    pnl = "net_pnl"
    agg_n = len(df)
    agg_pf = pf_of(df[pnl])
    agg_wr = 100 * (df[pnl] > 0).mean()
    print(f"  AGGREGATE: n={agg_n:,}  PF={agg_pf:.3f}  WR={agg_wr:.1f}%")
    if date_col:
        agg_sharpe = _sharpe_of(df.groupby(date_col)[pnl].sum())
        print(f"  AGGREGATE Sharpe (daily): {agg_sharpe:.3f}")
    if month_col:
        agg_m = _monthly_stats(df, pnl, month_col)
        print(f"  AGGREGATE monthly: n_mo={agg_m['n_months']} "
              f"win_mo={agg_m['win_mo_pct']:.1f}% "
              f"lose_mo={agg_m['lose_mo_pct']:.1f}% "
              f"top_mo_NET={agg_m['top_mo_pct']:.1f}%")

    dims = [d for d in dims if d in df.columns]
    print(f"  cell dims scanned: {dims}")

    cells = scan_cells(df, dims, pnl, max_combo=3,
                       date_col=date_col, month_col=month_col)
    if cells.empty:
        print("  NO CELLS")
        return

    # Survivor gate: n + PF + (optional) Sharpe > 0
    survivors_mask = (
        (cells["n"] >= N_MIN_SURVIVOR)
        & (cells["pf"] >= PF_MIN_SURVIVOR)
        & (cells["sharpe"] >= SHARPE_MIN_SURVIVOR)
    )
    survivors = cells[survivors_mask].sort_values(["pf", "n"], ascending=[False, False])

    # Ship gate: gauntlet-v2 aligned thresholds
    ship_mask = (
        (cells["n"] >= N_MIN_SHIP)
        & (cells["pf"] >= PF_MIN_SHIP)
        & (cells["sharpe"] >= SHARPE_MIN_SHIP)
        & (cells["lose_mo_pct"] <= LOSING_MONTHS_PCT_MAX_SHIP)
        & (cells["top_mo_pct"] < TOP_MONTH_CONCENTRATION_MAX)
    )
    ship_eligible = cells[ship_mask].sort_values(["pf", "n"], ascending=[False, False])

    print(f"\n  SURVIVORS (n>={N_MIN_SURVIVOR}, PF>={PF_MIN_SURVIVOR}, Sharpe>0): {len(survivors):,}")
    if len(survivors) == 0:
        print("    [none]")
    else:
        for _, r in survivors.head(20).iterrows():
            print(f"    [{r['dims']}] {r['cell']}  "
                  f"n={r['n']:,} PF={r['pf']:.3f} WR={r['wr']:.1f}% "
                  f"Sh={r['sharpe']:.2f} mo[win={r['win_mo_pct']}% "
                  f"lose={r['lose_mo_pct']}% top={r['top_mo_pct']}%] "
                  f"NET={r['net']:,.0f}")

    print(f"\n  SHIP-ELIGIBLE (gauntlet-v2 gates: n>={N_MIN_SHIP}, PF>={PF_MIN_SHIP}, "
          f"Sharpe>={SHARPE_MIN_SHIP}, lose_mo<={LOSING_MONTHS_PCT_MAX_SHIP}%, "
          f"top_mo<{TOP_MONTH_CONCENTRATION_MAX}%): {len(ship_eligible):,}")
    if len(ship_eligible) == 0:
        print("    [none]")
    else:
        for _, r in ship_eligible.head(10).iterrows():
            print(f"    [{r['dims']}] {r['cell']}  "
                  f"n={r['n']:,} PF={r['pf']:.3f} Sh={r['sharpe']:.2f} "
                  f"lose_mo={r['lose_mo_pct']}% top_mo={r['top_mo_pct']}% "
                  f"NET={r['net']:,.0f}")


def main():
    base = _REPO / "reports" / "sub9_sanity"
    setups = [
        ("gap_fill", base / "gap_fill_trades.csv",
         ["side", "cap_segment", "regime", "hour_bucket", "dow",
          "gap_bucket", "vp_bucket", "adv_bucket"]),
        ("first_hour_momentum", base / "first_hour_momentum_trades.csv",
         ["direction", "cap_segment", "trigger_bucket",
          "or_size_bucket", "rvol_bucket", "dow"]),
        ("failure_fade", base / "failure_fade_trades.csv",
         ["side", "cap_segment", "regime", "hour_bucket", "setup", "level",
          "pierce_bucket", "fade_bucket", "dow"]),
        ("vwap_reclaim_lose", base / "vwap_reclaim_lose_trades.csv",
         ["side", "cap_segment", "hour_bucket", "_dist_bucket", "dow",
          "vol_ratio_bucket", "below5_bucket", "above5_bucket"]),
    ]
    for name, csv, dims in setups:
        report_setup(name, csv, dims)


if __name__ == "__main__":
    main()
