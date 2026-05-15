"""OOS cell-check for RSI extreme reversal cluster.

Re-runs the EXACT mathematics of `sanity_rsi_extreme_reversal.py` but on the
OOS window (2025-10-01 .. 2026-04-30) ONLY, then applies the four candidate
filters (Cell A, Cell B, Combined, Broader cluster) to the resulting trade
ledger.

This script does NOT modify the original Discovery run. It imports the
underlying functions (build_universe, load, aggregate, detect, simulate) from
the existing sanity script for byte-identical mathematics.

CANDIDATE CELLS (from deep cell-mining, committed at 44b630f)
-------------------------------------------------------------
- Cell A: direction=SHORT AND rsi_severity_fine=='S80-85' AND R_size%>=3 AND dow=Tue
- Cell B: direction=SHORT AND cap_segment=='small_cap' AND rsi_severity_fine=='S85+' AND dow=Fri
- Cell Combined: union of A and B
- Broader cluster: direction=SHORT AND cap_segment=='small_cap' AND rsi_trigger>=80

PRE-REGISTERED PASS CRITERIA (LOCKED — no goalpost moving)
-----------------------------------------------------------
Cell-specific OOS:
  PF >= 1.30
  n >= 50 (lower floor due to thin 7-month OOS window)
  Sharpe > 0
  per-month winning >= 4 / 7 OOS months
  top-month NET < 40%

Usage:
    python tools/sub9_research/sanity_rsi_oos_cellcheck.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Re-use the exact same math from the original sanity script.
from tools.sub9_research.sanity_rsi_extreme_reversal import (  # noqa: E402
    OOS_START,
    OOS_END,
    build_universe,
    load_5m_for_period,
    aggregate_to_15m_with_rsi,
    detect_events,
    simulate,
)

_OUT_DIR = _REPO / "reports" / "sub9_sanity"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-registered pass gates
PASS_PF = 1.30
PASS_N = 50
PASS_SH = 0.0  # Sharpe > 0
PASS_WIN_MO_COUNT = 4  # >=4 winning months out of 7
PASS_TOP_MO_PCT = 40.0


# ---------------------------------------------------------------------------
# Bucketers (mirror _deep_cellmine_math_patterns.prep_rsi)
# ---------------------------------------------------------------------------
def _bucket_pct(s: pd.Series, edges, labels) -> pd.Categorical:
    return pd.cut(s, bins=edges, labels=labels, right=False)


def annotate_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Add dow, rsi_severity_fine, R_size_bucket columns to trade frame."""
    df = df.copy()
    d = pd.to_datetime(df["session_date"])
    df["dow"] = d.dt.day_name().str[:3]  # Mon, Tue, ...

    # Fine RSI severity (direction-aware) — same edges as deep cellmine
    rsi = df["rsi_trigger"]
    side = df["direction"].astype(str).str.lower()
    long_bins = [-np.inf, 15, 20, 25, 30, np.inf]
    long_lbl = ["L<15", "L15-20", "L20-25", "L25-30", "L30+"]
    short_bins = [-np.inf, 70, 75, 80, 85, np.inf]
    short_lbl = ["S<=70", "S70-75", "S75-80", "S80-85", "S85+"]

    long_bucket = pd.cut(rsi, bins=long_bins, labels=long_lbl, right=False)
    short_bucket = pd.cut(rsi, bins=short_bins, labels=short_lbl, right=False)

    df["rsi_severity_fine"] = np.where(side == "long", long_bucket.astype(str),
                                       short_bucket.astype(str))
    df.loc[df["rsi_severity_fine"] == "nan", "rsi_severity_fine"] = np.nan

    # R size (% of entry)
    r_pct = df["stop_distance"] / df["entry_price"] * 100.0
    df["R_size_pct"] = r_pct
    df["R_size_bucket"] = pd.cut(
        r_pct,
        bins=[-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf],
        labels=["<0.5%", "0.5-1%", "1-2%", "2-3%", "3%+"],
    ).astype(str)
    df.loc[df["R_size_bucket"] == "nan", "R_size_bucket"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _pf(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    return g / l if l > 0 else float("inf")


def _sharpe_daily(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    daily = df.groupby("session_date")["net_pnl"].sum()
    if daily.size < 2 or daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std())


def _monthly_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n_months=0, win_mo_count=0, win_mo_pct=0.0, top_mo_pct=0.0,
                    monthly_breakdown=[])
    m = df.copy()
    m["_mo"] = pd.to_datetime(m["session_date"]).dt.strftime("%Y-%m")
    monthly = m.groupby("_mo")["net_pnl"].sum().sort_index()
    n_mo = int(monthly.size)
    win_mo_count = int((monthly > 0).sum())
    win_mo_pct = 100.0 * win_mo_count / n_mo if n_mo > 0 else 0.0
    total = float(monthly.sum())
    if abs(total) > 1e-6:
        top_pct = 100.0 * float(monthly.abs().max()) / abs(total)
    else:
        top_pct = 0.0
    return dict(
        n_months=n_mo,
        win_mo_count=win_mo_count,
        win_mo_pct=round(win_mo_pct, 1),
        top_mo_pct=round(top_pct, 1),
        monthly_breakdown=[(k, round(v, 0)) for k, v in monthly.items()],
    )


def _agg_row(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n=0, pf=0.0, wr=0.0, net=0.0, sharpe=0.0,
                    n_months=0, win_mo_count=0, win_mo_pct=0.0,
                    top_mo_pct=0.0, monthly_breakdown=[])
    pnl = df["net_pnl"]
    pf = _pf(pnl)
    wr = 100.0 * float((pnl > 0).mean())
    sharpe = _sharpe_daily(df)
    m = _monthly_stats(df)
    return dict(
        n=int(len(df)),
        pf=float(pf),
        wr=float(wr),
        net=float(pnl.sum()),
        sharpe=float(sharpe),
        n_months=m["n_months"],
        win_mo_count=m["win_mo_count"],
        win_mo_pct=m["win_mo_pct"],
        top_mo_pct=m["top_mo_pct"],
        monthly_breakdown=m["monthly_breakdown"],
    )


def _verdict(agg: dict) -> str:
    if agg["n"] < PASS_N:
        return f"FAIL_N (n={agg['n']} < {PASS_N})"
    if agg["pf"] < PASS_PF:
        return f"FAIL_PF (pf={agg['pf']:.3f} < {PASS_PF})"
    if agg["sharpe"] <= PASS_SH:
        return f"FAIL_SHARPE (sh={agg['sharpe']:.3f} <= {PASS_SH})"
    if agg["win_mo_count"] < PASS_WIN_MO_COUNT:
        return f"FAIL_WIN_MO ({agg['win_mo_count']}/{agg['n_months']} mo winning, need >={PASS_WIN_MO_COUNT})"
    if agg["top_mo_pct"] >= PASS_TOP_MO_PCT:
        return f"FAIL_TOP_MO (top={agg['top_mo_pct']}% >= {PASS_TOP_MO_PCT}%)"
    return "PASS"


# ---------------------------------------------------------------------------
# Cell filters
# ---------------------------------------------------------------------------
def filter_cell_a(df: pd.DataFrame) -> pd.DataFrame:
    """Cell A: SHORT × S80-85 × R_size>=3% × Tue."""
    return df[
        (df["direction"].astype(str).str.lower() == "short")
        & (df["rsi_severity_fine"] == "S80-85")
        & (df["R_size_pct"] >= 3.0)
        & (df["dow"] == "Tue")
    ].copy()


def filter_cell_b(df: pd.DataFrame) -> pd.DataFrame:
    """Cell B: SHORT × small_cap × S85+ × Fri."""
    return df[
        (df["direction"].astype(str).str.lower() == "short")
        & (df["cap_segment"] == "small_cap")
        & (df["rsi_severity_fine"] == "S85+")
        & (df["dow"] == "Fri")
    ].copy()


def filter_combined(df: pd.DataFrame) -> pd.DataFrame:
    a = filter_cell_a(df)
    b = filter_cell_b(df)
    out = pd.concat([a, b], ignore_index=True)
    if out.empty:
        return out
    # Deduplicate on (symbol, session_date) — same event would only appear in
    # one cell anyway (Tue vs Fri), but keep this safe.
    out = out.drop_duplicates(subset=["symbol", "session_date"]).reset_index(drop=True)
    return out


def filter_broader(df: pd.DataFrame) -> pd.DataFrame:
    """Broader cluster: SHORT × small_cap × rsi_trigger >= 80."""
    return df[
        (df["direction"].astype(str).str.lower() == "short")
        & (df["cap_segment"] == "small_cap")
        & (df["rsi_trigger"] >= 80.0)
    ].copy()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_cell_report(name: str, sub: pd.DataFrame) -> dict:
    print(f"\n{'-' * 78}")
    print(f"CELL: {name}")
    print('-' * 78)
    if sub.empty:
        print("  [NO TRADES]")
        return dict(n=0)
    agg = _agg_row(sub)
    print(f"  n={agg['n']}  PF={agg['pf']:.3f}  WR={agg['wr']:.1f}%  "
          f"NET={agg['net']:,.0f}  Sharpe(daily)={agg['sharpe']:.3f}")
    print(f"  months={agg['n_months']}  win_mo={agg['win_mo_count']}/{agg['n_months']} "
          f"({agg['win_mo_pct']}%)  top_mo_NET={agg['top_mo_pct']}%")
    print(f"  monthly breakdown:")
    for mo, net in agg["monthly_breakdown"]:
        marker = "+" if net > 0 else "-"
        print(f"    {mo}  {marker}  NET={net:,.0f}")
    verdict = _verdict(agg)
    print(f"  VERDICT [pre-registered gates]: {verdict}")
    agg["verdict"] = verdict
    agg["name"] = name
    return agg


def main() -> None:
    print("=== RSI extreme reversal — OOS cell-check ===")
    print(f"Window: {OOS_START} .. {OOS_END}")
    print(f"Pre-registered gates: n>={PASS_N}  PF>={PASS_PF}  Sh>{PASS_SH}  "
          f"win_mo>={PASS_WIN_MO_COUNT}/7  top_mo<{PASS_TOP_MO_PCT}%")

    universe, cap_map = build_universe()
    print(f"\nUniverse: {len(universe):,} symbols")

    # OOS simulation
    print("\n--- OOS load + simulate ---")
    big5m = load_5m_for_period(OOS_START, OOS_END, universe)
    if big5m.empty:
        print("[empty 5m frame]")
        return
    df15 = aggregate_to_15m_with_rsi(big5m)
    events = detect_events(df15)
    print(f"  events: {len(events):,}  (long={int((events['direction']=='long').sum()):,}  "
          f"short={int((events['direction']=='short').sum()):,})")
    trades = simulate(events, df15, big5m, cap_map)
    print(f"  trades: {len(trades):,}")
    if trades.empty:
        print("[NO TRADES]")
        return

    # Save raw OOS trades
    oos_path = _OUT_DIR / "rsi_oos_cellcheck_all_trades.csv"
    trades.to_csv(oos_path, index=False)
    print(f"  raw OOS trades written: {oos_path}")

    # Annotate cells
    trades = annotate_cells(trades)

    # Aggregate baseline (informational)
    print(f"\n{'='*78}")
    print(f"BASELINE (all OOS trades)")
    print('=' * 78)
    base = _agg_row(trades)
    print(f"  n={base['n']:,}  PF={base['pf']:.3f}  WR={base['wr']:.1f}%  "
          f"NET={base['net']:,.0f}  Sharpe(daily)={base['sharpe']:.3f}")

    # Run each cell
    results = []
    for name, fn, fname in [
        ("Cell A (SHORT × S80-85 × R>=3% × Tue)", filter_cell_a, "cell_a"),
        ("Cell B (SHORT × small_cap × S85+ × Fri)", filter_cell_b, "cell_b"),
        ("Combined (A union B)", filter_combined, "combined"),
        ("Broader cluster (SHORT × small_cap × rsi>=80)", filter_broader, "broader"),
    ]:
        sub = fn(trades)
        # Persist per-cell trade CSV
        cell_csv = _OUT_DIR / f"rsi_oos_cellcheck_{fname}.csv"
        sub.to_csv(cell_csv, index=False)
        print(f"\n[saved {len(sub)} trades -> {cell_csv.name}]")
        res = print_cell_report(name, sub)
        res["fname"] = fname
        results.append(res)

    # Summary
    print(f"\n{'='*78}")
    print("SUMMARY")
    print('=' * 78)
    for r in results:
        if r.get("n", 0) == 0:
            print(f"  {r.get('name','?'):<60} [NO TRADES]")
            continue
        print(f"  {r['name']:<60}  n={r['n']:>4}  PF={r['pf']:.2f}  "
              f"Sh={r['sharpe']:>5.2f}  win_mo={r['win_mo_count']}/{r['n_months']}  "
              f"top_mo={r['top_mo_pct']}%  -> {r['verdict']}")

    any_pass = any(r.get("verdict") == "PASS" for r in results)
    print(f"\nFINAL: {'AT LEAST ONE CELL SURVIVES OOS' if any_pass else 'ALL CELLS FAIL OOS — RETIRE'}")


if __name__ == "__main__":
    main()
