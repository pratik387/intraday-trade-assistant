"""Re-compute sanity PF on PRODUCTION-aligned universe.

Original sanity scripts iterate every symbol in the monthly 5m feathers
and let per-bar filters reject non-qualifying signals at signal time.
Production's universe builder is STRICTER:

  1. cap_segment == required (e.g., 'unknown' for below_vwap_volume_revert_long)
  2. mis_enabled == True
  3. >= 30 prior trading days of DAILY history (in consolidated_daily.feather)
  4. avg daily volume >= 50,000 (in consolidated_daily.feather)

Sanity's 5m-level coverage filter is far more permissive — it only checks
% of window days with bars present, NOT prior history at signal time. And
its volume threshold is on closing-30m totals, effectively ~10× more
permissive than production's 50K daily floor.

This tool re-filters a sanity trades CSV against production's universe
filters and re-computes PF. Numbers BEFORE alignment vs AFTER tell us
the inflation caused by the universe mismatch.

Usage:
  python tools/sub9_research/sanity_to_production_universe_align.py \\
      --sanity-csv reports/sub9_sanity/_below_vwap_volume_revert_long_trades_holdout.csv \\
      --required-cap unknown \\
      --window-start 2025-12-08 --window-end 2026-04-30 \\
      --cell-filter "cap_segment=unknown,vol_ratio_bin=gte_10,hhmm_bucket=afternoon_1300_1500"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Set

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DAILY_PATH = _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather"
_NSE_ALL = _REPO_ROOT / "nse_all.json"


def _normalize_sym(s: str) -> str:
    """Strip 'NSE:' prefix and '.NS' suffix (matches services.symbol_metadata)."""
    if ":" in s:
        s = s.split(":")[-1]
    if "." in s:
        s = s.split(".")[0]
    return s


def _load_nse_all() -> Dict[str, dict]:
    with open(_NSE_ALL, encoding="utf-8") as f:
        entries = json.load(f)
    out = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        sym = _normalize_sym(e.get("symbol", ""))
        if sym:
            out[sym] = e
    return out


def production_universe_at(
    target_date: date,
    *,
    required_cap: str = "unknown",
    accepted_caps: tuple = None,
    min_trading_days: int = 30,
    min_daily_avg_volume: float = 50_000,
    daily_df: pd.DataFrame = None,
    nse_all: Dict[str, dict] = None,
) -> Set[str]:
    """Return the set of BARE symbols that pass production filters on `target_date`.

    Pass `accepted_caps` (tuple) for multi-cap setups; otherwise `required_cap`
    (str) for single-cap setups like below_vwap_volume_revert_long.
    """
    if daily_df is None:
        daily_df = pd.read_feather(_DAILY_PATH)
        daily_df["ts"] = pd.to_datetime(daily_df["ts"])
        daily_df["d"] = daily_df["ts"].dt.date
    if nse_all is None:
        nse_all = _load_nse_all()

    daily = daily_df[daily_df.d < target_date]
    grouped = daily.groupby("symbol")

    accepted = (
        set(accepted_caps) if accepted_caps is not None else {required_cap}
    )

    qual = set()
    for bare, ddf in grouped:
        if bare not in nse_all:
            continue
        e = nse_all[bare]
        if e.get("cap_segment") not in accepted:
            continue
        if not e.get("mis_enabled", False):
            continue
        if len(ddf) < min_trading_days:
            continue
        if float(ddf.volume.mean()) < min_daily_avg_volume:
            continue
        qual.add(bare)
    return qual


def realign_sanity(
    sanity_csv: Path,
    *,
    required_cap: str = None,
    accepted_caps: tuple = None,
    window_start: date = None,
    window_end: date = None,
    cell_filter: Dict[str, str] = None,
    min_trading_days: int = 30,
    min_daily_avg_volume: float = 50_000,
) -> dict:
    """Load sanity trades, filter to (cell, window), then re-check each trade's
    (symbol, signal_date) against the production universe.

    Returns a summary dict with original-vs-aligned PF.
    """
    df = pd.read_csv(sanity_csv)
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
    if window_start is not None:
        df = df[df["signal_date"] >= window_start]
    if window_end is not None:
        df = df[df["signal_date"] <= window_end]
    if cell_filter:
        for k, v in cell_filter.items():
            df = df[df[k] == v]
    print(f"After (window, cell) filter: {len(df)} trades")

    # Pre-load daily + nse_all once
    daily_df = pd.read_feather(_DAILY_PATH)
    daily_df["ts"] = pd.to_datetime(daily_df["ts"])
    daily_df["d"] = daily_df["ts"].dt.date
    nse_all = _load_nse_all()

    # Cache universe per signal_date (universe changes slowly day-to-day)
    universe_cache: Dict[date, Set[str]] = {}

    def _universe(d: date) -> Set[str]:
        if d not in universe_cache:
            universe_cache[d] = production_universe_at(
                d, required_cap=required_cap, accepted_caps=accepted_caps,
                min_trading_days=min_trading_days,
                min_daily_avg_volume=min_daily_avg_volume,
                daily_df=daily_df, nse_all=nse_all,
            )
        return universe_cache[d]

    df["bare_sym"] = df["symbol"].apply(_normalize_sym)
    print("Filtering against production universe (may take ~30s)...")
    accepted = []
    rejected_reasons = {"not_in_prod_universe": 0}
    for _, row in df.iterrows():
        u = _universe(row["signal_date"])
        if row["bare_sym"] in u:
            accepted.append(row)
        else:
            rejected_reasons["not_in_prod_universe"] += 1
    aligned_df = pd.DataFrame(accepted)

    # Stats
    def _pf_stats(d: pd.DataFrame) -> dict:
        if d.empty or "net_pnl_inr" not in d.columns:
            return {"n": 0, "pf": None, "wr": None, "net": 0.0, "mean": 0.0}
        s = d["net_pnl_inr"]
        g = float(s[s > 0].sum())
        l = float(-s[s < 0].sum())
        pf = (g / l) if l > 0 else float("inf")
        return {
            "n": len(d), "pf": pf,
            "wr": float((s > 0).mean() * 100.0),
            "net": float(s.sum()),
            "mean": float(s.mean()),
        }

    orig = _pf_stats(df)
    aligned = _pf_stats(aligned_df)

    return {
        "window": f"{window_start} to {window_end}",
        "cell_filter": cell_filter,
        "original": orig,
        "aligned": aligned,
        "rejected_count": rejected_reasons["not_in_prod_universe"],
        "rejected_pct": rejected_reasons["not_in_prod_universe"] / max(1, orig["n"]) * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-csv", required=True, type=Path)
    parser.add_argument("--required-cap", default=None,
                        help="Single cap_segment for the setup (e.g., 'unknown')")
    parser.add_argument("--accepted-caps", default=None,
                        help="Comma-separated for multi-cap setups (e.g., 'large_cap,mid_cap,small_cap,unknown')")
    parser.add_argument("--window-start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--window-end", default=None, help="YYYY-MM-DD")
    parser.add_argument("--cell-filter", default=None,
                        help="Comma-separated key=value pairs to subset (e.g., 'cap_segment=unknown,vol_ratio_bin=gte_10')")
    parser.add_argument("--min-trading-days", type=int, default=30)
    parser.add_argument("--min-daily-avg-volume", type=float, default=50000)
    args = parser.parse_args()

    accepted_caps = None
    if args.accepted_caps:
        accepted_caps = tuple(c.strip() for c in args.accepted_caps.split(","))
    cell_filter = None
    if args.cell_filter:
        cell_filter = dict(p.split("=") for p in args.cell_filter.split(","))
    ws = date.fromisoformat(args.window_start) if args.window_start else None
    we = date.fromisoformat(args.window_end) if args.window_end else None

    summary = realign_sanity(
        args.sanity_csv,
        required_cap=args.required_cap,
        accepted_caps=accepted_caps,
        window_start=ws, window_end=we,
        cell_filter=cell_filter,
        min_trading_days=args.min_trading_days,
        min_daily_avg_volume=args.min_daily_avg_volume,
    )

    print()
    print("=" * 80)
    print(f"Universe alignment summary")
    print("=" * 80)
    print(f"Window: {summary['window']}")
    print(f"Cell filter: {summary['cell_filter']}")
    print()
    o = summary["original"]; a = summary["aligned"]
    print(f"{'metric':<10} {'BEFORE (sanity)':>20} {'AFTER (prod-aligned)':>25}")
    print('-' * 60)
    print(f"{'n':<10} {o['n']:>20,} {a['n']:>25,}")
    if o['pf'] is not None and a['pf'] is not None:
        print(f"{'PF':<10} {o['pf']:>20.3f} {a['pf']:>25.3f}")
        print(f"{'WR':<10} {o['wr']:>19.1f}% {a['wr']:>24.1f}%")
        o_mean_str = f"+Rs {o['mean']:.0f}"
        a_mean_str = f"+Rs {a['mean']:.0f}"
        o_net_str = f"+Rs {o['net']:,.0f}"
        a_net_str = f"+Rs {a['net']:,.0f}"
        print(f"{'mean/trade':<10} {o_mean_str:>20} {a_mean_str:>25}")
        print(f"{'NET':<10} {o_net_str:>20} {a_net_str:>25}")
    print()
    print(f"Rejected (in sanity, not in production universe): {summary['rejected_count']:,} "
          f"({summary['rejected_pct']:.1f}% of original)")


if __name__ == "__main__":
    main()
