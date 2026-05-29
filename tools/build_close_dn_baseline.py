#!/usr/bin/env python3
"""Build the close_dn_overnight_long closing-25m volume baseline.

Pre-computes per-symbol (mean, std) of total 15:00-15:20 5m-bar volume across
the prior `rolling_days` trading sessions and writes a JSON file the 15:25
overnight cron can read in microseconds — replaces an inline 1118-symbol
batch fetch at cron time (which would burn ~60s of the 5-minute MOC window).

Output:
    data/close_dn_baseline/baseline_<YYYY-MM-DD>.json   # dated snapshot
    data/close_dn_baseline/baseline_latest.json         # symlink-style copy

JSON schema:
    {
      "session_date": "2026-05-29",            # the cron run-date this baseline serves
      "rolling_days": 20,
      "hhmm_window": ["15:00", "15:20"],
      "computed_at": "2026-05-29T08:00:00",
      "n_symbols_attempted": 1456,
      "n_symbols_with_baseline": 1118,
      "symbols": {
        "NSE:RELIANCE": {"vol_mean": 1234567.0, "vol_std": 234567.0, "n_sessions": 20},
        ...
      }
    }

Wire-up:
    Run daily at 08:00 IST via scripts/refresh-close-dn-baseline.sh cron.
    structures/close_dn_overnight_long_structure._closing_baseline() reads
    `baseline_latest.json` first; falls back to df_5m-based compute for
    backtest (where df_5m carries multi-day history natively).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date as _date, datetime as _dt, timedelta as _td
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "close_dn_baseline"
SIGNAL_HHMM = ["15:00", "15:05", "15:10", "15:15", "15:20"]


def _eligible_universe(sdk, config) -> list:
    """Return MIS-eligible NSE EQ symbols from current SDK instrument map.

    Matches close_dn_overnight_long_universe's filter (MIS + cap_segment-aware).
    We don't filter by daily-volume here — the baseline is keyed per symbol;
    the universe builder filters again at cron time.
    """
    from services.symbol_metadata import get_mis_info
    sm = sdk.get_symbol_map()
    out = []
    for sym in sm:
        try:
            if get_mis_info(sym).get("mis_enabled", False):
                out.append(sym)
        except Exception:
            continue
    return sorted(out)


def _compute_baseline_for_symbol(df_5m: pd.DataFrame, rolling_days: int) -> dict | None:
    """Per-symbol baseline: (mean, std) of 5-bar 15:00-15:20 total volume over
    the most recent `rolling_days` trading sessions present in df_5m.

    Returns None if fewer than max(10, rolling_days // 2) prior sessions
    available (mirrors the detector's existing guard).
    """
    if df_5m is None or df_5m.empty:
        return None
    sig = df_5m[df_5m.index.map(lambda ts: ts.strftime("%H:%M") in SIGNAL_HHMM)]
    if sig.empty:
        return None
    per_session = sig.groupby(sig.index.date)["volume"].sum().astype("float64")
    if len(per_session) < max(10, rolling_days // 2):
        return None
    recent = per_session.iloc[-rolling_days:]
    mean = float(recent.mean())
    std = float(recent.std(ddof=1))
    if std <= 0:
        return None
    return {"vol_mean": mean, "vol_std": std, "n_sessions": int(len(recent))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--session-date", type=str, default=None,
                        help="The cron date this baseline will serve (YYYY-MM-DD). "
                             "Defaults to today.")
    parser.add_argument("--rolling-days", type=int, default=20,
                        help="Number of prior trading sessions in the baseline window.")
    parser.add_argument("--rps", type=float, default=20.0,
                        help="Upstox API rate (RPS).")
    parser.add_argument("--concurrency", type=int, default=30,
                        help="Concurrent HTTP requests.")
    parser.add_argument("--max-symbols", type=int, default=None,
                        help="Cap universe size (debug).")
    args = parser.parse_args()

    session_date = (
        _date.fromisoformat(args.session_date) if args.session_date else _date.today()
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"build_close_dn_baseline: session_date={session_date} rolling_days={args.rolling_days}")

    # Build universe
    from broker.upstox.upstox_data_client import UpstoxDataClient
    from config.filters_setup import load_filters
    cfg = load_filters()
    sdk = UpstoxDataClient()
    universe = _eligible_universe(sdk, cfg)
    if args.max_symbols:
        universe = universe[:args.max_symbols]
    print(f"  universe size: {len(universe)} MIS-eligible NSE EQ symbols")

    # Fetch: rolling_days * 2 calendar days; async_fetch_historical_5m_batch
    # auto-chunks under the Upstox 1-month per-request limit.
    from_d = (session_date - _td(days=args.rolling_days * 2)).isoformat()
    to_d = (session_date - _td(days=1)).isoformat()
    print(f"  fetching 5m bars {from_d} -> {to_d}, rps={args.rps} concurrency={args.concurrency}")

    t0 = _dt.now()
    bars_map = asyncio.run(
        sdk.async_fetch_historical_5m_batch(
            universe, from_d, to_d,
            concurrency=args.concurrency, rps=args.rps,
        )
    )
    fetch_secs = (_dt.now() - t0).total_seconds()
    print(f"  fetch done in {fetch_secs:.1f}s — got bars for {len(bars_map)}/{len(universe)} symbols")

    # Compute per-symbol baselines
    symbols_out: dict = {}
    n_skipped = 0
    for sym in universe:
        df = bars_map.get(sym)
        b = _compute_baseline_for_symbol(df, args.rolling_days)
        if b is None:
            n_skipped += 1
            continue
        symbols_out[sym] = b

    payload = {
        "session_date": session_date.isoformat(),
        "rolling_days": args.rolling_days,
        "hhmm_window": [SIGNAL_HHMM[0], SIGNAL_HHMM[-1]],
        "computed_at": _dt.now().isoformat(),
        "n_symbols_attempted": len(universe),
        "n_symbols_with_baseline": len(symbols_out),
        "n_symbols_skipped": n_skipped,
        "fetch_seconds": round(fetch_secs, 2),
        "symbols": symbols_out,
    }

    dated_path = OUT_DIR / f"baseline_{session_date.isoformat()}.json"
    latest_path = OUT_DIR / "baseline_latest.json"
    dated_path.write_text(json.dumps(payload, indent=2))
    latest_path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {dated_path} ({len(symbols_out)} symbols)")
    print(f"  wrote {latest_path}")


if __name__ == "__main__":
    main()
