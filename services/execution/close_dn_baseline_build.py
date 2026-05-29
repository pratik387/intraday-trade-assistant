"""Closing-25m baseline + same-day candidate pre-filter builder.

Shared by:
  - run_verify_exit (services/execution/overnight_handlers.py) — runs once
    per day at 09:30 IST as the natural daily hook (verify-exit already
    runs there for AMO settlement; piggybacking saves a separate cron).
  - tools/build_close_dn_baseline.py — keep for manual one-off rebuilds.

Outputs two files in data/close_dn_baseline/:
  - baseline_<YYYY-MM-DD>.json + baseline_latest.json
    Per-symbol (vol_mean, vol_std) of 15:00-15:20 5-bar total volume across
    the trailing N trading sessions. close_dn_overnight_long_structure
    reads baseline_latest.json at detection time.
  - candidates_<YYYY-MM-DD>.json + candidates_latest.json
    Pre-filtered list of symbols whose prior-day return >= the cell-lock
    threshold (3.0% by default). Pre-computed prior_close + prev_prior_close
    let run_entry skip _gather_daily_dict entirely — it constructs a tiny
    2-row df_daily from these values for each candidate.

Why pre-filter at 09:30 instead of 15:25:
  The cell requires prior_day_return >= 3%; on a typical day ~50 of the
  1118 MIS-eligible NSE symbols qualify. Pre-filtering lets the 15:25
  entry cron fetch intraday bars for only ~50 symbols (~3s) instead of
  the full 1118 (~150s). Same edge, vastly cheaper.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from datetime import date as _date, datetime as _dt, timedelta as _td
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "data" / "close_dn_baseline"
SIGNAL_HHMM = ["15:00", "15:05", "15:10", "15:15", "15:20"]


def _eligible_universe(sdk) -> List[str]:
    """MIS-eligible NSE EQ symbols from the SDK instrument map."""
    from services.symbol_metadata import get_mis_info
    sm = sdk.get_symbol_map()
    return sorted(
        s for s in sm
        if get_mis_info(s).get("mis_enabled", False)
    )


def _baseline_for_symbol(df_5m: pd.DataFrame, rolling_days: int) -> Optional[dict]:
    """Per-symbol (vol_mean, vol_std) of 5-bar 15:00-15:20 total volume."""
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


def _prior_return_for_symbol(df_5m: pd.DataFrame) -> Optional[dict]:
    """Per-symbol prior_close, prev_prior_close, prior_day_return_pct.

    Derived from the last bar of each prior session in df_5m (close
    column). Matches the detector's df_5m fallback path semantically.
    Returns None when fewer than 2 prior sessions are available.
    """
    if df_5m is None or df_5m.empty:
        return None
    per_session_close = df_5m.groupby(df_5m.index.date)["close"].last().astype("float64")
    if len(per_session_close) < 2:
        return None
    prev_close = float(per_session_close.iloc[-1])
    prev_prev_close = float(per_session_close.iloc[-2])
    if prev_prev_close <= 0:
        return None
    ret_pct = (prev_close - prev_prev_close) / prev_prev_close * 100.0
    return {
        "prior_close": prev_close,
        "prev_prior_close": prev_prev_close,
        "prior_day_return_pct": float(ret_pct),
    }


def build_baseline_and_candidates(
    sdk,
    session_date: _date,
    *,
    rolling_days: int = 20,
    rps: float = 20.0,
    concurrency: int = 30,
    cell_min_prior_ret_pct: float = 3.0,
    max_symbols: Optional[int] = None,
) -> Dict[str, Any]:
    """Build baseline + candidate snapshots for `session_date`.

    Single batch fetch of prior 30 calendar days of 5m bars across the
    MIS-eligible universe, then per-symbol:
      * (vol_mean, vol_std) of 15:00-15:20 5-bar totals → baseline file
      * If prior_day_return >= cell_min_prior_ret_pct → candidates file

    Returns stats dict with counts + paths.
    """
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    universe = _eligible_universe(sdk)
    if max_symbols:
        universe = universe[:max_symbols]
    logger.info(
        "close_dn_baseline: building for %s | universe=%d rolling_days=%d "
        "cell_min_prior_ret=%.1f%%",
        session_date, len(universe), rolling_days, cell_min_prior_ret_pct,
    )

    from_d = (session_date - _td(days=30)).isoformat()
    to_d = (session_date - _td(days=1)).isoformat()

    t0 = _time.perf_counter()
    bars_map = asyncio.run(
        sdk.async_fetch_historical_5m_batch(
            universe, from_d, to_d,
            concurrency=concurrency, rps=rps,
        )
    )
    fetch_secs = _time.perf_counter() - t0
    logger.info(
        "close_dn_baseline: fetched bars for %d/%d symbols in %.1fs",
        len(bars_map), len(universe), fetch_secs,
    )

    baseline_symbols: Dict[str, Dict[str, float]] = {}
    candidate_entries: List[Dict[str, Any]] = []
    for sym in universe:
        df = bars_map.get(sym)
        b = _baseline_for_symbol(df, rolling_days)
        if b is not None:
            baseline_symbols[sym] = b
        r = _prior_return_for_symbol(df)
        if r is not None and r["prior_day_return_pct"] >= cell_min_prior_ret_pct:
            candidate_entries.append({"symbol": sym, **r})

    candidate_entries.sort(key=lambda e: e["prior_day_return_pct"], reverse=True)

    computed_at = _dt.now().isoformat()

    baseline_payload = {
        "session_date": session_date.isoformat(),
        "rolling_days": rolling_days,
        "hhmm_window": [SIGNAL_HHMM[0], SIGNAL_HHMM[-1]],
        "computed_at": computed_at,
        "n_symbols_attempted": len(universe),
        "n_symbols_with_baseline": len(baseline_symbols),
        "fetch_seconds": round(fetch_secs, 2),
        "symbols": baseline_symbols,
    }
    candidate_payload = {
        "session_date": session_date.isoformat(),
        "cell_min_prior_ret_pct": cell_min_prior_ret_pct,
        "computed_at": computed_at,
        "n_candidates": len(candidate_entries),
        "candidates": candidate_entries,
    }

    baseline_dated = BASELINE_DIR / f"baseline_{session_date.isoformat()}.json"
    baseline_latest = BASELINE_DIR / "baseline_latest.json"
    candidates_dated = BASELINE_DIR / f"candidates_{session_date.isoformat()}.json"
    candidates_latest = BASELINE_DIR / "candidates_latest.json"
    baseline_dated.write_text(json.dumps(baseline_payload, indent=2))
    baseline_latest.write_text(json.dumps(baseline_payload, indent=2))
    candidates_dated.write_text(json.dumps(candidate_payload, indent=2))
    candidates_latest.write_text(json.dumps(candidate_payload, indent=2))

    return {
        "session_date": session_date.isoformat(),
        "fetch_seconds": round(fetch_secs, 2),
        "n_symbols_attempted": len(universe),
        "n_symbols_with_baseline": len(baseline_symbols),
        "n_candidates": len(candidate_entries),
        "baseline_path": str(baseline_latest),
        "candidates_path": str(candidates_latest),
    }
