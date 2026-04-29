"""Module-level lookup for NSE index spot at a given 1m timestamp.

Public API:
  - get_nifty_spot(bar_timestamp) -> Optional[float]

Both `MainDetector._create_market_context` (per-bar scanner path) and
`PipelineOrchestrator._build_plan_from_sub7_detector` (post-trigger plan
path) call this so detectors that need cross-sectional index context
(currently `expiry_pin_strike_reversal`) see the same `nifty_spot` value
in `ctx.indicators` regardless of which path built the context.

In backtest mode, reads from
`backtest-cache-download/index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1minutes.feather`.

In live mode (deferred): the orchestrator should populate this at the
broker tick boundary instead of reading from disk. Same return type so
detector code is mode-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger

logger = get_agent_logger()


# Module-level cache: timestamp -> close. Populated lazily on first call.
# Keyed by IST-naive Timestamp (per project rule: no tzinfo on internal ts).
_NIFTY_SPOT_BY_TS: Optional[Dict[pd.Timestamp, float]] = None


def _resolve_index_path() -> Path:
    """Resolve the on-disk feather for NIFTY 50 1m bars."""
    # services/ is one level under repo root.
    return (
        Path(__file__).resolve().parent.parent
        / "backtest-cache-download" / "index_ohlcv"
        / "NSE_NIFTY_50" / "NSE_NIFTY_50_1minutes.feather"
    )


def _load_cache() -> Dict[pd.Timestamp, float]:
    """Read the NIFTY 50 1m feather; return {tz-naive Timestamp: close}.

    Returns an empty dict on missing/corrupt file (logged as warning) so
    callers don't propagate exceptions to detector hot path.
    """
    path = _resolve_index_path()
    if not path.exists():
        logger.warning(f"[NIFTY_SPOT] Index cache missing: {path}")
        return {}
    try:
        df = pd.read_feather(path)
        # Strip any tz so timestamps match the IST-naive bar timestamps the
        # orchestrator uses (per project's IST-naive rule).
        ts_col = pd.to_datetime(df["date"]).dt.tz_localize(None)
        out = dict(zip(ts_col, df["close"].astype(float)))
        logger.debug(f"[NIFTY_SPOT] Loaded {len(out):,} NIFTY 50 1m bars")
        return out
    except Exception as exc:   # noqa: BLE001 — broad on purpose; soft-fail
        logger.warning(f"[NIFTY_SPOT] Failed to load index cache: {exc}")
        return {}


def get_nifty_spot(bar_timestamp) -> Optional[float]:
    """Return NIFTY 50 close at the given 1m timestamp, or None if missing.

    Tolerates small grid mismatches by walking backward up to 5 minutes
    (covers detector ticks at 5m boundaries that don't perfectly align
    with the index 1m grid edge).
    """
    global _NIFTY_SPOT_BY_TS
    if _NIFTY_SPOT_BY_TS is None:
        _NIFTY_SPOT_BY_TS = _load_cache()

    ts = pd.Timestamp(bar_timestamp)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    if ts in _NIFTY_SPOT_BY_TS:
        return float(_NIFTY_SPOT_BY_TS[ts])
    for back in range(1, 6):
        t = ts - pd.Timedelta(minutes=back)
        if t in _NIFTY_SPOT_BY_TS:
            return float(_NIFTY_SPOT_BY_TS[t])
    return None


def clear_cache() -> None:
    """Reset the module cache (testing / live-mode reload utility)."""
    global _NIFTY_SPOT_BY_TS
    _NIFTY_SPOT_BY_TS = None
