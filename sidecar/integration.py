# sidecar/integration.py
"""
Integration helpers for main engine to use sidecar data.
"""

from __future__ import annotations

from datetime import datetime, time as dtime
from typing import TYPE_CHECKING, Optional

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from services.screener_live import ScreenerLive


def bootstrap_screener_from_sidecar(screener: "ScreenerLive") -> dict:
    """
    Bootstrap a ScreenerLive instance from sidecar data.

    Call this after creating ScreenerLive but before calling start().
    Populates:
    - ORB cache (orb_levels_cache)
    - 5m bars (bar_aggregator)

    Args:
        screener: ScreenerLive instance to bootstrap

    Returns:
        Dict with bootstrap results:
        - success: bool
        - orb_count: int
        - bars_count: int
        - skipped: bool (if data not available or not needed)
    """
    from .bootstrap import SidecarBootstrap

    result = {
        "success": False,
        "orb_count": 0,
        "daily_levels_count": 0,
        "bars_count": 0,
        "symbols_count": 0,
        "skipped": False,
        "reason": ""
    }

    # Check if we need bootstrap (late start)
    now = datetime.now().time()
    if now < dtime(9, 40):
        # Early start - ORB not computed yet, let main engine handle normally
        result["skipped"] = True
        result["reason"] = "early_start"
        logger.info("SIDECAR_BOOTSTRAP | Skipped: early start (before 09:40)")
        return result

    bootstrap = SidecarBootstrap()
    status = bootstrap.get_status()

    if not bootstrap.is_available():
        result["skipped"] = True
        result["reason"] = "no_sidecar_data"
        logger.warning("SIDECAR_BOOTSTRAP | Skipped: no sidecar data available")
        return result

    # Load ORB levels (ORH/ORL)
    orb_levels = {}
    if bootstrap.has_orb():
        orb_levels = bootstrap.load_orb() or {}
        if orb_levels:
            result["orb_count"] = len(orb_levels)
            logger.info(f"SIDECAR_BOOTSTRAP | Loaded ORB for {len(orb_levels)} symbols")

    # Load daily levels (PDH/PDL/PDC)
    daily_levels = {}
    if bootstrap.has_daily_levels():
        daily_levels = bootstrap.load_daily_levels() or {}
        if daily_levels:
            result["daily_levels_count"] = len(daily_levels)
            logger.info(f"SIDECAR_BOOTSTRAP | Loaded daily levels for {len(daily_levels)} symbols")

    # Merge ORB + daily levels and inject into screener's cache
    # Main engine stores both in _orb_levels_cache[session_date][symbol]
    if orb_levels or daily_levels:
        merged_levels = {}
        all_symbols = set(orb_levels.keys()) | set(daily_levels.keys())

        for symbol in all_symbols:
            merged_levels[symbol] = {}
            # Add ORB levels (ORH, ORL)
            if symbol in orb_levels:
                merged_levels[symbol].update(orb_levels[symbol])
            # Add daily levels (PDH, PDL, PDC)
            if symbol in daily_levels:
                merged_levels[symbol].update(daily_levels[symbol])

        if hasattr(screener, '_orb_levels_cache'):
            from datetime import datetime
            session_date = datetime.now().strftime("%Y-%m-%d")
            screener._orb_levels_cache[session_date] = merged_levels
            logger.info(f"SIDECAR_BOOTSTRAP | Injected merged levels for {len(merged_levels)} symbols")

    # Load 5m bars
    bars = bootstrap.load_bars()
    if bars:
        result["symbols_count"] = len(bars)
        result["bars_count"] = sum(len(df) for df in bars.values())

        # Inject into screener's bar aggregator
        if hasattr(screener, 'agg') and hasattr(screener.agg, '_bars_5m'):
            for symbol, df in bars.items():
                screener.agg._bars_5m[symbol] = df.copy()
            logger.info(f"SIDECAR_BOOTSTRAP | Injected {result['bars_count']} bars for {len(bars)} symbols")

    result["success"] = True
    logger.info(f"SIDECAR_BOOTSTRAP | Complete: {result}")

    return result


def maybe_bootstrap_from_sidecar(screener: "ScreenerLive") -> dict:
    """
    Convenience function that only bootstraps if it makes sense.

    Returns early if:
    - It's before 09:40 (ORB not computed yet)
    - Sidecar data not available
    """
    try:
        return bootstrap_screener_from_sidecar(screener)
    except Exception as e:
        logger.exception(f"SIDECAR_BOOTSTRAP | Error: {e}")
        return {
            "success": False,
            "skipped": True,
            "reason": f"error: {e}"
        }
