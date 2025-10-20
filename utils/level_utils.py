"""
Level Calculation Utilities
============================

Centralized utilities for computing previous day levels (PDH/PDL/PDC)
and other price level calculations.

This eliminates duplication between screener_live.py and planner_internal.py.
"""

import pandas as pd
import datetime
from typing import Dict, Optional
from config.logging_config import get_agent_logger

logger = get_agent_logger()


def get_previous_day_levels(
    daily_df: Optional[pd.DataFrame],
    session_date: Optional[datetime.date],
    fallback_df: Optional[pd.DataFrame] = None,
    enable_fallback: bool = False,
) -> Dict[str, float]:
    """
    Compute Previous Day High/Low/Close (PDH/PDL/PDC) from daily data.

    This is the centralized implementation used by both screener_live.py
    and planner_internal.py to avoid code duplication.

    Args:
        daily_df: Daily OHLC DataFrame (with 'high', 'low', 'close' columns)
        session_date: Target session date (will use last day BEFORE this date)
        fallback_df: Optional 5m DataFrame to estimate levels if daily data missing (for backtests)
        enable_fallback: Whether to use fallback estimation (default False for production)

    Returns:
        Dict with keys: "PDH", "PDL", "PDC" (float values, nan if not available)

    Algorithm:
        1. Filter daily data to dates before session_date
        2. Remove rows with zero/missing volume (if volume column exists)
        3. Remove rows with missing high/low
        4. Extract last valid row as previous day
        5. If no data and fallback enabled, estimate from early session bars

    Example:
        >>> daily = get_daily_data("RELIANCE", days=210)
        >>> levels = get_previous_day_levels(daily, datetime.date(2025, 1, 15))
        >>> print(levels)  # {"PDH": 2456.80, "PDL": 2432.15, "PDC": 2445.50}
    """
    try:
        if daily_df is None or daily_df.empty:
            logger.debug("get_previous_day_levels: daily_df is None or empty")
            if enable_fallback and fallback_df is not None:
                return _estimate_levels_from_fallback(fallback_df)
            return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}

        # Make copy to avoid modifying original
        d = daily_df.copy()

        # Normalize date index
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"])
            d = d.sort_values("date").set_index("date")
        else:
            d.index = pd.to_datetime(d.index)
            d = d.sort_index()

        # Filter to dates before session_date
        if session_date is not None:
            pre_filter_size = len(d)
            d = d[d.index.date < session_date]
            logger.debug(f"get_previous_day_levels: Filtered to dates < {session_date}: {len(d)}/{pre_filter_size}")

        # Remove rows with zero/missing volume
        if "volume" in d.columns:
            pre_vol_size = len(d)
            d = d[d["volume"].fillna(0) > 0]
            logger.debug(f"get_previous_day_levels: Removed zero volume: {len(d)}/{pre_vol_size}")

        # Remove rows with missing high/low
        pre_clean_size = len(d)
        d = d[d["high"].notna() & d["low"].notna()]
        logger.debug(f"get_previous_day_levels: Removed missing high/low: {len(d)}/{pre_clean_size}")

        if d.empty:
            logger.debug("get_previous_day_levels: No valid data after filtering")
            if enable_fallback and fallback_df is not None:
                return _estimate_levels_from_fallback(fallback_df)
            return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}

        # Extract previous day (last row)
        prev = d.iloc[-1]
        pdh = float(prev["high"])
        pdl = float(prev["low"])
        pdc = float(prev.get("close", float("nan")))

        logger.debug(f"get_previous_day_levels: PDH={pdh:.2f}, PDL={pdl:.2f}, PDC={pdc:.2f} from date {prev.name}")

        return {"PDH": pdh, "PDL": pdl, "PDC": pdc}

    except Exception as e:
        logger.error(f"get_previous_day_levels: Error computing levels: {e}", exc_info=True)
        if enable_fallback and fallback_df is not None:
            return _estimate_levels_from_fallback(fallback_df)
        return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}


def _estimate_levels_from_fallback(fallback_df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate PDH/PDL/PDC from early session bars when daily data unavailable.

    This is used as a fallback for backtests where previous day data may not exist.
    Uses first 10 bars of session and adds 2% buffer to high/low.

    Args:
        fallback_df: Intraday (typically 5m) DataFrame

    Returns:
        Estimated levels dict (or nan if fallback fails)
    """
    try:
        if fallback_df is None or fallback_df.empty or len(fallback_df) < 10:
            logger.debug("_estimate_levels_from_fallback: Insufficient fallback data")
            return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}

        early_bars = fallback_df.iloc[:min(10, len(fallback_df))]
        pdh_est = float(early_bars["high"].max() * 1.02)  # 2% above early high
        pdl_est = float(early_bars["low"].min() * 0.98)   # 2% below early low
        pdc_est = float(early_bars["close"].iloc[-1])

        logger.info(f"_estimate_levels_from_fallback: Using estimated levels - PDH={pdh_est:.2f}, PDL={pdl_est:.2f}, PDC={pdc_est:.2f}")

        return {"PDH": pdh_est, "PDL": pdl_est, "PDC": pdc_est}

    except Exception as e:
        logger.error(f"_estimate_levels_from_fallback: Error estimating levels: {e}", exc_info=True)
        return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}
