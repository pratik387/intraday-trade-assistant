"""
ORB Level Calculator - Self-contained computation for MDS.

Computes Opening Range High/Low (ORH/ORL) and Previous Day Levels (PDH/PDL/PDC)
for publishing to Redis. Pure pandas computation with no dependencies on
services/, utils/, structures/, or broker/ packages.

Logic mirrors:
  - services/levels.py:opening_range() for ORH/ORL
  - utils/level_utils.py:get_previous_day_levels() for PDH/PDL/PDC
"""

from __future__ import annotations

from datetime import datetime as pydt, time as dtime, date
from typing import Dict, Optional, Tuple

import pandas as pd

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


NAN = float("nan")


def compute_opening_range(
    df_5m: Optional[pd.DataFrame],
    session_open_hhmm: str,
    orb_minutes: int,
    symbol: str = "UNKNOWN",
) -> Tuple[float, float]:
    """
    Compute Opening Range High/Low from 5-minute bars.

    Bars must be START-STAMPED (Zerodha/broker convention):
      09:15 bar = data from [09:15, 09:20)
      09:20 bar = data from [09:20, 09:25)
      09:25 bar = data from [09:25, 09:30)

    Args:
        df_5m: 5-minute OHLCV DataFrame with DatetimeIndex (start-stamped)
        session_open_hhmm: Market open time as "HHMM" string (e.g. "0915")
        orb_minutes: Opening range duration in minutes (e.g. 15)
        symbol: Symbol name for logging

    Returns:
        (orb_high, orb_low) as floats, or (NaN, NaN) if insufficient data.
    """
    try:
        if df_5m is None or df_5m.empty:
            return NAN, NAN

        # Parse session open time
        open_time = dtime(int(session_open_hhmm[:2]), int(session_open_hhmm[2:]))

        # Use the last date in the DataFrame
        last_date = df_5m.index[-1].date()

        session_start = pd.Timestamp(pydt.combine(last_date, open_time))
        session_end = session_start + pd.Timedelta(minutes=orb_minutes)

        # Start-stamped bar convention: >= start AND < end
        # For 09:15-09:30 range: matches bars 09:15, 09:20, 09:25 (3 bars)
        win = df_5m.loc[(df_5m.index >= session_start) & (df_5m.index < session_end)]

        if win.empty:
            logger.debug(
                f"orb_calculator [{symbol}]: empty window last_date={last_date} "
                f"start={session_start} end={session_end}"
            )
            return NAN, NAN

        # Require full opening range period (e.g. 3 x 5m bars for 15 minutes)
        min_bars = orb_minutes // 5
        if len(win) < min_bars:
            logger.debug(
                f"orb_calculator [{symbol}]: insufficient bars ({len(win)} < {min_bars})"
            )
            return NAN, NAN

        orb_high = float(win["high"].max())
        orb_low = float(win["low"].min())
        return orb_high, orb_low

    except Exception as e:
        logger.debug(f"orb_calculator [{symbol}]: error: {e}")
        return NAN, NAN


def compute_previous_day_levels(
    daily_df: Optional[pd.DataFrame],
    session_date: date,
) -> Dict[str, float]:
    """
    Compute Previous Day High/Low/Close from daily OHLCV data.

    Args:
        daily_df: Daily OHLCV DataFrame with DatetimeIndex
        session_date: Current session date (uses last trading day BEFORE this)

    Returns:
        Dict with "PDH", "PDL", "PDC" (float values, NaN if unavailable)
    """
    nan_result = {"PDH": NAN, "PDL": NAN, "PDC": NAN}

    try:
        if daily_df is None or daily_df.empty:
            return nan_result

        d = daily_df.copy()

        # Normalize index to DatetimeIndex
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"])
            d = d.sort_values("date").set_index("date")
        else:
            d.index = pd.to_datetime(d.index)
            d = d.sort_index()

        # Filter to dates before session_date
        d = d[d.index.date < session_date]

        # Remove zero-volume rows
        if "volume" in d.columns:
            d = d[d["volume"].fillna(0) > 0]

        # Remove rows with missing high/low
        d = d[d["high"].notna() & d["low"].notna()]

        if d.empty:
            return nan_result

        # Previous day = last valid row
        prev = d.iloc[-1]
        return {
            "PDH": float(prev["high"]),
            "PDL": float(prev["low"]),
            "PDC": float(prev.get("close", NAN)),
        }

    except Exception as e:
        logger.debug(f"orb_calculator: previous day levels error: {e}")
        return nan_result
