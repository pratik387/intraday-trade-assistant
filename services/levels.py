# services/levels.py
"""
Price Levels Utilities
----------------------
Single responsibility utilities to compute key intraday levels:

- opening_range(df_5m) -> (orb_high, orb_low)
  Uses session open + ORB minutes from entry_config.json.

- yesterday_levels(df_daily) -> (y_high, y_low)
  Previous session's high/low.

- broke_above(level, close) / broke_below(level, close)
  Breakout checks using buffer from entry_config.json (in bps).

- distance_bpct(level, price) / distance_bps(level, price)
  Distance of price from a level.

Assumptions:
- DataFrames use naive IST DatetimeIndex (see utils.time_util.ensure_naive_ist_index).
- Strictly config-driven; NO in-code defaults.
"""

from __future__ import annotations
from datetime import datetime as pydt, time as dtime
from typing import Tuple

import pandas as pd

from config.logging_config import get_agent_logger
from config.filters_setup import load_filters
from utils.time_util import ensure_naive_ist_index

logger = get_agent_logger()


def _parse_hhmm(hhmm: str) -> dtime:
    """Parse 'HHMM' -> datetime.time."""
    hhmm = str(hhmm)
    if len(hhmm) != 4 or not hhmm.isdigit():
        raise ValueError(f"Invalid HHMM string: {hhmm!r}")
    return dtime(int(hhmm[:2]), int(hhmm[2:]))

def opening_range(df_5m: pd.DataFrame, symbol: str = "UNKNOWN") -> Tuple[float, float]:
    """
    Compute the Opening Range (ORB) high/low for the **current session**.

    Config (required in entry_config.json):
      - session_open_hhmm : "0915"
      - orb_minutes       : 15

    Returns
    -------
    (orb_high, orb_low) as floats. If window not found, returns (nan, nan).
    """
    try:
        if df_5m is None or df_5m.empty:
            return float("nan"), float("nan")

        cfg = load_filters()
        open_hhmm = _parse_hhmm(cfg["session_open_hhmm"])
        orb_minutes = int(cfg["orb_minutes"])

        d = ensure_naive_ist_index(df_5m).copy()
        # Work on the latest session only (rows with the last date in the frame)
        last_date = d.index[-1].date()

        session_start = pd.Timestamp(pydt.combine(last_date, open_hhmm))
        session_end = session_start + pd.Timedelta(minutes=orb_minutes)

        # CRITICAL: Bars are START-STAMPED (Zerodha/broker convention)
        # With START-STAMPED 5m bars:
        #   09:15 bar = [09:15-09:20) data (timestamped at start: 09:15)
        #   09:20 bar = [09:20-09:25) data (timestamped at start: 09:20)
        #   09:25 bar = [09:25-09:30) data (timestamped at start: 09:25)
        #   09:30 bar = [09:30-09:35) data (timestamped at start: 09:30) ← Should NOT be included!
        # For OR period 09:15-09:30, query: >= session_start AND < session_end
        # This matches bars timestamped: 09:15, 09:20, 09:25 = 15 minutes ✅
        win = d.loc[(d.index >= session_start) & (d.index < session_end)]

        if win.empty:
            # Enhanced diagnostics: show what data is actually available
            first_ts = d.index[0] if not d.empty else None
            last_ts = d.index[-1] if not d.empty else None
            total_rows = len(d)
            logger.warning(
                f"levels.opening_range [{symbol}]: empty window last_date={last_date} "
                f"start={session_start} end={session_end} | "
                f"data_range=[{first_ts} to {last_ts}] total_rows={total_rows}"
            )
            return float("nan"), float("nan")

        # CRITICAL FIX: Ensure we have full opening range period (3 x 5m bars minimum)
        if len(win) < 3:
            logger.warning(
                f"levels.opening_range [{symbol}]: insufficient bars ({len(win)} < 3) for full OR period "
                f"last_date={last_date} start={session_start} end={session_end}"
            )
            return float("nan"), float("nan")

        orb_high = float(win["high"].max())
        orb_low = float(win["low"].min())
        logger.debug(
            f"levels.opening_range: date={last_date} "
            f"high={orb_high:.4f} low={orb_low:.4f} rows={len(win)}"
        )
        return orb_high, orb_low

    except Exception as e:
        logger.exception(f"levels.opening_range error: {e}")
        return float("nan"), float("nan")


def yesterday_levels(df_daily: pd.DataFrame) -> Tuple[float, float]:
    """
    Previous session's high/low.

    Expects a daily bars DataFrame containing at least 2 rows.
    Works with either a DatetimeIndex or a 'date' column.
    """
    try:
        if df_daily is None or len(df_daily) < 2:
            return float("nan"), float("nan")

        d = df_daily.copy()
        # Normalize sorting ascending by date
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"])
            d = d.sort_values("date")
        elif isinstance(d.index, pd.DatetimeIndex):
            d = d.sort_index()
        else:
            # If neither, try to coerce an index
            d.index = pd.to_datetime(d.index)
            d = d.sort_index()

        prev = d.iloc[-2]
        y_high = float(prev["high"])
        y_low = float(prev["low"])
        logger.debug(f"levels.yesterday_levels: y_high={y_high:.4f} y_low={y_low:.4f}")
        return y_high, y_low

    except Exception as e:
        logger.exception(f"levels.yesterday_levels error: {e}")
        return float("nan"), float("nan")


def broke_above(level: float, close: float) -> bool:
    """
    True if `close` > level * (1 + buffer_bps/10000).
    Buffer comes from entry_config.json: breakout_buffer_bps (e.g., 8).
    """
    try:
        if pd.isna(level) or pd.isna(close):
            return False
        cfg = load_filters()
        bps = float(cfg["breakout_buffer_bps"])
        return close > level * (1.0 + bps / 1e4)
    except Exception as e:
        logger.exception(f"levels.broke_above error: {e}")
        return False


def broke_below(level: float, close: float) -> bool:
    """
    True if `close` < level * (1 - buffer_bps/10000).
    Symmetric to broke_above; same config key used.
    """
    try:
        if pd.isna(level) or pd.isna(close):
            return False
        cfg = load_filters()
        bps = float(cfg["breakout_buffer_bps"])
        return close < level * (1.0 - bps / 1e4)
    except Exception as e:
        logger.exception(f"levels.broke_below error: {e}")
        return False


def distance_bpct(level: float, price: float) -> float:
    """
    Percentage distance of price from level (positive if above).
    Returns NaN if inputs invalid.
    """
    try:
        if pd.isna(level) or pd.isna(price) or level == 0:
            return float("nan")
        return (price / level - 1.0) * 100.0
    except Exception as e:
        logger.exception(f"levels.distance_bpct error: {e}")
        return float("nan")


def distance_bps(level: float, price: float) -> float:
    """
    Basis-points distance of price from level (positive if above), i.e. 100 * bpct.
    """
    try:
        bpct = distance_bpct(level, price)
        return bpct * 100.0 if pd.notna(bpct) else float("nan")
    except Exception as e:
        logger.exception(f"levels.distance_bps error: {e}")
        return float("nan")
