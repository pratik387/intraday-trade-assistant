from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional, Union

IST_TZ = "Asia/Kolkata"

def _now_naive_ist() -> pd.Timestamp:
    """Return current time as a naive (tz-unaware) IST timestamp."""
    return pd.Timestamp.now(tz=IST_TZ).tz_localize(None)

def _to_naive_ist(ts: pd.Timestamp) -> pd.Timestamp:
    """Coerce any timestamp to naive IST (drop tz)."""
    if ts.tzinfo is None:
        # Assume it's already IST wall-time
        return ts
    # Convert to IST wall-time, then drop tz
    return ts.tz_convert(IST_TZ).tz_localize(None)

def ensure_naive_ist_index(df):
    if df.index.tz is None:
        return df
    out = df.copy()
    out.index = df.index.tz_convert(IST_TZ).tz_localize(None)
    return out

def _drop_forming_last_bar(
    df: pd.DataFrame,
    freq: str = "1min",
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Naive-IST version.
    Removes the last bar if it is still forming based on `freq`.

    Assumptions:
      - Index is DatetimeIndex in **naive IST** (tz-unaware IST wall-time).
      - If index is tz-aware (legacy data), we'll coerce it to naive IST once.
    """
    if df is None or df.empty or len(df.index) < 2:
        return df

    # Ensure DatetimeIndex and coerce to naive IST if needed (one-time pass)
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try a common timestamp column fallback
        if "timestamp" in df.columns:
            idx = pd.to_datetime(df["timestamp"])
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or 'timestamp' column.")
        # If tz-aware, convert to IST then drop tz
        if idx.tz is not None:
            idx = idx.tz_convert(IST_TZ).tz_localize(None)
        df = df.copy()
        df.index = idx
    elif df.index.tz is not None:
        # Convert legacy tz-aware indices to naive IST
        df = df.copy()
        df.index = df.index.tz_convert(IST_TZ).tz_localize(None)

    last_ts: pd.Timestamp = df.index[-1]

    # Determine "now" as naive IST
    if now is None:
        now = _now_naive_ist()
    else:
        # Coerce provided `now` to naive IST if it's tz-aware
        if isinstance(now, pd.Timestamp) and now.tzinfo is not None:
            now = now.tz_convert(IST_TZ).tz_localize(None)

    # Frequency delta
    try:
        freq_delta = pd.tseries.frequencies.to_offset(freq).delta
    except Exception:
        freq_delta = pd.Timedelta(freq)

    # If within the current bar window, drop it
    if (now - last_ts) < freq_delta:
        return df.iloc[:-1].copy()

    return df

# --- Intraday session helpers (IST, naive) ---

def _parse_hhmm_str(s: str) -> Tuple[int, int]:
    """'HH:MM' -> (HH, MM) with basic validation."""
    if not isinstance(s, str) or ":" not in s:
        raise ValueError(f"Invalid HH:MM string: {s}")
    hh, mm = s.split(":")
    return int(hh), int(mm)

def _hhmm(ts: pd.Timestamp) -> str:
    """Return naive-IST ts as 'HH:MM'."""
    ts = _to_naive_ist(ts)
    return ts.strftime("%H:%M")

def is_within_window(now: pd.Timestamp, start_hhmm: str, end_hhmm: str) -> bool:
    """Return True if 'now' is within [start_hhmm, end_hhmm] inclusive."""
    s_h, s_m = _parse_hhmm_str(start_hhmm)
    e_h, e_m = _parse_hhmm_str(end_hhmm)
    n = _to_naive_ist(now)
    n_h, n_m = n.hour, n.minute
    # same-day window; we do not handle cross-midnight here (not needed for market hours)
    start_tuple = (s_h, s_m)
    end_tuple   = (e_h, e_m)
    now_tuple   = (n_h, n_m)
    return start_tuple <= now_tuple <= end_tuple

def is_lunch(now: pd.Timestamp, cfg: dict) -> bool:
    if not bool(cfg.get("enable_lunch_pause", False)):
        return False
    start = str(cfg.get("lunch_start", "12:15"))
    end   = str(cfg.get("lunch_end", "13:15"))
    return is_within_window(now, start, end)

def is_after_cutoff(now: pd.Timestamp, cfg: dict) -> bool:
    cutoff = str(cfg.get("intraday_cutoff_hhmm", "1510"))
    # If now >= cutoff -> block new entries
    n = _to_naive_ist(now)
    return _hhmm(n) >= cutoff

# --- NEW shared helpers (used by planner, trade & exit executors) ---

def _minute_of_day(ts: Union[pd.Timestamp, str]) -> int:
    """
    Convert a timestamp (or parseable string) to minutes since midnight IST.
    """
    t = pd.Timestamp(ts)
    t = _to_naive_ist(t)
    return t.hour * 60 + t.minute

def _parse_hhmm_to_md(val: Optional[Union[str, int]]) -> Optional[int]:
    """
    Parse 'HH:MM', 'HHMM', or int like 1510 -> minute-of-day.
    Returns None if unparsable.
    """
    if val is None:
        return None
    try:
        s = str(val).strip()
        if ":" in s:
            hh, mm = s.split(":")
            return int(hh) * 60 + int(mm)
        if s.isdigit() and len(s) >= 3:
            hh = int(s[:-2]); mm = int(s[-2:])
            return hh * 60 + mm
        v = int(s)
        if v >= 1000:
            hh, mm = divmod(v, 100)
            return hh * 60 + mm
        return v
    except Exception:
        return None
