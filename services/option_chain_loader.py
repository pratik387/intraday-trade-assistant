"""Public option-chain loader API for the expiry_pin_strike_reversal detector.

Reads daily OI snapshots from data/option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet
and exposes the lookups the detector needs:
  - load_oi_snapshot(session_date, oi_root, symbol=None) -> pd.DataFrame
  - find_max_oi_strike(session_date, symbol="NIFTY", expiry="weekly") -> float
  - is_expiry_day(session_date) -> bool   (re-export from universe_filter)
  - is_monthly_expiry(session_date) -> bool

The actual NSE bhavcopy backfill runs via tools/option_chain/fetch_oi_snapshot.py
and is the user's deferred step. This module assumes the parquet store
has been populated; callers must catch FileNotFoundError on missing
sessions and reject the detection gracefully.

Per specs/2026-04-29-expiry_pin_strike_reversal-plan.md Phase A4 + A5.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

# Re-export is_expiry_day from symbol_metadata — single source of truth.
from services.symbol_metadata import is_expiry_day   # noqa: F401  (re-export)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OI_ROOT = _REPO_ROOT / "data" / "option_chain"

# Module-level LRU cache (max 32 entries) keyed by (session_date, oi_root).
# Bounds memory in long-running backtests where multiple symbols query the
# same session repeatedly.
_CACHE_MAX_ENTRIES = 32
_SESSION_CACHE: "OrderedDict[Tuple[date, Path], pd.DataFrame]" = OrderedDict()


class OISnapshotMissing(FileNotFoundError):
    """Raised when the requested session has no parquet file in the store."""


def _coerce_date(d: Union[date, datetime, pd.Timestamp, str]) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    return pd.Timestamp(d).date()


def _parquet_path(session_date: date, oi_root: Path) -> Path:
    return (
        oi_root
        / f"{session_date.year:04d}"
        / f"{session_date.month:02d}"
        / f"{session_date.isoformat()}.parquet"
    )


def _cache_get(key: Tuple[date, Path]) -> Optional[pd.DataFrame]:
    if key in _SESSION_CACHE:
        # LRU touch — move to end (most-recently-used)
        df = _SESSION_CACHE.pop(key)
        _SESSION_CACHE[key] = df
        return df
    return None


def _cache_put(key: Tuple[date, Path], df: pd.DataFrame) -> None:
    if key in _SESSION_CACHE:
        _SESSION_CACHE.pop(key)
    _SESSION_CACHE[key] = df
    while len(_SESSION_CACHE) > _CACHE_MAX_ENTRIES:
        _SESSION_CACHE.popitem(last=False)   # evict LRU (front)


def clear_cache() -> None:
    """Clear the module-level OI cache (testing utility)."""
    _SESSION_CACHE.clear()


def load_oi_snapshot(
    session_date,
    oi_root: Optional[Path] = None,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Load the parquet snapshot for `session_date`.

    Returns a DataFrame with the canonical schema (per
    tools/option_chain/_nse_bhavcopy_client.parse_bhavcopy):
        session_date, symbol, expiry_date, strike, option_type (CE/PE),
        oi, oi_change, vol, ltp, settlement_price, iv

    If `symbol` is provided, the result is filtered to that symbol.

    Raises OISnapshotMissing (FileNotFoundError subclass) if the session's
    parquet doesn't exist.
    """
    sd = _coerce_date(session_date)
    root = (oi_root if oi_root is not None else _DEFAULT_OI_ROOT)
    root = Path(root)
    cache_key = (sd, root)
    cached = _cache_get(cache_key)
    if cached is None:
        path = _parquet_path(sd, root)
        if not path.exists():
            raise OISnapshotMissing(
                f"no OI snapshot at {path} for {sd}"
            )
        df = pd.read_parquet(path)
        _cache_put(cache_key, df)
        cached = df
    if symbol is not None:
        return cached[cached["symbol"] == symbol].copy()
    return cached.copy()


def is_monthly_expiry(session_date) -> bool:
    """Return True if session_date is the LAST weekly-expiry day of its month
    (i.e., the monthly expiry).

    Uses is_expiry_day to confirm it's an expiry day, then checks whether
    the next expiry day falls in a different month (which makes the current
    one "the last of the month" = monthly).
    """
    sd = _coerce_date(session_date)
    if not is_expiry_day(sd):
        return False
    # Walk forward up to 14 days looking for the next expiry day. If the next
    # expiry's month differs, the current day is the monthly.
    from datetime import timedelta
    nxt = sd + timedelta(days=1)
    for _ in range(14):
        if is_expiry_day(nxt):
            return nxt.month != sd.month
        nxt = nxt + timedelta(days=1)
    # If we couldn't find a next expiry within 14 days, we're either at the
    # tail end of the calendar or the calendar is sparse — fall back to the
    # heuristic "last-Thursday-of-month" check.
    return _is_last_thursday_of_month(sd)


def _is_last_thursday_of_month(d: date) -> bool:
    """Heuristic fallback: True if `d` is the last Thursday of its month."""
    if d.weekday() != 3:   # 3 = Thursday
        return False
    from datetime import timedelta
    nxt = d + timedelta(days=7)
    return nxt.month != d.month


def find_max_oi_strike(
    session_date,
    symbol: str = "NIFTY",
    expiry: str = "weekly",
    oi_root: Optional[Path] = None,
) -> float:
    """Return the strike with the highest aggregate OI (CE + PE) for the
    given symbol's relevant expiry on `session_date`.

    `expiry`:
      - "weekly":  use the nearest expiry_date >= session_date (typically
        the same week's Thursday/Tuesday).
      - "monthly": use the last-Thursday-of-month expiry within the next
        ~31 days. Falls back to "weekly" if none found.
      - "auto":    monthly if is_monthly_expiry(session_date) else weekly.

    Tied strikes are broken by returning the LOWER strike (deterministic).

    Raises:
      - OISnapshotMissing if the snapshot file doesn't exist.
      - ValueError if no contracts match the (symbol, expiry) selection.
    """
    sd = _coerce_date(session_date)
    df = load_oi_snapshot(sd, oi_root=oi_root, symbol=symbol)
    if df.empty:
        raise ValueError(f"no contracts for {symbol} on {sd}")

    # Pick the relevant expiry_date
    expiries = sorted(df["expiry_date"].dropna().unique())
    if not expiries:
        raise ValueError(f"no expiry dates parsed for {symbol} on {sd}")

    chosen_expiry: Optional[date] = None
    if expiry == "weekly":
        # Nearest expiry_date >= session_date (NSE settles at end of expiry day,
        # so same-day expiry IS the relevant one).
        future = [e for e in expiries if e >= sd]
        chosen_expiry = future[0] if future else expiries[-1]
    elif expiry == "monthly":
        # Last-Thursday-of-month expiry within the next ~31 days
        from datetime import timedelta
        horizon_end = sd + timedelta(days=35)
        candidates = [e for e in expiries if sd <= e <= horizon_end
                      and _is_last_thursday_of_month(e)]
        if candidates:
            chosen_expiry = candidates[0]
        else:
            future = [e for e in expiries if e >= sd]
            chosen_expiry = future[0] if future else expiries[-1]
    elif expiry == "auto":
        if is_monthly_expiry(sd):
            return find_max_oi_strike(sd, symbol, "monthly", oi_root=oi_root)
        return find_max_oi_strike(sd, symbol, "weekly", oi_root=oi_root)
    else:
        raise ValueError(f"unknown expiry mode: {expiry!r}")

    if chosen_expiry is None:
        raise ValueError(
            f"could not select expiry for {symbol} on {sd} (mode={expiry!r})"
        )

    sub = df[df["expiry_date"] == chosen_expiry]
    if sub.empty:
        raise ValueError(
            f"no contracts at chosen expiry {chosen_expiry} for {symbol} on {sd}"
        )

    # Sum CE + PE OI per strike, return argmax.
    grp = sub.groupby("strike", as_index=False)["oi"].sum()
    max_oi = grp["oi"].max()
    if max_oi <= 0:
        raise ValueError(
            f"all-zero OI at chosen expiry {chosen_expiry} for {symbol} on {sd}"
        )
    # Deterministic tie-break: lowest strike wins
    matches = grp[grp["oi"] == max_oi].sort_values("strike")
    return float(matches.iloc[0]["strike"])
