"""Enrich per-symbol daily_df with `delivery_pct` from NSE bhavcopy archive.

Used by the `delivery_pct_anomaly_short` detector which expects
`delivery_pct` as a column in `ctx.df_daily`. Falls back to silent no-op
if the source parquet is unavailable.

Source: `data/delivery_pct/delivery_history.parquet`
  Schema: symbol, date, series, delivery_qty, total_traded_qty,
          delivery_pct (0-100), total_traded_value, close_price.
  Built by: `tools/delivery_pct/fetch_delivery.py` (idempotent CLI;
  re-run daily after market close to top up).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_PARQUET = _REPO / "data" / "delivery_pct" / "delivery_history.parquet"

# Module-level cache: (symbol, date) -> delivery_pct
_DELIVERY_LOOKUP: Optional[dict] = None
_LOAD_ATTEMPTED = False


def _load() -> Optional[dict]:
    """Load parquet once; cache as dict for O(1) lookup. Returns None on failure."""
    global _DELIVERY_LOOKUP, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _DELIVERY_LOOKUP
    _LOAD_ATTEMPTED = True
    if not _PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(_PARQUET, columns=["symbol", "date", "series", "delivery_pct"])
        df = df[df["series"] == "EQ"]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # Build dict for O(1) lookup
        _DELIVERY_LOOKUP = dict(zip(zip(df["symbol"], df["date"]), df["delivery_pct"]))
        return _DELIVERY_LOOKUP
    except Exception:
        return None


def enrich(daily_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add `delivery_pct` column to a per-symbol daily_df. Returns df with column added.

    `symbol` should be the NSE bare ticker (no "NSE:" prefix). If the input
    has a prefix, strips it before lookup.

    Idempotent: if `delivery_pct` already in df, returns unchanged.
    Lookup miss: column added with NaN values (detector will silently no-fire
    on those days).
    """
    if daily_df is None or daily_df.empty:
        return daily_df
    if "delivery_pct" in daily_df.columns:
        return daily_df

    lookup = _load()
    if lookup is None:
        return daily_df

    bare = symbol.replace("NSE:", "") if symbol.startswith("NSE:") else symbol

    # Try to extract dates from index or date column
    if isinstance(daily_df.index, pd.DatetimeIndex):
        dates = daily_df.index.date
    elif "date" in daily_df.columns:
        dates = pd.to_datetime(daily_df["date"]).dt.date
    elif "ts" in daily_df.columns:
        dates = pd.to_datetime(daily_df["ts"]).dt.date
    else:
        # Cannot determine dates; bail
        daily_df = daily_df.copy()
        daily_df["delivery_pct"] = float("nan")
        return daily_df

    daily_df = daily_df.copy()
    daily_df["delivery_pct"] = [lookup.get((bare, d), float("nan")) for d in dates]
    return daily_df


def enrich_dict(daily_dict: dict) -> dict:
    """Bulk enrichment: apply enrich() to each (symbol, daily_df) pair.

    Mutates in place via dict comprehension; returns same dict for chaining.
    """
    lookup = _load()
    if lookup is None:
        return daily_dict
    return {sym: enrich(dd, sym) for sym, dd in daily_dict.items()}
