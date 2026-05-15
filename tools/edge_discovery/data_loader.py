"""5m feather loader (IST-naive timestamps)."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd


_REPO = Path(__file__).resolve().parents[2]
_FEATHER_DIR = _REPO / "backtest-cache-download" / "monthly"
_KEEP_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def _months_in(start: date, end: date) -> List[tuple]:
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        if m == 12:
            y += 1; m = 1
        else:
            m += 1
    return months


def load_5m_period(start: date, end: date, symbols: Optional[Set[str]] = None) -> pd.DataFrame:
    """Load monthly 5m feathers covering [start, end] and concat. IST-naive."""
    frames: List[pd.DataFrame] = []
    for y, m in _months_in(start, end):
        fp = _FEATHER_DIR / f"{y:04d}_{m:02d}_5m_enriched.feather"
        if not fp.exists():
            continue
        df = pd.read_feather(fp, columns=_KEEP_COLS)
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert(None)
        if symbols is not None:
            df = df[df["symbol"].isin(symbols)]
        # Strict date filter
        day_floor = df["date"].dt.floor("D")
        df = df[(day_floor >= pd.Timestamp(start)) & (day_floor <= pd.Timestamp(end))]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=_KEEP_COLS)
    return pd.concat(frames, ignore_index=True)
