"""Cross-day RVOL baseline lookup for delivery_pct_anomaly_short.

Replaces the broken `_cross_day_rvol` path that filtered df_5m for prior
same-tod bars — that path returned None in production because the screener
caps df_5m at 120 bars (`screener_store_5m_max`), so 20+ days of prior
same-tod history is never available.

This module loads a precomputed (symbol, date, hhmm) → vol_mean20 lookup
built by `tools/cross_day_rvol/build_baseline.py`. Detector calls
`get_baseline_vol(symbol, session_date, hhmm)` and computes
RVOL = today_bar_vol / baseline.

Source: data/cross_day_rvol/rvol_baseline.parquet
  Built locally, uploaded to OCI bucket via
  oci/tools/upload_cross_day_rvol.py, downloaded at pod startup by
  oci/docker/entrypoint.py::download_cross_day_rvol().
"""
from __future__ import annotations

from datetime import date as _date
from pathlib import Path
from typing import Optional

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_PARQUET = _REPO / "data" / "cross_day_rvol" / "rvol_baseline.parquet"

_LOOKUP: Optional[dict] = None
_LOAD_ATTEMPTED = False


def _load() -> Optional[dict]:
    """Load parquet once; cache as dict keyed by (symbol, date, hhmm)."""
    global _LOOKUP, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LOOKUP
    _LOAD_ATTEMPTED = True
    if not _PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(_PARQUET)
        # Normalize date to python date
        df["date"] = pd.to_datetime(df["date"]).dt.date
        _LOOKUP = dict(
            zip(
                zip(df["symbol"], df["date"], df["hhmm"].astype(int)),
                df["vol_mean20"].astype(float),
            )
        )
        return _LOOKUP
    except Exception:
        return None


def get_baseline_vol(symbol: str, session_date: _date, hhmm: int) -> Optional[float]:
    """Return prior-20-session same-tod mean volume for (symbol, date, hhmm).

    `symbol` may be "NSE:XXX" or bare "XXX"; both work — strips the prefix
    once before lookup since the parquet uses bare tickers.

    Returns None if:
      - parquet file is missing (graceful no-op for environments without
        the cache uploaded yet)
      - the (symbol, date, hhmm) key isn't present (insufficient history,
        symbol newly listed, etc.)
    """
    lookup = _load()
    if lookup is None:
        return None
    bare = symbol.replace("NSE:", "") if symbol.startswith("NSE:") else symbol
    val = lookup.get((bare, session_date, int(hhmm)))
    if val is None or val <= 0:
        return None
    return val
