"""Cross-day RVOL baseline lookup for delivery_pct_anomaly_short.

Replaces the broken `_cross_day_rvol` path that filtered df_5m for prior
same-tod bars — that path returned None in production because the screener
caps df_5m at 120 bars (`screener_store_5m_max`), so 20+ days of prior
same-tod history is never available.

This module loads a precomputed (symbol, date, hhmm) → vol_mean20 lookup
built by `tools/cross_day_rvol/build_baseline.py`. Detector calls
`get_baseline_vol(symbol, session_date, hhmm)` and computes
RVOL = today_bar_vol / baseline.

Memory model
------------
The full parquet has ~21M rows for 2023-01..2026-04 — building a dict of
that size costs ~6 GB of resident memory and OOM-killed pods on the OCI
cluster. Each backtest pod only ever queries a single `session_date`, so
we lazily filter the parquet to just that date's slice (~30k rows, ~10 MB
dict) on the first call. If subsequent calls arrive for a different date
(e.g. when the screener spans multiple sessions in tests), we re-filter.

Source: data/cross_day_rvol/rvol_baseline.parquet
  Built locally, uploaded to OCI bucket via
  oci/tools/upload_cross_day_rvol.py, downloaded at pod startup by
  oci/docker/entrypoint.py::download_cross_day_rvol().
"""
from __future__ import annotations

from datetime import date as _date
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_PARQUET = _REPO / "data" / "cross_day_rvol" / "rvol_baseline.parquet"

# Per-date cache: maps session_date → {(symbol, hhmm): vol_mean20}
_LOOKUP_BY_DATE: Dict[_date, Optional[Dict[Tuple[str, int], float]]] = {}
_FILE_MISSING_LOGGED = False


def _load_for_date(target_date: _date) -> Optional[Dict[Tuple[str, int], float]]:
    """Load parquet filtered to `target_date` rows only; cache result.

    Returns dict keyed by (symbol, hhmm) → vol_mean20, or None when the
    parquet is missing or the date has no rows.
    """
    global _FILE_MISSING_LOGGED
    if target_date in _LOOKUP_BY_DATE:
        return _LOOKUP_BY_DATE[target_date]
    if not _PARQUET.exists():
        if not _FILE_MISSING_LOGGED:
            _FILE_MISSING_LOGGED = True
        _LOOKUP_BY_DATE[target_date] = None
        return None
    try:
        # PyArrow filter pushdown — only reads `target_date` rows from disk.
        target_ts = pd.Timestamp(target_date)
        df = pd.read_parquet(
            _PARQUET,
            filters=[("date", "=", target_ts)],
        )
        if df.empty:
            _LOOKUP_BY_DATE[target_date] = None
            return None
        lookup = dict(
            zip(
                zip(df["symbol"], df["hhmm"].astype(int)),
                df["vol_mean20"].astype(float),
            )
        )
        _LOOKUP_BY_DATE[target_date] = lookup
        return lookup
    except Exception:
        _LOOKUP_BY_DATE[target_date] = None
        return None


def get_baseline_vol(symbol: str, session_date: _date, hhmm: int) -> Optional[float]:
    """Return prior-20-session same-tod mean volume for (symbol, date, hhmm).

    `symbol` may be "NSE:XXX" or bare "XXX"; both work — strips the prefix
    once before lookup since the parquet uses bare tickers.

    Returns None if:
      - parquet file is missing (graceful no-op for environments without
        the cache uploaded yet)
      - the (symbol, date, hhmm) key isn't present (insufficient history,
        symbol newly listed, weekend/holiday, etc.)
    """
    lookup = _load_for_date(session_date)
    if lookup is None:
        return None
    bare = symbol.replace("NSE:", "") if symbol.startswith("NSE:") else symbol
    val = lookup.get((bare, int(hhmm)))
    if val is None or val <= 0:
        return None
    return val
