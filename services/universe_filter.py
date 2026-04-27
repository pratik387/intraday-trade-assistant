"""Per-setup universe membership lookup + cross-cutting exclusion helpers (sub8-T1, rev2).

Universe keys consumed by sub8 detector configs:
  - "nifty50"               — Nifty 50 only (CPR breakout context)
  - "banknifty"             — Bank Nifty only
  - "nifty50_banknifty"     — union of Nifty 50 + Bank Nifty (sub8 narrow_cpr_breakout)
  - "fno_liquid_200"        — F&O liquid ~120-200 names (ORB, VWAP, CHR)
  - "smallmid_fno"          — F&O liquid MINUS Nifty 50 (PDH/PDL fade)

Cross-cutting helpers (rev2 — design Section 10a):
  - is_expiry_day(date)              — NSE F&O weekly expiry exclusion
  - near_circuit_band(price, pdc, …) — universe-wide circuit-band proximity exclusion
"""
from __future__ import annotations

from datetime import date as _date
from pathlib import Path
from typing import Optional, Set

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = _REPO_ROOT / "assets"


def _load_csv_symbols(filename: str) -> Set[str]:
    """Load symbols from an NSE constituent CSV. Tries 'Symbol' column first,
    falls back to 'symbol'. Returns prefixed 'NSE:XYZ' set."""
    path = _ASSETS / filename
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else ("symbol" if "symbol" in df.columns else None)
    if col is None:
        return set()
    out = set()
    for s in df[col].astype(str):
        s = s.strip().upper()
        if not s:
            continue
        if not s.startswith("NSE:"):
            s = f"NSE:{s}"
        out.add(s)
    return out


# Load once at import time — cheap for ~50-200 row CSVs.
_NIFTY50: Set[str] = _load_csv_symbols("ind_nifty50list.csv")
_BANKNIFTY: Set[str] = _load_csv_symbols("ind_niftybanklist.csv")
_FNO_LIQUID_200: Set[str] = _load_csv_symbols("fno_liquid_200.csv")
_NIFTY50_BANKNIFTY: Set[str] = _NIFTY50 | _BANKNIFTY
_SMALLMID_FNO: Set[str] = _FNO_LIQUID_200 - _NIFTY50

_UNIVERSE_MAP = {
    "nifty50": _NIFTY50,
    "banknifty": _BANKNIFTY,
    "nifty50_banknifty": _NIFTY50_BANKNIFTY,
    "fno_liquid_200": _FNO_LIQUID_200,
    "smallmid_fno": _SMALLMID_FNO,
}


def in_nifty50(symbol: str) -> bool:
    return symbol in _NIFTY50


def in_banknifty(symbol: str) -> bool:
    return symbol in _BANKNIFTY


def in_fno_liquid_200(symbol: str) -> bool:
    return symbol in _FNO_LIQUID_200


def in_universe(symbol: str, universe_key: str) -> bool:
    """Dispatch by universe key. Raises KeyError on unknown key."""
    if universe_key not in _UNIVERSE_MAP:
        raise KeyError(f"Unknown universe key: {universe_key!r}. "
                       f"Valid: {sorted(_UNIVERSE_MAP.keys())}")
    return symbol in _UNIVERSE_MAP[universe_key]


# ---------------------------------------------------------------------------
# Cross-cutting exclusion helpers (sub8 design Section 10a)
# ---------------------------------------------------------------------------


def _load_expiry_dates() -> Set[_date]:
    """Load NSE F&O weekly expiry dates from assets/nse_fno_expiry_dates.csv.

    Expected CSV format: single column 'date' with ISO YYYY-MM-DD strings.
    Returns empty set if file missing — caller falls back to heuristic.
    """
    path = _ASSETS / "nse_fno_expiry_dates.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
    if col is None:
        return set()
    out: Set[_date] = set()
    for s in df[col].astype(str):
        try:
            out.add(pd.to_datetime(s.strip()).date())
        except (ValueError, TypeError):
            continue
    return out


_EXPIRY_DATES: Set[_date] = _load_expiry_dates()


def is_expiry_day(d) -> bool:
    """Return True if `d` is an NSE F&O weekly expiry day.

    Primary source: assets/nse_fno_expiry_dates.csv (curated calendar).

    Fallback heuristic when calendar is empty/missing:
      - Pre-2025-09-01: Thursday is Nifty weekly expiry (BankNifty was Wednesday
        but Wednesday discontinuation makes Thursday the dominant signal)
      - 2025-09-01 onwards: Tuesday is Nifty weekly expiry (post-SEBI consolidation)

    Sources: HDFC Sky, AlgoTest blog, Sahi.com expiry-day rules.
    """
    if d is None:
        return False
    # Coerce pandas Timestamp first (it inherits from datetime, so isinstance
    # check below would mis-handle it); call .date() to drop the time component.
    if isinstance(d, pd.Timestamp):
        d = d.date()
    elif not isinstance(d, _date):
        try:
            d = pd.to_datetime(d).date()
        except (ValueError, TypeError):
            return False

    if _EXPIRY_DATES:
        return d in _EXPIRY_DATES

    # Heuristic fallback (less precise than calendar but better than nothing)
    weekday = d.weekday()  # Mon=0..Sun=6
    cutoff = _date(2025, 9, 1)
    if d < cutoff:
        return weekday == 3  # Thursday
    return weekday == 1  # Tuesday


def near_circuit_band(
    current_price: float,
    prev_close: float,
    circuit_pct: float = 10.0,
    proximity_pct: float = 2.0,
) -> bool:
    """Return True if `current_price` is within `proximity_pct`% of either circuit
    boundary (upper = pdc × (1 + circuit_pct/100), lower = pdc × (1 − circuit_pct/100)).

    Circuit-band-near positions can become forced-delivery (StockGro, 5paisa),
    so all sub8 setups exclude symbols where price approaches the band.

    Default circuit_pct=10.0 is the most-common F&O / index-component band.
    Symbols with 5% or 20% bands are typically smaller / SME — the default is
    conservative (a 5% stock would be flagged "near band" earlier than necessary,
    which is fine).
    """
    if prev_close <= 0 or current_price <= 0:
        return True  # invalid prices — be safe, exclude
    upper = prev_close * (1.0 + circuit_pct / 100.0)
    lower = prev_close * (1.0 - circuit_pct / 100.0)
    prox_band = prev_close * (proximity_pct / 100.0)
    return (
        current_price >= upper - prox_band
        or current_price <= lower + prox_band
    )
