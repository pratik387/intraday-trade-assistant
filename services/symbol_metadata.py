"""Symbol metadata + universe membership lookups (Phase C consolidation).

This module is the single source of truth for everything detectors need to
know ABOUT a symbol that does NOT depend on its current bars/quotes:

  Cap segment / sizing
    - get_cap_segment(symbol)  — large_cap / mid_cap / small_cap / unknown
    - get_mis_info(symbol)     — MIS eligibility + leverage from nse_all.json
    - _normalize_symbol(...)   — strip 'NSE:' prefix and '.NS' suffix

  Universe membership (per-setup eligibility)
    - in_nifty50, in_banknifty, in_fno_liquid_200
    - in_universe(symbol, key) — generic dispatch over the universe map

  Cross-cutting per-bar exclusions (still per-symbol, no bar history needed)
    - is_expiry_day(date)              — NSE F&O weekly expiry
    - near_circuit_band(price, pdc, …) — circuit-band proximity guard

History — relocated 2026-04-30 from:
  pipelines/base_pipeline.py    (cap segment, MIS info, _normalize_symbol)
  services/universe_filter.py   (universe + expiry + circuit helpers)

Detectors should import from THIS module only — neither of the two source
modules will exist after Phase C completes.
"""
from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from config.logging_config import get_agent_logger


logger = get_agent_logger()

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = _REPO_ROOT / "assets"


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------


def _normalize_symbol(symbol: str) -> str:
    """Strip exchange prefix ('NSE:') and suffix ('.NS' / '.BO') so cache keys
    match nse_all.json's bare symbol names ('RELIANCE')."""
    if ":" in symbol:
        symbol = symbol.split(":")[-1]
    if "." in symbol:
        symbol = symbol.split(".")[0]
    return symbol


# ---------------------------------------------------------------------------
# nse_all.json-backed caches  (cap segment, MIS info)
# ---------------------------------------------------------------------------

_cap_segment_cache: Dict[str, str] = {}
_cap_segment_loaded: bool = False

_mis_info_cache: Dict[str, dict] = {}
_mis_info_loaded: bool = False

_NSE_ALL_PATH = _REPO_ROOT / "nse_all.json"


def _load_nse_all() -> Optional[list]:
    """Read nse_all.json once; return None on miss/parse-fail (caller decides
    fallback). Logged at debug to avoid noise on missing files."""
    if not _NSE_ALL_PATH.exists():
        return None
    try:
        with _NSE_ALL_PATH.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"symbol_metadata: failed to read {_NSE_ALL_PATH.name}: {e}")
        return None


def get_cap_segment(symbol: str) -> str:
    """Return market-cap bucket for `symbol`: large_cap / mid_cap / small_cap /
    unknown. Used for cap-aware sizing (Van Tharp evidence): large=1.2×,
    mid=1.0×, small=0.6× + wider stops. Unknown → caller treats as mid-cap.
    """
    global _cap_segment_cache, _cap_segment_loaded
    if not _cap_segment_loaded:
        data = _load_nse_all()
        if data is not None:
            _cap_segment_cache = {
                _normalize_symbol(item["symbol"]): item.get("cap_segment", "unknown")
                for item in data
            }
            logger.debug(f"CAP_SEGMENT: Loaded {len(_cap_segment_cache)} symbols")
        _cap_segment_loaded = True  # don't retry on miss
    return _cap_segment_cache.get(_normalize_symbol(symbol), "unknown")


def get_mis_info(symbol: str) -> dict:
    """Return MIS (Margin Intraday Square-off) info for `symbol`:
        {"mis_enabled": bool, "mis_leverage": float | None}
    Default {mis_enabled=False, mis_leverage=None} on cache miss.
    """
    global _mis_info_cache, _mis_info_loaded
    if not _mis_info_loaded:
        data = _load_nse_all()
        if data is not None:
            _mis_info_cache = {
                _normalize_symbol(item["symbol"]): {
                    "mis_enabled": item.get("mis_enabled", False),
                    "mis_leverage": item.get("mis_leverage"),
                }
                for item in data
            }
            logger.debug(f"MIS_INFO: Loaded {len(_mis_info_cache)} symbols")
        _mis_info_loaded = True
    return _mis_info_cache.get(
        _normalize_symbol(symbol),
        {"mis_enabled": False, "mis_leverage": None},
    )


# ---------------------------------------------------------------------------
# Universe membership  (NIFTY50 / BankNifty / F&O liquid 200 / smallmid F&O)
# ---------------------------------------------------------------------------


def _load_csv_symbols(filename: str) -> Set[str]:
    """Load a flat NSE constituent CSV, returning a set of 'NSE:XYZ' tickers.
    Tries 'Symbol' column first, falls back to 'symbol'. Empty set on miss."""
    path = _ASSETS / filename
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else ("symbol" if "symbol" in df.columns else None)
    if col is None:
        return set()
    out: Set[str] = set()
    for s in df[col].astype(str):
        s = s.strip().upper()
        if not s:
            continue
        if not s.startswith("NSE:"):
            s = f"NSE:{s}"
        out.add(s)
    return out


# Load once at import — cheap for ~50-200 row CSVs.
_NIFTY50: Set[str] = _load_csv_symbols("ind_nifty50list.csv")
_BANKNIFTY: Set[str] = _load_csv_symbols("ind_niftybanklist.csv")
_FNO_LIQUID_200: Set[str] = _load_csv_symbols("fno_liquid_200.csv")
_NIFTY50_BANKNIFTY: Set[str] = _NIFTY50 | _BANKNIFTY
_SMALLMID_FNO: Set[str] = _FNO_LIQUID_200 - _NIFTY50

_UNIVERSE_MAP: Dict[str, Set[str]] = {
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
    """Dispatch by universe key. Raises KeyError on unknown key — fail-fast so
    a typo in detector config is caught at first detect() invocation rather
    than producing a silently-empty universe."""
    if universe_key not in _UNIVERSE_MAP:
        raise KeyError(
            f"Unknown universe key: {universe_key!r}. "
            f"Valid: {sorted(_UNIVERSE_MAP.keys())}"
        )
    return symbol in _UNIVERSE_MAP[universe_key]


# ---------------------------------------------------------------------------
# Expiry-day calendar  (NSE F&O weekly)
# ---------------------------------------------------------------------------


def _load_expiry_dates() -> Set[_date]:
    """Load NSE F&O weekly expiry dates from assets/nse_fno_expiry_dates.csv
    (single 'date' column, ISO YYYY-MM-DD). Returns empty set if missing —
    callers fall back to weekday heuristic."""
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
      - Pre-2025-09-01: Thursday is Nifty weekly expiry
      - 2025-09-01 onwards: Tuesday (post-SEBI consolidation)
    """
    if d is None:
        return False
    # pd.Timestamp inherits from datetime so coerce first
    if isinstance(d, pd.Timestamp):
        d = d.date()
    elif not isinstance(d, _date):
        try:
            d = pd.to_datetime(d).date()
        except (ValueError, TypeError):
            return False

    if _EXPIRY_DATES:
        return d in _EXPIRY_DATES

    weekday = d.weekday()  # Mon=0..Sun=6
    cutoff = _date(2025, 9, 1)
    if d < cutoff:
        return weekday == 3  # Thursday
    return weekday == 1  # Tuesday


# ---------------------------------------------------------------------------
# Circuit-band proximity guard
# ---------------------------------------------------------------------------


def near_circuit_band(
    current_price: float,
    prev_close: float,
    circuit_pct: float = 10.0,
    proximity_pct: float = 2.0,
) -> bool:
    """Return True if `current_price` is within `proximity_pct`% of either
    circuit boundary (upper = pdc × (1 + circuit_pct/100), lower = pdc × …).

    Circuit-band-near positions can become forced-delivery (StockGro / 5paisa
    references), so all sub8 setups exclude symbols where price approaches
    the band. Default circuit_pct=10.0 covers most F&O / index components;
    smaller-band stocks are flagged earlier than strictly necessary, which is
    the safe direction.
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
