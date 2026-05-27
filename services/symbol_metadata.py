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

# Per-session-date cap_segment caches. Backtests can simulate multiple dates
# in one process — caching by date keyword prevents re-reading the same
# snapshot file repeatedly. Key None == "live/paper, use latest".
_cap_segment_caches: Dict[Optional[str], Dict[str, str]] = {}

_mis_info_cache: Dict[str, dict] = {}
_mis_info_loaded: bool = False

_NSE_ALL_PATH = _REPO_ROOT / "nse_all.json"
_CAP_SEGMENTS_DIR = _REPO_ROOT / "data" / "cap_segments"


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


def _load_cap_segments_json(path: Path) -> Optional[Dict[str, str]]:
    """Read one cap_segments_*.json snapshot. Returns the {symbol: segment}
    classification map or None on miss/parse-fail.

    Snapshot schema (produced by scripts/refresh_cap_segments.py):
        {
          "generated_at": "YYYY-MM-DD",
          "source": "...",
          "n_symbols": <int>,
          "classification": {"NSE:RELIANCE": "large_cap", ...}
        }
    """
    if not path.exists():
        return None
    try:
        with path.open() as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("symbol_metadata: failed to read %s: %s", path, e)
        return None
    if not isinstance(payload, dict) or "classification" not in payload:
        logger.warning("symbol_metadata: %s missing 'classification' key", path)
        return None
    return payload["classification"]


def _pick_snapshot_path(session_date) -> Optional[Path]:
    """Return the dated snapshot path effective for `session_date`, or the
    latest snapshot when session_date is None (live/paper).

    For backtests we pick the most recent snapshot dated <= session_date
    so we use the cap-segment classification that was in effect on that
    day, not today's. Symbols whose listing post-dated the snapshot get
    classified as 'unknown' downstream — correct for point-in-time fidelity.
    """
    if not _CAP_SEGMENTS_DIR.exists():
        return None

    if session_date is None:
        latest = _CAP_SEGMENTS_DIR / "cap_segments_latest.json"
        return latest if latest.exists() else None

    # Coerce session_date to a date
    if isinstance(session_date, pd.Timestamp):
        sd = session_date.date()
    elif isinstance(session_date, _date) and not isinstance(session_date, type(None)):
        sd = session_date
    else:
        try:
            sd = pd.to_datetime(session_date).date()
        except (ValueError, TypeError):
            return None

    candidates = []
    for p in _CAP_SEGMENTS_DIR.glob("cap_segments_*.json"):
        name = p.stem  # cap_segments_YYYY-MM-DD or cap_segments_latest
        if name == "cap_segments_latest":
            continue
        try:
            snap_date = pd.to_datetime(name.rsplit("_", 1)[-1]).date()
        except (ValueError, TypeError):
            continue
        if snap_date <= sd:
            candidates.append((snap_date, p))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _build_cap_segment_cache(session_date) -> Dict[str, str]:
    """Build the {NSE:symbol -> segment} cache for the given session_date.

    Precedence:
      1. Dated cap_segments snapshot from data/cap_segments/ (preferred)
      2. cap_segments_latest.json (live/paper)
      3. Legacy nse_all.json (kept so backtests with no snapshot still work)

    Returns the bare-symbol-keyed map for backwards compat with the original
    cache shape (`_normalize_symbol(item["symbol"])` keys). Snapshots use
    'NSE:XXX' keys so we strip the prefix here.
    """
    snap_path = _pick_snapshot_path(session_date)
    if snap_path is not None:
        snap = _load_cap_segments_json(snap_path)
        if snap:
            cache = {_normalize_symbol(k): v for k, v in snap.items()}
            logger.debug(
                "CAP_SEGMENT: Loaded %d symbols from %s (session_date=%s)",
                len(cache), snap_path.name, session_date,
            )
            return cache
        logger.warning(
            "CAP_SEGMENT: snapshot %s empty/invalid — falling back to nse_all.json",
            snap_path.name,
        )

    data = _load_nse_all()
    if data is None:
        return {}
    cache = {
        _normalize_symbol(item["symbol"]): item.get("cap_segment", "unknown")
        for item in data
    }
    logger.debug("CAP_SEGMENT: Loaded %d symbols from nse_all.json (legacy fallback)", len(cache))
    return cache


def get_cap_segment(symbol: str, session_date=None) -> str:
    """Return market-cap bucket for `symbol`: large_cap / mid_cap / small_cap /
    micro_cap / unknown. Used for cap-aware sizing (Van Tharp evidence):
    large=1.2x, mid=1.0x, small=0.6x + wider stops. Unknown -> caller treats
    as mid-cap.

    Args:
        symbol: 'NSE:XXX' or bare 'XXX' (normalized internally)
        session_date: when provided (backtest), looks up the cap_segments
            snapshot in effect on that date for point-in-time fidelity.
            When None (live/paper), uses cap_segments_latest.json.

    Resolution order (per session_date cohort):
        1. data/cap_segments/cap_segments_<YYYY-MM-DD>.json (dated snapshot)
        2. data/cap_segments/cap_segments_latest.json
        3. nse_all.json (legacy fallback)
    """
    key = None
    if session_date is not None:
        # Normalize to ISO-day string so multi-date backtests reuse the cache
        try:
            if isinstance(session_date, pd.Timestamp):
                key = session_date.date().isoformat()
            elif isinstance(session_date, _date):
                key = session_date.isoformat()
            else:
                key = pd.to_datetime(session_date).date().isoformat()
        except (ValueError, TypeError):
            key = None

    cache = _cap_segment_caches.get(key)
    if cache is None:
        cache = _build_cap_segment_cache(session_date)
        _cap_segment_caches[key] = cache
    return cache.get(_normalize_symbol(symbol), "unknown")


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
