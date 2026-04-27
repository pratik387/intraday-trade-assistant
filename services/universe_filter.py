"""Per-setup universe membership lookup (sub8-T1).

Loads NSE constituent CSVs once at import time from `assets/ind_nifty*.csv`
and `assets/fno_liquid_200.csv`. Returns True/False membership tests cheap
enough to call per-symbol-per-bar in detectors.

Universe keys consumed by sub8 detector configs:
  - "nifty50"               — Nifty 50 only (CPR breakout context)
  - "banknifty"             — Bank Nifty only
  - "nifty50_banknifty"     — union of Nifty 50 + Bank Nifty (sub8 narrow_cpr_breakout)
  - "fno_liquid_200"        — F&O liquid ~120-200 names (ORB, VWAP, CHR)
  - "smallmid_fno"          — F&O liquid MINUS Nifty 50 (PDH/PDL fade)
"""
from __future__ import annotations

from pathlib import Path
from typing import Set

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
