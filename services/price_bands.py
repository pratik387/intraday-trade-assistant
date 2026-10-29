"""NSE price bands (circuit limits) for a session, from the daily sec_list files.

The parquet is built by tools/fetch_nse_price_bands.py. A file dated D holds the
bands in force on the trading day after D, so session S reads the latest file
dated strictly before S. Live and backtest therefore read the same thing: the
file is published the evening before the session.

    circuit_distance_frac(symbol, session_date, ref_close, entry, bias)
        -> fraction from entry to the circuit that hurts the position
           (upper circuit for a short, lower for a long), or None when NSE lists
           the symbol with "No Band" (F&O names: no fixed circuit).

A session without the previous trading day's file raises
PriceBandDataError: missing data is never read as "no circuit". An entry outside
the band around the reference close raises InconsistentReferenceClose (the close
is not the band's base: unadjusted ex-date, bad print).
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


class PriceBandDataError(RuntimeError):
    pass


class SymbolNotInBandList(PriceBandDataError):
    pass


class InconsistentReferenceClose(PriceBandDataError):
    """Entry sits outside the band around the reference close: the close is not
    the band's base (ex-date adjustment, bad print), so the distance is unknown."""


@dataclass(frozen=True)
class SessionBands:
    file_date: date
    bands: Dict[str, float]          # symbol -> band pct (NaN = "No Band")


def load_nse_holidays(path: str) -> frozenset:
    """NSE trading holidays from assets/nse_holidays.json (tradingDate 'DD-Mon-YYYY').
    Raises on a missing or empty calendar: the previous-trading-day check depends on it."""
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    rows = json.load(open(p, encoding="utf-8"))
    out = frozenset(datetime.strptime(r["tradingDate"], "%d-%b-%Y").date() for r in rows)
    if not out:
        raise PriceBandDataError(f"NSE holiday calendar {p} is empty")
    return out


class PriceBandStore:
    def __init__(self, parquet_path: str, holidays: frozenset,
                 series_preference: Sequence[str], consistency_tolerance_pct: float) -> None:
        self.path = Path(parquet_path)
        if not self.path.is_absolute():
            self.path = _ROOT / self.path
        self.holidays = frozenset(holidays)
        self.series_rank = {s: i for i, s in enumerate(series_preference)}
        self.tol = float(consistency_tolerance_pct) / 100.0
        self._df: Optional[pd.DataFrame] = None
        self._mtime: Optional[float] = None
        self._sessions: Dict[date, SessionBands] = {}
        self._lock = threading.Lock()

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise PriceBandDataError(
                f"price-band parquet missing: {self.path} "
                f"(run tools/fetch_nse_price_bands.py)")
        mtime = self.path.stat().st_mtime
        if self._df is None or mtime != self._mtime:
            # Re-read when the morning refresh rewrote it: a daemon that outlives
            # its start day must see the next day's file.
            df = pd.read_parquet(self.path)
            df["file_date"] = pd.to_datetime(df["file_date"]).dt.date
            self._df, self._mtime = df, mtime
            self._sessions.clear()
        return self._df

    def previous_trading_day(self, d: date) -> date:
        cur = d - timedelta(days=1)
        while cur.weekday() >= 5 or cur in self.holidays:
            cur -= timedelta(days=1)
        return cur

    def session(self, session_date: date) -> SessionBands:
        with self._lock:
            df = self._load()
            got = self._sessions.get(session_date)
            if got is not None:
                return got
            days = sorted(d for d in set(df["file_date"]) if d < session_date)
            if not days:
                raise PriceBandDataError(f"no price-band file before {session_date}")
            fd = days[-1]
            need = self.previous_trading_day(session_date)
            if fd < need:
                # The bands for S are in the file of the trading day before S. An older
                # file holds the wrong day's bands (a band change would be missed).
                # >= rather than == lets a special session (e.g. a Saturday) through.
                raise PriceBandDataError(
                    f"session {session_date} needs the sec_list of {need} (previous trading "
                    f"day); newest available is {fd} (run tools/fetch_nse_price_bands.py)")
            day = df[df["file_date"] == fd].copy()
            day["_rank"] = day["series"].map(self.series_rank).fillna(len(self.series_rank))
            day = day.sort_values(["symbol", "_rank"]).drop_duplicates("symbol")
            got = SessionBands(fd, dict(zip(day["symbol"], day["band_pct"].astype(float))))
            self._sessions[session_date] = got
            return got

    def band_pct(self, symbol: str, session_date: date) -> Optional[float]:
        """Band in percent, or None for "No Band". Raises if the symbol is absent."""
        sb = self.session(session_date)
        sym = symbol.split(":", 1)[-1]
        if sym not in sb.bands:
            raise SymbolNotInBandList(f"{sym} not in price-band list {sb.file_date}")
        b = sb.bands[sym]
        return None if pd.isna(b) else float(b)

    def circuit_price(self, symbol: str, session_date: date, ref_close: float,
                      bias: str) -> Optional[float]:
        """The circuit that hurts the position: upper for a short, lower for a long.
        None for "No Band"."""
        band = self.band_pct(symbol, session_date)
        if band is None:
            return None
        ref = float(ref_close)
        return ref * (1 + band / 100.0) if bias == "short" else ref * (1 - band / 100.0)

    def circuit_distance_frac(self, symbol: str, session_date: date, ref_close: float,
                              entry: float, bias: str) -> Optional[float]:
        band = self.band_pct(symbol, session_date)
        if band is None:
            return None
        def ok(x):
            try:
                return x is not None and math.isfinite(float(x)) and float(x) > 0
            except (TypeError, ValueError):
                return False
        if not ok(ref_close) or not ok(entry):
            raise InconsistentReferenceClose(f"{symbol}: bad ref_close={ref_close} / entry={entry}")
        ref_close, entry = float(ref_close), float(entry)
        uc, lc = ref_close * (1 + band / 100.0), ref_close * (1 - band / 100.0)
        if entry > uc * (1 + self.tol) or entry < lc * (1 - self.tol):
            raise InconsistentReferenceClose(
                f"{symbol}: entry {entry} outside band {band}% of ref close {ref_close}")
        if bias == "short":
            return max(0.0, uc / entry - 1.0)
        if bias == "long":
            return max(0.0, 1.0 - lc / entry)
        raise ValueError(f"bias must be 'long' or 'short', got {bias!r}")


_store: Optional[PriceBandStore] = None
_store_lock = threading.Lock()


def get_price_band_store(cfg: dict) -> PriceBandStore:
    """Process-wide store built from the root config's price_bands block."""
    global _store
    with _store_lock:
        if _store is None:
            pb = cfg["price_bands"]
            _store = PriceBandStore(pb["parquet_path"], load_nse_holidays(pb["holidays_path"]),
                                    pb["series_preference"], pb["consistency_tolerance_pct"])
        return _store


def reset_price_band_store() -> None:
    """Tests only."""
    global _store
    with _store_lock:
        _store = None
