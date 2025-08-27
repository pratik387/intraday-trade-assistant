from __future__ import annotations
"""
BarBuilder — aggregates ticks → 1m bars → 5m bars (IST-naive).

Public API:
  - on_tick(symbol: str, price: float, volume: float, ts: datetime) -> None
  - get_df_1m_tail(symbol: str, n: int) -> pd.DataFrame
  - get_df_5m_tail(symbol: str, n: int) -> pd.DataFrame
  - last_ltp(symbol: str) -> float | None
  - index_df_5m(symbol: str | None = None) -> pd.DataFrame | dict[str, pd.DataFrame]

Constructor (no hidden config):
  - bar_5m_span_minutes: int (must be a multiple of 5)
  - on_1m_close: callable(symbol, bar_1m: pd.Series)
  - on_5m_close: callable(symbol, bar_5m: pd.Series)
  - index_symbols: Optional[list[str]] — symbols treated as index for convenience access

Notes:
  - Bars are time-indexed at their *close* times.
  - VWAP is per-bar; 5m VWAP is volume-weighted from the 1m bars.
  - Thread-safe: single RLock protects state. Callbacks are wrapped in try/except.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

Bar = pd.Series  # readability alias


@dataclass
class _LastTick:
    price: float
    volume: float
    ts: datetime


class BarBuilder:
    def __init__(
        self,
        *,
        bar_5m_span_minutes: int,
        on_1m_close: Callable[[str, Bar], None],
        on_5m_close: Callable[[str, Bar], None],
        index_symbols: Optional[List[str]] = None,
    ) -> None:
        if bar_5m_span_minutes % 5 != 0:
            raise ValueError("bar_5m_span_minutes must be a multiple of 5")
        self._span5 = int(bar_5m_span_minutes)
        self._on_1m_close = on_1m_close
        self._on_5m_close = on_5m_close

        self._lock = threading.RLock()
        self._ltp: Dict[str, _LastTick] = {}
        self._cur_1m: Dict[str, Bar] = {}
        self._bars_1m: Dict[str, pd.DataFrame] = defaultdict(_empty_df)
        self._bars_5m: Dict[str, pd.DataFrame] = defaultdict(_empty_df)
        self._index_symbols = set(index_symbols or [])

    # ----------------------------- Public API -----------------------------
    def on_tick(self, symbol: str, price: float, volume: float, ts: datetime) -> None:
        """Push a realtime tick for aggregation. Thread-safe."""
        with self._lock:
            self._ltp[symbol] = _LastTick(price=float(price), volume=float(volume), ts=ts)
            self._update_1m(symbol, float(price), float(volume), ts)

    def get_df_1m_tail(self, symbol: str, n: int) -> pd.DataFrame:
        with self._lock:
            df = self._bars_1m.get(symbol)
            if df is None or df.empty:
                return _empty_df()
            return df.tail(int(n)).copy()

    def get_df_5m_tail(self, symbol: str, n: int) -> pd.DataFrame:
        with self._lock:
            df = self._bars_5m.get(symbol)
            if df is None or df.empty:
                return _empty_df()
            return df.tail(int(n)).copy()

    def last_ltp(self, symbol: str) -> Optional[float]:
        with self._lock:
            t = self._ltp.get(symbol)
            return float(t.price) if t else None

    def index_df_5m(self, symbol: Optional[str] = None):
        """
        If `symbol` provided and tracked as an index symbol, return its 5m DF.
        If `symbol` is None:
          - If a single index symbol is configured, return its 5m DF;
          - If multiple, return a dict[symbol -> DF].
        """
        with self._lock:
            if symbol is not None:
                return self._bars_5m.get(symbol, _empty_df()).copy()
            idx_syms = list(self._index_symbols)
            if not idx_syms:
                return _empty_df()
            if len(idx_syms) == 1:
                return self._bars_5m.get(idx_syms[0], _empty_df()).copy()
            return {s: self._bars_5m.get(s, _empty_df()).copy() for s in idx_syms}

    # ---------------------------- Internals ------------------------------
    def _minute_bucket_end(self, ts: datetime) -> datetime:
        # Round up to the minute close (IST-naive). Example: 09:15:07 → 09:16:00
        return ts.replace(second=0, microsecond=0) + timedelta(minutes=1)

    def _update_1m(self, symbol: str, price: float, volume: float, ts: datetime) -> None:
        bucket_end = self._minute_bucket_end(ts)
        cur = self._cur_1m.get(symbol)
        if cur is None or cur.name != bucket_end:
            # Close previous bar if exists
            if cur is not None:
                self._close_1m(symbol, cur)
            # Start new bar
            ser = pd.Series(
                {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": float(volume),
                    "vwap_num": price * float(volume),  # temp accumulator
                    "vwap_den": float(volume),
                },
                name=bucket_end,
            )
            self._cur_1m[symbol] = ser
        else:
            # Update current 1m bar
            cur["high"] = max(cur["high"], price)
            cur["low"] = min(cur["low"], price)
            cur["close"] = price
            cur["volume"] += float(volume)
            cur["vwap_num"] += price * float(volume)
            cur["vwap_den"] += float(volume)

    def _close_1m(self, symbol: str, ser: Bar) -> None:
        # Finalize VWAP for the minute
        den = float(ser.get("vwap_den", 0.0) or 0.0)
        vwap = float((ser.get("vwap_num", 0.0) / den)) if den > 0 else float(ser["close"])
        ser = ser.drop(labels=["vwap_num", "vwap_den"], errors="ignore")
        ser["vwap"] = vwap

        df = self._bars_1m[symbol]
        self._bars_1m[symbol] = pd.concat([df, ser.to_frame().T])

        # Callback (non-fatal)
        try:
            self._on_1m_close(symbol, ser)
        except Exception:
            pass

        # Try rolling into 5m
        self._attempt_close_5m(symbol, ser.name)

    def _attempt_close_5m(self, symbol: str, minute_close_ts: datetime) -> None:
        # Close a 5m bar when minute_close_ts.minute % span == 0
        span = self._span5
        if (minute_close_ts.minute % span) != 0:
            return

        end_ts = minute_close_ts
        start_ts = end_ts - timedelta(minutes=span)

        df1 = self._bars_1m.get(symbol)
        if df1 is None or df1.empty:
            return
        window = df1[(df1.index > start_ts) & (df1.index <= end_ts)]
        if window.empty:
            return

        bar5 = _aggregate_window_to_ohlcv(window)
        bar5.name = end_ts

        df5 = self._bars_5m[symbol]
        self._bars_5m[symbol] = pd.concat([df5, bar5.to_frame().T])

        # 5m close callback
        try:
            self._on_5m_close(symbol, bar5)
        except Exception:
            pass


# ------------------------------ Helpers --------------------------------

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])  # type: ignore


def _aggregate_window_to_ohlcv(window: pd.DataFrame) -> pd.Series:
    o = float(window.iloc[0]["open"]) if len(window) else np.nan
    h = float(window["high"].max()) if len(window) else np.nan
    l = float(window["low"].min()) if len(window) else np.nan
    c = float(window.iloc[-1]["close"]) if len(window) else np.nan
    v = float(window["volume"].sum()) if len(window) else 0.0

    # 5m VWAP from 1m vwap * volume (more stable than simple average)
    if "vwap" in window.columns:
        vwap_num = float((window["vwap"] * window["volume"]).sum())
        vwap_den = float(window["volume"].sum())
        vwap = vwap_num / vwap_den if vwap_den > 0 else c
    else:
        vwap = c

    # BB width proxy: std(close, N=20) / vwap (dimensionless)
    closes = window["close"].tail(20)
    bb_width_proxy = float(closes.std(ddof=0) / vwap) if len(closes) >= 5 and vwap else 0.0

    return pd.Series(
        {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "vwap": float(vwap),
            "bb_width_proxy": bb_width_proxy,
        }
    )
