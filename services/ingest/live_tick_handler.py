from __future__ import annotations
"""
LiveTickHandler (formerly BarBuilder) — aggregates ticks → 1m → 5m OHLCV bars.

In the new architecture, this component ONLY handles:
  - Tick aggregation into 1m/5m/15m OHLCV bars (no indicators)
  - Real-time LTP for execution layer
  - Bar close callbacks for scan timing triggers

Indicators (VWAP, bb_width, ADX, RSI) are computed by:
  - Offline: tools/precompute_5m_cache.py → feather cache (backtest reads this)
  - Runtime: services/indicators/bar_enrichment.enrich_5m_bars() (paper/live)

Public API:
  - on_tick(symbol, price, volume, ts)
  - on_i1_candle(symbol, open, high, low, close, volume, ts_str)
  - get_df_1m_tail(symbol, n) / get_df_5m_tail(symbol, n) / get_df_15m_tail(symbol, n)
  - last_ltp(symbol) -> float | None
"""

import time as _time_mod
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from config.logging_config import get_agent_logger
# NOTE: Indicator computation (ADX, RSI, VWAP, bb_width) removed from BarBuilder.
# Indicators are now precomputed offline (tools/precompute_5m_cache.py) or
# computed at runtime via enrich_5m_bars (services/indicators/bar_enrichment.py).
# BarBuilder only aggregates OHLCV.

logger = get_agent_logger()

Bar = pd.Series  # readability alias


@dataclass
class _LastTick:
    price: float
    volume: float
    ts: datetime


# --------------------------------- Builder ------------------------------------

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])  # type: ignore


def _aggregate_window_to_ohlcv(window: pd.DataFrame) -> pd.Series:
    """Aggregate 1m bars into a single OHLCV bar. No indicators."""
    o = float(window.iloc[0]["open"]) if len(window) else np.nan
    h = float(window["high"].max()) if len(window) else np.nan
    l = float(window["low"].min()) if len(window) else np.nan
    c = float(window.iloc[-1]["close"]) if len(window) else np.nan
    v = float(window["volume"].sum()) if len(window) else 0.0

    return pd.Series({"open": o, "high": h, "low": l, "close": c, "volume": v})


class LiveTickHandler:
    def __init__(
        self,
        *,
        bar_5m_span_minutes: int,
        on_1m_close: Callable[[str, Bar], None],
        on_5m_close: Callable[[str, Bar], None],
        on_15m_close: Optional[Callable[[str, Bar], None]] = None,
        index_symbols: Optional[List[str]] = None,
    ) -> None:
        if bar_5m_span_minutes % 5 != 0:
            raise ValueError("bar_5m_span_minutes must be a multiple of 5")
        self._span5 = int(bar_5m_span_minutes)

        self._on_1m_close = on_1m_close
        self._on_5m_close = on_5m_close
        self._on_15m_close = on_15m_close or (lambda s, b: None)  # no-op if not provided

        self._lock = threading.RLock()
        self._ltp: Dict[str, _LastTick] = {}
        self._cur_1m: Dict[str, Bar] = {}
        self._bars_1m: Dict[str, pd.DataFrame] = defaultdict(_empty_df)
        self._bars_5m: Dict[str, pd.DataFrame] = defaultdict(_empty_df)
        self._bars_15m: Dict[str, pd.DataFrame] = defaultdict(_empty_df)
        self._index_symbols = set(index_symbols or [])

        # Additional handlers for trigger system
        self._additional_1m_handlers: List[Callable[[str, Bar], None]] = []
        self._additional_5m_handlers: List[Callable[[str, Bar], None]] = []
        self._additional_15m_handlers: List[Callable[[str, Bar], None]] = []

        # Symbols receiving I1 (broker-constructed 1m) candles — skip tick-based bar building
        self._i1_symbols: set = set()

        # Bar density tracking: count closed 1m bars per symbol to detect illiquid symbols
        # where WebSocket doesn't deliver all candles (causes backtest-live divergence)
        self._bar_1m_count: Dict[str, int] = {}
        self._first_bar_ts: Dict[str, datetime] = {}

        # Flag to clear pre-market data once at 9:15
        self._market_open_cleared = False

    def _clear_pre_market(self) -> None:
        """Clear all pre-market bar data at 9:15. Called once per session."""
        self._cur_1m.clear()
        self._bars_1m.clear()
        self._bars_5m.clear()
        self._bars_15m.clear()
        self._bar_1m_count.clear()
        self._first_bar_ts.clear()
        # Keep _ltp - we want last traded price
        logger.info("BAR_BUILDER | Cleared pre-market data at market open")

    # ----------------------------- Public API -----------------------------
    def on_tick(self, symbol: str, price: float, volume: float, ts: datetime) -> None:
        """Push a realtime tick for aggregation. Thread-safe."""
        with self._lock:
            # Clear pre-market data on first tick at/after 9:15
            if not self._market_open_cleared and ts.hour == 9 and ts.minute >= 15:
                self._clear_pre_market()
                self._market_open_cleared = True

            self._ltp[symbol] = _LastTick(price=float(price), volume=float(volume), ts=ts)

            # Skip tick-based bar building for symbols receiving I1 candles
            if symbol not in self._i1_symbols:
                self._update_1m(symbol, float(price), float(volume), ts)

    def on_i1_candle(
        self, symbol: str, open_: float, high: float, low: float,
        close: float, volume: int, ts_str: str,
    ) -> None:
        """
        Accept a running I1 (1-minute) candle from broker WebSocket.

        I1 candles update with every tick. When the minute changes, the
        previous candle is complete and flows through the same _close_1m →
        _attempt_close_5m → _attempt_close_15m path as tick-built bars.

        Parameters match the Upstox ticker adapter callback signature.
        ts_str: epoch milliseconds as string (UTC).
        """
        if not ts_str or open_ == 0:
            return

        with self._lock:
            # Convert epoch ms UTC → IST-naive datetime (minute start)
            try:
                ts_epoch_ms = int(ts_str)
                ts_ist = datetime.utcfromtimestamp(ts_epoch_ms / 1000) + timedelta(hours=5, minutes=30)
            except (ValueError, OverflowError):
                return

            # BarBuilder convention: bar index = bucket_end = minute_start + 1min
            bucket_end = ts_ist.replace(second=0, microsecond=0) + timedelta(minutes=1)

            # Clear pre-market data on first candle at/after 9:15
            if not self._market_open_cleared and ts_ist.hour == 9 and ts_ist.minute >= 15:
                self._clear_pre_market()
                self._market_open_cleared = True

            # Mark symbol as I1-sourced (disables tick-based bar building)
            self._i1_symbols.add(symbol)

            # Update LTP from I1 candle
            self._ltp[symbol] = _LastTick(price=float(close), volume=float(volume), ts=ts_ist)

            cur = self._cur_1m.get(symbol)
            if cur is not None and cur.name != bucket_end:
                # Minute changed — close the previous completed bar
                self._close_1m(symbol, cur)

            # Replace current running bar with broker's OHLCV
            # VWAP approximation: HLC3 (typical price) — standard proxy without tick data
            ser = pd.Series(
                {
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume),
                },
                name=bucket_end,
            )
            self._cur_1m[symbol] = ser

    def get_df_15m_tail(self, symbol: str, n: int) -> pd.DataFrame:
        with self._lock:
            df = self._bars_15m.get(symbol)
            if df is None or df.empty:
                return _empty_df()
            return df.tail(int(n)).copy()

    def last_ltp(self, symbol: str) -> Optional[float]:
        with self._lock:
            t = self._ltp.get(symbol)
            return float(t.price) if t else None

    def get_bar_density(self, symbol: str, current_ts: datetime) -> float:
        """Return bar density (0.0-1.0) = closed_1m_bars / elapsed_minutes since first bar.

        Used to detect illiquid symbols where WebSocket doesn't deliver all 1m candles.
        Symbols with low density produce unreliable 5m bars (backtest-live divergence).
        """
        with self._lock:
            count = self._bar_1m_count.get(symbol, 0)
            first_ts = self._first_bar_ts.get(symbol)
            if count == 0 or first_ts is None:
                return 0.0
            elapsed = (current_ts - first_ts).total_seconds() / 60.0
            if elapsed <= 0:
                return 1.0
            return min(1.0, count / elapsed)

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

    def _close_1m(self, symbol: str, ser: Bar) -> None:
        # Track bar density (for illiquid symbol detection)
        self._bar_1m_count[symbol] = self._bar_1m_count.get(symbol, 0) + 1
        if symbol not in self._first_bar_ts:
            self._first_bar_ts[symbol] = ser.name

        df = self._bars_1m[symbol]
        # Ensure the appended 1-row DataFrame uses the minute timestamp as its index
        row_df = ser.to_frame().T
        row_df.index = [ser.name]
        if df is None or getattr(df, "empty", True):
            self._bars_1m[symbol] = row_df
        else:
            # Preserve DatetimeIndex; no ignore_index to avoid integer index creation
            self._bars_1m[symbol] = pd.concat([df, row_df], copy=False)

        # Callback (non-fatal)
        try:
            self._on_1m_close(symbol, ser)
        except Exception as e:
            logger.exception("BarBuilder: on_1m_close callback failed: %s", e)
            pass
        
        for handler in self._additional_1m_handlers:
            try:
                handler(symbol, ser)
            except Exception as e:
                logger.exception("BarBuilder: additional 1m handler failed: %s", e)

        # Try rolling into 5m and 15m
        self._attempt_close_5m(symbol, ser.name)
        self._attempt_close_15m(symbol, ser.name)

    def _attempt_close_5m(self, symbol: str, minute_close_ts: datetime) -> None:
        span = self._span5
        if (minute_close_ts.minute % span) != 0:
            return

        end_ts = minute_close_ts
        start_ts = end_ts - timedelta(minutes=span)

        df1 = self._bars_1m.get(symbol)
        if df1 is None or df1.empty:
            return
        # Coerce to DatetimeIndex defensively (in case of legacy rows)
        if not isinstance(df1.index, pd.DatetimeIndex):
            df1 = df1.copy()
            df1.index = pd.to_datetime(df1.index, errors="coerce")
            df1 = df1[~df1.index.isna()]
        # START-STAMPED convention: window includes [start_ts, end_ts)
        window = df1[(df1.index >= start_ts) & (df1.index < end_ts)]
        if window.empty:
            return

        bar5 = _aggregate_window_to_ohlcv(window)
        bar5.name = start_ts  # ← START-LABELED (matches Zerodha/Upstox convention)

        df5 = self._bars_5m[symbol]
        row5 = bar5.to_frame().T
        row5.index = [start_ts]  # ← START-LABELED timestamp
        if df5 is None or getattr(df5, "empty", True):
            self._bars_5m[symbol] = row5
        else:
            self._bars_5m[symbol] = pd.concat([df5, row5], copy=False)

        # 5m close callback (in backtest this blocks for the entire scan — log duration)
        _t_cb = _time_mod.perf_counter()
        try:
            self._on_5m_close(symbol, bar5)
        except Exception as e:
            logger.exception("BarBuilder: on_5m_close callback failed: %s", e)
        _t_cb_elapsed = _time_mod.perf_counter() - _t_cb
        if _t_cb_elapsed > 1.0:
            logger.info("BAR_BUILDER_5M_CALLBACK | %s at %s | %.2fs", symbol, start_ts, _t_cb_elapsed)
        
        for handler in self._additional_5m_handlers:
            try:
                handler(symbol, bar5)
            except Exception as e:
                logger.exception("BarBuilder: additional 5m handler failed: %s", e)

    def _attempt_close_15m(self, symbol: str, minute_close_ts: datetime) -> None:
        if (minute_close_ts.minute % 15) != 0:
            return

        end_ts = minute_close_ts
        start_ts = end_ts - timedelta(minutes=15)

        df5 = self._bars_5m.get(symbol)
        if df5 is None or df5.empty:
            return

        # Coerce to DatetimeIndex defensively
        if not isinstance(df5.index, pd.DatetimeIndex):
            df5 = df5.copy()
            df5.index = pd.to_datetime(df5.index, errors="coerce")
            df5 = df5[~df5.index.isna()]

        # Aggregate 3 x 5m bars into 1 x 15m bar (START-STAMPED convention)
        # With START-STAMPED 5m bars: 09:15, 09:20, 09:25 form the 09:15 15m bar
        window = df5[(df5.index >= start_ts) & (df5.index < end_ts)]
        if window.empty:
            return

        bar15 = _aggregate_window_to_ohlcv(window)
        bar15.name = start_ts  # ← START-LABELED (matches convention)

        df15 = self._bars_15m[symbol]
        row15 = bar15.to_frame().T
        row15.index = [start_ts]  # ← START-LABELED timestamp
        if df15 is None or getattr(df15, "empty", True):
            self._bars_15m[symbol] = row15
        else:
            self._bars_15m[symbol] = pd.concat([df15, row15], copy=False)

        # 15m close callback
        try:
            self._on_15m_close(symbol, bar15)
        except Exception as e:
            logger.exception("BarBuilder: on_15m_close callback failed: %s", e)
            pass

        for handler in self._additional_15m_handlers:
            try:
                handler(symbol, bar15)
            except Exception as e:
                logger.exception("BarBuilder: additional 15m handler failed: %s", e)

    def get_df_1m_tail(self, symbol: str, n: int) -> pd.DataFrame:
        """Get last n 1-minute bars for a symbol"""
        with self._lock:
            df = self._bars_1m.get(symbol)
            if df is None or df.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            return df.tail(int(n)).copy()

    def get_df_5m_tail(self, symbol: str, n: int) -> pd.DataFrame:
        """Get last n 5-minute bars for a symbol"""
        with self._lock:
            df = self._bars_5m.get(symbol)
            if df is None or df.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            return df.tail(int(n)).copy()

    def register_1m_handler(self, handler: Callable[[str, Bar], None]) -> None:
        """Register additional 1m bar handler (for trigger system)"""
        if handler not in self._additional_1m_handlers:
            self._additional_1m_handlers.append(handler)

    def register_5m_handler(self, handler: Callable[[str, Bar], None]) -> None:
        """Register additional 5m bar handler"""
        if handler not in self._additional_5m_handlers:
            self._additional_5m_handlers.append(handler)

    def register_15m_handler(self, handler: Callable[[str, Bar], None]) -> None:
        """Register additional 15m bar handler"""
        if handler not in self._additional_15m_handlers:
            self._additional_15m_handlers.append(handler)
