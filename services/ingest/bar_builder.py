from __future__ import annotations
"""
BarBuilder — aggregates ticks → 1m bars → 5m bars (IST-naive).

Public API:
  - on_tick(symbol: str, price: float, volume: float, ts: datetime) -> None
  - get_df_1m_tail(symbol: str, n: int) -> pd.DataFrame
  - get_df_5m_tail(symbol: str, n: int) -> pd.DataFrame
  - get_df_15m_tail(symbol: str, n: int) -> pd.DataFrame
  - last_ltp(symbol: str) -> float | None
  - index_df_5m(symbol: str | None = None) -> pd.DataFrame | dict[str, pd.DataFrame]

Constructor (no hidden config):
  - bar_5m_span_minutes: int (must be a multiple of 5)
  - on_1m_close: callable(symbol, bar_1m: pd.Series)
  - on_5m_close: callable(symbol, bar_5m: pd.Series)
  - on_15m_close: callable(symbol, bar_15m: pd.Series)
  - index_symbols: Optional[list[str]] — symbols treated as index for convenience access

Notes:
  - Bars are time-indexed at their *start* times (5m bar closing at 10:35 has index 10:30).
  - VWAP is per-bar; 5m VWAP is volume-weighted from the 1m bars.
  - Thread-safe: single RLock protects state. Callbacks are wrapped in try/except.
"""

import math
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from config.logging_config import get_agent_logger
from services.indicators.indicators import RSIState, update_rsi_incremental

logger = get_agent_logger()

Bar = pd.Series  # readability alias


@dataclass
class _LastTick:
    price: float
    volume: float
    ts: datetime


# ------------------------------- Minimal ADX state ------------------------------

@dataclass
class _ADXState:
    prev_high: float = math.nan
    prev_low: float = math.nan
    prev_close: float = math.nan
    tr_s: float = 0.0         # Wilder-smoothed True Range
    plus_s: float = 0.0       # Wilder-smoothed +DM
    minus_s: float = 0.0      # Wilder-smoothed -DM
    adx: float = 0.0          # Wilder-smoothed DX (the ADX)


# --------------------------------- Builder ------------------------------------

def _empty_df() -> pd.DataFrame:
    # Initial columns; new fields (bb_width_proxy, adx) will be added on first append
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


class BarBuilder:
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

        # ADX state (minimal, incremental)
        self._adx_state: Dict[str, _ADXState] = {}
        self._adx_alpha: float = 1.0 / 14.0  # Wilder α for ADX(14)

        # RSI state (using indicators.RSIState for incremental calculation)
        self._rsi_state: Dict[str, RSIState] = {}

        # Additional handlers for trigger system
        self._additional_1m_handlers: List[Callable[[str, Bar], None]] = []
        self._additional_5m_handlers: List[Callable[[str, Bar], None]] = []
        self._additional_15m_handlers: List[Callable[[str, Bar], None]] = []

        # Flag to clear pre-market data once at 9:15
        self._market_open_cleared = False

    def _clear_pre_market(self) -> None:
        """Clear all pre-market bar data at 9:15. Called once per session."""
        self._cur_1m.clear()
        self._bars_1m.clear()
        self._bars_5m.clear()
        self._bars_15m.clear()
        self._adx_state.clear()
        self._rsi_state.clear()
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
            self._update_1m(symbol, float(price), float(volume), ts)

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
        # Close a 5m bar when minute_close_ts.minute % span == 0
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
        # CRITICAL FIX: Use START-STAMPED convention (Zerodha/broker standard)
        # START-STAMPED: 09:15 bar contains data from [09:15, 09:20) labeled at START (09:15)
        # 09:20 bar contains [09:20, 09:25) labeled at START (09:20)
        # Excludes end_ts minute (goes into next bar)
        window = df1[(df1.index >= start_ts) & (df1.index < end_ts)]
        if window.empty:
            return

        bar5 = _aggregate_window_to_ohlcv(window)
        bar5.name = start_ts  # ← START-LABELED (matches Zerodha/Upstox convention)

        # --- Incremental ADX(14) update (O(1)) ---
        try:
            adx_val = self._update_adx_5m(symbol, float(bar5["high"]), float(bar5["low"]), float(bar5["close"]))
        except Exception:
            adx_val = 0.0
        bar5["adx"] = float(adx_val)

        # --- Incremental RSI(14) update (O(1)) ---
        try:
            rsi_val = self._update_rsi_5m(symbol, float(bar5["close"]))
        except Exception:
            rsi_val = 50.0  # Neutral RSI on error
        bar5["rsi"] = float(rsi_val)

        df5 = self._bars_5m[symbol]
        row5 = bar5.to_frame().T
        row5.index = [start_ts]  # ← START-LABELED timestamp
        if df5 is None or getattr(df5, "empty", True):
            self._bars_5m[symbol] = row5
        else:
            self._bars_5m[symbol] = pd.concat([df5, row5], copy=False)

        # 5m close callback
        try:
            self._on_5m_close(symbol, bar5)
        except Exception as e:
            logger.exception("BarBuilder: on_5m_close callback failed: %s", e)
            pass
        
        for handler in self._additional_5m_handlers:
            try:
                handler(symbol, bar5)
            except Exception as e:
                logger.exception("BarBuilder: additional 5m handler failed: %s", e)

    def _attempt_close_15m(self, symbol: str, minute_close_ts: datetime) -> None:
        """Close a 15m bar when minute_close_ts.minute % 15 == 0"""
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

        # 15m bars inherit ADX from the last 5m bar in the window
        try:
            bar15["adx"] = float(window.iloc[-1]["adx"]) if "adx" in window.columns else 0.0
        except Exception:
            bar15["adx"] = 0.0

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

    # ------------------------ Incremental ADX(14) -------------------------
    def _update_adx_5m(self, symbol: str, high: float, low: float, close: float) -> float:
        """
        O(1) Wilder-style ADX update for the new 5m bar. Returns the latest ADX.
        """
        st = self._adx_state.get(symbol)
        if st is None:
            st = _ADXState()
            self._adx_state[symbol] = st

        # First bar warmup: initialize smoothed values
        if not math.isfinite(st.prev_close):
            st.prev_high = high
            st.prev_low = low
            st.prev_close = close
            tr = max(high - low, abs(high - close), abs(low - close))
            st.tr_s = tr
            st.plus_s = 0.0
            st.minus_s = 0.0
            st.adx = 0.0
            return 0.0

        up_move = high - st.prev_high
        dn_move = st.prev_low - low
        plus_dm = up_move if (up_move > dn_move and up_move > 0.0) else 0.0
        minus_dm = dn_move if (dn_move > up_move and dn_move > 0.0) else 0.0

        tr1 = high - low
        tr2 = abs(high - st.prev_close)
        tr3 = abs(st.prev_close - low)
        tr = max(tr1, tr2, tr3)

        a = self._adx_alpha
        # Wilder EMA (RMA) updates
        st.tr_s    = st.tr_s    + a * (tr      - st.tr_s)
        st.plus_s  = st.plus_s  + a * (plus_dm - st.plus_s)
        st.minus_s = st.minus_s + a * (minus_dm - st.minus_s)

        if st.tr_s <= 1e-12:
            cur_adx = st.adx
        else:
            plus_di  = 100.0 * (st.plus_s  / st.tr_s)
            minus_di = 100.0 * (st.minus_s / st.tr_s)
            denom = plus_di + minus_di
            dx = 0.0 if denom <= 1e-12 else 100.0 * abs(plus_di - minus_di) / denom
            st.adx = st.adx + a * (dx - st.adx)
            cur_adx = st.adx

        st.prev_high = high
        st.prev_low = low
        st.prev_close = close
        return float(cur_adx)

    def _update_rsi_5m(self, symbol: str, close: float) -> float:
        """
        O(1) Wilder-style RSI update for the new 5m bar. Returns the latest RSI.
        Uses indicators.update_rsi_incremental for the calculation.
        """
        st = self._rsi_state.get(symbol)
        if st is None:
            st = RSIState()
            self._rsi_state[symbol] = st

        return update_rsi_incremental(st, close, period=14, warmup=14)

    def get_df_1m_tail(self, symbol: str, n: int) -> pd.DataFrame:
        """Get last n 1-minute bars for a symbol"""
        with self._lock:
            df = self._bars_1m.get(symbol)
            if df is None or df.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
            return df.tail(int(n)).copy()

    def get_df_5m_tail(self, symbol: str, n: int) -> pd.DataFrame:
        """Get last n 5-minute bars for a symbol"""
        with self._lock:
            df = self._bars_5m.get(symbol)
            if df is None or df.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "bb_width_proxy", "adx"])
            return df.tail(int(n)).copy()

    def get_current_vwap(self, symbol: str) -> float:
        """Get current VWAP for a symbol"""
        with self._lock:
            df = self._bars_1m.get(symbol)
            if df is None or df.empty:
                return 0.0
            return float(df.iloc[-1].get("vwap", 0.0))
        
    def register_1m_handler(self, handler: Callable[[str, Bar], None]) -> None:
        """Register additional 1m bar handler (for trigger system)"""
        if handler not in self._additional_1m_handlers:
            self._additional_1m_handlers.append(handler)

    def register_5m_handler(self, handler: Callable[[str, Bar], None]) -> None:
        """Register additional 5m bar handler"""
        if handler not in self._additional_5m_handlers:
            self._additional_5m_handlers.append(handler)
