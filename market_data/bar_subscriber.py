"""
BarSubscriber - Receives bars from Redis pub/sub for PAPER mode.

This module allows PAPER instances to receive bars from LIVE instances,
ensuring identical market data for deterministic signal generation.

Usage:
    # PAPER mode setup
    subscriber = BarSubscriber(redis_url="redis://localhost:6379/0")

    # Register callbacks (same signature as BarBuilder callbacks)
    subscriber.on_5m_bar(process_5m_bar)

    # Start receiving bars
    subscriber.start()

The subscriber mimics BarBuilder's callback interface, so the rest of
the trading system doesn't need to know whether bars come from direct
tick aggregation (LIVE) or Redis subscription (PAPER).
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable, Optional, List, Dict

import pandas as pd

from config.logging_config import get_agent_logger
from .market_data_bus import MarketDataBus, BarEvent

logger = get_agent_logger()


class BarSubscriber:
    """
    Subscribes to bar events from Redis and invokes callbacks.

    This is the PAPER mode counterpart to BarBuilder - instead of
    aggregating ticks into bars, it receives pre-built bars from
    the LIVE instance via Redis pub/sub.

    Interface matches BarBuilder for seamless integration:
      - on_5m_close(symbol, bar) callbacks
      - get_df_5m_tail(symbol, n) for historical bars
      - last_ltp(symbol) via SharedLTPCache
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        symbols: Optional[List[str]] = None,
    ):
        """
        Initialize BarSubscriber.

        Args:
            redis_url: Redis connection URL
            symbols: Optional list of symbols to subscribe to (None = all)
        """
        self._redis_url = redis_url
        self._symbols = symbols

        # Callbacks (same signature as BarBuilder)
        self._on_1m_close: Optional[Callable] = None
        self._on_5m_close: Optional[Callable] = None
        self._on_15m_close: Optional[Callable] = None

        # Additional handlers (same as BarBuilder)
        self._additional_1m_handlers: List[Callable] = []
        self._additional_5m_handlers: List[Callable] = []
        self._additional_15m_handlers: List[Callable] = []

        # Bar storage (mirrors BarBuilder's internal structure)
        self._bars_1m: Dict[str, pd.DataFrame] = {}
        self._bars_5m: Dict[str, pd.DataFrame] = {}
        self._bars_15m: Dict[str, pd.DataFrame] = {}
        self._lock = threading.RLock()

        # Market data bus for subscription
        self._bus: Optional[MarketDataBus] = None
        self._started = False

    def set_callbacks(
        self,
        on_1m_close: Optional[Callable] = None,
        on_5m_close: Optional[Callable] = None,
        on_15m_close: Optional[Callable] = None,
    ) -> None:
        """Set bar close callbacks (same interface as BarBuilder constructor)."""
        self._on_1m_close = on_1m_close
        self._on_5m_close = on_5m_close
        self._on_15m_close = on_15m_close

    def register_1m_handler(self, handler: Callable) -> None:
        """Register additional 1m bar handler."""
        if handler not in self._additional_1m_handlers:
            self._additional_1m_handlers.append(handler)

    def register_5m_handler(self, handler: Callable) -> None:
        """Register additional 5m bar handler."""
        if handler not in self._additional_5m_handlers:
            self._additional_5m_handlers.append(handler)

    def register_15m_handler(self, handler: Callable) -> None:
        """Register additional 15m bar handler."""
        if handler not in self._additional_15m_handlers:
            self._additional_15m_handlers.append(handler)

    def start(self) -> None:
        """Start receiving bars from Redis."""
        if self._started:
            return

        self._bus = MarketDataBus(
            mode="subscriber",
            redis_url=self._redis_url,
        )

        # Subscribe to all timeframes
        self._bus.subscribe_bars("1m", self._on_bar_event, self._symbols)
        self._bus.subscribe_bars("5m", self._on_bar_event, self._symbols)
        self._bus.subscribe_bars("15m", self._on_bar_event, self._symbols)

        self._started = True
        logger.info(f"BAR_SUBSCRIBER | Started, symbols={self._symbols or 'all'}")

    def _on_bar_event(self, event: BarEvent) -> None:
        """Handle incoming bar event from Redis."""
        # Convert BarEvent to pandas Series (matches BarBuilder output)
        bar = pd.Series({
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
            "vwap": event.vwap,
            "adx": event.adx,
            "rsi": event.rsi,
            "bb_width_proxy": event.bb_width_proxy,
        })

        # Parse timestamp
        try:
            bar.name = pd.Timestamp(event.ts)
        except Exception:
            bar.name = datetime.fromisoformat(event.ts)

        symbol = event.symbol
        timeframe = event.timeframe

        # Store bar in local cache
        self._store_bar(symbol, timeframe, bar)

        # Invoke callbacks
        if timeframe == "1m":
            self._invoke_callbacks(symbol, bar, self._on_1m_close, self._additional_1m_handlers)
        elif timeframe == "5m":
            self._invoke_callbacks(symbol, bar, self._on_5m_close, self._additional_5m_handlers)
        elif timeframe == "15m":
            self._invoke_callbacks(symbol, bar, self._on_15m_close, self._additional_15m_handlers)

    def _store_bar(self, symbol: str, timeframe: str, bar: pd.Series) -> None:
        """Store bar in local cache for get_df_*_tail methods."""
        with self._lock:
            storage = {
                "1m": self._bars_1m,
                "5m": self._bars_5m,
                "15m": self._bars_15m,
            }.get(timeframe)

            if storage is None:
                return

            row_df = bar.to_frame().T
            row_df.index = [bar.name]

            if symbol not in storage or storage[symbol].empty:
                storage[symbol] = row_df
            else:
                storage[symbol] = pd.concat([storage[symbol], row_df], copy=False)

                # Keep only last 500 bars to prevent memory growth
                if len(storage[symbol]) > 500:
                    storage[symbol] = storage[symbol].iloc[-500:]

    def _invoke_callbacks(
        self,
        symbol: str,
        bar: pd.Series,
        main_callback: Optional[Callable],
        additional_handlers: List[Callable],
    ) -> None:
        """Invoke bar close callbacks (same error handling as BarBuilder)."""
        if main_callback:
            try:
                main_callback(symbol, bar)
            except Exception as e:
                logger.exception(f"BAR_SUBSCRIBER | Callback failed: {e}")

        for handler in additional_handlers:
            try:
                handler(symbol, bar)
            except Exception as e:
                logger.exception(f"BAR_SUBSCRIBER | Additional handler failed: {e}")

    # ============ BarBuilder-compatible API ============

    def get_df_1m_tail(self, symbol: str, n: int) -> pd.DataFrame:
        """Get last n 1-minute bars (matches BarBuilder interface)."""
        with self._lock:
            df = self._bars_1m.get(symbol)
            if df is None or df.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
            return df.tail(int(n)).copy()

    def get_df_5m_tail(self, symbol: str, n: int) -> pd.DataFrame:
        """Get last n 5-minute bars (matches BarBuilder interface)."""
        with self._lock:
            df = self._bars_5m.get(symbol)
            if df is None or df.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "bb_width_proxy", "adx", "rsi"])
            return df.tail(int(n)).copy()

    def get_df_15m_tail(self, symbol: str, n: int) -> pd.DataFrame:
        """Get last n 15-minute bars (matches BarBuilder interface)."""
        with self._lock:
            df = self._bars_15m.get(symbol)
            if df is None or df.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "bb_width_proxy", "adx"])
            return df.tail(int(n)).copy()

    def last_ltp(self, symbol: str) -> Optional[float]:
        """Get last close price (approximation for LTP in subscriber mode)."""
        with self._lock:
            df = self._bars_1m.get(symbol)
            if df is not None and not df.empty:
                return float(df.iloc[-1]["close"])

            df = self._bars_5m.get(symbol)
            if df is not None and not df.empty:
                return float(df.iloc[-1]["close"])

            return None

    def index_df_5m(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get index symbol 5m bars (BarBuilder compatibility).

        Note: BarSubscriber doesn't track index symbols separately.
        Returns empty DataFrame as ScreenerLive._index_symbols() returns [].
        """
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._started = False
        if self._bus:
            self._bus.shutdown()
            self._bus = None
        logger.info("BAR_SUBSCRIBER | Shutdown complete")
