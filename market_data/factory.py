"""
Market Data Factory - Creates appropriate market data components based on config.

This module provides factory functions to create the right market data
components based on the market_data_bus configuration:

  - LIVE mode (publisher): Uses BarBuilder + publishes to Redis
  - PAPER mode (subscriber): Uses BarSubscriber + reads from Redis
  - Standalone mode: Uses BarBuilder only (existing behavior)

Usage in main.py:
    from market_data.factory import create_market_data_components

    bar_builder, ltp_cache, market_data_bus = create_market_data_components(config)

    # bar_builder: BarBuilder or BarSubscriber (duck-typed interface)
    # ltp_cache: SharedLTPCache (mode-appropriate)
    # market_data_bus: MarketDataBus or None
"""

from __future__ import annotations

from typing import Optional, Tuple, Callable, Any

from config.logging_config import get_agent_logger
from .market_data_bus import MarketDataBus
from .shared_ltp import SharedLTPCache
from .bar_subscriber import BarSubscriber

logger = get_agent_logger()


def create_market_data_components(
    config: dict,
    on_1m_close: Optional[Callable] = None,
    on_5m_close: Optional[Callable] = None,
    on_15m_close: Optional[Callable] = None,
    index_symbols: Optional[list] = None,
    ltp_batcher: Optional[Any] = None,
) -> Tuple[Any, SharedLTPCache, Optional[MarketDataBus]]:
    """
    Create market data components based on configuration.

    Args:
        config: Configuration dict (should contain 'market_data_bus' section)
        on_1m_close: Callback for 1m bar closes
        on_5m_close: Callback for 5m bar closes
        on_15m_close: Callback for 15m bar closes
        index_symbols: List of index symbols for BarBuilder
        ltp_batcher: Optional LTP batcher for WebSocket broadcasts

    Returns:
        Tuple of (bar_builder_or_subscriber, ltp_cache, market_data_bus)

    Mode behavior:
        standalone: Returns BarBuilder + local LTPCache + None
        publisher:  Returns BarBuilder (with publish hooks) + publisher LTPCache + MarketDataBus
        subscriber: Returns BarSubscriber + subscriber LTPCache + MarketDataBus
    """
    mdb_config = config.get("market_data_bus", {})
    mode = mdb_config.get("mode", "standalone")
    redis_url = mdb_config.get("redis_url", "redis://localhost:6379/0")
    publish_1m = mdb_config.get("publish_1m_bars", False)
    ltp_mode = mdb_config.get("ltp_mode", mode)  # Default LTP mode follows bus mode

    logger.info(f"MARKET_DATA_FACTORY | mode={mode}, ltp_mode={ltp_mode}")

    if mode == "subscriber":
        return _create_subscriber_components(
            redis_url=redis_url,
            ltp_mode=ltp_mode,
            on_1m_close=on_1m_close,
            on_5m_close=on_5m_close,
            on_15m_close=on_15m_close,
            ltp_batcher=ltp_batcher,
        )
    elif mode == "publisher":
        return _create_publisher_components(
            config=config,
            redis_url=redis_url,
            publish_1m=publish_1m,
            ltp_mode=ltp_mode,
            on_1m_close=on_1m_close,
            on_5m_close=on_5m_close,
            on_15m_close=on_15m_close,
            index_symbols=index_symbols,
            ltp_batcher=ltp_batcher,
        )
    else:
        # Standalone mode - existing behavior
        return _create_standalone_components(
            config=config,
            on_1m_close=on_1m_close,
            on_5m_close=on_5m_close,
            on_15m_close=on_15m_close,
            index_symbols=index_symbols,
            ltp_batcher=ltp_batcher,
        )


def _create_standalone_components(
    config: dict,
    on_1m_close: Optional[Callable],
    on_5m_close: Optional[Callable],
    on_15m_close: Optional[Callable],
    index_symbols: Optional[list],
    ltp_batcher: Optional[Any],
) -> Tuple[Any, SharedLTPCache, None]:
    """Create standalone components (existing behavior)."""
    from services.ingest.bar_builder import BarBuilder

    bar_builder = BarBuilder(
        bar_5m_span_minutes=config.get("bar_5m_span_minutes", 5),
        on_1m_close=on_1m_close or (lambda s, b: None),
        on_5m_close=on_5m_close or (lambda s, b: None),
        on_15m_close=on_15m_close,
        index_symbols=index_symbols,
    )

    ltp_cache = SharedLTPCache(
        mode="standalone",
        ltp_batcher=ltp_batcher,
    )

    return bar_builder, ltp_cache, None


def _create_publisher_components(
    config: dict,
    redis_url: str,
    publish_1m: bool,
    ltp_mode: str,
    on_1m_close: Optional[Callable],
    on_5m_close: Optional[Callable],
    on_15m_close: Optional[Callable],
    index_symbols: Optional[list],
    ltp_batcher: Optional[Any],
) -> Tuple[Any, SharedLTPCache, MarketDataBus]:
    """Create publisher components (LIVE mode with Redis publishing)."""
    from services.ingest.bar_builder import BarBuilder

    # Create market data bus for publishing
    bus = MarketDataBus(
        mode="publisher",
        redis_url=redis_url,
        publish_1m_bars=publish_1m,
    )

    # Wrap callbacks to publish to Redis after processing
    def wrapped_1m_close(symbol, bar):
        if on_1m_close:
            on_1m_close(symbol, bar)
        bus.publish_bar(symbol, "1m", bar)

    def wrapped_5m_close(symbol, bar):
        if on_5m_close:
            on_5m_close(symbol, bar)
        bus.publish_bar(symbol, "5m", bar)

    def wrapped_15m_close(symbol, bar):
        if on_15m_close:
            on_15m_close(symbol, bar)
        bus.publish_bar(symbol, "15m", bar)

    bar_builder = BarBuilder(
        bar_5m_span_minutes=config.get("bar_5m_span_minutes", 5),
        on_1m_close=wrapped_1m_close,
        on_5m_close=wrapped_5m_close,
        on_15m_close=wrapped_15m_close if on_15m_close else None,
        index_symbols=index_symbols,
    )

    ltp_cache = SharedLTPCache(
        mode=ltp_mode if ltp_mode != "standalone" else "publisher",
        redis_url=redis_url,
        ltp_batcher=ltp_batcher,
    )

    logger.info(f"MARKET_DATA_FACTORY | Publisher mode active, Redis: {redis_url}")
    return bar_builder, ltp_cache, bus


def _create_subscriber_components(
    redis_url: str,
    ltp_mode: str,
    on_1m_close: Optional[Callable],
    on_5m_close: Optional[Callable],
    on_15m_close: Optional[Callable],
    ltp_batcher: Optional[Any],
) -> Tuple[BarSubscriber, SharedLTPCache, MarketDataBus]:
    """Create subscriber components (PAPER mode reading from Redis)."""

    # Create bar subscriber (receives bars from LIVE instance)
    subscriber = BarSubscriber(redis_url=redis_url)
    subscriber.set_callbacks(
        on_1m_close=on_1m_close,
        on_5m_close=on_5m_close,
        on_15m_close=on_15m_close,
    )
    subscriber.start()

    # Create LTP cache in subscriber mode (reads from Redis)
    ltp_cache = SharedLTPCache(
        mode=ltp_mode if ltp_mode != "standalone" else "subscriber",
        redis_url=redis_url,
        ltp_batcher=ltp_batcher,
    )

    # Create a dummy bus for compatibility (subscriber already handles subscription)
    bus = MarketDataBus(
        mode="subscriber",
        redis_url=redis_url,
    )

    logger.info(f"MARKET_DATA_FACTORY | Subscriber mode active, Redis: {redis_url}")
    return subscriber, ltp_cache, bus
