"""
MarketDataBus - Redis pub/sub for bar distribution across trading instances.

Publisher (LIVE mode):
  - on_bar_close(symbol, timeframe, bar) -> publishes to Redis channel
  - update_ltp(symbol, price, ts) -> updates Redis hash

Subscriber (PAPER mode):
  - subscribe_bars(callback) -> receives bar events from Redis
  - get_ltp(symbol) -> reads from Redis hash

Channels:
  - bars:5m:{symbol}  - Per-symbol 5m bar events
  - bars:1m:{symbol}  - Per-symbol 1m bar events (optional)

Hashes:
  - ltp:current       - symbol -> "price:timestamp" (for fast LTP lookups)
  - bars:5m:latest    - symbol -> last completed bar JSON (for late joiners)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Optional, Dict, Any

from config.logging_config import get_agent_logger

logger = get_agent_logger()

# Optional Redis import - graceful fallback if not installed
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


@dataclass
class BarEvent:
    """Serializable bar event for pub/sub distribution."""
    symbol: str
    timeframe: str  # "1m", "5m", "15m"
    ts: str  # ISO format timestamp (bar start time)
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    adx: float = 0.0
    rsi: float = 50.0
    bb_width_proxy: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "BarEvent":
        d = json.loads(data)
        return cls(**d)

    @classmethod
    def from_bar_series(cls, symbol: str, timeframe: str, bar) -> "BarEvent":
        """Create BarEvent from pandas Series (output of BarBuilder)."""
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            ts=bar.name.isoformat() if hasattr(bar.name, 'isoformat') else str(bar.name),
            open=float(bar.get("open", 0)),
            high=float(bar.get("high", 0)),
            low=float(bar.get("low", 0)),
            close=float(bar.get("close", 0)),
            volume=float(bar.get("volume", 0)),
            vwap=float(bar.get("vwap", 0)),
            adx=float(bar.get("adx", 0)),
            rsi=float(bar.get("rsi", 50)),
            bb_width_proxy=float(bar.get("bb_width_proxy", 0)),
        )


class MarketDataBus:
    """
    Redis-backed pub/sub for distributing market data across instances.

    Modes:
      - "publisher": Publishes bars/LTP to Redis (LIVE mode)
      - "subscriber": Receives bars/LTP from Redis (PAPER mode)
      - "standalone": No Redis, local-only (default, backward compatible)

    Config keys (from configuration.json):
      - market_data_bus.mode: "standalone" | "publisher" | "subscriber"
      - market_data_bus.redis_url: Redis connection URL
      - market_data_bus.publish_1m_bars: Whether to publish 1m bars (default: false)
    """

    def __init__(
        self,
        mode: str = "standalone",
        redis_url: str = "redis://localhost:6379/0",
        publish_1m_bars: bool = False,
    ):
        self._mode = mode
        self._redis_url = redis_url
        self._publish_1m = publish_1m_bars
        self._redis: Optional[Any] = None
        self._pubsub: Optional[Any] = None
        self._subscriber_thread: Optional[threading.Thread] = None
        self._running = False

        # Callbacks for subscribers
        self._bar_callbacks: Dict[str, list[Callable[[BarEvent], None]]] = {
            "1m": [],
            "5m": [],
            "15m": [],
        }
        # Tick callbacks for real-time execution (symbol, price, volume, ts)
        self._tick_callbacks: list[Callable[[str, float, float, datetime], None]] = []

        # Latency tracking for subscriber mode
        self._latency_samples: list[float] = []
        self._latency_count: int = 0

        if mode != "standalone":
            self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.error("MKT_DATA_BUS | Redis not installed. pip install redis")
            raise ImportError("redis package required for market_data_bus")

        try:
            self._redis = redis.Redis.from_url(self._redis_url, decode_responses=True)
            self._redis.ping()
            logger.info(f"MKT_DATA_BUS | Connected to Redis: {self._redis_url}")
        except Exception as e:
            logger.error(f"MKT_DATA_BUS | Redis connection failed: {e}")
            raise

    # ======================== Publisher Methods ========================

    def publish_bar(self, symbol: str, timeframe: str, bar) -> None:
        """
        Publish a completed bar to Redis (called by BarBuilder callbacks).

        Args:
            symbol: Trading symbol (e.g., "RELIANCE")
            timeframe: "1m", "5m", or "15m"
            bar: pandas Series from BarBuilder
        """
        if self._mode != "publisher":
            return

        if timeframe == "1m" and not self._publish_1m:
            return

        try:
            event = BarEvent.from_bar_series(symbol, timeframe, bar)
            channel = f"bars:{timeframe}:{symbol}"

            # Publish to channel for real-time subscribers
            self._redis.publish(channel, event.to_json())

            # Store latest bar in hash for late joiners
            self._redis.hset(f"bars:{timeframe}:latest", symbol, event.to_json())

            logger.debug(f"MKT_DATA_BUS | Published {timeframe} bar: {symbol} @ {event.ts}")
        except Exception as e:
            logger.warning(f"MKT_DATA_BUS | Failed to publish bar: {e}")

    def update_ltp(self, symbol: str, price: float, ts: datetime) -> None:
        """
        Update shared LTP in Redis hash (called on every tick).
        This is for point-in-time LTP lookups.

        Args:
            symbol: Trading symbol
            price: Last traded price
            ts: Tick timestamp
        """
        if self._mode != "publisher":
            return

        try:
            # Store as "price:timestamp_ns" for fast parsing
            ts_ns = int(ts.timestamp() * 1_000_000) if hasattr(ts, 'timestamp') else 0
            value = f"{price}:{ts_ns}"
            self._redis.hset("ltp:current", symbol, value)
        except Exception:
            # Don't log every tick failure - too noisy
            pass

    def publish_tick(self, symbol: str, price: float, volume: float, ts: datetime) -> None:
        """
        Publish tick to Redis pub/sub for real-time execution.
        Subscribers receive every tick for trigger validation.

        Args:
            symbol: Trading symbol
            price: Last traded price
            volume: Tick volume
            ts: Tick timestamp
        """
        if self._mode != "publisher":
            return

        try:
            ts_ns = int(ts.timestamp() * 1_000_000) if hasattr(ts, 'timestamp') else 0
            send_ns = time.time_ns()  # For latency measurement
            # Compact format: "symbol|price|volume|ts_ns|send_ns"
            tick_data = f"{symbol}|{price}|{volume}|{ts_ns}|{send_ns}"
            self._redis.publish("ticks:stream", tick_data)
        except Exception:
            # Don't log every tick failure - too noisy
            pass

    # ======================== Subscriber Methods ========================

    def subscribe_bars(
        self,
        timeframe: str,
        callback: Callable[[BarEvent], None],
        symbols: Optional[list[str]] = None,
    ) -> None:
        """
        Subscribe to bar events (PAPER mode).

        Args:
            timeframe: "1m", "5m", or "15m"
            callback: Function called with BarEvent on each bar close
            symbols: Optional list of symbols to filter (None = all)
        """
        if self._mode != "subscriber":
            logger.warning("MKT_DATA_BUS | subscribe_bars called but mode is not 'subscriber'")
            return

        self._bar_callbacks[timeframe].append(callback)

        # Start subscriber thread if not running
        if not self._running:
            self._start_subscriber(symbols)

    def subscribe_ticks(
        self,
        callback: Callable[[str, float, float, datetime], None],
    ) -> None:
        """
        Subscribe to real-time tick events for execution layer.

        Args:
            callback: Function(symbol, price, volume, ts) called on each tick
        """
        if self._mode != "subscriber":
            logger.warning("MKT_DATA_BUS | subscribe_ticks called but mode is not 'subscriber'")
            return

        self._tick_callbacks.append(callback)

        # Start subscriber thread if not running
        if not self._running:
            self._start_subscriber(None)

    def _start_subscriber(self, symbols: Optional[list[str]] = None) -> None:
        """Start background thread for Redis subscription."""
        if self._running:
            return

        self._running = True
        self._pubsub = self._redis.pubsub()

        # Subscribe to pattern for all symbols or specific ones
        patterns = []
        for tf in ["1m", "5m", "15m"]:
            if symbols:
                for sym in symbols:
                    patterns.append(f"bars:{tf}:{sym}")
            else:
                patterns.append(f"bars:{tf}:*")

        # Always subscribe to tick stream for real-time execution
        direct_channels = ["ticks:stream"]

        if symbols:
            self._pubsub.subscribe(*patterns, *direct_channels)
        else:
            self._pubsub.psubscribe(*patterns)
            self._pubsub.subscribe(*direct_channels)

        self._subscriber_thread = threading.Thread(
            target=self._subscriber_loop,
            daemon=True,
            name="MarketDataBus-Subscriber"
        )
        self._subscriber_thread.start()
        logger.info(f"MKT_DATA_BUS | Subscriber started, patterns: {patterns}")

    def _subscriber_loop(self) -> None:
        """Background loop processing Redis pub/sub messages.

        Uses blocking listen() instead of polling get_message() for
        lower latency tick delivery (~5-10ms improvement).
        """
        try:
            # Blocking iterator - immediate delivery on message arrival
            for message in self._pubsub.listen():
                if not self._running:
                    break

                try:
                    if message["type"] not in ("message", "pmessage"):
                        continue

                    channel = message.get("channel", "") or message.get("pattern", "")
                    data = message["data"]

                    # Handle tick stream messages (high priority - latency sensitive)
                    if channel == "ticks:stream":
                        self._handle_tick_message(data)
                        continue

                    # Handle bar messages: "bars:5m:RELIANCE" -> timeframe="5m"
                    parts = channel.split(":")
                    if len(parts) >= 2 and parts[0] == "bars":
                        timeframe = parts[1]
                        try:
                            event = BarEvent.from_json(data)
                            for callback in self._bar_callbacks.get(timeframe, []):
                                try:
                                    callback(event)
                                except Exception as e:
                                    logger.exception(f"MKT_DATA_BUS | Bar callback error: {e}")
                        except Exception as e:
                            logger.warning(f"MKT_DATA_BUS | Failed to parse bar event: {e}")

                except Exception as e:
                    logger.warning(f"MKT_DATA_BUS | Message processing error: {e}")

        except Exception as e:
            if self._running:
                logger.warning(f"MKT_DATA_BUS | Subscriber loop error: {e}")

    def _handle_tick_message(self, data: str) -> None:
        """Parse and dispatch tick message to callbacks."""
        try:
            # Format: "symbol|price|volume|ts_ns|send_ns"
            parts = data.split("|")
            if len(parts) < 4:
                return

            symbol = parts[0]
            price = float(parts[1])
            volume = float(parts[2])
            ts_ns = int(parts[3])
            ts = datetime.fromtimestamp(ts_ns / 1_000_000)

            # Calculate latency if send_ns is present
            if len(parts) >= 5:
                send_ns = int(parts[4])
                recv_ns = time.time_ns()
                latency_ms = (recv_ns - send_ns) / 1_000_000
                self._track_latency(latency_ms)

            for callback in self._tick_callbacks:
                try:
                    callback(symbol, price, volume, ts)
                except Exception as e:
                    logger.exception(f"MKT_DATA_BUS | Tick callback error: {e}")
        except Exception as e:
            # Don't log every parse failure - too noisy
            pass

    def _track_latency(self, latency_ms: float) -> None:
        """Track tick latency statistics."""
        self._latency_samples.append(latency_ms)
        self._latency_count += 1

        # Log stats every 10000 ticks
        if self._latency_count % 10000 == 0:
            samples = self._latency_samples
            if samples:
                avg = sum(samples) / len(samples)
                p50 = sorted(samples)[len(samples) // 2]
                p99 = sorted(samples)[int(len(samples) * 0.99)]
                max_lat = max(samples)
                logger.info(
                    f"MKT_DATA_BUS | TICK_LATENCY | count={self._latency_count} | "
                    f"avg={avg:.1f}ms p50={p50:.1f}ms p99={p99:.1f}ms max={max_lat:.1f}ms"
                )
            # Reset samples but keep count
            self._latency_samples = []

    def get_ltp(self, symbol: str) -> Optional[tuple[float, int]]:
        """
        Get shared LTP from Redis (PAPER mode).

        Returns:
            Tuple of (price, timestamp_ns) or None if not found
        """
        if self._mode == "standalone":
            return None

        try:
            value = self._redis.hget("ltp:current", symbol)
            if value:
                price_str, ts_str = value.split(":")
                return (float(price_str), int(ts_str))
            return None
        except Exception:
            return None

    def get_latest_bar(self, symbol: str, timeframe: str = "5m") -> Optional[BarEvent]:
        """
        Get latest completed bar from Redis (for late joiners).

        Args:
            symbol: Trading symbol
            timeframe: "1m", "5m", or "15m"

        Returns:
            BarEvent or None if not found
        """
        if self._mode == "standalone":
            return None

        try:
            data = self._redis.hget(f"bars:{timeframe}:latest", symbol)
            if data:
                return BarEvent.from_json(data)
            return None
        except Exception:
            return None

    # ======================== ORB Levels (PDH/PDL/PDC/ORH/ORL) ========================

    def publish_orb_levels(self, session_date: str, levels_by_symbol: Dict[str, Dict[str, float]]) -> None:
        """
        Publish ORB levels to Redis (computed once at 09:40 by publisher).

        All subscriber instances will read these identical values, ensuring
        deterministic signals across LIVE/PAPER/FIXED modes.

        Args:
            session_date: Date string (YYYY-MM-DD)
            levels_by_symbol: Dict mapping symbol to levels dict (PDH/PDL/PDC/ORH/ORL)
        """
        if self._mode != "publisher":
            return

        try:
            key = f"orb:levels:{session_date}"
            pipeline = self._redis.pipeline()

            for symbol, lvls in levels_by_symbol.items():
                # Convert NaN to None for JSON serialization
                clean_lvls = {
                    k: (None if v != v else v)  # NaN check: NaN != NaN
                    for k, v in lvls.items()
                }
                pipeline.hset(key, symbol, json.dumps(clean_lvls))

            # Set 24h TTL for automatic cleanup
            pipeline.expire(key, 86400)
            pipeline.execute()

            valid_orb = sum(1 for v in levels_by_symbol.values()
                           if v.get("ORH") is not None and v.get("ORL") is not None
                           and v.get("ORH") == v.get("ORH"))  # NaN check
            logger.info(
                f"MKT_DATA_BUS | Published ORB levels for {len(levels_by_symbol)} symbols "
                f"({valid_orb} with valid ORH/ORL) to Redis"
            )
        except Exception as e:
            logger.warning(f"MKT_DATA_BUS | Failed to publish ORB levels: {e}")

    def get_orb_levels(self, session_date: str) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get ORB levels from Redis (subscriber mode).

        Returns cached PDH/PDL/PDC/ORH/ORL levels computed by the publisher.

        Args:
            session_date: Date string (YYYY-MM-DD)

        Returns:
            Dict mapping symbol to levels dict, or None if not found
        """
        if self._mode == "standalone":
            return None

        if self._redis is None:
            return None

        try:
            key = f"orb:levels:{session_date}"
            data = self._redis.hgetall(key)

            if not data:
                return None

            levels_by_symbol = {}
            for symbol, lvls_json in data.items():
                lvls = json.loads(lvls_json)
                # Convert None back to NaN for consistency with local computation
                levels_by_symbol[symbol] = {
                    k: (float("nan") if v is None else v)
                    for k, v in lvls.items()
                }

            logger.debug(f"MKT_DATA_BUS | Retrieved ORB levels for {len(levels_by_symbol)} symbols from Redis")
            return levels_by_symbol

        except Exception as e:
            logger.warning(f"MKT_DATA_BUS | Failed to get ORB levels: {e}")
            return None

    def has_orb_levels(self, session_date: str) -> bool:
        """Check if ORB levels exist in Redis for the given date."""
        if self._mode == "standalone" or self._redis is None:
            return False

        try:
            key = f"orb:levels:{session_date}"
            return self._redis.exists(key) > 0
        except Exception:
            return False

    # ======================== Lifecycle ========================

    def shutdown(self) -> None:
        """Clean shutdown of Redis connections."""
        self._running = False

        if self._pubsub:
            try:
                self._pubsub.close()
            except Exception:
                pass

        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass

        logger.info("MKT_DATA_BUS | Shutdown complete")

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_publisher(self) -> bool:
        return self._mode == "publisher"

    @property
    def is_subscriber(self) -> bool:
        return self._mode == "subscriber"
