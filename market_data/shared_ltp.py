"""
SharedLTPCache - Redis-backed LTP cache for multi-instance trading.

This is a drop-in replacement for the existing LTPCache (in main.py) that
can share LTP data across LIVE and PAPER instances via Redis.

Modes:
  - "standalone": Local-only cache (existing behavior, default)
  - "publisher": Writes to local cache AND Redis (LIVE mode)
  - "subscriber": Reads from Redis, no local writes (PAPER mode)
  - "hybrid": Writes to local AND reads from Redis as fallback

For PAPER mode with shared market data:
  - LIVE instance publishes LTP to Redis on every tick
  - PAPER instance reads from Redis for LTP queries
  - Both instances see identical prices for P&L calculation
"""

from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Optional, Tuple, Any

import pandas as pd

from config.logging_config import get_agent_logger

logger = get_agent_logger()

# Optional Redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class SharedLTPCache:
    """
    Thread-safe LTP cache with optional Redis sharing.

    Compatible with existing LTPCache interface:
      - update(symbol, ltp, ts) -> stores price
      - get_ltp(symbol) -> returns price or None
      - get_ltp_ts(symbol) -> returns (price, ts) tuple
    """

    # Redis key for LTP hash
    REDIS_LTP_KEY = "ltp:current"

    def __init__(
        self,
        mode: str = "standalone",
        redis_url: str = "redis://localhost:6379/0",
        ltp_batcher=None,
    ):
        """
        Initialize SharedLTPCache.

        Args:
            mode: "standalone" | "publisher" | "subscriber" | "hybrid"
            redis_url: Redis connection URL
            ltp_batcher: Optional WebSocket batcher for dashboard broadcasts
        """
        self._mode = mode
        self._redis_url = redis_url
        self._ltp_batcher = ltp_batcher

        # Local cache (always used for standalone/publisher/hybrid)
        self._d: dict[str, Tuple[float, pd.Timestamp]] = {}
        self._lock = Lock()

        # Redis connection (for non-standalone modes)
        self._redis: Optional[Any] = None

        if mode != "standalone":
            self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("SHARED_LTP | Redis not installed, falling back to standalone mode")
            self._mode = "standalone"
            return

        try:
            self._redis = redis.Redis.from_url(self._redis_url, decode_responses=True)
            self._redis.ping()
            logger.info(f"SHARED_LTP | Connected to Redis ({self._mode} mode)")
        except Exception as e:
            logger.warning(f"SHARED_LTP | Redis connection failed: {e}, falling back to standalone")
            self._mode = "standalone"
            self._redis = None

    def set_ltp_batcher(self, batcher) -> None:
        """Set LTP batcher for WebSocket broadcasts (dashboard)."""
        self._ltp_batcher = batcher

    def update(self, symbol: str, ltp: float, ts: pd.Timestamp) -> None:
        """
        Update LTP for a symbol.

        In publisher mode: Updates local cache AND Redis
        In subscriber mode: Does nothing (reads from Redis only)
        In standalone mode: Updates local cache only
        In hybrid mode: Updates local cache AND Redis
        """
        if self._mode == "subscriber":
            # Subscriber doesn't write - it reads from LIVE instance's Redis data
            return

        # Update local cache
        with self._lock:
            self._d[symbol] = (float(ltp), pd.Timestamp(ts))

        # Queue LTP for WebSocket broadcast (dashboard)
        if self._ltp_batcher:
            self._ltp_batcher.update(symbol, float(ltp), str(ts))

        # Publish to Redis (publisher/hybrid modes)
        if self._mode in ("publisher", "hybrid") and self._redis:
            try:
                # Store as "price:timestamp_ns" for fast parsing
                ts_val = ts.timestamp() if hasattr(ts, 'timestamp') else 0
                ts_ns = int(ts_val * 1_000_000)
                value = f"{ltp}:{ts_ns}"
                self._redis.hset(self.REDIS_LTP_KEY, symbol, value)
            except Exception:
                # Don't log every tick failure - too noisy
                pass

    def get_ltp(self, symbol: str) -> Optional[float]:
        """
        Get LTP for a symbol.

        In subscriber mode: Reads from Redis
        In hybrid mode: Tries local cache first, then Redis
        In standalone/publisher mode: Reads from local cache
        """
        if self._mode == "subscriber":
            return self._get_ltp_from_redis(symbol)

        if self._mode == "hybrid":
            # Try local first
            with self._lock:
                tup = self._d.get(symbol)
                if tup:
                    return float(tup[0])
            # Fall back to Redis
            return self._get_ltp_from_redis(symbol)

        # standalone/publisher: local cache only
        with self._lock:
            tup = self._d.get(symbol)
            return float(tup[0]) if tup else None

    def get_ltp_ts(self, symbol: str) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
        """
        Get LTP and timestamp for a symbol.

        Returns:
            (price, timestamp) tuple, or (None, None) if not found
        """
        if self._mode == "subscriber":
            result = self._get_ltp_ts_from_redis(symbol)
            return result if result else (None, None)

        if self._mode == "hybrid":
            # Try local first
            with self._lock:
                tup = self._d.get(symbol)
                if tup:
                    return tup
            # Fall back to Redis
            result = self._get_ltp_ts_from_redis(symbol)
            return result if result else (None, None)

        # standalone/publisher: local cache only
        with self._lock:
            tup = self._d.get(symbol)
            return tup if tup else (None, None)

    def _get_ltp_from_redis(self, symbol: str) -> Optional[float]:
        """Get LTP from Redis."""
        if not self._redis:
            return None

        try:
            value = self._redis.hget(self.REDIS_LTP_KEY, symbol)
            if value:
                price_str, _ = value.split(":")
                return float(price_str)
            return None
        except Exception:
            return None

    def _get_ltp_ts_from_redis(self, symbol: str) -> Optional[Tuple[float, pd.Timestamp]]:
        """Get LTP and timestamp from Redis."""
        if not self._redis:
            return None

        try:
            value = self._redis.hget(self.REDIS_LTP_KEY, symbol)
            if value:
                price_str, ts_str = value.split(":")
                price = float(price_str)
                # Convert microseconds back to timestamp
                ts_us = int(ts_str)
                ts = pd.Timestamp.fromtimestamp(ts_us / 1_000_000)
                return (price, ts)
            return None
        except Exception:
            return None

    def all_symbols(self) -> list[str]:
        """Get all symbols with LTP data."""
        if self._mode == "subscriber" and self._redis:
            try:
                return list(self._redis.hkeys(self.REDIS_LTP_KEY))
            except Exception:
                return []

        with self._lock:
            return list(self._d.keys())

    def clear(self) -> None:
        """Clear local cache (doesn't affect Redis)."""
        with self._lock:
            self._d.clear()

    def shutdown(self) -> None:
        """Clean shutdown."""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
        self._redis = None

    @property
    def mode(self) -> str:
        return self._mode
