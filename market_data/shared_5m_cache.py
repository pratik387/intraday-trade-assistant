"""
Shared 5m Bar Cache — Redis-backed cross-instance cache for enriched API 5m bars.

Problem: Multiple instances (paper + live, or multiple papers) all fetch API 5m bars.
At 20 RPS × 3 instances = 60 RPS → 429 rate limit errors.

Solution: First instance to fetch writes enriched bars to Redis. Other instances
read from Redis instead of hitting the API. Non-destructive reads — all instances
get the same data.

Key format: api5m:{bar_timestamp_iso}:{symbol}
TTL: 6 minutes (bars are fetched every 5m cycle, stale after one cycle)
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger("agent")

# Compact serialization: store as JSON list of [ts_iso, o, h, l, c, v, vwap, bb, adx, rsi]
_COLS = ["open", "high", "low", "close", "volume", "vwap", "bb_width_proxy", "adx", "rsi"]
_TTL_SEC = 360  # 6 minutes


class Shared5mCache:
    """Redis-backed shared cache for enriched API 5m bars."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis = None
        self._redis_url = redis_url
        self._enabled = False
        try:
            import redis
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._enabled = True
            logger.info("SHARED_5M_CACHE | Connected to Redis: %s", redis_url)
        except Exception as e:
            logger.warning("SHARED_5M_CACHE | Redis unavailable (%s), running without shared cache", e)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _bar_key(self, bar_ts: str) -> str:
        return f"api5m:{bar_ts}"

    def get_cached_bars(self, bar_ts: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Try to load enriched 5m bars for a given bar timestamp from Redis.
        Returns dict of {symbol: DataFrame} or None if cache miss.
        """
        if not self._enabled:
            return None

        try:
            key = self._bar_key(bar_ts)
            data = self._redis.get(key)
            if data is None:
                return None

            result = {}
            payload = json.loads(data)
            for sym, rows in payload.items():
                if not rows:
                    continue
                df = pd.DataFrame(rows, columns=["date"] + _COLS)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                for col in _COLS:
                    df[col] = df[col].astype(float)
                result[sym] = df

            logger.info("SHARED_5M_CACHE | HIT for %s — %d symbols from Redis", bar_ts, len(result))
            return result

        except Exception as e:
            logger.debug("SHARED_5M_CACHE | Read failed: %s", e)
            return None

    def store_bars(self, bar_ts: str, bars: Dict[str, pd.DataFrame]) -> None:
        """
        Store enriched 5m bars in Redis for other instances to read.
        """
        if not self._enabled or not bars:
            return

        try:
            payload = {}
            for sym, df in bars.items():
                if df is None or df.empty:
                    continue
                rows = []
                for ts, row in df.iterrows():
                    rows.append([ts.isoformat()] + [float(row.get(c, 0)) for c in _COLS])
                payload[sym] = rows

            key = self._bar_key(bar_ts)
            self._redis.set(key, json.dumps(payload), ex=_TTL_SEC)
            logger.info("SHARED_5M_CACHE | STORED %d symbols for %s (TTL=%ds)", len(payload), bar_ts, _TTL_SEC)

        except Exception as e:
            logger.debug("SHARED_5M_CACHE | Write failed: %s", e)

    def shutdown(self):
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
