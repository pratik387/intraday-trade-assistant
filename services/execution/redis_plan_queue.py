"""
Redis-backed Plan Queue — trade plan handoff between scan and exec processes.

Two duck-type replacements for OrderQueue:

  RedisPlanPublisher  — scan process side (enqueue only)
    Interface: enqueue(order: dict) -> None
    Used by: ScreenerLive (self.oq.enqueue(exec_item))

  RedisPlanConsumer   — exec process side (dequeue only)
    Interface: get_next(timeout: float) -> Optional[dict]
    Used by: TriggerAwareExecutor (self.oq.get_next(timeout=0.1))

Transport: Redis list with RPUSH (publish) / BLPOP (consume).
Serialization: JSON with default=str for Timestamp/datetime fields.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from config.logging_config import get_agent_logger

logger = get_agent_logger()

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisPlanPublisher:
    """
    Drop-in replacement for OrderQueue on the scan side.

    Publishes trade plan dicts to a Redis list via RPUSH.
    ScreenerLive calls self.oq.enqueue(exec_item) — this class
    implements that interface so zero changes needed in ScreenerLive.
    """

    def __init__(self, redis_url: str, queue_key: str) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required for RedisPlanPublisher")

        self._queue_key = queue_key
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._redis.ping()
        logger.info(
            "PLAN_QUEUE | Publisher connected to Redis, queue_key=%s", queue_key
        )

    def enqueue(self, order: Dict[str, Any]) -> None:
        """Publish a trade plan to Redis (same signature as OrderQueue.enqueue)."""
        try:
            payload = json.dumps(order, default=str)
            self._redis.rpush(self._queue_key, payload)
            symbol = order.get("symbol", "?")
            logger.info("PLAN_QUEUE | Published plan for %s", symbol)
        except Exception as e:
            logger.error("PLAN_QUEUE | Failed to publish plan: %s", e)

    def approx_backlog(self) -> int:
        """Return approximate queue length (for monitoring)."""
        try:
            return self._redis.llen(self._queue_key)
        except Exception:
            return 0

    def shutdown(self) -> None:
        """Close Redis connection."""
        try:
            self._redis.close()
        except Exception:
            pass


class RedisPlanConsumer:
    """
    Drop-in replacement for OrderQueue on the exec side.

    Consumes trade plan dicts from a Redis list via BLPOP.
    TriggerAwareExecutor calls self.oq.get_next(timeout=0.1) — this class
    implements that interface so zero changes needed in TriggerAwareExecutor.
    """

    def __init__(self, redis_url: str, queue_key: str) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required for RedisPlanConsumer")

        self._queue_key = queue_key
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._redis.ping()
        logger.info(
            "PLAN_QUEUE | Consumer connected to Redis, queue_key=%s", queue_key
        )

    def get_next(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Dequeue the next trade plan from Redis (same signature as OrderQueue.get_next).

        Uses BLPOP for blocking wait. Returns None on timeout (matches OrderQueue).
        """
        try:
            # BLPOP timeout is in seconds (int). 0 means block forever.
            # Convert float timeout to int, minimum 1 second for Redis.
            if timeout is None or timeout <= 0:
                blpop_timeout = 1  # Avoid infinite block; re-check every 1s
            else:
                blpop_timeout = max(1, int(timeout))

            result = self._redis.blpop(self._queue_key, timeout=blpop_timeout)
            if result is None:
                return None

            # BLPOP returns (key, value) tuple
            _, payload = result
            order = json.loads(payload)
            symbol = order.get("symbol", "?")
            logger.info("PLAN_QUEUE | Consumed plan for %s", symbol)
            return order

        except Exception as e:
            logger.error("PLAN_QUEUE | Failed to consume plan: %s", e)
            return None

    def approx_backlog(self) -> int:
        """Return approximate queue length (for monitoring)."""
        try:
            return self._redis.llen(self._queue_key)
        except Exception:
            return 0

    def shutdown(self) -> None:
        """Close Redis connection."""
        try:
            self._redis.close()
        except Exception:
            pass
