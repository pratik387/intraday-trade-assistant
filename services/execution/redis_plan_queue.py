"""
Redis-backed Plan Queue — trade plan handoff between scan and exec processes.

Two duck-type replacements for OrderQueue:

  RedisPlanPublisher  — scan process side (enqueue only)
    Interface: enqueue(order: dict) -> None
    Used by: ScreenerLive (self.oq.enqueue(exec_item))

  RedisPlanConsumer   — exec process side (dequeue only)
    Interface: get_next(timeout: float) -> Optional[dict]
    Used by: TriggerAwareExecutor (self.oq.get_next(timeout=0.1))

Transport: Redis Pub/Sub with PUBLISH (fan-out) / SUBSCRIBE (all consumers).
Each consumer receives ALL plans — supports multiple exec instances
(1 live + N paper) without BLPOP race conditions.

Serialization: JSON with default=str for Timestamp/datetime fields.
"""
from __future__ import annotations

import json
import queue
import threading
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

    Publishes trade plan dicts via Redis PUBLISH (fan-out to all subscribers).
    ScreenerLive calls self.oq.enqueue(exec_item) — this class
    implements that interface so zero changes needed in ScreenerLive.
    """

    def __init__(self, redis_url: str, queue_key: str) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required for RedisPlanPublisher")

        self._channel = queue_key
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._redis.ping()
        logger.info(
            "PLAN_QUEUE | Publisher connected to Redis, channel=%s", queue_key
        )

    def enqueue(self, order: Dict[str, Any]) -> None:
        """Publish a trade plan to all subscribers via Redis Pub/Sub."""
        try:
            payload = json.dumps(order, default=str)
            num_receivers = self._redis.publish(self._channel, payload)
            symbol = order.get("symbol", "?")
            logger.info("PLAN_QUEUE | Published plan for %s (receivers=%d)", symbol, num_receivers)
            if num_receivers == 0:
                logger.warning("PLAN_QUEUE | No subscribers received plan for %s", symbol)
        except Exception as e:
            logger.error("PLAN_QUEUE | Failed to publish plan: %s", e)

    def approx_backlog(self) -> int:
        """Pub/Sub has no backlog concept — always returns 0."""
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

    Subscribes to Redis Pub/Sub channel and buffers incoming plans.
    Every consumer instance receives ALL plans (fan-out), solving the
    BLPOP race condition where multiple exec processes compete for plans.

    TriggerAwareExecutor calls self.oq.get_next(timeout=0.1) — this class
    implements that interface so zero changes needed in TriggerAwareExecutor.
    """

    def __init__(self, redis_url: str, queue_key: str) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required for RedisPlanConsumer")

        self._channel = queue_key
        self._buffer: queue.Queue = queue.Queue()
        self._running = True

        # Subscriber needs its own connection (Redis requirement for pub/sub)
        self._redis_sub = redis.Redis.from_url(redis_url, decode_responses=True)
        self._redis_sub.ping()

        # Start subscriber thread
        self._sub_thread = threading.Thread(
            target=self._subscribe_loop, daemon=True, name="PlanQueueSub"
        )
        self._sub_thread.start()

        logger.info(
            "PLAN_QUEUE | Consumer subscribed to Redis, channel=%s", queue_key
        )

    def _subscribe_loop(self) -> None:
        """Background thread: subscribe to channel and buffer incoming plans."""
        pubsub = self._redis_sub.pubsub()
        pubsub.subscribe(self._channel)

        try:
            while self._running:
                message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    continue
                if message["type"] != "message":
                    continue

                try:
                    order = json.loads(message["data"])
                    self._buffer.put(order)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error("PLAN_QUEUE | Failed to parse plan: %s", e)
        except Exception as e:
            if self._running:
                logger.error("PLAN_QUEUE | Subscriber loop error: %s", e)
        finally:
            try:
                pubsub.unsubscribe()
                pubsub.close()
            except Exception:
                pass

    def get_next(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Dequeue the next trade plan from the local buffer.

        Plans are received via Pub/Sub in a background thread and buffered.
        Returns None on timeout (matches OrderQueue interface).
        """
        try:
            wait = max(0.1, timeout) if timeout else 0.1
            order = self._buffer.get(timeout=wait)
            symbol = order.get("symbol", "?")
            logger.info("PLAN_QUEUE | Consumed plan for %s", symbol)
            return order
        except queue.Empty:
            return None

    def approx_backlog(self) -> int:
        """Return number of buffered plans waiting to be consumed."""
        return self._buffer.qsize()

    def shutdown(self) -> None:
        """Stop subscriber thread and close Redis connection."""
        self._running = False
        try:
            self._sub_thread.join(timeout=3)
        except Exception:
            pass
        try:
            self._redis_sub.close()
        except Exception:
            pass
