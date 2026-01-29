"""
Exec Process Heartbeat — liveness signal between scan and exec processes.

ExecHeartbeatPublisher: Exec process publishes a heartbeat key with TTL.
ExecHeartbeatChecker:   Scan process checks if exec is alive before enqueuing.

Uses a Redis key with TTL. If the key expires, exec is considered dead.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from config.logging_config import get_agent_logger

logger = get_agent_logger()

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class ExecHeartbeatPublisher:
    """
    Background thread that publishes exec process heartbeat to Redis.

    Sets a Redis key with TTL every interval_sec seconds.
    If the exec process dies, the key expires after ttl_sec.
    """

    def __init__(
        self,
        redis_url: str,
        key: str,
        interval_sec: int,
        ttl_sec: int,
    ) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required for ExecHeartbeatPublisher")

        self._key = key
        self._interval = interval_sec
        self._ttl = ttl_sec
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._redis.ping()

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start heartbeat publishing in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._publish_loop,
            name="ExecHeartbeat",
            daemon=True,
        )
        self._thread.start()
        logger.info("EXEC_HEARTBEAT | Publisher started (interval=%ds, ttl=%ds)", self._interval, self._ttl)

    def _publish_loop(self) -> None:
        while self._running:
            try:
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                self._redis.set(self._key, f"alive:{ts}", ex=self._ttl)
            except Exception as e:
                logger.warning("EXEC_HEARTBEAT | Publish failed: %s", e)
            time.sleep(self._interval)

    def stop(self) -> None:
        """Stop heartbeat publishing."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        # Remove heartbeat key so scan process sees stale immediately
        try:
            self._redis.delete(self._key)
        except Exception:
            pass
        try:
            self._redis.close()
        except Exception:
            pass
        logger.info("EXEC_HEARTBEAT | Publisher stopped")


class ExecHeartbeatChecker:
    """
    Check if the exec process is alive by reading its heartbeat key.

    Used by the scan process before enqueuing trade plans.
    """

    def __init__(self, redis_url: str, key: str) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required for ExecHeartbeatChecker")

        self._key = key
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)

    def is_alive(self) -> bool:
        """Return True if exec process heartbeat is present."""
        try:
            return self._redis.exists(self._key) > 0
        except Exception as e:
            logger.warning("EXEC_HEARTBEAT | Check failed: %s", e)
            return False

    def shutdown(self) -> None:
        """Close Redis connection."""
        try:
            self._redis.close()
        except Exception:
            pass
