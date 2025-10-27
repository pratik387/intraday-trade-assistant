from __future__ import annotations
"""
subscription_manager.py
-----------------------
Batched, debounced WebSocket subscription manager.

Goal: avoid 429s / churn by coalescing subscribe/unsubscribe operations into
periodic flushes. Keeps a persistent "core" set (always on) and a dynamic
"hot" set (changes during the session).

Public API
----------
class SubscriptionManager:
    def __init__(self, ws_client) -> None
    def start(self) -> None
    def stop(self) -> None

    def set_core(self, tokens: set[int]) -> None
    def set_hot(self, tokens: set[int]) -> None

    # For imperative updates if needed
    def add_hot(self, tokens: set[int]) -> None
    def remove_hot(self, tokens: set[int]) -> None

Notes
-----
- Reads config via config.filters_setup.load_filters() and *requires* keys:
    ws_batch_size, ws_flush_enabled, ws_flush_interval_ms
- Expects ws_client to expose subscribe(List[int]) and unsubscribe(List[int]).
- Thread‑safe. Use start()/stop() to run the periodic flusher.
"""
import threading
import time
from typing import Iterable, List, Optional, Set

from config.filters_setup import load_filters
from config.logging_config import get_agent_logger

logger = get_agent_logger()


class SubscriptionManager:
    def __init__(self, ws_client) -> None:
        self.ws = ws_client
        cfg = load_filters()
        try:
            self.batch_size: int = int(cfg["ws_batch_size"])  # e.g., 100–500
            self.flush_enabled: bool = bool(cfg["ws_flush_enabled"])  # 1/0
            self.flush_interval_ms: int = int(cfg["ws_flush_interval_ms"])  # e.g., 1500
        except KeyError as e:  # no defaults allowed in this project
            raise KeyError(f"SubscriptionManager: missing config key `{e.args[0]}`")

        # Threading state
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Token sets
        self._core: Set[int] = set()
        self._hot: Set[int] = set()
        self._current: Set[int] = set()  # currently subscribed at WS

        # Pending diffs accumulated since last flush
        self._pending_add: Set[int] = set()
        self._pending_del: Set[int] = set()

    # ------------------------------ lifecycle ------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="SubMgr", daemon=True)
        self._thread.start()
        logger.info("subscription-manager: started")

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None
        logger.info("subscription-manager: stopped")

    # ------------------------------ public API -----------------------------
    def set_core(self, tokens: Set[int]) -> None:
        with self._lock:
            tokens = set(int(x) for x in tokens)
            self._core = tokens
            self._recompute_desired_locked()

    def set_hot(self, tokens: Set[int]) -> None:
        with self._lock:
            tokens = set(int(x) for x in tokens)
            self._hot = tokens
            self._recompute_desired_locked()

    def add_hot(self, tokens: Iterable[int]) -> None:
        with self._lock:
            for t in tokens:
                self._hot.add(int(t))
            self._recompute_desired_locked()

    def remove_hot(self, tokens: Iterable[int]) -> None:
        with self._lock:
            for t in tokens:
                self._hot.discard(int(t))
            self._recompute_desired_locked()

    # Optional: immediate flush (blocking)
    def flush_now(self) -> None:
        with self._lock:
            self._apply_diffs_locked()

    # ------------------------------ internals ------------------------------
    def _run(self) -> None:
        # Periodically apply diffs
        interval = max(50, self.flush_interval_ms) / 1000.0
        while not self._stop.is_set():
            if self.flush_enabled:
                try:
                    with self._lock:
                        self._apply_diffs_locked()
                except Exception as e:
                    logger.warning(f"subscription-manager flush failed: {e}")
            self._stop.wait(interval)

    def _recompute_desired_locked(self) -> None:
        desired = self._core | self._hot
        to_add = desired - self._current
        to_del = self._current - desired
        self._pending_add |= to_add
        self._pending_del |= to_del
        # Do not apply immediately — let the flusher coalesce

    def _apply_diffs_locked(self) -> None:
        if not self._pending_add and not self._pending_del:
            return

        # Batch unsubscribe first (to reduce stream load), then subscribe
        if self._pending_del:
            dels = list(self._pending_del)
            self._pending_del.clear()
            for i in range(0, len(dels), self.batch_size):
                batch = dels[i : i + self.batch_size]
                try:
                    self.ws.unsubscribe_batch(batch)
                    for t in batch:
                        self._current.discard(t)
                except Exception as e:
                    logger.warning(f"WS unsubscribe failed for {len(batch)} tokens: {e}")

        if self._pending_add:
            adds = list(self._pending_add)
            self._pending_add.clear()
            total_added = 0
            for i in range(0, len(adds), self.batch_size):
                batch = adds[i : i + self.batch_size]
                try:
                    self.ws.subscribe_batch(batch)
                    for t in batch:
                        self._current.add(t)
                    total_added += len(batch)
                except Exception as e:
                    logger.warning(f"WS subscribe failed for {len(batch)} tokens: {e}")

            if total_added > 0:
                logger.info(f"Subscribed to {total_added} tokens (total: {len(self._current)})")
            elif adds:  # Only warn if there were tokens to subscribe
                logger.warning("No tokens were successfully subscribed")     
