from __future__ import annotations
"""
OrderQueue — a thin, deterministic, rate‑limited order spool.

Responsibilities
- Accept order requests from the screener/planner thread via `enqueue(order)`.
- Serve orders to the TradeExecutor via `get_next(timeout)` while enforcing
  broker‑safe throughput caps: per‑second, per‑minute, and per‑day.
- No defaults: reads absolute limits from config.filters_setup.load_filters().

Notes
- This queue does **not** place orders; it only meters them out.
- If limits are hit, `get_next` blocks (up to its timeout) and returns None when it
  could not legally emit an order before the deadline.
- Day counter resets when the local calendar date changes.

Contract
- Public API: enqueue(dict) -> None; get_next(timeout: float|None) -> dict|None
- Thread‑safe.
- Non‑destructive on limit: if an item is popped but cannot be emitted due to
  limits, it is held internally and retried on the next `get_next` call.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

from config.filters_setup import load_filters


@dataclass
class _Limits:
    per_sec: int
    per_min: int
    per_day: int
    idle_ms: int  # backoff sleep when throttled


class OrderQueue:
    def __init__(self) -> None:
        cfg = load_filters()  # must contain required keys — no defaults
        try:
            self._limits = _Limits(
                per_sec=int(cfg["order_max_per_sec"]),
                per_min=int(cfg["order_max_per_min"]),
                per_day=int(cfg["order_max_per_day"]),
                idle_ms=int(cfg["order_idle_ms"]),
            )
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"OrderQueue: missing config key `{missing}`")

        self._q: Deque[Dict[str, Any]] = deque()
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)

        # Rate windows
        self._sec_epoch = 0  # integer seconds bucket
        self._min_epoch = 0  # integer minutes bucket
        self._day_ordinal = 0  # days since epoch
        self._sec_count = 0
        self._min_count = 0
        self._day_count = 0

        # Held item when throttled (so we don't lose the head of queue)
        self._carry: Optional[Dict[str, Any]] = None

    # ------------------------------- Public API -------------------------------
    def enqueue(self, order: Dict[str, Any]) -> None:
        """Add an order request to the tail of the queue."""
        with self._not_empty:
            self._q.append(order)
            self._not_empty.notify()

    def get_next(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Pop the next order if available **and** allowed by rate limits.
        Blocks up to `timeout` seconds. Returns None on timeout.
        """
        deadline = None if timeout is None else (time.time() + max(0.0, timeout))
        backoff = self._limits.idle_ms / 1000.0

        with self._not_empty:
            while True:
                # Refresh windows before any decision
                self._roll_windows_locked()

                # Ensure we have something to consider
                if self._carry is None:
                    # Fast path: if queue empty, wait
                    if not self._q:
                        if deadline is None:
                            self._not_empty.wait()
                            # loop back and re-check
                            continue
                        else:
                            remaining = deadline - time.time()
                            if remaining <= 0:
                                return None
                            self._not_empty.wait(timeout=remaining)
                            if not self._q and (deadline - time.time()) <= 0:
                                return None
                            continue
                    # Pull head candidate
                    self._carry = self._q.popleft()

                # Check limits for emission
                if self._can_emit_locked():
                    self._increment_counters_locked()
                    order = self._carry
                    self._carry = None
                    return order

                # Throttled — wait a bit, but respect outer timeout
                if deadline is not None and (deadline - time.time()) <= 0:
                    return None
                # Put the thread to sleep briefly while holding the lock? No.
                # Release the lock so enqueues can proceed; then reacquire.
                self._not_empty.release()
                time.sleep(backoff)
                self._not_empty.acquire()
                # loop continues

    # ------------------------------ Internals ---------------------------------
    def _roll_windows_locked(self) -> None:
        now = time.time()
        sec_bucket = int(now)
        min_bucket = int(now // 60)
        day_ordinal = int(now // 86400)

        if sec_bucket != self._sec_epoch:
            self._sec_epoch = sec_bucket
            self._sec_count = 0
        if min_bucket != self._min_epoch:
            self._min_epoch = min_bucket
            self._min_count = 0
        if day_ordinal != self._day_ordinal:
            self._day_ordinal = day_ordinal
            self._day_count = 0

    def _can_emit_locked(self) -> bool:
        return (
            self._sec_count < self._limits.per_sec
            and self._min_count < self._limits.per_min
            and self._day_count < self._limits.per_day
        )

    def _increment_counters_locked(self) -> None:
        self._sec_count += 1
        self._min_count += 1
        self._day_count += 1

    # ---------------------------- Introspection -------------------------------
    def approx_backlog(self) -> int:
        with self._lock:
            n = len(self._q)
            if self._carry is not None:
                n += 1
            return n

    def limits(self) -> _Limits:
        return self._limits
