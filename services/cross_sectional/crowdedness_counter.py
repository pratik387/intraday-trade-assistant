"""F2: Crowdedness counter — backward-only 5-min sliding window per setup_type.

At decision time for candidate (setup_type, decision_ts), returns the count of
OTHER recorded events with same setup_type in [decision_ts - window, decision_ts].
Backward-only: future events not counted (live-realistic, no lookahead bias).

Memory-bounded via pruning: events older than 2x window are discarded on record.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict


class CrowdednessCounter:
    """Per-setup-type backward-only sliding window counter.

    Args:
        window_min: window size in minutes (e.g., 5)
    """

    def __init__(self, window_min: int):
        if window_min <= 0:
            raise ValueError(f"window_min must be positive, got {window_min}")
        self.window_min = window_min
        self._events: Dict[str, Deque[datetime]] = defaultdict(deque)

    def record(self, setup_type: str, ts: datetime) -> None:
        """Record that a signal fired for setup_type at ts."""
        dq = self._events[setup_type]
        dq.append(ts)
        # Prune events older than 2x window to bound memory
        prune_before = ts - timedelta(minutes=2 * self.window_min)
        while dq and dq[0] < prune_before:
            dq.popleft()

    def count(self, setup_type: str, ts: datetime) -> int:
        """Count events for setup_type in [ts - window, ts] (inclusive both)."""
        dq = self._events.get(setup_type)
        if not dq:
            return 0
        lo = ts - timedelta(minutes=self.window_min)
        return sum(1 for t in dq if lo <= t <= ts)

    def reset(self) -> None:
        """Clear all state (used in backtest between sessions or tests)."""
        self._events.clear()
