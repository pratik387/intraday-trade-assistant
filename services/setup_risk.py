"""Per-setup risk controls: concurrency cap, per-symbol cooloff, per-bar rate limit.

2026-05-12 architectural refactor: replaces the deleted gate_chain
(DedupGate + CrossSectionalGate). Each setup config declares its own risk
limits; this module enforces them per-decision, using PositionStore for
concurrency lookups.

Reuses services.state.position_store for open-position tracking — no
duplicate state.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, Tuple

import pandas as pd


_RATE_WINDOW_MIN = 5   # max_fires_per_5min uses a hard 5-minute sliding window


class SetupRiskTracker:
    """Enforces per-setup concurrency / cooloff / rate-limit constraints.

    Configuration is read from `setups_cfg[setup_type]`. Required keys per
    active setup: max_concurrent_positions, per_symbol_cooloff_min,
    max_fires_per_5min. If a setup is absent from setups_cfg, all checks
    pass (used for testing or new setups in development).
    """

    def __init__(self, setups_cfg: Dict[str, Dict[str, Any]], position_store):
        self._cfg = setups_cfg or {}
        self._positions = position_store
        # (setup_type, symbol) -> last admit ts
        self._last_admit_ts: Dict[Tuple[str, str], pd.Timestamp] = {}
        # setup_type -> deque of admit timestamps (for rate-limit window)
        self._recent_admits: Dict[str, Deque[pd.Timestamp]] = defaultdict(deque)

    def can_admit(
        self, symbol: str, setup_type: str, ts: pd.Timestamp,
    ) -> Tuple[bool, str]:
        """Check whether a new admit is allowed.

        Returns (ok, reason). reason is empty when ok=True; otherwise a
        short tag for logging (e.g. 'concurrent_cap', 'cooloff', 'rate_limit').
        """
        scfg = self._cfg.get(setup_type)
        if scfg is None:
            return True, ""

        # 1. Concurrency cap
        max_concurrent = int(scfg.get("max_concurrent_positions", 0)) or 0
        if max_concurrent > 0:
            open_for_setup = self._open_count_for_setup(setup_type)
            if open_for_setup >= max_concurrent:
                return False, f"concurrent_cap_{open_for_setup}/{max_concurrent}"

        # 2. Per-symbol cooloff
        cooloff_min = float(scfg.get("per_symbol_cooloff_min", 0))
        if cooloff_min > 0:
            last_ts = self._last_admit_ts.get((setup_type, symbol))
            if last_ts is not None:
                elapsed_min = (ts - last_ts).total_seconds() / 60.0
                if elapsed_min < cooloff_min:
                    return False, f"cooloff_{elapsed_min:.1f}m<{cooloff_min}m"

        # 3. Rate-limit (max fires in trailing _RATE_WINDOW_MIN)
        max_fires = int(scfg.get("max_fires_per_5min", 0))
        if max_fires > 0:
            window_start = ts - pd.Timedelta(minutes=_RATE_WINDOW_MIN)
            q = self._recent_admits[setup_type]
            # Drop stale entries
            while q and q[0] < window_start:
                q.popleft()
            if len(q) >= max_fires:
                return False, f"rate_limit_{len(q)}/{max_fires}_in_{_RATE_WINDOW_MIN}min"

        return True, ""

    def record_admit(
        self, symbol: str, setup_type: str, ts: pd.Timestamp,
    ) -> None:
        """Record that a position was admitted. Updates cooloff + rate-limit state."""
        self._last_admit_ts[(setup_type, symbol)] = ts
        self._recent_admits[setup_type].append(ts)

    def _open_count_for_setup(self, setup_type: str) -> int:
        """Count currently-open positions for this setup_type.

        Uses position_store's list_open() and filters by plan.strategy
        (which equals setup_type in our pipeline).
        """
        try:
            open_dict = self._positions.list_open()
        except AttributeError:
            # Test fakes: try open_by_setup if available
            try:
                return len(self._positions.open_by_setup(setup_type))
            except AttributeError:
                return 0
        count = 0
        for pos in open_dict.values():
            plan = getattr(pos, "plan", None) or {}
            if plan.get("strategy") == setup_type:
                count += 1
        return count
