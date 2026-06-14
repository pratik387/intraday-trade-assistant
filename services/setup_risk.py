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
_IN_FLIGHT_TTL_MIN = 10  # admit -> fill window; bridges the race where
                         # multiple admits in the same bar all see list_open()=0


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
        # setup_type -> {symbol: admit_ts}. Tracks admits between
        # record_admit and the moment the position lands in
        # position_store. Without this, N admits in the same bar all see
        # list_open()=0 and bypass max_concurrent_positions
        # (paper-trade 9-day data Jun 2026: gap_fade_short cap=5 saw
        # 9-10 concurrent entries at 09:15 on 5/9 sessions).
        self._in_flight: Dict[str, Dict[str, pd.Timestamp]] = defaultdict(dict)

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
            open_for_setup = self._open_count_for_setup(setup_type, ts)
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
        """Record that a position was admitted. Updates cooloff, rate-limit,
        and in-flight concurrency state."""
        self._last_admit_ts[(setup_type, symbol)] = ts
        self._recent_admits[setup_type].append(ts)
        # Reserve the concurrency slot until position_store sees the fill
        # (or until the TTL expires for never-filled admits).
        self._in_flight[setup_type][symbol] = ts

    def _open_count_for_setup(
        self, setup_type: str, ts: pd.Timestamp = None,
    ) -> int:
        """Count currently-open + in-flight admits for this setup_type.

        Uses position_store's list_open() and filters by plan.strategy
        (which equals setup_type in our pipeline). Adds in-flight admits
        (admitted this bar but not yet visible in position_store) so the
        cap is respected when N signals fire in the same bar.

        Dedupes by symbol so an admit that has since landed in
        position_store isn't double-counted. TTL-purges in-flight entries
        older than _IN_FLIGHT_TTL_MIN so an admit that never fills
        doesn't permanently block the cap.
        """
        open_symbols = set()
        open_count = 0
        try:
            open_dict = self._positions.list_open()
            for pos in open_dict.values():
                plan = getattr(pos, "plan", None) or {}
                if plan.get("strategy") == setup_type:
                    open_count += 1
                    sym = plan.get("symbol") or getattr(pos, "symbol", None)
                    if sym:
                        open_symbols.add(sym)
        except AttributeError:
            # Test fakes / older stores: fall back to open_by_setup.
            try:
                positions = self._positions.open_by_setup(setup_type)
                open_count = len(positions)
                for pos in positions:
                    sym = (
                        pos.get("symbol") if isinstance(pos, dict)
                        else getattr(pos, "symbol", None)
                    )
                    if sym:
                        open_symbols.add(sym)
            except AttributeError:
                pass

        in_flight = self._in_flight.get(setup_type)
        if in_flight:
            cutoff = (ts - pd.Timedelta(minutes=_IN_FLIGHT_TTL_MIN)) if ts is not None else None
            # Purge admits that have either materialized into list_open()
            # or aged past the TTL.
            for sym in list(in_flight.keys()):
                admit_ts = in_flight[sym]
                if sym in open_symbols:
                    in_flight.pop(sym)
                elif cutoff is not None and admit_ts < cutoff:
                    in_flight.pop(sym)
            open_count += len(in_flight)

        return open_count
