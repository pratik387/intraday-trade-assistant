"""F1: UniverseRVOLState — per-symbol 20-session rolling same-mod volume mean
and cross-sectional rank within cap_segment tiers.

On each 5m bar close, called with all MIS-universe volumes at that bar. Maintains
per-(symbol, mod) deque of last N sessions' volumes and computes rvol per symbol.
Cross-sectional rank is computed per cap_segment tier.

Thread-safe contract: assume single-threaded caller (trading decision pipeline
is single-threaded; backtest replay is single-threaded).
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, Optional, Tuple

import numpy as np


class UniverseRVOLState:
    """Per-symbol rolling-mean volume + cross-sectional RVOL rank per cap_segment.

    Args:
        rolling_sessions: number of historical sessions for same-mod volume mean
        min_sessions: minimum sessions of history required before rvol_pct is returned
    """

    def __init__(self, rolling_sessions: int, min_sessions: int):
        if rolling_sessions <= 0:
            raise ValueError(f"rolling_sessions must be positive, got {rolling_sessions}")
        if min_sessions <= 0 or min_sessions > rolling_sessions:
            raise ValueError(f"min_sessions must be 1..{rolling_sessions}, got {min_sessions}")
        self.rolling_sessions = rolling_sessions
        self.min_sessions = min_sessions
        # (symbol, mod) -> deque of (session_date, volume) for last N sessions
        self._history: Dict[Tuple[str, int], Deque[Tuple[object, int]]] = defaultdict(deque)
        # (ts) -> {symbol -> (rvol_pct, cap_tier)} — last computed snapshot
        self._last_snapshot_ts: Optional[datetime] = None
        self._last_rvol_pct: Dict[str, float] = {}

    def on_bar_close(
        self,
        ts: datetime,
        bar_volumes: Dict[str, int],
        symbol_caps: Dict[str, str],
    ) -> None:
        """Record bar close + recompute cross-sectional ranks.

        ts: timestamp at bar close (IST-naive)
        bar_volumes: {symbol -> volume for just-closed bar}
        symbol_caps: {symbol -> cap_segment} for all universe symbols
        """
        mod = ts.hour * 60 + ts.minute
        session_date = ts.date()

        # Update per-(symbol, mod) history
        for symbol, volume in bar_volumes.items():
            key = (symbol, mod)
            dq = self._history[key]
            # Only record one datapoint per session for this mod (overwrite-if-same-date)
            if dq and dq[-1][0] == session_date:
                dq[-1] = (session_date, volume)
            else:
                dq.append((session_date, volume))
            # Trim to rolling_sessions
            while len(dq) > self.rolling_sessions:
                dq.popleft()

        # Compute rvol per symbol (current volume / mean of prior sessions, excluding current)
        rvol_by_tier: Dict[str, Dict[str, float]] = defaultdict(dict)
        for symbol, current_vol in bar_volumes.items():
            key = (symbol, mod)
            dq = self._history[key]
            prior_vols = [v for d, v in dq if d < session_date]
            if len(prior_vols) < self.min_sessions:
                continue
            mean_prior = float(np.mean(prior_vols))
            if mean_prior <= 0:
                continue
            rvol = float(current_vol) / mean_prior
            cap = symbol_caps.get(symbol, "unknown")
            rvol_by_tier[cap][symbol] = rvol

        # Cross-sectional rank within each cap_segment tier
        self._last_rvol_pct.clear()
        for cap, rvol_map in rvol_by_tier.items():
            if len(rvol_map) < 2:
                # Cannot rank a tier of 1 — assign mid-percentile
                for sym in rvol_map:
                    self._last_rvol_pct[sym] = 50.0
                continue
            syms = list(rvol_map.keys())
            vals = np.array([rvol_map[s] for s in syms], dtype=float)
            n = len(vals)
            # Average-rank percentile (handles ties): for each value, rank is the
            # average of positions it occupies after sorting. Ties share pct.
            order = vals.argsort()
            ranks = np.empty(n, dtype=float)
            i = 0
            while i < n:
                j = i
                # Group contiguous ties in sorted order
                while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                    j += 1
                avg_rank = (i + j) / 2.0  # average 0-based rank
                for k in range(i, j + 1):
                    ranks[order[k]] = avg_rank
                i = j + 1
            if n > 1:
                pcts = ranks / (n - 1) * 100.0
            else:
                pcts = np.array([50.0])
            for sym, pct in zip(syms, pcts):
                self._last_rvol_pct[sym] = float(pct)
        self._last_snapshot_ts = ts

    def get_rvol_pct_tier(self, symbol: str, ts: datetime) -> Optional[float]:
        """Return the cross-sectional rvol percentile (0-100) for symbol at ts
        within its cap_segment tier. None if insufficient history or ts doesn't
        match the last-computed snapshot."""
        if self._last_snapshot_ts != ts:
            return None
        return self._last_rvol_pct.get(symbol)

    def reset(self) -> None:
        self._history.clear()
        self._last_rvol_pct.clear()
        self._last_snapshot_ts = None
