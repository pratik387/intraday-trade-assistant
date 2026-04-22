"""ConvictionGate — online top-N selection + minimum-conviction threshold.

Stateful: tracks per-session admitted count. Session boundary detected via
candidate.session_date change; counter resets.

Scorer runs UPSTREAM — this gate takes predicted_r as input (already scored).
Separation keeps gate testable without model artifacts.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Tuple


class ConvictionGate:
    """Applies top-N + threshold filter per session.

    Config keys:
        enabled (bool)
        daily_cap (int)
        min_predicted_r (float)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._current_session: Optional[date] = None
        self._admitted_today: int = 0
        self._stats_admitted: int = 0
        self._stats_rejected: int = 0

    def evaluate(self, cand: Dict[str, Any], predicted_r: float) -> Tuple[bool, str]:
        """Return (allow, reason).

        cand must include `session_date` (date) for daily cap tracking.
        """
        if not self.cfg.get("enabled", False):
            return True, "gate_disabled"

        # Session boundary reset
        sess = cand.get("session_date")
        if sess != self._current_session:
            self._current_session = sess
            self._admitted_today = 0

        # Threshold check
        min_r = float(self.cfg["min_predicted_r"])
        if predicted_r < min_r:
            self._stats_rejected += 1
            return False, f"below_threshold predicted_r={predicted_r:.3f}<{min_r}"

        # Daily cap check
        cap = int(self.cfg["daily_cap"])
        if self._admitted_today >= cap:
            self._stats_rejected += 1
            return False, f"daily_cap_reached count={self._admitted_today}>={cap}"

        # Admit
        self._admitted_today += 1
        self._stats_admitted += 1
        return True, "admitted"

    def stats(self) -> Dict[str, int]:
        return {
            "admitted": self._stats_admitted,
            "rejected": self._stats_rejected,
        }
