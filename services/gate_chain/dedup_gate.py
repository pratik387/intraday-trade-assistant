"""DedupGate — sub-project #4 gate chain stage D.

Streaming stateful gate that enforces same-symbol cooloff + setup-change +
strength requirements on admits. Called by LiveGateChain AFTER Conviction.
Replaces the _dedupe_ok method on ScreenerLive (refactored out as part of
sub-project #4 to move gate logic into the gate chain).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple


class DedupGate:
    """Stateful per-session dedup gate.

    Config keys (from top-level cfg dict, typically under `dedup_gate:`):
        enabled (bool) — master switch. When False, evaluate() is a pass-through.
        cooloff_bars (int) — minimum bar gap between two admits on same symbol
        require_setup_change (bool) — if True, second admit on same symbol
            rejected unless setup_type differs from prior admit

    State:
        _last_entry[sym] = {"ts": <datetime>, "setup": <str>, "score": <float>}
        _current_session: date — session boundary detection for reset

    Session reset: when candidate's session_date differs from _current_session,
    clear _last_entry. Matches ConvictionGate reset semantics.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._current_session: Optional[date] = None
        self._last_entry: Dict[str, Dict[str, Any]] = {}
        self._stats_admitted: int = 0
        self._stats_rejected: int = 0

    @staticmethod
    def _bars_since(older: datetime, newer: datetime) -> int:
        """Number of 5-min bars between two datetimes. 9999 on any error."""
        try:
            delta = (newer - older).total_seconds() / 60.0
            return int(delta // 5)
        except Exception:
            return 9999

    def evaluate(
        self,
        sym: str,
        now_ts: datetime,
        setup_type: Optional[str],
        score: float,
        pctl_score: float,
        session_date: Optional[date] = None,
    ) -> Tuple[bool, str]:
        """Return (allow, reason).

        session_date: when provided, triggers _last_entry reset on session change.
            Parity with ConvictionGate. None = no session-boundary tracking.
        """
        if not self.cfg.get("enabled", False):
            return True, "gate_disabled"

        # Session boundary reset
        if session_date is not None and session_date != self._current_session:
            self._current_session = session_date
            self._last_entry.clear()

        cool = int(self.cfg["cooloff_bars"])
        need_change = bool(self.cfg["require_setup_change"])
        last = self._last_entry.get(sym)
        if not last:
            # no prior accept → allow AND record
            self._last_entry[sym] = {"ts": now_ts, "setup": setup_type, "score": float(score)}
            self._stats_admitted += 1
            return True, "admitted"

        bars_gap = self._bars_since(last["ts"], now_ts)
        if bars_gap < cool:
            self._stats_rejected += 1
            return False, f"cooloff bars_gap={bars_gap}<{cool}"

        if need_change and setup_type is not None and last.get("setup") == setup_type:
            self._stats_rejected += 1
            return False, f"setup_unchanged setup={setup_type}"

        last_score = float(last.get("score") or float("-inf"))
        required = max(pctl_score, last_score)
        if float(score) < required:
            self._stats_rejected += 1
            return False, f"score_weak score={score:.3f}<required={required:.3f}"

        # Admit + update last_entry
        self._last_entry[sym] = {"ts": now_ts, "setup": setup_type, "score": float(score)}
        self._stats_admitted += 1
        return True, "admitted"

    def stats(self) -> Dict[str, int]:
        return {
            "admitted": self._stats_admitted,
            "rejected": self._stats_rejected,
        }
