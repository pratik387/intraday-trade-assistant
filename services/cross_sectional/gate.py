"""F1+F2 composition gate. Config-driven, stateless (state lives in injected components)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Protocol, Tuple


@dataclass(frozen=True)
class Candidate:
    """Minimal decision-time context needed by CrossSectionalGate."""
    symbol: str
    setup_type: str
    cap_segment: str
    hour_bucket: str
    decision_ts: datetime


class _RVOLLike(Protocol):
    def get_rvol_pct_tier(self, symbol: str, ts: datetime): ...


class _CrowdLike(Protocol):
    def count(self, setup_type: str, ts: datetime) -> int: ...
    def record(self, setup_type: str, ts: datetime) -> None: ...


class CrossSectionalGate:
    """Applies F1 (RVOL cap-conditional) + F2 (crowdedness universal) filters.

    Config keys (see specs/2026-04-21-sub-project-3-cross-sectional-design.md §4.5).
    All thresholds injected via `cfg` dict — no hardcoded defaults.
    """

    def __init__(self, cfg: Dict[str, Any], rvol: _RVOLLike, crowdedness: _CrowdLike):
        self.cfg = cfg
        self.rvol = rvol
        self.crowdedness = crowdedness

    def evaluate(self, cand: Candidate) -> Tuple[bool, str]:
        """Return (allow, reason). reason is comma-separated list of failed checks
        if allow=False, or 'allowed' if allow=True."""
        if not self.cfg.get("enabled", False):
            return True, "gate_disabled"

        failures: List[str] = []

        # F1: RVOL cap-conditional
        if self.cfg.get("f1_rvol_enabled", False):
            applicable_caps = set(self.cfg.get("f1_applicable_caps", []))
            skip_hours = set(self.cfg.get("f1_skip_hour_buckets", []))
            if cand.cap_segment in applicable_caps and cand.hour_bucket not in skip_hours:
                rvol_pct = self.rvol.get_rvol_pct_tier(cand.symbol, cand.decision_ts)
                threshold = float(self.cfg["f1_rvol_threshold_pct"])
                if rvol_pct is not None and rvol_pct >= threshold:
                    failures.append(f"f1_rvol_pct={rvol_pct:.1f}>={threshold}")

        # F2: Crowdedness universal
        if self.cfg.get("f2_crowdedness_enabled", False):
            crowd = self.crowdedness.count(cand.setup_type, cand.decision_ts)
            threshold = int(self.cfg["f2_crowdedness_threshold"])
            if crowd >= threshold:
                failures.append(f"f2_crowded_count={crowd}>={threshold}")

        if failures:
            return False, ",".join(failures)
        return True, "allowed"
