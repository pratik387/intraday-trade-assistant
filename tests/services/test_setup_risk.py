"""Tests for services.setup_risk.SetupRiskTracker."""
from datetime import datetime, timedelta
import pandas as pd
import pytest


def _ts(hhmm: str, day: str = "2024-08-29") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}")


def _cfg(max_concurrent: int = 2) -> dict:
    return {
        "gap_fade_short": {
            "max_concurrent_positions": max_concurrent,
            "per_symbol_cooloff_min": 15,
            "max_fires_per_5min": 3,
        },
    }


class _FakePositionStore:
    """Stand-in for services.state.position_store.PositionStore."""
    def __init__(self):
        self._open = {}   # trade_id -> position-like

    def list_open_by_symbol(self, sym):
        return [p for p in self._open.values() if p["symbol"] == sym]

    def open_by_setup(self, setup_type):
        return [p for p in self._open.values() if p["setup_type"] == setup_type]

    def add(self, sym, setup_type):
        tid = f"{sym}_{len(self._open)}"
        self._open[tid] = {"symbol": sym, "setup_type": setup_type}


def test_concurrency_cap_blocks_when_at_limit():
    from services.setup_risk import SetupRiskTracker
    ps = _FakePositionStore()
    rt = SetupRiskTracker(_cfg(), ps)
    # Fill to cap (max_concurrent_positions=2)
    ps.add("NSE:A", "gap_fade_short")
    ps.add("NSE:B", "gap_fade_short")
    ok, reason = rt.can_admit("NSE:C", "gap_fade_short", _ts("09:20"))
    assert not ok
    assert "concurrent" in reason


def test_cooloff_blocks_recent_same_symbol():
    from services.setup_risk import SetupRiskTracker
    rt = SetupRiskTracker(_cfg(), _FakePositionStore())
    # Record a fire at 09:15
    rt.record_admit("NSE:A", "gap_fade_short", _ts("09:15"))
    # At 09:20 (5 min later), cooloff_min=15 → blocked
    ok, reason = rt.can_admit("NSE:A", "gap_fade_short", _ts("09:20"))
    assert not ok
    assert "cooloff" in reason


def test_cooloff_clears_after_window():
    from services.setup_risk import SetupRiskTracker
    rt = SetupRiskTracker(_cfg(), _FakePositionStore())
    rt.record_admit("NSE:A", "gap_fade_short", _ts("09:15"))
    # 16 minutes later, past cooloff_min=15
    ok, _ = rt.can_admit("NSE:A", "gap_fade_short", _ts("09:31"))
    assert ok


def test_rate_limit_blocks_after_max_fires_in_window():
    from services.setup_risk import SetupRiskTracker
    # Raise the concurrency cap so this test isolates rate-limit behaviour.
    rt = SetupRiskTracker(_cfg(max_concurrent=99), _FakePositionStore())
    # 3 fires at 09:15:00 — at cap (max_fires_per_5min=3)
    for sym in ("NSE:A", "NSE:B", "NSE:C"):
        rt.record_admit(sym, "gap_fade_short", _ts("09:15"))
    # 4th fire at 09:17 (within 5-min window) → blocked
    ok, reason = rt.can_admit("NSE:D", "gap_fade_short", _ts("09:17"))
    assert not ok
    assert "rate" in reason


def test_rate_limit_clears_after_window():
    from services.setup_risk import SetupRiskTracker
    rt = SetupRiskTracker(_cfg(max_concurrent=99), _FakePositionStore())
    for sym in ("NSE:A", "NSE:B", "NSE:C"):
        rt.record_admit(sym, "gap_fade_short", _ts("09:15"))
    # 6 minutes later → window slid, OK
    ok, _ = rt.can_admit("NSE:D", "gap_fade_short", _ts("09:21"))
    assert ok


def test_concurrent_admits_same_bar_respect_cap():
    """Regression for the 9-day paper-trade in-flight bypass.

    With max_concurrent_positions=2, three admits at the SAME ts should see
    the 3rd blocked — even though none have landed in position_store yet.
    Before the fix, list_open() returned 0 for each admit and all three got
    through (paper-trade Jun 2026 saw 9-10 concurrent gap_fade_short entries
    at 09:15 with cap=5).
    """
    from services.setup_risk import SetupRiskTracker
    rt = SetupRiskTracker(_cfg(max_concurrent=2), _FakePositionStore())
    # First two admits land within the cap.
    ok1, _ = rt.can_admit("NSE:A", "gap_fade_short", _ts("09:15"))
    assert ok1
    rt.record_admit("NSE:A", "gap_fade_short", _ts("09:15"))
    ok2, _ = rt.can_admit("NSE:B", "gap_fade_short", _ts("09:15"))
    assert ok2
    rt.record_admit("NSE:B", "gap_fade_short", _ts("09:15"))
    # Third admit at the same ts: list_open() is still empty, but the
    # in-flight counter has 2 → cap hit.
    ok3, reason = rt.can_admit("NSE:C", "gap_fade_short", _ts("09:15"))
    assert not ok3
    assert "concurrent" in reason


def test_in_flight_dedupes_after_position_lands():
    """Once an admit shows up in position_store, the in-flight slot must be
    released so a single trade isn't counted twice."""
    from services.setup_risk import SetupRiskTracker
    ps = _FakePositionStore()
    rt = SetupRiskTracker(_cfg(max_concurrent=2), ps)
    rt.record_admit("NSE:A", "gap_fade_short", _ts("09:15"))
    rt.record_admit("NSE:B", "gap_fade_short", _ts("09:15"))
    # Both fills land in the next bar.
    ps.add("NSE:A", "gap_fade_short")
    ps.add("NSE:B", "gap_fade_short")
    # list_open()=2 + in_flight={A,B} would naively count 4. After dedupe
    # it must be 2 — and the cap fires only because of the 2 fills.
    ok, reason = rt.can_admit("NSE:C", "gap_fade_short", _ts("09:20"))
    assert not ok
    assert "concurrent_cap_2/2" in reason


def test_in_flight_expires_via_ttl():
    """A never-filled admit must not permanently block the cap."""
    from services.setup_risk import SetupRiskTracker
    rt = SetupRiskTracker(_cfg(max_concurrent=2), _FakePositionStore())
    rt.record_admit("NSE:A", "gap_fade_short", _ts("09:15"))
    rt.record_admit("NSE:B", "gap_fade_short", _ts("09:15"))
    # Past the 10-min in-flight TTL with no fill → slots freed.
    ok, _ = rt.can_admit("NSE:C", "gap_fade_short", _ts("09:26"))
    assert ok


def test_admit_for_unknown_setup_passes_open():
    """If setup_type isn't in config, treat as no per-setup limits."""
    from services.setup_risk import SetupRiskTracker
    rt = SetupRiskTracker(_cfg(), _FakePositionStore())
    ok, _ = rt.can_admit("NSE:A", "unknown_setup", _ts("09:15"))
    assert ok
