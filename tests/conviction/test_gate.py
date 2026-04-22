"""ConvictionGate tests — online top-N + threshold + session boundary reset."""
from datetime import date, datetime

import pytest

from services.conviction.gate import ConvictionGate


CFG = {
    "enabled": True,
    "daily_cap": 5,
    "min_predicted_r": 0.3,
}


def _candidate(symbol="SYM", ts=None, session=None):
    return {
        "symbol": symbol,
        "decision_ts": ts or datetime(2026, 4, 22, 10, 0),
        "session_date": session or date(2026, 4, 22),
    }


def test_disabled_allows_everything():
    gate = ConvictionGate({**CFG, "enabled": False})
    ok, reason = gate.evaluate(_candidate(), predicted_r=-1.0)
    assert ok is True


def test_admit_when_above_threshold_and_under_cap():
    gate = ConvictionGate(CFG)
    ok, _ = gate.evaluate(_candidate(), predicted_r=0.5)
    assert ok is True


def test_reject_when_below_threshold():
    gate = ConvictionGate(CFG)
    ok, reason = gate.evaluate(_candidate(), predicted_r=0.2)
    assert ok is False
    assert "threshold" in reason.lower()


def test_reject_when_daily_cap_reached():
    gate = ConvictionGate(CFG)
    # Admit 5 candidates — fills cap
    for i in range(5):
        ok, _ = gate.evaluate(_candidate(symbol=f"S{i}"), predicted_r=0.5)
        assert ok is True
    # 6th should be rejected
    ok, reason = gate.evaluate(_candidate(symbol="S6"), predicted_r=0.9)
    assert ok is False
    assert "cap" in reason.lower()


def test_session_boundary_resets_counter():
    gate = ConvictionGate(CFG)
    # Fill cap on day 1
    for i in range(5):
        gate.evaluate(_candidate(symbol=f"S{i}", session=date(2026, 4, 22)), 0.5)
    # Day 2 — fresh start
    ok, _ = gate.evaluate(_candidate(symbol="S6", session=date(2026, 4, 23)), predicted_r=0.5)
    assert ok is True


def test_only_admitted_trades_count_toward_cap():
    """Rejected-below-threshold trades don't consume cap slots."""
    gate = ConvictionGate(CFG)
    # Reject 3 low-conviction trades
    for i in range(3):
        gate.evaluate(_candidate(symbol=f"low{i}"), predicted_r=0.1)
    # Admit 5 high-conviction (cap is 5)
    for i in range(5):
        ok, _ = gate.evaluate(_candidate(symbol=f"hi{i}"), predicted_r=0.5)
        assert ok is True
    # 6th high-conviction: rejected (cap reached)
    ok, _ = gate.evaluate(_candidate(symbol="hi6"), predicted_r=0.5)
    assert ok is False


def test_gate_stats_reports_admitted_and_rejected():
    gate = ConvictionGate(CFG)
    gate.evaluate(_candidate(symbol="a"), 0.5)
    gate.evaluate(_candidate(symbol="b"), 0.1)
    stats = gate.stats()
    assert stats["admitted"] == 1
    assert stats["rejected"] == 1
