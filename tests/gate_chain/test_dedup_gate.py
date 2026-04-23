"""DedupGate tests — cooloff + setup-change + score-beats-max(pctl,last) + session reset."""
from datetime import date, datetime, timedelta

import pytest

from services.gate_chain.dedup_gate import DedupGate


CFG = {
    "enabled": True,
    "cooloff_bars": 6,
    "require_setup_change": True,
}


def _ts(hour: int = 10, minute: int = 0) -> datetime:
    """Helper: return an IST-naive datetime on 2026-04-22."""
    return datetime(2026, 4, 22, hour, minute)


def test_disabled_gate_passes_all():
    """cfg.enabled=False → always (True, 'gate_disabled'), no state change."""
    gate = DedupGate({**CFG, "enabled": False})
    ok, reason = gate.evaluate(
        sym="SBIN",
        now_ts=_ts(10, 0),
        setup_type="premium_zone_short",
        score=0.5,
        pctl_score=0.9,
    )
    assert ok is True
    assert reason == "gate_disabled"
    # No state recorded when disabled
    assert gate._last_entry == {}


def test_first_admit_allowed():
    """No prior entry on the symbol → admit; state recorded."""
    gate = DedupGate(CFG)
    ok, reason = gate.evaluate(
        sym="SBIN",
        now_ts=_ts(10, 0),
        setup_type="premium_zone_short",
        score=0.4,
        pctl_score=0.9,
    )
    assert ok is True
    assert reason == "admitted"
    # State updated
    assert "SBIN" in gate._last_entry
    assert gate._last_entry["SBIN"]["setup"] == "premium_zone_short"
    assert gate._last_entry["SBIN"]["score"] == 0.4
    assert gate._last_entry["SBIN"]["ts"] == _ts(10, 0)


def test_cooloff_blocks_before_window():
    """Second admit on same symbol within cooloff_bars → reject with 'cooloff'."""
    gate = DedupGate(CFG)
    # First admit at 10:00
    ok1, _ = gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 0), setup_type="premium_zone_short",
        score=0.5, pctl_score=0.4,
    )
    assert ok1 is True

    # Second attempt 10 min later = 2 bars — under cooloff_bars=6
    ok2, reason2 = gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 10), setup_type="range_bounce_short",
        score=0.9, pctl_score=0.4,
    )
    assert ok2 is False
    assert "cooloff" in reason2.lower()
    # Prior state not overwritten
    assert gate._last_entry["SBIN"]["score"] == 0.5
    assert gate._last_entry["SBIN"]["setup"] == "premium_zone_short"


def test_cooloff_passes_after_window():
    """Second admit after cooloff_bars → proceeds to further checks (setup test)."""
    gate = DedupGate(CFG)
    # First admit at 10:00
    gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 0), setup_type="premium_zone_short",
        score=0.5, pctl_score=0.4,
    )
    # Second attempt 35 min later = 7 bars — beyond cooloff. Same setup so it
    # still rejects here (setup_unchanged), but NOT with 'cooloff'.
    ok, reason = gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 35), setup_type="premium_zone_short",
        score=0.9, pctl_score=0.4,
    )
    assert ok is False
    assert "cooloff" not in reason.lower()
    assert "setup_unchanged" in reason


def test_setup_change_required():
    """require_setup_change=True + same setup_type → reject 'setup_unchanged'."""
    gate = DedupGate(CFG)
    gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 0), setup_type="premium_zone_short",
        score=0.5, pctl_score=0.4,
    )
    # After cooloff, SAME setup → rejected
    ok, reason = gate.evaluate(
        sym="SBIN", now_ts=_ts(11, 0), setup_type="premium_zone_short",
        score=0.9, pctl_score=0.4,
    )
    assert ok is False
    assert "setup_unchanged" in reason


def test_setup_change_different_setup():
    """require_setup_change=True + different setup + score passes → admit."""
    gate = DedupGate(CFG)
    gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 0), setup_type="premium_zone_short",
        score=0.5, pctl_score=0.4,
    )
    # After cooloff, DIFFERENT setup, score beats max(pctl=0.4, last=0.5) = 0.5
    ok, reason = gate.evaluate(
        sym="SBIN", now_ts=_ts(11, 0), setup_type="range_bounce_short",
        score=0.6, pctl_score=0.4,
    )
    assert ok is True
    assert reason == "admitted"
    # State updated with new setup + score
    assert gate._last_entry["SBIN"]["setup"] == "range_bounce_short"
    assert gate._last_entry["SBIN"]["score"] == 0.6
    assert gate._last_entry["SBIN"]["ts"] == _ts(11, 0)


def test_score_must_beat_max_pctl_last():
    """Second admit with score < max(pctl, last) → reject 'score_weak'."""
    gate = DedupGate(CFG)
    # Record a prior entry with last_score=0.7
    gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 0), setup_type="premium_zone_short",
        score=0.7, pctl_score=0.3,
    )
    # After cooloff, different setup so setup-check passes. But score=0.5 < max(pctl=0.4, last=0.7)=0.7
    ok, reason = gate.evaluate(
        sym="SBIN", now_ts=_ts(11, 0), setup_type="range_bounce_short",
        score=0.5, pctl_score=0.4,
    )
    assert ok is False
    assert "score_weak" in reason


def test_score_beats_requirement():
    """score >= max(pctl, last) → admit."""
    gate = DedupGate(CFG)
    gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 0), setup_type="premium_zone_short",
        score=0.3, pctl_score=0.2,
    )
    # After cooloff, different setup, score=0.6 >= max(pctl=0.5, last=0.3)=0.5
    ok, reason = gate.evaluate(
        sym="SBIN", now_ts=_ts(11, 0), setup_type="range_bounce_short",
        score=0.6, pctl_score=0.5,
    )
    assert ok is True
    assert reason == "admitted"


def test_session_reset_clears_state():
    """New session_date → _last_entry cleared; previously-blocked symbol is allowed again."""
    gate = DedupGate(CFG)
    # Day 1 admit
    gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 0), setup_type="premium_zone_short",
        score=0.5, pctl_score=0.4,
        session_date=date(2026, 4, 22),
    )
    assert "SBIN" in gate._last_entry

    # Day 2 — same symbol, same setup, same (small) score. Would be blocked in
    # same-session. Must be admitted here because state resets.
    ok, reason = gate.evaluate(
        sym="SBIN", now_ts=datetime(2026, 4, 23, 10, 0), setup_type="premium_zone_short",
        score=0.1, pctl_score=0.05,
        session_date=date(2026, 4, 23),
    )
    assert ok is True
    assert reason == "admitted"
    # New state (only SBIN from day 2)
    assert gate._last_entry["SBIN"]["ts"] == datetime(2026, 4, 23, 10, 0)
    assert gate._current_session == date(2026, 4, 23)


def test_stats_tracking():
    """admit/reject counters increment correctly across a mix of outcomes."""
    gate = DedupGate(CFG)
    # 1 admit (first entry)
    gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 0), setup_type="premium_zone_short",
        score=0.5, pctl_score=0.4,
    )
    # 1 reject (cooloff)
    gate.evaluate(
        sym="SBIN", now_ts=_ts(10, 10), setup_type="range_bounce_short",
        score=0.9, pctl_score=0.4,
    )
    # 1 reject (setup_unchanged after cooloff)
    gate.evaluate(
        sym="SBIN", now_ts=_ts(11, 0), setup_type="premium_zone_short",
        score=0.9, pctl_score=0.4,
    )
    # 1 admit (different symbol, first entry)
    gate.evaluate(
        sym="RELIANCE", now_ts=_ts(10, 5), setup_type="premium_zone_short",
        score=0.5, pctl_score=0.4,
    )
    stats = gate.stats()
    assert stats["admitted"] == 2
    assert stats["rejected"] == 2
