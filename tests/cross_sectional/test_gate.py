"""Tests for CrossSectionalGate — F1 + F2 composition + config-driven logic."""
from datetime import datetime
from dataclasses import dataclass

import pytest

from services.cross_sectional.gate import CrossSectionalGate, Candidate


CFG_BASE = {
    "enabled": True,
    "f1_rvol_enabled": True,
    "f1_rvol_threshold_pct": 70.0,
    "f1_applicable_caps": ["small_cap", "mid_cap", "micro_cap"],
    "f1_skip_hour_buckets": ["late"],
    "f1_min_history_sessions": 3,
    "f1_rolling_window_sessions": 20,
    "f2_crowdedness_enabled": True,
    "f2_crowdedness_threshold": 40,
    "f2_crowdedness_window_min": 5,
}


class _FakeRVOL:
    def __init__(self, pct_by_symbol):
        self.pct_by_symbol = pct_by_symbol

    def get_rvol_pct_tier(self, symbol, ts):
        return self.pct_by_symbol.get(symbol)


class _FakeCrowd:
    def __init__(self, count_by_setup):
        self.count_by_setup = count_by_setup

    def count(self, setup_type, ts):
        return self.count_by_setup.get(setup_type, 0)

    def record(self, setup_type, ts):
        pass


def _candidate(symbol="SYM", setup="premium_zone_short", cap="small_cap",
               hb="morning", ts=None):
    return Candidate(
        symbol=symbol,
        setup_type=setup,
        cap_segment=cap,
        hour_bucket=hb,
        decision_ts=ts or datetime(2026, 4, 21, 11, 0),
    )


def test_gate_disabled_allows_everything():
    cfg = {**CFG_BASE, "enabled": False}
    gate = CrossSectionalGate(cfg, rvol=_FakeRVOL({}), crowdedness=_FakeCrowd({}))
    ok, reason = gate.evaluate(_candidate())
    assert ok is True
    assert "disabled" in reason.lower()


def test_f1_rejects_high_rvol_in_applicable_cap():
    """small_cap + morning + rvol_pct=80 → reject (top 30%)."""
    gate = CrossSectionalGate(
        CFG_BASE,
        rvol=_FakeRVOL({"SYM": 80.0}),
        crowdedness=_FakeCrowd({}),
    )
    ok, reason = gate.evaluate(_candidate(cap="small_cap", hb="morning"))
    assert ok is False
    assert "f1_rvol" in reason.lower()


def test_f1_skips_for_unknown_cap():
    """unknown cap → F1 doesn't apply regardless of rvol."""
    gate = CrossSectionalGate(
        CFG_BASE,
        rvol=_FakeRVOL({"SYM": 99.0}),
        crowdedness=_FakeCrowd({}),
    )
    ok, reason = gate.evaluate(_candidate(cap="unknown", hb="morning"))
    assert ok is True


def test_f1_skips_for_large_cap():
    gate = CrossSectionalGate(
        CFG_BASE,
        rvol=_FakeRVOL({"SYM": 99.0}),
        crowdedness=_FakeCrowd({}),
    )
    ok, _ = gate.evaluate(_candidate(cap="large_cap", hb="morning"))
    assert ok is True


def test_f1_skips_for_late_hour():
    """late hour → F1 skipped even in applicable cap."""
    gate = CrossSectionalGate(
        CFG_BASE,
        rvol=_FakeRVOL({"SYM": 99.0}),
        crowdedness=_FakeCrowd({}),
    )
    ok, _ = gate.evaluate(_candidate(cap="small_cap", hb="late"))
    assert ok is True


def test_f1_allows_below_threshold():
    gate = CrossSectionalGate(
        CFG_BASE,
        rvol=_FakeRVOL({"SYM": 50.0}),
        crowdedness=_FakeCrowd({}),
    )
    ok, _ = gate.evaluate(_candidate(cap="small_cap", hb="morning"))
    assert ok is True


def test_f1_allows_when_rvol_pct_unavailable():
    """Insufficient history → no rvol_pct → allow (don't block on missing data)."""
    gate = CrossSectionalGate(
        CFG_BASE,
        rvol=_FakeRVOL({}),  # SYM not in map
        crowdedness=_FakeCrowd({}),
    )
    ok, _ = gate.evaluate(_candidate(cap="small_cap", hb="morning"))
    assert ok is True


def test_f2_rejects_high_crowdedness():
    gate = CrossSectionalGate(
        CFG_BASE,
        rvol=_FakeRVOL({}),
        crowdedness=_FakeCrowd({"premium_zone_short": 50}),  # > 40 threshold
    )
    ok, reason = gate.evaluate(_candidate(setup="premium_zone_short"))
    assert ok is False
    assert "f2_crowded" in reason.lower()


def test_f2_applies_universally_including_unknown_cap():
    gate = CrossSectionalGate(
        CFG_BASE,
        rvol=_FakeRVOL({}),
        crowdedness=_FakeCrowd({"premium_zone_short": 50}),
    )
    ok, reason = gate.evaluate(_candidate(cap="unknown", setup="premium_zone_short"))
    assert ok is False
    assert "f2_crowded" in reason.lower()


def test_f2_allows_below_threshold():
    gate = CrossSectionalGate(
        CFG_BASE,
        rvol=_FakeRVOL({}),
        crowdedness=_FakeCrowd({"premium_zone_short": 10}),
    )
    ok, _ = gate.evaluate(_candidate(setup="premium_zone_short"))
    assert ok is True


def test_f2_disabled_allows_crowded():
    cfg = {**CFG_BASE, "f2_crowdedness_enabled": False}
    gate = CrossSectionalGate(
        cfg,
        rvol=_FakeRVOL({}),
        crowdedness=_FakeCrowd({"premium_zone_short": 100}),
    )
    ok, _ = gate.evaluate(_candidate(setup="premium_zone_short"))
    assert ok is True


def test_reject_reason_reports_both_failures():
    """If both F1 and F2 fail, reason lists both."""
    gate = CrossSectionalGate(
        CFG_BASE,
        rvol=_FakeRVOL({"SYM": 90.0}),
        crowdedness=_FakeCrowd({"premium_zone_short": 50}),
    )
    ok, reason = gate.evaluate(_candidate(cap="small_cap", hb="morning", setup="premium_zone_short"))
    assert ok is False
    assert "f1_rvol" in reason.lower()
    assert "f2_crowded" in reason.lower()


def test_candidate_dataclass_fields():
    """Candidate dataclass has expected fields."""
    c = Candidate(
        symbol="NSE:ACI",
        setup_type="premium_zone_short",
        cap_segment="small_cap",
        hour_bucket="morning",
        decision_ts=datetime(2026, 4, 21, 11, 0),
    )
    assert c.symbol == "NSE:ACI"
    assert c.setup_type == "premium_zone_short"
