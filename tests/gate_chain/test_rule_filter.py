"""RuleFilterGate tests."""
import json
from pathlib import Path

import pytest

from services.gate_chain.rule_filter import RuleFilterGate


# Test fixture: tiny survivors JSON with 3 known rules
@pytest.fixture
def tiny_survivors_file(tmp_path):
    p = tmp_path / "survivors.json"
    p.write_text(json.dumps({
        "survivors": [
            {"setup": "premium_zone_short", "rule_id": "premium_zone_short__cap_segment+hour_bucket=small_cap+morning"},
            {"setup": "range_bounce_short", "rule_id": "range_bounce_short__regime=chop"},
            {"setup": "order_block_short", "rule_id": "order_block_short__cap_segment+hour_bucket=large_cap+opening"},
        ]
    }), encoding="utf-8")
    return p


def test_loads_rules_from_survivors_json(tiny_survivors_file):
    gate = RuleFilterGate(tiny_survivors_file)
    # Internal set should have 3 entries
    assert len(gate._survivor_set) == 3


def test_admits_candidate_matching_a_rule(tiny_survivors_file):
    gate = RuleFilterGate(tiny_survivors_file)
    cand = {
        "setup_type": "premium_zone_short",
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
        "regime": "trend_up",  # irrelevant for this rule
    }
    ok, reason = gate.evaluate(cand)
    assert ok is True
    assert reason == "matched_rule"


def test_admits_single_dim_rule(tiny_survivors_file):
    """range_bounce_short__regime=chop matches any cap/hour as long as regime=chop."""
    gate = RuleFilterGate(tiny_survivors_file)
    cand = {
        "setup_type": "range_bounce_short",
        "cap_segment": "mid_cap",
        "hour_bucket": "afternoon",
        "regime": "chop",
    }
    ok, _ = gate.evaluate(cand)
    assert ok is True


def test_rejects_unknown_setup(tiny_survivors_file):
    gate = RuleFilterGate(tiny_survivors_file)
    cand = {
        "setup_type": "vwap_lose_short",  # not in any rule
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
        "regime": "chop",
    }
    ok, reason = gate.evaluate(cand)
    assert ok is False
    assert "no_matching" in reason


def test_rejects_when_conditioner_value_doesnt_match(tiny_survivors_file):
    gate = RuleFilterGate(tiny_survivors_file)
    # Setup is in a rule, but cap_segment doesn't match
    cand = {
        "setup_type": "premium_zone_short",
        "cap_segment": "large_cap",  # rule wants small_cap
        "hour_bucket": "morning",
        "regime": "chop",
    }
    ok, _ = gate.evaluate(cand)
    assert ok is False


def test_rejects_when_conditioner_key_missing(tiny_survivors_file):
    """If a candidate is missing the conditioner key the rule references, reject (don't crash)."""
    gate = RuleFilterGate(tiny_survivors_file)
    cand = {
        "setup_type": "premium_zone_short",
        # cap_segment missing
        "hour_bucket": "morning",
    }
    ok, _ = gate.evaluate(cand)
    assert ok is False


def test_loads_real_74_survivors():
    """Smoke-test against the actual production survivors JSON."""
    real_path = Path("analysis/edge_discovery_runs/2026-04-22-validation-gate/stage6_validation_survivors.json")
    if not real_path.exists():
        pytest.skip("survivors json not present (CI env)")
    gate = RuleFilterGate(real_path)
    assert len(gate._survivor_set) == 74
