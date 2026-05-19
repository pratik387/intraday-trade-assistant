"""Tests for plan_orchestrator._setup_should_fire helper."""
from services.plan_orchestrator import _setup_should_fire


def test_setup_disabled_does_not_fire():
    block = {"enabled": True, "cb_state": "disabled"}
    should, mult = _setup_should_fire(block)
    assert should is False
    assert mult == 0.0


def test_setup_forward_validation_fires_at_quarter_size():
    block = {"enabled": True, "cb_state": "forward_validation",
             "position_size_multiplier": 0.25}
    should, mult = _setup_should_fire(block)
    assert should is True
    assert mult == 0.25


def test_setup_forward_validation_default_multiplier_is_quarter():
    """If cb_state=forward_validation but no multiplier set, default to 0.25."""
    block = {"enabled": True, "cb_state": "forward_validation"}
    should, mult = _setup_should_fire(block)
    assert should is True
    assert mult == 0.25


def test_setup_enabled_fires_at_full_size():
    block = {"enabled": True, "cb_state": "enabled"}
    should, mult = _setup_should_fire(block)
    assert should is True
    assert mult == 1.0


def test_setup_with_enabled_false_does_not_fire_regardless_of_cb_state():
    """`enabled: false` always wins."""
    block = {"enabled": False, "cb_state": "enabled"}
    should, mult = _setup_should_fire(block)
    assert should is False
    assert mult == 0.0


def test_setup_default_cb_state_is_enabled():
    """Missing cb_state defaults to enabled (backward compatible)."""
    block = {"enabled": True}  # no cb_state
    should, mult = _setup_should_fire(block)
    assert should is True
    assert mult == 1.0
