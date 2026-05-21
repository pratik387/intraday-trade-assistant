"""Tests for SetupSpec.mode field and SetupRegistry.get_active_setups."""
import pytest
from services.dispatch.setup_registry import SetupRegistry, SetupSpec, Trigger
from datetime import time


def _minimal_setup_config(**overrides):
    base = {
        "enabled": True,
        "detector_class": "structures.gap_fade_short_structure.GapFadeShortStructure",
        "universe_builder": "services.setup_universe.gap_fade_universe",
        "universe_trigger": "session_start",
        "active_window_start": "09:15",
        "active_window_end": "10:30",
    }
    base.update(overrides)
    return base


def test_mode_defaults_to_intraday_when_not_specified():
    cfg = {"setups": {"x": _minimal_setup_config()}}
    reg = SetupRegistry.load_from_config(cfg)
    assert reg.get("x").mode == "intraday"


def test_mode_overnight_loads_correctly():
    cfg = {"setups": {"x": _minimal_setup_config(mode="overnight")}}
    reg = SetupRegistry.load_from_config(cfg)
    assert reg.get("x").mode == "overnight"


def test_invalid_mode_raises():
    cfg = {"setups": {"x": _minimal_setup_config(mode="hybrid")}}
    with pytest.raises(ValueError, match="mode must be"):
        SetupRegistry.load_from_config(cfg)


def test_get_active_setups_filters_by_mode():
    cfg = {"setups": {
        "intra1": _minimal_setup_config(mode="intraday"),
        "intra2": _minimal_setup_config(mode="intraday"),
        "over1": _minimal_setup_config(mode="overnight"),
    }}
    reg = SetupRegistry.load_from_config(cfg)
    intra = reg.get_active_setups("intraday")
    over = reg.get_active_setups("overnight")
    assert len(intra) == 2
    assert len(over) == 1
    assert over[0].name == "over1"


def test_get_active_setups_excludes_disabled():
    cfg = {"setups": {
        "intra_on": _minimal_setup_config(mode="intraday", enabled=True),
        "intra_off": _minimal_setup_config(mode="intraday", enabled=False),
    }}
    reg = SetupRegistry.load_from_config(cfg)
    intra = reg.get_active_setups("intraday")
    assert len(intra) == 1
    assert intra[0].name == "intra_on"


def test_get_active_setups_invalid_mode_raises():
    reg = SetupRegistry.load_from_config({"setups": {}})
    with pytest.raises(ValueError, match="mode must be"):
        reg.get_active_setups("hybrid")
