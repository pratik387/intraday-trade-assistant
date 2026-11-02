"""immediate_execution skips the entry-zone wait in the executor (2026-11-02).

The flag used to be read only by trigger_validation_engine; the executor's tick,
poll and 1m paths bypassed the zone only for wide_open, so an immediate plan still
waited (and often expired) in its +/-0.3% zone.
"""
from unittest.mock import patch

import pytest

import services.config_loader as cl


def _cfg(**setup):
    return {"wide_open_mode": False, "setups": {"s": setup}}


@pytest.mark.parametrize("setup, skipped", [
    ({}, False),
    ({"immediate_execution": True}, True),
    ({"immediate_execution": False}, False),
    ({"wide_open": True}, True),
])
def test_entry_zone_skipped_for_setup(setup, skipped):
    with patch("config.filters_setup.load_filters", return_value=_cfg(**setup)):
        assert cl.entry_zone_skipped_for_setup("s") is skipped


def test_immediate_execution_does_not_switch_on_wide_open():
    with patch("config.filters_setup.load_filters", return_value=_cfg(immediate_execution=True)):
        assert cl.is_wide_open_for_setup("s") is False      # duplicate guard / min_bars untouched


def test_1m_zone_check_lets_an_immediate_plan_through_far_from_the_zone():
    from services.execution.trigger_aware_executor import TriggerAwareExecutor
    with patch.object(TriggerAwareExecutor, "__init__", lambda self: None):
        ex = TriggerAwareExecutor()
    zone = [100.0, 100.6]
    with patch("services.config_loader.entry_zone_skipped_for_setup", return_value=True):
        assert ex._is_price_in_entry_zone(98.0, zone, "short", setup_type="s") is True
    with patch("services.config_loader.entry_zone_skipped_for_setup", return_value=False):
        assert ex._is_price_in_entry_zone(98.0, zone, "short", setup_type="s") is False


def test_only_the_up_spike_family_uses_immediate_execution():
    import io, json
    from pathlib import Path
    cfg = json.load(io.open(Path(__file__).resolve().parents[3] / "config" / "configuration.json",
                            encoding="utf-8"))
    on = [n for n, r in cfg["setups"].items()
          if isinstance(r, dict) and r.get("enabled") and r.get("immediate_execution")]
    # up_spike since 2026-11-02 (unified replay); its results-day cells (paper,
    # 2026-11-03) were validated with the same market-at-decision execution.
    assert on == ["up_spike_fade_short", "results_breadth_day_fade_short", "results_breadth_late_fade_short"]
