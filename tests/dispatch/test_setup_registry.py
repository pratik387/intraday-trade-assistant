import pytest
from datetime import time
from services.dispatch.setup_registry import Trigger, parse_trigger, SetupSpec


def test_parse_session_start_trigger():
    t = parse_trigger("session_start")
    assert t == Trigger.session_start()


def test_parse_bar_trigger():
    t = parse_trigger("bar:09:15")
    assert t.kind == "bar"
    assert t.at == time(9, 15)


def test_parse_bar_trigger_malformed_raises():
    with pytest.raises(ValueError, match="malformed"):
        parse_trigger("bar:9:15")  # missing leading zero
    with pytest.raises(ValueError, match="malformed"):
        parse_trigger("bar:09")
    with pytest.raises(ValueError, match="malformed"):
        parse_trigger("foo")


def test_setup_spec_creation():
    spec = SetupSpec(
        name="gap_fade_short",
        enabled=True,
        detector_class_path="structures.gap_fade_short_structure.GapFadeShortStructure",
        universe_builder_path="services.setup_universe.gap_fade_universe",
        universe_trigger=Trigger.bar(time(9, 15)),
        active_window=(time(9, 15), time(9, 30)),
        raw_config={"foo": "bar"},
    )
    assert spec.name == "gap_fade_short"
    assert spec.enabled is True
    assert spec.active_window == (time(9, 15), time(9, 30))


def test_setup_spec_rejects_inverted_window():
    with pytest.raises(ValueError, match="active_window_start <= active_window_end"):
        SetupSpec(
            name="bad", enabled=True,
            detector_class_path="x.Y",
            universe_builder_path="x.y",
            universe_trigger=Trigger.session_start(),
            active_window=(time(10, 0), time(9, 0)),  # inverted
            raw_config={},
        )
