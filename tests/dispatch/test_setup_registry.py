import pytest
from datetime import time
from services.dispatch.setup_registry import Trigger, parse_trigger


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
