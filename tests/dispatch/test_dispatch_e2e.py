"""End-to-end: feed a synthetic bar stream, assert dispatch matches expected (det, sym) pairs."""
import pandas as pd
import pytest
from datetime import datetime, time
from unittest.mock import MagicMock, patch

from services.dispatch.setup_registry import SetupSpec, SetupRegistry, Trigger
from services.dispatch.transition_calendar import TransitionCalendar
from services.dispatch.tag_map import TagMap
from services.dispatch.planner import DispatchPlanner


def test_e2e_morning_dispatch_pattern():
    """At 09:30, only delivery_pct (opening) should be active.

    Calendar sorts open before close at same minute, so gap_fade
    (which closes at 09:30) is no longer active by end of walk.
    """
    reg = SetupRegistry({})
    reg._specs = {
        "gap_fade_short": SetupSpec(
            "gap_fade_short", True,
            "structures.gap_fade_short_structure.GapFadeShortStructure",
            "services.setup_universe.gap_fade_universe",
            Trigger.bar(time(9, 15)),
            (time(9, 15), time(9, 30)),
            {},
        ),
        "delivery_pct_anomaly_short": SetupSpec(
            "delivery_pct_anomaly_short", True,
            "structures.delivery_pct_anomaly_short_structure.DeliveryPctAnomalyShortStructure",
            "services.setup_universe.delivery_pct_universe",
            Trigger.session_start(),
            (time(9, 30), time(10, 30)),
            {},
        ),
    }
    cal = TransitionCalendar.from_registry(reg)
    tm = TagMap()

    # Simulate walking from session-open through to 09:30 bar
    for ev in cal.events_in(after=time(0, 0), until=time(9, 30)):
        if ev.kind == "build_universe":
            syms = {"NSE:GAP1", "NSE:GAP2"} if ev.setup == "gap_fade_short" else {"NSE:DEL1"}
            tm.add_universe(ev.setup, syms)
        elif ev.kind == "open_window":
            tm.open_window(ev.setup)
        elif ev.kind == "close_window":
            tm.close_window(ev.setup)

    active = tm.active_symbols()
    # gap_fade_short closes at 09:30, delivery_pct opens at 09:30.
    # Calendar sorts open before close at same minute, BUT the walk applies
    # in time order: gap_fade close fires after gap_fade open and after
    # delivery_pct open. So at the end of the walk:
    assert "NSE:GAP1" not in active
    assert "NSE:DEL1" in active


def test_e2e_idle_bar_skips_scan():
    """At 10:35 (between or_window_failure close and round_number open), zero active."""
    reg = SetupRegistry({})
    reg._specs = {
        "or_window_failure_fade_short": SetupSpec(
            "or_window_failure_fade_short", True,
            "x.Y", "x.y", Trigger.session_start(),
            (time(9, 30), time(10, 30)), {},
        ),
        "round_number_sweep_short": SetupSpec(
            "round_number_sweep_short", True,
            "x.Y", "x.y", Trigger.session_start(),
            (time(11, 0), time(12, 30)), {},
        ),
    }
    cal = TransitionCalendar.from_registry(reg)
    tm = TagMap()
    for ev in cal.events_in(after=time(0, 0), until=time(10, 35)):
        if ev.kind == "build_universe":
            tm.add_universe(ev.setup, {"NSE:A"})
        elif ev.kind == "open_window":
            tm.open_window(ev.setup)
        elif ev.kind == "close_window":
            tm.close_window(ev.setup)

    # or_window_failure closed at 10:30. round_number opens at 11:00. At 10:35: idle.
    assert tm.active_symbols() == set()
