import pytest
from datetime import time
from services.dispatch.setup_registry import SetupSpec, Trigger, SetupRegistry
from services.dispatch.transition_calendar import TransitionCalendar, TransitionEvent


def _mk_spec(name, trigger, win_start, win_end):
    return SetupSpec(
        name=name, enabled=True,
        detector_class_path="x.Y", universe_builder_path="x.y",
        universe_trigger=trigger,
        active_window=(win_start, win_end),
        raw_config={},
    )


def _mk_registry(*specs):
    reg = SetupRegistry({})
    reg._specs = {s.name: s for s in specs}
    return reg


def test_events_sorted_by_time():
    reg = _mk_registry(
        _mk_spec("gap_fade_short", Trigger.bar(time(9, 15)), time(9, 15), time(9, 30)),
        _mk_spec("c8", Trigger.session_start(), time(14, 30), time(15, 0)),
    )
    cal = TransitionCalendar.from_registry(reg)
    times = [ev.at for ev in cal.all_events()]
    assert times == sorted(times)


def test_session_start_trigger_emits_build_at_market_open():
    reg = _mk_registry(_mk_spec("c8", Trigger.session_start(), time(14, 30), time(15, 0)))
    cal = TransitionCalendar.from_registry(reg)
    builds = [ev for ev in cal.all_events() if ev.kind == "build_universe"]
    assert len(builds) == 1
    assert builds[0].at == time(9, 15)
    assert builds[0].setup == "c8"


def test_bar_trigger_emits_build_at_that_bar():
    reg = _mk_registry(_mk_spec("gap_fade_short", Trigger.bar(time(9, 15)), time(9, 15), time(9, 30)))
    cal = TransitionCalendar.from_registry(reg)
    builds = [ev for ev in cal.all_events() if ev.kind == "build_universe"]
    assert len(builds) == 1
    assert builds[0].at == time(9, 15)


def test_each_setup_emits_open_and_close_window():
    reg = _mk_registry(_mk_spec("gap_fade_short", Trigger.bar(time(9, 15)), time(9, 15), time(9, 30)))
    cal = TransitionCalendar.from_registry(reg)
    kinds = [(ev.at, ev.kind) for ev in cal.all_events()]
    assert (time(9, 15), "open_window") in kinds
    assert (time(9, 30), "close_window") in kinds


def test_events_in_range_inclusive_exclusive():
    """events_in(after, until) returns events where after < ev.at <= until."""
    reg = _mk_registry(_mk_spec("gap_fade_short", Trigger.bar(time(9, 15)), time(9, 15), time(9, 30)))
    cal = TransitionCalendar.from_registry(reg)
    events = cal.events_in(after=time(9, 10), until=time(9, 15))
    kinds = [ev.kind for ev in events]
    assert "build_universe" in kinds
    assert "open_window" in kinds
    events_late = cal.events_in(after=time(9, 15), until=time(9, 30))
    kinds_late = [ev.kind for ev in events_late]
    assert "close_window" in kinds_late
    assert "open_window" not in kinds_late


def test_close_strictly_after_open_at_same_minute():
    """One-shot windows: open must come before close at same minute."""
    reg = _mk_registry(_mk_spec("circuit_t1", Trigger.session_start(), time(10, 30), time(10, 30)))
    cal = TransitionCalendar.from_registry(reg)
    same_minute = [ev for ev in cal.all_events() if ev.at == time(10, 30)]
    kinds = [ev.kind for ev in same_minute]
    assert kinds.index("open_window") < kinds.index("close_window")
