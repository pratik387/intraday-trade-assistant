"""Tests for CrowdednessCounter — backward-only 5-min sliding window per setup_type."""
from datetime import datetime, timedelta

import pytest

from services.cross_sectional.crowdedness_counter import CrowdednessCounter


def test_empty_counter_returns_zero():
    c = CrowdednessCounter(window_min=5)
    count = c.count("premium_zone_short", datetime(2026, 4, 21, 10, 0))
    assert count == 0


def test_single_event_within_window_counted():
    c = CrowdednessCounter(window_min=5)
    t0 = datetime(2026, 4, 21, 10, 0)
    c.record("premium_zone_short", t0)
    count = c.count("premium_zone_short", t0 + timedelta(minutes=3))
    assert count == 1


def test_event_outside_window_not_counted():
    c = CrowdednessCounter(window_min=5)
    t0 = datetime(2026, 4, 21, 10, 0)
    c.record("premium_zone_short", t0)
    count = c.count("premium_zone_short", t0 + timedelta(minutes=6))
    assert count == 0


def test_future_events_not_counted_in_query_at_earlier_time():
    """Backward-only: event at t=5 is NOT counted when queried at t=0."""
    c = CrowdednessCounter(window_min=5)
    c.record("premium_zone_short", datetime(2026, 4, 21, 10, 5))
    count = c.count("premium_zone_short", datetime(2026, 4, 21, 10, 0))
    assert count == 0


def test_different_setup_types_counted_separately():
    c = CrowdednessCounter(window_min=5)
    t0 = datetime(2026, 4, 21, 10, 0)
    c.record("premium_zone_short", t0)
    c.record("range_bounce_short", t0)
    assert c.count("premium_zone_short", t0) == 1
    assert c.count("range_bounce_short", t0) == 1
    assert c.count("order_block_short", t0) == 0


def test_multiple_events_same_setup_counted():
    c = CrowdednessCounter(window_min=5)
    t0 = datetime(2026, 4, 21, 10, 0)
    for i in range(5):
        c.record("premium_zone_short", t0 + timedelta(minutes=i))
    # Query at t=4: events at 0,1,2,3,4 are all in [-5,0] window => 5
    assert c.count("premium_zone_short", t0 + timedelta(minutes=4)) == 5


def test_boundary_exclusive_on_past_edge():
    """Event exactly at t-5min IS included (inclusive past boundary).
    Event at t-5:01 is NOT included."""
    c = CrowdednessCounter(window_min=5)
    t_query = datetime(2026, 4, 21, 10, 10)
    c.record("s", t_query - timedelta(minutes=5))  # exactly at t-5
    c.record("s", t_query - timedelta(minutes=5, seconds=1))  # just before
    assert c.count("s", t_query) == 1


def test_prune_discards_old_events_on_record():
    """Implementation detail: events older than 2x window are pruned to
    keep memory bounded. Visible via inspecting internal state."""
    c = CrowdednessCounter(window_min=5)
    t0 = datetime(2026, 4, 21, 10, 0)
    c.record("s", t0)
    c.record("s", t0 + timedelta(minutes=20))  # triggers prune of t0
    # After prune, only the recent event remains
    assert len(c._events["s"]) == 1


def test_reset_clears_all_state():
    c = CrowdednessCounter(window_min=5)
    c.record("s1", datetime(2026, 4, 21, 10, 0))
    c.record("s2", datetime(2026, 4, 21, 10, 0))
    c.reset()
    assert c.count("s1", datetime(2026, 4, 21, 10, 1)) == 0
    assert c.count("s2", datetime(2026, 4, 21, 10, 1)) == 0
