"""
Unit tests for the per-plan time-stop entry guard in TriggerAwareExecutor.

Regression for the 2026-06-05 bug: below_vwap_volume_revert_long has
time_stop_at=14:30 but active_window_end=14:55 and the GLOBAL entry cutoff
is 14:45. A trade entered at 14:36 passed the global cutoff, entered, then
was squared off on the very next tick (14:37) with reason time_stop_14:30 —
a guaranteed instant loss. The guard rejects entries at/after the plan's
per-setup time-stop, independent of the global cutoff.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _make_executor():
    from services.execution.trigger_aware_executor import TriggerAwareExecutor
    with patch.object(TriggerAwareExecutor, "__init__", lambda self: None):
        ex = TriggerAwareExecutor()
    return ex


def _plan(time_stop="14:30", strategy="below_vwap_volume_revert_long"):
    return {"strategy": strategy, "exits": {"time_stop_hhmm": time_stop}}


# Default: setup is NOT in wide_open research-capture mode, and ticks are
# fresh (the stale-ts guard is a live-only WS-snapshot safety layer — here we
# isolate the core minute-of-day logic, then test the stale path explicitly).
@pytest.fixture(autouse=True)
def _not_wide_open():
    with patch("services.config_loader.is_wide_open_for_setup", return_value=False), \
         patch("services.execution.exit_executor._is_stale_ts_for_live", return_value=False):
        yield


def test_entry_after_time_stop_blocked():
    """14:36 entry with a 14:30 time-stop must be rejected (the reported bug)."""
    ex = _make_executor()
    now = pd.Timestamp("2026-06-05 14:36:58")
    assert ex._past_setup_time_stop(_plan("14:30"), now) is True


def test_entry_exactly_at_time_stop_blocked():
    """Entry at exactly the time-stop minute would square off immediately."""
    ex = _make_executor()
    now = pd.Timestamp("2026-06-05 14:30:00")
    assert ex._past_setup_time_stop(_plan("14:30"), now) is True


def test_entry_before_time_stop_allowed():
    """A 14:25 entry with a 14:30 time-stop is fine (holds to the stop)."""
    ex = _make_executor()
    now = pd.Timestamp("2026-06-05 14:25:00")
    assert ex._past_setup_time_stop(_plan("14:30"), now) is False


def test_no_time_stop_in_plan_allowed():
    """No per-setup time-stop -> guard is a no-op (global cutoff still applies)."""
    ex = _make_executor()
    now = pd.Timestamp("2026-06-05 14:36:00")
    assert ex._past_setup_time_stop({"strategy": "x", "exits": {}}, now) is False
    assert ex._past_setup_time_stop({"strategy": "x"}, now) is False


def test_wide_open_research_capture_bypasses_guard():
    """wide_open capture mode rides to natural EOD; must NOT block at entry."""
    ex = _make_executor()
    now = pd.Timestamp("2026-06-05 14:36:00")
    with patch("services.config_loader.is_wide_open_for_setup", return_value=True):
        assert ex._past_setup_time_stop(_plan("14:30"), now) is False


def test_stale_live_tick_does_not_block_entry():
    """A stale WS snapshot tick (live/paper) must NOT falsely block entry."""
    ex = _make_executor()
    now = pd.Timestamp("2026-06-05 14:36:00")  # md past 14:30, but stale
    with patch("services.execution.exit_executor._is_stale_ts_for_live", return_value=True):
        assert ex._past_setup_time_stop(_plan("14:30"), now) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
