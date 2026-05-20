"""Regression tests for services.execution.exit_executor.classify_sl_exit_reason.

The bug: SL exits were labeled `sl_post_t1` whenever `t1_done=True`, even when
no T1 partial was ACTUALLY booked. T1 done flag fires for two skip cases:

  1. t1_skipped_low_r — partial-R below `t1_min_partial_r` threshold, partial
     skipped to avoid fee-dominated entry. SL fires at original hard_sl.
  2. t1_skipped_plan_zero — config has `t1_partial_qty_pct: 0.0`, full
     quantity rides to T2 or SL. SL fires at original hard_sl.

The pre-fix code at exit_executor.py:222-228 and :473-479 checked
`t1_skipped_low_r` but missed `t1_skipped_plan_zero`. This caused setups
configured for full-ride (or_window_failure_fade_short, circuit_release_fade_short)
to log SL exits as `sl_post_t1` despite never booking a T1 partial.

OCI evidence (both May 19 OCI runs): ~150 mis-labeled cases across these
setups, aggregate -Rs 170K of -1R stop losses (expected behavior, just
mis-attributed in trade reports).
"""
from services.execution.exit_executor import classify_sl_exit_reason


def test_t2_done_returns_sl_post_t2():
    st = {"t2_done": True, "t1_done": True}
    assert classify_sl_exit_reason(st) == "sl_post_t2"


def test_t1_done_with_actual_booking_returns_sl_post_t1():
    st = {"t1_done": True, "t1_booked_qty": 50}
    assert classify_sl_exit_reason(st) == "sl_post_t1"


def test_t1_skipped_low_r_returns_hard_sl_not_sl_post_t1():
    """REGRESSION (2026-05-20): low-R skip means no partial booked → hard_sl."""
    st = {"t1_done": True, "t1_skipped_low_r": True, "t1_booked_qty": 0}
    assert classify_sl_exit_reason(st) == "hard_sl"


def test_t1_skipped_plan_zero_returns_hard_sl_not_sl_post_t1():
    """REGRESSION (2026-05-20): plan qty_pct=0 means no partial booked → hard_sl.

    Pre-fix bug: this case was MISSED, returned `sl_post_t1` despite no T1
    booking. Affected ~150 trades for or_window_failure_fade_short +
    circuit_release_fade_short across OCI runs.
    """
    st = {"t1_done": True, "t1_skipped_plan_zero": True, "t1_booked_qty": 0}
    assert classify_sl_exit_reason(st) == "hard_sl"


def test_no_t1_done_returns_default_no_t1_tick_sl():
    """Per-tick check uses tick_sl as default fallback."""
    st = {}
    assert classify_sl_exit_reason(st, default_no_t1="tick_sl") == "tick_sl"


def test_no_t1_done_returns_default_no_t1_hard_sl():
    """Bar-boundary check uses hard_sl as default fallback."""
    st = {}
    assert classify_sl_exit_reason(st, default_no_t1="hard_sl") == "hard_sl"


def test_t1_skipped_low_r_and_plan_zero_both_true_returns_hard_sl():
    """Edge case: both skip flags set → still hard_sl, never sl_post_t1."""
    st = {
        "t1_done": True,
        "t1_skipped_low_r": True,
        "t1_skipped_plan_zero": True,
    }
    assert classify_sl_exit_reason(st) == "hard_sl"


def test_t2_done_supersedes_t1_state():
    """t2_done is checked first regardless of t1 skip flags."""
    st = {
        "t2_done": True,
        "t1_done": True,
        "t1_skipped_plan_zero": True,
    }
    assert classify_sl_exit_reason(st) == "sl_post_t2"
