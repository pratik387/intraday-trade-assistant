"""earnings_downshock_continuation_short detector unit tests.

Frozen construction: earnings reaction-day move <= -8% on T-1, SHORT at the
close of the 09:15-labelled 5m bar (the 09:20 print), hard time exit 15:15,
geometry none/none with a 15% catastrophe backstop that never binds.

Brief: specs/2026-07-29-brief-earnings_downshock_continuation_short.md
Cell lock: tools/sub9_research/earnings_downshock_continuation_short_cell_lock.json
Discovery n=220 PF 1.55 @CENTRAL slippage; demoted n=219 PF 1.44; fresh-pool
n=77 PASS by 1.4bp (one month carries it — forward expectation ~zero).
"""
from datetime import date, time

import pandas as pd
import pytest

from structures.data_models import MarketContext
from structures.earnings_downshock_continuation_short_structure import (
    EarningsDownshockContinuationShortStructure,
)

SESSION = date(2024, 6, 12)
PX = 100.0


def _cfg(**over):
    cfg = {
        "_setup_name": "earnings_downshock_continuation_short",
        "enabled": False,
        "paper_enabled": False,
        "active_window_start": "09:15",
        "active_window_end": "09:15",
        "time_stop_at": "15:10",
        "shock_threshold_pct": -8.0,
        "reaction_same_day_classes": ["intraday", "BMO"],
        "reaction_same_day_hour_max": 15,
        "reaction_max_staleness_days": 7,
        "catastrophe_stop_pct": 15.0,
        "min_stop_distance_pct": 0.5,
        "exclude_circuit_blocked_open": True,
        "circuit_block_min_gap_pct": -1.5,   # PRODUCTION value (negative)
        "min_bars_required": 1,
        "entry_order_working_required": True,
        "entry_order_working_supported": False,
        "entry_zone_pct": 0.3,
        "entry_zone_mode": "symmetric",
    }
    cfg.update(over)
    return cfg


def _daily(reaction_move_pct: float, *, flag: bool = True, n: int = 30) -> pd.DataFrame:
    """Daily frame ending at T-1 (the reaction session), pre-enriched so the
    detector does not need the parquet."""
    idx = pd.bdate_range(end=pd.Timestamp(SESSION) - pd.Timedelta(days=1), periods=n)
    close = [PX] * n
    close[-1] = PX * (1.0 + reaction_move_pct / 100.0)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": [1e5] * n},
        index=idx,
    )
    df["is_earnings_reaction_day"] = [False] * (n - 1) + [flag]
    return df


def _ctx(df_daily, *, hh=9, mm=15, close=PX, hi=None, lo=None, sym="NSE:TESTCO"):
    ts = pd.Timestamp(SESSION) + pd.Timedelta(hours=hh, minutes=mm)
    d5 = pd.DataFrame(
        {"open": [close * 1.002], "high": [hi if hi is not None else close * 1.004],
         "low": [lo if lo is not None else close * 0.998], "close": [close],
         "volume": [1e4]},
        index=[ts],
    )
    return MarketContext(
        symbol=sym, df_5m=d5, df_daily=df_daily, session_date=SESSION,
        regime="chop", current_price=close, timestamp=ts,
    )


def _det(**over):
    return EarningsDownshockContinuationShortStructure(_cfg(**over))


# ---------------------------------------------------------------- config

def test_missing_config_key_raises():
    """CLAUDE.md rule 1: no hardcoded defaults — missing key must fail fast."""
    for key in (
        "shock_threshold_pct", "time_stop_at", "catastrophe_stop_pct",
        "active_window_start", "min_bars_required",
    ):
        bad = _cfg()
        del bad[key]
        with pytest.raises(KeyError):
            EarningsDownshockContinuationShortStructure(bad)


def test_locked_parameters_are_read_from_config():
    d = _det()
    assert d.shock_threshold_pct == -8.0
    assert d.time_stop_at == "15:10"          # broker mechanics, not performance
    assert d.catastrophe_stop_pct == 15.0
    assert d.active_start == time(9, 15) and d.active_end == time(9, 15)


# ---------------------------------------------------------------- signal

def test_fires_on_deep_down_shock():
    r = _det().detect(_ctx(_daily(-10.0)))
    assert r.structure_detected
    ev = r.events[0]
    assert ev.side == "short"
    assert ev.context["reaction_move_pct"] == pytest.approx(-10.0, abs=1e-6)


def test_does_not_fire_when_shock_too_shallow():
    """-5% is a down day but not a down-SHOCK; threshold is -8%."""
    r = _det().detect(_ctx(_daily(-5.0)))
    assert not r.structure_detected


def test_threshold_is_inclusive_at_minus_eight():
    assert _det().detect(_ctx(_daily(-8.0))).structure_detected


def test_does_not_fire_on_up_move():
    assert not _det().detect(_ctx(_daily(+10.0))).structure_detected


def test_does_not_fire_when_t1_is_not_a_reaction_day():
    """A -10% day with no earnings behind it is not this setup."""
    assert not _det().detect(_ctx(_daily(-10.0, flag=False))).structure_detected


def test_stale_reaction_is_rejected():
    """A reaction more than reaction_max_staleness_days back is a data gap."""
    df = _daily(-10.0)
    df.index = df.index - pd.Timedelta(days=20)
    assert not _det().detect(_ctx(df)).structure_detected


# ---------------------------------------------------------------- timing

def test_does_not_fire_outside_the_single_bar_window():
    for hh, mm in ((9, 20), (10, 30), (14, 55)):
        r = _det().detect(_ctx(_daily(-10.0), hh=hh, mm=mm))
        assert not r.structure_detected
        assert "Outside active window" in (r.rejection_reason or "")


def test_latch_allows_only_one_fire_per_session():
    d = _det()
    assert d.detect(_ctx(_daily(-10.0))).structure_detected
    second = d.detect(_ctx(_daily(-10.0)))
    assert not second.structure_detected
    assert "already fired" in (second.rejection_reason or "")


# ---------------------------------------------------------------- guards

def test_circuit_blocked_when_zero_range_AND_gapped_down_past_threshold():
    """REGRESSION: the guard read `abs(gap_pct) >= threshold` with a NEGATIVE
    production threshold (-1.5), which is always true — it rejected every
    zero-range bar regardless of gap direction, over-rejecting vs the validated
    construction (research measured this filter at 0.26-0.28% of events)."""
    # reaction close is PX*0.90; open PX*0.80 => gap ~-11% (well past -1.5)
    df = _daily(-10.0)
    r = _det().detect(_ctx(df, close=PX * 0.80, hi=PX * 0.80, lo=PX * 0.80))
    assert not r.structure_detected
    assert "circuit-blocked" in (r.rejection_reason or "")


def test_zero_range_bar_that_did_NOT_gap_down_still_fires():
    """The magnitude gate must be live: a flat zero-range open is not a lower
    circuit lock. Under the old abs() bug this was wrongly rejected."""
    df = _daily(-10.0)
    flat = PX * 0.90                      # == the reaction close, gap ~0%
    r = _det().detect(_ctx(df, close=flat, hi=flat, lo=flat))
    assert r.structure_detected, r.rejection_reason


def test_circuit_guard_can_be_disabled_by_config():
    d = _det(exclude_circuit_blocked_open=False)
    r = d.detect(_ctx(_daily(-10.0), close=PX * 0.80, hi=PX * 0.80, lo=PX * 0.80))
    assert r.structure_detected


def test_execution_working_flags_are_fail_fast():
    """The order layer cannot yet WORK an order across the 5m block; the flags
    that surface that gap must be validated like any other trading parameter."""
    for key in ("entry_order_working_required", "entry_order_working_supported"):
        bad = _cfg()
        del bad[key]
        with pytest.raises(KeyError):
            EarningsDownshockContinuationShortStructure(bad)


def test_execution_working_unsupported_warning_is_emitted_on_every_fire(caplog):
    """config documents this warning fires on EVERY fire so the order-layer gap
    can never be silent. It previously did not exist in code at all."""
    import logging
    d = _det()
    with caplog.at_level(logging.WARNING):
        assert d.detect(_ctx(_daily(-10.0))).structure_detected
    assert any("EXECUTION_WORKING_UNSUPPORTED" in rec.getMessage() for rec in caplog.records)


def test_insufficient_bars_rejected():
    d = _det(min_bars_required=5)
    assert not d.detect(_ctx(_daily(-10.0))).structure_detected


# ---------------------------------------------------------------- plan

def test_plan_is_short_with_no_targets_and_a_time_exit():
    d = _det()
    ctx = _ctx(_daily(-10.0))
    ev = d.detect(ctx).events[0]
    plan = d.plan_short_strategy(ctx, ev)
    assert plan is not None
    assert plan.side == "short"
    # geometry none/none: no profit targets, single exit on the time stop
    assert plan.exit_levels.targets == []
    assert plan.exit_levels.time_exit == "15:10"
    # catastrophe backstop ABOVE entry for a short, at the configured pct
    assert plan.risk_params.hard_sl == pytest.approx(plan.entry_price * 1.15, rel=1e-6)


def test_catastrophe_stop_is_far_enough_never_to_bind():
    """Max adverse excursion across all 220 Discovery trades was 12.97%."""
    assert _det().catastrophe_stop_pct > 12.97


def test_long_side_is_never_planned():
    d = _det()
    ctx = _ctx(_daily(-10.0))
    ev = d.detect(ctx).events[0]
    assert d.plan_long_strategy(ctx, ev) is None


def test_plan_rejects_a_mismatched_event_side():
    d = _det()
    ctx = _ctx(_daily(-10.0))
    ev = d.detect(ctx).events[0]
    ev.side = "long"
    assert d.plan_short_strategy(ctx, ev) is None


# ---------------------------------------------------------------- ABC contract

def test_validate_timing_matches_the_single_bar_window():
    d = _det()
    assert d.validate_timing(pd.Timestamp(SESSION) + pd.Timedelta(hours=9, minutes=15))
    assert not d.validate_timing(
        pd.Timestamp(SESSION) + pd.Timedelta(hours=10, minutes=30)
    )


def test_get_exit_levels_carries_the_time_exit():
    d = _det()
    ctx = _ctx(_daily(-10.0))
    plan = d.plan_short_strategy(ctx, d.detect(ctx).events[0])
    levels = d.get_exit_levels(plan)
    assert levels.targets == []
    assert levels.time_exit == "15:10"
