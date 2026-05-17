"""Circuit Release Fade SHORT detector unit tests.

Aggregate validated on Discovery/OOS/Holdout (see
specs/2026-05-16-new-setup-candidates.md -> C-03 and
tools/sub9_research/sanity_circuit_release_fade.py).
"""
import pandas as pd
import pytest

from structures.circuit_release_fade_short_structure import CircuitReleaseFadeShortStructure
from structures.data_models import MarketContext


def _cfg(**overrides):
    base = {
        "_setup_name": "circuit_release_fade_short",
        "enabled": True,
        "active_window_start": "12:00",
        "active_window_end": "15:10",
        "min_day_gain_pct": 4.5,
        "morning_high_by_hhmm": "10:30",
        "morning_high_tolerance_pct": 0.1,
        "retest_tol_pct": 0.3,
        "rejection_close_pct": 0.3,
        "volume_confirm_lookback": 5,
        "allowed_cap_segments": ["small_cap", "mid_cap"],
        "sl_buffer_above_rejection_high_pct": 0.3,
        "min_stop_pct": 0.5,
        "t1_r_multiple": 1.0,
        "t2_r_multiple": 2.0,
        "t1_partial_qty_pct": 0.5,
        "time_stop_at": "15:10",
        "min_bars_required": 8,
        "entry_zone_pct": 0.3,
        "entry_zone_mode": "symmetric",
        "min_stop_distance_pct": 0.3,
    }
    base.update(overrides)
    return base


def _bars(morning_high_price, current_bar_high, current_bar_close,
          current_bar_low=None, current_bar_volume=10000, pdc=100.0,
          symbol="TEST", session_date=pd.Timestamp("2026-05-20").date(),
          current_hhmm="13:00", n_morning_bars=10, n_afternoon_bars_before=5):
    """Build a 5m bar DataFrame that simulates:
    - Morning bars (09:15 onwards) hitting morning_high_price
    - Mid-day bars (post-12:00) leading up to the current rejection bar
    - Current rejection bar at the given hhmm.

    The current bar is positioned such that:
      bar.high = current_bar_high
      bar.close = current_bar_close
      bar.low = current_bar_low (defaults to slight below close)
      bar.volume = current_bar_volume
    All prior bars have volume = 5000 (so current bar's volume is the comparison).
    """
    if current_bar_low is None:
        current_bar_low = current_bar_close * 0.998

    rows = []
    # Morning bars 09:15-10:30 (16 bars)
    morning_times = pd.date_range(
        start=f"{session_date} 09:15:00", end=f"{session_date} 10:30:00", freq="5min"
    )
    for i, t in enumerate(morning_times):
        # The 5th morning bar makes the morning high
        if i == 5:
            rows.append({"date": t, "open": pdc * 1.02, "high": morning_high_price,
                         "low": pdc * 1.01, "close": morning_high_price * 0.999, "volume": 5000})
        else:
            rows.append({"date": t, "open": pdc * 1.02, "high": pdc * 1.025,
                         "low": pdc * 1.01, "close": pdc * 1.022, "volume": 5000})

    # 10:35-12:00 bars (mild pullback from high)
    mid_times = pd.date_range(
        start=f"{session_date} 10:35:00", end=f"{session_date} 11:55:00", freq="5min"
    )
    for t in mid_times:
        rows.append({"date": t, "open": morning_high_price * 0.99,
                     "high": morning_high_price * 0.995, "low": morning_high_price * 0.985,
                     "close": morning_high_price * 0.988, "volume": 5000})

    # Afternoon bars before the current bar (post-12:00 up to current_hhmm)
    cur_h, cur_m = map(int, current_hhmm.split(":"))
    cur_ts = pd.Timestamp(f"{session_date} {current_hhmm}:00")
    before_ts = pd.date_range(end=cur_ts - pd.Timedelta(minutes=5),
                               periods=n_afternoon_bars_before, freq="5min")
    for t in before_ts:
        rows.append({"date": t, "open": morning_high_price * 0.99,
                     "high": morning_high_price * 0.995,
                     "low": morning_high_price * 0.985,
                     "close": morning_high_price * 0.988, "volume": 5000})

    # CURRENT REJECTION BAR
    rows.append({"date": cur_ts, "open": morning_high_price * 0.99,
                 "high": current_bar_high, "low": current_bar_low,
                 "close": current_bar_close, "volume": current_bar_volume})

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def _ctx(df, symbol="TEST", pdc=100.0, cap_segment="small_cap"):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df.iloc[-1]["close"]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.date(),
        pdh=pdc * 0.98,  # arbitrary - not used
        pdl=pdc * 0.95,  # arbitrary
        pdc=pdc,
        cap_segment=cap_segment,
    )


# --------- Detection tests ---------

def test_canonical_failed_retest_fires():
    """Morning pin at 106 (+6% gain), afternoon re-test at 105.8 with rejection."""
    df = _bars(morning_high_price=106.0,
               current_bar_high=105.8, current_bar_close=105.0,  # 0.76% rejection
               current_bar_volume=10000)  # higher than 5000 prior median
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert r.structure_detected, f"Should fire on canonical failed retest, got: {r.rejection_reason}"
    e = r.events[0]
    assert e.side == "short"
    assert e.structure_type == "circuit_release_fade_short"
    assert e.levels["session_high"] == 106.0
    assert e.levels["rejection_high"] == 105.8
    assert e.context["day_gain_pct"] == pytest.approx(6.0, rel=1e-3)


def test_rejects_day_gain_too_small():
    """Day gain only 3% -> below 4.5% threshold."""
    df = _bars(morning_high_price=103.0,  # only +3%
               current_bar_high=102.9, current_bar_close=102.5,
               current_bar_volume=10000)
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "day_gain" in r.rejection_reason


def test_rejects_when_new_high_after_morning():
    """Morning high 106, but a new high 107 happened post-10:30."""
    df = _bars(morning_high_price=106.0,
               current_bar_high=107.0,  # new session high - not a morning pin
               current_bar_close=106.5,
               current_bar_volume=10000)
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "morning_high" in r.rejection_reason


def test_rejects_when_no_retest():
    """Current bar high is well below session high - no re-test."""
    df = _bars(morning_high_price=106.0,
               current_bar_high=103.0,  # way below morning high 106 (>0.3% away)
               current_bar_close=102.5,
               current_bar_volume=10000)
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "no re-test" in r.rejection_reason or "retest_threshold" in r.rejection_reason


def test_rejects_when_no_rejection_wick():
    """Bar reaches high but closes near it (no rejection)."""
    df = _bars(morning_high_price=106.0,
               current_bar_high=105.8,
               current_bar_close=105.79,  # close almost at high
               current_bar_volume=10000)
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "rejection" in r.rejection_reason


def test_rejects_when_volume_below_recent_median():
    """Volume confirmation: bar volume < prior 5-bar median -> reject."""
    df = _bars(morning_high_price=106.0,
               current_bar_high=105.8, current_bar_close=105.0,
               current_bar_volume=2000)  # below 5000 prior median
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "recent_median" in r.rejection_reason


# cap_segment early-reject removed: universe builders filter before dispatch


def test_outside_active_window_rejects():
    """Re-test at 11:30 IST is before active window (12:00)."""
    df = _bars(morning_high_price=106.0,
               current_bar_high=105.8, current_bar_close=105.0,
               current_bar_volume=10000,
               current_hhmm="11:30")  # before active window
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "active window" in r.rejection_reason


def test_latch_prevents_repeat_within_session():
    """Once fired for (symbol, session), subsequent calls do not re-fire."""
    df = _bars(morning_high_price=106.0,
               current_bar_high=105.8, current_bar_close=105.0,
               current_bar_volume=10000)
    det = CircuitReleaseFadeShortStructure(_cfg())
    r1 = det.detect(_ctx(df))
    assert r1.structure_detected
    # Second call with same context = same (symbol, session_date) - should be latched
    r2 = det.detect(_ctx(df))
    assert not r2.structure_detected
    assert "already fired" in r2.rejection_reason


def test_insufficient_bars_rejects():
    """Less than min_bars_required - reject."""
    # Build a df with only 3 bars - way below min_bars_required=8
    df = pd.DataFrame({
        "open": [102, 103, 104], "high": [103, 104, 105],
        "low": [101, 102, 103], "close": [102.5, 103.5, 104.5], "volume": [5000, 5000, 5000],
    }, index=pd.date_range(start="2026-05-20 13:00", periods=3, freq="5min"))
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "Insufficient bars" in r.rejection_reason


# --------- Plan tests ---------

def test_plan_has_correct_target_geometry():
    """T1 = entry - 1R, T2 = entry - 2R, SL = max(rejection_high * 1.003, entry * 1.005)."""
    df = _bars(morning_high_price=106.0,
               current_bar_high=105.8, current_bar_close=105.0,
               current_bar_volume=10000)
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert r.structure_detected
    plan = det.plan_short_strategy(_ctx(df), event=r.events[0])
    assert plan is not None
    assert plan.side == "short"
    entry = plan.entry_price
    sl = plan.risk_params.hard_sl
    R = sl - entry
    assert R > 0
    # T1 should be ~1R below entry, T2 ~2R below
    t1 = plan.exit_levels.targets[0]["level"]
    t2 = plan.exit_levels.targets[1]["level"]
    assert t1 == pytest.approx(entry - R, rel=1e-6)
    assert t2 == pytest.approx(entry - 2 * R, rel=1e-6)


def test_long_plan_returns_none():
    """Short-only setup - plan_long_strategy returns None."""
    df = _bars(morning_high_price=106.0,
               current_bar_high=105.8, current_bar_close=105.0,
               current_bar_volume=10000)
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    plan = det.plan_long_strategy(_ctx(df), event=r.events[0])
    assert plan is None


def test_sl_takes_farther_of_two_stops():
    """SL = max(rejection_high * 1.003, entry * (1 + min_stop_pct/100)) - the FARTHER one."""
    df = _bars(morning_high_price=106.0,
               current_bar_high=105.8, current_bar_close=105.0,
               current_bar_volume=10000)
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    plan = det.plan_short_strategy(_ctx(df), event=r.events[0])
    entry = plan.entry_price
    rej_high = r.events[0].levels["rejection_high"]
    sl_from_high = rej_high * 1.003
    sl_from_min = entry * 1.005  # 0.5% min_stop
    expected_sl = max(sl_from_high, sl_from_min)
    assert plan.risk_params.hard_sl == pytest.approx(expected_sl, rel=1e-6)


def test_warmup_bars_do_not_pollute_detection():
    """df_5m may contain prior-day warmup bars. session_high should be computed
    from TODAY's bars only, not from warmup."""
    # Build today's normal bars
    today_df = _bars(morning_high_price=106.0,
                     current_bar_high=105.8, current_bar_close=105.0,
                     current_bar_volume=10000)
    # Prepend prior-day warmup bars with a much higher artificial high (108 > today's 106)
    prior_ts = pd.date_range(start="2026-05-19 14:00", end="2026-05-19 15:25", freq="5min")
    prior_df = pd.DataFrame({
        "open": [107.0] * len(prior_ts), "high": [108.0] * len(prior_ts),  # spurious high
        "low": [106.5] * len(prior_ts), "close": [107.5] * len(prior_ts), "volume": [10000] * len(prior_ts),
    }, index=prior_ts)
    df = pd.concat([prior_df, today_df])
    det = CircuitReleaseFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert r.structure_detected, f"Warmup pollution broke detection: {r.rejection_reason}"
    # session_high should be 106 (today's), not 108 (prior day)
    assert r.events[0].levels["session_high"] == 106.0
