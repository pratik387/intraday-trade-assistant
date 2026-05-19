"""OR-Window Failure Fade SHORT detector unit tests.

Cell-locked detector validated via 3-window cross-stability test
(see specs/2026-05-16-new-setup-candidates.md C-10).
"""
import pandas as pd
import pytest

from structures.or_window_failure_fade_short_structure import OrWindowFailureFadeShortStructure
from structures.data_models import MarketContext


def _cfg(**overrides):
    base = {
        "_setup_name": "or_window_failure_fade_short",
        "enabled": True,
        "or_bars": 3,
        "active_window_start": "09:30",
        "active_window_end": "10:30",
        "poke_pct": 0.15,
        "vol_ratio_min": 8.0,
        "vol_ratio_max": 15.0,
        "allowed_cap_segments": ["small_cap"],
        "sl_pct_above_sweep_high": 0.3,
        "min_stop_pct": 0.5,
        "t1_r_multiple": 1.0,
        "t2_r_multiple": 2.0,
        "t1_partial_qty_pct": 0.0,
        "time_stop_at": "15:10",
        "min_bars_required": 5,
        "entry_zone_pct": 0.3,
        "entry_zone_mode": "symmetric",
        "min_stop_distance_pct": 0.3,
    }
    base.update(overrides)
    return base


def _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=99.0,
          sweep_volume=10000, prior_vol_avg=1000, session_date=pd.Timestamp("2026-05-20").date(),
          sweep_hhmm="10:00"):
    """Build today's bars: OR bars (09:15-09:30) + intermediate bars + sweep + confirm."""
    rows = []
    # OR bars: 09:15, 09:20, 09:25 - establish ORH
    or_times = [
        pd.Timestamp(f"{session_date} 09:15:00"),
        pd.Timestamp(f"{session_date} 09:20:00"),
        pd.Timestamp(f"{session_date} 09:25:00"),
    ]
    # First OR bar hits the ORH; rest lower
    rows.append({"date": or_times[0], "open": 99.0, "high": orh,
                 "low": 98.5, "close": 99.5, "volume": prior_vol_avg})
    rows.append({"date": or_times[1], "open": 99.5, "high": 99.8,
                 "low": 99.0, "close": 99.3, "volume": prior_vol_avg})
    rows.append({"date": or_times[2], "open": 99.3, "high": 99.6,
                 "low": 98.8, "close": 99.0, "volume": prior_vol_avg})

    # Intermediate bars before sweep
    sweep_ts = pd.Timestamp(f"{session_date} {sweep_hhmm}:00")
    inter_count = int(((sweep_ts - or_times[-1]).total_seconds() / 300)) - 1
    if inter_count > 0:
        inter_times = pd.date_range(start=or_times[-1] + pd.Timedelta(minutes=5),
                                     periods=inter_count, freq="5min")
        for t in inter_times:
            rows.append({"date": t, "open": 99.0, "high": 99.5,
                         "low": 98.5, "close": 99.0, "volume": prior_vol_avg})

    # Sweep bar
    rows.append({"date": sweep_ts, "open": 99.5, "high": sweep_high,
                 "low": 99.0, "close": sweep_close, "volume": sweep_volume})
    # Confirmation bar
    confirm_ts = sweep_ts + pd.Timedelta(minutes=5)
    rows.append({"date": confirm_ts, "open": sweep_close, "high": sweep_close,
                 "low": confirm_close, "close": confirm_close, "volume": prior_vol_avg})

    return pd.DataFrame(rows).set_index("date").sort_index()


def _ctx(df, symbol="TEST", cap_segment="small_cap"):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol, current_price=float(df.iloc[-1]["close"]), timestamp=last_ts,
        df_5m=df, session_date=last_ts.date(),
        pdh=100.0, pdl=95.0, pdc=98.0, cap_segment=cap_segment,
    )


# --------- Detection tests ---------

def test_canonical_failed_pierce_fires():
    """ORH=100, sweep pokes to 100.5, closes 99.5 (back below), confirm at 99.0."""
    # vol_ratio = 10000 / 1000 = 10 (in 8-15 cell band)
    df = _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=99.0,
               sweep_volume=10000, prior_vol_avg=1000)
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert r.structure_detected, f"Should fire on canonical failed pierce, got: {r.rejection_reason}"
    e = r.events[0]
    assert e.side == "short"
    assert e.structure_type == "or_window_failure_fade_short"
    assert e.levels["orh"] == 100.0


def test_rejects_no_pierce():
    """Sweep bar high doesn't reach ORH * 1.0015 poke threshold."""
    df = _bars(orh=100.0, sweep_high=100.1, sweep_close=99.5, confirm_close=99.0,
               sweep_volume=10000, prior_vol_avg=1000)
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected


def test_rejects_close_above_orh():
    """Sweep bar pierces but closes ABOVE ORH (genuine breakout)."""
    df = _bars(orh=100.0, sweep_high=100.5, sweep_close=100.2, confirm_close=100.1,
               sweep_volume=10000, prior_vol_avg=1000)
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected


def test_rejects_no_recovery_confirm():
    """Sweep happens, but next bar closes back above ORH."""
    df = _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=100.2,
               sweep_volume=10000, prior_vol_avg=1000)
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected


def test_rejects_vol_ratio_too_low():
    """vol_ratio = 4 (below 8). Outside cell band."""
    df = _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=99.0,
               sweep_volume=4000, prior_vol_avg=1000)  # ratio = 4
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "vol_ratio" in r.rejection_reason


def test_rejects_vol_ratio_too_high():
    """vol_ratio = 20 (above 15). Outside cell band (parabolic continuation)."""
    df = _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=99.0,
               sweep_volume=20000, prior_vol_avg=1000)  # ratio = 20
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "vol_ratio" in r.rejection_reason


def test_rejects_outside_active_window():
    """Sweep at 11:00 IST is past active window end (10:30)."""
    df = _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=99.0,
               sweep_volume=10000, prior_vol_avg=1000, sweep_hhmm="11:00")
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "active window" in r.rejection_reason


# cap_segment early-reject removed: universe builders filter before dispatch


def test_latch_prevents_repeat():
    df = _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=99.0,
               sweep_volume=10000, prior_vol_avg=1000)
    det = OrWindowFailureFadeShortStructure(_cfg())
    r1 = det.detect(_ctx(df))
    assert r1.structure_detected
    r2 = det.detect(_ctx(df))
    assert not r2.structure_detected
    assert "already fired" in r2.rejection_reason


# --------- Plan tests ---------

def test_plan_uses_ride_to_t2():
    """T1 should have qty_pct=0.0 (no exit), T2 should have qty_pct=1.0 (full exit)."""
    df = _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=99.0,
               sweep_volume=10000, prior_vol_avg=1000)
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    plan = det.plan_short_strategy(_ctx(df), event=r.events[0])
    assert plan is not None
    assert plan.side == "short"
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    # T1 = no exit (informational)
    assert targets[0]["name"] == "T1"
    assert targets[0]["qty_pct"] == 0.0
    # T2 = full exit
    assert targets[1]["name"] == "T2"
    assert targets[1]["qty_pct"] == 1.0


def test_plan_sl_takes_farther():
    df = _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=99.0,
               sweep_volume=10000, prior_vol_avg=1000)
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    plan = det.plan_short_strategy(_ctx(df), event=r.events[0])
    entry = plan.entry_price
    sweep_high = 100.5
    sl_from_sweep = sweep_high * 1.003
    sl_from_min = entry * 1.005
    expected_sl = max(sl_from_sweep, sl_from_min)
    assert plan.risk_params.hard_sl == pytest.approx(expected_sl, rel=1e-6)


def test_long_plan_returns_none():
    df = _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=99.0,
               sweep_volume=10000, prior_vol_avg=1000)
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert det.plan_long_strategy(_ctx(df), event=r.events[0]) is None


def test_warmup_bars_do_not_pollute():
    """Prior-day warmup bars must not contaminate today's ORH calculation."""
    today_df = _bars(orh=100.0, sweep_high=100.5, sweep_close=99.5, confirm_close=99.0,
                     sweep_volume=10000, prior_vol_avg=1000)
    prior_ts = pd.date_range(start="2026-05-19 14:00", periods=10, freq="5min")
    prior_df = pd.DataFrame({
        "open": [200.0] * 10, "high": [210.0] * 10,  # spurious high
        "low": [195.0] * 10, "close": [200.0] * 10, "volume": [10000] * 10,
    }, index=prior_ts)
    df = pd.concat([prior_df, today_df])
    det = OrWindowFailureFadeShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert r.structure_detected, f"Warmup pollution broke detection: {r.rejection_reason}"
    # ORH should be 100 (from today's OR bars), not 210 (from yesterday's warmup)
    assert r.events[0].levels["orh"] == 100.0
