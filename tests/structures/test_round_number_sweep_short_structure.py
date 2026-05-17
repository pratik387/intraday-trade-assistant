"""Round-Number Sweep SHORT detector unit tests.

Cell-locked detector validated via 3-window cross-stability test
(see specs/2026-05-16-new-setup-candidates.md C-02).
"""
import pandas as pd
import pytest

from structures.round_number_sweep_short_structure import RoundNumberSweepShortStructure
from structures.data_models import MarketContext


def _cfg(**overrides):
    base = {
        "_setup_name": "round_number_sweep_short",
        "enabled": True,
        "active_window_start": "11:00",
        "active_window_end": "12:30",
        "min_price": 100.0,
        "max_price": 250.0,
        "round_number_increment": 50.0,
        "poke_pct": 0.15,
        "vol_ratio_min": 2.0,
        "allowed_cap_segments": ["small_cap"],
        "sl_pct_above_sweep_high": 0.5,
        "min_stop_pct": 0.5,
        "t1_r_multiple": 1.0,
        "t2_r_multiple": 2.0,
        "t1_partial_qty_pct": 1.0,
        "time_stop_at": "15:00",
        "min_bars_required": 8,
        "entry_zone_pct": 0.3,
        "entry_zone_mode": "symmetric",
        "min_stop_distance_pct": 0.3,
    }
    base.update(overrides)
    return base


def _bars(prior_bars_avg_volume=5000, sweep_bar_high=150.5, sweep_bar_close=149.5,
          confirm_bar_close=149.0, sweep_bar_volume=15000, rn=150.0,
          session_date=pd.Timestamp("2026-05-20").date(), sweep_hhmm="12:00",
          price_band=150.0):
    """Build 5m bars: many prior bars + sweep bar (current-1) + confirm bar (current).

    sweep_bar = pokes above rn, closes back below
    confirm_bar = close stays below rn (recovery confirmation)
    """
    rows = []
    # 10 prior bars (09:15 onwards) - establish session
    prior_times = pd.date_range(start=f"{session_date} 09:15:00", periods=10, freq="5min")
    for t in prior_times:
        rows.append({"date": t, "open": price_band, "high": price_band * 1.005,
                     "low": price_band * 0.995, "close": price_band,
                     "volume": prior_bars_avg_volume})

    # More bars up to sweep
    sweep_ts = pd.Timestamp(f"{session_date} {sweep_hhmm}:00")
    intermediate_count = int(((sweep_ts - prior_times[-1]).total_seconds() / 300)) - 1
    if intermediate_count > 0:
        inter_times = pd.date_range(start=prior_times[-1] + pd.Timedelta(minutes=5),
                                     periods=intermediate_count, freq="5min")
        for t in inter_times:
            rows.append({"date": t, "open": price_band, "high": price_band * 1.005,
                         "low": price_band * 0.995, "close": price_band,
                         "volume": prior_bars_avg_volume})

    # Sweep bar
    rows.append({"date": sweep_ts, "open": price_band, "high": sweep_bar_high,
                 "low": price_band * 0.998, "close": sweep_bar_close,
                 "volume": sweep_bar_volume})
    # Confirmation bar (sweep + 5min)
    confirm_ts = sweep_ts + pd.Timedelta(minutes=5)
    rows.append({"date": confirm_ts, "open": sweep_bar_close, "high": sweep_bar_close,
                 "low": confirm_bar_close, "close": confirm_bar_close,
                 "volume": prior_bars_avg_volume})

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def _ctx(df, symbol="TEST", cap_segment="small_cap"):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol, current_price=float(df.iloc[-1]["close"]), timestamp=last_ts,
        df_5m=df, session_date=last_ts.date(),
        pdh=150.0, pdl=140.0, pdc=145.0, cap_segment=cap_segment,
    )


# --------- Detection tests ---------

def test_canonical_failed_sweep_fires():
    """Bar pokes above Rs.150, closes back below; next bar confirms recovery."""
    df = _bars(sweep_bar_high=150.5, sweep_bar_close=149.5, confirm_bar_close=149.0,
               sweep_bar_volume=15000, price_band=150.0)
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert r.structure_detected, f"Should fire on canonical sweep, got: {r.rejection_reason}"
    e = r.events[0]
    assert e.side == "short"
    assert e.structure_type == "round_number_sweep_short"
    assert e.levels["round_number"] == 150.0


def test_rejects_price_outside_band():
    """Stock price Rs.300 is outside Rs.100-250 cell."""
    df = _bars(sweep_bar_high=300.5, sweep_bar_close=299.5, confirm_bar_close=299.0,
               sweep_bar_volume=15000, price_band=300.0)
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "outside cell band" in r.rejection_reason


def test_rejects_no_poke():
    """Bar high doesn't reach round-number poke threshold."""
    df = _bars(sweep_bar_high=149.5, sweep_bar_close=149.0, confirm_bar_close=148.5,
               sweep_bar_volume=15000, price_band=150.0)  # high < 150
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "No qualifying round-number sweep" in r.rejection_reason


def test_rejects_close_above_rn():
    """Bar pierces but closes ABOVE round number (no failure - genuine breakout)."""
    df = _bars(sweep_bar_high=151.0, sweep_bar_close=150.5, confirm_bar_close=150.8,
               sweep_bar_volume=15000, price_band=150.0)  # close > rn=150
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected


def test_rejects_no_confirm_recovery():
    """Sweep happens but next bar closes ABOVE the round number (no confirmation)."""
    df = _bars(sweep_bar_high=150.5, sweep_bar_close=149.5, confirm_bar_close=150.2,
               sweep_bar_volume=15000, price_band=150.0)
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected


def test_rejects_volume_below_threshold():
    """Sweep bar volume is below 2x session average."""
    df = _bars(sweep_bar_high=150.5, sweep_bar_close=149.5, confirm_bar_close=149.0,
               sweep_bar_volume=4000, prior_bars_avg_volume=5000, price_band=150.0)
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "Sweep volume" in r.rejection_reason


def test_rejects_outside_active_window():
    """Sweep at 13:00 IST is past active window end."""
    df = _bars(sweep_bar_high=150.5, sweep_bar_close=149.5, confirm_bar_close=149.0,
               sweep_bar_volume=15000, price_band=150.0, sweep_hhmm="13:00")
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "active window" in r.rejection_reason


# cap_segment early-reject removed: universe builders filter before dispatch


def test_latch_prevents_repeat_within_session():
    df = _bars(sweep_bar_high=150.5, sweep_bar_close=149.5, confirm_bar_close=149.0,
               sweep_bar_volume=15000, price_band=150.0)
    det = RoundNumberSweepShortStructure(_cfg())
    r1 = det.detect(_ctx(df))
    assert r1.structure_detected
    r2 = det.detect(_ctx(df))
    assert not r2.structure_detected
    assert "already fired" in r2.rejection_reason


# --------- Plan tests ---------

def test_plan_uses_full_exit_at_t1():
    """T1 should have qty_pct=1.0 (FULL exit), T2 should have qty_pct=0.0 (inert backup)."""
    df = _bars(sweep_bar_high=150.5, sweep_bar_close=149.5, confirm_bar_close=149.0,
               sweep_bar_volume=15000, price_band=150.0)
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    plan = det.plan_short_strategy(_ctx(df), event=r.events[0])
    assert plan is not None
    assert plan.side == "short"
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    # T1 = full exit
    assert targets[0]["name"] == "T1"
    assert targets[0]["qty_pct"] == 1.0
    # T2 = inert
    assert targets[1]["name"] == "T2"
    assert targets[1]["qty_pct"] == 0.0


def test_plan_sl_takes_farther_of_two_stops():
    """SL = max(sweep_high * 1.005, entry * 1.005)."""
    df = _bars(sweep_bar_high=150.5, sweep_bar_close=149.5, confirm_bar_close=149.0,
               sweep_bar_volume=15000, price_band=150.0)
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    plan = det.plan_short_strategy(_ctx(df), event=r.events[0])
    entry = plan.entry_price
    sweep_high = 150.5
    sl_from_sweep = sweep_high * 1.005
    sl_from_min = entry * 1.005
    expected_sl = max(sl_from_sweep, sl_from_min)
    assert plan.risk_params.hard_sl == pytest.approx(expected_sl, rel=1e-6)


def test_long_plan_returns_none():
    df = _bars(sweep_bar_high=150.5, sweep_bar_close=149.5, confirm_bar_close=149.0,
               sweep_bar_volume=15000, price_band=150.0)
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert det.plan_long_strategy(_ctx(df), event=r.events[0]) is None


def test_warmup_bars_do_not_pollute_detection():
    """df_5m may contain prior-day warmup. session_high check should be today-only."""
    today_df = _bars(sweep_bar_high=150.5, sweep_bar_close=149.5, confirm_bar_close=149.0,
                     sweep_bar_volume=15000, price_band=150.0)
    prior_ts = pd.date_range(start="2026-05-19 14:00", periods=10, freq="5min")
    prior_df = pd.DataFrame({
        "open": [200.0] * 10, "high": [205.0] * 10,
        "low": [195.0] * 10, "close": [200.0] * 10, "volume": [10000] * 10,
    }, index=prior_ts)
    df = pd.concat([prior_df, today_df])
    det = RoundNumberSweepShortStructure(_cfg())
    r = det.detect(_ctx(df))
    # The warmup bars are at price 200 (outside the today 150-band).
    # If detection accidentally uses warmup, it'd fail or use the wrong RN.
    # Today's bars are at price 150, so detection should work correctly.
    assert r.structure_detected, f"Warmup pollution broke detection: {r.rejection_reason}"
    assert r.events[0].levels["round_number"] == 150.0
