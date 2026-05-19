"""MIS-Unwind VWAP-Mean-Revert SHORT detector unit tests.

Aggregate-ships detector validated via 3-window LPGD cycle (PF 1.92/1.70/1.60).
"""
import pandas as pd
import pytest

from structures.mis_unwind_vwap_revert_short_structure import MisUnwindVwapRevertShortStructure
from structures.data_models import MarketContext


def _cfg(**overrides):
    base = {
        "_setup_name": "mis_unwind_vwap_revert_short",
        "enabled": True,
        "active_window_start": "14:30",
        "active_window_end": "15:10",
        "vwap_extension_pct": 0.5,
        "rsi_overbought": 65,
        "vol_ratio_min": 2.0,
        "allowed_cap_segments": ["small_cap", "mid_cap"],
        "sl_pct_above_entry": 0.4,
        "min_stop_pct": 0.5,
        "t1_r_multiple": 1.0,
        "t2_r_multiple": 2.0,
        "t1_partial_qty_pct": 0.0,
        "time_stop_at": "15:10",
        "min_bars_required": 8,
        "entry_zone_pct": 0.2,
        "entry_zone_mode": "symmetric",
        "min_stop_distance_pct": 0.3,
    }
    base.update(overrides)
    return base


def _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=5000, prior_vol_avg=1000,
          session_date=pd.Timestamp("2026-05-20").date(), signal_hhmm="14:45"):
    """Build today's bars: filler bars BEFORE signal_hhmm + the signal bar at signal_hhmm.

    Signal bar is always the LAST bar in the df (latest timestamp), so the
    detector's `df.index[-1]` correctly references it.
    """
    rows = []
    sig_ts = pd.Timestamp(f"{session_date} {signal_hhmm}:00")
    # Filler bars from 09:15 up to 5 min BEFORE signal_hhmm
    end_filler = sig_ts - pd.Timedelta(minutes=5)
    if end_filler >= pd.Timestamp(f"{session_date} 09:15:00"):
        filler_times = pd.date_range(start=f"{session_date} 09:15:00", end=end_filler, freq="5min")
        for t in filler_times:
            rows.append({"date": t, "open": vwap, "high": vwap * 1.005,
                         "low": vwap * 0.995, "close": vwap,
                         "volume": prior_vol_avg, "vwap": vwap, "rsi": 55.0})

    # Signal bar (last bar in df)
    rows.append({"date": sig_ts, "open": vwap, "high": close_px * 1.001,
                 "low": vwap, "close": close_px,
                 "volume": sweep_volume, "vwap": vwap, "rsi": float(rsi)})

    return pd.DataFrame(rows).set_index("date").sort_index()


def _ctx(df, symbol="TEST", cap_segment="small_cap"):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol, current_price=float(df.iloc[-1]["close"]), timestamp=last_ts,
        df_5m=df, session_date=last_ts.date(),
        pdh=100.0, pdl=95.0, pdc=98.0, cap_segment=cap_segment,
    )


# --------- Detection tests ---------

def test_canonical_signal_fires():
    """VWAP=100, close=101 (1% ext), RSI=70, vol_ratio=5x. Should fire."""
    df = _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=5000, prior_vol_avg=1000)
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert r.structure_detected, f"Should fire on canonical signal: {r.rejection_reason}"
    e = r.events[0]
    assert e.side == "short"
    assert e.structure_type == "mis_unwind_vwap_revert_short"
    assert e.context["vwap_ext_pct"] == pytest.approx(1.0, rel=1e-3)
    assert e.context["rsi"] == 70.0


def test_rejects_vwap_extension_too_small():
    """close=100.3 (only 0.3% extension)."""
    df = _bars(vwap=100.0, close_px=100.3, rsi=70, sweep_volume=5000, prior_vol_avg=1000)
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "VWAP extension" in r.rejection_reason


def test_rejects_rsi_not_overbought():
    """RSI=60, below 65 threshold."""
    df = _bars(vwap=100.0, close_px=101.0, rsi=60, sweep_volume=5000, prior_vol_avg=1000)
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "RSI" in r.rejection_reason


def test_rejects_volume_below_ratio():
    """Volume = 1.5x prior avg, below 2x threshold."""
    df = _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=1500, prior_vol_avg=1000)
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "vol_ratio" in r.rejection_reason


def test_rejects_outside_active_window():
    """Signal at 12:00 IST is before active window (14:30)."""
    df = _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=5000, prior_vol_avg=1000,
               signal_hhmm="12:00")
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected
    assert "active window" in r.rejection_reason


# cap_segment early-reject removed: universe builders filter before dispatch


def test_accepts_mid_cap():
    """Both small_cap AND mid_cap are allowed."""
    df = _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=5000, prior_vol_avg=1000)
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df, cap_segment="mid_cap"))
    assert r.structure_detected


def test_latch_prevents_repeat():
    df = _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=5000, prior_vol_avg=1000)
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r1 = det.detect(_ctx(df))
    assert r1.structure_detected
    r2 = det.detect(_ctx(df))
    assert not r2.structure_detected
    assert "already fired" in r2.rejection_reason


def test_missing_vwap_column_rejects():
    """If df_5m has no vwap column, reject gracefully."""
    df = _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=5000, prior_vol_avg=1000)
    df = df.drop(columns=["vwap"])
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert not r.structure_detected


# --------- Plan tests ---------

def test_plan_uses_ride_to_t2():
    df = _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=5000, prior_vol_avg=1000)
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df))
    plan = det.plan_short_strategy(_ctx(df), event=r.events[0])
    assert plan is not None
    assert plan.side == "short"
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["name"] == "T1"
    assert targets[0]["qty_pct"] == 0.0  # T1 informational
    assert targets[1]["name"] == "T2"
    assert targets[1]["qty_pct"] == 1.0  # T2 full exit


def test_plan_sl_at_min_stop_floor():
    """SL = max(entry*1.004, entry*1.005). min_stop_pct dominates here."""
    df = _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=5000, prior_vol_avg=1000)
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df))
    plan = det.plan_short_strategy(_ctx(df), event=r.events[0])
    entry = plan.entry_price
    # min_stop_pct is 0.5%, sl_pct_above is 0.4%. max(0.4, 0.5) = 0.5%.
    expected_sl = entry * 1.005
    assert plan.risk_params.hard_sl == pytest.approx(expected_sl, rel=1e-6)


def test_long_plan_returns_none():
    df = _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=5000, prior_vol_avg=1000)
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert det.plan_long_strategy(_ctx(df), event=r.events[0]) is None


def test_warmup_bars_do_not_pollute():
    """Prior-day warmup bars must not affect today's VWAP/RSI lookup."""
    today_df = _bars(vwap=100.0, close_px=101.0, rsi=70, sweep_volume=5000, prior_vol_avg=1000)
    prior_ts = pd.date_range(start="2026-05-19 14:00", periods=10, freq="5min")
    prior_df = pd.DataFrame({
        "open": [200.0] * 10, "high": [210.0] * 10,
        "low": [195.0] * 10, "close": [200.0] * 10, "volume": [10000] * 10,
        "vwap": [200.0] * 10, "rsi": [50.0] * 10,
    }, index=prior_ts)
    df = pd.concat([prior_df, today_df])
    det = MisUnwindVwapRevertShortStructure(_cfg())
    r = det.detect(_ctx(df))
    assert r.structure_detected, f"Warmup pollution broke detection: {r.rejection_reason}"
    assert r.events[0].levels["vwap"] == 100.0  # today's VWAP, not prior's 200
