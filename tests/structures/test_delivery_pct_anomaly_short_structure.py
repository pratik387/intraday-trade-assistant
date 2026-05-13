"""delivery_pct_anomaly_short detector unit tests (sub-9 round-7 cell-ship).

Verifies the SHORT-side detector for the validated delivery_pct anomaly:
T-1 delivery_pct < 20% + daily_return > 3% + ADV in [100, 1000]cr → fade
SHORT on the 09:30-10:00 5m confirmation bar (close < open, close < VWAP,
cross-day RVOL >= 1.0).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import structures.delivery_pct_anomaly_short_structure as detector_module
from structures.delivery_pct_anomaly_short_structure import (
    DeliveryPctAnomalyShortStructure,
)
from structures.data_models import MarketContext


@pytest.fixture(autouse=True)
def _stub_rvol_baseline(monkeypatch):
    """Stub the cross-day RVOL baseline lookup for all tests in this file.

    Tests synthesize df_5m without writing to data/cross_day_rvol/. The
    detector's _cross_day_rvol() now reads baseline from a precomputed
    parquet via services.cross_day_rvol_enrichment.get_baseline_vol —
    which would return None in tests, blocking every fire. Replace with
    a simple stub that uses the previous bar's volume as the baseline so
    test data still exercises the gate semantics (today_vol/baseline >= 1).
    """
    import services.cross_day_rvol_enrichment as _enrich

    def _stub_get_baseline_vol(symbol, session_date, hhmm):
        # Constant baseline = 1000; tests build today_vol > 1000 to fire.
        return 1000.0

    monkeypatch.setattr(_enrich, "get_baseline_vol", _stub_get_baseline_vol)


def _cfg(**overrides):
    cfg = {
        "_setup_name": "delivery_pct_anomaly_short",
        "enabled": True,
        "active_window_start": "09:30",
        "active_window_end": "10:00",
        "time_stop_at": "13:00",
        "delivery_pct_max": 20.0,
        "min_prior_day_return_pct": 3.0,
        "min_adv_inr_cr": 100.0,
        "max_adv_inr_cr": 1000.0,
        "min_gap_pct": -2.0,
        "max_gap_pct": 3.0,
        "min_volume_ratio_to_20d_avg": 1.0,
        "stop_open_high_buffer_pct": 0.5,
        "stop_tday_close_buffer_pct": 1.2,
        "t1_r_multiple": 0.25,
        "t2_r_multiple": 0.75,
        "t1_partial_qty_pct": 0.5,
        "min_stop_distance_pct": 0.3,
        "min_bars_required": 4,
        "entry_zone_pct": 0.10,
        "entry_zone_mode": "directional",
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def _disable_wide_open(monkeypatch):
    # detector calls services.config_loader.is_wide_open_for_setup at runtime
    # (per-setup flag). Force False so cell filters (ADV, gap, active_window)
    # remain enforced in unit tests regardless of configuration.json contents.
    import services.config_loader as _cfg_mod
    monkeypatch.setattr(_cfg_mod, "is_wide_open_for_setup", lambda *a, **kw: False)


def _build_5m(
    session_date: date,
    n_today_bars: int = 4,
    gap_pct: float = 0.5,
    pdc: float = 100.0,
    confirmation_red: bool = True,
    confirmation_below_vwap: bool = True,
    confirmation_volume_mult: float = 2.0,
    n_prior_sessions: int = 22,
) -> pd.DataFrame:
    """Build a 5m bar tail spanning prior sessions + today.

    n_today_bars = 4 → today's bars at 09:15, 09:20, 09:25, 09:30.
    Prior sessions populate same-time-of-day history for cross-day RVOL.

    confirmation_volume_mult: multiplies the 09:30 bar's volume vs prior-day
    same-time average (which is set to 1000). Default 2.0 satisfies RVOL>=1.

    Trading-calendar gymnastics not required: this fixture builds N
    consecutive calendar days with same-time-of-day bars. Index-level date
    comparison + date-time match matters for both df_5m's session filter
    and the cross-day RVOL.
    """
    base_open = pdc * (1.0 + gap_pct / 100.0)

    rows = []
    # Prior sessions — each contributes one 09:30 same-time-of-day bar with
    # vol=1000 so cross-day mean is exactly 1000.
    for k in range(n_prior_sessions, 0, -1):
        d = pd.Timestamp(session_date) - pd.Timedelta(days=k)
        ts_0930 = d.replace(hour=9, minute=30)
        rows.append({
            "ts": ts_0930, "open": pdc, "high": pdc * 1.001,
            "low": pdc * 0.999, "close": pdc, "volume": 1000,
        })

    # Today's bars at 09:15, 09:20, 09:25, 09:30, ...
    today = pd.Timestamp(session_date)
    bar_minute_offsets = [15, 20, 25, 30, 35, 40][:n_today_bars]
    for i, mo in enumerate(bar_minute_offsets):
        ts = today.replace(hour=9, minute=mo)
        if i == 0:
            o = base_open
            h = base_open * 1.005   # morning swing high
            l = base_open * 0.998
            c = base_open * 1.002   # green-ish first bar so VWAP starts above PDC
            v = 5000
        elif i == n_today_bars - 1:
            # Confirmation bar (last bar of today fed to detector)
            if confirmation_red:
                # Red bar: close < open
                if confirmation_below_vwap:
                    # Drop deeper to get below VWAP
                    o = base_open * 1.000
                    c = base_open * 0.985
                    h = base_open * 1.001
                    l = base_open * 0.984
                else:
                    # Red but stay above VWAP
                    o = base_open * 1.010
                    c = base_open * 1.008
                    h = base_open * 1.012
                    l = base_open * 1.007
            else:
                # Green confirmation candle (should reject)
                o = base_open * 0.998
                c = base_open * 1.005
                h = base_open * 1.007
                l = base_open * 0.997
            v = int(1000 * confirmation_volume_mult)
        else:
            o = base_open * 1.001
            c = base_open * 1.000
            h = base_open * 1.003
            l = base_open * 0.998
            v = 5000
        rows.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})

    df = pd.DataFrame(rows).set_index("ts").sort_index()
    return df


def _build_daily(
    session_date: date,
    n_days: int = 30,
    delivery_pct: float = 12.0,
    daily_return_pct: float = 5.0,
    adv_target_cr: float = 300.0,
    pdc: float = 100.0,
    drop_delivery_col: bool = False,
) -> pd.DataFrame:
    """Build a daily bar tail with a `delivery_pct` column.

    The most recent bar is T-1 (one day before session_date). The bar at
    T-1 has delivery_pct = `delivery_pct` and close/prev_close producing
    `daily_return_pct`. ADV-20d × close ≈ adv_target_cr (in Cr).

    Setting drop_delivery_col=True simulates upstream pipeline NOT having
    enriched daily_df (detector should silently skip).
    """
    rows = []
    # Daily prev_close (so T-1 daily_return = (pdc / prev_close - 1))
    prev_close = pdc / (1.0 + daily_return_pct / 100.0)
    # ADV target: close * volume / 1e7 = adv_target_cr → volume = cr * 1e7 / close
    target_vol = adv_target_cr * 1e7 / prev_close
    for k in range(n_days, 1, -1):
        d = pd.Timestamp(session_date) - pd.Timedelta(days=k)
        rows.append({
            "ts": d, "open": prev_close, "high": prev_close * 1.001,
            "low": prev_close * 0.999, "close": prev_close,
            "volume": target_vol,
            "delivery_pct": 30.0,   # baseline for prior days (not anomaly)
        })
    # T-1 row
    t_minus_1 = pd.Timestamp(session_date) - pd.Timedelta(days=1)
    rows.append({
        "ts": t_minus_1, "open": pdc * 0.99, "high": pdc * 1.005,
        "low": pdc * 0.985, "close": pdc, "volume": target_vol,
        "delivery_pct": delivery_pct,
    })
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    if drop_delivery_col:
        df = df.drop(columns=["delivery_pct"])
    return df


def _make_ctx(
    df_5m: pd.DataFrame,
    df_daily: pd.DataFrame,
    sd: date,
    cap_segment: str = "mid_cap",
    regime: str = "trend_up",
):
    return MarketContext(
        symbol="NSE:TESTSYM",
        current_price=float(df_5m["close"].iloc[-1]),
        timestamp=df_5m.index[-1],
        df_5m=df_5m,
        df_daily=df_daily,
        session_date=sd,
        cap_segment=cap_segment,
        regime=regime,
        indicators={"atr": 1.0},
        pdc=100.0,
    )


# --------------------------------------------------------------------------
# Test 1: fires when all conditions met
# --------------------------------------------------------------------------

def test_fires_when_all_conditions_met():
    det = DeliveryPctAnomalyShortStructure(_cfg())
    sd = date(2024, 6, 6)
    df_5m = _build_5m(sd, n_today_bars=4, gap_pct=0.5)
    df_daily = _build_daily(sd, delivery_pct=12.0, daily_return_pct=5.0,
                             adv_target_cr=300.0)
    ctx = _make_ctx(df_5m, df_daily, sd)
    result = det.detect(ctx)
    assert result.structure_detected, f"expected fire: {result.rejection_reason}"
    assert result.events[0].side == "short"


# --------------------------------------------------------------------------
# Test 2: outside active window → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_outside_active_window():
    det = DeliveryPctAnomalyShortStructure(_cfg())
    sd = date(2024, 6, 6)
    # Last bar at 09:25 (n=3) — before 09:30 window start. min_bars_required=4
    # so make 4 bars but include one before window: shift the today bars to
    # 09:00-09:15 to land outside the window. Easiest: build 4 bars where
    # last is 09:35 (still inside) — instead start one earlier and run 4
    # bars ending at 09:25 by changing min_bars to 3.
    cfg = _cfg(min_bars_required=3)
    det = DeliveryPctAnomalyShortStructure(cfg)
    df_5m = _build_5m(sd, n_today_bars=3, gap_pct=0.5)  # ends at 09:25
    df_daily = _build_daily(sd)
    ctx = _make_ctx(df_5m, df_daily, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "window" in (result.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 3: delivery_pct >= 20 → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_when_delivery_pct_above_threshold():
    det = DeliveryPctAnomalyShortStructure(_cfg())
    sd = date(2024, 6, 6)
    df_5m = _build_5m(sd, n_today_bars=4, gap_pct=0.5)
    df_daily = _build_daily(sd, delivery_pct=25.0, daily_return_pct=5.0,
                             adv_target_cr=300.0)  # delivery >= 20
    ctx = _make_ctx(df_5m, df_daily, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "anomaly" in (result.rejection_reason or "").lower() \
        or "qualifier" in (result.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 4: daily_return <= 3% → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_when_daily_return_below_threshold():
    det = DeliveryPctAnomalyShortStructure(_cfg())
    sd = date(2024, 6, 6)
    df_5m = _build_5m(sd, n_today_bars=4, gap_pct=0.5)
    df_daily = _build_daily(sd, delivery_pct=12.0, daily_return_pct=2.0,
                             adv_target_cr=300.0)  # 2% < 3% min
    ctx = _make_ctx(df_5m, df_daily, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "qualifier" in (result.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 5: ADV outside [100, 1000]cr → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_when_adv_outside_band():
    det = DeliveryPctAnomalyShortStructure(_cfg())
    sd = date(2024, 6, 6)
    df_5m = _build_5m(sd, n_today_bars=4, gap_pct=0.5)
    # ADV = 50cr → below 100cr min
    df_daily = _build_daily(sd, delivery_pct=12.0, daily_return_pct=5.0,
                             adv_target_cr=50.0)
    ctx = _make_ctx(df_5m, df_daily, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "qualifier" in (result.rejection_reason or "").lower()

    # Also test above max
    det2 = DeliveryPctAnomalyShortStructure(_cfg())
    df_daily2 = _build_daily(sd, delivery_pct=12.0, daily_return_pct=5.0,
                              adv_target_cr=2000.0)  # > 1000cr
    ctx2 = _make_ctx(df_5m, df_daily2, sd)
    r2 = det2.detect(ctx2)
    assert not r2.structure_detected


# --------------------------------------------------------------------------
# Test 6: gap_pct outside [-2, +3] → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_when_gap_outside_band():
    det = DeliveryPctAnomalyShortStructure(_cfg())
    sd = date(2024, 6, 6)
    # Below min: gap_pct = -3% < -2%
    df_5m = _build_5m(sd, n_today_bars=4, gap_pct=-3.0)
    df_daily = _build_daily(sd)
    ctx = _make_ctx(df_5m, df_daily, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "gap_pct" in (result.rejection_reason or "").lower()

    # Above max: gap_pct = +5% > +3%
    det2 = DeliveryPctAnomalyShortStructure(_cfg())
    df_5m_2 = _build_5m(sd, n_today_bars=4, gap_pct=5.0)
    ctx2 = _make_ctx(df_5m_2, df_daily, sd)
    r2 = det2.detect(ctx2)
    assert not r2.structure_detected
    assert "gap_pct" in (r2.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 7: confirmation candle missing one criterion → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_when_confirmation_candle_invalid():
    sd = date(2024, 6, 6)
    df_daily = _build_daily(sd)

    # 7a: green bar (close > open)
    det = DeliveryPctAnomalyShortStructure(_cfg())
    df_5m = _build_5m(sd, n_today_bars=4, confirmation_red=False)
    ctx = _make_ctx(df_5m, df_daily, sd)
    r = det.detect(ctx)
    assert not r.structure_detected
    assert any(s in (r.rejection_reason or "").lower()
               for s in ("red", "vwap", "rvol"))

    # 7b: red but close > VWAP (still bullish)
    det2 = DeliveryPctAnomalyShortStructure(_cfg())
    df_5m_2 = _build_5m(sd, n_today_bars=4, confirmation_red=True,
                        confirmation_below_vwap=False)
    ctx2 = _make_ctx(df_5m_2, df_daily, sd)
    r2 = det2.detect(ctx2)
    assert not r2.structure_detected
    assert any(s in (r2.rejection_reason or "").lower()
               for s in ("vwap", "rvol", "red"))

    # 7c: rvol < 1.0 (volume dries up — too quiet for confirmation)
    det3 = DeliveryPctAnomalyShortStructure(_cfg())
    df_5m_3 = _build_5m(sd, n_today_bars=4, confirmation_red=True,
                        confirmation_below_vwap=True,
                        confirmation_volume_mult=0.5)
    ctx3 = _make_ctx(df_5m_3, df_daily, sd)
    r3 = det3.detect(ctx3)
    assert not r3.structure_detected
    assert "rvol" in (r3.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 8: plan_short geometry — SL > entry, T1 < entry, T2 < T1
# --------------------------------------------------------------------------

def test_plan_short_geometry():
    det = DeliveryPctAnomalyShortStructure(_cfg())
    sd = date(2024, 6, 6)
    df_5m = _build_5m(sd, n_today_bars=4, gap_pct=0.5)
    df_daily = _build_daily(sd)
    ctx = _make_ctx(df_5m, df_daily, sd)
    result = det.detect(ctx)
    assert result.structure_detected, f"setup pre-cond: {result.rejection_reason}"
    plan = det.plan_short_strategy(ctx, result.events[0])
    assert plan is not None
    entry = plan.entry_price
    sl = plan.risk_params.hard_sl
    t1 = plan.exit_levels.targets[0]["level"]
    t2 = plan.exit_levels.targets[1]["level"]
    assert sl > entry, f"SL {sl} must be ABOVE short entry {entry}"
    assert t1 < entry, f"T1 {t1} must be BELOW short entry {entry}"
    assert t2 < t1, f"T2 {t2} must be FURTHER BELOW T1 {t1}"
    assert plan.target_anchor_type == "arithmetic"
    # plan_long returns None (SHORT-only)
    assert det.plan_long_strategy(ctx, result.events[0]) is None


# --------------------------------------------------------------------------
# Test 9: latch prevents double-fire same session
# --------------------------------------------------------------------------

def test_latch_prevents_double_fire():
    det = DeliveryPctAnomalyShortStructure(_cfg())
    sd = date(2024, 6, 6)
    df_5m = _build_5m(sd, n_today_bars=4, gap_pct=0.5)
    df_daily = _build_daily(sd)
    ctx = _make_ctx(df_5m, df_daily, sd)
    r1 = det.detect(ctx)
    assert r1.structure_detected
    plan = det.plan_short_strategy(ctx, r1.events[0])
    assert plan is not None
    r2 = det.detect(ctx)
    assert not r2.structure_detected
    assert "already fired" in (r2.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 10: wide_open does NOT bypass detector signal
# --------------------------------------------------------------------------
# This detector has no meta-filters (no cap_segment / regime / universe key
# gates) for wide_open to legitimately bypass — the T-1 anomaly + gap +
# confirmation candle + RVOL gates ARE the setup signal. A prior version of
# this test asserted that wide_open bypassed the signal too; that
# behavior caused 7000x trade-count inflation in OCI capture (every
# symbol fired every bar in the 09:30-10:00 window). The detector now
# applies the full signal regardless of wide_open_mode.

def test_wide_open_does_not_bypass_signal(monkeypatch):
    monkeypatch.setattr(detector_module, "_is_wide_open", lambda: True)
    det = DeliveryPctAnomalyShortStructure(_cfg())
    sd = date(2024, 6, 6)
    # Failing conditions: delivery_pct=25 (>=20 threshold) — should still
    # reject under wide_open because the signal IS the setup identity.
    df_5m = _build_5m(sd, n_today_bars=4, gap_pct=0.5)
    df_daily = _build_daily(sd, delivery_pct=25.0)
    ctx = _make_ctx(df_5m, df_daily, sd)
    result = det.detect(ctx)
    assert not result.structure_detected, \
        "wide_open must NOT bypass T-1 anomaly signal"
    assert "qualifier" in (result.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 11 (extra): silently skips when delivery_pct column absent (upstream
# pipeline not yet enriching daily_df). Pipe is still in development; the
# detector must not crash, just decline to fire.
# --------------------------------------------------------------------------

def test_does_not_fire_when_delivery_pct_column_missing():
    det = DeliveryPctAnomalyShortStructure(_cfg())
    sd = date(2024, 6, 6)
    df_5m = _build_5m(sd, n_today_bars=4, gap_pct=0.5)
    df_daily = _build_daily(sd, drop_delivery_col=True)
    ctx = _make_ctx(df_5m, df_daily, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "qualifier" in (result.rejection_reason or "").lower()
