"""capitulation_long_morning detector unit tests (sub-9 round-6 cell-ship).

Verifies the LONG-side mirror of gap_fade_short with cell filters
(trend_down × mid_cap × liq=10-30cr) locked from sanity cell-select.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import structures.capitulation_long_morning_structure as detector_module
from structures.capitulation_long_morning_structure import (
    CapitulationLongMorningStructure,
)
from structures.data_models import MarketContext


def _cfg(**overrides):
    cfg = {
        "_setup_name": "capitulation_long_morning",
        "enabled": True,
        "gap_min_pct": 1.5,
        "gap_max_pct": 8.0,
        "lower_wick_ratio_min": 0.5,
        "body_size_max_pct": 30.0,
        "active_window_start": "09:25",
        "active_window_end": "10:00",
        "stop_pct": 0.01,
        "gap_low_buffer_pct": 0.5,
        "atr_stop_multiple": 1.5,
        "t1_r_multiple": 1.0,
        "t2_r_multiple": 2.0,
        "t1_qty_pct": 0.5,
        "allowed_cap_segments": ["mid_cap"],
        "allowed_regimes": ["trend_down"],
        "allowed_liquidity_band_cr": [10.0, 30.0],
        "universe_key": None,
        "min_bars_required": 4,
        "entry_zone_pct": 0.10,
        "entry_zone_mode": "directional",
        "min_stop_distance_pct": 0.3,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def _disable_wide_open(monkeypatch):
    monkeypatch.setattr(detector_module, "_is_wide_open", lambda: False)


def _build_5m(session_date: date, gap_pct: float, n_bars: int = 4,
              red_first: bool = True, exhaustion_bar: bool = True,
              fresh_low: bool = False,
              first_bar_low_pct: float = 1.0) -> pd.DataFrame:
    """Build 5m bars from 09:15 to 09:15 + 5*(n-1) min.

    n=4 → 09:30 last bar (in window).
    Bar 0 (09:15) opens at prev_close × (1 + gap_pct/100); falls 1% intrabar
    creating gap_low.
    Bar n-1: optionally an exhaustion candle (lower wick + small body + green).
    """
    base = pd.Timestamp(session_date).replace(hour=9, minute=15)
    prev_close = 100.0
    first_open = prev_close * (1.0 + gap_pct / 100.0)
    bars = []
    for i in range(n_bars):
        if i == 0:
            o = first_open
            l = first_open * (1.0 - first_bar_low_pct / 100.0)
            h = first_open * 1.001
            c = first_open * (1.0 - 0.005) if red_first else first_open * 1.005
        elif i == n_bars - 1:
            if exhaustion_bar:
                # Green bar with lower wick:
                # body = open(0.998) -> close(1.005), body_bottom = 0.998
                # bar_low must be BELOW body_bottom (to create wick) but ABOVE
                # 09:15 low (to avoid fresh-low rejection).
                o = first_open * 0.998
                if fresh_low:
                    l = first_open * (1 - first_bar_low_pct / 100.0) * 0.999
                else:
                    l = first_open * 0.993   # below body bottom, above 09:15 low
                c = first_open * 1.005   # green
                h = c * 1.001
            else:
                # Red continuation
                o = first_open * 0.998
                l = first_open * 0.985
                c = first_open * 0.99
                h = first_open * 0.999
        else:
            # Drift down through the morning
            o = first_open * 0.998
            l = first_open * 0.99
            c = first_open * 0.995
            h = first_open * 0.999
        bars.append({"open": o, "high": h, "low": l, "close": c, "volume": 5000})
    df = pd.DataFrame(
        bars, index=[base + pd.Timedelta(minutes=5 * i) for i in range(n_bars)],
    )
    df["adv_20d_cr"] = 20.0   # in band
    return df


def _build_daily(session_date: date, prev_close: float = 100.0) -> pd.DataFrame:
    bars = [{"open": prev_close, "high": prev_close, "low": prev_close,
             "close": prev_close, "volume": 100000}]
    return pd.DataFrame(
        bars, index=pd.DatetimeIndex([pd.Timestamp(session_date) - pd.Timedelta(days=1)])
    )


def _make_ctx(df, sd, cap_segment="mid_cap", regime="trend_down"):
    return MarketContext(
        symbol="NSE:TESTSYM",
        current_price=float(df["close"].iloc[-1]),
        timestamp=df.index[-1],
        df_5m=df,
        df_daily=_build_daily(sd, 100.0),
        session_date=sd,
        cap_segment=cap_segment,
        regime=regime,
        indicators={"atr": 0.5},
        pdc=100.0,
    )


# --------------------------------------------------------------------------
# Test 1: fires when all conditions met
# --------------------------------------------------------------------------

def test_fires_when_all_conditions_met():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, gap_pct=-3.0, n_bars=4, exhaustion_bar=True, fresh_low=False)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert result.structure_detected, f"expected fire: {result.rejection_reason}"
    assert result.events[0].side == "long"


# --------------------------------------------------------------------------
# Test 2: outside active window → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_outside_active_window():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    # n=2 → last bar at 09:20 (before 09:25 window start)
    df = _build_5m(sd, gap_pct=-3.0, n_bars=2)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert any(s in (result.rejection_reason or "").lower()
               for s in ("window", "bars"))


# --------------------------------------------------------------------------
# Test 3: gap not deep enough → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_when_gap_below_threshold():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, gap_pct=-1.0, n_bars=4)   # |gap|=1% < 1.5% min
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "gap_pct" in (result.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 4: gap too deep (fundamental shock) → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_when_gap_exceeds_max():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, gap_pct=-9.0, n_bars=4)   # |gap|=9% > 8% max
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "gap_pct" in (result.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 5: not green confirmation → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_when_not_green():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, gap_pct=-3.0, n_bars=4, exhaustion_bar=False)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    # Either fresh-low or not-green
    assert any(s in (result.rejection_reason or "").lower()
               for s in ("green", "fresh low"))


# --------------------------------------------------------------------------
# Test 6: fresh low → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_when_fresh_low():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, gap_pct=-3.0, n_bars=4,
                   exhaustion_bar=True, fresh_low=True,
                   first_bar_low_pct=0.5)   # 09:15 low only 0.5% below open
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "fresh low" in (result.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 7: disallowed cap → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_on_disallowed_cap():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, gap_pct=-3.0, n_bars=4)
    ctx = _make_ctx(df, sd, cap_segment="large_cap")  # cell-locked: mid_cap only
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "cap segment" in (result.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 8: disallowed regime → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_on_disallowed_regime():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, gap_pct=-3.0, n_bars=4)
    ctx = _make_ctx(df, sd, regime="trend_up")  # cell-locked: trend_down only
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "regime" in (result.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 9: liquidity band — too liquid → no fire
# --------------------------------------------------------------------------

def test_does_not_fire_when_liquidity_above_band():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, gap_pct=-3.0, n_bars=4)
    df["adv_20d_cr"] = 50.0   # outside [10, 30]
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "adv_20d_cr" in (result.rejection_reason or "").lower()


# --------------------------------------------------------------------------
# Test 10: plan_long geometry
# --------------------------------------------------------------------------

def test_plan_long_geometry():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, gap_pct=-3.0, n_bars=4)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert result.structure_detected
    plan = det.plan_long_strategy(ctx, result.events[0])
    assert plan is not None
    entry = plan.entry_price
    sl = plan.risk_params.hard_sl
    t1 = plan.exit_levels.targets[0]["level"]
    t2 = plan.exit_levels.targets[1]["level"]
    assert sl < entry, f"SL {sl} must be BELOW long entry {entry}"
    assert t1 > entry, f"T1 {t1} must be ABOVE long entry {entry}"
    assert t2 > t1
    assert plan.target_anchor_type == "arithmetic"
    # plan_short returns None (LONG-only)
    assert det.plan_short_strategy(ctx, result.events[0]) is None


# --------------------------------------------------------------------------
# Test 11: latch prevents double-fire same session
# --------------------------------------------------------------------------

def test_latch_prevents_double_fire():
    det = CapitulationLongMorningStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, gap_pct=-3.0, n_bars=4)
    ctx = _make_ctx(df, sd)
    r1 = det.detect(ctx)
    assert r1.structure_detected
    plan = det.plan_long_strategy(ctx, r1.events[0])
    assert plan is not None
    r2 = det.detect(ctx)
    assert not r2.structure_detected
    assert "already fired" in (r2.rejection_reason or "").lower()
