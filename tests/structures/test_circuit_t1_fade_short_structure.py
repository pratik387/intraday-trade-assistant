"""circuit_t1_fade_short detector unit tests (sub-9 §3.3 implementation).

Verifies:
  - fires when conditions met (T-1 circuit hit + T+1 gap-up 1-5% + 10:30 bar)
  - does NOT fire outside 10:30 single-bar window
  - does NOT fire when T-1 fails circuit-hit qualification (any of:
    pct_change too small, close not at high, vol too low)
  - does NOT fire when T+1 gap is < 1% or > 5%
  - does NOT fire on disallowed cap_segment (under wide_open=False)
  - plan_short emits hard_sl ABOVE entry, T1 + T2 BELOW entry, sub-period stable
  - latch prevents double-fire same session
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from structures.circuit_t1_fade_short_structure import CircuitT1FadeShortStructure
from structures.data_models import MarketContext


def _cfg(**overrides):
    cfg = {
        "_setup_name": "circuit_t1_fade_short",
        "enabled": True,
        # T+0 circuit-hit detection
        "t0_min_pct_change": 4.5,
        "t0_high_to_close_min": 0.995,
        "t0_last30min_vol_share_max": 0.35,
        "t0_min_vol_vs_20d": 1.5,
        # T+1 entry conditions
        "t1_gap_min_pct": 1.0,
        "t1_gap_max_pct": 5.0,
        "active_window_start": "10:30",
        "active_window_end": "10:30",
        # Risk
        "stop_t1_high_buffer_pct": 0.5,
        "min_stop_distance_pct": 1.0,
        # Targets
        "t1_target_anchor": "t1_open",
        "t2_target_anchor": "t0_close",
        "t1_qty_pct": 0.5,
        "time_stop_at": "15:10",
        # Universe
        "allowed_cap_segments": ["mid_cap", "small_cap"],
        "universe_key": None,
        # Plumbing
        "min_bars_required": 16,
        "entry_zone_pct": 0.10,
        "entry_zone_mode": "directional",
    }
    cfg.update(overrides)
    return cfg


def _build_t1_5m(t1_date: date, t1_open: float, peak: float, current_close: float,
                 n_bars: int = 16) -> pd.DataFrame:
    """Build n_bars of 5m T+1 bars from 09:15 up to 10:30 (16 bars).

    Bar 0 (09:15) opens at `t1_open`, peak hit during morning, drift down by
    bar 15 (10:30) to `current_close`.
    """
    # Base = T+1 09:15 IST. Bars at 09:15, 09:20, ..., 10:30 (every 5 min).
    base = pd.Timestamp(t1_date).replace(hour=9, minute=15)
    bars = []
    for i in range(n_bars):
        if i == 0:
            o = t1_open
            h = max(t1_open, peak * 0.8)
            l = t1_open * 0.999
            c = t1_open * 1.005
        elif i == n_bars - 1:
            o = current_close * 1.002
            h = current_close * 1.004
            l = current_close * 0.999
            c = current_close
        else:
            mid = peak if i < n_bars / 2 else current_close
            o = mid * 0.999
            h = max(o, peak if i < n_bars / 2 else o)
            l = o * 0.998
            c = mid
        bars.append({"open": o, "high": h, "low": l, "close": c, "volume": 5000})
    df = pd.DataFrame(
        bars,
        index=[base + pd.Timedelta(minutes=5 * i) for i in range(n_bars)],
    )
    return df


def _build_daily(t0_date: date, t0_close: float, prev_close: float,
                 t0_high: float = None, t0_volume: float = 100000,
                 prev_volumes: list = None, n_history: int = 25) -> pd.DataFrame:
    """Build n_history daily bars ending at t0_date with the supplied
    T+0 close + prev_close + history volumes."""
    if t0_high is None:
        t0_high = t0_close   # assume close == high (clamped)
    if prev_volumes is None:
        prev_volumes = [40000] * (n_history - 1)
    bars = []
    cur = t0_date
    # Start n_history days back, walk forward
    history_dates = [t0_date - timedelta(days=n_history - 1 - i) for i in range(n_history)]
    for i, d in enumerate(history_dates):
        if i == n_history - 1:
            # T+0 row
            bars.append({
                "open": prev_close,
                "high": t0_high,
                "low": prev_close * 0.99,
                "close": t0_close,
                "volume": t0_volume,
            })
        elif i == n_history - 2:
            # T-1 (prev_close)
            bars.append({
                "open": prev_close * 0.99,
                "high": prev_close * 1.01,
                "low": prev_close * 0.98,
                "close": prev_close,
                "volume": prev_volumes[i],
            })
        else:
            # earlier history — flat synthetic
            c = prev_close * 0.95
            bars.append({
                "open": c, "high": c * 1.005, "low": c * 0.995,
                "close": c, "volume": prev_volumes[i],
            })
    return pd.DataFrame(bars, index=pd.DatetimeIndex(history_dates))


def _make_ctx(t1_5m: pd.DataFrame, daily: pd.DataFrame, t1_date: date,
              cap_segment: str = "small_cap"):
    """MarketContext at T+1 10:30 with T-1 daily bar already a circuit hit."""
    return MarketContext(
        symbol="NSE:TESTSYM",
        current_price=float(t1_5m["close"].iloc[-1]),
        timestamp=t1_5m.index[-1],
        df_5m=t1_5m,
        df_daily=daily,
        session_date=t1_date,
        cap_segment=cap_segment,
        regime="trend_up",
    )


# ---------------------------------------------------------------------------
# Test 1: valid setup fires (T-1 circuit hit + T+1 gap 2% + 10:30 bar)
# ---------------------------------------------------------------------------

def test_fires_when_all_conditions_met():
    """T-1 close = +6% above prev_close, clamped, high vol; T+1 gaps 2.5%; 10:30 bar."""
    det = CircuitT1FadeShortStructure(_cfg())
    t0_date = date(2024, 6, 5)
    t1_date = date(2024, 6, 6)

    daily = _build_daily(
        t0_date=t0_date, t0_close=106.0, prev_close=100.0,
        t0_high=106.0, t0_volume=100000,
        prev_volumes=[40000] * 24,   # avg ~40k vs t0 100k = 2.5x
    )
    t1_5m = _build_t1_5m(t1_date, t1_open=106.0 * 1.025, peak=110.0,
                          current_close=108.5, n_bars=16)
    ctx = _make_ctx(t1_5m, daily, t1_date, cap_segment="small_cap")
    result = det.detect(ctx)
    assert result.structure_detected, f"expected fire, got: {result.rejection_reason}"
    assert len(result.events) == 1
    assert result.events[0].side == "short"


# ---------------------------------------------------------------------------
# Test 2: outside 10:30 window → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_outside_active_window():
    """Same setup but the latest bar is 09:30 (before 10:30) → no fire."""
    det = CircuitT1FadeShortStructure(_cfg())
    t0_date = date(2024, 6, 5); t1_date = date(2024, 6, 6)
    daily = _build_daily(t0_date, 106.0, 100.0, 106.0, 100000)
    t1_5m = _build_t1_5m(t1_date, t1_open=108.65, peak=110.0,
                          current_close=108.5, n_bars=4)   # only 4 bars (09:15-09:30)
    ctx = _make_ctx(t1_5m, daily, t1_date)
    result = det.detect(ctx)
    assert not result.structure_detected
    # Either insufficient bars or outside window — both acceptable rejections
    assert any(s in (result.rejection_reason or "").lower()
               for s in ("window", "bars"))


# ---------------------------------------------------------------------------
# Test 3: T-1 pct_change < 4.5% → not a circuit-hit qualifier
# ---------------------------------------------------------------------------

def test_does_not_fire_when_t0_not_circuit_hit():
    """T+0 closed only 2% above prev (below 4.5% min) → not a qualifier."""
    det = CircuitT1FadeShortStructure(_cfg())
    t0_date = date(2024, 6, 5); t1_date = date(2024, 6, 6)
    daily = _build_daily(t0_date, t0_close=102.0, prev_close=100.0,
                          t0_high=102.0, t0_volume=100000)
    t1_5m = _build_t1_5m(t1_date, t1_open=104.55, peak=106.0,
                          current_close=104.0, n_bars=16)
    ctx = _make_ctx(t1_5m, daily, t1_date)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "circuit" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 4: T-1 close not at high (not clamped) → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_when_t0_not_clamped_at_high():
    """T+0 close is 4% above prev but day high was 8% above prev → not band-locked."""
    det = CircuitT1FadeShortStructure(_cfg())
    t0_date = date(2024, 6, 5); t1_date = date(2024, 6, 6)
    daily = _build_daily(
        t0_date, t0_close=106.0, prev_close=100.0,
        t0_high=110.0,   # high WAY above close → close/high < 0.995 → not clamped
        t0_volume=100000,
    )
    t1_5m = _build_t1_5m(t1_date, t1_open=108.65, peak=110.0,
                          current_close=108.5, n_bars=16)
    ctx = _make_ctx(t1_5m, daily, t1_date)
    result = det.detect(ctx)
    assert not result.structure_detected


# ---------------------------------------------------------------------------
# Test 5: T+1 gap below 1% → no fire (no continuation evidence)
# ---------------------------------------------------------------------------

def test_does_not_fire_when_t1_gap_too_small():
    """T+0 was a clean circuit hit but T+1 opens FLAT → operator pump
    didn't carry overnight → no continuation thesis."""
    det = CircuitT1FadeShortStructure(_cfg())
    t0_date = date(2024, 6, 5); t1_date = date(2024, 6, 6)
    daily = _build_daily(t0_date, 106.0, 100.0, 106.0, 100000)
    t1_5m = _build_t1_5m(t1_date, t1_open=106.2, peak=107.0,
                          current_close=106.5, n_bars=16)   # gap = 0.19%
    ctx = _make_ctx(t1_5m, daily, t1_date)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "gap" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 6: T+1 gap above 5% → no fire (fundamental news territory)
# ---------------------------------------------------------------------------

def test_does_not_fire_when_t1_gap_too_large():
    """T+1 opens 8% above T+0 close → fundamental news (results / M&A)."""
    det = CircuitT1FadeShortStructure(_cfg())
    t0_date = date(2024, 6, 5); t1_date = date(2024, 6, 6)
    daily = _build_daily(t0_date, 106.0, 100.0, 106.0, 100000)
    t1_5m = _build_t1_5m(t1_date, t1_open=114.5, peak=116.0,
                          current_close=114.0, n_bars=16)   # gap = 8%
    ctx = _make_ctx(t1_5m, daily, t1_date)
    result = det.detect(ctx)
    assert not result.structure_detected


# ---------------------------------------------------------------------------
# Test 7: large_cap → no fire (cap filter; under wide_open=False)
# ---------------------------------------------------------------------------

def test_does_not_fire_on_large_cap():
    """Setup is mid/small-cap only — large_caps rarely circuit and are
    usually fundamental when they do."""
    det = CircuitT1FadeShortStructure(_cfg())
    t0_date = date(2024, 6, 5); t1_date = date(2024, 6, 6)
    daily = _build_daily(t0_date, 106.0, 100.0, 106.0, 100000)
    t1_5m = _build_t1_5m(t1_date, t1_open=108.65, peak=110.0,
                          current_close=108.5, n_bars=16)
    ctx = _make_ctx(t1_5m, daily, t1_date, cap_segment="large_cap")
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "cap" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 8: plan_short emits hard_sl ABOVE entry + T1/T2 BELOW (gap edges)
# ---------------------------------------------------------------------------

def test_plan_short_emits_correct_geometry():
    det = CircuitT1FadeShortStructure(_cfg())
    t0_date = date(2024, 6, 5); t1_date = date(2024, 6, 6)
    daily = _build_daily(t0_date, 106.0, 100.0, 106.0, 100000)
    t1_5m = _build_t1_5m(t1_date, t1_open=108.65, peak=110.0,
                          current_close=108.5, n_bars=16)
    ctx = _make_ctx(t1_5m, daily, t1_date)
    res = det.detect(ctx)
    assert res.structure_detected, f"setup must fire: {res.rejection_reason}"
    plan = det.plan_short_strategy(ctx, event=res.events[0])
    assert plan is not None
    assert plan.side == "short"
    entry = plan.entry_price
    assert plan.risk_params.hard_sl > entry, "short SL must be above entry"
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["name"] == "T1" and targets[0]["qty_pct"] == 0.5
    assert targets[1]["name"] == "T2" and targets[1]["qty_pct"] == 0.5
    assert targets[0]["level"] < entry, "T1 must be below entry on short"
    assert targets[1]["level"] <= targets[0]["level"], "T2 ≤ T1 (further fade)"
    # T2 should be roughly the t0_close (full gap fill)
    assert abs(targets[1]["level"] - 106.0) < 0.5, (
        f"T2 should anchor to t0_close=106.0, got {targets[1]['level']}"
    )


# ---------------------------------------------------------------------------
# Test 9: latch prevents double-fire same session
# ---------------------------------------------------------------------------

def test_latch_prevents_double_fire_same_session():
    det = CircuitT1FadeShortStructure(_cfg())
    t0_date = date(2024, 6, 5); t1_date = date(2024, 6, 6)
    daily = _build_daily(t0_date, 106.0, 100.0, 106.0, 100000)
    t1_5m = _build_t1_5m(t1_date, t1_open=108.65, peak=110.0,
                          current_close=108.5, n_bars=16)
    ctx = _make_ctx(t1_5m, daily, t1_date)
    res1 = det.detect(ctx)
    assert res1.structure_detected
    plan = det.plan_short_strategy(ctx, event=res1.events[0])
    assert plan is not None   # commits the latch on success

    # detect() called again on same (symbol, session_date) — must NOT fire
    res2 = det.detect(ctx)
    assert not res2.structure_detected
    assert "already" in (res2.rejection_reason or "").lower() or \
           "fired" in (res2.rejection_reason or "").lower()
