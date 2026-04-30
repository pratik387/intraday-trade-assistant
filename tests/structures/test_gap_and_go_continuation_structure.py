"""Gap-and-Go Continuation detector unit tests.

Mechanic per
specs/2026-04-29-gap_and_go_continuation-plan.md and
specs/2026-04-29-research-new-indian-setup-candidates.md (Candidate 5).

Two-bar state machine:
  bar 0 (09:15): qualifier — gap_pct >= threshold + PDC vs 20SMA aligned
                 with gap direction + bar 0 prints new intraday extreme +
                 first-bar volume >= ratio × 14d baseline
  bar 1+:       trigger — last_bar tags bar 0's high (long) / low (short)

Tests cover:
  - Happy-path long + short fires
  - Sub-threshold gap → no fire
  - Daily-trend filter blocks counter-trend gaps (PDC against 20SMA)
  - First-bar volume below threshold → no fire
  - Bar 0 fails to print new extreme → no fire
  - Trigger bar fails to tag bar 0's extreme → no fire
  - Outside active window → reject
  - Symbol outside fno_liquid_200 universe → reject
  - small_cap → reject (regime-complement to gap_fade_short)
  - First-trigger latch prevents same-day double-fire
  - plan_*_strategy emits hard_sl + tiered T1 (1R) / T2 (2R) in correct direction
  - Wide_open bypasses daily-trend + first-bar volume filters
"""
from __future__ import annotations

import pandas as pd
import pytest

from structures.gap_and_go_continuation_structure import GapAndGoContinuationStructure
from structures.data_models import MarketContext


def _cfg(**overrides):
    cfg = {
        "_setup_name": "gap_and_go_continuation",
        "enabled": True,
        "active_window_start": "09:15",
        "active_window_end": "09:30",
        "gap_threshold_pct": 1.0,
        "daily_trend_lookback_days": 20,
        "daily_trend_min_distance_pct": 0.0,
        "min_first_bar_volume_ratio": 1.2,
        "volume_baseline_lookback_days": 14,
        "t1_target_r": 1.0,
        "t2_target_r": 2.0,
        "t1_qty_pct": 0.5,
        "stop_below_first_bar_low_buffer_pct": 0.05,
        "allowed_sides": ["long", "short"],
        "allowed_cap_segments": ["large_cap", "mid_cap"],
        "universe_key": "fno_liquid_200",
        "min_bars_required": 2,
        "entry_zone_pct": 0.15,
        "entry_zone_mode": "directional",
        "min_stop_distance_pct": 0.3,
    }
    cfg.update(overrides)
    return cfg


def _build_daily_df(pdc=100.0, sma20=98.0, n=21):
    """Build a 21-row daily df ending at 2025-01-07.

    Last row's close = pdc; mean of last `daily_trend_lookback_days` (20) =
    sma20 — i.e. the closes are arranged so the lookback mean equals sma20
    while the last bar = pdc.
    """
    end = pd.Timestamp("2025-01-07")
    idx = pd.date_range(end - pd.Timedelta(days=n - 1), periods=n, freq="D")
    # Set 20 prior closes to (sma20 * 20 - pdc) / 19 so mean of last 20 ≈ sma20
    # ... actually simpler: set first 19 prior closes to a value that makes the
    # mean of the last 20 close to sma20 even with the final pdc.
    # mean_target = sma20; sum_target = sma20 * 20; last close = pdc.
    # First 19 closes contribute (20 * sma20 - pdc); each ≈ (20*sma20 - pdc)/19.
    base = (20 * sma20 - pdc) / 19.0
    closes = [base] * 19 + [pdc, pdc]   # last 2 rows: prior day + pdc
    # Adjust so the LAST 20 rows have mean = sma20 exactly.
    # last_20 = closes[-20:] = [base]*18 + [pdc, pdc]; sum = 18*base + 2*pdc.
    # We want sum = 20*sma20 → 18*base + 2*pdc = 20*sma20 → base = (20*sma20 - 2*pdc)/18
    base = (20 * sma20 - 2 * pdc) / 18.0
    closes = [base] * 19 + [pdc, pdc]
    return pd.DataFrame({
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,    # baseline volume
    }, index=idx)


def _build_long_session(
    pdc=100.0,
    gap_pct=1.5,
    sma20=98.0,
    first_bar_vol=80_000,
    baseline_vol=10_000,    # 09:15 mean baseline (per-bar, not daily)
    n_subsequent=2,
    new_high_above_open_pct=0.5,
    trigger_at_or_above_first_high=True,
    session_date_str="2025-01-08",
):
    """Build a 5m intraday DataFrame for the long-side setup.

    bar 0 (09:15): opens at pdc * (1 + gap_pct/100); prints new high above
      open by `new_high_above_open_pct` of open; volume = first_bar_vol.
    bars 1..n_subsequent (09:20, 09:25): each bar's high tags or exceeds
      bar 0's high if `trigger_at_or_above_first_high=True`, else stays
      strictly below.
    """
    bar0_open = pdc * (1.0 + gap_pct / 100.0)
    bar0_high = bar0_open * (1.0 + new_high_above_open_pct / 100.0)
    bar0_close = (bar0_open + bar0_high) / 2.0
    bar0_low = bar0_open - bar0_open * 0.001

    rows = [{
        "ts": pd.Timestamp(f"{session_date_str} 09:15:00"),
        "open": bar0_open, "high": bar0_high, "low": bar0_low,
        "close": bar0_close, "volume": first_bar_vol,
    }]
    for i in range(1, n_subsequent + 1):
        ts = pd.Timestamp(f"{session_date_str} 09:15:00") + pd.Timedelta(minutes=5 * i)
        if trigger_at_or_above_first_high:
            high = bar0_high * 1.001
        else:
            high = bar0_high * 0.999
        rows.append({
            "ts": ts,
            "open": bar0_close,
            "high": high,
            "low": bar0_close - bar0_close * 0.001,
            "close": bar0_close,
            "volume": baseline_vol,
        })
    return pd.DataFrame(rows).set_index("ts")


def _build_short_session(
    pdc=100.0,
    gap_pct=-1.5,
    sma20=102.0,
    first_bar_vol=80_000,
    baseline_vol=10_000,
    n_subsequent=2,
    new_low_below_open_pct=0.5,
    trigger_at_or_below_first_low=True,
    session_date_str="2025-01-08",
):
    """Mirror of _build_long_session for the gap-down short setup."""
    bar0_open = pdc * (1.0 + gap_pct / 100.0)   # gap_pct negative → bar0_open < pdc
    bar0_low = bar0_open * (1.0 - new_low_below_open_pct / 100.0)
    bar0_close = (bar0_open + bar0_low) / 2.0
    bar0_high = bar0_open + bar0_open * 0.001

    rows = [{
        "ts": pd.Timestamp(f"{session_date_str} 09:15:00"),
        "open": bar0_open, "high": bar0_high, "low": bar0_low,
        "close": bar0_close, "volume": first_bar_vol,
    }]
    for i in range(1, n_subsequent + 1):
        ts = pd.Timestamp(f"{session_date_str} 09:15:00") + pd.Timedelta(minutes=5 * i)
        if trigger_at_or_below_first_low:
            low = bar0_low * 0.999
        else:
            low = bar0_low * 1.001
        rows.append({
            "ts": ts,
            "open": bar0_close,
            "high": bar0_close + bar0_close * 0.001,
            "low": low,
            "close": bar0_close,
            "volume": baseline_vol,
        })
    return pd.DataFrame(rows).set_index("ts")


def _make_ctx(
    df_5m,
    df_daily=None,
    pdc=100.0,
    cap_segment="large_cap",
    symbol="NSE:HDFCBANK",
    atr=1.0,
    volume_baseline_open_5m=10000.0,
):
    """Build MarketContext with df_daily (for trend filter) + indicators."""
    last_ts = df_5m.index[-1]
    indicators = {"atr": atr}
    if volume_baseline_open_5m is not None:
        indicators["volume_baseline_open_5m"] = volume_baseline_open_5m
    return MarketContext(
        symbol=symbol,
        current_price=float(df_5m["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df_5m,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        df_daily=df_daily,
        cap_segment=cap_segment,
        regime="trend_up",
        pdc=pdc,
        indicators=indicators,
    )


# =============================================================================
# Happy-path tests
# =============================================================================

def test_fires_long_on_canonical_gap_up_trend_new_high():
    """Gap-up 1.5% + PDC > 20SMA + bar 0 new high + bar 1 tags it → fires LONG."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)   # PDC > 20SMA
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"
    assert res.events[0].structure_type == "gap_and_go_continuation"


def test_fires_short_on_canonical_gap_down_trend_new_low():
    """Gap-down -1.5% + PDC < 20SMA + bar 0 new low + bar 1 tags it → fires SHORT."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_short_session(pdc=100.0, gap_pct=-1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=102.0)  # PDC < 20SMA (downtrend)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"


# =============================================================================
# Gap qualification
# =============================================================================

def test_does_not_fire_when_gap_below_threshold():
    """gap_pct=0.5 < threshold 1.0 → reject."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=0.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "gap_pct" in (res.rejection_reason or "").lower()


# =============================================================================
# Daily-trend filter
# =============================================================================

def test_does_not_fire_long_when_pdc_below_20sma():
    """Gap-up 1.5% but PDC < 20SMA → counter-trend → reject."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=102.0)   # PDC < 20SMA
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "trend" in (res.rejection_reason or "").lower()


def test_does_not_fire_short_when_pdc_above_20sma():
    """Gap-down -1.5% but PDC > 20SMA → counter-trend → reject."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_short_session(pdc=100.0, gap_pct=-1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)    # PDC > 20SMA
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "trend" in (res.rejection_reason or "").lower()


# =============================================================================
# First-bar volume filter
# =============================================================================

def test_does_not_fire_when_first_bar_volume_below_threshold():
    """first_bar_vol = baseline * 1.0 < min_ratio 1.2 → reject."""
    det = GapAndGoContinuationStructure(_cfg())
    # first_bar_vol == baseline (ratio 1.0, below threshold 1.2)
    df = _build_long_session(pdc=100.0, gap_pct=1.5, first_bar_vol=10000)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0, volume_baseline_open_5m=10000.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "vol" in (res.rejection_reason or "").lower()


# =============================================================================
# First-bar new-extreme check
# =============================================================================

def test_does_not_fire_when_first_bar_no_new_high_long():
    """Bar 0 high == open (no extension above gap-open) → reject for long."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=1.5, new_high_above_open_pct=0.0)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "new_high" in (res.rejection_reason or "").lower() or \
           "first_bar" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_first_bar_no_new_low_short():
    """Mirror: bar 0 low == open → reject for short."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_short_session(pdc=100.0, gap_pct=-1.5, new_low_below_open_pct=0.0)
    df_daily = _build_daily_df(pdc=100.0, sma20=102.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False


# =============================================================================
# Trigger bar check
# =============================================================================

def test_does_not_fire_when_trigger_bar_below_first_bar_high_long():
    """Bar 1 high < bar 0 high (no tag) → reject."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(
        pdc=100.0, gap_pct=1.5, trigger_at_or_above_first_high=False,
    )
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "trigger" in (res.rejection_reason or "").lower() or \
           "first_bar_high" in (res.rejection_reason or "").lower()


# =============================================================================
# Outside active window
# =============================================================================

def test_does_not_fire_outside_active_window():
    """Last bar at 09:35 is past active_window_end 09:30."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=1.5, n_subsequent=4)  # 09:15 + 4 bars → ends 09:35
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


# =============================================================================
# Universe + cap-segment guards
# =============================================================================

def test_does_not_fire_when_symbol_outside_universe():
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0, symbol="NSE:NONEXISTENT")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_cap_segment_small_cap():
    """small_cap excluded from allowed_cap_segments — confirms regime-complement
    to gap_fade_short which fades small_cap gaps."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0, cap_segment="small_cap")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "cap_segment" in (res.rejection_reason or "").lower()


# =============================================================================
# First-trigger latch
# =============================================================================

def test_first_trigger_latch_prevents_double_fire():
    """After plan_long fires, a SECOND detect on same session = no fire."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    # First call fires (and the plan registers the latch)
    res1 = det.detect(ctx)
    assert res1.structure_detected is True
    plan = det.plan_long_strategy(ctx, event=res1.events[0])
    assert plan is not None
    # Second call same session = blocked by latch
    res2 = det.detect(ctx)
    assert res2.structure_detected is False
    assert "already_fired" in (res2.rejection_reason or "").lower() or \
           "latch" in (res2.rejection_reason or "").lower()


# =============================================================================
# Plan emission
# =============================================================================

def test_plan_emits_hard_sl_and_tiered_t1_t2_for_long():
    """LONG plan: hard_sl < entry, T1 (1R) > entry, T2 (2R) > T1."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    plan = det.plan_long_strategy(ctx, event=res.events[0])
    assert plan is not None
    assert plan.side == "long"
    assert plan.risk_params.hard_sl < plan.entry_price
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["level"] > plan.entry_price
    assert targets[1]["level"] > targets[0]["level"]
    assert targets[0]["qty_pct"] == 0.5
    assert targets[1]["qty_pct"] == 0.5
    # T1 RR ~ 1.0 and T2 RR ~ 2.0 (tolerant of slight risk-floor rounding)
    assert abs(targets[0]["rr"] - 1.0) < 0.01
    assert abs(targets[1]["rr"] - 2.0) < 0.01


def test_plan_emits_hard_sl_and_tiered_t1_t2_for_short():
    """SHORT plan: hard_sl > entry, T1 < entry, T2 < T1."""
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_short_session(pdc=100.0, gap_pct=-1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=102.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    plan = det.plan_short_strategy(ctx, event=res.events[0])
    assert plan is not None
    assert plan.side == "short"
    assert plan.risk_params.hard_sl > plan.entry_price
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["level"] < plan.entry_price
    assert targets[1]["level"] < targets[0]["level"]


# =============================================================================
# Wide-open mode bypass
# =============================================================================

def test_wide_open_bypasses_daily_trend_filter(monkeypatch):
    """Under wide_open, counter-trend gap (PDC < 20SMA on long) still fires."""
    import structures.gap_and_go_continuation_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=102.0)   # would normally block long
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, (
        f"wide_open should bypass daily-trend: {res.rejection_reason}"
    )


def test_wide_open_bypasses_first_bar_volume_filter(monkeypatch):
    """Under wide_open, first-bar volume below ratio still fires."""
    import structures.gap_and_go_continuation_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    det = GapAndGoContinuationStructure(_cfg())
    # first_bar_vol == baseline (ratio 1.0, below 1.2 threshold)
    df = _build_long_session(pdc=100.0, gap_pct=1.5, first_bar_vol=10000)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0, volume_baseline_open_5m=10000.0)
    res = det.detect(ctx)
    assert res.structure_detected is True


def test_wide_open_preserves_trigger_geometry(monkeypatch):
    """Even under wide_open, sub-threshold gap must NOT fire — trigger geometry
    is mechanical and always enforced."""
    import structures.gap_and_go_continuation_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=0.5)   # below threshold
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False


def test_wide_open_bypasses_universe_and_cap_segment(monkeypatch):
    """Under wide_open, off-universe symbol AND off-cap-segment still fires —
    gauntlet decides which slice the detector works in (lessons.md 2026-04-15)."""
    import structures.gap_and_go_continuation_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    det = GapAndGoContinuationStructure(_cfg())
    df = _build_long_session(pdc=100.0, gap_pct=1.5)
    df_daily = _build_daily_df(pdc=100.0, sma20=98.0)
    ctx = _make_ctx(df, df_daily=df_daily, pdc=100.0,
                    symbol="NSE:NONEXISTENT", cap_segment="micro_cap")
    res = det.detect(ctx)
    assert res.structure_detected is True, (
        f"wide_open should bypass universe + cap_segment: {res.rejection_reason}"
    )
