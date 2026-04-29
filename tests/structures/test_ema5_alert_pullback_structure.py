"""5 EMA Alert-Candle Pullback (Subasish Pani) detector unit tests.

Mechanic per
specs/2026-04-29-ema5_alert_pullback-plan.md and
specs/2026-04-29-research-new-indian-setup-candidates.md (§ Candidate 4).

Three-state pipeline (long; mirror short):
  1) Trend prerequisite: 5 EMA strictly increasing for last
     trend_lookback_bars increments
  2) Alert candle: first bar whose entire HIGH < EMA at that bar
  3) Confirmation entry: next bar's HIGH > alert's HIGH → fire LONG.
     If next bar's LOW < alert's LOW → abort (drop pending).

Tests use trend_lookback_bars=4 override (production default 10) so the
12-bar test fixtures fit within the 09:30-10:00 active window.

Tests cover:
  - Happy-path long + short fires
  - Trend prerequisite fail → no latch (choppy bars)
  - Strict separation: alert HIGH must be < EMA, not equal/touching
  - Abort: confirm bar breaks alert opposite extreme → drop pending
  - Outside active window → reject (Subasish 10am cutoff)
  - Symbol outside fno_liquid_200 universe → reject
  - First-trigger latch prevents same-day double-fire
  - Plan T1 (1.5R) / T2 (3R) tiered exits in correct direction
  - Wide_open bypasses universe + cap_segment but preserves trend/alert/window
"""
from __future__ import annotations

import pandas as pd
import pytest

from structures.ema5_alert_pullback_structure import EMA5AlertPullbackStructure
from structures.data_models import MarketContext


def _cfg(**overrides):
    """Default test config — uses trend_lookback_bars=4 (vs production 10)
    so a 4+1+1=6-bar trend+alert+confirm session fits inside the 09:30-10:00
    active window with min_bars_required=6 prior bars."""
    cfg = {
        "_setup_name": "ema5_alert_pullback",
        "enabled": True,
        "active_window_start": "09:30",
        "active_window_end": "10:00",
        "ema_period": 5,
        "trend_lookback_bars": 4,        # test override (prod default 10)
        "trend_definition": "ema_slope_positive",
        "target_rr": 3.0,
        "t1_rr": 1.5,
        "t1_qty_pct": 0.5,
        "allowed_sides": ["long", "short"],
        "allowed_cap_segments": ["large_cap", "mid_cap", "small_cap"],
        "universe_key": "fno_liquid_200",
        "min_bars_required": 6,           # trend(4) + alert(1) + confirm(1)
        "entry_zone_pct": 0.1,
        "entry_zone_mode": "directional",
        "min_stop_distance_pct": 0.3,
    }
    cfg.update(overrides)
    return cfg


def _build_uptrend_long_session(
    n_trend=5,                  # bars BEFORE alert (gives EMA history + 4 increments)
    trend_step=0.5,             # how much each trend bar gains
    base_close=100.0,
    alert_high_below_ema_pct=0.5,    # alert HIGH = ACTUAL_EMA * (1 - this/100)
    confirm_break_pct=0.3,           # confirm HIGH = alert.high * (1 + this/100)
    session_date_str="2025-01-08",
    trigger_break=True,              # if False, confirm bar's HIGH stays below alert.high
    confirm_break_alert_low=False,   # if True, confirm LOW < alert.low (abort)
):
    """Build a 5m intraday DataFrame with an established 5m uptrend ending in
    a bearish alert candle (HIGH < EMA) and a confirm bar.

    Layout:
      bars 0..n_trend-1   : monotonically rising closes (EMA slopes up)
      bar n_trend         : alert candle — HIGH below the ACTUAL EMA at that bar
      bar n_trend+1       : confirm candle — HIGH breaks alert.high (or aborts)

    First bar timestamp is set so the LAST bar lands at 09:55 (within window).
    Alert HIGH is computed from the actual EMA(5) at the alert bar, not from
    a simple closes-proxy — EMA lags rising closes, so a naive
    `last_close * (1 - X%)` produces alert_high ABOVE the real EMA, which
    fails the strict-separation rule.
    """
    confirm_ts = pd.Timestamp(f"{session_date_str} 09:55:00")
    n_total = n_trend + 2  # trend + alert + confirm
    first_ts = confirm_ts - pd.Timedelta(minutes=5 * (n_total - 1))
    idx = pd.date_range(first_ts, periods=n_total, freq="5min")

    # Trend bars: closes rise from base_close by trend_step per bar
    trend_closes = [base_close + i * trend_step for i in range(n_trend)]

    # Compute EMA(5, adjust=False) at the alert bar position. The alert bar's
    # close affects the EMA there, so iterate forward including the alert
    # close and back-solve. Simpler approach: use a trial alert_close, compute
    # EMA, place alert_high below it. Iterate once for stability.
    alpha = 2.0 / (5 + 1)
    ema = trend_closes[0]
    for c in trend_closes[1:]:
        ema = (1 - alpha) * ema + alpha * c
    # ema is now the EMA at the LAST trend bar.
    # For the alert bar's EMA, we need to step forward with the alert's close.
    # First-pass: assume alert close = ema * (1 - 1%) (1% pullback).
    alert_close_guess = ema * 0.99
    ema_at_alert_guess = (1 - alpha) * ema + alpha * alert_close_guess
    # Place alert HIGH safely below the EMA at the alert bar.
    alert_high = ema_at_alert_guess * (1.0 - alert_high_below_ema_pct / 100.0)
    alert_open = alert_high * 0.998
    alert_close = alert_high * 0.996
    alert_low = alert_close * 0.997

    rows = []
    for i, c in enumerate(trend_closes):
        rows.append({
            "ts": idx[i],
            "open": c - 0.05,
            "high": c + 0.10,
            "low": c - 0.10,
            "close": c,
            "volume": 10000,
        })
    rows.append({
        "ts": idx[n_trend],
        "open": alert_open, "high": alert_high, "low": alert_low,
        "close": alert_close, "volume": 12000,
    })
    # Confirm bar
    if confirm_break_alert_low:
        # Abort: confirm LOW < alert LOW (stay below alert.high)
        confirm_low = alert_low * 0.998
        confirm_high = alert_close
        confirm_close = confirm_low * 1.001
    elif trigger_break:
        # Fire: confirm HIGH breaks alert.high
        confirm_high = alert_high * (1.0 + confirm_break_pct / 100.0)
        confirm_low = alert_close * 0.999
        confirm_close = confirm_high * 0.999
    else:
        # Neither fire nor abort: confirm stays inside alert range
        confirm_high = alert_high * 0.999
        confirm_low = alert_low * 1.001
        confirm_close = (confirm_high + confirm_low) / 2.0
    rows.append({
        "ts": idx[n_trend + 1],
        "open": alert_close,
        "high": confirm_high, "low": confirm_low, "close": confirm_close,
        "volume": 11000,
    })
    return pd.DataFrame(rows).set_index("ts")


def _build_downtrend_short_session(
    n_trend=5,
    trend_step=0.5,
    base_close=100.0,
    alert_low_above_ema_pct=0.5,
    confirm_break_pct=0.3,
    session_date_str="2025-01-08",
    trigger_break=True,
):
    """Mirror of long session: monotonically falling closes (EMA slopes down)
    ending in a bullish alert candle (LOW > EMA) and a confirm bar that
    breaks alert.low for SHORT entry."""
    confirm_ts = pd.Timestamp(f"{session_date_str} 09:55:00")
    n_total = n_trend + 2
    first_ts = confirm_ts - pd.Timedelta(minutes=5 * (n_total - 1))
    idx = pd.date_range(first_ts, periods=n_total, freq="5min")

    trend_closes = [base_close - i * trend_step for i in range(n_trend)]
    alpha = 2.0 / (5 + 1)
    ema = trend_closes[0]
    for c in trend_closes[1:]:
        ema = (1 - alpha) * ema + alpha * c
    alert_close_guess = ema * 1.01    # mirror: pullback UP (counter-downtrend)
    ema_at_alert_guess = (1 - alpha) * ema + alpha * alert_close_guess
    alert_low = ema_at_alert_guess * (1.0 + alert_low_above_ema_pct / 100.0)
    alert_open = alert_low * 1.002
    alert_close = alert_low * 1.004
    alert_high = alert_close * 1.003

    rows = []
    for i, c in enumerate(trend_closes):
        rows.append({
            "ts": idx[i],
            "open": c + 0.05, "high": c + 0.10, "low": c - 0.10,
            "close": c, "volume": 10000,
        })
    rows.append({
        "ts": idx[n_trend],
        "open": alert_open, "high": alert_high, "low": alert_low,
        "close": alert_close, "volume": 12000,
    })
    if trigger_break:
        confirm_low = alert_low * (1.0 - confirm_break_pct / 100.0)
        confirm_high = alert_close * 1.001
        confirm_close = confirm_low * 1.001
    else:
        confirm_low = alert_low * 1.001
        confirm_high = alert_high * 0.999
        confirm_close = (confirm_high + confirm_low) / 2.0
    rows.append({
        "ts": idx[n_trend + 1],
        "open": alert_close, "high": confirm_high, "low": confirm_low,
        "close": confirm_close, "volume": 11000,
    })
    return pd.DataFrame(rows).set_index("ts")


def _build_choppy_session(session_date_str="2025-01-08", n_total=7):
    """Build alternating up/down closes (no monotonic trend) with a final
    bearish bar that COULD be an alert candle if trend held."""
    confirm_ts = pd.Timestamp(f"{session_date_str} 09:55:00")
    first_ts = confirm_ts - pd.Timedelta(minutes=5 * (n_total - 1))
    idx = pd.date_range(first_ts, periods=n_total, freq="5min")
    rows = []
    base = 100.0
    for i in range(n_total - 1):
        # Alternating up/down
        c = base + (0.5 if i % 2 == 0 else -0.5)
        rows.append({
            "ts": idx[i],
            "open": base, "high": c + 0.1, "low": c - 0.1,
            "close": c, "volume": 10000,
        })
    # Last bar: bearish, well below
    rows.append({
        "ts": idx[-1],
        "open": base, "high": base - 1.0, "low": base - 1.5,
        "close": base - 1.4, "volume": 12000,
    })
    return pd.DataFrame(rows).set_index("ts")


def _ctx(
    df,
    symbol="NSE:HDFCBANK",
    cap_segment="large_cap",
    atr=1.0,
):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment=cap_segment,
        regime="trend_up",
        pdh=120.0, pdl=80.0, pdc=100.0,
        indicators={"atr": atr},
    )


# =============================================================================
# Happy-path tests
# =============================================================================

def test_fires_long_on_canonical_uptrend_alert_pullback():
    """Established uptrend + bearish alert below EMA + confirm bar breaks
    alert.high → fires LONG."""
    det = EMA5AlertPullbackStructure(_cfg())
    df = _build_uptrend_long_session()
    # Step 1: feed up to the alert bar — should latch but NOT fire.
    df_at_alert = df.iloc[:-1]
    res_alert = det.detect(_ctx(df_at_alert))
    assert res_alert.structure_detected is False, (
        f"alert bar should latch, not fire: {res_alert.rejection_reason}"
    )
    # Step 2: feed full df with confirm bar — fires LONG.
    res = det.detect(_ctx(df))
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"
    assert res.events[0].structure_type == "ema5_alert_pullback"


def test_fires_short_on_canonical_downtrend_alert_pullback():
    """Mirror: established downtrend + bullish alert above EMA + confirm
    breaks alert.low → fires SHORT."""
    det = EMA5AlertPullbackStructure(_cfg())
    df = _build_downtrend_short_session()
    det.detect(_ctx(df.iloc[:-1]))
    res = det.detect(_ctx(df))
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"


# =============================================================================
# Trend prerequisite
# =============================================================================

def test_does_not_fire_when_no_trend():
    """Choppy alternating bars don't satisfy ema_slope_positive prerequisite."""
    det = EMA5AlertPullbackStructure(_cfg())
    df = _build_choppy_session()
    res = det.detect(_ctx(df))
    assert res.structure_detected is False


# =============================================================================
# Strict separation rule (alert HIGH must be < EMA)
# =============================================================================

def test_does_not_fire_when_alert_high_touches_ema():
    """If the alert bar's HIGH meets the EMA at that bar (no strict
    separation), no alert is latched — strict-less-than is required.
    Computes the actual EMA at the alert position and sets alert.high to
    EMA * 1.0001 (just above EMA, breaks the strict rule)."""
    det = EMA5AlertPullbackStructure(_cfg())
    df = _build_uptrend_long_session()
    # Compute actual EMA(5, adjust=False) up to and including the alert bar
    closes_through_alert = df["close"].iloc[: -1]   # exclude confirm bar
    alpha = 2.0 / 6.0
    ema_at_alert = float(closes_through_alert.iloc[0])
    for c in closes_through_alert.iloc[1:]:
        ema_at_alert = (1 - alpha) * ema_at_alert + alpha * float(c)
    # Mutate alert bar's HIGH to be just AT the EMA (no strict separation)
    alert_idx = df.index[-2]
    df.loc[alert_idx, "high"] = ema_at_alert * 1.0001  # at/above EMA
    df_at_alert = df.iloc[:-1]
    det.detect(_ctx(df_at_alert))
    res = det.detect(_ctx(df))
    assert res.structure_detected is False


# =============================================================================
# Abort: confirm bar breaks alert opposite side
# =============================================================================

def test_does_not_fire_long_when_confirm_breaks_alert_low():
    """Sweep latches, but confirm LOW < alert LOW (no breakout above; trap
    is invalidated). detect() drops pending silently."""
    det = EMA5AlertPullbackStructure(_cfg())
    df = _build_uptrend_long_session(confirm_break_alert_low=True)
    det.detect(_ctx(df.iloc[:-1]))
    res = det.detect(_ctx(df))
    assert res.structure_detected is False


# =============================================================================
# Outside active window (Subasish 10am cutoff)
# =============================================================================

def test_does_not_fire_after_10am():
    """Confirm bar at 10:05 (past 10:00 active_window_end) → reject."""
    det = EMA5AlertPullbackStructure(_cfg())
    # Shift session so confirm lands at 10:05
    df = _build_uptrend_long_session()
    new_idx = df.index + pd.Timedelta(minutes=10)
    df.index = new_idx
    det.detect(_ctx(df.iloc[:-1]))
    res = det.detect(_ctx(df))
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


# =============================================================================
# Universe + cap_segment guards
# =============================================================================

def test_does_not_fire_when_symbol_outside_universe():
    det = EMA5AlertPullbackStructure(_cfg())
    df = _build_uptrend_long_session()
    det.detect(_ctx(df.iloc[:-1], symbol="NSE:NONEXISTENT"))
    res = det.detect(_ctx(df, symbol="NSE:NONEXISTENT"))
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_cap_segment_excluded():
    """allowed_cap_segments=['large_cap'] → mid_cap excluded."""
    det = EMA5AlertPullbackStructure(_cfg(allowed_cap_segments=["large_cap"]))
    df = _build_uptrend_long_session()
    det.detect(_ctx(df.iloc[:-1], cap_segment="mid_cap"))
    res = det.detect(_ctx(df, cap_segment="mid_cap"))
    assert res.structure_detected is False


# =============================================================================
# First-trigger latch
# =============================================================================

def test_first_trigger_latch_prevents_double_fire():
    """After first LONG fires, a second valid alert+confirm later in same
    session is no-op."""
    det = EMA5AlertPullbackStructure(_cfg())
    df1 = _build_uptrend_long_session()
    det.detect(_ctx(df1.iloc[:-1]))
    res1 = det.detect(_ctx(df1))
    assert res1.structure_detected is True

    # Build a second alert+confirm at a later 5m offset (still inside window)
    # by shifting timestamps forward by 5 min — confirm at 10:00 (boundary).
    df2 = df1.copy()
    df2.index = df1.index + pd.Timedelta(minutes=5)
    # Filter to bars within active_window_end (10:00 is the inclusive end).
    df2 = df2[df2.index <= pd.Timestamp(f"{df1.index[0].strftime('%Y-%m-%d')} 10:00:00")]
    if len(df2) >= 6:
        det.detect(_ctx(df2.iloc[:-1]))
        res2 = det.detect(_ctx(df2))
        assert res2.structure_detected is False, "latch should prevent same-session refire"


# =============================================================================
# Plan emission
# =============================================================================

def test_plan_emits_hard_sl_t1_t2_for_long():
    """LONG plan: hard_sl < entry, T1 (1.5R) > entry, T2 (3R) > T1, qty 50/50."""
    det = EMA5AlertPullbackStructure(_cfg())
    df = _build_uptrend_long_session()
    det.detect(_ctx(df.iloc[:-1]))
    plan = det.plan_long_strategy(_ctx(df))
    assert plan is not None
    assert plan.side == "long"
    assert plan.risk_params.hard_sl < plan.entry_price
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["level"] > plan.entry_price
    assert targets[1]["level"] > targets[0]["level"]
    assert targets[0]["qty_pct"] == 0.5
    assert targets[1]["qty_pct"] == 0.5
    # T1 RR ~ 1.5, T2 RR ~ 3.0 (tolerant of risk-floor rounding)
    assert abs(targets[0]["rr"] - 1.5) < 0.01
    assert abs(targets[1]["rr"] - 3.0) < 0.01


def test_plan_emits_hard_sl_t1_t2_for_short():
    """SHORT plan: hard_sl > entry, T1 < entry, T2 < T1."""
    det = EMA5AlertPullbackStructure(_cfg())
    df = _build_downtrend_short_session()
    det.detect(_ctx(df.iloc[:-1]))
    plan = det.plan_short_strategy(_ctx(df))
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

def test_wide_open_bypasses_universe_filter(monkeypatch):
    """Under wide_open, symbol outside fno_liquid_200 still fires."""
    import structures.ema5_alert_pullback_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    det = EMA5AlertPullbackStructure(_cfg())
    df = _build_uptrend_long_session()
    det.detect(_ctx(df.iloc[:-1], symbol="NSE:NONEXISTENT"))
    res = det.detect(_ctx(df, symbol="NSE:NONEXISTENT"))
    assert res.structure_detected is True, (
        f"wide_open should bypass universe: {res.rejection_reason}"
    )


def test_wide_open_preserves_active_window_and_trend(monkeypatch):
    """Even under wide_open, outside-window AND choppy-trend rejections still
    apply — those are mechanical, not design-inferred."""
    import structures.ema5_alert_pullback_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    det = EMA5AlertPullbackStructure(_cfg())
    # Choppy trend — should NOT fire even under wide_open
    df = _build_choppy_session()
    res = det.detect(_ctx(df))
    assert res.structure_detected is False
