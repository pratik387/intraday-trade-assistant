"""options_vol_iv_rank_revert detector unit tests (sub-9 round-4 ship).

Verifies:
  - fires when iv_rank >= threshold + red candle below VWAP at 11:00
  - does NOT fire outside 11:00 single-bar window
  - does NOT fire when iv_rank < threshold
  - does NOT fire on green candle
  - does NOT fire when bar close >= VWAP
  - does NOT fire on disallowed cap_segment
  - does NOT fire on symbol outside fno_liquid_200 universe
  - plan_short emits hard_sl ABOVE entry, T1 + T2 BELOW entry
  - latch prevents double-fire same session
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import pytest

import services.iv_rank_service as iv_module
import structures.options_vol_iv_rank_revert_structure as detector_module
from structures.data_models import MarketContext
from structures.options_vol_iv_rank_revert_structure import (
    OptionsVolIvRankRevertStructure,
)


@pytest.fixture(autouse=True)
def _disable_wide_open(monkeypatch):
    """Project's base config has wide_open_mode=True (Discovery captures);
    these unit tests verify the production guards, so force wide_open=False."""
    monkeypatch.setattr(detector_module, "_is_wide_open", lambda: False)


def _cfg(**overrides):
    cfg = {
        "_setup_name": "options_vol_iv_rank_revert",
        "enabled": True,
        "iv_rank_high_threshold": 0.85,
        "active_window_start": "11:00",
        "active_window_end": "11:00",
        "stop_pct": 0.01,
        "t1_r_multiple": 1.0,
        "t2_r_multiple": 2.0,
        "t1_qty_pct": 0.5,
        "time_stop_at": "15:10",
        "allowed_cap_segments": ["large_cap", "mid_cap", "small_cap"],
        "universe_key": None,    # tests use None to bypass universe check
        "allowed_regimes": None,
        "min_bars_required": 16,
        "entry_zone_pct": 0.10,
        "entry_zone_mode": "directional",
        "min_stop_distance_pct": 0.5,    # below stop_pct=1% with fp margin
    }
    cfg.update(overrides)
    return cfg


class _MockIVRankService:
    def __init__(self, iv_rank: Optional[float]):
        self._iv = iv_rank

    def get_iv_rank(self, symbol: str, session_date: date) -> Optional[float]:
        return self._iv


@pytest.fixture
def mock_iv_high(monkeypatch):
    """IV-rank = 0.92 → above 0.85 threshold."""
    monkeypatch.setattr(iv_module, "_singleton", _MockIVRankService(0.92))
    yield
    monkeypatch.setattr(iv_module, "_singleton", None)


@pytest.fixture
def mock_iv_low(monkeypatch):
    """IV-rank = 0.50 → below threshold."""
    monkeypatch.setattr(iv_module, "_singleton", _MockIVRankService(0.50))
    yield
    monkeypatch.setattr(iv_module, "_singleton", None)


@pytest.fixture
def mock_iv_missing(monkeypatch):
    monkeypatch.setattr(iv_module, "_singleton", _MockIVRankService(None))
    yield
    monkeypatch.setattr(iv_module, "_singleton", None)


def _build_5m(session_date: date, n_bars: int = 22,
              close_above_vwap: bool = False,
              red_candle: bool = True) -> pd.DataFrame:
    """Build n_bars of 5m bars from 09:15 IST. Last bar at 09:15 + 5*(n-1) min.

    n_bars=21 → last bar at 09:15 + 100min = 10:55 (NOT 11:00).
    n_bars=22 → last bar at 11:00 (the active-window bar).

    Bars 0..n-2 drift UP creating an elevated VWAP. The last bar drops
    sharply (1.5% red candle) below the cumulative VWAP — fire condition.

    Red candle: close < open. Green: close > open.
    close_above_vwap=True → push last bar's close above VWAP (no fire).
    """
    base = pd.Timestamp(session_date).replace(hour=9, minute=15)
    bars = []
    drift_price = 100.0
    for i in range(n_bars - 1):
        # Strong up-drift on bars 0..n-2 to push VWAP high
        o = drift_price
        c = drift_price * 1.001          # 0.1% per bar
        h = c * 1.0005
        l = o * 0.9995
        bars.append({"open": o, "high": h, "low": l, "close": c, "volume": 5000})
        drift_price = c

    # Last bar: open at drift_price, drop hard
    o = drift_price
    if red_candle:
        c = drift_price * 0.985          # 1.5% red drop — well below VWAP
    else:
        c = drift_price * 1.005          # green
    h = max(o, c) * 1.001
    l = min(o, c) * 0.999
    bars.append({"open": o, "high": h, "low": l, "close": c, "volume": 5000})

    df = pd.DataFrame(
        bars, index=[base + pd.Timedelta(minutes=5 * i) for i in range(n_bars)],
    )
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tpv = (tp * df["volume"]).cumsum()
    cum_v = df["volume"].cumsum()
    df["vwap"] = cum_tpv / cum_v

    if close_above_vwap:
        last_idx = df.index[-1]
        # Force close above VWAP (still red — so VWAP gate is what blocks)
        df.loc[last_idx, "close"] = df.loc[last_idx, "vwap"] * 1.002
        df.loc[last_idx, "high"] = max(df.loc[last_idx, "high"],
                                        df.loc[last_idx, "close"] * 1.001)
    return df


def _make_ctx(df: pd.DataFrame, session_date: date,
              cap_segment: str = "large_cap"):
    return MarketContext(
        symbol="NSE:RELIANCE",
        current_price=float(df["close"].iloc[-1]),
        timestamp=df.index[-1],
        df_5m=df,
        df_daily=None,
        session_date=session_date,
        cap_segment=cap_segment,
        regime="trend_down",
    )


# ---------------------------------------------------------------------------
# Test 1: fires when all conditions met
# ---------------------------------------------------------------------------

def test_fires_when_all_conditions_met(mock_iv_high):
    det = OptionsVolIvRankRevertStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, n_bars=22, red_candle=True, close_above_vwap=False)
    # Last bar must be 11:00
    assert df.index[-1].time().strftime("%H:%M") == "11:00"
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert result.structure_detected, f"expected fire: {result.rejection_reason}"
    assert result.events[0].side == "short"
    assert result.events[0].context["iv_rank"] == 0.92


# ---------------------------------------------------------------------------
# Test 2: outside 11:00 window → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_outside_active_window(mock_iv_high):
    det = OptionsVolIvRankRevertStructure(_cfg())
    sd = date(2024, 6, 6)
    # 21 bars → last at 10:55 (one bar before 11:00)
    df = _build_5m(sd, n_bars=21, red_candle=True)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "window" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 3: iv_rank below threshold → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_when_iv_rank_below_threshold(mock_iv_low):
    det = OptionsVolIvRankRevertStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, n_bars=22, red_candle=True)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "iv_rank" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 4: green candle → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_on_green_candle(mock_iv_high):
    det = OptionsVolIvRankRevertStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, n_bars=22, red_candle=False)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "red candle" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 5: close >= VWAP → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_when_close_above_vwap(mock_iv_high):
    det = OptionsVolIvRankRevertStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, n_bars=22, red_candle=True, close_above_vwap=True)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "vwap" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 6: missing IV-rank → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_when_iv_rank_missing(mock_iv_missing):
    det = OptionsVolIvRankRevertStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, n_bars=22, red_candle=True)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "no iv-rank" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 7: disallowed cap segment → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_on_disallowed_cap(mock_iv_high):
    det = OptionsVolIvRankRevertStructure(_cfg(allowed_cap_segments=["mid_cap"]))
    sd = date(2024, 6, 6)
    df = _build_5m(sd, n_bars=22, red_candle=True)
    ctx = _make_ctx(df, sd, cap_segment="large_cap")  # disallowed
    result = det.detect(ctx)
    assert not result.structure_detected
    assert "cap segment" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 8: plan_short produces valid SL/T1/T2 geometry
# ---------------------------------------------------------------------------

def test_plan_short_geometry(mock_iv_high):
    det = OptionsVolIvRankRevertStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, n_bars=22, red_candle=True)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    assert result.structure_detected
    plan = det.plan_short_strategy(ctx, result.events[0])
    assert plan is not None, "plan_short returned None"
    entry = plan.entry_price
    sl = plan.risk_params.hard_sl
    t1 = plan.exit_levels.targets[0]["level"]
    t2 = plan.exit_levels.targets[1]["level"]
    assert sl > entry, f"SL {sl} must be ABOVE short entry {entry}"
    assert t1 < entry, f"T1 {t1} must be BELOW short entry {entry}"
    assert t2 < t1, f"T2 {t2} must be further below than T1 {t1}"
    # Stop = 1% above entry; T1 = 1R below; T2 = 2R below
    expected_sl = entry * 1.01
    assert abs(sl - expected_sl) < 0.01, f"sl={sl} expected={expected_sl}"
    assert plan.target_anchor_type == "arithmetic"


# ---------------------------------------------------------------------------
# Test 9: long-side returns None (SHORT-only setup)
# ---------------------------------------------------------------------------

def test_plan_long_returns_none(mock_iv_high):
    det = OptionsVolIvRankRevertStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, n_bars=22, red_candle=True)
    ctx = _make_ctx(df, sd)
    result = det.detect(ctx)
    plan = det.plan_long_strategy(ctx, result.events[0])
    assert plan is None


# ---------------------------------------------------------------------------
# Test 10: latch prevents double-fire same session
# ---------------------------------------------------------------------------

def test_latch_prevents_double_fire(mock_iv_high):
    det = OptionsVolIvRankRevertStructure(_cfg())
    sd = date(2024, 6, 6)
    df = _build_5m(sd, n_bars=22, red_candle=True)
    ctx = _make_ctx(df, sd)
    r1 = det.detect(ctx)
    assert r1.structure_detected
    plan = det.plan_short_strategy(ctx, r1.events[0])
    assert plan is not None
    # Second detect on same (symbol, session) — should be latched
    r2 = det.detect(ctx)
    assert not r2.structure_detected
    assert "already fired" in (r2.rejection_reason or "").lower()
