"""CPR Mean Revert detector unit tests (sub7-T6)."""
import pandas as pd
from structures.cpr_mean_revert_structure import CPRMeanRevertStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "cpr_mean_revert",
        "enabled": True,
        "active_window_start": "11:30",
        "active_window_end": "13:30",
        "tc_bc_touch_tolerance_pct": 0.1,
        "max_volume_pct_of_intraday_avg": 30.0,
        "require_reversion_candle": True,
        "reversion_patterns": ["hammer", "doji", "shooting_star"],
        "allowed_cap_segments": ["small_cap", "mid_cap", "large_cap"],
        "stop_at_extreme_atr_buffer": 0.5,
        "target_type": "cpr_midpoint",
        "time_stop_at": "14:00",
        "min_cpr_width_pct": 0.0,
        "min_bars_required": 30,
    }


def _build_lull_df(
    now_time,
    last_close,
    cpr_mid=100.0,
    atr=1.0,
    cur_volume=2000,
    intraday_avg_volume=10000,
    candle_pattern="hammer",
    n_bars=40,
):
    """Build a 40-bar DataFrame for CPR mean revert tests.

    Bars 0..n-2: close=cpr_mid, normal volume (intraday_avg_volume).
    Last bar: specified close + low volume + reversion candle pattern.

    For "hammer" (last_close > cpr_mid): small body at top + long lower wick.
      o=last_close - 0.1*atr (open near close), c=last_close
      h=last_close + 0.05*atr, l=last_close - 3*atr

    For "shooting_star" (last_close < cpr_mid): small body at bottom + long upper wick.
      o=last_close (open near close), c=last_close + 0.1*atr
      h=last_close + 3*atr, l=last_close - 0.05*atr

    For "doji": very small body.
      o=last_close, c=last_close + 0.01*atr, h=last_close + atr, l=last_close - atr
    """
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5 * (n_bars - 1)), periods=n_bars, freq="5min")

    # Default bars: close at cpr_mid, normal volume
    opens = [cpr_mid] * n_bars
    highs = [cpr_mid + 0.3 * atr] * n_bars
    lows = [cpr_mid - 0.3 * atr] * n_bars
    closes = [cpr_mid] * n_bars
    volumes = [intraday_avg_volume] * n_bars

    # Override the last bar with the specified candle pattern + low volume
    if candle_pattern == "hammer":
        # Small body near top, long lower wick — signals potential bullish reversal
        # Used when close > cpr_mid (short setup), but hammer shows selling rejection
        last_open = last_close - 0.1 * atr
        last_close_val = last_close
        last_high = last_close + 0.05 * atr
        last_low = last_close - 3.0 * atr
    elif candle_pattern == "shooting_star":
        # Small body near bottom, long upper wick — signals potential bearish reversal
        # Used when close < cpr_mid (long setup)
        last_open = last_close
        last_close_val = last_close + 0.1 * atr
        last_high = last_close + 3.0 * atr
        last_low = last_close - 0.05 * atr
    else:  # doji
        last_open = last_close
        last_close_val = last_close + 0.01 * atr
        last_high = last_close + atr
        last_low = last_close - atr

    opens[-1] = last_open
    highs[-1] = last_high
    lows[-1] = last_low
    closes[-1] = last_close_val
    volumes[-1] = cur_volume

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )
    return df


def _make_ctx(df, pdh=103.0, pdl=98.0, pdc=100.0, cap_segment="small_cap", atr=1.0):
    """Build a MarketContext for CPR mean revert tests.

    CPR levels are now computed by the detector from pdh/pdl/pdc (production
    MarketContext provides these as direct fields; indicators only has 'vol_z'
    and 'atr' in production).

    Default PDH=103, PDL=98, PDC=100 gives:
      pivot = (103+98+100)/3 = 100.33
      bc    = (103+98)/2     = 100.5
      tc    = 2*100.33-100.5 = 100.17
      CPR_TOP=100.5, CPR_BOTTOM=100.17, CPR_MID=100.33
    """
    last_ts = df.index[-1]
    return MarketContext(
        symbol="NSE:TESTSYM",
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment=cap_segment,
        regime="chop",
        pdh=pdh,
        pdl=pdl,
        pdc=pdc,
        indicators={"atr": atr},
    )


# ---------------------------------------------------------------------------
# Test 1: fires short reversion above CPR
# ---------------------------------------------------------------------------

def test_fires_short_reversion_above_cpr():
    """Bar high touches TC + shooting_star (bearish rejection) → fires short.

    PDH=103, PDL=98, PDC=100 → pivot=100.33, bc=100.5, tc=100.17
    Normalized: CPR_TOP=100.5, CPR_BOTTOM=100.17, CPR_MID=100.33
    shooting_star with last_close=102 → high=105.0 >= TC=100.5 → short trigger.
    """
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)

    atr = 1.0
    last_close = 102.0

    df = _build_lull_df(
        now_time="12:00:00",
        last_close=last_close,
        cpr_mid=100.33,  # visual guide for df builder (prior bars), not used by detector
        atr=atr,
        cur_volume=2000,
        intraday_avg_volume=10000,
        candle_pattern="shooting_star",
    )
    # PDH=103, PDL=98, PDC=100 → CPR_MID=100.33
    ctx = _make_ctx(df, pdh=103.0, pdl=98.0, pdc=100.0, cap_segment="small_cap", atr=atr)

    result = det.detect(ctx)
    assert result.structure_detected is True, f"Expected fire, got: {result.rejection_reason}"
    assert result.events[0].side == "short"
    assert result.events[0].context.get("bias") == "short"

    plan = det.plan_short_strategy(ctx)
    assert plan is not None
    assert plan.side == "short"
    assert plan.structure_type == "cpr_mean_revert"
    assert plan.notes.get("bias") == "short"
    # Stop should be ABOVE entry (short)
    assert plan.risk_params.hard_sl > plan.entry_price
    # Target should be at CPR midpoint (~100.33), below entry (102.0)
    assert plan.exit_levels.targets[0]["level"] < plan.entry_price


# ---------------------------------------------------------------------------
# Test 2: fires long reversion below CPR
# ---------------------------------------------------------------------------

def test_fires_long_reversion_below_cpr():
    """Bar low touches BC + hammer (bullish rejection) → fires long.

    PDH=103, PDL=98, PDC=100 → CPR_MID=100.33, CPR_BOTTOM=100.17, CPR_TOP=100.5
    hammer with last_close=98 → low=95.0 <= BC=100.17 → long trigger.
    """
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)

    atr = 1.0
    last_close = 98.0

    df = _build_lull_df(
        now_time="12:00:00",
        last_close=last_close,
        cpr_mid=100.33,  # visual guide for df builder, not used by detector
        atr=atr,
        cur_volume=2000,
        intraday_avg_volume=10000,
        candle_pattern="hammer",
    )
    # PDH=103, PDL=98, PDC=100 → CPR_MID=100.33
    ctx = _make_ctx(df, pdh=103.0, pdl=98.0, pdc=100.0, cap_segment="small_cap", atr=atr)

    result = det.detect(ctx)
    assert result.structure_detected is True, f"Expected fire, got: {result.rejection_reason}"
    assert result.events[0].side == "long"
    assert result.events[0].context.get("bias") == "long"

    plan = det.plan_long_strategy(ctx)
    assert plan is not None
    assert plan.side == "long"
    assert plan.structure_type == "cpr_mean_revert"
    assert plan.notes.get("bias") == "long"
    # Stop should be BELOW entry (long)
    assert plan.risk_params.hard_sl < plan.entry_price
    # Target should be at CPR midpoint (~100.33), above entry (~98.1)
    assert plan.exit_levels.targets[0]["level"] > plan.entry_price


# ---------------------------------------------------------------------------
# Test 3: does not fire outside active window
# ---------------------------------------------------------------------------

def test_does_not_fire_outside_window():
    """Bar at 10:00 is outside 11:30-13:30 window → no fire."""
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)

    atr = 1.0
    last_close = 102.0

    df = _build_lull_df(
        now_time="10:00:00",
        last_close=last_close,
        cpr_mid=100.33,
        atr=atr,
        cur_volume=2000,
        intraday_avg_volume=10000,
        candle_pattern="hammer",
    )
    # PDH=103, PDL=98, PDC=100 → CPR_MID=100.33
    ctx = _make_ctx(df, pdh=103.0, pdl=98.0, pdc=100.0, cap_segment="small_cap", atr=atr)

    result = det.detect(ctx)
    assert result.structure_detected is False
    assert "window" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 4: does not fire when bar is entirely inside the CPR band
# ---------------------------------------------------------------------------

def test_does_not_fire_when_neither_boundary_touched():
    """Last bar entirely inside CPR [BC=100.17, TC=100.5] → no boundary touch → no fire.

    PDH=103, PDL=98, PDC=100 → CPR_BOTTOM=100.17, CPR_TOP=100.5.
    Build a bar with high=100.45, low=100.25 — both inside CPR.
    Industry-standard CPR mean-revert REQUIRES touching TC or BC; a bar that
    just oscillates inside the band is not a fade signal.
    """
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)

    atr = 1.0
    end = pd.Timestamp("2025-01-02 12:00:00")
    n_bars = 40
    idx = pd.date_range(
        end - pd.Timedelta(minutes=5 * (n_bars - 1)), periods=n_bars, freq="5min"
    )

    # Prior bars: warmup, normal volume; last bar: tight inside-CPR, low volume.
    # Tolerance band: TC*(1-0.001)=100.3995, BC*(1+0.001)=100.2667.
    # Bar must stay strictly inside [100.2667, 100.3995] to avoid triggering.
    opens = [100.30] * n_bars
    highs = [100.35] * n_bars
    lows = [100.28] * n_bars
    closes = [100.31] * n_bars
    volumes = [10000] * (n_bars - 1) + [2000]
    opens[-1] = 100.30
    highs[-1] = 100.35   # < TC*(1-tol) = 100.3995
    lows[-1] = 100.28    # > BC*(1+tol) = 100.2667
    closes[-1] = 100.32
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )
    ctx = _make_ctx(df, pdh=103.0, pdl=98.0, pdc=100.0, cap_segment="small_cap", atr=atr)

    result = det.detect(ctx)
    assert result.structure_detected is False
    assert "boundary" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 5: does not fire with high volume
# ---------------------------------------------------------------------------

def test_does_not_fire_with_high_volume():
    """cur_volume=8000 is 80% of intraday_avg (10000), above 30% threshold → no fire."""
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)

    atr = 1.0
    last_close = 102.0

    df = _build_lull_df(
        now_time="12:00:00",
        last_close=last_close,
        cpr_mid=100.33,
        atr=atr,
        cur_volume=8000,       # 80% of avg — well above 30% max
        intraday_avg_volume=10000,
        candle_pattern="hammer",
    )
    # PDH=103, PDL=98, PDC=100 → CPR_MID=100.33
    ctx = _make_ctx(df, pdh=103.0, pdl=98.0, pdc=100.0, cap_segment="small_cap", atr=atr)

    result = det.detect(ctx)
    assert result.structure_detected is False
    assert "volume" in (result.rejection_reason or "").lower()
