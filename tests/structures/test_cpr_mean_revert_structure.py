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
        "min_distance_atr_from_cpr": 1.0,
        "max_volume_pct_of_intraday_avg": 30.0,
        "require_reversion_candle": True,
        "reversion_patterns": ["hammer", "doji", "shooting_star"],
        "allowed_cap_segments": ["small_cap", "mid_cap", "large_cap"],
        "stop_at_extreme_atr_buffer": 0.2,
        "target_type": "cpr_midpoint",
        "time_stop_at": "13:45",
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


def _make_ctx(df, cpr_top=101.0, cpr_bottom=99.0, cap_segment="small_cap", atr=1.0):
    """Build a MarketContext for CPR mean revert tests.

    CPR levels are passed via indicators dict (CPR_TOP, CPR_BOTTOM, atr).
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
        indicators={
            "atr": atr,
            "CPR_TOP": cpr_top,
            "CPR_BOTTOM": cpr_bottom,
        },
    )


# ---------------------------------------------------------------------------
# Test 1: fires short reversion above CPR
# ---------------------------------------------------------------------------

def test_fires_short_reversion_above_cpr():
    """close=102 (1.0 ATR above cpr_mid=101), low volume, hammer → fires short."""
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)

    cpr_top, cpr_bot = 101.0, 99.0   # cpr_mid = 100.0
    atr = 1.0
    last_close = 102.0  # 2 ATR above cpr_mid... wait, mid=100, dist=2 >= 1.0 ok

    df = _build_lull_df(
        now_time="12:00:00",
        last_close=last_close,
        cpr_mid=100.0,
        atr=atr,
        cur_volume=2000,
        intraday_avg_volume=10000,
        candle_pattern="hammer",
    )
    ctx = _make_ctx(df, cpr_top=cpr_top, cpr_bottom=cpr_bot, cap_segment="small_cap", atr=atr)

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
    # Target should be at or near CPR midpoint (100.0), below entry (102.0)
    assert plan.exit_levels.targets[0]["level"] < plan.entry_price


# ---------------------------------------------------------------------------
# Test 2: fires long reversion below CPR
# ---------------------------------------------------------------------------

def test_fires_long_reversion_below_cpr():
    """close=98 (2.0 ATR below cpr_mid=100), low volume, shooting_star → fires long."""
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)

    cpr_top, cpr_bot = 101.0, 99.0   # cpr_mid = 100.0
    atr = 1.0
    last_close = 98.0  # 2 ATR below cpr_mid

    df = _build_lull_df(
        now_time="12:00:00",
        last_close=last_close,
        cpr_mid=100.0,
        atr=atr,
        cur_volume=2000,
        intraday_avg_volume=10000,
        candle_pattern="shooting_star",
    )
    ctx = _make_ctx(df, cpr_top=cpr_top, cpr_bottom=cpr_bot, cap_segment="small_cap", atr=atr)

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
    # Target should be at or near CPR midpoint (100.0), above entry (98.0)
    assert plan.exit_levels.targets[0]["level"] > plan.entry_price


# ---------------------------------------------------------------------------
# Test 3: does not fire outside active window
# ---------------------------------------------------------------------------

def test_does_not_fire_outside_window():
    """Bar at 10:00 is outside 11:30-13:30 window → no fire."""
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)

    cpr_top, cpr_bot = 101.0, 99.0
    atr = 1.0
    last_close = 102.0

    df = _build_lull_df(
        now_time="10:00:00",
        last_close=last_close,
        cpr_mid=100.0,
        atr=atr,
        cur_volume=2000,
        intraday_avg_volume=10000,
        candle_pattern="hammer",
    )
    ctx = _make_ctx(df, cpr_top=cpr_top, cpr_bottom=cpr_bot, cap_segment="small_cap", atr=atr)

    result = det.detect(ctx)
    assert result.structure_detected is False
    assert "window" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 4: does not fire when close is too close to CPR midpoint
# ---------------------------------------------------------------------------

def test_does_not_fire_close_to_cpr():
    """close=100.5 (only 0.5 ATR above cpr_mid=100), < min 1.0 ATR → no fire."""
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)

    cpr_top, cpr_bot = 101.0, 99.0   # cpr_mid = 100.0
    atr = 1.0
    last_close = 100.5  # 0.5 ATR above cpr_mid — below threshold

    df = _build_lull_df(
        now_time="12:00:00",
        last_close=last_close,
        cpr_mid=100.0,
        atr=atr,
        cur_volume=2000,
        intraday_avg_volume=10000,
        candle_pattern="hammer",
    )
    ctx = _make_ctx(df, cpr_top=cpr_top, cpr_bottom=cpr_bot, cap_segment="small_cap", atr=atr)

    result = det.detect(ctx)
    assert result.structure_detected is False
    assert "distance" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 5: does not fire with high volume
# ---------------------------------------------------------------------------

def test_does_not_fire_with_high_volume():
    """cur_volume=8000 is 80% of intraday_avg (10000), above 30% threshold → no fire."""
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)

    cpr_top, cpr_bot = 101.0, 99.0
    atr = 1.0
    last_close = 102.0

    df = _build_lull_df(
        now_time="12:00:00",
        last_close=last_close,
        cpr_mid=100.0,
        atr=atr,
        cur_volume=8000,       # 80% of avg — well above 30% max
        intraday_avg_volume=10000,
        candle_pattern="hammer",
    )
    ctx = _make_ctx(df, cpr_top=cpr_top, cpr_bottom=cpr_bot, cap_segment="small_cap", atr=atr)

    result = det.detect(ctx)
    assert result.structure_detected is False
    assert "volume" in (result.rejection_reason or "").lower()
