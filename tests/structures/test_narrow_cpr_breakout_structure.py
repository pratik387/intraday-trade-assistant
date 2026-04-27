"""Narrow CPR Trending Breakout detector unit tests (sub8-T5)."""
import pandas as pd

from structures.narrow_cpr_breakout_structure import NarrowCPRBreakoutStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "narrow_cpr_breakout",
        "enabled": True,
        "active_window_start": "09:30",
        "active_window_end": "14:00",
        "max_cpr_width_pct": 0.40,
        "min_volume_x_20d_median": 1.3,
        "anti_whipsaw_lookback_bars": 2,
        "stop_at_pivot": True,
        "t1_target": "r1_s1",
        "t2_target": "r2_s2",
        "t1_qty_pct": 0.5,
        "universe_key": "nifty50_banknifty",
        "min_bars_required": 30,
    }


def _build_df(now_time="11:00:00", breakout_close=101.0, breakout_volume=13000,
              median_volume=10000, n_bars=40):
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5 * (n_bars - 1)),
                        periods=n_bars, freq="5min")
    # Tight pre-breakout: hovering 100.0 +/- 0.1
    opens = [100.0] * n_bars
    highs = [100.1] * n_bars
    lows = [99.9] * n_bars
    closes = [100.0] * n_bars
    volumes = [median_volume] * n_bars
    # Last bar = breakout
    opens[-1] = 100.0
    closes[-1] = breakout_close
    highs[-1] = max(breakout_close, 100.5) + 0.05
    lows[-1] = min(opens[-1], breakout_close) - 0.05
    volumes[-1] = breakout_volume
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes}, index=idx)


def _ctx(df, symbol="NSE:HDFCBANK", pdh=100.6, pdl=99.8, pdc=100.0):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment="large_cap",
        regime="chop",
        pdh=pdh, pdl=pdl, pdc=pdc,
        indicators={"atr": 0.5, "median_volume_20d": 10000},
    )


def test_fires_long_on_close_above_tc_with_volume():
    """Narrow CPR (~0.13% width) + close above TC + volume 1.3x → long.

    PDH=100.6, PDL=99.8, PDC=100.0:
      pivot=(100.6+99.8+100.0)/3 = 100.13
      bc=(100.6+99.8)/2 = 100.20
      tc=2*100.13 - 100.20 = 100.07
      Normalized: cpr_top=100.20, cpr_bottom=100.07
      Width = (100.20-100.07)/100.13 * 100 = 0.13% < 0.40% threshold ✓
    Breakout close 101.0 > cpr_top 100.20 → long.
    """
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(breakout_close=101.0, breakout_volume=13000)
    ctx = _ctx(df, pdh=100.6, pdl=99.8, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"
    plan = det.plan_long_strategy(ctx)
    assert plan is not None
    assert len(plan.exit_levels.targets) == 2  # T1+T2 mandatory
    assert plan.exit_levels.targets[0]["qty_pct"] == 0.5
    assert plan.risk_params.hard_sl < plan.entry_price  # stop below entry for long


def test_fires_short_on_close_below_bc_with_volume():
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(breakout_close=99.0, breakout_volume=13000)
    ctx = _ctx(df, pdh=100.6, pdl=99.8, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"


def test_does_not_fire_on_wide_cpr():
    """Wide CPR (width > 0.40%) should be rejected."""
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(breakout_close=101.0, breakout_volume=13000)
    # Wider PDH/PDL → wider CPR
    ctx = _ctx(df, pdh=102.0, pdl=98.0, pdc=100.0)  # width ~2%
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "cpr_width" in (res.rejection_reason or "").lower()


def test_does_not_fire_outside_universe():
    """Non-Nifty50/BankNifty symbol rejected."""
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(breakout_close=101.0, breakout_volume=13000)
    ctx = _ctx(df, symbol="NSE:RANDOMSMALLCAP", pdh=100.6, pdl=99.8, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_on_low_volume():
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(breakout_close=101.0, breakout_volume=8000)  # below 1.3× median
    ctx = _ctx(df, pdh=100.6, pdl=99.8, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "volume" in (res.rejection_reason or "").lower()


def test_does_not_fire_outside_window():
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(now_time="14:30:00", breakout_close=101.0, breakout_volume=13000)
    ctx = _ctx(df, pdh=100.6, pdl=99.8, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()
