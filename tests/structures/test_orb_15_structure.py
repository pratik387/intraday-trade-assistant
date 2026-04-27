"""ORB-15 detector unit tests (sub8-T3)."""
import pandas as pd
import pytest

from structures.orb_15_structure import ORB15Structure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "orb_15",
        "enabled": True,
        "active_window_start": "09:30",
        "active_window_end": "11:15",
        "range_window_start": "09:20",
        "range_window_end": "09:30",
        "min_range_pct": 0.4,
        "max_range_pct": 2.0,
        "min_volume_x_30d_median": 1.5,
        "stop_at_range_midpoint": False,
        "wick_buffer_pct": 0.10,
        "t1_r_multiple": 1.0,
        "t2_r_multiple": 2.0,
        "t1_qty_pct": 0.5,
        "universe_key": "fno_liquid_200",
        "min_bars_required": 4,
        "max_gap_pct_for_orb": 0.5,
    }


def _build_orb_df(now_time="09:35:00", range_high=102.0, range_low=100.0,
                  breakout_close=102.5, breakout_volume=15000, median_volume=10000):
    """Build a 6+ bar 5m DataFrame: range bars + entry-window bars.

    Range window: 09:20-09:30 (2 bars: 09:20, 09:25).
    Last bar = breakout candidate.
    """
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    n_entry_bars = max(1, int((end - pd.Timestamp("2025-01-02 09:30:00")).total_seconds() / 300) + 1)
    n_total = 2 + n_entry_bars  # 2 range bars (09:20, 09:25) + entry bars
    idx = pd.date_range("2025-01-02 09:20:00", periods=n_total, freq="5min")

    opens = [(range_high + range_low) / 2] * n_total
    highs = [range_high] * n_total
    lows = [range_low] * n_total
    closes = [(range_high + range_low) / 2] * n_total
    volumes = [median_volume] * n_total

    opens[-1] = (range_high + range_low) / 2
    closes[-1] = breakout_close
    highs[-1] = max(breakout_close, range_high) + 0.05
    lows[-1] = min(opens[-1], breakout_close) - 0.05
    volumes[-1] = breakout_volume
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes}, index=idx)


def _ctx(df, symbol="NSE:RELIANCE", cap_segment="large_cap", median_volume=10000, pdc=101.0):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment=cap_segment,
        regime="trend_up",
        pdh=110.0, pdl=98.0, pdc=pdc,
        indicators={"atr": 1.0, "median_volume_30d": median_volume},
    )


def test_fires_long_on_upside_break_with_volume():
    """Range [100, 102], breakout close 102.5, volume 1.5×, on F&O sym → long."""
    det = ORB15Structure(_cfg())
    df = _build_orb_df(now_time="09:35:00", breakout_close=102.5, breakout_volume=15000)
    ctx = _ctx(df, symbol="NSE:RELIANCE", pdc=101.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"

    plan = det.plan_long_strategy(ctx)
    assert plan is not None
    assert plan.side == "long"
    assert plan.risk_params.hard_sl < plan.entry_price
    targets = plan.exit_levels.targets
    assert len(targets) == 2, "ORB MUST emit T1 + T2 (tiered exits required)"
    assert targets[0]["qty_pct"] == 0.5
    assert targets[1]["qty_pct"] == 0.5
    assert targets[0]["level"] < targets[1]["level"]


def test_fires_short_on_downside_break_with_volume():
    det = ORB15Structure(_cfg())
    df = _build_orb_df(now_time="09:35:00", breakout_close=99.5, breakout_volume=15000)
    ctx = _ctx(df, symbol="NSE:RELIANCE", pdc=101.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"


def test_does_not_fire_on_low_volume_break():
    det = ORB15Structure(_cfg())
    df = _build_orb_df(breakout_close=102.5, breakout_volume=8000, median_volume=10000)
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "volume" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_close_inside_range():
    det = ORB15Structure(_cfg())
    df = _build_orb_df(breakout_close=101.0, breakout_volume=15000)
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False


def test_does_not_fire_outside_active_window():
    det = ORB15Structure(_cfg())
    df = _build_orb_df(now_time="14:00:00", breakout_close=102.5, breakout_volume=15000)
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_symbol_outside_universe():
    """ORB only runs on F&O liquid universe — small caps rejected."""
    det = ORB15Structure(_cfg())
    df = _build_orb_df(breakout_close=102.5, breakout_volume=15000)
    ctx = _ctx(df, symbol="NSE:RANDOMSMALLCAP")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_on_gap_day_routes_to_gap_fade():
    """rev2: ORB disabled if open gap > 0.5% (route to gap_fade_short)."""
    det = ORB15Structure(_cfg())
    df = _build_orb_df(breakout_close=102.5, breakout_volume=15000)
    # PDC=100, range opens at midpoint 101 → gap is 1% → > 0.5% threshold → exclude
    ctx = _ctx(df, symbol="NSE:RELIANCE", pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "gap_day" in (res.rejection_reason or "").lower()
