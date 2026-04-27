"""VWAP First-Pullback detector unit tests (sub8-T7)."""
import pandas as pd

from structures.vwap_first_pullback_structure import VWAPFirstPullbackStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "vwap_first_pullback",
        "enabled": True,
        "active_window_start": "10:00",
        "active_window_end": "14:30",
        "trend_lookback_bars": 6,
        "trend_min_bars_same_side": 4,
        "pullback_proximity_pct": 0.10,
        "reversal_min_range_pct": 0.20,
        "max_stop_distance_pct": 0.6,
        "t1_target": "prev_swing_extreme",
        "t2_r_multiple": 2.0,
        "t1_qty_pct": 0.5,
        "universe_key": "fno_liquid_200",
        "min_bars_required": 30,
    }


def _build_uptrend_pullback_df(now_time="11:00:00", n_bars=40, vwap=100.0):
    """Uptrend: bars 0..N-3 trend up above VWAP, bar N-2 pulls back to VWAP,
    bar N-1 (last) is reversal candle that closes back above VWAP."""
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5 * (n_bars - 1)),
                        periods=n_bars, freq="5min")
    closes = [vwap + 0.5 + i * 0.05 for i in range(n_bars - 2)]
    closes.append(vwap + 0.05)            # pullback bar (closes near VWAP)
    closes.append(vwap + 0.4)             # reversal bar (closes back above VWAP)

    opens = [c - 0.05 for c in closes]
    highs = [c + 0.10 for c in closes]
    lows = [c - 0.10 for c in closes]
    lows[-2] = vwap - 0.02                # pullback bar low touches VWAP
    opens[-1] = closes[-2]
    highs[-1] = closes[-1] + 0.05
    lows[-1] = opens[-1] - 0.02
    volumes = [10000] * n_bars
    volumes[-1] = 12000                   # reversal vol > prior
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes,
                         "vwap": [vwap] * n_bars}, index=idx)


def _ctx(df, symbol="NSE:RELIANCE"):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment="large_cap",
        regime="trend_up",
        pdh=110.0, pdl=98.0, pdc=100.0,
        indicators={"atr": 0.5, "vwap": 100.0},
    )


def test_fires_long_on_first_vwap_pullback_in_uptrend():
    det = VWAPFirstPullbackStructure(_cfg())
    df = _build_uptrend_pullback_df()
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"
    plan = det.plan_long_strategy(ctx)
    assert plan is not None
    assert len(plan.exit_levels.targets) == 2
    assert plan.exit_levels.targets[0]["qty_pct"] == 0.5
    assert plan.risk_params.hard_sl < plan.entry_price


def test_does_not_fire_outside_universe():
    det = VWAPFirstPullbackStructure(_cfg())
    df = _build_uptrend_pullback_df()
    ctx = _ctx(df, symbol="NSE:UNKNOWNSMALLCAP")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_outside_window():
    """rev2: window is 10:00-14:30, NOT 13:30."""
    det = VWAPFirstPullbackStructure(_cfg())
    df = _build_uptrend_pullback_df(now_time="15:00:00")  # past 14:30 cutoff
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


def test_fires_within_extended_window_at_14_15():
    """rev2: window extended to 14:30 — verify 14:15 still fires (was rejected by rev1's 13:30 cap)."""
    det = VWAPFirstPullbackStructure(_cfg())
    df = _build_uptrend_pullback_df(now_time="14:15:00")
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire at 14:15 (rev2 window): {res.rejection_reason}"


def test_does_not_fire_when_no_trend():
    """If the trend window alternates above/below VWAP, no clear trend, skip."""
    det = VWAPFirstPullbackStructure(_cfg())
    df = _build_uptrend_pullback_df()
    # Alternate closes: 3 above, 3 below — neither side reaches trend_min_same=4
    close_col = df.columns.get_loc("close")
    for j, i in enumerate(range(-8, -2)):
        df.iloc[i, close_col] = 100.3 if j % 2 == 0 else 99.7
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "trend" in (res.rejection_reason or "").lower()
