"""Closing Hour Reversal detector unit tests (sub8-T11)."""
import pandas as pd

from structures.closing_hour_reversal_structure import ClosingHourReversalStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "closing_hour_reversal",
        "enabled": True,
        "active_window_start": "14:30",
        "active_window_end": "15:15",
        "min_intraday_move_pct": 1.5,
        "exhaustion_min_body_pct_of_range": 60.0,
        "exhaustion_min_volume_x_recent": 1.3,
        "stop_atr_multiplier": 1.5,
        "t1_target": "vwap",
        "t2_target": "pivot_or_50pct_retrace",
        "t1_qty_pct": 0.5,
        "hard_time_stop_hhmm": "15:22",
        "universe_key": "fno_liquid_200",
        "min_bars_required": 60,
    }


def _build_chr_df(now_time="14:35:00", n_bars=70, open_price=100.0, hod=104.0):
    """Stock ran from 100 to 104 (+4%) by 14:30, last bar prints exhaustion bearish."""
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5 * (n_bars - 1)),
                        periods=n_bars, freq="5min")
    # Trend up: 100 → 104 over first n_bars-1 bars
    trend_closes = [open_price + (hod - open_price) * (i / (n_bars - 2)) for i in range(n_bars - 1)]
    closes = trend_closes + [trend_closes[-1] - 0.5]  # last bar drops
    opens = [c - 0.05 for c in closes]
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    # Last bar = bearish exhaustion: large body (>= 60% of range), high volume
    opens[-1] = trend_closes[-1]
    closes[-1] = trend_closes[-1] - 0.6
    highs[-1] = trend_closes[-1] + 0.05
    lows[-1] = closes[-1] - 0.05
    volumes = [10000] * n_bars
    volumes[-1] = 14000  # 1.4× recent (above 1.3× threshold)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes,
                         "vwap": [102.0] * n_bars}, index=idx)


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
        indicators={"atr": 0.5, "vwap": 102.0},
    )


def test_fires_short_on_up_move_exhaustion():
    """Stock ran +4% then prints bearish exhaustion = SHORT (fade the up move)."""
    det = ClosingHourReversalStructure(_cfg())
    df = _build_chr_df()
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"
    plan = det.plan_short_strategy(ctx)
    assert plan is not None
    assert len(plan.exit_levels.targets) == 2
    assert plan.exit_levels.targets[0]["qty_pct"] == 0.5
    assert plan.risk_params.atr is not None


def test_does_not_fire_outside_window():
    det = ClosingHourReversalStructure(_cfg())
    df = _build_chr_df(now_time="13:00:00")
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_intraday_move_too_small():
    """If stock moved < 1.5% intraday, no exhaustion to fade."""
    det = ClosingHourReversalStructure(_cfg())
    df = _build_chr_df(open_price=100.0, hod=100.5)  # only 0.5%
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "move" in (res.rejection_reason or "").lower()


def test_does_not_fire_outside_universe():
    det = ClosingHourReversalStructure(_cfg())
    df = _build_chr_df()
    ctx = _ctx(df, symbol="NSE:UNKNOWNSMALLCAP")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()
