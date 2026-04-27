"""PDH/PDL Touch-and-Reject detector unit tests (sub8-T9)."""
import pandas as pd

from structures.pdh_pdl_reject_structure import PDHPDLRejectStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "pdh_pdl_reject",
        "enabled": True,
        "active_window_start": "10:00",
        "active_window_end": "14:30",
        "level_proximity_pct": 0.10,
        "max_body_size_pct": 40.0,
        "min_upper_wick_x_body": 1.5,
        "volume_polarity": "absence",
        "max_volume_x_recent_for_absence": 1.5,
        "min_volume_x_recent_for_spike": 1.5,
        "wick_buffer_pct": 0.10,
        "t1_target": "vwap",
        "t2_target": "today_opposite_extreme",
        "t1_qty_pct": 0.5,
        "universe_key": "smallmid_fno",
        "min_bars_required": 30,
    }


def _build_pdh_reject_df(now_time="11:00:00", n_bars=40, prev_recent_vol=10000):
    """Build a session where price tags PDH=105.0 and prints rejection."""
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5 * (n_bars - 1)),
                        periods=n_bars, freq="5min")
    closes = [103.0 + 0.05 * (i % 5) for i in range(n_bars)]
    opens = [c - 0.05 for c in closes]
    highs = [c + 0.10 for c in closes]
    lows = [c - 0.10 for c in closes]
    volumes = [prev_recent_vol] * n_bars
    # Last bar: tag PDH with rejection candle
    # Body=0.05, upper_wick=0.50 (10x body), low rejection
    opens[-1] = 104.5
    closes[-1] = 104.55
    highs[-1] = 105.05
    lows[-1] = 104.45
    volumes[-1] = 10000  # NOT above 1.5x recent (absence polarity signal)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes,
                         "vwap": [103.0] * n_bars}, index=idx)


def _ctx(df, symbol="NSE:YESBANK", pdh=105.0, pdl=98.0, pdc=103.0):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment="mid_cap",
        regime="chop",
        pdh=pdh, pdl=pdl, pdc=pdc,
        indicators={"atr": 1.0, "vwap": 103.0},
    )


def test_fires_short_on_pdh_reject_with_no_breakout_volume():
    """Default polarity=absence: rejection at PDH without breakout volume -> short."""
    det = PDHPDLRejectStructure(_cfg())
    df = _build_pdh_reject_df()
    ctx = _ctx(df, pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"
    plan = det.plan_short_strategy(ctx)
    assert plan is not None
    assert len(plan.exit_levels.targets) == 2
    assert plan.risk_params.hard_sl > plan.entry_price  # short: stop above entry


def test_does_not_fire_on_breakout_volume_in_absence_polarity():
    """absence polarity (default): if volume > 1.5x recent, that's a breakout -- skip."""
    det = PDHPDLRejectStructure(_cfg())
    df = _build_pdh_reject_df()
    df.iloc[-1, df.columns.get_loc("volume")] = 20000  # 2x recent
    ctx = _ctx(df, pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "absence_polarity" in (res.rejection_reason or "").lower()


def test_fires_on_breakout_volume_in_spike_polarity():
    """rev2: spike polarity (A/B variant) -- bar vol >= 1.5x recent IS the signal."""
    cfg = _cfg()
    cfg["volume_polarity"] = "spike"
    det = PDHPDLRejectStructure(cfg)
    df = _build_pdh_reject_df()
    df.iloc[-1, df.columns.get_loc("volume")] = 20000  # 2x recent -> spike
    ctx = _ctx(df, pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire in spike polarity: {res.rejection_reason}"
    assert res.events[0].side == "short"


def test_does_not_fire_on_low_volume_in_spike_polarity():
    """spike polarity: if volume < 1.5x recent, no spike -- skip."""
    cfg = _cfg()
    cfg["volume_polarity"] = "spike"
    det = PDHPDLRejectStructure(cfg)
    df = _build_pdh_reject_df()
    df.iloc[-1, df.columns.get_loc("volume")] = 8000  # below 1.5x recent
    ctx = _ctx(df, pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "spike_polarity" in (res.rejection_reason or "").lower()


def test_does_not_fire_outside_universe():
    """Reject Nifty50 majors -- this fade is for retail-watched small/mid."""
    det = PDHPDLRejectStructure(_cfg())
    df = _build_pdh_reject_df()
    ctx = _ctx(df, symbol="NSE:RELIANCE", pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_high_far_from_pdh():
    """If bar high doesn't tag within 0.10% of PDH, skip."""
    det = PDHPDLRejectStructure(_cfg())
    df = _build_pdh_reject_df()
    df.iloc[-1, df.columns.get_loc("high")] = 104.5  # PDH=105.0, 0.5% away
    ctx = _ctx(df, pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
