"""MIS unwind short detector unit tests (sub7-T2)."""
import pandas as pd
from datetime import datetime
from structures.mis_unwind_short_structure import MISUnwindShortStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "mis_unwind_short",
        "enabled": True,
        "active_window_start": "14:55",
        "active_window_end": "15:15",
        "min_distance_above_vwap_pct": 0.5,
        "min_intraday_high_recency_min": 30,
        "max_momentum_3bar_pct": 0.0,
        "min_rvol": 1.2,
        "allowed_cap_segments": ["small_cap", "mid_cap"],
        "stop_atr_buffer": 0.8,
        "target_type": "vwap",
        "time_stop_min_before_close": 5,
        "min_bars_required": 30,
    }


def _build_df(now_time, n_bars=40, last_close=105.0, vwap=100.0,
              recent_high_offset_min=15, momentum_3bar_pct=-0.3):
    """Build a minimal df where the last bar is at `now_time`,
    with a fresh intraday high `recent_high_offset_min` ago and weakening momentum."""
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5*(n_bars-1)), periods=n_bars, freq="5min")
    closes = [vwap] * n_bars
    highs = [vwap + 0.5] * n_bars
    # Set a fresh intraday high N minutes ago
    high_bar = max(0, n_bars - 1 - (recent_high_offset_min // 5))
    highs[high_bar] = last_close + 1.0  # the fresh high
    closes[-1] = last_close             # current close above VWAP
    # Momentum_3bar_pct calculated from closes[-4] to closes[-1]
    closes[-4] = last_close - (momentum_3bar_pct / 100.0) * last_close
    df = pd.DataFrame({
        "open": closes, "high": highs,
        "low": [c - 0.5 for c in closes], "close": closes,
        "volume": [10000] * n_bars,
        "vwap": [vwap] * n_bars,
    }, index=idx)
    return df


def _make_ctx(df, cap_segment="small_cap", rvol=1.5, atr=1.0):
    """Build a MarketContext compatible with the actual data_models signature.
    rvol and atr are passed via the indicators dict."""
    last_ts = df.index[-1]
    return MarketContext(
        symbol="NSE:SYM",
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment=cap_segment,
        regime="trend_up",
        indicators={"rvol": rvol, "atr": atr},
    )


def test_fires_in_window_with_valid_setup():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("15:00:00", recent_high_offset_min=15, momentum_3bar_pct=-0.5)
    ctx = _make_ctx(df, cap_segment="small_cap", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is True
    assert any(e.structure_type == "mis_unwind_short" for e in result.events)


def test_does_not_fire_outside_window():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("11:00:00")  # well before active window
    ctx = _make_ctx(df, cap_segment="small_cap", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is False


def test_does_not_fire_below_vwap():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("15:00:00", last_close=99.0, vwap=100.0)  # below VWAP
    ctx = _make_ctx(df, cap_segment="small_cap", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is False


def test_does_not_fire_with_positive_momentum():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("15:00:00", momentum_3bar_pct=+0.5)  # still going up
    ctx = _make_ctx(df, cap_segment="small_cap", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is False


def test_does_not_fire_in_disallowed_cap_segment():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("15:00:00")
    ctx = _make_ctx(df, cap_segment="large_cap", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is False
