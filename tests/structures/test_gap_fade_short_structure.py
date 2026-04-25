"""Gap Fade Short detector unit tests (sub7-T4)."""
import pandas as pd
from datetime import datetime
from structures.gap_fade_short_structure import GapFadeShortStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "gap_fade_short",
        "enabled": True,
        "active_window_start": "09:15",
        "active_window_end": "09:30",
        "min_gap_pct_above_pdc": 1.5,
        "max_gap_pct_above_pdc": 8.0,
        "min_upper_wick_ratio": 0.5,
        "max_body_size_pct": 30.0,
        "require_volume_decline_after_gap": True,
        "allowed_cap_segments": ["small_cap", "mid_cap", "micro_cap"],
        "stop_above_gap_high_atr": 0.3,
        "target_type": "pdc_or_open",
        "time_stop_at": "10:00",
        "min_bars_required": 1,
    }


def _build_df(now_time, pdc=100.0, gap_pct=2.5,
              upper_wick_ratio=0.8, body_pct=20.0,
              vol_bar0=50000, vol_bar1=30000):
    """Build a 2-bar DataFrame for gap fade tests.

    Bar 0: the opening gap-up bar (09:15) — high volume, gap-up open above PDC.
    Bar 1: the current bar (now_time) — exhaustion candle (big upper wick, small body).

    Args:
        pdc: Previous day close
        gap_pct: How much above PDC the open gaps (%)
        upper_wick_ratio: upper_wick / body for bar 1
        body_pct: body / open * 100 for bar 1
        vol_bar0: volume for opening bar (gap bar)
        vol_bar1: volume for current bar
    """
    # --- Bar 0: opening gap-up bar at 09:15 ---
    bar0_open = pdc * (1 + gap_pct / 100.0)
    bar0_close = bar0_open * 1.005   # slight bullish close
    bar0_high = bar0_open * 1.015
    bar0_low = bar0_open * 0.998

    # --- Bar 1: exhaustion candle at now_time ---
    bar1_open = bar0_close
    # body_pct = body / open * 100  =>  body = open * body_pct / 100
    # The full body IS the candle body (close - open = body, bullish)
    body = bar1_open * body_pct / 100.0
    bar1_close = bar1_open + body          # full body: body_size_pct = body/open*100 exactly
    # upper_wick = upper_wick_ratio * body
    upper_wick = upper_wick_ratio * body
    bar1_high = bar1_close + upper_wick    # top of body + upper wick
    bar1_low = bar1_open - body * 0.1      # tiny lower wick

    dates = [
        pd.Timestamp(f"2025-01-02 09:15:00"),
        pd.Timestamp(f"2025-01-02 {now_time}"),
    ]
    df = pd.DataFrame(
        {
            "open":   [bar0_open,   bar1_open],
            "high":   [bar0_high,   bar1_high],
            "low":    [bar0_low,    bar1_low],
            "close":  [bar0_close,  bar1_close],
            "volume": [vol_bar0,    vol_bar1],
        },
        index=dates,
    )
    return df


def _make_ctx(df, pdc=100.0, cap_segment="small_cap", atr=1.0):
    """Build a MarketContext compatible with data_models signature.
    atr is passed via indicators dict; pdc via the pdc field."""
    last_ts = df.index[-1]
    return MarketContext(
        symbol="NSE:TESTSYM",
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment=cap_segment,
        regime="chop",
        pdc=pdc,
        indicators={"atr": atr},
    )


# ---------------------------------------------------------------------------
# Test 1: valid gap-fade pattern fires
# ---------------------------------------------------------------------------

def test_fires_with_valid_gap_fade_pattern():
    """gap=2.5%, upper_wick_ratio=0.8, body_pct=20%, vol decline -> fires."""
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    pdc = 100.0
    df = _build_df("09:20:00", pdc=pdc, gap_pct=2.5,
                   upper_wick_ratio=0.8, body_pct=20.0,
                   vol_bar0=50000, vol_bar1=30000)
    ctx = _make_ctx(df, pdc=pdc, cap_segment="small_cap", atr=1.0)
    result = det.detect(ctx)
    assert result.structure_detected is True, f"Expected fire, got: {result.rejection_reason}"
    assert len(result.events) == 1
    assert result.events[0].side == "short"


# ---------------------------------------------------------------------------
# Test 2: outside active window → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_outside_window():
    """Bar at 11:00 is outside 09:15-09:30 window."""
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    pdc = 100.0
    df = _build_df("11:00:00", pdc=pdc, gap_pct=2.5,
                   upper_wick_ratio=0.8, body_pct=20.0,
                   vol_bar0=50000, vol_bar1=30000)
    ctx = _make_ctx(df, pdc=pdc, cap_segment="small_cap", atr=1.0)
    result = det.detect(ctx)
    assert result.structure_detected is False
    assert "window" in (result.rejection_reason or "").lower()


# ---------------------------------------------------------------------------
# Test 3: gap below min threshold → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_below_gap_threshold():
    """gap=0.5% < min_gap_pct_above_pdc (1.5%) → no fire."""
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    pdc = 100.0
    df = _build_df("09:20:00", pdc=pdc, gap_pct=0.5,
                   upper_wick_ratio=0.8, body_pct=20.0,
                   vol_bar0=50000, vol_bar1=30000)
    ctx = _make_ctx(df, pdc=pdc, cap_segment="small_cap", atr=1.0)
    result = det.detect(ctx)
    assert result.structure_detected is False


# ---------------------------------------------------------------------------
# Test 4: gap above max threshold → no fire (halted territory)
# ---------------------------------------------------------------------------

def test_does_not_fire_above_max_gap():
    """gap=10% > max_gap_pct_above_pdc (8.0%) → no fire."""
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    pdc = 100.0
    df = _build_df("09:20:00", pdc=pdc, gap_pct=10.0,
                   upper_wick_ratio=0.8, body_pct=20.0,
                   vol_bar0=50000, vol_bar1=30000)
    ctx = _make_ctx(df, pdc=pdc, cap_segment="small_cap", atr=1.0)
    result = det.detect(ctx)
    assert result.structure_detected is False


# ---------------------------------------------------------------------------
# Test 5: strong body (no exhaustion) → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_with_strong_body():
    """body_pct=80% far exceeds max_body_size_pct (30%) → no exhaustion → no fire."""
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    pdc = 100.0
    df = _build_df("09:20:00", pdc=pdc, gap_pct=2.5,
                   upper_wick_ratio=0.8, body_pct=80.0,
                   vol_bar0=50000, vol_bar1=30000)
    ctx = _make_ctx(df, pdc=pdc, cap_segment="small_cap", atr=1.0)
    result = det.detect(ctx)
    assert result.structure_detected is False


# ---------------------------------------------------------------------------
# Test 6: disallowed cap segment → no fire
# ---------------------------------------------------------------------------

def test_does_not_fire_in_disallowed_cap_segment():
    """large_cap is not in allowed_cap_segments → no fire."""
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    pdc = 100.0
    df = _build_df("09:20:00", pdc=pdc, gap_pct=2.5,
                   upper_wick_ratio=0.8, body_pct=20.0,
                   vol_bar0=50000, vol_bar1=30000)
    ctx = _make_ctx(df, pdc=pdc, cap_segment="large_cap", atr=1.0)
    result = det.detect(ctx)
    assert result.structure_detected is False
