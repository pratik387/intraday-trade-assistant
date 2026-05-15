"""Long Panic Gap Down detector unit tests.

Cell B + 1.5R/2.5R targets validated on Discovery/OOS/Holdout (see
specs/2026-05-15-edge-discovery-findings.md and
tools/sub9_research/sanity_long_panic_gap_down.py).
"""
import pandas as pd
import pytest

from structures.long_panic_gap_down_structure import LongPanicGapDownStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "long_panic_gap_down",
        "enabled": True,
        "active_window_start": "09:15",
        "active_window_end": "09:20",
        "gap_pct_max": -1.0,
        "dist_from_pdh_pct_max": -5.5,
        "dist_from_pdl_pct_min": -5.0,
        "dist_from_pdl_pct_max": -3.0,
        "allowed_cap_segments": ["small_cap", "mid_cap"],
        "sl_buffer_below_bar_low_pct": 0.5,
        "min_stop_pct": 1.0,
        "t1_r_multiple": 1.5,
        "t2_r_multiple": 2.5,
        "t1_partial_qty_pct": 0.5,
        "time_stop_at": "15:10",
        "min_bars_required": 1,
        "entry_zone_pct": 0.3,
        "entry_zone_mode": "symmetric",
        "min_stop_distance_pct": 0.5,
    }


def _ctx(symbol="TEST", *, bar_open, bar_high, bar_low, bar_close,
         pdh, pdl, pdc, cap_segment="small_cap",
         ts=pd.Timestamp("2026-05-20 09:15:00")):
    df = pd.DataFrame({
        "open": [bar_open], "high": [bar_high],
        "low": [bar_low], "close": [bar_close], "volume": [50000],
    }, index=[ts])
    return MarketContext(
        symbol=symbol, current_price=bar_close, timestamp=ts,
        df_5m=df, session_date=ts.date(),
        pdh=pdh, pdl=pdl, pdc=pdc, cap_segment=cap_segment,
    )


def test_cell_b_trigger_fires():
    """Cell B canonical: gap=-6%, dist_pdh=-13.3%, dist_pdl=-4.2% (in band)."""
    det = LongPanicGapDownStructure(_cfg())
    ctx = _ctx(bar_open=94.0, bar_high=94.5, bar_low=89.5, bar_close=91.0,
               pdh=105.0, pdl=95.0, pdc=100.0)
    r = det.detect(ctx)
    assert r.structure_detected
    e = r.events[0]
    assert e.side == "long"
    assert e.structure_type == "long_panic_gap_down"
    assert abs(e.context["gap_pct"] - (-6.0)) < 1e-9
    assert -5.0 <= e.context["dist_from_pdl_pct"] <= -3.0


def test_rejects_shallow_dist_from_pdl():
    """dist_from_pdl outside [-5%, -3%] band should reject."""
    det = LongPanicGapDownStructure(_cfg())
    # close=94, pdl=95 → dist_pdl ≈ -1.05% (above band)
    ctx = _ctx(bar_open=97.0, bar_high=97.5, bar_low=93.5, bar_close=94.0,
               pdh=105.0, pdl=95.0, pdc=100.0)
    r = det.detect(ctx)
    assert not r.structure_detected
    assert "dist_from_pdl" in r.rejection_reason


def test_rejects_when_gap_too_small():
    """gap_pct > -1% (i.e., not enough gap-down) should reject."""
    det = LongPanicGapDownStructure(_cfg())
    # open=99.7 vs pdc=100 → gap_pct = -0.3% (above max=-1.0)
    ctx = _ctx(bar_open=99.7, bar_high=99.8, bar_low=90.0, bar_close=91.0,
               pdh=105.0, pdl=95.0, pdc=100.0)
    r = det.detect(ctx)
    assert not r.structure_detected
    assert "gap_pct" in r.rejection_reason


def test_rejects_cap_segment_not_allowed():
    """large_cap not in allowed_cap_segments → reject."""
    det = LongPanicGapDownStructure(_cfg())
    ctx = _ctx(bar_open=94.0, bar_high=94.5, bar_low=89.5, bar_close=91.0,
               pdh=105.0, pdl=95.0, pdc=100.0, cap_segment="large_cap")
    r = det.detect(ctx)
    assert not r.structure_detected
    assert "Cap segment" in r.rejection_reason


def test_rejects_dist_from_pdh_too_shallow():
    """If symbol is < 5.5% below PDH, reject."""
    det = LongPanicGapDownStructure(_cfg())
    # close=101 vs pdh=105 → dist_pdh = -3.8% (above max=-5.5)
    ctx = _ctx(bar_open=99.0, bar_high=101.5, bar_low=98.5, bar_close=101.0,
               pdh=105.0, pdl=95.0, pdc=100.0)
    r = det.detect(ctx)
    assert not r.structure_detected
    assert "dist_from_pdh" in r.rejection_reason


def test_latch_prevents_repeat_within_session():
    """One trigger per (symbol, session_date)."""
    det = LongPanicGapDownStructure(_cfg())
    ctx = _ctx(bar_open=94.0, bar_high=94.5, bar_low=89.5, bar_close=91.0,
               pdh=105.0, pdl=95.0, pdc=100.0)
    r1 = det.detect(ctx)
    assert r1.structure_detected
    r2 = det.detect(ctx)
    assert not r2.structure_detected
    assert "already fired" in r2.rejection_reason


def test_outside_active_window_rejects():
    """Bars after 09:20 must not trigger."""
    det = LongPanicGapDownStructure(_cfg())
    ts_late = pd.Timestamp("2026-05-20 10:00:00")
    df = pd.DataFrame({
        "open": [94.0], "high": [94.5],
        "low": [89.5], "close": [91.0], "volume": [50000],
    }, index=[ts_late])
    ctx = MarketContext(
        symbol="TEST", current_price=91.0, timestamp=ts_late,
        df_5m=df, session_date=ts_late.date(),
        pdh=105.0, pdl=95.0, pdc=100.0, cap_segment="small_cap",
    )
    r = det.detect(ctx)
    assert not r.structure_detected
    assert "active window" in r.rejection_reason


def test_plan_has_correct_target_geometry():
    """T1 must be at +1.5R and T2 at +2.5R from entry."""
    det = LongPanicGapDownStructure(_cfg())
    ctx = _ctx(bar_open=94.0, bar_high=94.5, bar_low=89.5, bar_close=91.0,
               pdh=105.0, pdl=95.0, pdc=100.0)
    r = det.detect(ctx)
    plan = det.plan_long_strategy(ctx, event=r.events[0])
    assert plan is not None
    entry = plan.entry_price
    sl = plan.risk_params.hard_sl
    assert sl < entry, "long SL must be below entry"
    R = entry - sl
    t1 = next(t for t in plan.exit_levels.targets if t["name"] == "T1")
    t2 = next(t for t in plan.exit_levels.targets if t["name"] == "T2")
    assert abs(t1["level"] - (entry + 1.5 * R)) < 1e-6
    assert abs(t2["level"] - (entry + 2.5 * R)) < 1e-6
    assert t1["qty_pct"] == 0.5
    assert t2["qty_pct"] == 0.5
    assert plan.exit_levels.time_exit == "15:10"


def test_sl_takes_deeper_of_two_stops():
    """SL = min(entry_bar_low × (1 - 0.5%), entry × (1 - 1.0%))."""
    det = LongPanicGapDownStructure(_cfg())
    # entry_bar_low=89.5, close=91 → sl_from_low = 89.05; sl_from_min = 90.09
    # min(89.05, 90.09) = 89.05 (from low, deeper)
    ctx = _ctx(bar_open=94.0, bar_high=94.5, bar_low=89.5, bar_close=91.0,
               pdh=105.0, pdl=95.0, pdc=100.0)
    r = det.detect(ctx)
    plan = det.plan_long_strategy(ctx, event=r.events[0])
    assert plan is not None
    # bar_low * (1 - 0.005) = 89.5 * 0.995 = 89.0525
    assert abs(plan.risk_params.hard_sl - 89.0525) < 1e-4


def test_short_plan_returns_none():
    """Long-only setup: plan_short_strategy must return None."""
    det = LongPanicGapDownStructure(_cfg())
    ctx = _ctx(bar_open=94.0, bar_high=94.5, bar_low=89.5, bar_close=91.0,
               pdh=105.0, pdl=95.0, pdc=100.0)
    r = det.detect(ctx)
    assert det.plan_short_strategy(ctx, event=r.events[0]) is None
