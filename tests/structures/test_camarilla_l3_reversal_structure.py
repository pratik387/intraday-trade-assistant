"""Camarilla L3/H3 Reversal detector unit tests.

Mechanic per
specs/2026-04-29-camarilla_l3_reversal-plan.md and
specs/2026-04-29-research-new-indian-setup-candidates.md (§ Candidate 1).

Camarilla pivot formulas (CANONICAL — NOT Floor Pivot):
  H3 = PDC + 0.275 * (PDH - PDL)
  L3 = PDC - 0.275 * (PDH - PDL)
  H4 = PDC + 0.55  * (PDH - PDL)
  L4 = PDC - 0.55  * (PDH - PDL)

L3-long sweep+reclaim (mirror H3-short):
  bar t-1: low < L3*(1-pen) AND close >= L3*(1+rec_buf) AND bullish bar
  bar t:   close > L3*(1+rec_buf) AND > pending.sweep_close → fire LONG

Tests cover:
  - Camarilla formula correctness (PINNED)
  - Happy-path L3 long + H3 short
  - ADX regime gate (>= max blocks)
  - Mutual exclusion vs pdh_pdl_sweep_reclaim (L3 ≈ PDL)
  - Sub-threshold penetration / no reclaim / no continuation → no fire
  - Outside active window / outside universe / cap_segment guards
  - First-trigger latch
  - Plan emits hard_sl ≈ L4/H4, T1 = pivot P, T2 = opposite L3/H3
  - Wide-open bypasses ADX + mutual-exclusion (geometry stays active)
"""
from __future__ import annotations

import pandas as pd
import pytest

from structures.camarilla_l3_reversal_structure import (
    CamarillaL3ReversalStructure,
    _camarilla_levels,
)
from structures.data_models import MarketContext


def _cfg(**overrides):
    cfg = {
        "_setup_name": "camarilla_l3_reversal",
        "enabled": True,
        "active_window_start": "10:00",
        "active_window_end": "14:00",
        "sweep_penetration_pct": 0.10,
        "reclaim_buffer_pct": 0.05,
        "max_adx_for_revert": 25,
        "pdh_pdl_proximity_skip_pct": 0.3,
        "t1_target": "pivot_p",
        "t2_target": "opposite_l3_or_h3",
        "t1_qty_pct": 0.5,
        "stop_atr_buffer": 0.0,
        "wick_buffer_pct": 0.05,
        "allowed_sides": ["long", "short"],
        "allowed_cap_segments": ["large_cap", "mid_cap", "small_cap"],
        "universe_key": "fno_liquid_200",
        "min_bars_required": 30,
        "entry_zone_pct": 0.10,
        "entry_zone_mode": "symmetric",
        "min_stop_distance_pct": 0.3,
    }
    cfg.update(overrides)
    return cfg


# Test PDH/PDL/PDC values that produce clean Camarilla levels for fixtures
_PDH = 110.0
_PDL = 100.0
_PDC = 105.0
# Range = 10 → H3 = 105 + 2.75 = 107.75, L3 = 105 - 2.75 = 102.25
#                H4 = 105 + 5.5 = 110.5,  L4 = 105 - 5.5 =  99.5


def _build_l3_long_session(
    sweep_time="11:25:00",
    confirm_time="11:30:00",
    pdh=_PDH, pdl=_PDL, pdc=_PDC,
    n_filler=30,
    adx=18.0,
):
    """Build a session ending at confirm_time with the canonical L3-long
    sweep+reclaim+confirm 3-bar sequence.

    Camarilla L3 = PDC - 0.275 * (PDH - PDL). With test PDH/PDL/PDC: L3=102.25.
    Sweep bar: low penetrates 0.15% below L3 (= 102.097), close 0.10% above
    L3 (= 102.352), bullish.
    Confirm bar: close > L3*(1+rec_buf) AND > sweep_close.
    """
    levels = _camarilla_levels(pdh, pdl, pdc)
    l3 = levels["L3"]

    end_ts = pd.Timestamp(f"2025-01-08 {confirm_time}")  # Wed
    sweep_ts = pd.Timestamp(f"2025-01-08 {sweep_time}")
    range_idx = pd.date_range(
        sweep_ts - pd.Timedelta(minutes=5 * n_filler),
        periods=n_filler,
        freq="5min",
    )

    rows = []
    # Filler bars: chop band between L3 and H3 (above L3)
    base = (l3 + levels["H3"]) / 2.0
    for ts in range_idx:
        rows.append({
            "ts": ts,
            "open": base, "high": base + 0.05, "low": base - 0.05,
            "close": base + 0.02, "volume": 10000,
            "adx": adx,
        })
    # Sweep bar: bullish trap candle at L3
    sweep_low = l3 * (1.0 - 0.0015)              # -0.15% penetration
    sweep_close = l3 * (1.0 + 0.0010)            # +0.10% reclaim
    sweep_high = sweep_close + 0.05
    sweep_open = l3 - 0.10
    rows.append({
        "ts": sweep_ts,
        "open": sweep_open, "high": sweep_high, "low": sweep_low,
        "close": sweep_close, "volume": 12000, "adx": adx,
    })
    # Confirm bar: close > L3*(1+rec_buf) AND > sweep_close
    confirm_close = sweep_close + 0.10
    rows.append({
        "ts": end_ts,
        "open": sweep_close,
        "high": confirm_close + 0.02,
        "low": sweep_close - 0.02,
        "close": confirm_close,
        "volume": 11000,
        "adx": adx,
    })
    return pd.DataFrame(rows).set_index("ts")


def _build_h3_short_session(
    sweep_time="11:25:00",
    confirm_time="11:30:00",
    pdh=_PDH, pdl=_PDL, pdc=_PDC,
    n_filler=30,
    adx=18.0,
):
    """Mirror of _build_l3_long_session for H3-short."""
    levels = _camarilla_levels(pdh, pdl, pdc)
    h3 = levels["H3"]

    end_ts = pd.Timestamp(f"2025-01-08 {confirm_time}")
    sweep_ts = pd.Timestamp(f"2025-01-08 {sweep_time}")
    range_idx = pd.date_range(
        sweep_ts - pd.Timedelta(minutes=5 * n_filler),
        periods=n_filler,
        freq="5min",
    )

    rows = []
    base = (levels["L3"] + h3) / 2.0
    for ts in range_idx:
        rows.append({
            "ts": ts,
            "open": base, "high": base + 0.05, "low": base - 0.05,
            "close": base, "volume": 10000, "adx": adx,
        })
    sweep_high = h3 * (1.0 + 0.0015)             # +0.15% penetration
    sweep_close = h3 * (1.0 - 0.0010)            # -0.10% reclaim
    sweep_low = sweep_close - 0.05
    sweep_open = h3 + 0.10
    rows.append({
        "ts": sweep_ts,
        "open": sweep_open, "high": sweep_high, "low": sweep_low,
        "close": sweep_close, "volume": 12000, "adx": adx,
    })
    confirm_close = sweep_close - 0.10
    rows.append({
        "ts": end_ts,
        "open": sweep_close,
        "high": sweep_close + 0.02,
        "low": confirm_close - 0.02,
        "close": confirm_close,
        "volume": 11000, "adx": adx,
    })
    return pd.DataFrame(rows).set_index("ts")


def _ctx(
    df,
    symbol="NSE:HDFCBANK",
    pdh=_PDH, pdl=_PDL, pdc=_PDC,
    cap_segment="mid_cap",
    atr=1.0,
    adx_override=None,
):
    last_ts = df.index[-1]
    indicators = {"atr": atr}
    if adx_override is not None:
        indicators["adx"] = adx_override
    elif "adx" in df.columns and pd.notna(df["adx"].iloc[-1]):
        indicators["adx"] = float(df["adx"].iloc[-1])
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment=cap_segment,
        regime="chop",
        pdh=pdh, pdl=pdl, pdc=pdc,
        indicators=indicators,
    )


# =============================================================================
# Camarilla formula correctness — pinned
# =============================================================================

def test_camarilla_levels_match_canonical_formula():
    """Hand-computed values for PDH=110, PDL=100, PDC=105 (Range=10):
        P  = (110+100+105)/3 = 105
        H3 = 105 + 0.275*10 = 107.75
        L3 = 105 - 0.275*10 = 102.25
        H4 = 105 + 0.55 *10 = 110.5
        L4 = 105 - 0.55 *10 =  99.5
    """
    levels = _camarilla_levels(110.0, 100.0, 105.0)
    assert levels["P"] == pytest.approx(105.0, abs=1e-9)
    assert levels["H3"] == pytest.approx(107.75, abs=1e-9)
    assert levels["L3"] == pytest.approx(102.25, abs=1e-9)
    assert levels["H4"] == pytest.approx(110.5, abs=1e-9)
    assert levels["L4"] == pytest.approx(99.5, abs=1e-9)


# =============================================================================
# Happy-path tests
# =============================================================================

def test_fires_long_on_canonical_l3_sweep_reclaim_confirm():
    """Sweep bar latches; confirm bar fires LONG with all Camarilla levels."""
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_l3_long_session()
    df_at_sweep = df.iloc[:-1]
    res_sweep = det.detect(_ctx(df_at_sweep))
    assert res_sweep.structure_detected is False, (
        f"sweep bar should latch, not fire: {res_sweep.rejection_reason}"
    )
    res = det.detect(_ctx(df))
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"
    assert res.events[0].levels["L3"] == pytest.approx(102.25, abs=1e-9)
    assert res.events[0].levels["L4"] == pytest.approx(99.5, abs=1e-9)
    assert res.events[0].levels["P"] == pytest.approx(105.0, abs=1e-9)


def test_fires_short_on_canonical_h3_sweep_reclaim_confirm():
    """H3 short mirror."""
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_h3_short_session()
    det.detect(_ctx(df.iloc[:-1]))
    res = det.detect(_ctx(df))
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"
    assert res.events[0].levels["H3"] == pytest.approx(107.75, abs=1e-9)
    assert res.events[0].levels["H4"] == pytest.approx(110.5, abs=1e-9)


# =============================================================================
# ADX regime gate
# =============================================================================

def test_does_not_fire_when_adx_above_max_for_revert():
    """ADX=30 (above max=25) → reject (trending day, Camarilla mean-revert
    fails when price runs through L3 to L4)."""
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_l3_long_session(adx=30.0)
    det.detect(_ctx(df.iloc[:-1]))
    res = det.detect(_ctx(df))
    assert res.structure_detected is False
    assert "adx" in (res.rejection_reason or "").lower()


# =============================================================================
# Mutual exclusion vs pdh_pdl_sweep_reclaim
# =============================================================================

def test_does_not_fire_when_l3_within_skip_pct_of_pdl():
    """Choose PDH/PDL/PDC such that L3 is within 0.3% of PDL → cede to
    pdh_pdl_sweep_reclaim. Long side should be dropped from allowed_sides_today.

    Want: |L3 - PDL| / PDL <= 0.003.
    L3 = PDC - 0.275 * (PDH - PDL). For L3 = PDL exactly:
      PDC - 0.275*(PDH-PDL) = PDL  →  PDC = PDL + 0.275*(PDH-PDL).
    Pick PDH=102, PDL=100, then PDC = 100 + 0.55 = 100.55. L3 = 100.55-0.55 = 100.0 = PDL.
    """
    det = CamarillaL3ReversalStructure(_cfg())
    pdh, pdl, pdc = 102.0, 100.0, 100.55
    df = _build_l3_long_session(pdh=pdh, pdl=pdl, pdc=pdc)
    det.detect(_ctx(df.iloc[:-1], pdh=pdh, pdl=pdl, pdc=pdc))
    res = det.detect(_ctx(df, pdh=pdh, pdl=pdl, pdc=pdc))
    # Long side ceded → no fire from L3
    assert res.structure_detected is False, (
        f"long should be ceded to pdh_pdl_sweep_reclaim: {res.rejection_reason}"
    )


# =============================================================================
# Sub-threshold penetration
# =============================================================================

def test_does_not_fire_when_penetration_below_threshold():
    """Sweep bar's low only 0.05% below L3 (< sweep_pen=0.10%) → no latch."""
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_l3_long_session()
    sweep_idx = df.index[-2]
    levels = _camarilla_levels(_PDH, _PDL, _PDC)
    df.loc[sweep_idx, "low"] = levels["L3"] * (1.0 - 0.0005)   # only 0.05% below
    det.detect(_ctx(df.iloc[:-1]))
    res = det.detect(_ctx(df))
    assert res.structure_detected is False


# =============================================================================
# No reclaim
# =============================================================================

def test_does_not_fire_when_no_reclaim():
    """Sweep bar's close stays BELOW L3 (no reclaim above level) → no latch."""
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_l3_long_session()
    sweep_idx = df.index[-2]
    levels = _camarilla_levels(_PDH, _PDL, _PDC)
    df.loc[sweep_idx, "close"] = levels["L3"] * (1.0 - 0.0010)   # below L3
    det.detect(_ctx(df.iloc[:-1]))
    res = det.detect(_ctx(df))
    assert res.structure_detected is False


# =============================================================================
# Confirm bar lacks continuation
# =============================================================================

def test_does_not_fire_when_confirm_bar_breaks_back_below_l3():
    """Confirm bar close <= L3*(1+rec_buf) → fail continuation, drop pending."""
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_l3_long_session()
    confirm_idx = df.index[-1]
    levels = _camarilla_levels(_PDH, _PDL, _PDC)
    df.loc[confirm_idx, "close"] = levels["L3"] * 0.999   # below L3
    det.detect(_ctx(df.iloc[:-1]))
    res = det.detect(_ctx(df))
    assert res.structure_detected is False


# =============================================================================
# Outside active window
# =============================================================================

def test_does_not_fire_outside_active_window():
    """Sweep+confirm at 09:25/09:30 — before the 10:00 active_window_start."""
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_l3_long_session(sweep_time="09:25:00", confirm_time="09:30:00")
    res = det.detect(_ctx(df))
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


# =============================================================================
# Universe + cap_segment guards
# =============================================================================

def test_does_not_fire_when_symbol_outside_universe():
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_l3_long_session()
    ctx = _ctx(df, symbol="NSE:NONEXISTENT")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_cap_segment_micro():
    """allowed_cap_segments excludes micro_cap by default."""
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_l3_long_session()
    ctx = _ctx(df, cap_segment="micro_cap")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "cap_segment" in (res.rejection_reason or "").lower()


# =============================================================================
# First-trigger latch
# =============================================================================

def test_first_trigger_latch_prevents_double_fire():
    """After L3-long fires once, a second canonical sequence later in the
    same session is no-op."""
    det = CamarillaL3ReversalStructure(_cfg())
    df1 = _build_l3_long_session(sweep_time="11:25:00", confirm_time="11:30:00")
    det.detect(_ctx(df1.iloc[:-1]))
    res1 = det.detect(_ctx(df1))
    assert res1.structure_detected is True

    df2 = _build_l3_long_session(sweep_time="13:25:00", confirm_time="13:30:00")
    det.detect(_ctx(df2.iloc[:-1]))
    res2 = det.detect(_ctx(df2))
    assert res2.structure_detected is False, "latch should block same-day double-fire"


# =============================================================================
# Plan emission
# =============================================================================

def test_plan_emits_hard_sl_l4_t1_pivot_t2_h3_for_long():
    """LONG plan: hard_sl ≈ L4 (within wick buffer), T1 == P, T2 == H3,
    T2 > T1 > entry."""
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_l3_long_session()
    det.detect(_ctx(df.iloc[:-1]))
    plan = det.plan_long_strategy(_ctx(df))
    assert plan is not None
    assert plan.side == "long"
    levels = _camarilla_levels(_PDH, _PDL, _PDC)
    # hard_sl should be at or just below L4 (wick_buf_pct adjustment)
    assert plan.risk_params.hard_sl <= levels["L4"]
    # Tolerate the wick-buffer subtraction (close * 0.05% ≈ 0.05 at close ≈ 100)
    assert plan.risk_params.hard_sl >= levels["L4"] - plan.entry_price * 0.001
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    # T1 = pivot P
    assert targets[0]["level"] == pytest.approx(levels["P"], abs=1e-6)
    # T2 = H3
    assert targets[1]["level"] == pytest.approx(levels["H3"], abs=1e-6)
    # Direction: T1 > entry, T2 > T1
    assert targets[0]["level"] > plan.entry_price
    assert targets[1]["level"] > targets[0]["level"]


def test_plan_emits_hard_sl_h4_t1_pivot_t2_l3_for_short():
    """SHORT plan: hard_sl ≈ H4, T1 == P, T2 == L3."""
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_h3_short_session()
    det.detect(_ctx(df.iloc[:-1]))
    plan = det.plan_short_strategy(_ctx(df))
    assert plan is not None
    assert plan.side == "short"
    levels = _camarilla_levels(_PDH, _PDL, _PDC)
    assert plan.risk_params.hard_sl >= levels["H4"]
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["level"] == pytest.approx(levels["P"], abs=1e-6)
    assert targets[1]["level"] == pytest.approx(levels["L3"], abs=1e-6)
    assert targets[0]["level"] < plan.entry_price
    assert targets[1]["level"] < targets[0]["level"]


# =============================================================================
# Wide-open mode bypass
# =============================================================================

def test_wide_open_bypasses_adx_gate_and_mutual_exclusion(monkeypatch):
    """Under wide_open: high ADX (would normally block) AND L3 ≈ PDL (would
    normally cede) BOTH bypassed — sweep+reclaim still fires."""
    import structures.camarilla_l3_reversal_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    det = CamarillaL3ReversalStructure(_cfg())
    # Choose PDH/PDL/PDC where L3 is within 0.3% of PDL (would cede)
    pdh, pdl, pdc = 102.0, 100.0, 100.55
    df = _build_l3_long_session(pdh=pdh, pdl=pdl, pdc=pdc, adx=40.0)  # high ADX too
    det.detect(_ctx(df.iloc[:-1], pdh=pdh, pdl=pdl, pdc=pdc, adx_override=40.0))
    res = det.detect(_ctx(df, pdh=pdh, pdl=pdl, pdc=pdc, adx_override=40.0))
    assert res.structure_detected is True, (
        f"wide_open should bypass ADX + mutual-exclusion: {res.rejection_reason}"
    )


def test_wide_open_preserves_trigger_geometry(monkeypatch):
    """Sub-threshold penetration must not fire even under wide_open."""
    import structures.camarilla_l3_reversal_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    det = CamarillaL3ReversalStructure(_cfg())
    df = _build_l3_long_session()
    sweep_idx = df.index[-2]
    levels = _camarilla_levels(_PDH, _PDL, _PDC)
    df.loc[sweep_idx, "low"] = levels["L3"] * (1.0 - 0.0005)
    det.detect(_ctx(df.iloc[:-1]))
    res = det.detect(_ctx(df))
    assert res.structure_detected is False
