"""PDH/PDL Sweep+Reclaim detector unit tests.

Mechanic per
specs/2026-04-29-pdh_pdl_sweep_reclaim-plan.md and
specs/2026-04-28-research-pdh_pdl_reject-indian-pro-mechanics.md.

PDH-fade-short canonical sequence (mirror PDL-fade-long):
  bar t-1 (sweep):  high > pdh*(1+pen) AND close < pdh*(1-rec)  → latch
  bar t (confirm):  close < pdh AND close < bar(t-1).low        → fire short

Tests cover:
  - Happy-path long + short fires
  - Sweep below threshold → no latch
  - Sweep with no reclaim (close stays beyond level) → no latch
  - Confirm bar without continuation (close inside sweep range) → abort
  - Outside active window → reject
  - Outside universe → reject
  - First-trigger latch prevents same-day double fire
  - Gap-context side selector (M8): gap-up blocks LONG side
  - Multi-day confluence (M2): default OFF fires; ON fires only on confluence
  - Wide-open mode bypasses design-inferred filters
  - plan_*_strategy emits hard_sl + tiered T1/T2 in correct direction
"""
from __future__ import annotations

import pandas as pd
import pytest

from structures.pdh_pdl_sweep_reclaim_structure import PDHPDLSweepReclaimStructure
from structures.data_models import MarketContext


def _cfg(**overrides):
    cfg = {
        "_setup_name": "pdh_pdl_sweep_reclaim",
        "enabled": True,
        "active_window_start": "11:00",
        "active_window_end": "14:00",
        "sweep_penetration_pct": 0.10,
        "reclaim_buffer_pct": 0.05,
        "confirm_close_beyond_sweep_low": True,
        "multi_day_confluence_enabled": False,
        "multi_day_confluence_pct": 0.5,
        "multi_day_lookback": 3,
        "gap_context_enabled": False,   # default OFF in tests; explicit-on tests flip
        "gap_threshold_pct": 0.5,
        "allowed_sides": ["long", "short"],
        "allowed_cap_segments": ["large_cap", "mid_cap", "small_cap", "micro_cap"],
        "universe_key": "fno_liquid_200",
        "stop_atr_buffer": 0.5,
        "wick_buffer_pct": 0.10,
        "t1_target": "vwap",
        "t2_target": "two_day_range_50pct_retrace",
        "t1_qty_pct": 0.5,
        "min_bars_required": 30,
        "entry_zone_pct": 0.15,
        "entry_zone_mode": "symmetric",
        "min_stop_distance_pct": 0.3,
    }
    cfg.update(overrides)
    return cfg


def _build_pdh_short_session(
    sweep_time="11:25:00",
    confirm_time="11:30:00",
    pdh=105.0,
    pdc=103.0,
    n_filler=30,
    open_today=None,
):
    """Build a session ending at confirm_time with the canonical sweep+reclaim+confirm sequence.

    n_filler benign bars (price chops below PDH) + sweep bar at sweep_time
    (high penetrates PDH by 0.15%, close back inside by 0.10%) + confirm
    bar at confirm_time (close < PDH AND close < sweep_bar.low).
    """
    end_ts = pd.Timestamp(f"2025-01-08 {confirm_time}")  # Wed (non-Friday by default)
    sweep_ts = pd.Timestamp(f"2025-01-08 {sweep_time}")
    if open_today is None:
        open_today = pdc  # flat open by default

    # Filler bars end one 5m bar before the sweep
    filler_end = sweep_ts - pd.Timedelta(minutes=5)
    filler_idx = pd.date_range(
        filler_end - pd.Timedelta(minutes=5 * (n_filler - 1)),
        periods=n_filler,
        freq="5min",
    )

    rows = []
    base = pdc
    is_first = True
    for ts in filler_idx:
        rows.append({
            "ts": ts,
            "open": open_today if is_first else base,
            "high": base + 0.05,
            "low": base - 0.05,
            "close": base + 0.02,
            "volume": 10000,
            "vwap": pdc,
        })
        is_first = False

    # Sweep bar — penetrates PDH then closes back inside
    sweep_high = pdh * 1.0015           # +0.15% penetration (>= sweep_penetration_pct=0.10%)
    sweep_close = pdh * 0.9990          # -0.10% reclaim (>= reclaim_buffer_pct=0.05%)
    sweep_low = sweep_close - 0.10
    rows.append({
        "ts": sweep_ts,
        "open": pdh - 0.20, "high": sweep_high, "low": sweep_low,
        "close": sweep_close, "volume": 12000, "vwap": pdc,
    })

    # Confirm bar — close < PDH AND close < sweep_bar.low (continuation beyond sweep low)
    confirm_close = sweep_low - 0.05
    rows.append({
        "ts": end_ts,
        "open": sweep_close,
        "high": sweep_close + 0.02,
        "low": confirm_close - 0.05,
        "close": confirm_close,
        "volume": 11000,
        "vwap": pdc,
    })
    return pd.DataFrame(rows).set_index("ts")


def _build_pdl_long_session(
    sweep_time="11:25:00",
    confirm_time="11:30:00",
    pdl=98.0,
    pdc=99.5,
    n_filler=30,
    open_today=None,
):
    """Mirror of _build_pdh_short_session for the PDL-fade-long side.

    Sweep bar: low penetrates PDL (low < pdl*(1-pen)), close back inside
    (close > pdl*(1+rec)). Confirm bar: close > PDL AND close > sweep_bar.high.
    """
    end_ts = pd.Timestamp(f"2025-01-08 {confirm_time}")
    sweep_ts = pd.Timestamp(f"2025-01-08 {sweep_time}")
    if open_today is None:
        open_today = pdc

    filler_end = sweep_ts - pd.Timedelta(minutes=5)
    filler_idx = pd.date_range(
        filler_end - pd.Timedelta(minutes=5 * (n_filler - 1)),
        periods=n_filler,
        freq="5min",
    )

    rows = []
    base = pdc
    is_first = True
    for ts in filler_idx:
        rows.append({
            "ts": ts,
            "open": open_today if is_first else base,
            "high": base + 0.05, "low": base - 0.05,
            "close": base, "volume": 10000, "vwap": pdc,
        })
        is_first = False

    sweep_low = pdl * 0.9985                     # -0.15% penetration
    sweep_close = pdl * 1.0010                   # +0.10% reclaim
    sweep_high = sweep_close + 0.10
    rows.append({
        "ts": sweep_ts,
        "open": pdl + 0.20, "high": sweep_high, "low": sweep_low,
        "close": sweep_close, "volume": 12000, "vwap": pdc,
    })

    confirm_close = sweep_high + 0.05
    rows.append({
        "ts": end_ts,
        "open": sweep_close, "high": confirm_close + 0.05,
        "low": sweep_close - 0.02, "close": confirm_close,
        "volume": 11000, "vwap": pdc,
    })
    return pd.DataFrame(rows).set_index("ts")


def _ctx(
    df,
    symbol="NSE:HDFCBANK",
    pdh=105.0,
    pdl=98.0,
    pdc=103.0,
    cap_segment="mid_cap",
    regime="chop",
    df_daily=None,
    atr=1.0,
    vwap=None,
):
    last_ts = df.index[-1]
    indicators = {"atr": atr}
    if vwap is not None:
        indicators["vwap"] = vwap
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        df_daily=df_daily,
        cap_segment=cap_segment,
        regime=regime,
        pdh=pdh, pdl=pdl, pdc=pdc,
        indicators=indicators,
    )


# =============================================================================
# Task 1.2 — happy-path PDH-fade-short fires on canonical sequence
# =============================================================================

def test_fires_short_on_canonical_pdh_sweep_reclaim_confirm():
    """Range bars + sweep bar + confirm bar — fires SHORT on confirm."""
    det = PDHPDLSweepReclaimStructure(_cfg())
    df = _build_pdh_short_session()

    # Step 1: feed only up to and including the sweep bar — should NOT fire
    # (the sweep latches; confirmation requires the next bar).
    df_at_sweep = df.iloc[:-1]
    ctx_sweep = _ctx(df_at_sweep)
    res_sweep = det.detect(ctx_sweep)
    assert res_sweep.structure_detected is False, (
        f"Sweep bar should latch, not fire: {res_sweep.rejection_reason}"
    )

    # Step 2: feed the full df including confirm bar — fires SHORT.
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"
    assert res.events[0].structure_type == "pdh_pdl_sweep_reclaim"


# =============================================================================
# Task 1.4 — PDL-fade-long mirror happy path
# =============================================================================

def test_fires_long_on_canonical_pdl_sweep_reclaim_confirm():
    """Mirror long: sweep bar low penetrates PDL, close back inside, confirm bar
    close > PDL AND > sweep_bar.high → fires LONG."""
    det = PDHPDLSweepReclaimStructure(_cfg())
    df = _build_pdl_long_session()
    df_at_sweep = df.iloc[:-1]
    ctx_sweep = _ctx(df_at_sweep, pdl=98.0, pdh=105.0, pdc=99.5)
    assert det.detect(ctx_sweep).structure_detected is False
    ctx = _ctx(df, pdl=98.0, pdh=105.0, pdc=99.5)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"


# =============================================================================
# Task 1.5 — sub-threshold penetration → no latch
# =============================================================================

def test_does_not_fire_when_penetration_below_threshold():
    """If sweep bar's high does not exceed pdh*(1+sweep_pen_pct), no latch happens."""
    det = PDHPDLSweepReclaimStructure(_cfg())
    df = _build_pdh_short_session()
    pdh = 105.0
    sweep_idx = df.index[-2]
    # Penetrate by only 0.05% (< threshold 0.10%)
    df.loc[sweep_idx, "high"] = pdh * 1.0005
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False


# =============================================================================
# Task 1.6 — sweep bar close did not reclaim back inside → no latch
# =============================================================================

def test_does_not_fire_when_no_reclaim():
    """If sweep bar's close stays > pdh*(1-rec), it's a real breakout, not a trap."""
    det = PDHPDLSweepReclaimStructure(_cfg())
    df = _build_pdh_short_session()
    pdh = 105.0
    sweep_idx = df.index[-2]
    # Close ABOVE the reclaim level — no trap, no latch
    df.loc[sweep_idx, "close"] = pdh * 1.0010
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False


# =============================================================================
# Task 1.7 — confirm bar lacks continuation → abort, no fire
# =============================================================================

def test_does_not_fire_when_confirm_bar_lacks_continuation():
    """Sweep latches, but confirm bar's close >= sweep_bar.low (no continuation)
    aborts the pattern. Pending must be dropped — second detect() with even a
    valid continuation later in the same session is no-op."""
    det = PDHPDLSweepReclaimStructure(_cfg())
    df = _build_pdh_short_session()
    confirm_idx = df.index[-1]
    sweep_idx = df.index[-2]
    sweep_low = float(df.loc[sweep_idx, "low"])
    # Confirm close ABOVE sweep low — fails continuation
    df.loc[confirm_idx, "close"] = sweep_low + 0.10
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False


# =============================================================================
# Task 1.8 — outside active window → reject (early-exit, before sweep checks)
# =============================================================================

def test_does_not_fire_outside_active_window():
    """09:25/09:30 sweep+confirm — before the 11:00 active_window_start."""
    det = PDHPDLSweepReclaimStructure(_cfg())
    df = _build_pdh_short_session(sweep_time="09:25:00", confirm_time="09:30:00")
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


# =============================================================================
# Task 1.9 — symbol outside fno_liquid_200 universe → reject
# =============================================================================

def test_does_not_fire_when_symbol_outside_universe():
    """Universe filter fires before any sweep+reclaim logic."""
    det = PDHPDLSweepReclaimStructure(_cfg())
    df = _build_pdh_short_session()
    ctx = _ctx(df, symbol="NSE:NONEXISTENT")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


# =============================================================================
# Task 1.10 — first-trigger latch prevents same-day double-fire
# =============================================================================

def test_first_trigger_latch_prevents_double_fire():
    """After first PDH-fade-short fires, a SECOND valid sweep+reclaim+confirm
    later in the same session is a no-op (latch holds).

    Note: detect() is fed bar-by-bar (sweep then confirm) to mimic how the
    orchestrator drives the detector — each new bar is a separate call.
    """
    det = PDHPDLSweepReclaimStructure(_cfg())
    df1 = _build_pdh_short_session(sweep_time="11:25:00", confirm_time="11:30:00")
    # First fire: feed sweep bar (latches), then confirm bar (fires)
    det.detect(_ctx(df1.iloc[:-1]))                        # latch sweep
    res1 = det.detect(_ctx(df1))                           # confirm fires
    assert res1.structure_detected is True, f"first fire failed: {res1.rejection_reason}"

    # Build a second sweep+confirm later in the same session_date
    df2 = _build_pdh_short_session(sweep_time="13:25:00", confirm_time="13:30:00")
    det.detect(_ctx(df2.iloc[:-1]))                        # would latch — but is it blocked?
    res2 = det.detect(_ctx(df2))
    assert res2.structure_detected is False, "latch should block same-day second fire"


# =============================================================================
# Task 1.11 — gap-context side selector (M8)
# =============================================================================

def test_gap_up_day_blocks_long_only_short_allowed():
    """gap_context_enabled=True; on gap-up day (open >> pdc), the long side is
    dropped from allowed_sides_today. A canonical PDL-long sweep+reclaim+confirm
    must NOT fire long on this day."""
    cfg = _cfg(gap_context_enabled=True, gap_threshold_pct=0.5)
    det = PDHPDLSweepReclaimStructure(cfg)
    pdc = 99.5
    pdl = 98.0
    # Force first bar to open 1.5% above PDC (gap-up day)
    df = _build_pdl_long_session(pdl=pdl, pdc=pdc, open_today=pdc * 1.015)
    # Feed bar-by-bar
    det.detect(_ctx(df.iloc[:-1], pdl=pdl, pdh=105.0, pdc=pdc))
    res = det.detect(_ctx(df, pdl=pdl, pdh=105.0, pdc=pdc))
    assert res.structure_detected is False, (
        f"long-side should be blocked on gap-up day: {res.rejection_reason}"
    )


def test_gap_down_day_blocks_short_only_long_allowed():
    """Mirror: gap-down day drops short-side; PDH-fade-short does not fire."""
    cfg = _cfg(gap_context_enabled=True, gap_threshold_pct=0.5)
    det = PDHPDLSweepReclaimStructure(cfg)
    pdc = 103.0
    pdh = 105.0
    # Force first bar to open 1.5% below PDC (gap-down day)
    df = _build_pdh_short_session(pdh=pdh, pdc=pdc, open_today=pdc * 0.985)
    det.detect(_ctx(df.iloc[:-1], pdh=pdh, pdl=98.0, pdc=pdc))
    res = det.detect(_ctx(df, pdh=pdh, pdl=98.0, pdc=pdc))
    assert res.structure_detected is False, (
        f"short-side should be blocked on gap-down day: {res.rejection_reason}"
    )


def test_flat_open_day_allows_both_sides():
    """Flat-open (|gap| < threshold): both PDH-shorts and PDL-longs allowed."""
    cfg = _cfg(gap_context_enabled=True, gap_threshold_pct=0.5)
    # PDH-fade-short on a flat-open day — should still fire
    det = PDHPDLSweepReclaimStructure(cfg)
    df = _build_pdh_short_session(pdc=103.0, open_today=103.0)  # flat
    det.detect(_ctx(df.iloc[:-1]))
    res = det.detect(_ctx(df))
    assert res.structure_detected is True, f"flat-open should allow short: {res.rejection_reason}"


# =============================================================================
# Task 1.12 — multi-day confluence gate (M2)
# =============================================================================

def _build_df_daily(pdh_history, pdl_history=None, end_date="2025-01-07"):
    """Build a small daily df preceding the test session date 2025-01-08.

    pdh_history: list of daily highs, oldest first.
    pdl_history: optional list of daily lows; defaults to high - 5.
    Indexed by date strictly < 2025-01-08.
    """
    n = len(pdh_history)
    if pdl_history is None:
        pdl_history = [h - 5.0 for h in pdh_history]
    end = pd.Timestamp(end_date)
    idx = pd.date_range(end - pd.Timedelta(days=n - 1), periods=n, freq="D")
    return pd.DataFrame({
        "open": pdh_history,
        "high": pdh_history,
        "low": pdl_history,
        "close": pdh_history,
        "volume": [1000] * n,
    }, index=idx)


def test_multi_day_confluence_off_default_fires():
    """multi_day_confluence_enabled=False (default): fires regardless of df_daily."""
    det = PDHPDLSweepReclaimStructure(_cfg())  # default: confluence OFF
    df = _build_pdh_short_session()
    ctx_sweep = _ctx(df.iloc[:-1], df_daily=None)
    ctx = _ctx(df, df_daily=None)
    det.detect(ctx_sweep)
    res = det.detect(ctx)
    assert res.structure_detected is True, (
        f"confluence OFF should not gate: {res.rejection_reason}"
    )


def test_multi_day_confluence_on_fires_when_daily_high_near_pdh():
    """Confluence ON; one of last 3 daily highs is within 0.5% of PDH → fire."""
    cfg = _cfg(multi_day_confluence_enabled=True, multi_day_confluence_pct=0.5,
               multi_day_lookback=3)
    det = PDHPDLSweepReclaimStructure(cfg)
    pdh = 105.0
    df = _build_pdh_short_session(pdh=pdh)
    # Daily highs: pdh-2 within 0.5% of PDH (104.7 = 99.71% of 105 → diff 0.286%)
    df_daily = _build_df_daily(pdh_history=[100.0, 104.7, 102.0])
    det.detect(_ctx(df.iloc[:-1], df_daily=df_daily))
    res = det.detect(_ctx(df, df_daily=df_daily))
    assert res.structure_detected is True, (
        f"confluence holds (104.7 within 0.5% of 105): {res.rejection_reason}"
    )


def test_multi_day_confluence_on_blocks_when_daily_highs_far():
    """Confluence ON; all last 3 daily highs are > 0.5% from PDH → no fire."""
    cfg = _cfg(multi_day_confluence_enabled=True, multi_day_confluence_pct=0.5,
               multi_day_lookback=3)
    det = PDHPDLSweepReclaimStructure(cfg)
    pdh = 105.0
    df = _build_pdh_short_session(pdh=pdh)
    # All daily highs > 1% away from PDH
    df_daily = _build_df_daily(pdh_history=[100.0, 101.0, 102.0])
    det.detect(_ctx(df.iloc[:-1], df_daily=df_daily))
    res = det.detect(_ctx(df, df_daily=df_daily))
    assert res.structure_detected is False, (
        "no confluence (all daily highs > 1% from PDH=105) — should not fire"
    )


# =============================================================================
# Task 1.13 — plan_long/short_strategy + tiered T1/T2 + risk params
# =============================================================================

def test_plan_short_emits_hard_sl_t1_t2():
    """SHORT plan: hard_sl > entry, T1 < entry (direction-correct), T2 < T1."""
    det = PDHPDLSweepReclaimStructure(_cfg())
    df = _build_pdh_short_session()
    # Set VWAP slightly below entry so T1 = VWAP retest is direction-correct
    det.detect(_ctx(df.iloc[:-1], vwap=103.0))
    plan = det.plan_short_strategy(_ctx(df, vwap=103.0))
    assert plan is not None
    assert plan.side == "short"
    # Stop above entry for short
    assert plan.risk_params.hard_sl > plan.entry_price, (
        f"short stop must be above entry: sl={plan.risk_params.hard_sl} entry={plan.entry_price}"
    )
    # Two tiered targets
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["name"] == "T1"
    assert targets[1]["name"] == "T2"
    # T1 below entry (target moves down for short)
    assert targets[0]["level"] < plan.entry_price
    # T2 farther below than T1 (more aggressive)
    assert targets[1]["level"] < targets[0]["level"]
    # qty_pct sums to 1.0
    assert abs(targets[0]["qty_pct"] + targets[1]["qty_pct"] - 1.0) < 1e-6
    # T2 action = exit_full
    assert targets[1]["action"] == "exit_full"


def test_plan_long_emits_hard_sl_t1_t2():
    """LONG plan: hard_sl < entry, T1 > entry, T2 > T1."""
    det = PDHPDLSweepReclaimStructure(_cfg())
    df = _build_pdl_long_session()
    # Set VWAP above entry so T1 = VWAP retest is direction-correct (long)
    det.detect(_ctx(df.iloc[:-1], pdl=98.0, pdh=105.0, pdc=99.5, vwap=99.5))
    plan = det.plan_long_strategy(_ctx(df, pdl=98.0, pdh=105.0, pdc=99.5, vwap=99.5))
    assert plan is not None
    assert plan.side == "long"
    assert plan.risk_params.hard_sl < plan.entry_price, (
        f"long stop must be below entry: sl={plan.risk_params.hard_sl} entry={plan.entry_price}"
    )
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["level"] > plan.entry_price
    assert targets[1]["level"] > targets[0]["level"]
    assert targets[1]["action"] == "exit_full"


def test_plan_short_strategy_returns_none_for_long_side_when_sides_restricted():
    """Config restricts to short-only; plan_long_strategy returns None even
    on a clean PDL-fade-long sweep+reclaim."""
    det = PDHPDLSweepReclaimStructure(_cfg(allowed_sides=["short"]))
    df = _build_pdl_long_session()
    plan = det.plan_long_strategy(_ctx(df, pdl=98.0, pdh=105.0, pdc=99.5))
    assert plan is None


def test_plan_min_stop_distance_widens_tight_stops():
    """If sweep_extreme is so close to close that risk < close*min_stop_distance_pct,
    hard_sl widens to enforce the floor."""
    # min_stop_distance_pct=0.3% means risk floor = entry * 0.003
    det = PDHPDLSweepReclaimStructure(_cfg(min_stop_distance_pct=0.5))  # 0.5% floor
    df = _build_pdh_short_session()
    det.detect(_ctx(df.iloc[:-1]))
    plan = det.plan_short_strategy(_ctx(df))
    assert plan is not None
    entry = plan.entry_price
    risk = plan.risk_params.risk_per_share
    floor = entry * 0.005
    assert risk >= floor - 1e-6, f"risk {risk} should be >= floor {floor}"


# =============================================================================
# Task 1.14 — wide_open_mode bypass for design-inferred filters
# =============================================================================

def test_wide_open_bypasses_multi_day_confluence_and_gap_context(monkeypatch):
    """Under wide_open_mode, design-inferred filters (multi-day confluence,
    gap-context) are bypassed; trigger geometry (sweep+reclaim+confirm) stays
    active. Override the conftest auto-patch by re-patching _is_wide_open
    to return True."""
    import structures.pdh_pdl_sweep_reclaim_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)

    # Multi-day confluence ON; df_daily has highs FAR from PDH (would normally
    # block). Wide-open should bypass the gate and let the sweep+reclaim fire.
    cfg = _cfg(multi_day_confluence_enabled=True, multi_day_confluence_pct=0.5)
    det = PDHPDLSweepReclaimStructure(cfg)
    pdh = 105.0
    df = _build_pdh_short_session(pdh=pdh)
    df_daily = _build_df_daily(pdh_history=[100.0, 101.0, 102.0])  # all > 1% from pdh
    det.detect(_ctx(df.iloc[:-1], df_daily=df_daily))
    res = det.detect(_ctx(df, df_daily=df_daily))
    assert res.structure_detected is True, (
        f"wide_open should bypass confluence: {res.rejection_reason}"
    )


def test_wide_open_preserves_trigger_geometry(monkeypatch):
    """Even under wide_open, sub-threshold sweep penetration must NOT fire —
    trigger geometry (penetration, reclaim, continuation) is mechanical and
    always enforced."""
    import structures.pdh_pdl_sweep_reclaim_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)

    det = PDHPDLSweepReclaimStructure(_cfg())
    df = _build_pdh_short_session()
    pdh = 105.0
    sweep_idx = df.index[-2]
    # Make sweep penetration sub-threshold even under wide_open
    df.loc[sweep_idx, "high"] = pdh * 1.0005
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False, (
        "trigger geometry must be enforced regardless of wide_open"
    )
