"""ORB-15 detector unit tests — sweep+reclaim redesign.

Redesigned per specs/2026-04-29-orb_15-redesign-plan.md (Phase 1 TDD).

Tests cover the new mechanic:
  - Sweep+reclaim trigger (penetrate range + close back inside, then reclaim
    on next bar) — replaces broken first-close-outside trigger.
  - NR7 pre-condition: today's OR range must be <= multiplier × min(prior 7).
  - Relative-volume "in play" filter at OR-end (>= 2.0× 14-day baseline).
  - First-trigger latch per (symbol, side, session_date).
  - Friday exclusion (DOW gate).
  - Regime + side gates (explicit-allowlist semantics — empty list = none).
  - Plan emits hard_sl + tiered T1/T2 (existing exit mechanic preserved).

Negative regression: the old "first close outside range" pattern (without a
preceding sweep candle) MUST NOT fire — guards against trigger regression.
"""
import pandas as pd
import pytest

from structures.orb_15_structure import ORB15Structure
from structures.data_models import MarketContext


def _cfg(**overrides):
    """Default test config covering every key the constructor reads.

    Overrides let individual tests flip specific gates (e.g., empty
    allowed_regimes, alternate exclude_dow) without touching others.
    """
    cfg = {
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
        # Redesign keys (Phase 0):
        "sweep_reclaim_lookback_bars": 6,
        "nr7_lookback_days": 7,
        "nr7_multiplier": 1.0,
        "min_rvol_at_or_close": 2.0,
        "rvol_baseline_lookback_days": 14,
        "exclude_dow": [4],
        "allowed_regimes": ["trend_up", "trend_down", "chop", "squeeze"],
        "allowed_sides": ["long", "short"],
    }
    cfg.update(overrides)
    return cfg


def _build_sweep_reclaim_long_df(
    sweep_time="09:35:00",
    reclaim_time="09:40:00",
    range_high=102.0,
    range_low=100.0,
    sweep_high=102.5,
    sweep_close=101.5,
    reclaim_close=102.6,
    or_volume=25000,
    body_volume=10000,
):
    """Build LONG sweep+reclaim: range bars + sweep bar + reclaim trigger bar.

    Range window 09:20-09:30 (2 bars: 09:20, 09:25).
    Sweep bar at sweep_time: high > range_high (penetration), close back inside [range_low, range_high].
    Reclaim trigger bar at reclaim_time: close > range_high (the trigger).

    Cumulative volume in 09:20-09:30 range bars = `or_volume`/2 × 2 = `or_volume`.
    """
    end = pd.Timestamp(f"2025-01-08 {reclaim_time}")
    sweep_ts = pd.Timestamp(f"2025-01-08 {sweep_time}")
    range_idx = pd.date_range("2025-01-08 09:20:00", periods=2, freq="5min")
    # Bars between range-end (09:30) and sweep_time
    pre_sweep_count = max(0, int((sweep_ts - pd.Timestamp("2025-01-08 09:30:00")).total_seconds() / 300))
    pre_sweep_idx = pd.date_range("2025-01-08 09:30:00", periods=pre_sweep_count, freq="5min") if pre_sweep_count > 0 else pd.DatetimeIndex([])
    # Bars between sweep_time and reclaim_time
    inter_count = max(0, int((end - sweep_ts).total_seconds() / 300) - 1)
    inter_idx = pd.date_range(sweep_ts + pd.Timedelta(minutes=5), periods=inter_count, freq="5min") if inter_count > 0 else pd.DatetimeIndex([])

    rows = []
    mid = (range_high + range_low) / 2
    # Range bars (09:20, 09:25): each contributes or_volume/2
    for ts in range_idx:
        rows.append({"ts": ts, "open": mid, "high": range_high, "low": range_low,
                     "close": mid, "volume": or_volume / len(range_idx)})
    # Pre-sweep bars (between 09:30 and sweep): inside range, low volume
    for ts in pre_sweep_idx:
        rows.append({"ts": ts, "open": mid, "high": mid + 0.1, "low": mid - 0.1,
                     "close": mid, "volume": body_volume})
    # Sweep bar: high penetrates above range_high, close back inside
    rows.append({"ts": sweep_ts, "open": mid, "high": sweep_high, "low": mid - 0.05,
                 "close": sweep_close, "volume": body_volume})
    # Inter bars (between sweep and reclaim): inside range
    for ts in inter_idx:
        rows.append({"ts": ts, "open": sweep_close, "high": sweep_close + 0.1,
                     "low": sweep_close - 0.1, "close": sweep_close, "volume": body_volume})
    # Reclaim trigger bar: close above range_high
    rows.append({"ts": end, "open": sweep_close, "high": reclaim_close + 0.05,
                 "low": sweep_close - 0.05, "close": reclaim_close, "volume": body_volume})
    return pd.DataFrame(rows).set_index("ts")


def _build_sweep_reclaim_short_df(
    sweep_time="09:35:00",
    reclaim_time="09:40:00",
    range_high=102.0,
    range_low=100.0,
    sweep_low=99.5,
    sweep_close=100.5,
    reclaim_close=99.4,
    or_volume=25000,
    body_volume=10000,
):
    """Build SHORT sweep+reclaim: range bars + sweep bar + reclaim trigger bar.

    Mirror of long: low < range_low (penetration), close back inside, then
    next bar close < range_low.
    """
    end = pd.Timestamp(f"2025-01-08 {reclaim_time}")
    sweep_ts = pd.Timestamp(f"2025-01-08 {sweep_time}")
    range_idx = pd.date_range("2025-01-08 09:20:00", periods=2, freq="5min")
    pre_sweep_count = max(0, int((sweep_ts - pd.Timestamp("2025-01-08 09:30:00")).total_seconds() / 300))
    pre_sweep_idx = pd.date_range("2025-01-08 09:30:00", periods=pre_sweep_count, freq="5min") if pre_sweep_count > 0 else pd.DatetimeIndex([])
    inter_count = max(0, int((end - sweep_ts).total_seconds() / 300) - 1)
    inter_idx = pd.date_range(sweep_ts + pd.Timedelta(minutes=5), periods=inter_count, freq="5min") if inter_count > 0 else pd.DatetimeIndex([])

    rows = []
    mid = (range_high + range_low) / 2
    for ts in range_idx:
        rows.append({"ts": ts, "open": mid, "high": range_high, "low": range_low,
                     "close": mid, "volume": or_volume / len(range_idx)})
    for ts in pre_sweep_idx:
        rows.append({"ts": ts, "open": mid, "high": mid + 0.1, "low": mid - 0.1,
                     "close": mid, "volume": body_volume})
    rows.append({"ts": sweep_ts, "open": mid, "high": mid + 0.05, "low": sweep_low,
                 "close": sweep_close, "volume": body_volume})
    for ts in inter_idx:
        rows.append({"ts": ts, "open": sweep_close, "high": sweep_close + 0.1,
                     "low": sweep_close - 0.1, "close": sweep_close, "volume": body_volume})
    rows.append({"ts": end, "open": sweep_close, "high": sweep_close + 0.05,
                 "low": reclaim_close - 0.05, "close": reclaim_close, "volume": body_volume})
    return pd.DataFrame(rows).set_index("ts")


def _build_simple_breakout_df(
    breakout_time="09:35:00",
    range_high=102.0,
    range_low=100.0,
    breakout_close=102.5,
    or_volume=25000,
    body_volume=10000,
):
    """Build the OLD-STYLE first-close-outside pattern (no preceding sweep candle).

    Used for negative tests: the new sweep+reclaim mechanic MUST reject this
    pattern. Range bars + a single bar that closes outside the range, with
    NO prior penetration-and-reclaim sequence.
    """
    end = pd.Timestamp(f"2025-01-08 {breakout_time}")
    range_idx = pd.date_range("2025-01-08 09:20:00", periods=2, freq="5min")
    pre_count = max(0, int((end - pd.Timestamp("2025-01-08 09:30:00")).total_seconds() / 300))
    pre_idx = pd.date_range("2025-01-08 09:30:00", periods=pre_count, freq="5min") if pre_count > 0 else pd.DatetimeIndex([])

    rows = []
    mid = (range_high + range_low) / 2
    for ts in range_idx:
        rows.append({"ts": ts, "open": mid, "high": range_high, "low": range_low,
                     "close": mid, "volume": or_volume / len(range_idx)})
    # All pre-bars stay strictly inside the range — no penetration
    for ts in pre_idx:
        rows.append({"ts": ts, "open": mid, "high": mid + 0.1, "low": mid - 0.1,
                     "close": mid, "volume": body_volume})
    # Final bar simply closes outside, no prior sweep
    rows.append({"ts": end, "open": mid, "high": breakout_close + 0.05,
                 "low": mid - 0.05, "close": breakout_close, "volume": body_volume})
    return pd.DataFrame(rows).set_index("ts")


def _ctx(
    df,
    symbol="NSE:RELIANCE",
    cap_segment="large_cap",
    pdc=101.0,
    regime="squeeze",
    prior_or_ranges_7d=None,
    or_window_volume_baseline_14d=10000.0,
    session_date_override=None,
):
    """Build MarketContext with NR7 + rvol baseline pre-populated in indicators.

    Default fixtures pass NR7 (range=2.0, prior_or_ranges_7d ≥ 2.0 each),
    pass rvol (default or_volume=25000 vs baseline 10000 = 2.5× ≥ 2.0×),
    are non-Friday (2025-01-08 = Wednesday), and squeeze regime.
    """
    last_ts = df.index[-1]
    if prior_or_ranges_7d is None:
        # Default: makes today's OR width 2.0 the new minimum (passes NR7
        # because nr7_multiplier=1.0 means ≤ min_prior=2.0 → boundary inclusive).
        prior_or_ranges_7d = [2.5, 3.0, 2.8, 3.2, 2.6, 2.9, 2.7]
    indicators = {
        "atr": 1.0,
        "median_volume_30d": 10000,
        "prior_or_ranges_7d": prior_or_ranges_7d,
        "or_window_volume_baseline_14d": or_window_volume_baseline_14d,
    }
    session_date = session_date_override or last_ts.to_pydatetime().replace(
        hour=0, minute=0, second=0,
    )
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=session_date,
        cap_segment=cap_segment,
        regime=regime,
        pdh=110.0, pdl=98.0, pdc=pdc,
        indicators=indicators,
    )


# =============================================================================
# Sweep+reclaim trigger tests (replaces first-close-outside)
# =============================================================================

def test_fires_long_on_sweep_then_reclaim():
    """Range [100, 102], sweep bar penetrates above and closes back inside,
    next bar closes above range — fires LONG."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df()
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"


def test_fires_short_on_sweep_then_reclaim():
    """Mirror short: penetrate below, close inside, next bar closes below."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_short_df()
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"


def test_does_not_fire_on_simple_close_outside_no_prior_sweep():
    """REGRESSION GUARD: the old 'first close outside' pattern must NOT fire.

    A bar closing above range_high without a preceding sweep candle is now
    rejected — ensures we don't regress to the broken trigger.
    """
    det = ORB15Structure(_cfg())
    df = _build_simple_breakout_df()
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "sweep" in (res.rejection_reason or "").lower() or \
           "reclaim" in (res.rejection_reason or "").lower()


# =============================================================================
# NR7 pre-condition tests
# =============================================================================

def test_does_not_fire_when_or_range_exceeds_nr7_threshold():
    """prior_or_ranges_7d min = 1.0; today's OR width 2.0 — NR7 violated."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df()
    ctx = _ctx(df, prior_or_ranges_7d=[1.0] * 7)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "nr7" in (res.rejection_reason or "").lower()


def test_fires_at_nr7_boundary():
    """Today's OR width = min(prior_7) — boundary inclusive (<=)."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df()
    # min = 2.0 = today's OR width; should pass (boundary inclusive)
    ctx = _ctx(df, prior_or_ranges_7d=[2.0, 2.5, 3.0, 2.2, 2.8, 2.4, 2.6])
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire at boundary: {res.rejection_reason}"


def test_does_not_fire_when_nr7_data_missing():
    """No prior_or_ranges_7d in indicators AND no df_daily — fail-closed."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df()
    ctx = _ctx(df, prior_or_ranges_7d=[])  # empty list = unavailable
    # Manually strip the key from indicators to simulate missing data
    ctx.indicators.pop("prior_or_ranges_7d", None)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "nr7" in (res.rejection_reason or "").lower()


# =============================================================================
# Relative-volume filter tests
# =============================================================================

def test_does_not_fire_when_rvol_below_threshold():
    """OR-window vol = 15000 vs baseline 10000 = 1.5× < 2.0× threshold."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df(or_volume=15000)
    ctx = _ctx(df, or_window_volume_baseline_14d=10000)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "rvol" in (res.rejection_reason or "").lower()


def test_fires_when_rvol_meets_threshold():
    """OR-window vol = 20000 vs baseline 10000 = 2.0× — boundary inclusive."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df(or_volume=20000)
    ctx = _ctx(df, or_window_volume_baseline_14d=10000)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire at rvol=2.0: {res.rejection_reason}"


def test_does_not_fire_when_rvol_baseline_missing():
    """No or_window_volume_baseline_14d AND no df_daily — fail-closed."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df()
    ctx = _ctx(df)
    ctx.indicators.pop("or_window_volume_baseline_14d", None)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "rvol" in (res.rejection_reason or "").lower()


# =============================================================================
# First-trigger latch tests
# =============================================================================

def test_first_trigger_latch_suppresses_subsequent_long_fires():
    """After plan_long fires, subsequent detect() on a later bar same session = no fire."""
    det = ORB15Structure(_cfg())
    df1 = _build_sweep_reclaim_long_df(sweep_time="09:35:00", reclaim_time="09:40:00")
    ctx1 = _ctx(df1)
    res1 = det.detect(ctx1)
    assert res1.structure_detected is True
    plan = det.plan_long_strategy(ctx1)  # plan_*_strategy registers the latch
    assert plan is not None
    # Now try a SECOND sweep+reclaim later in same session
    df2 = _build_sweep_reclaim_long_df(sweep_time="09:50:00", reclaim_time="09:55:00")
    ctx2 = _ctx(df2)
    res2 = det.detect(ctx2)
    assert res2.structure_detected is False
    assert "already" in (res2.rejection_reason or "").lower() or \
           "latched" in (res2.rejection_reason or "").lower() or \
           "fired" in (res2.rejection_reason or "").lower()


def test_first_trigger_latch_per_side_independent():
    """LONG latch does NOT block SHORT — they're independent keys."""
    det = ORB15Structure(_cfg())
    df1 = _build_sweep_reclaim_long_df(sweep_time="09:35:00", reclaim_time="09:40:00")
    ctx1 = _ctx(df1)
    assert det.detect(ctx1).structure_detected is True
    det.plan_long_strategy(ctx1)
    # Same session, opposite side — should still fire
    df2 = _build_sweep_reclaim_short_df(sweep_time="09:50:00", reclaim_time="09:55:00")
    ctx2 = _ctx(df2)
    res2 = det.detect(ctx2)
    assert res2.structure_detected is True, f"SHORT side independent: {res2.rejection_reason}"
    assert res2.events[0].side == "short"


def test_first_trigger_latch_resets_on_new_session_date():
    """Day 2 fresh session — latch from Day 1 must clear."""
    det = ORB15Structure(_cfg())
    df1 = _build_sweep_reclaim_long_df()
    ctx1 = _ctx(df1)  # session 2025-01-08 (Wed)
    det.detect(ctx1)
    det.plan_long_strategy(ctx1)
    # Build a second day's session using a Thursday (DOW != Friday)
    end_d2 = pd.Timestamp("2025-01-09 09:40:00")
    sweep_d2 = pd.Timestamp("2025-01-09 09:35:00")
    range_idx = pd.date_range("2025-01-09 09:20:00", periods=2, freq="5min")
    rows = []
    for ts in range_idx:
        rows.append({"ts": ts, "open": 101, "high": 102, "low": 100, "close": 101, "volume": 12500})
    rows.append({"ts": sweep_d2, "open": 101, "high": 102.5, "low": 100.95, "close": 101.5, "volume": 10000})
    rows.append({"ts": end_d2, "open": 101.5, "high": 102.65, "low": 101.45, "close": 102.6, "volume": 10000})
    df2 = pd.DataFrame(rows).set_index("ts")
    ctx2 = _ctx(df2)
    res2 = det.detect(ctx2)
    assert res2.structure_detected is True, f"New session must allow fire: {res2.rejection_reason}"


# =============================================================================
# Friday DOW exclusion tests
# =============================================================================

def test_does_not_fire_on_friday():
    """exclude_dow=[4] — 2025-01-10 is a Friday."""
    det = ORB15Structure(_cfg())
    end = pd.Timestamp("2025-01-10 09:40:00")  # Friday
    sweep_ts = pd.Timestamp("2025-01-10 09:35:00")
    range_idx = pd.date_range("2025-01-10 09:20:00", periods=2, freq="5min")
    rows = []
    for ts in range_idx:
        rows.append({"ts": ts, "open": 101, "high": 102, "low": 100, "close": 101, "volume": 12500})
    rows.append({"ts": sweep_ts, "open": 101, "high": 102.5, "low": 100.95, "close": 101.5, "volume": 10000})
    rows.append({"ts": end, "open": 101.5, "high": 102.65, "low": 101.45, "close": 102.6, "volume": 10000})
    df = pd.DataFrame(rows).set_index("ts")
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    reason = (res.rejection_reason or "").lower()
    assert "dow" in reason or "friday" in reason or "exclude" in reason


@pytest.mark.parametrize("date_str,expected_dow", [
    ("2025-01-06", 0),  # Mon
    ("2025-01-07", 1),  # Tue
    ("2025-01-08", 2),  # Wed
    ("2025-01-09", 3),  # Thu
])
def test_fires_on_non_friday_dows(date_str, expected_dow):
    det = ORB15Structure(_cfg())
    end = pd.Timestamp(f"{date_str} 09:40:00")
    assert end.weekday() == expected_dow
    sweep_ts = pd.Timestamp(f"{date_str} 09:35:00")
    range_idx = pd.date_range(f"{date_str} 09:20:00", periods=2, freq="5min")
    rows = []
    for ts in range_idx:
        rows.append({"ts": ts, "open": 101, "high": 102, "low": 100, "close": 101, "volume": 12500})
    rows.append({"ts": sweep_ts, "open": 101, "high": 102.5, "low": 100.95, "close": 101.5, "volume": 10000})
    rows.append({"ts": end, "open": 101.5, "high": 102.65, "low": 101.45, "close": 102.6, "volume": 10000})
    df = pd.DataFrame(rows).set_index("ts")
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"DOW={expected_dow} should fire: {res.rejection_reason}"


# =============================================================================
# Regime + side gate tests (explicit-allowlist semantics)
# =============================================================================

def test_respects_allowed_regimes_config():
    """Regime not in allowlist → reject."""
    det = ORB15Structure(_cfg(allowed_regimes=["squeeze"]))  # only squeeze allowed
    df = _build_sweep_reclaim_long_df()
    ctx = _ctx(df, regime="trend_up")  # not in allowlist
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "regime" in (res.rejection_reason or "").lower()


def test_empty_allowed_regimes_blocks_all():
    """Empty allowlist = no regime allowed (explicit-allowlist semantics)."""
    det = ORB15Structure(_cfg(allowed_regimes=[]))
    df = _build_sweep_reclaim_long_df()
    ctx = _ctx(df, regime="squeeze")  # any regime
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "regime" in (res.rejection_reason or "").lower()


def test_respects_allowed_sides_config_short_only():
    """allowed_sides=['short']: long candidate → reject; short candidate → fire."""
    det = ORB15Structure(_cfg(allowed_sides=["short"]))
    df_long = _build_sweep_reclaim_long_df()
    ctx_long = _ctx(df_long)
    res_long = det.detect(ctx_long)
    assert res_long.structure_detected is False
    assert "side" in (res_long.rejection_reason or "").lower()

    df_short = _build_sweep_reclaim_short_df()
    ctx_short = _ctx(df_short)
    res_short = det.detect(ctx_short)
    assert res_short.structure_detected is True


# =============================================================================
# Regression suite — preserve existing behavior
# =============================================================================

def test_does_not_fire_outside_active_window():
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df(sweep_time="14:00:00", reclaim_time="14:05:00")
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_symbol_outside_universe():
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df()
    ctx = _ctx(df, symbol="NSE:RANDOMSMALLCAP")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_on_gap_day_routes_to_gap_fade():
    """rev2: ORB disabled if open gap > 0.5% (route to gap_fade_short)."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df()
    # Range opens at 101 (mid of [100,102]); pdc=99 → gap 2% → exclude
    ctx = _ctx(df, pdc=99.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "gap_day" in (res.rejection_reason or "").lower()


def test_plan_long_emits_tiered_t1_t2():
    """LONG plan emits hard_sl + T1 (1R, 50% qty) + T2 (2R, 50% qty)."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df()
    ctx = _ctx(df)
    plan = det.plan_long_strategy(ctx)
    assert plan is not None
    assert plan.side == "long"
    assert plan.risk_params.hard_sl < plan.entry_price  # stop below entry
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["qty_pct"] == 0.5
    assert targets[1]["qty_pct"] == 0.5
    assert targets[0]["level"] < targets[1]["level"]  # T1 below T2 (long)
    assert targets[0]["level"] > plan.entry_price  # T1 above entry (long)


def test_plan_short_emits_tiered_t1_t2():
    """SHORT plan emits hard_sl + T1 + T2 with stop ABOVE entry, targets BELOW."""
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_short_df()
    ctx = _ctx(df)
    plan = det.plan_short_strategy(ctx)
    assert plan is not None
    assert plan.side == "short"
    assert plan.risk_params.hard_sl > plan.entry_price  # stop above entry
    targets = plan.exit_levels.targets
    assert len(targets) == 2
    assert targets[0]["level"] > targets[1]["level"]  # T1 above T2 (short)
    assert targets[0]["level"] < plan.entry_price  # T1 below entry (short)


# =============================================================================
# Wide-open mode bypass — universe is a design-inferred filter
# =============================================================================

def test_wide_open_bypasses_universe_filter(monkeypatch):
    """Under wide_open, an off-universe symbol still fires (gauntlet decides
    which universe slice the detector works in — see lessons.md 2026-04-15)."""
    import structures.orb_15_structure as mod
    monkeypatch.setattr(mod, "_is_wide_open", lambda: True)
    det = ORB15Structure(_cfg())
    df = _build_sweep_reclaim_long_df()
    ctx = _ctx(df, symbol="NSE:RANDOMSMALLCAP")   # not in fno_liquid_200
    res = det.detect(ctx)
    assert res.structure_detected is True, (
        f"wide_open should bypass universe: {res.rejection_reason}"
    )
