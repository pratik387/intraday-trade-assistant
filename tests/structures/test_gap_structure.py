"""Regression tests for structures/gap_structure.py audit fixes.

Per docs/edge_discovery/audit/15-gap_structure.md.
Each test captures a specific bug discovered during the FIXED-AND-TRUSTED audit.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structures.gap_structure import GapStructure
from structures.data_models import MarketContext


def _build_df(bars, start='2026-04-15 09:15', freq='5min'):
    idx = pd.date_range(start, periods=len(bars), freq=freq)
    df = pd.DataFrame(
        bars,
        columns=['open', 'high', 'low', 'close', 'volume'],
        index=idx,
    )
    # Provide vol_z for volume confirmation (Tier-A fix #1)
    df['vol_z'] = 2.0  # default high vol_z so volume confirmation passes
    return df


def _config(setup_name, require_volume=False):
    return {
        "_setup_name": setup_name,
        "min_gap_pct": 0.3,
        "max_gap_pct": 2.5,
        "require_volume_confirmation": require_volume,
        "min_volume_mult": 1.2,
        "gap_fill_start_hhmm": "0915",
        "gap_fill_end_hhmm": "1030",
        "target_mult_t1": 1.0,
        "target_mult_t2": 2.0,
        "confidence_level": 0.7,
        "gap_sl_buffer_atr": 0.3,
        "min_stop_distance_pct": 0.3,
    }


def make_gap_up_bars(open_price=102.0, current_price=101.5, current_open=None):
    """Build bars where session opened with a gap up vs PDC (100).

    The LAST bar's open/close determines reversal-candle behavior:
      - For gap_fill_short: last bar must be bearish (close < open)
      - For gap_breakout_long: last bar can be anything; price > current_open
    """
    bars = []
    bars.append((open_price, open_price + 0.3, open_price - 0.3, open_price + 0.1, 1000))
    for _ in range(4):
        bars.append((open_price - 0.1, open_price + 0.1, open_price - 0.5, current_price, 1000))
    # Last bar — open higher than close to satisfy bearish reversal for gap_fill_short
    last_open = current_open if current_open is not None else current_price + 0.1
    bars.append((last_open, last_open + 0.05, current_price - 0.1, current_price, 1000))
    return _build_df(bars)


def make_gap_down_bars(open_price=98.0, current_price=98.5, current_open=None):
    """Build bars where session opened with a gap down vs PDC (100).

    Last bar must be bullish (close > open) for gap_fill_long reversal confirmation.
    """
    bars = []
    bars.append((open_price, open_price + 0.3, open_price - 0.3, open_price + 0.1, 1000))
    for _ in range(4):
        bars.append((open_price + 0.1, open_price + 0.5, open_price - 0.1, current_price, 1000))
    # Last bar — open lower than close to satisfy bullish reversal for gap_fill_long
    last_open = current_open if current_open is not None else current_price - 0.1
    bars.append((last_open, current_price + 0.1, last_open - 0.05, current_price, 1000))
    return _build_df(bars)


def _ctx(df, symbol="TEST", cap_segment=None, pdc=100.0, timestamp=None):
    return MarketContext(
        symbol=symbol,
        current_price=float(df['close'].iloc[-1]),
        timestamp=timestamp if timestamp is not None else df.index[-1],
        df_5m=df,
        session_date=datetime(2026, 4, 15),
        cap_segment=cap_segment,
        pdc=pdc,
    )


# -----------------------------------------------------------------------------
# P1 #1 — levels dict must include side-aware support/resistance/broken_level
# -----------------------------------------------------------------------------

def test_gap_fill_short_levels_dict_has_resistance_and_broken_level():
    """Audit P1 #1: gap_fill_short (gap-up + price below open) emits resistance = current_open."""
    detector = GapStructure(_config("gap_fill_short"))
    df = make_gap_up_bars(open_price=102.0, current_price=101.5)  # gap up; price below open
    analysis = detector.detect(_ctx(df))

    assert analysis.structure_detected, f"Expected detection, got: {analysis.rejection_reason}"
    event = analysis.events[0]
    assert event.structure_type == "gap_fill_short"
    assert event.side == "short"
    assert "resistance" in event.levels
    assert "broken_level" in event.levels
    # For fill, structural level is the gap_open (the high of the gap on gap-up)
    assert event.levels["resistance"] == pytest.approx(event.levels["gap_open"])
    assert event.levels["broken_level"] == pytest.approx(event.levels["gap_open"])


def test_gap_fill_long_levels_dict_has_support_and_broken_level():
    """Audit P1 #1 mirror: gap_fill_long (gap-down + price above open) emits support = current_open."""
    detector = GapStructure(_config("gap_fill_long"))
    df = make_gap_down_bars(open_price=98.0, current_price=98.5)
    analysis = detector.detect(_ctx(df))

    assert analysis.structure_detected, f"Expected detection, got: {analysis.rejection_reason}"
    event = analysis.events[0]
    assert event.structure_type == "gap_fill_long"
    assert event.side == "long"
    assert "support" in event.levels
    assert "broken_level" in event.levels
    assert event.levels["support"] == pytest.approx(event.levels["gap_open"])
    assert event.levels["broken_level"] == pytest.approx(event.levels["gap_open"])


def test_gap_breakout_long_levels_dict_uses_pdc_for_support():
    """Audit P1 #1: gap_breakout_long (gap-up + price above open) — structural level is PDC."""
    detector = GapStructure(_config("gap_breakout_long"))
    df = make_gap_up_bars(open_price=102.0, current_price=102.5)  # gap up; price above open
    analysis = detector.detect(_ctx(df))

    assert analysis.structure_detected, f"Expected detection, got: {analysis.rejection_reason}"
    event = analysis.events[0]
    assert event.structure_type == "gap_breakout_long"
    assert event.side == "long"
    assert "support" in event.levels
    # For breakout, structural level is PDC (gap edge to hold above)
    assert event.levels["support"] == pytest.approx(event.levels["gap_level"])


def test_gap_breakout_short_levels_dict_uses_pdc_for_resistance():
    """Audit P1 #1: gap_breakout_short (gap-down + price below open) — structural level is PDC."""
    detector = GapStructure(_config("gap_breakout_short"))
    df = make_gap_down_bars(open_price=98.0, current_price=97.5)  # gap down; price below open
    analysis = detector.detect(_ctx(df))

    assert analysis.structure_detected, f"Expected detection, got: {analysis.rejection_reason}"
    event = analysis.events[0]
    assert event.structure_type == "gap_breakout_short"
    assert event.side == "short"
    assert "resistance" in event.levels
    assert event.levels["resistance"] == pytest.approx(event.levels["gap_level"])


# -----------------------------------------------------------------------------
# P2 #2 — blocked_cap_segments parity with other detectors
# -----------------------------------------------------------------------------

def test_blocked_cap_segments_list_format_skips_detection():
    """Audit P2 #2: list-format blocked_cap_segments (parity with other detectors) must work."""
    cfg = _config("gap_fill_short")
    cfg["blocked_cap_segments"] = ["micro_cap"]
    detector = GapStructure(cfg)

    df = make_gap_up_bars(open_price=102.0, current_price=101.5)
    analysis = detector.detect(_ctx(df, cap_segment="micro_cap"))
    assert not analysis.structure_detected
    assert "micro_cap" in (analysis.rejection_reason or "")


def test_blocked_cap_segments_dict_format_backward_compat():
    """Audit P2 #2: dict-with-segments format also accepted (FHM-style legacy schema)."""
    cfg = _config("gap_fill_short")
    cfg["blocked_cap_segments"] = {"segments": ["micro_cap"]}
    detector = GapStructure(cfg)

    df = make_gap_up_bars(open_price=102.0, current_price=101.5)
    analysis = detector.detect(_ctx(df, cap_segment="micro_cap"))
    assert not analysis.structure_detected


def test_blocked_cap_segments_does_not_block_other_caps():
    """Sanity: blocking micro_cap must not affect mid_cap detection."""
    cfg = _config("gap_fill_short")
    cfg["blocked_cap_segments"] = ["micro_cap"]
    detector = GapStructure(cfg)

    df = make_gap_up_bars(open_price=102.0, current_price=101.5)
    analysis = detector.detect(_ctx(df, cap_segment="mid_cap"))
    assert analysis.structure_detected, f"Should detect for mid_cap, got: {analysis.rejection_reason}"


# =============================================================================
# Tier-A: missing config keys must FAIL FAST (CLAUDE.md)
# =============================================================================

def test_missing_min_volume_mult_fails_fast():
    """Audit/15 Tier-A #1: min_volume_mult is required (no silent default)."""
    cfg = _config("gap_fill_short")
    del cfg["min_volume_mult"]
    with pytest.raises(KeyError, match="min_volume_mult"):
        GapStructure(cfg)


def test_missing_gap_fill_start_hhmm_fails_fast():
    """Audit/15 Tier-A #2: gap_fill_start_hhmm is required."""
    cfg = _config("gap_fill_short")
    del cfg["gap_fill_start_hhmm"]
    with pytest.raises(KeyError, match="gap_fill_start_hhmm"):
        GapStructure(cfg)


def test_missing_gap_fill_end_hhmm_fails_fast():
    """Audit/15 Tier-A #2: gap_fill_end_hhmm is required."""
    cfg = _config("gap_fill_short")
    del cfg["gap_fill_end_hhmm"]
    with pytest.raises(KeyError, match="gap_fill_end_hhmm"):
        GapStructure(cfg)


# =============================================================================
# Tier-A #1: volume confirmation actually wired up (was DEAD code)
# =============================================================================

def test_volume_confirmation_blocks_low_vol_z_fill():
    """Tier-A #1: when require_volume_confirmation=true, low vol_z blocks the signal."""
    detector = GapStructure(_config("gap_fill_short", require_volume=True))
    df = make_gap_up_bars(open_price=102.0, current_price=101.5)
    df['vol_z'] = 0.5  # below 1.2 threshold
    analysis = detector.detect(_ctx(df))
    assert not analysis.structure_detected
    assert "volume confirmation" in (analysis.rejection_reason or "").lower()


def test_volume_confirmation_passes_with_high_vol_z():
    """Tier-A #1: high vol_z passes volume confirmation."""
    detector = GapStructure(_config("gap_fill_short", require_volume=True))
    df = make_gap_up_bars(open_price=102.0, current_price=101.5)
    df['vol_z'] = 2.0  # well above 1.2 threshold
    analysis = detector.detect(_ctx(df))
    assert analysis.structure_detected, f"Expected detection, got: {analysis.rejection_reason}"


def test_volume_confirmation_disabled_skips_check():
    """Tier-A #1: when require_volume_confirmation=false, vol_z is ignored."""
    detector = GapStructure(_config("gap_fill_short", require_volume=False))
    df = make_gap_up_bars(open_price=102.0, current_price=101.5)
    df['vol_z'] = 0.1  # very low; should pass since flag is off
    analysis = detector.detect(_ctx(df))
    assert analysis.structure_detected


# =============================================================================
# Tier-A #2: time-of-day window for fills
# =============================================================================

def test_gap_fill_outside_window_rejected():
    """Tier-A #2: gap_fill at 11:30 (outside 09:15-10:30 window) is rejected."""
    detector = GapStructure(_config("gap_fill_short"))
    df = make_gap_up_bars(open_price=102.0, current_price=101.5)
    late_ts = pd.Timestamp("2026-04-15 11:30:00")
    analysis = detector.detect(_ctx(df, timestamp=late_ts))
    assert not analysis.structure_detected
    assert "time window" in (analysis.rejection_reason or "").lower()


def test_gap_fill_inside_window_accepted():
    """Tier-A #2: gap_fill at 10:00 (inside window) is accepted."""
    detector = GapStructure(_config("gap_fill_short"))
    df = make_gap_up_bars(open_price=102.0, current_price=101.5)
    in_window_ts = pd.Timestamp("2026-04-15 10:00:00")
    analysis = detector.detect(_ctx(df, timestamp=in_window_ts))
    assert analysis.structure_detected, f"Got: {analysis.rejection_reason}"


def test_gap_breakout_NOT_subject_to_fill_window():
    """Tier-A #2: gap_breakout setups can fire any time (continuation pattern)."""
    detector = GapStructure(_config("gap_breakout_long"))
    # Gap up + price above current_open → breakout_long
    df = make_gap_up_bars(open_price=102.0, current_price=102.5)
    late_ts = pd.Timestamp("2026-04-15 13:30:00")
    analysis = detector.detect(_ctx(df, timestamp=late_ts))
    assert analysis.structure_detected, f"Breakout should fire late: {analysis.rejection_reason}"


# =============================================================================
# Tier-A #3: reversal candle confirmation for fills
# =============================================================================

def test_gap_fill_short_requires_bearish_reversal_candle():
    """Tier-A #3: gap_fill_short rejects when current bar is bullish (close > open)."""
    detector = GapStructure(_config("gap_fill_short"))
    # Build bars with last bar BULLISH (close > open) — should fail reversal check
    bars = []
    bars.append((102.0, 102.3, 101.7, 102.1, 1000))
    for _ in range(4):
        bars.append((101.9, 102.0, 101.4, 101.5, 1000))
    # Last bar bullish: open=101.4, close=101.5 (close > open)
    bars.append((101.4, 101.6, 101.3, 101.5, 1000))
    df = _build_df(bars)
    analysis = detector.detect(_ctx(df))
    assert not analysis.structure_detected
    assert "bearish reversal" in (analysis.rejection_reason or "").lower()


def test_gap_fill_long_requires_bullish_reversal_candle():
    """Tier-A #3: gap_fill_long rejects when current bar is bearish (close < open)."""
    detector = GapStructure(_config("gap_fill_long"))
    bars = []
    bars.append((98.0, 98.3, 97.7, 98.1, 1000))
    for _ in range(4):
        bars.append((98.1, 98.5, 98.0, 98.5, 1000))
    # Last bar bearish: open=98.6, close=98.5 (close < open)
    bars.append((98.6, 98.7, 98.4, 98.5, 1000))
    df = _build_df(bars)
    analysis = detector.detect(_ctx(df))
    assert not analysis.structure_detected
    assert "bullish reversal" in (analysis.rejection_reason or "").lower()


def test_gap_breakout_NOT_subject_to_reversal_check():
    """Tier-A #3: breakouts don't need reversal candles (continuation pattern)."""
    detector = GapStructure(_config("gap_breakout_long"))
    # Gap up; price > current_open; last bar can be anything
    bars = []
    bars.append((102.0, 102.3, 101.7, 102.1, 1000))
    for _ in range(4):
        bars.append((102.1, 102.7, 102.0, 102.5, 1000))
    # Last bar bearish (close < open) but price still > current_open → breakout
    bars.append((102.6, 102.7, 102.4, 102.5, 1000))
    df = _build_df(bars)
    analysis = detector.detect(_ctx(df))
    assert analysis.structure_detected, f"Got: {analysis.rejection_reason}"


# =============================================================================
# Tier-A #4: per-session dedup
# =============================================================================

def test_per_session_dedup_blocks_second_fire_same_session():
    """Tier-A #4: same setup_type fires at most once per (symbol, session)."""
    detector = GapStructure(_config("gap_fill_short"))
    df = make_gap_up_bars(open_price=102.0, current_price=101.5)

    # First call at 10:00 should fire
    analysis1 = detector.detect(_ctx(df, timestamp=pd.Timestamp("2026-04-15 10:00:00")))
    assert analysis1.structure_detected

    # Second call at 10:05 same session should be blocked
    analysis2 = detector.detect(_ctx(df, timestamp=pd.Timestamp("2026-04-15 10:05:00")))
    assert not analysis2.structure_detected
    assert "already fired" in (analysis2.rejection_reason or "").lower()


def test_per_session_dedup_resets_on_new_session():
    """Tier-A #4: new session resets the dedup state."""
    detector = GapStructure(_config("gap_fill_short"))
    df = make_gap_up_bars(open_price=102.0, current_price=101.5)

    # First session
    ctx1 = MarketContext(
        symbol="TEST",
        current_price=float(df['close'].iloc[-1]),
        timestamp=pd.Timestamp("2026-04-15 10:00:00"),
        df_5m=df,
        session_date=datetime(2026, 4, 15),
        cap_segment=None,
        pdc=100.0,
    )
    analysis1 = detector.detect(ctx1)
    assert analysis1.structure_detected

    # Next session — should fire again
    ctx2 = MarketContext(
        symbol="TEST",
        current_price=float(df['close'].iloc[-1]),
        timestamp=pd.Timestamp("2026-04-16 10:00:00"),
        df_5m=df,
        session_date=datetime(2026, 4, 16),
        cap_segment=None,
        pdc=100.0,
    )
    analysis2 = detector.detect(ctx2)
    assert analysis2.structure_detected, f"Got: {analysis2.rejection_reason}"


def test_per_session_dedup_per_symbol():
    """Tier-A #4: dedup is per-symbol, not global."""
    detector = GapStructure(_config("gap_fill_short"))
    df = make_gap_up_bars(open_price=102.0, current_price=101.5)

    # Symbol A fires
    ctx_a = MarketContext(
        symbol="SYM_A",
        current_price=float(df['close'].iloc[-1]),
        timestamp=pd.Timestamp("2026-04-15 10:00:00"),
        df_5m=df,
        session_date=datetime(2026, 4, 15),
        cap_segment=None,
        pdc=100.0,
    )
    analysis_a = detector.detect(ctx_a)
    assert analysis_a.structure_detected

    # Symbol B same session — should also fire (different symbol)
    ctx_b = MarketContext(
        symbol="SYM_B",
        current_price=float(df['close'].iloc[-1]),
        timestamp=pd.Timestamp("2026-04-15 10:00:00"),
        df_5m=df,
        session_date=datetime(2026, 4, 15),
        cap_segment=None,
        pdc=100.0,
    )
    analysis_b = detector.detect(ctx_b)
    assert analysis_b.structure_detected, f"Got: {analysis_b.rejection_reason}"
