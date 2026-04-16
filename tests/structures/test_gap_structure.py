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
    return pd.DataFrame(
        bars,
        columns=['open', 'high', 'low', 'close', 'volume'],
        index=idx,
    )


def _config(setup_name):
    return {
        "_setup_name": setup_name,
        "min_gap_pct": 0.3,
        "max_gap_pct": 2.5,
        "require_volume_confirmation": True,
        "target_mult_t1": 1.0,
        "target_mult_t2": 2.0,
        "confidence_level": 0.7,
        "gap_sl_buffer_atr": 0.3,
        "min_stop_distance_pct": 0.3,
    }


def make_gap_up_bars(open_price=102.0, current_price=101.5):
    """Build bars where session opened with a gap up vs PDC (100), current price below open."""
    bars = []
    # First bar opens at 102 (gap up vs PDC=100)
    bars.append((open_price, open_price + 0.3, open_price - 0.3, open_price + 0.1, 1000))
    # Subsequent bars drift toward current_price
    for _ in range(5):
        bars.append((open_price - 0.1, open_price + 0.1, open_price - 0.5, current_price, 1000))
    return _build_df(bars)


def make_gap_down_bars(open_price=98.0, current_price=98.5):
    """Build bars where session opened with a gap down vs PDC (100), current price above open."""
    bars = []
    bars.append((open_price, open_price + 0.3, open_price - 0.3, open_price + 0.1, 1000))
    for _ in range(5):
        bars.append((open_price + 0.1, open_price + 0.5, open_price - 0.1, current_price, 1000))
    return _build_df(bars)


def _ctx(df, symbol="TEST", cap_segment=None, pdc=100.0):
    return MarketContext(
        symbol=symbol,
        current_price=float(df['close'].iloc[-1]),
        timestamp=df.index[-1],
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
