"""Regression tests for structures/flag_continuation_structure.py audit fixes.

Per docs/edge_discovery/audit/14-flag_continuation_structure.md.
Each test captures a specific bug discovered during the FIXED-AND-TRUSTED audit.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structures.flag_continuation_structure import FlagContinuationStructure
from structures.data_models import MarketContext


def _build_df(bars, start='2026-04-15 09:30', freq='5min'):
    idx = pd.date_range(start, periods=len(bars), freq=freq)
    return pd.DataFrame(
        bars,
        columns=['open', 'high', 'low', 'close', 'volume'],
        index=idx,
    )


def _config(setup_name="flag_continuation_long", flag_volume_decline_ratio=1.0):
    """Build a default config. flag_volume_decline_ratio=1.0 disables the
    Tier-A volume-decline filter (existing tests use flat volume)."""
    return {
        "_setup_name": setup_name,
        "min_consolidation_bars": 3,
        "max_consolidation_bars": 12,
        "min_trend_strength": 1.5,
        "trend_lookback_period": 10,
        "max_consolidation_range_pct": 2.0,
        "breakout_confirmation_pct": 0.1,
        "require_volume_confirmation": False,
        "min_volume_mult": 1.5,
        "flag_volume_decline_ratio": flag_volume_decline_ratio,
        "target_mult_t1": 1.5,
        "target_mult_t2": 2.5,
        "stop_mult": 1.0,
        "confidence_strong_flag": 0.85,
        "confidence_weak_flag": 0.65,
    }


def make_flag_long_bars():
    """Build 24+ bars with up-trend then tight consolidation then breakout above flag_high.

    Detector requires len(df) >= trend_lookback(10) + max_consol(12) + 2 = 24.
    With consol_period=3 the analyzer reads:
      trend_data = bars[10:20] (the bars between idx 10 and 19 inclusive)
      consol_data = bars[20:23] (last 3 bars before current)
      current = bar 23

    Layout: 14 trend bars (100 -> ~109) + 9 consolidation bars (~110) + 1 breakout bar.
    """
    bars = []
    # Trend up: 100 -> ~109 over 14 bars (large trend for safety)
    for i in range(14):
        p = 100.0 + i * 0.7
        bars.append((p, p + 0.2, p - 0.2, p + 0.4, 1000))
    # Consolidation around 110 (9 bars; range ~0.4 << 2%)
    for _ in range(9):
        bars.append((110.0, 110.2, 109.8, 110.0, 1000))
    # Breakout bar (current_price > consol_high(110.2) by > 0.1%)
    bars.append((110.0, 110.8, 109.95, 110.6, 2000))
    return _build_df(bars)


def make_flag_short_bars():
    """Mirror image: down-trend then consolidation then breakdown below flag_low."""
    bars = []
    for i in range(14):
        p = 100.0 - i * 0.7
        bars.append((p, p + 0.2, p - 0.2, p - 0.4, 1000))
    for _ in range(9):
        bars.append((90.0, 90.2, 89.8, 90.0, 1000))
    # Breakdown bar
    bars.append((90.0, 90.05, 89.2, 89.4, 2000))
    return _build_df(bars)


def _ctx(df, symbol="TEST", cap_segment=None):
    return MarketContext(
        symbol=symbol,
        current_price=float(df['close'].iloc[-1]),
        timestamp=df.index[-1],
        df_5m=df,
        session_date=datetime(2026, 4, 15),
        cap_segment=cap_segment,
    )


# -----------------------------------------------------------------------------
# P1 #1 — levels dict must include side-aware support/resistance/broken_level keys
# -----------------------------------------------------------------------------

def test_levels_dict_long_has_support_and_broken_level():
    """Audit P1 #1: long flag_continuation must emit 'support' AND 'broken_level' keys
    so main_detector can resolve detected_level."""
    detector = FlagContinuationStructure(_config("flag_continuation_long"))
    df = make_flag_long_bars()
    analysis = detector.detect(_ctx(df))

    assert analysis.structure_detected, f"Expected detection, got: {analysis.rejection_reason}"
    event = analysis.events[0]
    assert event.side == "long"
    # main_detector reads: support OR nearest_support OR broken_level for long side
    assert "support" in event.levels, f"levels missing 'support': {event.levels}"
    assert "broken_level" in event.levels, f"levels missing 'broken_level': {event.levels}"
    # support and broken_level should equal flag_high (the broken consolidation top)
    assert event.levels["support"] == pytest.approx(event.levels["flag_high"])
    assert event.levels["broken_level"] == pytest.approx(event.levels["flag_high"])


def test_levels_dict_short_has_resistance_and_broken_level():
    """Audit P1 #1 mirror: short flag_continuation must emit 'resistance' AND 'broken_level'."""
    detector = FlagContinuationStructure(_config("flag_continuation_short"))
    df = make_flag_short_bars()
    analysis = detector.detect(_ctx(df))

    assert analysis.structure_detected, f"Expected detection, got: {analysis.rejection_reason}"
    event = analysis.events[0]
    assert event.side == "short"
    assert "resistance" in event.levels, f"levels missing 'resistance': {event.levels}"
    assert "broken_level" in event.levels, f"levels missing 'broken_level': {event.levels}"
    assert event.levels["resistance"] == pytest.approx(event.levels["flag_low"])
    assert event.levels["broken_level"] == pytest.approx(event.levels["flag_low"])


# -----------------------------------------------------------------------------
# P2 #2 — blocked_cap_segments parity with other detectors
# -----------------------------------------------------------------------------

def test_blocked_cap_segments_list_format_skips_detection():
    """Audit P2 #2: list-format blocked_cap_segments (parity with other detectors) must work."""
    cfg = _config("flag_continuation_long")
    cfg["blocked_cap_segments"] = ["micro_cap"]
    detector = FlagContinuationStructure(cfg)

    df = make_flag_long_bars()
    analysis = detector.detect(_ctx(df, cap_segment="micro_cap"))
    assert not analysis.structure_detected
    assert "micro_cap" in (analysis.rejection_reason or "")


def test_blocked_cap_segments_dict_format_backward_compat():
    """Audit P2 #2: dict-with-segments format also accepted (FHM-style legacy schema)."""
    cfg = _config("flag_continuation_long")
    cfg["blocked_cap_segments"] = {"segments": ["micro_cap"]}
    detector = FlagContinuationStructure(cfg)

    df = make_flag_long_bars()
    analysis = detector.detect(_ctx(df, cap_segment="micro_cap"))
    assert not analysis.structure_detected


def test_blocked_cap_segments_does_not_block_other_caps():
    """Sanity: blocking micro_cap must not affect mid_cap detection."""
    cfg = _config("flag_continuation_long")
    cfg["blocked_cap_segments"] = ["micro_cap"]
    detector = FlagContinuationStructure(cfg)

    df = make_flag_long_bars()
    analysis = detector.detect(_ctx(df, cap_segment="mid_cap"))
    assert analysis.structure_detected, f"Should detect for mid_cap, got: {analysis.rejection_reason}"


# =============================================================================
# Tier-A: missing config keys must FAIL FAST (CLAUDE.md)
# =============================================================================

def test_missing_flag_volume_decline_ratio_fails_fast():
    """Audit/14 Tier-A: flag_volume_decline_ratio is required (no silent default)."""
    cfg = _config()
    del cfg["flag_volume_decline_ratio"]
    with pytest.raises(KeyError, match="flag_volume_decline_ratio"):
        FlagContinuationStructure(cfg)


# =============================================================================
# Tier-A: volume-DECLINE-through-flag canonical filter
# =============================================================================

def _make_flag_long_with_decline():
    """24-bar flag long where consol volume DECLINED (canonical flag)."""
    bars = []
    # Trend bars 0-13 with HIGH volume (the flagpole)
    for i in range(14):
        p = 100.0 + i * 0.7
        bars.append((p, p + 0.2, p - 0.2, p + 0.4, 5000))
    # Consol bars 14-22 with LOW volume (volume dries up — canonical flag)
    for _ in range(9):
        bars.append((110.0, 110.2, 109.8, 110.0, 1000))  # 1000 / 5000 = 0.2 ratio
    # Breakout bar
    bars.append((110.0, 110.8, 109.95, 110.6, 6000))
    return _build_df(bars)


def _make_flag_long_without_decline():
    """24-bar flag long where consol volume did NOT decline (not canonical)."""
    bars = []
    # Trend bars with same volume as consol — no decline
    for i in range(14):
        p = 100.0 + i * 0.7
        bars.append((p, p + 0.2, p - 0.2, p + 0.4, 1000))
    # Consol bars with SAME volume (no decline)
    for _ in range(9):
        bars.append((110.0, 110.2, 109.8, 110.0, 1000))  # ratio 1.0
    bars.append((110.0, 110.8, 109.95, 110.6, 2000))
    return _build_df(bars)


def test_volume_decline_filter_accepts_canonical_flag():
    """Tier-A: flag with volume DECLINE through consol is accepted."""
    detector = FlagContinuationStructure(_config(flag_volume_decline_ratio=0.85))
    df = _make_flag_long_with_decline()
    analysis = detector.detect(_ctx(df))
    assert analysis.structure_detected, f"Got: {analysis.rejection_reason}"


def test_volume_decline_filter_rejects_non_canonical_flag():
    """Tier-A: flag WITHOUT volume decline (consol vol == trend vol) is rejected
    AND the specific volume-decline reason surfaces in rejection_reason
    (audit/14 Tier-A diagnostic — must be visible to OCI gauntlet analysis)."""
    detector = FlagContinuationStructure(_config(flag_volume_decline_ratio=0.85))
    df = _make_flag_long_without_decline()
    analysis = detector.detect(_ctx(df))
    assert not analysis.structure_detected
    # Specific volume-decline rejection must propagate (was previously buried
    # in _analyze_flag_pattern returning None and lost as generic message)
    assert analysis.rejection_reason is not None
    assert "volume did not decline" in analysis.rejection_reason.lower(), (
        f"Expected volume-decline rejection; got: {analysis.rejection_reason}"
    )


def test_volume_decline_filter_disabled_at_ratio_one():
    """Tier-A: flag_volume_decline_ratio=1.0 disables the filter (back-compat)."""
    detector = FlagContinuationStructure(_config(flag_volume_decline_ratio=1.0))
    df = _make_flag_long_without_decline()
    analysis = detector.detect(_ctx(df))
    assert analysis.structure_detected, f"Filter should be off; got: {analysis.rejection_reason}"
