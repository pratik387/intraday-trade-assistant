"""Regression tests for structures/level_breakout_structure.py audit fixes.

Per docs/edge_discovery/audit/05-level_breakout_structure.md.
Each test captures a specific bug discovered during the FIXED-AND-TRUSTED audit.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structures.level_breakout_structure import LevelBreakoutStructure
from structures.main_detector import MainDetector
from structures.data_models import MarketContext


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_df(bars, start_date='2026-04-15 10:00'):
    """Build a 5m bar dataframe from a list of (open, high, low, close, volume) tuples.

    Default start is 10:00 AM (past institutional timing filter at 9:45).
    """
    idx = pd.date_range(start_date, periods=len(bars), freq='5min')
    df = pd.DataFrame(
        bars,
        columns=['open', 'high', 'low', 'close', 'volume'],
        index=idx,
    )
    # Pre-compute a strong vol_z so volume gates pass.
    df['vol_z'] = 2.5
    return df


def make_context(df, **kwargs):
    last = df.iloc[-1]
    indicators = kwargs.pop('indicators', {'atr': 0.5, 'vol_z': 2.5})
    current_price = kwargs.pop('current_price', float(last['close']))
    defaults = {
        'symbol': 'NSE:TEST',
        'timestamp': df.index[-1].to_pydatetime(),
        'df_5m': df,
        'session_date': df.index[0].to_pydatetime(),
        'current_price': current_price,
        'pdh': kwargs.pop('pdh', None),
        'pdl': kwargs.pop('pdl', None),
        'orh': kwargs.pop('orh', None),
        'orl': kwargs.pop('orl', None),
        'indicators': indicators,
        'cap_segment': kwargs.pop('cap_segment', 'mid_cap'),
    }
    defaults.update(kwargs)
    return MarketContext(**defaults)


def make_level_breakout_config(setup_name: str = 'level_breakout_long'):
    """Load a level_breakout sub-config from configuration.json."""
    with open(ROOT / 'config' / 'configuration.json') as f:
        cfg = json.load(f)
    setups = cfg.get('setups') or cfg.get('trading_setups') or {}
    lb = setups.get(setup_name)
    assert lb is not None, f"{setup_name} config not found in configuration.json"
    return dict(lb)


def make_pdh_breakout_bars():
    """Build bars that break above PDH=100 with high volume and bullish conviction.

    Bars 0-16 consolidate below 100; bars 17-19 close above 100 to avoid the
    liquidity-grab filter (which rejects if ANY of last-3 closes < level after
    a break above level).
    """
    bars = []
    # 17 consolidation bars (well below level, highs well below 99.7 so no false "broke_above" trigger)
    for i in range(17):
        bars.append((99.3, 99.5, 99.0, 99.4, 1000))
    # 3 sustained-above bars (all closes > 100.0)
    bars.append((99.5, 100.5, 99.5, 100.3, 2500))
    bars.append((100.3, 100.6, 100.2, 100.5, 2500))
    # Final conviction bar: close in top 10% of range, strong volume
    bars.append((100.5, 100.8, 100.5, 100.75, 3500))
    return bars


def make_pdl_breakdown_bars_short():
    """Build bars that break below PDL=100 with high volume and bearish conviction.

    Bars 0-16 consolidate above 100; bars 17-19 close below 100 to avoid the
    liquidity-grab filter.
    """
    bars = []
    # 17 consolidation bars (well above level)
    for i in range(17):
        bars.append((100.7, 101.0, 100.5, 100.6, 1000))
    # 3 sustained-below bars (all closes < 100.0)
    bars.append((100.5, 100.5, 99.4, 99.6, 2500))
    bars.append((99.6, 99.8, 99.4, 99.5, 2500))
    # Final conviction bar: close in bottom 10% of range
    bars.append((99.5, 99.5, 99.2, 99.23, 3500))
    return bars


def make_orb_breakout_bars_long():
    """Bars where first 3 bars form opening range, later bar breaks above ORH."""
    bars = []
    # First 3 (opening range): highs up to 100
    bars.append((99.5, 100.0, 99.3, 99.8, 1000))
    bars.append((99.8, 100.0, 99.5, 99.9, 1000))
    bars.append((99.9, 99.95, 99.6, 99.7, 1000))
    # 16 filler bars consolidating below 100
    for i in range(16):
        bars.append((99.3, 99.8, 99.0, 99.5, 1000))
    # Breakout bar
    bars.append((99.5, 100.8, 99.5, 100.7, 3500))
    return bars


# ---------------------------------------------------------------------------
# Fix 1: P1 — orb_level_breakout_long/_short missing from main_detector mapping
# ---------------------------------------------------------------------------

def test_main_detector_direct_mappings_include_orb_level_breakout_long():
    """Regression: main_detector's direct_mappings must include orb_level_breakout_long.

    Bug: LevelBreakoutStructure emits 'orb_level_breakout_long' when ORH is broken,
    but main_detector._map_structure_to_setup_type had no entry for it, returning
    None → event silently dropped at SetupCandidate conversion.
    """
    with open(ROOT / 'config' / 'configuration.json') as f:
        cfg = json.load(f)
    detector = MainDetector(config=cfg)
    mapped = detector._map_structure_to_setup_type('orb_level_breakout_long')
    assert mapped is not None, (
        "orb_level_breakout_long must be mapped in main_detector.direct_mappings. "
        "Currently None → events silently dropped at _convert_to_setup_candidates."
    )


def test_main_detector_direct_mappings_include_orb_level_breakout_short():
    """Regression: main_detector's direct_mappings must include orb_level_breakout_short."""
    with open(ROOT / 'config' / 'configuration.json') as f:
        cfg = json.load(f)
    detector = MainDetector(config=cfg)
    mapped = detector._map_structure_to_setup_type('orb_level_breakout_short')
    assert mapped is not None, (
        "orb_level_breakout_short must be mapped in main_detector.direct_mappings."
    )


# ---------------------------------------------------------------------------
# Fix 2: P1 #2 — Short path asymmetry (entry_mode, retest_zone, dedupe)
# ---------------------------------------------------------------------------

def test_detect_short_breakdown_emits_entry_mode_in_context():
    """Regression: short breakdown must emit 'entry_mode' in context (mirror of long path)."""
    bars = make_pdl_breakdown_bars_short()
    df = make_df(bars)
    # Use long config — short config in JSON is minimal/disabled stub; the
    # class handles both long and short via the same parameter set.
    cfg = make_level_breakout_config('level_breakout_long')
    detector = LevelBreakoutStructure(cfg)
    ctx = make_context(df, pdl=100.0, current_price=99.3)
    result = detector.detect(ctx)
    short_events = [e for e in result.events if e.structure_type == 'level_breakout_short']
    assert len(short_events) >= 1, (
        f"Expected level_breakout_short event; got events: "
        f"{[(e.structure_type, e.context) for e in result.events]}, "
        f"rejection: {result.rejection_reason}"
    )
    event = short_events[0]
    assert 'entry_mode' in event.context, (
        f"Short context must include entry_mode (mirror of long). Got: {event.context}"
    )
    assert event.context['entry_mode'] in ('immediate', 'retest', 'pending')


def test_detect_short_breakdown_emits_retest_zone_in_context():
    """Regression: short breakdown must emit 'retest_zone' key in context (mirror of long)."""
    bars = make_pdl_breakdown_bars_short()
    df = make_df(bars)
    # Use long config — short config in JSON is minimal/disabled stub; the
    # class handles both long and short via the same parameter set.
    cfg = make_level_breakout_config('level_breakout_long')
    detector = LevelBreakoutStructure(cfg)
    ctx = make_context(df, pdl=100.0, current_price=99.3)
    result = detector.detect(ctx)
    short_events = [e for e in result.events if e.structure_type == 'level_breakout_short']
    assert len(short_events) >= 1
    event = short_events[0]
    assert 'retest_zone' in event.context, (
        f"Short context must include retest_zone key (may be None). Got: {event.context}"
    )


def test_short_breakdown_registers_in_traded_breakouts_today():
    """Regression: short path must populate traded_breakouts_today like long path.

    Without this, _determine_entry_mode is never called on shorts, so:
    - aggressive-vs-retest quality filter is bypassed on shorts
    - same PDL breakdown could be entered multiple times
    """
    bars = make_pdl_breakdown_bars_short()
    df = make_df(bars)
    # Use long config — short config in JSON is minimal/disabled stub; the
    # class handles both long and short via the same parameter set.
    cfg = make_level_breakout_config('level_breakout_long')
    detector = LevelBreakoutStructure(cfg)
    ctx = make_context(df, pdl=100.0, current_price=99.3)
    result = detector.detect(ctx)
    short_events = [e for e in result.events if e.structure_type == 'level_breakout_short']
    assert len(short_events) >= 1, "Setup should fire on first detection"

    # After first detect(), PDL breakdown should be tracked in dedupe state.
    # Representation may be set or dict-of-sets; either way it should be non-empty.
    tracked = detector.traded_breakouts_today
    if isinstance(tracked, set):
        count = len(tracked)
    elif isinstance(tracked, dict):
        count = sum(len(v) for v in tracked.values())
    else:
        count = 0
    assert count > 0, (
        "Short breakdown must register in traded_breakouts_today (like long path does). "
        "Currently short path skips _determine_entry_mode → dedupe never populated."
    )


# ---------------------------------------------------------------------------
# Fix 3: P1 #3 — traded_breakouts_today session reset
# ---------------------------------------------------------------------------

def test_traded_breakouts_today_scopes_to_session_date():
    """Regression: dedupe must scope to session date; yesterday's breakouts shouldn't block today's.

    Bug: traded_breakouts_today was a single set() never reset across sessions,
    accumulating keys forever. Fix: key by session date (dict of sets) so old
    sessions naturally don't match today's keys.
    """
    cfg = make_level_breakout_config('level_breakout_long')
    detector = LevelBreakoutStructure(cfg)

    # Day 1 breakout (same symbol, same PDH level, timestamp on 2026-04-14)
    bars_day1 = make_pdh_breakout_bars()
    df1 = make_df(bars_day1, start_date='2026-04-14 10:00')
    ctx1 = make_context(df1, pdh=100.0, current_price=100.7)
    result1 = detector.detect(ctx1)
    assert len(result1.events) >= 1, (
        f"Day 1 breakout should fire. Got events: {result1.events}, "
        f"rejection: {result1.rejection_reason}"
    )

    # Day 2: same symbol and level — must NOT be blocked by Day 1 dedupe.
    bars_day2 = make_pdh_breakout_bars()
    df2 = make_df(bars_day2, start_date='2026-04-15 10:00')
    ctx2 = make_context(df2, pdh=100.0, current_price=100.7)
    result2 = detector.detect(ctx2)
    assert len(result2.events) >= 1, (
        "Day 2 breakout must NOT be blocked by Day 1 dedupe. "
        "traded_breakouts_today must scope to session date."
    )

    # Bound-check: dedupe store must be dict-keyed by session date (not a
    # single set that grows unboundedly across days). With two distinct
    # sessions, exactly two buckets should exist.
    tracked = detector.traded_breakouts_today
    assert isinstance(tracked, dict), (
        "traded_breakouts_today must be a dict keyed by session date "
        "(not a flat set that accumulates forever across sessions)"
    )
    assert len(tracked) == 2, (
        f"Expected exactly 2 session buckets after Day 1+Day 2 detections; "
        f"got {len(tracked)} buckets: {list(tracked.keys())}"
    )


# ---------------------------------------------------------------------------
# Fix 4: P2 — normalize long/short key names for breakout size
# ---------------------------------------------------------------------------

def test_long_and_short_emit_symmetric_breakout_size_keys():
    """Regression: long and short events must use same key names for size features.

    Bug: long emitted 'breakout_size'/'breakout_size_atr' while short emitted
    'breakdown_size'/'breakdown_size_atr'. Downstream attribution analysis
    would see 2 separate columns instead of 1 unified feature.
    """
    # Use long config for both sides — same class, both paths live in same detector.
    # (The short setup config in configuration.json is a minimal disabled stub
    # missing the dual-mode keys, so we instantiate from the long config which
    # has the full parameter set.)
    cfg_long = make_level_breakout_config('level_breakout_long')
    cfg_short = make_level_breakout_config('level_breakout_long')

    detector_long = LevelBreakoutStructure(cfg_long)
    detector_short = LevelBreakoutStructure(cfg_short)

    bars_long = make_pdh_breakout_bars()
    bars_short = make_pdl_breakdown_bars_short()
    ctx_long = make_context(make_df(bars_long), pdh=100.0, current_price=100.7)
    ctx_short = make_context(make_df(bars_short), pdl=100.0, current_price=99.3)

    result_long = detector_long.detect(ctx_long)
    result_short = detector_short.detect(ctx_short)

    long_events = [e for e in result_long.events if e.structure_type == 'level_breakout_long']
    short_events = [e for e in result_short.events if e.structure_type == 'level_breakout_short']
    assert len(long_events) >= 1, (
        f"Long detector must fire on PDH breakout fixture. "
        f"rejection={result_long.rejection_reason}, events={result_long.events}"
    )
    assert len(short_events) >= 1, (
        f"Short detector must fire on PDL breakdown fixture. "
        f"rejection={result_short.rejection_reason}, events={result_short.events}"
    )

    long_evt = long_events[0]
    short_evt = short_events[0]

    # Unified context key
    assert 'breakout_size_atr' in long_evt.context, "long.context missing breakout_size_atr"
    assert 'breakout_size_atr' in short_evt.context, (
        "short.context must use 'breakout_size_atr' for parity with long "
        "(currently emits 'breakdown_size_atr')."
    )

    # Unified levels key
    assert 'breakout_size' in long_evt.levels, "long.levels missing breakout_size"
    assert 'breakout_size' in short_evt.levels, (
        "short.levels must use 'breakout_size' for parity with long "
        "(currently emits 'breakdown_size')."
    )

    # Values should be positive magnitudes on both sides
    assert short_evt.context['breakout_size_atr'] > 0
    assert short_evt.levels['breakout_size'] > 0
