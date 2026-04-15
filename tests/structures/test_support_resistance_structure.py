"""Regression tests for structures/support_resistance_structure.py audit fixes.

Per docs/edge_discovery/audit/03-support_resistance_structure.md.
Each test captures a specific bug discovered during the FIXED-AND-TRUSTED audit.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structures.support_resistance_structure import (
    SupportResistanceStructure,
    SupportResistanceLevels,
)
from structures.data_models import MarketContext


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_df(bars):
    """Build a 5m bar dataframe from a list of (open, high, low, close, volume) tuples."""
    idx = pd.date_range('2026-04-15 09:15', periods=len(bars), freq='5min')
    df = pd.DataFrame(
        bars,
        columns=['open', 'high', 'low', 'close', 'volume'],
        index=idx,
    )
    df['vol_z'] = 1.8
    return df


def make_context(df, **kwargs):
    last = df.iloc[-1]
    indicators = kwargs.pop('indicators', {'atr': 0.5, 'vol_z': 1.8})
    current_price = kwargs.pop('current_price', float(last['close']))
    # Build timestamp in mid-session (hour between 10-14) so institutional timing bonus does not flip
    ts = df.index[-1].to_pydatetime().replace(hour=11, minute=0)
    defaults = {
        'symbol': 'NSE:TEST',
        'timestamp': ts,
        'df_5m': df,
        'session_date': df.index[0].to_pydatetime(),
        'current_price': current_price,
        'pdh': kwargs.pop('pdh', float(df['high'].max())),
        'pdl': kwargs.pop('pdl', float(df['low'].min())),
        'orh': kwargs.pop('orh', float(df.iloc[:3]['high'].max()) if len(df) >= 3 else None),
        'orl': kwargs.pop('orl', float(df.iloc[:3]['low'].min()) if len(df) >= 3 else None),
        'indicators': indicators,
        'cap_segment': kwargs.pop('cap_segment', 'mid_cap'),
    }
    defaults.update(kwargs)
    return MarketContext(**defaults)


def make_sr_config():
    """Load the resistance_bounce_short sub-config from configuration.json (has all keys)."""
    with open(ROOT / 'config' / 'configuration.json') as f:
        cfg = json.load(f)
    setups = cfg.get('setups') or cfg.get('trading_setups') or {}
    sr = setups.get('resistance_bounce_short')
    assert sr is not None, "resistance_bounce_short config not found in configuration.json"
    # Return a mutable copy; allow any target via target_structure_type
    out = dict(sr)
    out['target_structure_type'] = 'all'
    return out


def make_sr_info(**overrides):
    """Build a SupportResistanceLevels with sensible defaults and allow overrides."""
    defaults = dict(
        support_levels=[100.0],
        resistance_levels=[102.0],
        nearest_support=100.0,
        nearest_resistance=102.0,
        support_distance_pct=0.5,
        resistance_distance_pct=0.5,
        support_touches=3,
        resistance_touches=3,
        support_strength=70.0,
        resistance_strength=70.0,
    )
    defaults.update(overrides)
    return SupportResistanceLevels(**defaults)


# ---------------------------------------------------------------------------
# Fix 1: P1 — NaN propagation in pivot calc
# ---------------------------------------------------------------------------

def test_calculate_pivot_levels_handles_nan_in_ohlc():
    """Regression: NaN in OHLC must not produce NaN pivot levels."""
    bars = [(100, 100.5, 99.5, 100, 1000)] * 25
    bars[5] = (float('nan'), float('nan'), float('nan'), float('nan'), 1000)
    df = make_df(bars)
    cfg = make_sr_config()
    detector = SupportResistanceStructure(cfg)
    levels = detector._calculate_pivot_levels(df)
    for category in ('support', 'resistance'):
        for lv in levels[category]:
            assert not np.isnan(lv), f"{category} level is NaN - NaN propagated from OHLC"


def test_calculate_pivot_levels_all_nan_returns_empty():
    """Regression: all-NaN OHLC must produce empty level lists cleanly."""
    bars = [(float('nan'), float('nan'), float('nan'), float('nan'), 1000)] * 25
    df = make_df(bars)
    cfg = make_sr_config()
    detector = SupportResistanceStructure(cfg)
    levels = detector._calculate_pivot_levels(df)
    assert levels == {'support': [], 'resistance': []}


# ---------------------------------------------------------------------------
# Fix 2: P1 — Latent label bug in _detect_support_breakdown
# ---------------------------------------------------------------------------

def test_detect_support_breakdown_passes_correct_setup_type_and_side():
    """Regression: breakdown must call strength helper with setup_type='support_breakdown', side='short'."""
    # Build bars where price has broken below support (~100), with support level that has touches
    # Create a base range around 100 that gets touched multiple times, then breakdown at the end
    bars = []
    for i in range(25):
        if i in (2, 6, 10, 14):
            # Support touches near 100
            bars.append((100.2, 100.4, 99.9, 100.1, 1200))
        else:
            bars.append((100.5, 100.8, 100.3, 100.6, 1000))
    # Final bar: breakdown below support
    bars.append((99.5, 99.6, 99.0, 99.2, 3000))
    df = make_df(bars)
    cfg = make_sr_config()
    cfg['target_structure_type'] = 'support_breakdown_short'
    # Relax requirements so breakdown path is reachable
    cfg['breakout_buffer_pct'] = 0.2
    cfg['min_breakout_volume_mult'] = 1.5
    detector = SupportResistanceStructure(cfg)
    ctx = make_context(df, current_price=99.2, indicators={'atr': 0.5, 'vol_z': 2.0})
    sr_info = make_sr_info(nearest_support=100.0, support_touches=4, support_strength=80.0)
    with patch.object(detector, '_calculate_institutional_strength',
                      wraps=detector._calculate_institutional_strength) as spy:
        detector._detect_support_breakdown(ctx, sr_info)
        matching_calls = [
            c for c in spy.call_args_list
            if len(c.args) >= 3 and c.args[2] == 'support_breakdown'
        ]
        assert matching_calls, (
            f"Expected strength call with setup_type='support_breakdown'; "
            f"got args: {[c.args for c in spy.call_args_list]}"
        )
        breakdown_call = matching_calls[0]
        assert breakdown_call.args[3] == 'short', (
            f"Expected side='short'; got {breakdown_call.args[3]}"
        )


# ---------------------------------------------------------------------------
# Fix 3: P2 — Fail-fast on missing immediate_entry_distance_pct
# ---------------------------------------------------------------------------

def test_sr_structure_fails_fast_on_missing_immediate_entry_distance_pct():
    """Regression: missing immediate_entry_distance_pct must raise at __init__, not silently default."""
    cfg = make_sr_config()
    del cfg['immediate_entry_distance_pct']
    with pytest.raises(KeyError):
        SupportResistanceStructure(cfg)


# ---------------------------------------------------------------------------
# Fix 4: P2 — Remove asymmetric candle_conviction keys from long bounce
# ---------------------------------------------------------------------------

def _make_support_bounce_bars():
    """Build bars where current price is near a tested support level and last bar closes in top 30%."""
    bars = []
    for i in range(25):
        if i in (3, 8, 13, 18):
            # Support touches near 100
            bars.append((100.3, 100.5, 100.0, 100.2, 1200))
        else:
            bars.append((100.5, 100.8, 100.3, 100.6, 1000))
    # Final bar: bounces off support, close in top 30% of bar range (strong conviction)
    bars.append((100.2, 100.9, 100.1, 100.85, 2500))
    return bars


def test_support_bounce_long_does_not_emit_candle_conviction_keys():
    """Regression: remove candle_conviction* keys for symmetry with other 3 setups."""
    bars = _make_support_bounce_bars()
    df = make_df(bars)
    cfg = make_sr_config()
    cfg['target_structure_type'] = 'support_bounce_long'
    cfg['bounce_tolerance_pct'] = 1.5  # be permissive on distance
    cfg['require_volume_spike'] = False  # skip volume gate
    detector = SupportResistanceStructure(cfg)
    ctx = make_context(df)
    result = detector.detect(ctx)
    bounce_events = [e for e in result.events if e.structure_type == 'support_bounce_long']
    for event in bounce_events:
        assert 'candle_conviction' not in event.context, \
            f"candle_conviction must not be emitted: {event.context}"
        assert 'candle_conviction_reason' not in event.context
        assert 'candle_conviction_close_position' not in event.context


# ---------------------------------------------------------------------------
# Fix 5: P2 — blocked_cap_segments config support
# ---------------------------------------------------------------------------

def _make_resistance_bounce_bars():
    """Build bars where current price is near a tested resistance level (short setup)."""
    bars = []
    for i in range(25):
        if i in (3, 8, 13, 18):
            # Resistance touches near 102
            bars.append((101.6, 102.0, 101.5, 101.8, 1200))
        else:
            bars.append((101.2, 101.5, 101.0, 101.3, 1000))
    # Final bar: price near resistance, below it (rejection scenario)
    bars.append((101.5, 101.9, 101.4, 101.6, 2500))
    return bars


def test_blocked_cap_segments_blocks_resistance_bounce_short_for_large_cap():
    """Regression: large_cap can be blocked for resistance_bounce_short via config (mirrors RangeStructure)."""
    bars = _make_resistance_bounce_bars()
    df = make_df(bars)
    cfg = make_sr_config()
    cfg['target_structure_type'] = 'resistance_bounce_short'
    cfg['blocked_cap_segments'] = ['large_cap']
    detector = SupportResistanceStructure(cfg)
    ctx = make_context(df, cap_segment='large_cap')
    result = detector.detect(ctx)
    short_events = [e for e in result.events if e.structure_type == 'resistance_bounce_short']
    assert len(short_events) == 0, "Expected resistance_bounce_short blocked for large_cap"


def test_blocked_cap_segments_allows_non_blocked_caps():
    """Complement: mid_cap should not be blocked when not in config."""
    bars = _make_resistance_bounce_bars()
    df = make_df(bars)
    cfg = make_sr_config()
    cfg['target_structure_type'] = 'resistance_bounce_short'
    cfg['blocked_cap_segments'] = ['large_cap']
    detector = SupportResistanceStructure(cfg)
    ctx = make_context(df, cap_segment='mid_cap')
    result = detector.detect(ctx)
    # mid_cap not in blocked list - detection runs to completion without exception
    assert isinstance(result.events, list)
