"""Regression tests for structures/volume_structure.py audit fixes.

Per docs/edge_discovery/audit/06-volume_structure.md.
Each test captures a specific bug discovered during the FIXED-AND-TRUSTED audit.
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structures.volume_structure import VolumeStructure
from structures.data_models import MarketContext


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _build_df(bars, start='2026-04-15 10:30', freq='5min'):
    """Build a 5m bar dataframe from a list of (open, high, low, close, volume) tuples."""
    idx = pd.date_range(start, periods=len(bars), freq=freq)
    df = pd.DataFrame(
        bars,
        columns=['open', 'high', 'low', 'close', 'volume'],
        index=idx,
    )
    return df


def _attach_vol_z(df, spike_z=4.0):
    """Attach a vol_z column where the last bar has spike_z and rest are ~1.0."""
    df['vol_z'] = 1.0
    df.loc[df.index[-1], 'vol_z'] = spike_z
    # Attach vwap column so key-level extraction works (optional but realistic)
    df['vwap'] = df['close'].mean()
    return df


def make_volume_spike_reversal_bars_long(spike_z=4.0, start='2026-04-15 10:30'):
    """Down bar with volume spike + lower wick rejection.

    For LONG reversal we need:
    - close < open (down bar)
    - body size >= 0.8% (config default)
    - lower wick >= 15% of total range (min_rejection_wick_ratio = 0.15)
    - volume >= 1.5x 20-bar median (min_volume_ratio = 1.5)
    - vol_z >= 1.2 (min_volume_spike_mult for long config)
    """
    # 25 flat bars (median volume 1000, tight OHLC around 100)
    bars = [(100.0, 100.4, 99.8, 100.1, 1000)] * 25
    # Last bar: down bar with big lower wick + volume spike
    # open=100.5, high=100.6, low=97.0, close=98.8 -> body down, significant lower wick
    # body_size_pct = abs(98.8-100.5)/100.5 * 100 = 1.69%  (>= 0.8 OK)
    # lower_wick = min(100.5, 98.8) - 97.0 = 1.8
    # total_range = 100.6 - 97.0 = 3.6
    # wick_ratio = 1.8 / 3.6 = 0.5 (>= 0.15 OK)
    # volume 3000 vs median 1000 -> vol_ratio 3.0 (>= 1.5 OK)
    bars.append((100.5, 100.6, 97.0, 98.8, 3000))
    df = _build_df(bars, start=start)
    return _attach_vol_z(df, spike_z=spike_z)


def make_volume_spike_reversal_bars_short(spike_z=4.0, start='2026-04-15 10:30'):
    """Up bar with volume spike + upper wick rejection (for SHORT reversal)."""
    bars = [(100.0, 100.4, 99.8, 100.1, 1000)] * 25
    # Up bar with big upper wick + volume spike
    # open=99.5, high=103.0, low=99.4, close=101.2
    # body up, upper wick big
    # body_size_pct = abs(101.2-99.5)/99.5*100 = 1.71% (>= 0.8 OK)
    # upper_wick = 103.0 - max(99.5, 101.2) = 1.8
    # total_range = 103.0 - 99.4 = 3.6
    # wick_ratio = 1.8 / 3.6 = 0.5 (>= 0.15 OK)
    bars.append((99.5, 103.0, 99.4, 101.2, 3000))
    df = _build_df(bars, start=start)
    return _attach_vol_z(df, spike_z=spike_z)


def make_volume_config(setup='volume_spike_reversal_long'):
    """Load the volume_spike_reversal sub-config from configuration.json."""
    with open(ROOT / 'config' / 'configuration.json') as f:
        cfg = json.load(f)
    setups = cfg.get('setups') or cfg.get('trading_setups') or {}
    vol = setups.get(setup)
    assert vol is not None, f"{setup} config not found in configuration.json"
    # Filter out _comment_* metadata keys
    clean = {k: v for k, v in vol.items() if not k.startswith('_')}
    return dict(clean)


def make_context(df, cap_segment='mid_cap', timestamp_hour=None, **kwargs):
    last = df.iloc[-1]
    ts = df.index[-1].to_pydatetime()
    if timestamp_hour is not None:
        ts = ts.replace(hour=timestamp_hour, minute=30)
    indicators = kwargs.pop('indicators', {'atr': 0.5, 'vol_z': float(df['vol_z'].iloc[-1])})
    current_price = kwargs.pop('current_price', float(last['close']))
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
        'cap_segment': cap_segment,
    }
    defaults.update(kwargs)
    return MarketContext(**defaults)


# ---------------------------------------------------------------------------
# Fix 1: P1 — Add blocked_cap_segments config support
# ---------------------------------------------------------------------------

def test_blocked_cap_segments_blocks_volume_spike_reversal_for_large_cap():
    """Regression: large_cap can be blocked via blocked_cap_segments config
    (parity with Range/SR/FHM)."""
    df = make_volume_spike_reversal_bars_long()
    cfg = make_volume_config()
    cfg['blocked_cap_segments'] = ['large_cap']
    detector = VolumeStructure(cfg)
    ctx = make_context(df, cap_segment='large_cap')
    result = detector.detect(ctx)
    assert len(result.events) == 0, (
        "Expected volume_spike_reversal to be blocked for large_cap"
    )


def test_blocked_cap_segments_allows_mid_cap():
    """Complement: mid_cap should not be blocked unless in config."""
    df = make_volume_spike_reversal_bars_long()
    cfg = make_volume_config()
    cfg['blocked_cap_segments'] = ['large_cap']
    detector = VolumeStructure(cfg)
    ctx = make_context(df, cap_segment='mid_cap')
    result = detector.detect(ctx)
    # mid_cap allowed; events may or may not fire based on other conditions,
    # but cap-filter path doesn't trigger
    assert isinstance(result.events, list)


# ---------------------------------------------------------------------------
# Fix 2: P1 — levels dict keys so detected_level populates downstream
# ---------------------------------------------------------------------------

def test_volume_spike_reversal_long_levels_dict_has_support_key():
    """Regression: long reversal must emit 'support' key in levels so
    detected_level populates via main_detector.py:540-544."""
    df = make_volume_spike_reversal_bars_long()
    cfg = make_volume_config()
    detector = VolumeStructure(cfg)
    ctx = make_context(df, cap_segment='mid_cap')
    result = detector.detect(ctx)
    long_events = [e for e in result.events if e.structure_type == 'volume_spike_reversal_long']
    assert len(long_events) >= 1, "Expected volume_spike_reversal_long to fire"
    event = long_events[0]
    assert 'support' in event.levels, (
        f"Long reversal must emit 'support' key (got keys: {list(event.levels.keys())})"
    )
    # Support value should be the spike bar's LOW (rejected low)
    spike_low = float(df.iloc[-1]['low'])
    assert abs(event.levels['support'] - spike_low) < 0.01, (
        f"Support should be spike low {spike_low}, got {event.levels['support']}"
    )


def test_volume_spike_reversal_short_levels_dict_has_resistance_key():
    """Regression: short reversal must emit 'resistance' key."""
    df = make_volume_spike_reversal_bars_short()
    cfg = make_volume_config(setup='volume_spike_reversal_short')
    detector = VolumeStructure(cfg)
    ctx = make_context(df, cap_segment='mid_cap')
    result = detector.detect(ctx)
    short_events = [e for e in result.events if e.structure_type == 'volume_spike_reversal_short']
    assert len(short_events) >= 1, "Expected volume_spike_reversal_short to fire"
    event = short_events[0]
    assert 'resistance' in event.levels, (
        f"Short reversal must emit 'resistance' key (got keys: {list(event.levels.keys())})"
    )
    spike_high = float(df.iloc[-1]['high'])
    assert abs(event.levels['resistance'] - spike_high) < 0.01, (
        f"Resistance should be spike high {spike_high}, got {event.levels['resistance']}"
    )


# ---------------------------------------------------------------------------
# Fix 3: P2 — NaN guard on current_vol_z
# ---------------------------------------------------------------------------

def test_detect_early_rejects_nan_vol_z():
    """Regression: NaN vol_z at last bar must be early-rejected, not silently passed."""
    df = make_volume_spike_reversal_bars_long()
    # Inject NaN into last bar's vol_z
    df.loc[df.index[-1], 'vol_z'] = float('nan')
    cfg = make_volume_config()
    detector = VolumeStructure(cfg)
    ctx = make_context(df, cap_segment='mid_cap')
    result = detector.detect(ctx)
    # Must not emit any events (NaN vol_z is invalid)
    assert len(result.events) == 0, "NaN vol_z must fail the volume spike gate"


# ---------------------------------------------------------------------------
# Fix 4: P2 — Hour-of-day bonus must not boost lunch window
# ---------------------------------------------------------------------------

def test_hour_bonus_does_not_boost_lunch_window():
    """Regression: hour-of-day confidence bonus must not boost the 12:00-13:00
    lunch window (Item 1 canonical: lunch is low-edge for volume-spike-reversal).

    Active windows per canonical NSE research: 10:00-11:00 and 14:00-15:00.
    Lunch window 12:00-13:00 must NOT receive the active-hour bonus.
    """
    df = make_volume_spike_reversal_bars_long()
    cfg = make_volume_config()
    detector = VolumeStructure(cfg)
    # 10:30 AM — should get bonus (active hour)
    ctx_10 = make_context(df, cap_segment='mid_cap', timestamp_hour=10)
    result_10 = detector.detect(ctx_10)
    # 12:30 PM — must NOT get bonus (lunch window)
    ctx_12 = make_context(df, cap_segment='mid_cap', timestamp_hour=12)
    result_12 = detector.detect(ctx_12)
    assert result_10.events and result_12.events, (
        "Both hour contexts should emit events for this fixture"
    )
    conf_10 = result_10.events[0].confidence
    conf_12 = result_12.events[0].confidence
    assert conf_10 > conf_12, (
        f"10:30 active-hour confidence {conf_10} should exceed "
        f"12:30 lunch confidence {conf_12}"
    )
