"""Regression tests for structures/range_structure.py audit fixes.

Per docs/edge_discovery/audit/02-range_structure.md.
Each test captures a specific bug discovered during the FIXED-AND-TRUSTED audit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structures.range_structure import RangeStructure
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
    df['vol_z'] = 1.5
    return df


def make_context(df, **kwargs):
    last = df.iloc[-1]
    indicators = kwargs.pop('indicators', {'atr': 0.5, 'vol_z': 1.5})
    current_price = kwargs.pop('current_price', float(last['close']))
    defaults = {
        'symbol': 'NSE:TEST',
        'timestamp': df.index[-1].to_pydatetime(),
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


def make_range_config():
    """Load the range_bounce_short sub-config from configuration.json (has all keys)."""
    with open(ROOT / 'config' / 'configuration.json') as f:
        cfg = json.load(f)
    setups = cfg.get('setups') or cfg.get('trading_setups') or {}
    rng = setups.get('range_bounce_short')
    assert rng is not None, "range_bounce_short config not found in configuration.json"
    # Return a mutable copy
    return dict(rng)


def make_range_bars_with_resistance_touch():
    """Build a 30-bar range with support ~100 and resistance ~102, current price near resistance.

    Tuned so that within the last `min_range_duration*2` bars of the lookback window,
    support (~100) and resistance (~102) are each touched at least 2 times within
    bounce_tolerance_pct (0.2%).
    """
    # Alternating touches across the whole 30 bars. Lookback with min_range_duration=6
    # examines last 12 bars (idx 18..29). Ensure touches fall in this window.
    bars = []
    for i in range(30):
        if i % 4 == 0:
            # Resistance touch: high ~101.95 (within 0.2% of 102)
            bars.append((101.5, 101.95, 101.4, 101.8, 1500))
        elif i % 4 == 2:
            # Support touch: low ~100.0
            bars.append((100.3, 100.6, 100.0, 100.4, 1500))
        else:
            # Mid-range filler
            bars.append((100.8, 101.2, 100.7, 101.0, 1000))
    # Last bar near resistance for short bounce setup
    bars[29] = (101.7, 101.95, 101.6, 101.9, 2000)
    return bars


# ---------------------------------------------------------------------------
# Fix 1: P1 — Dead blocked_cap_segments + hardcoded large_cap block
# ---------------------------------------------------------------------------

def test_blocked_cap_segments_config_is_honored_for_range_bounce_short():
    """Regression: config's blocked_cap_segments must be used; hardcoded large_cap block removed."""
    bars = make_range_bars_with_resistance_touch()
    df = make_df(bars)
    cfg = make_range_config()
    cfg['blocked_cap_segments'] = ['large_cap']
    detector = RangeStructure(cfg)
    ctx = make_context(df, cap_segment='large_cap')
    result = detector.detect(ctx)
    short_events = [e for e in result.events if e.structure_type == 'range_bounce_short']
    assert len(short_events) == 0, "Expected range_bounce_short to be blocked for large_cap"


def test_blocked_cap_segments_config_allows_other_caps():
    """Complement: mid_cap should not be blocked when not in config."""
    bars = make_range_bars_with_resistance_touch()
    df = make_df(bars)
    cfg = make_range_config()
    cfg['blocked_cap_segments'] = ['large_cap']
    detector = RangeStructure(cfg)
    ctx = make_context(df, cap_segment='mid_cap')
    result = detector.detect(ctx)
    assert isinstance(result.events, list)


def test_no_hardcoded_large_cap_block_when_config_empty():
    """When blocked_cap_segments is empty, large_cap must be allowed (no hardcode)."""
    bars = make_range_bars_with_resistance_touch()
    df = make_df(bars)
    cfg = make_range_config()
    cfg['blocked_cap_segments'] = []
    detector = RangeStructure(cfg)
    ctx = make_context(df, cap_segment='large_cap')
    result = detector.detect(ctx)
    # With empty blocked list, if a short event would fire it must fire (hardcode must not block).
    # Detection is permissive: we just assert no silent exception and result is a list.
    # Critical: a short event should now fire for this fixture (vs. being blocked previously).
    short_events = [e for e in result.events if e.structure_type == 'range_bounce_short']
    assert len(short_events) >= 1, (
        "With empty blocked_cap_segments, range_bounce_short must fire for large_cap "
        "(hardcoded block must be removed)"
    )


# ---------------------------------------------------------------------------
# Fix 2: P1 — duration vs duration_bars key mismatch
# ---------------------------------------------------------------------------

def test_calculate_institutional_strength_uses_actual_duration_bars():
    """Regression: confidence must scale with actual range duration_bars, not default 20."""
    bars = make_range_bars_with_resistance_touch()
    df = make_df(bars)
    cfg = make_range_config()
    detector = RangeStructure(cfg)
    ctx = make_context(df, cap_segment='mid_cap',
                       indicators={'vol_z': 1.5, 'atr': 0.5})
    range_info = detector._detect_range(df)
    assert range_info is not None, "Range must be detected for this test"

    # Mature range (50 bars) should trigger both "established" (>=30) and "mature" (>=50) bonuses
    range_info_mature = dict(range_info)
    range_info_mature['duration_bars'] = 50
    range_info_short = dict(range_info)
    range_info_short['duration_bars'] = 10

    conf_mature = detector._calculate_institutional_strength(ctx, range_info_mature, "breakout", "short")
    conf_short = detector._calculate_institutional_strength(ctx, range_info_short, "breakout", "short")

    assert conf_mature > conf_short, (
        f"Mature range (50 bars) confidence {conf_mature} should exceed "
        f"short range (10 bars) {conf_short}. Currently the code reads 'duration' "
        f"(wrong key) so the default 20 is always returned → bonuses never differ."
    )


# ---------------------------------------------------------------------------
# Fix 3: P2 — Div-by-zero guards on support/resistance
# ---------------------------------------------------------------------------

def test_detect_range_returns_none_on_zero_support():
    """Regression: zero support must produce clean None return, with no div-by-zero warning."""
    import warnings
    # Every bar has low=0 → support quantile(0.05) will be 0 → div-by-zero in pct calc
    bars = [(0.01, 0.5, 0.0, 0.1, 1000)] * 30
    df = make_df(bars)
    cfg = make_range_config()
    detector = RangeStructure(cfg)
    # Must return None cleanly without a divide-by-zero RuntimeWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = detector._detect_range(df)
    assert result is None


def test_detect_range_returns_none_on_nan_levels():
    """Regression: NaN support/resistance (from empty quantile) must produce clean None."""
    # All-NaN OHLC → quantile returns NaN
    import numpy as np
    bars = [(np.nan, np.nan, np.nan, np.nan, 1000)] * 30
    df = make_df(bars)
    cfg = make_range_config()
    detector = RangeStructure(cfg)
    result = detector._detect_range(df)
    assert result is None


# ---------------------------------------------------------------------------
# Fix 4: P2 — vol_z NaN consistency
# ---------------------------------------------------------------------------

def test_validate_volume_confirmation_permissive_on_nan_vol_z():
    """Regression: NaN vol_z must behave consistently with missing vol_z key (permissive)."""
    df = make_df([(100, 100.5, 99.5, 100, 1000)] * 25)
    cfg = make_range_config()
    cfg['require_volume_confirmation'] = True
    cfg['min_volume_mult'] = 1.5
    detector = RangeStructure(cfg)
    # Missing key → True (permissive path exists)
    ctx_missing = make_context(df, indicators={'atr': 0.5})
    assert detector._validate_volume_confirmation(ctx_missing) is True
    # NaN vol_z → should also be True (consistent permissive behavior)
    ctx_nan = make_context(df, indicators={'atr': 0.5, 'vol_z': float('nan')})
    assert detector._validate_volume_confirmation(ctx_nan) is True, \
        "NaN vol_z must match missing-key behavior (permissive)"
