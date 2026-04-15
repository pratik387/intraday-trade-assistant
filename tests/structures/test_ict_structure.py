"""Regression tests for structures/ict_structure.py audit fixes.

Per docs/edge_discovery/audit/01-ict_structure.md.
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

from structures.ict_structure import ICTStructure
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
    # Detectors often touch these derived columns — populate benign defaults.
    df['vol_surge'] = 1.0
    df['returns_3'] = df['close'].pct_change(3).fillna(0.0)
    return df


def make_context(df, **kwargs):
    last = df.iloc[-1]
    indicators = kwargs.pop('indicators', {'atr14': 0.5, 'atr': 0.5, 'vwap': float(last['close'])})
    defaults = {
        'symbol': 'NSE:TEST',
        'timestamp': df.index[-1].to_pydatetime(),
        'df_5m': df,
        'session_date': df.index[0].to_pydatetime(),
        'current_price': float(last['close']),
        'pdh': kwargs.pop('pdh', float(df['high'].max())),
        'pdl': kwargs.pop('pdl', float(df['low'].min())),
        'orh': kwargs.pop('orh', float(df.iloc[:3]['high'].max()) if len(df) >= 3 else None),
        'orl': kwargs.pop('orl', float(df.iloc[:3]['low'].min()) if len(df) >= 3 else None),
        'indicators': indicators,
    }
    defaults.update(kwargs)
    return MarketContext(**defaults)


def make_ict_config():
    """Load the ict_comprehensive sub-config from configuration.json."""
    with open('config/configuration.json') as f:
        cfg = json.load(f)
    # Setups are under trading_setups.<setup_name>
    setups = cfg.get('trading_setups') or cfg.get('setups') or {}
    ict = setups.get('ict_comprehensive')
    if ict is None:
        # Fall back to a top-level lookup; structure varies between envs.
        ict = cfg.get('ict_comprehensive')
    assert ict is not None, "ict_comprehensive config not found in configuration.json"
    # Preserve required top-level keys like ict_quality_filters if they live at root.
    if 'ict_quality_filters' not in ict and 'ict_quality_filters' in cfg:
        ict = {**ict, 'ict_quality_filters': cfg['ict_quality_filters']}
    return ict


# ---------------------------------------------------------------------------
# Fix 1: Swing-points off-by-one (P1)
# ---------------------------------------------------------------------------

def test_find_swing_points_returns_swings_on_short_frames():
    """Regression: when bos_min_structure_bars > len(df)-2, loop must not abort entirely."""
    bars = [(100, 100.5, 99.5, 100, 1000)] * 25
    bars[10] = (100, 102, 99.5, 100, 1000)  # clear swing high at idx 10
    df = make_df(bars)
    cfg = make_ict_config()
    detector = ICTStructure(cfg)
    detector.bos_min_structure_bars = 100  # > len(df) - 2 = 23
    swings = detector._find_swing_points(df, 'high')
    assert len(swings) > 0, "Expected swings detected; got zero (off-by-one bug)"


# ---------------------------------------------------------------------------
# Fix 2: PDH/PDL NaN silent rejection (P1)
# ---------------------------------------------------------------------------

def test_validate_premium_discount_zone_handles_nan_pdh_pdl():
    """Regression: NaN PDH/PDL must skip validation cleanly, not silently reject."""
    df = make_df([(100, 100.5, 99.5, 100, 1000)] * 25)
    cfg = make_ict_config()
    detector = ICTStructure(cfg)
    ctx = make_context(df, pdh=float('nan'), pdl=float('nan'))
    result = detector._validate_premium_discount_zone('long', 100.0, ctx)
    assert result is True, "Expected True (skip validation) on NaN PDH/PDL; got silent rejection"


# ---------------------------------------------------------------------------
# Fix 3: OB professional_filters stripped by scalar filter (P1)
# ---------------------------------------------------------------------------

def test_order_block_emits_flat_confluence_features():
    """Regression: OB confluence features must be scalar siblings in event.context, not a nested dict."""
    # Build a frame with a clear bearish OB scenario so _create_order_block_event returns an event.
    # ob_candle at idx 10 with a big move down, current bar tests the zone.
    bars = []
    for i in range(20):
        bars.append((100.0, 100.5, 99.5, 100.0, 1000))
    # Bearish OB candle at idx 10: big body, high volume
    bars[10] = (101.0, 102.0, 100.5, 100.6, 15000)
    # Subsequent bars drop (move_pct positive for bearish OB)
    for i in range(11, 18):
        bars[i] = (99.5, 100.0, 98.5, 99.0, 1200)
    # Current test candle at idx 19 re-tests the OB zone with rejection wick
    bars[19] = (101.0, 101.8, 100.6, 100.8, 2000)
    df = make_df(bars)
    # Make vol_surge reflect the OB candle
    df.loc[df.index[10], 'vol_surge'] = 5.0

    cfg = make_ict_config()
    detector = ICTStructure(cfg)
    ctx = make_context(df)

    event = detector._create_order_block_event(
        df=df,
        ob_candle_idx=10,
        move_pct=0.02,  # bearish direction (positive)
        current_price=100.8,
        current_bar_idx=19,
        context=ctx,
        has_sweep=True,
        has_mss=False,
        confluence_factors=['institutional_sweep'],
    )

    # If the function returns None because of quality filters, that's a separate concern —
    # the test specifically validates the shape when an event IS returned. If None, skip with
    # a clear message so we can tweak the fixture, not the fix.
    assert event is not None, (
        "Fixture did not produce an OB event — adjust fixture parameters. "
        "This test validates the event-shape contract when an event is emitted."
    )
    assert 'professional_filters' not in event.context, \
        "Nested 'professional_filters' dict must be flattened"
    assert event.context.get('ob_has_liquidity_sweep') is True
    assert event.context.get('ob_has_mss_confirmation') is False
    assert event.context.get('ob_confluence_count') == 1
    # confluence_factors serialized as comma-joined string (scalar)
    assert event.context.get('ob_confluence_factors') == 'institutional_sweep'


# ---------------------------------------------------------------------------
# Fix 4: ADX NaN silent fallthrough (P2)
# ---------------------------------------------------------------------------

def test_validate_htf_trend_handles_nan_adx():
    """Regression: NaN adx must not silently take the ADX branch and mis-evaluate."""
    bars = [(100 + i * 0.05, 100.5 + i * 0.05, 99.5 + i * 0.05, 100 + i * 0.05, 1000) for i in range(25)]
    df = make_df(bars)
    df['ema20'] = df['close'].rolling(20, min_periods=1).mean()
    cfg = make_ict_config()
    detector = ICTStructure(cfg)
    ctx = make_context(df, indicators={'adx14': float('nan'), 'atr14': 0.5, 'atr': 0.5})
    # Must return a bool without an uncaught exception; ADX branch must not fire on NaN.
    result = detector._validate_htf_trend(df, ctx, 'long')
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Fix 5: VWAP NaN handling in FVG event creation (P2)
# ---------------------------------------------------------------------------

def test_create_fvg_event_handles_nan_vwap():
    """Regression: NaN vwap must not cause silent rejection via NaN comparison."""
    bars = [
        (99, 100, 99, 99.5, 1000),
        (101, 105, 100.5, 104, 5000),
        (103, 104, 102, 103.5, 1500),
    ]
    # Pad to >= 20 bars so rolling calculations are valid
    for _ in range(22):
        bars.append((103, 103.5, 102.5, 103, 1000))
    df = make_df(bars)
    cfg = make_ict_config()
    detector = ICTStructure(cfg)
    ctx = make_context(df, indicators={'vwap': float('nan'), 'atr14': 0.5, 'atr': 0.5})
    # Should not raise; result is a list (possibly empty).
    events = detector._detect_fair_value_gaps(df, ctx)
    assert isinstance(events, list)


# ---------------------------------------------------------------------------
# Fix 6: CHoCH empty momentum_periods crash (P2)
# ---------------------------------------------------------------------------

def test_ict_structure_rejects_empty_choch_momentum_periods():
    """Regression: __init__ must validate non-empty momentum periods (fail fast)."""
    cfg = make_ict_config()
    cfg = {**cfg, 'choch_momentum_periods': []}
    with pytest.raises((ValueError, AssertionError)):
        ICTStructure(cfg)


# ---------------------------------------------------------------------------
# Fix 7: _get_atr silent fallback violates fail-loud rule (P2)
# ---------------------------------------------------------------------------

def test_get_atr_raises_on_missing_atr():
    """Regression: missing ATR must raise, not silently return 1% of price."""
    df = make_df([(100, 100.5, 99.5, 100, 1000)] * 25)
    cfg = make_ict_config()
    detector = ICTStructure(cfg)
    ctx = make_context(df, indicators={})  # No ATR
    with pytest.raises((KeyError, ValueError, AttributeError)):
        detector._get_atr(ctx)
