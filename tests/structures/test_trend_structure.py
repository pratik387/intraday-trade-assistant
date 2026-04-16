"""Regression tests for structures/trend_structure.py audit fixes.

Per docs/edge_discovery/audit/08-trend_structure.md.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structures.trend_structure import TrendStructure
from structures.data_models import MarketContext


def _build_df(bars, start='2026-04-15 10:30', freq='5min'):
    idx = pd.date_range(start, periods=len(bars), freq=freq)
    return pd.DataFrame(
        bars,
        columns=['open', 'high', 'low', 'close', 'volume'],
        index=idx,
    )


def make_trend_config():
    """Load trend_pullback_long config block from configuration.json + add required keys."""
    with open(ROOT / 'config' / 'configuration.json') as f:
        cfg = json.load(f)
    setups = cfg.get('setups') or cfg.get('trading_setups') or {}
    tr = setups.get('trend_pullback_long')
    assert tr is not None, "trend_pullback_long config missing"
    clean = {k: v for k, v in tr.items() if not k.startswith('_')}
    defaults = {
        'min_trend_strength': 30.0,
        'min_trend_bars': 25,
        'max_pullback_pct': 1.5,
        'min_pullback_pct': 0.3,
        'require_volume_confirmation': False,
        'min_volume_mult': 1.0,
        'min_momentum_score': 30.0,
        'require_volume_decline': True,
        'max_volume_mult_during_pullback': 0.8,
        'require_rsi_oversold': False,
        'rsi_oversold_threshold': 35.0,
        'require_rsi_overbought': False,
        'rsi_overbought_threshold': 65.0,
        'min_stop_distance_pct': 0.3,
        'stop_distance_mult': 2.0,
        'target_mult_t1': 1.5,
        'target_mult_t2': 2.5,
        'swing_sl_buffer_atr': 0.5,
        'swing_lookback_bars': 20,
        'confidence_strong_trend': 0.85,
        'confidence_weak_trend': 0.55,
        'continuation_momentum_boost': 10,
    }
    for k, v in defaults.items():
        clean.setdefault(k, v)
    return clean


def make_context(df, cap_segment='mid_cap', timestamp_hour=None, **kwargs):
    last = df.iloc[-1]
    ts = df.index[-1].to_pydatetime()
    if timestamp_hour is not None:
        ts = ts.replace(hour=timestamp_hour, minute=30)
    indicators = kwargs.pop('indicators', {'atr': 1.0})
    current_price = kwargs.pop('current_price', float(last['close']))
    defaults = {
        'symbol': 'NSE:TEST',
        'timestamp': ts,
        'df_5m': df,
        'session_date': df.index[0].to_pydatetime(),
        'current_price': current_price,
        'pdh': kwargs.pop('pdh', float(df['high'].max())),
        'pdl': kwargs.pop('pdl', float(df['low'].min())),
        'orh': kwargs.pop('orh', float(df.iloc[:3]['high'].max())),
        'orl': kwargs.pop('orl', float(df.iloc[:3]['low'].min())),
        'indicators': indicators,
        'cap_segment': cap_segment,
    }
    defaults.update(kwargs)
    return MarketContext(**defaults)


# ---------------------------------------------------------------------------
# Fix 1 (P1): blocked_cap_segments fast-fail
# ---------------------------------------------------------------------------

def test_blocked_cap_segments_blocks_trend_for_large_cap():
    """Regression: large_cap can be blocked via blocked_cap_segments config (parity)."""
    df = _build_df([(100.0, 100.5, 99.5, 100.0, 1000)] * 30)
    cfg = make_trend_config()
    cfg['blocked_cap_segments'] = ['large_cap']
    detector = TrendStructure(cfg)
    ctx = make_context(df, cap_segment='large_cap')
    result = detector.detect(ctx)
    assert len(result.events) == 0
    assert 'large_cap' in (result.rejection_reason or '') or 'blocked' in (result.rejection_reason or '').lower()


def test_blocked_cap_segments_allows_mid_cap():
    """Complement: mid_cap not blocked unless in config."""
    df = _build_df([(100.0, 100.5, 99.5, 100.0, 1000)] * 30)
    cfg = make_trend_config()
    cfg['blocked_cap_segments'] = ['large_cap']
    detector = TrendStructure(cfg)
    ctx = make_context(df, cap_segment='mid_cap')
    result = detector.detect(ctx)
    if result.rejection_reason:
        assert 'cap_segment' not in result.rejection_reason


# ---------------------------------------------------------------------------
# Fix 2 (P1): levels dict has support/resistance for detected_level flow
# ---------------------------------------------------------------------------

def test_trend_pullback_long_emits_support_key_when_event_fires():
    """If a long trend pullback event fires, it MUST include 'support' key in levels.

    This is a code-path verification — we test the detector's internal _detect_trend_pullback
    method by injecting a mock TrendInfo, since constructing real bar data that satisfies
    all 6 filters is fragile.
    """
    from structures.trend_structure import TrendInfo

    df = _build_df([(100.0, 100.5, 99.5, 100.0, 1000)] * 30)
    df['vol_z'] = 0.5  # low vol_z (volume decline)
    cfg = make_trend_config()
    cfg['require_rsi_oversold'] = False  # remove RSI gate for this test
    cfg['require_volume_decline'] = False
    cfg['min_pullback_pct'] = 0.0  # accept any pullback
    cfg['max_pullback_pct'] = 100.0
    cfg['min_momentum_score'] = 0.0  # no momentum gate
    detector = TrendStructure(cfg)
    ctx = make_context(df, cap_segment='mid_cap', indicators={'atr': 1.0, 'vol_z': 0.5})
    # Inject TrendInfo directly
    trend_info = TrendInfo(
        trend_direction="up",
        trend_strength=80.0,
        pullback_depth_pct=0.5,
        momentum_score=50.0,
        trend_age_bars=10,
        trend_quality=70.0,
    )
    events, _ = detector._detect_trend_pullback(ctx, trend_info)
    assert len(events) >= 1, "Expected long pullback event with relaxed gates"
    assert 'support' in events[0].levels, \
        f"Long pullback must emit 'support' key (got: {list(events[0].levels.keys())})"


def test_trend_pullback_short_emits_resistance_key_when_event_fires():
    """Symmetric short test."""
    from structures.trend_structure import TrendInfo

    df = _build_df([(100.0, 100.5, 99.5, 100.0, 1000)] * 30)
    df['vol_z'] = 0.5
    cfg = make_trend_config()
    cfg['require_rsi_oversold'] = False
    cfg['require_rsi_overbought'] = False
    cfg['require_volume_decline'] = False
    cfg['min_pullback_pct'] = 0.0
    cfg['max_pullback_pct'] = 100.0
    cfg['min_momentum_score'] = 0.0
    detector = TrendStructure(cfg)
    ctx = make_context(df, cap_segment='mid_cap', indicators={'atr': 1.0, 'vol_z': 0.5})
    trend_info = TrendInfo(
        trend_direction="down",
        trend_strength=80.0,
        pullback_depth_pct=0.5,
        momentum_score=50.0,
        trend_age_bars=10,
        trend_quality=70.0,
    )
    events, _ = detector._detect_trend_pullback(ctx, trend_info)
    assert len(events) >= 1, "Expected short pullback event with relaxed gates"
    assert 'resistance' in events[0].levels


# ---------------------------------------------------------------------------
# Verify trend_continuation pipeline-level disable from audit/04 still in place
# ---------------------------------------------------------------------------

def test_trend_continuation_disabled_per_audit_04():
    """Regression: trend_continuation_long/_short must remain enabled:false in
    config/pipelines/momentum_config.json per audit/04 disposition.

    Both MomentumStructure AND TrendStructure emit these setup types. The
    pipeline-level disable correctly catches both detectors' emissions.
    """
    with open(ROOT / 'config' / 'pipelines' / 'momentum_config.json') as f:
        cfg = json.load(f)
    sf = cfg.get('gates', {}).get('setup_filters', {}) or cfg.get('setup_filters', {})
    for setup in ('trend_continuation_long', 'trend_continuation_short'):
        assert sf.get(setup, {}).get('enabled') is False, \
            f"{setup} must remain DISABLED per audit/04 (catches both Momentum and Trend emissions)"
