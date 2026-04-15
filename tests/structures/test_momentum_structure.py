"""Regression tests for structures/momentum_structure.py audit fixes.

Per docs/edge_discovery/audit/04-momentum_structure.md (SPLIT disposition):
- momentum_breakout_* : FIXED-AND-TRUSTED
- trend_continuation_*: DISABLED

Each test captures a specific bug discovered during the audit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structures.momentum_structure import MomentumStructure
from structures.data_models import MarketContext


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_df(bars):
    """Build a 5m bar dataframe from (open, high, low, close, volume) tuples."""
    idx = pd.date_range("2026-04-15 09:15", periods=len(bars), freq="5min")
    df = pd.DataFrame(
        bars,
        columns=["open", "high", "low", "close", "volume"],
        index=idx,
    )
    return df


def make_context(df, **kwargs):
    last = df.iloc[-1]
    indicators = kwargs.pop("indicators", {"atr14": 0.5, "atr": 0.5})
    defaults = {
        "symbol": "NSE:TEST",
        "timestamp": df.index[-1].to_pydatetime(),
        "df_5m": df,
        "session_date": df.index[0].to_pydatetime(),
        "current_price": float(last["close"]),
        "indicators": indicators,
    }
    defaults.update(kwargs)
    return MarketContext(**defaults)


def make_momentum_config():
    """Load momentum_breakout_long config from configuration.json."""
    cfg_path = ROOT / "config" / "configuration.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    setups = cfg.get("trading_setups") or cfg.get("setups") or {}
    mb = setups.get("momentum_breakout_long")
    assert mb is not None, "momentum_breakout_long not found in configuration.json"
    # Tag the setup name so the detector filters to the configured direction.
    return {**mb, "_setup_name": "momentum_breakout_long"}


def make_momentum_breakout_bars_long():
    """Build 15 bars of strong upward momentum with clear volume surge at the end."""
    bars = []
    price = 100.0
    # First 12 flat-ish bars with baseline volume
    for _ in range(12):
        bars.append((price, price + 0.05, price - 0.05, price, 5000))
    # Last 3 bars: strong upward momentum with volume surge
    for _ in range(3):
        new_price = price * 1.01  # 1% per bar -> ~3% over 3 bars
        bars.append((price, new_price + 0.05, price - 0.02, new_price, 25000))
        price = new_price
    return bars


# ---------------------------------------------------------------------------
# Fix 1 (P1): NaN vol_surge must fail the gate
# ---------------------------------------------------------------------------


def test_momentum_breakout_rejects_nan_vol_surge(monkeypatch):
    """Regression: NaN vol_surge must fail the volume-surge gate, not silently bypass.

    Per audit/04-momentum_structure.md P1 #1.
    Bug: last_bar.get('vol_surge', 1.0) returns NaN (not fallback) when key exists
    with NaN value. NaN < threshold is False -> gate passes silently.

    To isolate the vol_surge gate we build bars that pass every other gate
    (momentum thresholds + vol_z) and then patch the indicator calculation to
    force vol_surge=NaN on the last bar only.
    """
    bars = make_momentum_breakout_bars_long()
    df = make_df(bars)
    cfg = make_momentum_config()
    detector = MomentumStructure(cfg)

    original = detector._calculate_momentum_indicators

    def _patched(frame: pd.DataFrame) -> pd.DataFrame:
        out = original(frame)
        if out is not None:
            out.loc[out.index[-1], "vol_surge"] = np.nan
        return out

    monkeypatch.setattr(detector, "_calculate_momentum_indicators", _patched)

    ctx = make_context(df)
    result = detector.detect(ctx)
    momentum_events = [
        e for e in result.events if e.structure_type.startswith("momentum_breakout")
    ]
    assert len(momentum_events) == 0, (
        f"Expected 0 momentum events with NaN vol_surge, got {len(momentum_events)}"
    )


# ---------------------------------------------------------------------------
# Fix 2+3 (P1): trend_continuation_long/short DISABLED in pipeline configs
# ---------------------------------------------------------------------------


def test_trend_continuation_setups_are_disabled_in_pipeline_config():
    """Regression: trend_continuation_long/short must be enabled:false per audit/04 P1 #2."""
    configs_to_check = [
        ROOT / "config" / "pipelines" / "breakout_config.json",
        ROOT / "config" / "pipelines" / "momentum_config.json",
    ]
    found_any = False
    for cfg_path in configs_to_check:
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        sf = (
            cfg.get("gates", {}).get("setup_filters", {})
            or cfg.get("setup_filters", {})
        )
        for setup_name in ("trend_continuation_long", "trend_continuation_short"):
            if setup_name in sf:
                assert sf[setup_name].get("enabled", True) is False, (
                    f"{setup_name} must be enabled:false in {cfg_path}"
                )
                found_any = True
    assert found_any, (
        "Expected trend_continuation_long/short to be explicitly disabled in at "
        "least one pipeline config (breakout_config.json or momentum_config.json)"
    )


# ---------------------------------------------------------------------------
# Fix 5 (P3): emit two_bar_sum_pct in momentum_breakout context for Stage 3 attribution
# ---------------------------------------------------------------------------


def test_momentum_breakout_long_emits_two_bar_sum_pct():
    """Regression: momentum_breakout_long must emit two_bar_sum_pct feature.

    Per audit/04-momentum_structure.md P3 #11 (optional enhancement).
    The 2-bar cumulative momentum is a gate threshold but was not emitted as a
    feature. Adding it enables downstream edge-attribution analysis in gauntlet Stage 3.
    """
    bars = make_momentum_breakout_bars_long()
    df = make_df(bars)
    cfg = make_momentum_config()
    detector = MomentumStructure(cfg)
    ctx = make_context(df)
    result = detector.detect(ctx)
    long_events = [
        e for e in result.events if e.structure_type == "momentum_breakout_long"
    ]
    assert len(long_events) >= 1, "Expected momentum_breakout_long to fire on strong up bars"
    ev = long_events[0]
    assert "two_bar_sum_pct" in ev.context, (
        "momentum_breakout_long must emit 'two_bar_sum_pct' in event.context"
    )
    # Verify value matches df['returns_1'].tail(2).sum() * 100
    df_calc = df.copy()
    df_calc["returns_1"] = df_calc["close"].pct_change()
    expected = float(df_calc["returns_1"].tail(2).sum() * 100)
    assert abs(ev.context["two_bar_sum_pct"] - expected) < 0.01, (
        f"two_bar_sum_pct={ev.context['two_bar_sum_pct']} vs expected={expected}"
    )


def test_momentum_breakout_short_emits_two_bar_sum_pct():
    """Regression: momentum_breakout_short must also emit two_bar_sum_pct."""
    bars = []
    price = 100.0
    for _ in range(12):
        bars.append((price, price + 0.05, price - 0.05, price, 5000))
    for _ in range(3):
        new_price = price * 0.99  # -1% per bar
        bars.append((price, price + 0.02, new_price - 0.05, new_price, 25000))
        price = new_price
    df = make_df(bars)
    cfg = make_momentum_config()
    # Use short-directional config
    cfg["_setup_name"] = "momentum_breakout_short"
    detector = MomentumStructure(cfg)
    ctx = make_context(df)
    result = detector.detect(ctx)
    short_events = [
        e for e in result.events if e.structure_type == "momentum_breakout_short"
    ]
    assert len(short_events) >= 1, "Expected momentum_breakout_short to fire on strong down bars"
    ev = short_events[0]
    assert "two_bar_sum_pct" in ev.context, (
        "momentum_breakout_short must emit 'two_bar_sum_pct' in event.context"
    )
