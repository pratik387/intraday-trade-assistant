"""Daily regime detector needs adequate history — characterizes the live bug.

screener_live fetched the index daily with days=30; DailyRegimeDetector bails to
'chop'/insufficient_data below 50 bars (and wants MIN_BARS_REQUIRED=210 for
EMA200 + a 100-bar BB-width window). Fed 30, the daily regime layer was inert —
squeeze/trend could never fire for the broad-market gate. These tests lock the
bar-count contract the screener fix must honor.
"""
import numpy as np
import pandas as pd

from services.gates.multi_timeframe_regime import DailyRegimeDetector


def _daily(n, seed=0):
    rng = np.random.RandomState(seed)
    close = 1000 + np.cumsum(rng.normal(0, 5, n))
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": close, "high": close + 5, "low": close - 5,
        "close": close, "volume": 1e6,
    }, index=idx)


def test_thirty_bars_is_insufficient_returns_chop():
    res = DailyRegimeDetector().classify(_daily(30))
    assert res.regime == "chop"
    assert res.metrics.get("error") == "insufficient_data"


def test_min_bars_required_window_classifies():
    n = DailyRegimeDetector.MIN_BARS_REQUIRED
    res = DailyRegimeDetector().classify(_daily(n))
    # With a full window it actually classifies — not the insufficient default.
    assert res.metrics.get("error") != "insufficient_data"
    assert res.regime in ("trend_up", "trend_down", "chop", "squeeze")


def test_min_bars_required_is_enough_for_ema200():
    # The fetch in screener_live is now driven by this constant; guard it stays
    # above the EMA200 + the 50-bar hard floor.
    assert DailyRegimeDetector.MIN_BARS_REQUIRED >= 200
