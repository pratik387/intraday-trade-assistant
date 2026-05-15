"""Stage 5d tests: conviction gate replay."""
from datetime import date, datetime

import pandas as pd
import pytest

from tools.edge_discovery_legacy_gauntlet.stages.stage5d_conviction_simulation import (
    simulate_conviction_filter,
)


class _FakeScorer:
    """Mock scorer that returns fixed per-symbol scores."""
    def __init__(self, scores):
        self.scores = scores
        self.features = ["dummy"]

    def predict(self, feat_or_frame):
        if isinstance(feat_or_frame, dict):
            return self.scores.get(feat_or_frame.get("symbol"), 0.0)
        raise TypeError


def _trade(symbol, ts, pnl, r, session=None):
    return {
        "symbol": symbol,
        "setup_type": "premium_zone_short",
        "regime": "chop",
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
        "decision_ts": ts,
        "session_date_dt": session or date(2025, 1, 2),
        "minute_of_day": 600,
        "total_trade_pnl": pnl,
        "r_multiple": r,
    }


def test_daily_cap_enforced():
    """With cap=2, only 2 trades per session admitted."""
    trades = pd.DataFrame([
        _trade("A", datetime(2025, 1, 2, 10, 0), 100, 1.0),
        _trade("B", datetime(2025, 1, 2, 10, 5), 100, 1.0),
        _trade("C", datetime(2025, 1, 2, 10, 10), 100, 1.0),
    ])
    scorer = _FakeScorer({"A": 0.9, "B": 0.8, "C": 0.7})
    cfg = {"enabled": True, "daily_cap": 2, "min_predicted_r": 0.0}
    result = simulate_conviction_filter(trades, scorer, cfg)
    admitted = result[result["admitted"] == True]
    # All three score above threshold — but cap is 2
    assert len(admitted) == 2


def test_threshold_enforced():
    """Low-score candidates rejected."""
    trades = pd.DataFrame([
        _trade("A", datetime(2025, 1, 2, 10, 0), 100, 1.0),
        _trade("B", datetime(2025, 1, 2, 10, 5), 100, 1.0),
    ])
    scorer = _FakeScorer({"A": 0.9, "B": 0.1})
    cfg = {"enabled": True, "daily_cap": 50, "min_predicted_r": 0.5}
    result = simulate_conviction_filter(trades, scorer, cfg)
    admitted = result[result["admitted"] == True]
    assert len(admitted) == 1
    assert admitted.iloc[0]["symbol"] == "A"


def test_session_boundary_resets_cap():
    """Cap resets between sessions."""
    trades = pd.DataFrame([
        _trade("A", datetime(2025, 1, 2, 10, 0), 100, 1.0, session=date(2025, 1, 2)),
        _trade("B", datetime(2025, 1, 2, 10, 5), 100, 1.0, session=date(2025, 1, 2)),
        _trade("C", datetime(2025, 1, 3, 10, 0), 100, 1.0, session=date(2025, 1, 3)),
    ])
    scorer = _FakeScorer({"A": 0.9, "B": 0.8, "C": 0.7})
    cfg = {"enabled": True, "daily_cap": 1, "min_predicted_r": 0.0}
    result = simulate_conviction_filter(trades, scorer, cfg)
    admitted = result[result["admitted"] == True]
    # 1 admitted on day 1, 1 on day 2 = 2 total
    assert len(admitted) == 2


def test_admitted_column_preserves_chronological_order():
    """Output trades preserve input order; admitted is a column, not reordering."""
    trades = pd.DataFrame([
        _trade("A", datetime(2025, 1, 2, 10, 0), 100, 1.0),
        _trade("B", datetime(2025, 1, 2, 10, 5), 100, 1.0),
    ])
    scorer = _FakeScorer({"A": 0.9, "B": 0.1})
    cfg = {"enabled": True, "daily_cap": 5, "min_predicted_r": 0.5}
    result = simulate_conviction_filter(trades, scorer, cfg)
    assert list(result["symbol"]) == ["A", "B"]
    assert list(result["admitted"]) == [True, False]
