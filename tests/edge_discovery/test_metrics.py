"""Tests for edge metrics (PF, Sharpe, DD, WR, expectancy)."""
import math
import numpy as np
import pandas as pd
import pytest

from datetime import date

from tools.edge_discovery.metrics import (
    profit_factor,
    win_rate_pct,
    sharpe_ratio,
    session_sharpe_ratio,
    max_drawdown_pct,
    expectancy,
    summary_stats,
)


def test_profit_factor_basic():
    # Winners sum = 300, losers sum = -150 → PF = 2.0
    pnl = pd.Series([100, 200, -50, -100])
    assert profit_factor(pnl) == pytest.approx(2.0)


def test_profit_factor_all_losers():
    pnl = pd.Series([-10, -20, -30])
    assert profit_factor(pnl) == 0.0


def test_profit_factor_all_winners():
    pnl = pd.Series([10, 20, 30])
    assert profit_factor(pnl) == float("inf")


def test_profit_factor_empty():
    assert profit_factor(pd.Series([], dtype=float)) == 0.0


def test_win_rate_basic():
    pnl = pd.Series([10, -5, 20, -3, 0])
    # 2 winners out of 5 = 40%. Zero PnL does NOT count as winner.
    assert win_rate_pct(pnl) == pytest.approx(40.0)


def test_sharpe_basic():
    # Constant positive returns → high Sharpe; use per-trade scale (no annualization)
    pnl = pd.Series([100, 100, 100, 100])
    assert sharpe_ratio(pnl) == float("inf")  # zero std


def test_sharpe_mixed():
    pnl = pd.Series([100, -50, 80, -30, 120])
    mean = pnl.mean()
    std = pnl.std(ddof=1)
    expected = mean / std if std > 0 else float("inf")
    assert sharpe_ratio(pnl) == pytest.approx(expected)


def test_max_drawdown_pct():
    # Equity curve: cumsum = [100, 300, 200, 400, 100]
    # Running peak:           [100, 300, 300, 400, 400]
    # Drawdown:               [  0,   0,-100,   0,-300]  → max_dd_abs = 300
    # Net PnL = sum = 100  → max_dd_pct = 300 / 100 * 100 = 300%
    pnl = pd.Series([100, 200, -100, 200, -300])
    assert max_drawdown_pct(pnl) == pytest.approx(300.0)


def test_max_drawdown_no_drawdown():
    pnl = pd.Series([100, 100, 100])
    assert max_drawdown_pct(pnl) == 0.0


def test_expectancy():
    # WR=50%, avg_win=100, avg_loss=-50. Expectancy = 0.5*100 + 0.5*(-50) = 25
    pnl = pd.Series([100, -50, 100, -50])
    assert expectancy(pnl) == pytest.approx(25.0)


def test_summary_stats_full():
    pnl = pd.Series([100, -50, 200, -30, 50])
    s = summary_stats(pnl)
    assert s["n"] == 5
    assert s["wr_pct"] == pytest.approx(60.0)
    assert s["pf"] == pytest.approx(350 / 80)
    assert s["avg_pnl"] == pytest.approx(54.0)
    assert s["total_pnl"] == pytest.approx(270.0)
    assert "sharpe" in s
    assert "max_dd_pct" in s


def test_summary_stats_empty():
    s = summary_stats(pd.Series([], dtype=float))
    assert s["n"] == 0
    assert s["pf"] == 0.0
    assert s["wr_pct"] == 0.0


def test_session_sharpe_aggregates_per_day_before_computing():
    """Session Sharpe sums pnl per session date, then computes mean/std on the
    daily series. For a setup that trades every day with consistent daily PnL
    but high per-trade variance, per-trade Sharpe is low but session Sharpe is
    high — this is the whole point of the metric."""
    # 3 sessions × 100 trades each. Per-trade: [+100, -50] cycle → high variance.
    # Daily sum: 100*(50*100 - 50*50) = 2500 per session. Zero std → inf.
    pnl = pd.Series([100, -50] * 150)
    sessions = pd.Series(
        [date(2023, 1, 1)] * 100 + [date(2023, 1, 2)] * 100 + [date(2023, 1, 3)] * 100
    )
    assert session_sharpe_ratio(pnl, sessions) == float("inf")


def test_session_sharpe_varied_daily_pnl():
    # Daily sums: [+1000, +500, +1500]. mean=1000, std=500 (ddof=1). Sharpe=2.0
    pnl = pd.Series([1000, 500, 1500])
    sessions = pd.Series([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)])
    assert session_sharpe_ratio(pnl, sessions) == pytest.approx(2.0)


def test_session_sharpe_single_session_returns_zero():
    """Single-session data has undefined daily std (n=1, ddof=1) → return 0.0."""
    pnl = pd.Series([100, -50, 200])
    sessions = pd.Series([date(2023, 1, 1)] * 3)
    assert session_sharpe_ratio(pnl, sessions) == 0.0


def test_session_sharpe_empty():
    assert session_sharpe_ratio(pd.Series([], dtype=float), pd.Series([], dtype=object)) == 0.0


def test_session_sharpe_length_mismatch_raises():
    with pytest.raises(ValueError):
        session_sharpe_ratio(pd.Series([1, 2, 3]), pd.Series([date(2023, 1, 1)]))


def test_summary_stats_with_session_dates_adds_session_sharpe():
    pnl = pd.Series([1000, 500, 1500])
    sessions = pd.Series([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)])
    s = summary_stats(pnl, session_dates=sessions)
    assert "session_sharpe" in s
    assert s["session_sharpe"] == pytest.approx(2.0)
    # Per-trade sharpe still present for reference
    assert "sharpe" in s


def test_summary_stats_without_session_dates_omits_session_sharpe():
    """Backwards compat: callers that don't pass session_dates get no
    session_sharpe key (Stage 1 / Stage 3 informational use)."""
    s = summary_stats(pd.Series([100, -50, 200]))
    assert "session_sharpe" not in s
