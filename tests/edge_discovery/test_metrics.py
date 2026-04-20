"""Tests for edge metrics (PF, Sharpe, DD, WR, expectancy)."""
import math
import numpy as np
import pandas as pd
import pytest

from tools.edge_discovery.metrics import (
    profit_factor,
    win_rate_pct,
    sharpe_ratio,
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
