"""Tests for tools.methodology.confidence.cnc_daily_returns.

Builds the daily mark-to-market portfolio-return series from overlapping multi-day
CNC positions — the input the stationary block bootstrap / Lo-Sharpe machinery in
cnc_stats.py operates on. Entry/exit positions mirror the cell-mine ledger exactly:
signal at day-t close -> ENTRY at T+1 open -> EXIT at close of T+1+K (positional,
per-symbol, on the sorted clean-daily panel).
"""
import numpy as np
import pandas as pd
import pytest

from tools.methodology.confidence.cnc_daily_returns import (
    expand_trade_to_daily,
    build_daily_portfolio_returns,
    simulate_slot_admission,
)


def _panel():
    """Two symbols, 8 trading days each, prices chosen for hand-computable returns
    (days 6-7 are filler so multi-day holds late in the window have a full window)."""
    d = pd.to_datetime([f"2023-01-0{i}" for i in range(2, 10)])  # d0..d7
    rows = []
    # Symbol A: d1 open100->close110 (+10%), d2 close121 (+10%), d3 close121 (0%)
    a_open = [90, 100, 115, 120, 130, 140, 150, 160]
    a_close = [95, 110, 121, 121, 125, 145, 155, 165]
    # Symbol B: d2 open50->close55 (+10%), d3 close55 (0%), d4 close60.5 (+10%)
    b_open = [40, 45, 50, 56, 60, 65, 70, 75]
    b_close = [42, 48, 55, 55, 60.5, 66, 71, 76]
    for i, dt in enumerate(d):
        rows.append(dict(symbol="A", date=dt, open=a_open[i], close=a_close[i]))
        rows.append(dict(symbol="B", date=dt, open=b_open[i], close=b_close[i]))
    return pd.DataFrame(rows)


D = pd.to_datetime([f"2023-01-0{i}" for i in range(2, 10)])  # D[0]..D[7]


# ---------------------------------------------------------------------------
# expand_trade_to_daily — one position's daily MTM returns over its hold window
# ---------------------------------------------------------------------------

def test_expand_trade_daily_returns_match_open_close_path():
    """Signal at D0, K=2 -> entry at D1 open, exit at D3 close. Daily returns:
    D1 = close/open, D2 = close/prev_close, D3 = close/prev_close."""
    s = expand_trade_to_daily(_panel(), symbol="A", signal_date=D[0], k_hold=2)
    assert list(s.index) == [D[1], D[2], D[3]]
    assert s.loc[D[1]] == pytest.approx(110 / 100 - 1)   # +0.10
    assert s.loc[D[2]] == pytest.approx(121 / 110 - 1)   # +0.10
    assert s.loc[D[3]] == pytest.approx(121 / 121 - 1)   # 0.0


def test_expand_trade_daily_compounds_to_total_fwd_return():
    """Product of (1+daily) equals the ledger's close_{T+1+K}/open_{T+1} - 1."""
    s = expand_trade_to_daily(_panel(), symbol="A", signal_date=D[0], k_hold=2)
    compounded = float(np.prod(1 + s.values)) - 1
    assert compounded == pytest.approx(121 / 100 - 1)    # +0.21


def test_expand_trade_cost_charged_on_entry_day():
    """Round-trip cost is subtracted from the entry-day return."""
    s = expand_trade_to_daily(_panel(), symbol="A", signal_date=D[0], k_hold=2, cost=0.01)
    assert s.loc[D[1]] == pytest.approx(110 / 100 - 1 - 0.01)
    assert s.loc[D[2]] == pytest.approx(121 / 110 - 1)   # later days unaffected


def test_expand_trade_insufficient_forward_rows_returns_empty():
    """A signal too close to the end of the panel (no full hold window) yields
    no daily returns rather than raising."""
    s = expand_trade_to_daily(_panel(), symbol="A", signal_date=D[6], k_hold=2)
    assert len(s) == 0


# ---------------------------------------------------------------------------
# build_daily_portfolio_returns — equal-weight across overlapping positions
# ---------------------------------------------------------------------------

def test_portfolio_returns_equal_weight_across_open_positions():
    """T1: A signal D0 (held D1-D3). T2: B signal D1 (held D2-D4). On overlap
    days the portfolio return is the equal-weight mean of open positions."""
    trades = pd.DataFrame([
        {"signal_date": D[0], "symbol": "A"},
        {"signal_date": D[1], "symbol": "B"},
    ])
    s = build_daily_portfolio_returns(trades, _panel(), k_hold=2)
    assert s.loc[D[1]] == pytest.approx(0.10)                    # only T1
    assert s.loc[D[2]] == pytest.approx((0.10 + 0.10) / 2)       # T1 & T2
    assert s.loc[D[3]] == pytest.approx((0.0 + 0.0) / 2)         # T1 & T2
    assert s.loc[D[4]] == pytest.approx(0.10)                    # only T2


def test_portfolio_returns_indexed_by_calendar_day_sorted():
    trades = pd.DataFrame([
        {"signal_date": D[0], "symbol": "A"},
        {"signal_date": D[1], "symbol": "B"},
    ])
    s = build_daily_portfolio_returns(trades, _panel(), k_hold=2)
    assert list(s.index) == sorted(s.index)
    assert list(s.index) == [D[1], D[2], D[3], D[4]]


def test_portfolio_returns_empty_trades_returns_empty_series():
    s = build_daily_portfolio_returns(pd.DataFrame(columns=["signal_date", "symbol"]),
                                      _panel(), k_hold=2)
    assert len(s) == 0


def test_portfolio_returns_sum_mode_adds_open_positions():
    """aggregate='sum' models a fixed-Rs/slot book: the day's return is the SUM of
    open-position returns (scale-invariant Sharpe), not the equal-weight mean."""
    trades = pd.DataFrame([
        {"signal_date": D[0], "symbol": "A"},
        {"signal_date": D[1], "symbol": "B"},
    ])
    s = build_daily_portfolio_returns(trades, _panel(), k_hold=2, aggregate="sum")
    assert s.loc[D[1]] == pytest.approx(0.10)              # only T1
    assert s.loc[D[2]] == pytest.approx(0.10 + 0.10)       # T1 + T2 summed
    assert s.loc[D[3]] == pytest.approx(0.0 + 0.0)
    assert s.loc[D[4]] == pytest.approx(0.10)              # only T2


# ---------------------------------------------------------------------------
# simulate_slot_admission — fixed-slot capacity book (reject when oversubscribed)
# ---------------------------------------------------------------------------

def test_slot_admission_caps_concurrent_positions_by_score():
    """Three signals on the same day, cap=2: keep the 2 best (lowest score =
    deepest loser = strongest signal), reject the worst."""
    trades = pd.DataFrame([
        {"signal_date": D[0], "symbol": "A", "score": 0.01},
        {"signal_date": D[0], "symbol": "B", "score": 0.02},
        {"signal_date": D[0], "symbol": "C", "score": 0.09},  # worst -> rejected
    ])
    admitted = simulate_slot_admission(trades, _panel(), k_hold=2, max_slots=2, score_col="score")
    assert set(admitted["symbol"]) == {"A", "B"}


def test_slot_admission_frees_slots_after_exit():
    """A slot occupied by a position entered earlier frees once that position
    exits, letting a later signal in even at cap=1."""
    # T1 signals D0 -> held D1..D3 (entry D1, exit close D3). T2 signals D3 ->
    # entry D4, by which time T1 has exited, so cap=1 admits both.
    trades = pd.DataFrame([
        {"signal_date": D[0], "symbol": "A", "score": 0.01},
        {"signal_date": D[3], "symbol": "B", "score": 0.01},
    ])
    admitted = simulate_slot_admission(trades, _panel(), k_hold=2, max_slots=1, score_col="score")
    assert set(admitted["symbol"]) == {"A", "B"}
