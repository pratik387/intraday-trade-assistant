"""Edge metrics computed on a pandas Series of realized trade PnL (in INR).

All functions accept a Series and return scalar floats. None handle missing
values — caller is responsible for pre-cleaning (drop NaN, filter is_final_exit).
"""
from typing import Dict
import math
import pandas as pd


def profit_factor(pnl: pd.Series) -> float:
    """Sum of winners / |sum of losers|. Returns 0.0 for all-losers, inf for
    all-winners, 0.0 for empty."""
    if len(pnl) == 0:
        return 0.0
    winners = pnl[pnl > 0].sum()
    losers_abs = abs(pnl[pnl < 0].sum())
    if losers_abs == 0:
        return float("inf") if winners > 0 else 0.0
    return float(winners / losers_abs)


def win_rate_pct(pnl: pd.Series) -> float:
    """Percent of trades with PnL > 0 (zero does NOT count as winner)."""
    if len(pnl) == 0:
        return 0.0
    return float((pnl > 0).mean() * 100)


def sharpe_ratio(pnl: pd.Series) -> float:
    """Per-trade Sharpe = mean / std (ddof=1). No annualization — scale is
    per-trade risk units. Returns inf for zero-std positive series, -inf for
    zero-std negative series, 0.0 for empty."""
    if len(pnl) == 0:
        return 0.0
    std = pnl.std(ddof=1)
    if std == 0 or math.isnan(std):
        mean = pnl.mean()
        if mean > 0:
            return float("inf")
        if mean < 0:
            return float("-inf")
        return 0.0
    return float(pnl.mean() / std)


def max_drawdown_pct(pnl: pd.Series) -> float:
    """Max drawdown in rupees, expressed as PERCENT of total gross profit.

    Cumulative equity curve → trough-to-peak absolute drop. Convention per spec:
    'Max DD < 30% of total profit' → we return the percentage.
    Returns 0.0 for empty or monotonically increasing series.
    """
    if len(pnl) == 0:
        return 0.0
    equity = pnl.cumsum()
    running_peak = equity.cummax()
    drawdown = equity - running_peak  # negative values
    max_dd_abs = abs(drawdown.min())  # rupees
    if max_dd_abs == 0:
        return 0.0
    total_profit = float(pnl.sum())  # net PnL; spec: "Max DD < 30% of total profit"
    if total_profit <= 0:
        return float("inf")
    return float(max_dd_abs / total_profit * 100)


def expectancy(pnl: pd.Series) -> float:
    """Average PnL per trade. Redundant with mean() but named for clarity."""
    if len(pnl) == 0:
        return 0.0
    return float(pnl.mean())


def summary_stats(pnl: pd.Series) -> Dict[str, float]:
    """Full stat dict used by all stages."""
    return {
        "n": int(len(pnl)),
        "total_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "avg_pnl": expectancy(pnl),
        "pf": profit_factor(pnl),
        "wr_pct": win_rate_pct(pnl),
        "sharpe": sharpe_ratio(pnl),
        "max_dd_pct": max_drawdown_pct(pnl),
    }
