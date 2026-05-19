"""Edge metrics computed on a pandas Series of realized trade PnL (in INR).

All functions accept a Series and return scalar floats. None handle missing
values — caller is responsible for pre-cleaning (drop NaN, filter is_final_exit).
"""
from typing import Dict, Optional
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
    zero-std negative series, 0.0 for empty.

    Informational only. For edge gating, use `session_sharpe_ratio` — intraday
    systems have per-trade variance that makes per-trade Sharpe structurally
    0.1-0.3 even for genuinely profitable strategies. Session-aggregated Sharpe
    is the finance-convention meaning of 'Sharpe'.
    """
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


def session_sharpe_ratio(pnl: pd.Series, session_dates: pd.Series) -> float:
    """Session-aggregated Sharpe (finance-convention).

    Sums per-trade pnl by session_date, then computes mean/std of the resulting
    daily PnL series (ddof=1). No annualization — scale is per-session risk
    units. This is what 'Sharpe ≥ X' conventionally means for an intraday
    system: daily PnL variability, not per-trade variability.

    Edge cases:
      - Empty pnl → 0.0
      - All days profitable, zero std → inf (finite std of zero)
      - Single session → 0.0 (std undefined with ddof=1, n=1 is meaningless)

    Args:
        pnl: per-trade realized PnL series
        session_dates: aligned series of session dates (same length as pnl)
    """
    if len(pnl) == 0:
        return 0.0
    if len(pnl) != len(session_dates):
        raise ValueError(
            f"pnl length ({len(pnl)}) must match session_dates length ({len(session_dates)})"
        )
    daily = pnl.groupby(session_dates.values).sum()
    if len(daily) < 2:
        return 0.0
    std = daily.std(ddof=1)
    if std == 0 or math.isnan(std):
        mean = daily.mean()
        if mean > 0:
            return float("inf")
        if mean < 0:
            return float("-inf")
        return 0.0
    return float(daily.mean() / std)


def max_drawdown_pct(pnl: pd.Series) -> float:
    """Max drawdown in rupees, expressed as PERCENT of total NET profit.

    Cumulative equity curve → trough-to-peak absolute drop. Denominator is
    net PnL (pnl.sum()), NOT gross winners, so a strategy that nearly gives
    back all its gains fails the 30% filter aggressively. Per spec criterion:
    'Max DD < 30% of total profit' where total profit = net realized.

    Edge cases:
      - Empty series → 0.0
      - No drawdown (monotonically increasing equity) → 0.0
      - Net PnL ≤ 0 → inf if any drawdown exists, else 0.0
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


def summary_stats(pnl: pd.Series, session_dates: Optional[pd.Series] = None) -> Dict[str, float]:
    """Full stat dict used by all stages.

    If `session_dates` is provided, adds `session_sharpe` (daily-aggregated).
    The per-trade `sharpe` is always present for reference.
    """
    stats = {
        "n": int(len(pnl)),
        "total_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "avg_pnl": expectancy(pnl),
        "pf": profit_factor(pnl),
        "wr_pct": win_rate_pct(pnl),
        "sharpe": sharpe_ratio(pnl),
        "max_dd_pct": max_drawdown_pct(pnl),
    }
    if session_dates is not None:
        stats["session_sharpe"] = session_sharpe_ratio(pnl, session_dates)
    return stats
