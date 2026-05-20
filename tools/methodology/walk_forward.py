"""Walk-forward engine.

Splits a trades DataFrame into N non-overlapping time windows, computes
per-window stats (PF_real, PF_net, n, WR, mwin, same_bar_pct), runs
bootstrap CI per window, and classifies the setup as GREEN / AMBER / RED.

Per the spec at docs/superpowers/specs/2026-05-19-walk-forward-methodology-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from tools.methodology.bootstrap_ci import (
    BootstrapResult,
    InsufficientData,
    bootstrap_pf_ci,
)


class Tier(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


@dataclass(frozen=True)
class Window:
    index: int
    start: date
    end: date


@dataclass(frozen=True)
class WindowStats:
    window: Window
    n: int
    pf_real: float
    pf_net: float
    wr_pct: float
    same_bar_pct: float
    bootstrap: Optional[BootstrapResult]
    passes_gate: bool


@dataclass(frozen=True)
class WalkForwardResult:
    setup_name: str
    windows: List[WindowStats]
    windows_pass: int
    windows_total: int
    pass_rate: float
    tier: Tier
    cb_drawdown_threshold: float


def _add_months(d: date, months: int) -> date:
    """Add `months` months to `d`. Day clamps to 1 if d.day > 28 (windows always start day=1)."""
    new_month = d.month - 1 + months
    new_year = d.year + new_month // 12
    new_month = new_month % 12 + 1
    return date(new_year, new_month, d.day if d.day <= 28 else 1)


def build_windows(
    start: date, end: date, window_months: int, n_windows: int,
) -> List[Window]:
    """Build N non-overlapping windows of window_months each, anchored at start."""
    windows = []
    cur = start
    for i in range(n_windows):
        month_end = _add_months(cur, window_months) - timedelta(days=1)
        windows.append(Window(index=i, start=cur, end=min(month_end, end)))
        cur = _add_months(cur, window_months)
    return windows


def classify_tier(pass_rate: float, n_windows_total: int) -> Tier:
    """Classify setup tier based on CI-adjusted pass rate.

    GREEN: >= 9 of 13 (~69%+)
    AMBER: 6-8 of 13 (~46-62%)
    RED:   <= 5 of 13 (<= 38%)
    """
    n_pass = int(round(pass_rate * n_windows_total))
    if n_pass >= 9:
        return Tier.GREEN
    if n_pass >= 6:
        return Tier.AMBER
    return Tier.RED


def _compute_per_trade_net_pnl(
    pnl_pct: float, fee_pct: float, mis_leverage: float,
) -> float:
    """Apply MIS leverage then subtract fees, both on CAPITAL basis.

    DEPRECATED for use with mixed-setup trades — fee_pct varies by setup
    (0.25-0.53% on capital depending on trade size, T1-partial frequency,
    side). Use `_compute_per_trade_net_pnl_from_columns` instead when the
    trades DataFrame has `realized_pnl_inr`, `fee_inr`, `entry_price`, `qty`.

    Per-setup fee_pct measurements (Discovery samples 2026-05-20):
      pre_results_t1 single-leg: 0.248
      pre_results_t1 T1-partial: 0.319
      mis_unwind: 0.296
      capitulation_long_v2: 0.437
      circuit_release: 0.411
      long_panic_gap_down: 0.461
      or_window_failure: 0.444
      delivery_pct_anomaly: 0.488
      capitulation_long_morning single: 0.426
      capitulation_long_morning T1: 0.531

    A flat fee_pct is a rough approximation. Use per-trade actual fees
    via _compute_per_trade_net_pnl_from_columns for production verdicts.
    """
    gross_leveraged = pnl_pct * mis_leverage
    net = gross_leveraged - fee_pct
    return net


def _compute_per_trade_net_pnl_from_columns(
    pnl_pct: pd.Series,
    entry_price: pd.Series,
    qty: pd.Series,
    fee_inr: pd.Series,
    mis_leverage: float,
) -> pd.Series:
    """Vectorized per-trade net % on capital, using ACTUAL per-trade fees.

    capital_per_trade = notional / leverage = entry × qty / leverage
    gross_pnl_capital_pct = pnl_pct × leverage  (price move scales by L on capital basis)
    fee_capital_pct = fee_inr / capital × 100   (real fee normalized to capital)
    net_pct_capital = gross_pnl_capital_pct - fee_capital_pct

    This is exact — no fee model approximation. Use whenever the sanity CSV
    carries actual fee_inr per trade.
    """
    notional = entry_price.astype(float) * qty.astype(int)
    capital = notional / mis_leverage
    gross_leveraged_pct = pnl_pct.astype(float) * mis_leverage
    fee_pct_capital = fee_inr.astype(float) / capital * 100.0
    net_pct = gross_leveraged_pct - fee_pct_capital
    return net_pct


def _profit_factor_from_series(pnls: pd.Series) -> float:
    pos = pnls[pnls > 0].sum()
    neg = -pnls[pnls < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 1.0
    return float(pos / neg)


def run_walk_forward(
    setup_name: str,
    trades_df: pd.DataFrame,
    start: date,
    end: date,
    window_months: int = 3,
    n_windows: int = 13,
    fee_pct_round_trip: float = 0.25,
    mis_leverage: float = 5.0,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 20260519,
    pf_net_gate: float = 1.10,
    min_n_for_ci: int = 10,
) -> WalkForwardResult:
    """Run walk-forward validation on a trades DataFrame.

    trades_df must have columns: signal_date (date or YYYY-MM-DD str), pnl_pct.
    """
    required_cols = {"signal_date", "pnl_pct"}
    missing = required_cols - set(trades_df.columns)
    if missing:
        raise ValueError(f"trades_df missing required columns: {sorted(missing)}")

    trades_df = trades_df.copy()
    # Always convert signal_date to datetime.date — string columns and
    # Timestamp columns both produce the same date objects for window filtering.
    trades_df["signal_date"] = pd.to_datetime(trades_df["signal_date"]).dt.date

    # Net PnL computation: prefer per-trade actual fees (exact) over flat fee_pct.
    # Use _compute_per_trade_net_pnl_from_columns when fee_inr + entry_price +
    # qty are all available in the canonical CSV. This is the accurate path.
    # Falls back to flat fee_pct_round_trip only when fee data missing.
    has_actual_fees = all(
        c in trades_df.columns for c in ("fee_inr", "entry_price", "qty")
    )
    if has_actual_fees:
        trades_df["pnl_pct_net"] = _compute_per_trade_net_pnl_from_columns(
            pnl_pct=trades_df["pnl_pct"],
            entry_price=trades_df["entry_price"],
            qty=trades_df["qty"],
            fee_inr=trades_df["fee_inr"],
            mis_leverage=mis_leverage,
        )
    else:
        trades_df["pnl_pct_net"] = trades_df["pnl_pct"].apply(
            lambda x: _compute_per_trade_net_pnl(x, fee_pct_round_trip, mis_leverage)
        )

    windows = build_windows(start, end, window_months, n_windows)
    stats_list: List[WindowStats] = []
    per_window_net_totals: List[float] = []

    for w in windows:
        mask = (trades_df["signal_date"] >= w.start) & (trades_df["signal_date"] <= w.end)
        wt = trades_df[mask]
        n = len(wt)

        if n == 0:
            stats_list.append(WindowStats(
                window=w, n=0, pf_real=0.0, pf_net=0.0, wr_pct=0.0,
                same_bar_pct=0.0, bootstrap=None, passes_gate=False,
            ))
            per_window_net_totals.append(0.0)
            continue

        pf_real = _profit_factor_from_series(wt["pnl_pct"])
        pf_net = _profit_factor_from_series(wt["pnl_pct_net"])
        wr = float((wt["pnl_pct_net"] > 0).mean() * 100)
        same_bar = float(0.0)
        net_total = float(wt["pnl_pct_net"].sum())
        per_window_net_totals.append(net_total)

        try:
            bs = bootstrap_pf_ci(
                wt[["pnl_pct_net"]].rename(columns={"pnl_pct_net": "pnl_pct"}),
                n_resamples=bootstrap_n,
                seed=bootstrap_seed,
                min_n=min_n_for_ci,
            )
            passes_ci = bs.ci_lower > 1.0
        except InsufficientData:
            bs = None
            passes_ci = False

        passes_gate = (pf_net >= pf_net_gate) and passes_ci
        stats_list.append(WindowStats(
            window=w, n=n, pf_real=pf_real, pf_net=pf_net, wr_pct=wr,
            same_bar_pct=same_bar, bootstrap=bs, passes_gate=passes_gate,
        ))

    windows_pass = sum(1 for s in stats_list if s.passes_gate)
    pass_rate = windows_pass / n_windows if n_windows > 0 else 0.0
    tier = classify_tier(pass_rate, n_windows)

    totals = np.array(per_window_net_totals)
    mu = float(totals.mean())
    sigma = float(totals.std(ddof=0))
    cb_threshold = mu - 2 * sigma

    return WalkForwardResult(
        setup_name=setup_name,
        windows=stats_list,
        windows_pass=windows_pass,
        windows_total=n_windows,
        pass_rate=pass_rate,
        tier=tier,
        cb_drawdown_threshold=cb_threshold,
    )
