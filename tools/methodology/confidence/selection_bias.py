"""Selection-bias correction for multiple-testing on trading strategies.

When you test N candidate setups + cell variations and report stats on the
SURVIVORS, those stats are inflated by selection. The 'best of N' Sharpe ratio
is overstated even under the null hypothesis of no edge.

Two correction methods, both research-backed:

1. Harvey & Liu (2015), "Backtesting" (Duke + Man Group):
   Convert Sharpe -> t-statistic via t = SR * sqrt(T), apply Bonferroni / Holm
   / BHY p-value adjustment for M tests, back-out adjusted SR.

2. Lopez de Prado & Lewis (2019), "Detection of False Investment Strategies":
   Cluster the N strategies' equity curves by correlation. Use cluster count k
   as 'effective N' (since correlated strategies don't represent independent
   trials). Avoids over-penalizing when 30 setups share features (universe,
   sides, indicators).

Both methods assume per-period (daily) returns on the equity curve, not
per-trade outcomes. This module builds the daily equity curve from per-trade
data and applies the haircut on that.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


@dataclass(frozen=True)
class HaircutResult:
    setup_name: str
    raw_sharpe: float
    n_daily_obs: int
    effective_M: int        # number of effective independent trials
    adjusted_sharpe: float
    haircut_pct: float      # percent reduction from raw_sharpe
    method: str             # "Bonferroni" | "Holm" | "BHY"


# ---------------------------------------------------------------------------
# Build daily equity curves from per-trade outcomes
# ---------------------------------------------------------------------------

def build_daily_equity_curve(
    trades_df: pd.DataFrame,
    *,
    pnl_column: str = "net_pnl_inr",
    date_column: str = "signal_date",
) -> pd.Series:
    """Aggregate per-trade PnL into daily returns.

    For each trading day, sum NET PnL across trades. Days with no trades
    have PnL = 0. Returns a date-indexed pd.Series of daily Rs PnL.

    To convert to % returns later, divide by deployed capital (set by caller).
    For Sharpe computation here we use Rs PnL directly — Sharpe is scale-
    invariant for the simple ratio mean/std on any consistent unit.
    """
    df = trades_df.copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.date
    daily = df.groupby(date_column)[pnl_column].sum()
    return daily


def daily_sharpe(daily_pnl: pd.Series, *, annualization_factor: float = np.sqrt(252)) -> float:
    """Annualized daily Sharpe from a per-day PnL series.

    Per Harvey-Liu convention: SR = mean(daily) / std(daily), annualized by
    multiplying by sqrt(252) for daily data.

    Returns 0.0 if daily PnL series has zero variance.
    """
    if len(daily_pnl) < 2:
        return 0.0
    mean = float(daily_pnl.mean())
    std = float(daily_pnl.std(ddof=1))
    if std == 0:
        return 0.0
    return mean / std * annualization_factor


# ---------------------------------------------------------------------------
# ONC effective-N via correlation clustering
# ---------------------------------------------------------------------------

def compute_effective_N(
    daily_pnl_by_setup: Dict[str, pd.Series],
    *,
    distance_threshold: float = 0.5,
    min_correlation: float = 0.3,
) -> Tuple[int, Dict[str, int]]:
    """Cluster setups by daily-PnL correlation; return cluster count as
    effective N.

    Per Lopez de Prado & Lewis (2019): correlated strategies are not
    independent trials. The optimal-number-of-clusters (ONC) algorithm uses
    K-means with silhouette scoring; we use a simpler hierarchical clustering
    with correlation-based distance.

    Args:
        daily_pnl_by_setup: dict of setup_name -> daily PnL series
        distance_threshold: hierarchical clustering cutoff (distance = 1 - |corr|).
            0.5 means: setups with correlation > 0.5 cluster together.
        min_correlation: setups with all pairwise correlations below this are
            considered fully independent.

    Returns:
        (effective_N, dict of setup_name -> cluster_id)
    """
    setup_names = list(daily_pnl_by_setup.keys())
    if len(setup_names) <= 1:
        return 1, {n: 1 for n in setup_names}

    # Align all daily series to a common date index
    all_dates = sorted(set().union(*[s.index for s in daily_pnl_by_setup.values()]))
    aligned = pd.DataFrame(index=all_dates)
    for name, s in daily_pnl_by_setup.items():
        aligned[name] = s.reindex(all_dates).fillna(0)

    # Compute correlation matrix
    corr = aligned.corr()
    # Distance = 1 - |corr| (correlated → low distance; anti-correlated → low distance too)
    dist = 1.0 - corr.abs()
    np.fill_diagonal(dist.values, 0.0)
    # Symmetric square distance matrix → condensed for scipy
    condensed = squareform(dist.values, checks=False)

    # Hierarchical clustering with average linkage
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=distance_threshold, criterion="distance")
    cluster_map = dict(zip(setup_names, clusters))
    effective_N = len(set(clusters))
    return effective_N, cluster_map


# ---------------------------------------------------------------------------
# Harvey-Liu haircut: Bonferroni / Holm / BHY
# ---------------------------------------------------------------------------

def harvey_liu_haircut(
    setup_name: str,
    daily_pnl: pd.Series,
    *,
    M: int,
    method: str = "Bonferroni",
    alpha: float = 0.05,
) -> HaircutResult:
    """Apply Harvey-Liu Sharpe-ratio haircut for multiple testing.

    Standard interpretation (per Harvey & Liu 2015 + Harvey, Liu & Zhu 2016):
    The haircut FACTOR is the ratio of critical t-statistics:
        haircut_factor = t_crit(1 test) / t_crit(M tests)
    Adjusted Sharpe = Raw Sharpe × haircut_factor

    This preserves the sign of the raw Sharpe (negative setups stay negative,
    closer to zero) and avoids the formula breakdown when raw p-value is
    near zero (highly significant in either direction).

    Args:
        setup_name: for reporting
        daily_pnl: per-day PnL series (used to compute T and raw Sharpe)
        M: effective number of independent trials (from ONC clustering)
        method: "Bonferroni" (most conservative), "BHY" (less conservative;
                Benjamini-Hochberg-Yekutieli)
        alpha: significance level (0.05 = 95% confidence)

    Returns:
        HaircutResult with raw vs adjusted Sharpe. Sign-preserving.
    """
    raw_sr = daily_sharpe(daily_pnl)
    T = len(daily_pnl)

    if T < 2 or raw_sr == 0 or M < 1:
        return HaircutResult(
            setup_name=setup_name, raw_sharpe=raw_sr,
            n_daily_obs=T, effective_M=M, adjusted_sharpe=raw_sr,
            haircut_pct=0.0, method=method,
        )

    # Compute critical t-statistics (two-tailed)
    # Single test: alpha
    # M tests Bonferroni: alpha/M
    # M tests BHY: alpha * c(M) / M where c(M) = sum(1/i for i in 1..M)
    if method == "Bonferroni":
        alpha_adj = alpha / M
    elif method == "BHY":
        c_M = float(np.sum(1.0 / np.arange(1, M + 1)))
        alpha_adj = alpha * c_M / M
    else:
        raise ValueError(f"Unknown method: {method}; use Bonferroni or BHY")

    # Critical t-statistics at single vs multiple test thresholds
    t_crit_1 = stats.t.ppf(1 - alpha / 2, df=T - 1)
    t_crit_M = stats.t.ppf(1 - alpha_adj / 2, df=T - 1)

    # Haircut factor = ratio of critical t's (M tests is higher bar)
    haircut_factor = t_crit_1 / t_crit_M

    # Apply factor to raw Sharpe — preserves sign
    sr_adjusted = raw_sr * haircut_factor

    # Haircut percentage: how much closer to zero is the adjusted Sharpe
    haircut_pct = (1.0 - haircut_factor) * 100.0

    return HaircutResult(
        setup_name=setup_name, raw_sharpe=raw_sr,
        n_daily_obs=T, effective_M=M, adjusted_sharpe=sr_adjusted,
        haircut_pct=haircut_pct, method=method,
    )


# ---------------------------------------------------------------------------
# Convenience: full selection-bias analysis on multiple setups
# ---------------------------------------------------------------------------

def analyze_setups_selection_bias(
    setups_trades: Dict[str, pd.DataFrame],
    *,
    pnl_column: str = "net_pnl_inr",
    date_column: str = "signal_date",
    haircut_method: str = "Bonferroni",
) -> Dict[str, HaircutResult]:
    """Apply selection-bias correction across N setups.

    1. Build daily equity curve for each setup
    2. Compute effective N via ONC clustering
    3. Apply Harvey-Liu haircut to each setup's raw Sharpe

    Returns dict of setup_name -> HaircutResult.
    """
    daily_pnl_by_setup = {
        name: build_daily_equity_curve(df, pnl_column=pnl_column, date_column=date_column)
        for name, df in setups_trades.items()
    }

    effective_N, cluster_map = compute_effective_N(daily_pnl_by_setup)

    results = {}
    for setup_name, daily_pnl in daily_pnl_by_setup.items():
        results[setup_name] = harvey_liu_haircut(
            setup_name=setup_name,
            daily_pnl=daily_pnl,
            M=effective_N,
            method=haircut_method,
        )
    return results, effective_N, cluster_map
