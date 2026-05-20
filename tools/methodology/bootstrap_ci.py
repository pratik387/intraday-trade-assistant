"""Bootstrap confidence interval for Profit Factor on per-window trades.

PF (Profit Factor) = sum(positive PnLs) / abs(sum(negative PnLs)).
Bootstrap resamples trades with replacement N times to produce a PF
distribution, then reports the 2.5th and 97.5th percentiles as the
95% CI bounds.

A window "passes" the walk-forward gate only if its CI lower bound > 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


class InsufficientData(ValueError):
    """Raised when n < min_n; CI is meaningless on tiny samples."""


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n: int
    n_resamples: int


def _profit_factor(pnls: np.ndarray) -> float:
    """PF = sum(positives) / abs(sum(negatives)). Returns inf if no losses."""
    pos = pnls[pnls > 0].sum()
    neg = -pnls[pnls < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 1.0
    return float(pos / neg)


def bootstrap_pf_ci(
    trades_df: pd.DataFrame,
    n_resamples: int = 1000,
    seed: int = 20260519,
    min_n: int = 10,
    pnl_col: str = "pnl_pct",
) -> BootstrapResult:
    """Compute bootstrap CI for Profit Factor.

    Raises InsufficientData if len(trades_df) < min_n.
    """
    if len(trades_df) < min_n:
        raise InsufficientData(f"n={len(trades_df)} < min_n={min_n}")

    pnls = trades_df[pnl_col].to_numpy()
    point = _profit_factor(pnls)

    rng = np.random.default_rng(seed)
    n_trades = len(pnls)
    # Vectorized: shape (n_resamples, n_trades) — ~10-50x faster than a Python loop
    samples = rng.choice(pnls, size=(n_resamples, n_trades), replace=True)
    pos_sums = np.where(samples > 0, samples, 0.0).sum(axis=1)
    neg_sums = -np.where(samples < 0, samples, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        resampled_pfs = np.where(
            neg_sums > 0,
            pos_sums / neg_sums,
            np.where(pos_sums > 0, np.inf, 1.0),
        )

    arr = resampled_pfs
    finite = arr[np.isfinite(arr)]
    if not np.isfinite(point):
        # All-wins window: PF is genuinely inf. CI is trivially [inf, inf].
        ci_lower = float("inf")
        ci_upper = float("inf")
    elif len(finite) < 10:
        # Bootstrap distribution too degenerate to compute meaningful CI.
        # Fail-safe: lower=1.0 means window does not pass the >1.0 gate from CI alone.
        ci_lower = 1.0
        ci_upper = float(np.percentile(finite, 97.5)) if len(finite) > 0 else float("inf")
    else:
        ci_lower = float(np.percentile(finite, 2.5))
        ci_upper = float(np.percentile(finite, 97.5))

    return BootstrapResult(
        point_estimate=point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=len(trades_df),
        n_resamples=n_resamples,
    )
