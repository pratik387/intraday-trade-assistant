"""Bootstrap BCa (Bias-Corrected accelerated) confidence intervals for
trading strategy metrics.

Per the research literature (Efron & Tibshirani 1993, "An Introduction to the
Bootstrap"; Efron 1987 for BCa specifically), bootstrap is the only
research-backed CI method for non-normal statistics like Profit Factor.
No closed-form CI exists for PF.

Why BCa over percentile bootstrap:
- BCa corrects for bias and skewness of the bootstrap distribution
- PF is bounded below by 0; bootstrap distribution is heavily right-skewed
- Plain percentile CI under-covers for skewed statistics

Per PyBroker convention (`pybroker.com/.../3.%20Evaluating%20with%20Bootstrap%20Metrics.html`),
B=5000-10000 resamples is standard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CIResult:
    """95% confidence interval result for a statistic."""
    point_estimate: float
    ci_lower: float        # 2.5th percentile after BCa adjustment
    ci_upper: float        # 97.5th percentile after BCa adjustment
    n: int                 # sample size
    n_resamples: int       # bootstrap iterations
    method: str = "BCa"

    def __repr__(self) -> str:
        return (f"{self.point_estimate:.4f} "
                f"[CI {self.ci_lower:.4f}, {self.ci_upper:.4f}] "
                f"(n={self.n}, {self.method})")


# ---------------------------------------------------------------------------
# Statistic functions (operate on per-trade pnl array)
# ---------------------------------------------------------------------------

def stat_pf(pnls: np.ndarray) -> float:
    """Profit Factor = sum(positives) / abs(sum(negatives))."""
    pos = pnls[pnls > 0].sum()
    neg = -pnls[pnls < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 1.0
    return float(pos / neg)


def stat_expectancy(pnls: np.ndarray) -> float:
    """Expectancy = mean per-trade PnL."""
    return float(np.mean(pnls))


def stat_win_rate(pnls: np.ndarray) -> float:
    """Fraction of trades with PnL > 0."""
    return float(np.mean(pnls > 0))


# ---------------------------------------------------------------------------
# BCa bootstrap implementation
# ---------------------------------------------------------------------------

def _bca_ci(
    sample: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_resamples: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Compute BCa (Bias-Corrected accelerated) CI bounds.

    Implementation follows Efron 1987 / DiCiccio & Romano 1996.

    Args:
        sample: 1-D array of observations
        statistic_fn: function mapping sample → scalar statistic
        n_resamples: bootstrap iterations
        alpha: significance level (0.05 = 95% CI)
        seed: RNG seed for reproducibility

    Returns:
        (lower_bound, upper_bound) of BCa CI.
    """
    rng = np.random.default_rng(seed)
    n = len(sample)

    # Bootstrap resamples (vectorized for speed)
    # Shape: (n_resamples, n)
    resample_idx = rng.integers(0, n, size=(n_resamples, n))
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        boot_stats[i] = statistic_fn(sample[resample_idx[i]])
    boot_stats = boot_stats[np.isfinite(boot_stats)]
    if len(boot_stats) < n_resamples * 0.5:
        # Too many infs/nans — degenerate distribution
        # Fall back to point estimate
        point = statistic_fn(sample)
        return (point, point)

    # Theta-hat (point estimate on original sample)
    theta_hat = statistic_fn(sample)

    # Bias correction z0
    p_below = np.sum(boot_stats < theta_hat) / len(boot_stats)
    # Avoid p=0 or p=1
    p_below = np.clip(p_below, 1.0 / (2 * n_resamples), 1.0 - 1.0 / (2 * n_resamples))
    z0 = stats.norm.ppf(p_below)

    # Acceleration via jackknife
    jack_stats = np.empty(n)
    for i in range(n):
        jack_stats[i] = statistic_fn(np.delete(sample, i))
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    acc = num / den if den > 0 else 0.0

    # Adjusted alpha levels
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    alpha_lo_adj = stats.norm.cdf(z0 + (z0 + z_alpha_lo) / (1 - acc * (z0 + z_alpha_lo)))
    alpha_hi_adj = stats.norm.cdf(z0 + (z0 + z_alpha_hi) / (1 - acc * (z0 + z_alpha_hi)))

    # Clip to valid percentile range
    alpha_lo_adj = np.clip(alpha_lo_adj, 0.0, 1.0)
    alpha_hi_adj = np.clip(alpha_hi_adj, 0.0, 1.0)

    ci_lower = float(np.percentile(boot_stats, alpha_lo_adj * 100))
    ci_upper = float(np.percentile(boot_stats, alpha_hi_adj * 100))
    return ci_lower, ci_upper


def bootstrap_ci(
    sample: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 20260520,
    method: str = "BCa",
) -> CIResult:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        sample: 1-D array of per-trade PnL values
        statistic_fn: e.g., stat_pf, stat_expectancy, stat_win_rate
        n_resamples: 5000 is standard (Politis-Romano convention)
        alpha: 0.05 = 95% CI
        seed: RNG seed for reproducibility
        method: "BCa" (default) or "percentile" (simpler fallback)

    Returns:
        CIResult with point estimate + CI bounds.
    """
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    n = len(sample)
    point = statistic_fn(sample) if n > 0 else float("nan")

    if n < 10:
        # Insufficient data — return point estimate as CI (degenerate)
        return CIResult(
            point_estimate=point, ci_lower=point, ci_upper=point,
            n=n, n_resamples=0, method="insufficient_data",
        )

    if method == "BCa":
        ci_lower, ci_upper = _bca_ci(sample, statistic_fn, n_resamples, alpha, seed)
    elif method == "percentile":
        rng = np.random.default_rng(seed)
        boot_stats = np.empty(n_resamples)
        for i in range(n_resamples):
            boot_stats[i] = statistic_fn(rng.choice(sample, size=n, replace=True))
        finite = boot_stats[np.isfinite(boot_stats)]
        ci_lower = float(np.percentile(finite, alpha / 2 * 100))
        ci_upper = float(np.percentile(finite, (1 - alpha / 2) * 100))
    else:
        raise ValueError(f"Unknown method: {method}")

    return CIResult(
        point_estimate=point, ci_lower=ci_lower, ci_upper=ci_upper,
        n=n, n_resamples=n_resamples, method=method,
    )


# ---------------------------------------------------------------------------
# Convenience: compute all three CIs from a trades DataFrame
# ---------------------------------------------------------------------------

def compute_aggregate_ci(
    trades_df: pd.DataFrame,
    *,
    pnl_column: str = "net_pnl_inr",
    n_resamples: int = 5000,
    seed: int = 20260520,
) -> dict:
    """Compute BCa CI on PF, expectancy, and win rate from a trades DataFrame.

    Uses `net_pnl_inr` column by default (real production NET PnL). Returns dict
    of CIResults keyed by metric name.
    """
    if pnl_column not in trades_df.columns:
        raise ValueError(f"{pnl_column!r} not in trades_df columns: {list(trades_df.columns)}")
    pnls = trades_df[pnl_column].to_numpy()

    return {
        "pf": bootstrap_ci(pnls, stat_pf, n_resamples=n_resamples, seed=seed),
        "expectancy": bootstrap_ci(pnls, stat_expectancy, n_resamples=n_resamples, seed=seed),
        "win_rate": bootstrap_ci(pnls, stat_win_rate, n_resamples=n_resamples, seed=seed),
    }
