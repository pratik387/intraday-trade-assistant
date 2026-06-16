"""Statistical core for the CNC/MTF confidence card.

The intraday confidence card (`confidence_card.py`) assumes per-trade observations
are ~independent (same-day-closed intraday trades). That assumption is FALSE for a
multi-day-hold, daily-rebalanced, cross-sectional reversion book:

  (a) each signal day enters a basket of correlated names (within-day clustering), and
  (b) overlapping multi-day holds induce positive autocorrelation in the daily
      portfolio-return series.

Naive iid per-trade bootstrap CIs and a √252 Sharpe are therefore too tight /
overstated. This module implements the deep-research recipe documented in
`specs/2026-06-15-cnc-confidence-card-methodology-research.md`:

  F1/F2/F3 — `stationary_bootstrap_ci`: stationary block bootstrap (Politis-Romano
             1994) of the daily portfolio-return series, with the data-adaptive
             block length of Politis-White (2004), via the `arch` library.
  F4       — `lo_sharpe_se`: Lo (2002) autocorrelation-corrected Sharpe standard
             error via a Newey-West HAC covariance of the GMM moment conditions.
  F5       — `lo_eta`: Lo's η(q) time-aggregation factor (use instead of √q under
             autocorrelation).
  F7/F8    — `expected_max_sharpe` + `deflated_sharpe_ratio`: Deflated Sharpe Ratio
             (Bailey & Lopez de Prado 2014) — selection-bias / multiple-testing /
             non-normality haircut, replacing Harvey-Liu Bonferroni for a large
             dispersed cell-mine sweep.
  F9       — `effective_n_independent`: reduce the raw trial count M to the number
             of INDEPENDENT trials N fed to the DSR.

All formulas are unit-faithful to their citations; `sr_hat`/`sr0` are PER-PERIOD
(non-annualized) Sharpe ratios unless otherwise noted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats

# Euler-Mascheroni constant (Bailey-Lopez de Prado getExpMaxSR).
_EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# F5: Lo (2002) eta(q) time-aggregation factor
# ---------------------------------------------------------------------------

def lo_eta(autocorrs: Sequence[float], q: int) -> float:
    """Lo's η(q) factor for aggregating a per-period Sharpe to q periods.

    η(q) = q / sqrt(q + 2 * Σ_{k=1}^{q-1} (q-k) * ρ_k)

    Under zero serial correlation η(q) = √q. POSITIVE autocorrelation (overlapping
    multi-day holds) pushes η(q) below √q, so naive √q OVERSTATES the aggregated
    Sharpe. Use η(q), not √q (Lo 2002, FAJ 58(4)).

    Args:
        autocorrs: ρ_1 .. ρ_{q-1} (length >= q-1; extra entries ignored).
        q: aggregation horizon in periods.
    """
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}")
    if q == 1:
        return 1.0
    if len(autocorrs) < q - 1:
        raise ValueError(f"need at least q-1={q-1} autocorrelations, got {len(autocorrs)}")
    s = sum((q - k) * autocorrs[k - 1] for k in range(1, q))
    denom = q + 2.0 * s
    if denom <= 0:
        raise ValueError(f"non-positive variance ratio (denom={denom}); autocorrs imply a degenerate series")
    return q / np.sqrt(denom)


# ---------------------------------------------------------------------------
# F8: expected-max-Sharpe SR0 (DSR deflation benchmark)
# ---------------------------------------------------------------------------

def expected_max_sharpe(var_sr: float, n_trials: float) -> float:
    """Expected maximum Sharpe of N unskilled trials (False Strategy Theorem).

    SR0 = sqrt(V[SR]) * ((1-γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(N·e)])

    where γ is the Euler-Mascheroni constant and Z⁻¹ the standard-normal quantile.
    SR0 grows with the trial count N AND with the cross-sectional variance of the
    trial Sharpes V[SR] (Bailey & Lopez de Prado 2014, Eq. 1).

    Args:
        var_sr: cross-sectional variance of the trial Sharpe ratios.
        n_trials: number of INDEPENDENT trials N (see effective_n_independent).
    """
    if n_trials <= 1:
        # A single trial has no selection bias; expected max == 0 hurdle.
        return 0.0
    g = _EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1.0 - g) * z1 + g * z2))


# ---------------------------------------------------------------------------
# F7: Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    sr_hat: float,
    sr0: float,
    n_obs: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio = probability the true Sharpe exceeds the SR0 hurdle.

    DSR = Φ( (SR_hat - SR0)·√(n-1) / √(1 - skew·SR_hat + ((kurt-1)/4)·SR_hat²) )

    A PSR (Bailey & Lopez de Prado 2014) whose threshold is the expected-max-Sharpe
    SR0 rather than zero. Incorporates non-normality via skew and (full) kurtosis;
    `kurt=3` is the Gaussian baseline. SR_hat and SR0 are PER-PERIOD Sharpe ratios.

    Returns a probability in [0, 1]; DSR = 0.5 when SR_hat == SR0.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    denom = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2
    if denom <= 0:
        raise ValueError(f"non-positive Sharpe-variance denominator ({denom}); check skew/kurt inputs")
    z = (sr_hat - sr0) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


# ---------------------------------------------------------------------------
# F4: Lo (2002) autocorrelation-corrected Sharpe standard error
# ---------------------------------------------------------------------------

def _default_hac_lag(t: int) -> int:
    """Truncation lag ~ T^(1/3) (research: ~5-10 for a few-hundred-obs series)."""
    return max(1, int(round(t ** (1.0 / 3.0))))


def lo_sharpe_se(returns: np.ndarray, lag: Optional[int] = None) -> float:
    """Autocorrelation-corrected SE of the per-period Sharpe (Lo 2002).

    GMM estimator with a Newey-West / Bartlett HAC covariance of the moment
    conditions g_t = [r_t - μ, (r_t-μ)² - m₂]. The Sharpe SR = μ/√m₂; by the delta
    method Var(SR) = ∇f' Σ_HAC ∇f / T with ∇f = [1/√m₂, -μ/(2 m₂^{3/2})].

    With `lag=0` and iid-normal returns this reduces to the closed-form
    √((1 + SR²/2)/T). Positive serial correlation (overlapping holds) inflates the
    HAC covariance and hence the SE.

    Args:
        returns: per-period return series.
        lag: Bartlett truncation lag; default ~T^(1/3).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3:
        return float("nan")
    if lag is None:
        lag = _default_hac_lag(T)
    lag = min(lag, T - 1)

    mu = r.mean()
    dev = r - mu
    m2 = float(np.mean(dev ** 2))
    if m2 <= 0:
        return float("nan")

    # Moment-condition series g_t (T x 2).
    g = np.column_stack([dev, dev ** 2 - m2])

    # Newey-West HAC long-run covariance with Bartlett weights.
    gamma0 = (g.T @ g) / T
    sigma = gamma0.copy()
    for k in range(1, lag + 1):
        gk = (g[k:].T @ g[:-k]) / T
        w = 1.0 - k / (lag + 1.0)  # Bartlett kernel
        sigma += w * (gk + gk.T)

    grad = np.array([1.0 / np.sqrt(m2), -mu / (2.0 * m2 ** 1.5)])
    var_sr = float(grad @ sigma @ grad) / T
    if var_sr <= 0:
        return float("nan")
    return float(np.sqrt(var_sr))


# ---------------------------------------------------------------------------
# F9: effective independent N from correlated trials
# ---------------------------------------------------------------------------

def effective_n_independent(M: int, rho_bar: float) -> float:
    """Independent-trial count implied by M correlated trials with mean
    pairwise correlation rho_bar: N = M·(1 - ρ̄) + ρ̄.

    Boundary conditions: ρ̄=0 → N=M (fully independent); ρ̄=1 → N=1 (fully
    redundant). Fed as `n_trials` to expected_max_sharpe (Bailey-Lopez de Prado
    2014, App. A.3). For a more robust estimate the authors recommend
    information-theoretic redundancy / clustering.
    """
    rho_bar = float(np.clip(rho_bar, 0.0, 1.0))
    return M * (1.0 - rho_bar) + rho_bar


# ---------------------------------------------------------------------------
# F1/F2/F3: stationary block bootstrap of the daily return series
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CNCBootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    block_length: float       # data-adaptive (Politis-White) mean block length
    n_resamples: int
    method: str = "stationary_bootstrap"


def stationary_bootstrap_ci(
    returns: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 20260615,
    block_size: Optional[float] = None,
    ci_method: str = "percentile",
) -> CNCBootstrapResult:
    """Stationary block-bootstrap CI for a statistic of a daily return series.

    Preserves temporal dependence (within-day basket correlation captured by the
    daily aggregation, day-to-day overlap by the blocks), so the CI is honestly
    wider than an iid per-observation bootstrap. Block length is set
    data-adaptively via `arch.bootstrap.optimal_block_length` (Politis-White 2004,
    N^(1/3) rule) unless `block_size` is supplied.

    Args:
        returns: daily portfolio-return series (1-D).
        statistic_fn: maps a resampled 1-D array -> scalar (e.g. np.mean, stat_pf).
        n_resamples: bootstrap iterations.
        alpha: 0.05 -> 95% CI.
        seed: RNG seed for reproducibility.
        block_size: override the adaptive mean block length.
        ci_method: arch conf_int method ("percentile" default, or "bca").
    """
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    point = float(statistic_fn(x)) if n > 0 else float("nan")
    if n < 10:
        return CNCBootstrapResult(point, point, point, 1.0, 0, "insufficient_data")

    if block_size is None:
        block_size = float(optimal_block_length(x)["stationary"].iloc[0])
    block_size = max(1.0, float(block_size))

    bs = StationaryBootstrap(block_size, x, seed=seed)
    ci = bs.conf_int(lambda d: np.asarray([statistic_fn(d)]),
                     reps=n_resamples, method=ci_method, size=1.0 - alpha)
    ci_lower = float(ci[0, 0])
    ci_upper = float(ci[1, 0])
    return CNCBootstrapResult(
        point_estimate=point, ci_lower=ci_lower, ci_upper=ci_upper,
        block_length=block_size, n_resamples=n_resamples,
    )


# ---------------------------------------------------------------------------
# Daily-return statistic helpers (operate on a daily portfolio-return series)
# ---------------------------------------------------------------------------

def stat_pf(returns: np.ndarray) -> float:
    """Profit Factor on a daily-return series: Σ(gains) / |Σ(losses)|."""
    r = np.asarray(returns, dtype=float)
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 1.0
    return float(pos / neg)


def stat_mean(returns: np.ndarray) -> float:
    """Expectancy = mean daily return."""
    return float(np.mean(returns))
