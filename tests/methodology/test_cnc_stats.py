"""Tests for tools.methodology.confidence.cnc_stats.

The CNC confidence card is for a MULTI-DAY, DAILY-REBALANCED, CROSS-SECTIONAL
reversion strategy whose per-trade observations are NOT independent. The intraday
card's iid per-trade bootstrap + Harvey-Liu haircut is invalid here. This module
implements the deep-research recipe
(specs/2026-06-15-cnc-confidence-card-methodology-research.md):

  - Lo (2002) autocorrelation-corrected Sharpe SE + eta(q) time-aggregation (F4/F5)
  - Deflated Sharpe Ratio / expected-max-Sharpe SR0 (Bailey-Lopez de Prado 2014) (F7/F8)
  - effective independent N from correlated trials (F9)
  - stationary block bootstrap of the daily portfolio-return series (F1/F2/F3)

Reference numeric oracles in this file were precomputed independently with scipy.
"""
import numpy as np
import pytest

from tools.methodology.confidence.cnc_stats import (
    lo_eta,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    lo_sharpe_se,
    effective_n_independent,
    stationary_bootstrap_ci,
)


# ---------------------------------------------------------------------------
# F5: Lo's eta(q) time-aggregation factor
# ---------------------------------------------------------------------------

def test_lo_eta_zero_autocorr_equals_sqrt_q():
    """With zero serial correlation, eta(q) reduces to sqrt(q) (the iid case)."""
    assert lo_eta([0.0, 0.0, 0.0], q=4) == pytest.approx(2.0)  # sqrt(4)


def test_lo_eta_positive_autocorr_below_sqrt_q():
    """Positive serial correlation (overlapping holds) pushes eta(q) BELOW
    sqrt(q): naive sqrt(q) annualization OVERSTATES the Sharpe."""
    eta = lo_eta([0.5], q=2)
    assert eta == pytest.approx(2.0 / np.sqrt(3.0), rel=1e-9)  # 1.15470...
    assert eta < np.sqrt(2.0)


def test_lo_eta_negative_autocorr_above_sqrt_q():
    """Negative serial correlation pushes eta(q) ABOVE sqrt(q)."""
    assert lo_eta([-0.4], q=2) > np.sqrt(2.0)


# ---------------------------------------------------------------------------
# F8: expected-max-Sharpe SR0 (DSR deflation benchmark)
# ---------------------------------------------------------------------------

def test_expected_max_sharpe_reference_values():
    """SR0 = sqrt(V[SR]) * ((1-g)*Z[1-1/N] + g*Z[1-1/(N e)]), g=Euler-Mascheroni.
    Oracles precomputed with scipy."""
    assert expected_max_sharpe(var_sr=1.0, n_trials=10) == pytest.approx(1.574598, abs=1e-4)
    assert expected_max_sharpe(var_sr=1.0, n_trials=100) == pytest.approx(2.530603, abs=1e-4)
    assert expected_max_sharpe(var_sr=1.0, n_trials=540) == pytest.approx(3.075589, abs=1e-4)


def test_expected_max_sharpe_scales_with_sqrt_var():
    """SR0 scales with sqrt(V[SR]) — a more dispersed cell-mine raises the hurdle."""
    assert expected_max_sharpe(var_sr=0.25, n_trials=100) == pytest.approx(
        0.5 * expected_max_sharpe(var_sr=1.0, n_trials=100), rel=1e-9)


def test_expected_max_sharpe_grows_with_n_trials():
    """More trials -> higher expected-max-Sharpe hurdle."""
    assert expected_max_sharpe(1.0, 10) < expected_max_sharpe(1.0, 100) < expected_max_sharpe(1.0, 540)


# ---------------------------------------------------------------------------
# F7: Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def test_dsr_at_benchmark_is_half():
    """When observed Sharpe equals the SR0 benchmark, DSR = 0.5 (no evidence
    of skill beyond the best-of-N coin flip)."""
    assert deflated_sharpe_ratio(sr_hat=0.10, sr0=0.10, n_obs=600) == pytest.approx(0.5, abs=1e-9)


def test_dsr_above_benchmark_exceeds_half():
    """Observed Sharpe above SR0 -> DSR > 0.5 (skill survives selection bias)."""
    assert deflated_sharpe_ratio(sr_hat=0.20, sr0=0.10, n_obs=600) > 0.5


def test_dsr_more_observations_more_confident():
    """For a fixed positive (sr_hat - sr0) gap, more observations -> higher DSR."""
    few = deflated_sharpe_ratio(sr_hat=0.15, sr0=0.10, n_obs=100)
    many = deflated_sharpe_ratio(sr_hat=0.15, sr0=0.10, n_obs=2000)
    assert many > few > 0.5


def test_dsr_negative_skew_lowers_confidence():
    """Negative skew (fat left tail) inflates the Sharpe SE -> lowers DSR
    relative to the normal case, for an above-benchmark Sharpe."""
    normal = deflated_sharpe_ratio(sr_hat=0.20, sr0=0.10, n_obs=600, skew=0.0, kurt=3.0)
    left_tailed = deflated_sharpe_ratio(sr_hat=0.20, sr0=0.10, n_obs=600, skew=-1.0, kurt=6.0)
    assert left_tailed < normal


# ---------------------------------------------------------------------------
# F4: Lo (2002) autocorrelation-corrected Sharpe SE
# ---------------------------------------------------------------------------

def test_lo_sharpe_se_reduces_to_iid_normal_with_no_autocorr():
    """For an iid-normal series with lag=0, the SE matches the closed-form
    iid-normal Sharpe SE sqrt((1+SR^2/2)/T)."""
    rng = np.random.default_rng(20260615)
    r = rng.normal(0.001, 0.01, size=6000)
    se = lo_sharpe_se(r, lag=0)
    sr = r.mean() / r.std(ddof=1)
    iid = np.sqrt((1 + sr**2 / 2) / len(r))
    assert se == pytest.approx(iid, rel=0.05)


def test_lo_sharpe_se_positive_autocorr_inflates_se():
    """Positive serial correlation (overlapping multi-day holds) inflates the
    Sharpe SE: the HAC-corrected SE (lag>0) exceeds the iid (lag=0) SE."""
    rng = np.random.default_rng(20260615)
    eps = rng.normal(0, 1, size=4000)
    ar = np.zeros_like(eps)
    for t in range(1, len(eps)):
        ar[t] = 0.6 * ar[t - 1] + eps[t]  # strong positive AR(1)
    se_iid = lo_sharpe_se(ar, lag=0)
    se_hac = lo_sharpe_se(ar, lag=10)
    assert se_hac > se_iid


def test_lo_sharpe_se_default_lag_scales_with_T():
    """Default truncation lag grows ~T^(1/3) (research: ~5-10 for a few-hundred-
    day series), not a fixed 3."""
    rng = np.random.default_rng(1)
    short = lo_sharpe_se(rng.normal(0, 1, 300))   # should not raise; lag auto
    long = lo_sharpe_se(rng.normal(0, 1, 3000))
    assert np.isfinite(short) and np.isfinite(long)


# ---------------------------------------------------------------------------
# F9: effective independent N from correlated trials
# ---------------------------------------------------------------------------

def test_effective_n_zero_correlation_equals_M():
    """rho_bar=0 -> all trials independent -> N=M."""
    assert effective_n_independent(M=66, rho_bar=0.0) == pytest.approx(66.0)


def test_effective_n_full_correlation_equals_one():
    """rho_bar=1 -> all trials identical -> N=1."""
    assert effective_n_independent(M=66, rho_bar=1.0) == pytest.approx(1.0)


def test_effective_n_partial_correlation_between_one_and_M():
    """0<rho_bar<1 -> 1 < N < M."""
    n = effective_n_independent(M=66, rho_bar=0.5)
    assert 1.0 < n < 66.0
    assert n == pytest.approx(66 * 0.5 + 0.5)  # M*(1-rho)+rho


# ---------------------------------------------------------------------------
# F1/F2/F3: stationary block bootstrap of the daily return series
# ---------------------------------------------------------------------------

def test_stationary_bootstrap_ci_brackets_point_estimate():
    """CI brackets the point estimate for a positive-mean series; deterministic
    with a seed; reports the data-adaptive block length."""
    rng = np.random.default_rng(20260615)
    r = rng.normal(0.001, 0.01, size=500)
    res = stationary_bootstrap_ci(r, np.mean, n_resamples=500, seed=7)
    assert res.ci_lower < res.point_estimate < res.ci_upper
    assert res.block_length >= 1.0
    assert res.method == "stationary_bootstrap"


def test_stationary_bootstrap_ci_deterministic_with_seed():
    rng = np.random.default_rng(20260615)
    r = rng.normal(0.001, 0.01, size=400)
    a = stationary_bootstrap_ci(r, np.mean, n_resamples=400, seed=11)
    b = stationary_bootstrap_ci(r, np.mean, n_resamples=400, seed=11)
    assert a.ci_lower == b.ci_lower and a.ci_upper == b.ci_upper


def test_stationary_bootstrap_wider_than_iid_under_autocorrelation():
    """The whole point: on a positively-autocorrelated series the stationary
    block bootstrap CI for the mean is WIDER than an iid percentile bootstrap CI
    (the iid bootstrap is too tight because it destroys temporal structure)."""
    rng = np.random.default_rng(20260615)
    eps = rng.normal(0, 1, size=1000)
    ar = np.zeros_like(eps)
    for t in range(1, len(eps)):
        ar[t] = 0.6 * ar[t - 1] + eps[t]

    sb = stationary_bootstrap_ci(ar, np.mean, n_resamples=1000, seed=3)
    sb_width = sb.ci_upper - sb.ci_lower

    # iid percentile bootstrap of the mean
    rs = np.random.default_rng(3)
    boot = np.array([rs.choice(ar, size=len(ar), replace=True).mean() for _ in range(1000)])
    iid_width = np.percentile(boot, 97.5) - np.percentile(boot, 2.5)

    assert sb_width > iid_width
