"""Tests for the confidence framework (Component 1+2+3).

Each component is tested for:
- Correctness on synthetic data with known properties
- Proper handling of edge cases (small n, zero variance, no losers)
- Reproducibility with fixed seed
"""
import numpy as np
import pandas as pd
import pytest

from tools.methodology.confidence.bootstrap_ci import (
    bootstrap_ci, compute_aggregate_ci, stat_pf, stat_expectancy, stat_win_rate,
    CIResult,
)
from tools.methodology.confidence.regime_breakdown import (
    load_regime_schema, assign_regime, compute_per_regime_stats,
)
from tools.methodology.confidence.selection_bias import (
    build_daily_equity_curve, daily_sharpe, compute_effective_N,
    harvey_liu_haircut, analyze_setups_selection_bias,
)


# ---------------------------------------------------------------------------
# Component 1: Bootstrap BCa CI
# ---------------------------------------------------------------------------

def test_pf_stat_basic():
    """Profit factor on simple sample."""
    pnls = np.array([100, 100, -50, -50])
    assert stat_pf(pnls) == 2.0


def test_pf_stat_no_losers():
    pnls = np.array([100, 200, 300])
    assert stat_pf(pnls) == float("inf")


def test_bootstrap_ci_normal_distribution_brackets_point():
    """Bootstrap CI should bracket point estimate for non-degenerate sample."""
    rng = np.random.default_rng(seed=42)
    sample = rng.normal(loc=0.5, scale=1.0, size=200)
    result = bootstrap_ci(sample, stat_expectancy, n_resamples=500, seed=42)
    assert result.ci_lower < result.point_estimate < result.ci_upper


def test_bootstrap_ci_deterministic_with_seed():
    """Same seed → same CI."""
    rng = np.random.default_rng(seed=1)
    sample = rng.normal(loc=0.0, scale=1.0, size=100)
    r1 = bootstrap_ci(sample, stat_expectancy, n_resamples=500, seed=99)
    r2 = bootstrap_ci(sample, stat_expectancy, n_resamples=500, seed=99)
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper


def test_bootstrap_ci_insufficient_data():
    """n < 10 returns degenerate CI."""
    sample = np.array([1, -1, 2])
    r = bootstrap_ci(sample, stat_pf)
    assert r.method == "insufficient_data"
    assert r.ci_lower == r.point_estimate == r.ci_upper


def test_compute_aggregate_ci_from_dataframe():
    """Aggregate CI returns dict with pf/expectancy/win_rate keys."""
    rng = np.random.default_rng(seed=7)
    df = pd.DataFrame({
        "net_pnl_inr": rng.normal(50, 100, size=200).tolist(),
        "signal_date": pd.date_range("2024-01-01", periods=200).date,
    })
    result = compute_aggregate_ci(df, n_resamples=500)
    assert set(result.keys()) == {"pf", "expectancy", "win_rate"}
    assert isinstance(result["pf"], CIResult)


# ---------------------------------------------------------------------------
# Component 2: Per-regime decomposition
# ---------------------------------------------------------------------------

def test_load_regime_schema_returns_7_regimes():
    regimes = load_regime_schema()
    assert len(regimes) == 7
    assert regimes[0].id == "R1"
    assert regimes[-1].id == "R7"


def test_regime_schema_dates_non_overlapping_and_contiguous():
    """Each regime's start = previous regime's end + 1 day (no gap, no overlap)."""
    from datetime import timedelta
    regimes = load_regime_schema()
    for prev, cur in zip(regimes[:-1], regimes[1:]):
        assert cur.start == prev.end + timedelta(days=1), (
            f"Gap/overlap between {prev.id} end {prev.end} and {cur.id} start {cur.start}"
        )


def test_assign_regime_finds_correct_bucket():
    from datetime import date
    regimes = load_regime_schema()
    # Known dates from schema
    r = assign_regime(date(2023, 6, 15), regimes)
    assert r is not None and r.id == "R1"
    r = assign_regime(date(2024, 6, 4), regimes)  # election day
    assert r is not None and r.id == "R2"
    r = assign_regime(date(2025, 1, 15), regimes)  # FII outflow window
    assert r is not None and r.id == "R4"
    r = assign_regime(date(2026, 3, 15), regimes)  # war
    assert r is not None and r.id == "R7"


def test_assign_regime_outside_range_returns_none():
    from datetime import date
    regimes = load_regime_schema()
    r = assign_regime(date(2022, 6, 15), regimes)
    assert r is None
    r = assign_regime(date(2027, 1, 1), regimes)
    assert r is None


def test_compute_per_regime_stats_returns_7_buckets():
    """Even if some regimes have 0 trades, output has 7 entries (one per regime)."""
    rng = np.random.default_rng(seed=11)
    # Trades only in R1 window
    df = pd.DataFrame({
        "signal_date": pd.date_range("2023-06-01", periods=100, freq="D").date,
        "net_pnl_inr": rng.normal(20, 50, size=100).tolist(),
    })
    results = compute_per_regime_stats(df, n_resamples=300)
    assert len(results) == 7
    # R1 should have trades; others should have 0
    r1 = results[0]
    assert r1.regime.id == "R1"
    assert r1.n_trades > 0


# ---------------------------------------------------------------------------
# Component 3: Selection-bias correction
# ---------------------------------------------------------------------------

def test_build_daily_equity_curve_aggregates_by_day():
    df = pd.DataFrame({
        "signal_date": ["2024-01-15", "2024-01-15", "2024-01-16"],
        "net_pnl_inr": [100, 50, -25],
    })
    curve = build_daily_equity_curve(df)
    from datetime import date
    assert curve[date(2024, 1, 15)] == 150
    assert curve[date(2024, 1, 16)] == -25


def test_daily_sharpe_positive_drift():
    """Positive-drift series should have positive Sharpe."""
    rng = np.random.default_rng(seed=3)
    pnl = pd.Series(rng.normal(10, 5, size=252))  # mean=10, std=5 → high Sharpe
    sr = daily_sharpe(pnl)
    assert sr > 5  # very high since mean/std = 2 annualized x sqrt(252) ≈ 31


def test_daily_sharpe_zero_variance():
    pnl = pd.Series([0] * 100)
    assert daily_sharpe(pnl) == 0.0


def test_effective_N_uncorrelated_setups():
    """Independent random setups should cluster into multiple clusters."""
    rng = np.random.default_rng(seed=4)
    dates = pd.date_range("2024-01-01", periods=100).date
    setups = {
        f"setup_{i}": pd.Series(rng.normal(0, 1, size=100), index=dates)
        for i in range(5)
    }
    eff_N, clusters = compute_effective_N(setups, distance_threshold=0.5)
    # 5 independent random series should mostly NOT cluster together
    assert eff_N >= 2


def test_effective_N_perfectly_correlated_setups_cluster_together():
    """Perfectly correlated setups should cluster as effective_N = 1."""
    dates = pd.date_range("2024-01-01", periods=100).date
    base = pd.Series(np.linspace(0, 100, 100), index=dates)
    setups = {f"setup_{i}": base.copy() for i in range(5)}
    eff_N, clusters = compute_effective_N(setups, distance_threshold=0.5)
    # 5 identical series → 1 cluster
    assert eff_N == 1


def test_harvey_liu_haircut_reduces_sharpe_for_M_gt_1():
    """Higher M should reduce adjusted Sharpe."""
    rng = np.random.default_rng(seed=5)
    pnl = pd.Series(rng.normal(2, 5, size=250))  # moderate-edge series

    r_m1 = harvey_liu_haircut("test", pnl, M=1)
    r_m10 = harvey_liu_haircut("test", pnl, M=10)

    # Bonferroni with M=10 should haircut more than M=1
    assert r_m10.haircut_pct >= r_m1.haircut_pct


def test_harvey_liu_haircut_preserves_sign_for_negative_sharpe():
    """REGRESSION: negative Sharpe must stay negative after haircut.

    Bug discovered 2026-05-20: prior implementation took abs(t_stat) in
    p-value calc, then back-out via ppf(1 - p/2) which is always positive.
    Result: a setup with raw Sharpe -2.6 had adjusted Sharpe +2.4 (sign flip).
    Fixed by using haircut-factor approach: SR_adj = SR_raw * t_crit_1 / t_crit_M.
    """
    rng = np.random.default_rng(seed=99)
    # Build a negative-Sharpe series
    pnl = pd.Series(rng.normal(-5, 10, size=300))  # negative mean → negative Sharpe
    result = harvey_liu_haircut("losing_setup", pnl, M=8)

    assert result.raw_sharpe < 0, f"setup is winning unexpectedly: {result.raw_sharpe}"
    assert result.adjusted_sharpe < 0, (
        f"adjusted Sharpe flipped sign: raw={result.raw_sharpe}, "
        f"adj={result.adjusted_sharpe}"
    )
    # Adjusted should be CLOSER TO ZERO than raw (correction toward null)
    assert abs(result.adjusted_sharpe) < abs(result.raw_sharpe)


def test_harvey_liu_haircut_factor_sensible():
    """Haircut factor for M=8 Bonferroni should be ~0.7 (well-known result)."""
    rng = np.random.default_rng(seed=77)
    pnl = pd.Series(rng.normal(2, 5, size=250))
    result = harvey_liu_haircut("test", pnl, M=8)
    factor = result.adjusted_sharpe / result.raw_sharpe
    # For M=8, two-tailed alpha=0.05: t_crit_1 / t_crit_8 ≈ 1.96/2.72 ≈ 0.72
    # Allow some variance (T-distribution with finite df vs normal)
    assert 0.65 < factor < 0.85


def test_analyze_setups_selection_bias_e2e():
    """End-to-end on multiple synthetic setups."""
    rng = np.random.default_rng(seed=6)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    setups = {
        "winning": pd.DataFrame({
            "signal_date": dates.date,
            "net_pnl_inr": rng.normal(50, 100, size=200).tolist(),
        }),
        "losing": pd.DataFrame({
            "signal_date": dates.date,
            "net_pnl_inr": rng.normal(-30, 100, size=200).tolist(),
        }),
    }
    results, eff_N, _ = analyze_setups_selection_bias(setups)
    assert "winning" in results
    assert "losing" in results
    # Winning setup should have positive raw Sharpe
    assert results["winning"].raw_sharpe > 0
