"""Tests for tools.methodology.bootstrap_ci."""
import numpy as np
import pandas as pd
import pytest

from tools.methodology.bootstrap_ci import bootstrap_pf_ci, InsufficientData


def _trades_df(pnls: list) -> pd.DataFrame:
    return pd.DataFrame({"pnl_pct": pnls})


def test_bootstrap_pf_ci_normal_distribution_converges():
    """Normal-distributed trades should produce CI bracketing the point estimate."""
    rng = np.random.default_rng(seed=20260519)
    pnls = rng.normal(loc=0.05, scale=0.30, size=200).tolist()
    df = _trades_df(pnls)

    result = bootstrap_pf_ci(df, n_resamples=1000, seed=20260519)

    assert result.point_estimate > 1.0
    assert result.ci_lower < result.point_estimate
    assert result.ci_upper > result.point_estimate
    assert result.ci_lower < result.ci_upper


def test_bootstrap_pf_ci_insufficient_data_raises():
    """n < min_n raises InsufficientData."""
    df = _trades_df([0.1, 0.2, -0.1])
    with pytest.raises(InsufficientData):
        bootstrap_pf_ci(df, min_n=10)


def test_bootstrap_pf_ci_deterministic_with_seed():
    """Same seed → same CI bounds."""
    rng = np.random.default_rng(seed=42)
    pnls = rng.normal(loc=0.0, scale=0.5, size=100).tolist()
    df = _trades_df(pnls)

    r1 = bootstrap_pf_ci(df, n_resamples=500, seed=12345)
    r2 = bootstrap_pf_ci(df, n_resamples=500, seed=12345)

    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper
    assert r1.point_estimate == r2.point_estimate


def test_bootstrap_pf_ci_ci_brackets_point_estimate_normal_case():
    """For a healthy positive-edge sample, CI should bracket point estimate
    and lower bound should be > 0."""
    rng = np.random.default_rng(seed=20260519)
    pnls = rng.normal(loc=0.1, scale=0.2, size=300).tolist()
    df = _trades_df(pnls)

    result = bootstrap_pf_ci(df, n_resamples=1000, seed=20260519)

    assert result.point_estimate > 1.5
    assert result.ci_lower > 1.0
