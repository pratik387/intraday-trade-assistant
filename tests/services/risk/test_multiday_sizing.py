"""Tests for volatility-targeted, correlation-aware multi-day sizing."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.risk.multiday_sizing import (  # noqa: E402
    implied_book_vol_pct,
    per_position_risk_inr,
    size_position,
)

CAP = 500_000.0


# --------------------------- per_position_risk_inr ---------------------------

def test_independent_case_matches_sqrt_n():
    """rho=0 must reduce to the textbook r = target / sqrt(n)."""
    r = per_position_risk_inr(capital_inr=CAP, daily_vol_target_pct=1.0,
                              n_planned=9, mean_pairwise_corr=0.0)
    assert r == pytest.approx((CAP * 0.01) / 3.0)  # sqrt(9)=3


def test_correlation_shrinks_the_per_position_budget():
    """Correlated positions carry more book risk, so each must be smaller."""
    indep = per_position_risk_inr(capital_inr=CAP, daily_vol_target_pct=1.0,
                                  n_planned=8, mean_pairwise_corr=0.0)
    corr = per_position_risk_inr(capital_inr=CAP, daily_vol_target_pct=1.0,
                                 n_planned=8, mean_pairwise_corr=0.227)
    assert corr < indep
    # measured book rho: 8 positions carry ~1.61x independent risk
    assert indep / corr == pytest.approx(1.61, abs=0.02)


def test_the_measured_cluster_correlation_halves_the_budget():
    """crash2d x zscore is +0.68 — a sleeve at that rho must size right down."""
    indep = per_position_risk_inr(capital_inr=CAP, daily_vol_target_pct=1.0,
                                  n_planned=8, mean_pairwise_corr=0.0)
    sleeve = per_position_risk_inr(capital_inr=CAP, daily_vol_target_pct=1.0,
                                   n_planned=8, mean_pairwise_corr=0.68)
    assert indep / sleeve == pytest.approx(2.40, abs=0.03)


def test_budget_scales_linearly_with_capital_and_target():
    a = per_position_risk_inr(capital_inr=CAP, daily_vol_target_pct=1.0,
                              n_planned=8, mean_pairwise_corr=0.227)
    b = per_position_risk_inr(capital_inr=2 * CAP, daily_vol_target_pct=1.0,
                              n_planned=8, mean_pairwise_corr=0.227)
    c = per_position_risk_inr(capital_inr=CAP, daily_vol_target_pct=2.0,
                              n_planned=8, mean_pairwise_corr=0.227)
    assert b == pytest.approx(2 * a)
    assert c == pytest.approx(2 * a)


@pytest.mark.parametrize("kw", [
    dict(capital_inr=0, daily_vol_target_pct=1.0, n_planned=8, mean_pairwise_corr=0.2),
    dict(capital_inr=CAP, daily_vol_target_pct=0, n_planned=8, mean_pairwise_corr=0.2),
    dict(capital_inr=CAP, daily_vol_target_pct=1.0, n_planned=0, mean_pairwise_corr=0.2),
    dict(capital_inr=CAP, daily_vol_target_pct=1.0, n_planned=8, mean_pairwise_corr=1.0),
])
def test_invalid_inputs_fail_fast(kw):
    with pytest.raises(ValueError):
        per_position_risk_inr(**kw)


# ------------------------------- size_position -------------------------------

_BASE = dict(close=100.0, leverage=1.0, min_notional_inr=10_000.0,
             max_notional_inr=1_000_000.0, fallback_sigma_pct=3.87)


def test_higher_vol_gets_smaller_notional():
    """The core inversion: risk-equal, not rupee-equal."""
    lo = size_position(risk_budget_inr=5_000.0, sigma_pct=2.0, **_BASE)
    hi = size_position(risk_budget_inr=5_000.0, sigma_pct=8.0, **_BASE)
    assert lo.notional_inr > hi.notional_inr
    assert lo.notional_inr / hi.notional_inr == pytest.approx(4.0, rel=0.02)


def test_notional_equals_risk_over_sigma():
    r = size_position(risk_budget_inr=5_000.0, sigma_pct=4.0, **_BASE)
    assert r.notional_inr == pytest.approx(5_000.0 / 0.04, rel=0.01)
    assert r.reason == "ok"


def test_missing_sigma_uses_fallback_and_is_reported():
    """A missing sigma is a data gap, not a signal — trade it, but flag it."""
    for bad in (None, 0.0, -1.0, float("nan")):
        r = size_position(risk_budget_inr=5_000.0, sigma_pct=bad, **_BASE)
        assert r.reason == "sigma_missing"
        assert r.qty > 0


def test_concentration_cap_binds():
    """Vol targeting alone would hand a very low-vol name an unbounded position."""
    kw = dict(_BASE); kw["max_notional_inr"] = 50_000.0
    r = size_position(risk_budget_inr=5_000.0, sigma_pct=0.5, **kw)
    assert r.notional_inr <= 50_000.0


def test_below_min_notional_is_rejected_not_shrunk():
    kw = dict(_BASE); kw["min_notional_inr"] = 100_000.0
    r = size_position(risk_budget_inr=1_000.0, sigma_pct=5.0, **kw)
    assert r.qty == 0 and r.reason == "below_min_notional"


def test_margin_reflects_leverage():
    kw = dict(_BASE); kw["leverage"] = 2.9
    r = size_position(risk_budget_inr=5_000.0, sigma_pct=4.0, **kw)
    assert r.margin_inr == pytest.approx(r.notional_inr / 2.9, rel=0.01)


def test_leverage_below_one_never_inflates_margin():
    kw = dict(_BASE); kw["leverage"] = 0.4
    r = size_position(risk_budget_inr=5_000.0, sigma_pct=4.0, **kw)
    assert r.margin_inr == pytest.approx(r.notional_inr, rel=0.01)


def test_zero_or_negative_close_is_safe():
    kw = dict(_BASE); kw["close"] = 0.0
    assert size_position(risk_budget_inr=5_000.0, sigma_pct=4.0, **kw).qty == 0


def test_expensive_share_that_cannot_fill_one_lot():
    kw = dict(_BASE); kw["close"] = 90_000.0; kw["min_notional_inr"] = 1.0
    r = size_position(risk_budget_inr=100.0, sigma_pct=4.0, **kw)
    assert r.qty == 0 and r.reason == "qty_zero"


# ---------------------------- round-trip identity ----------------------------

def test_sizing_reproduces_the_target_book_vol():
    """The whole point: n positions at the derived budget must hit the target."""
    for n, rho in [(8, 0.227), (13, 0.227), (8, 0.68), (4, 0.0)]:
        r = per_position_risk_inr(capital_inr=CAP, daily_vol_target_pct=0.95,
                                  n_planned=n, mean_pairwise_corr=rho)
        implied = implied_book_vol_pct(risk_budget_inr=r, n_positions=n,
                                       mean_pairwise_corr=rho, capital_inr=CAP)
        assert implied == pytest.approx(0.95, rel=1e-9)


def test_holding_more_positions_than_planned_overshoots_the_target():
    """Guards the reason n_planned must be the CAP, not today's count."""
    r = per_position_risk_inr(capital_inr=CAP, daily_vol_target_pct=0.95,
                              n_planned=8, mean_pairwise_corr=0.227)
    over = implied_book_vol_pct(risk_budget_inr=r, n_positions=16,
                                mean_pairwise_corr=0.227, capital_inr=CAP)
    assert over > 0.95
