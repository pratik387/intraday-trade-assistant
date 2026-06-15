"""Tests for tools.methodology.confidence.confidence_card_cnc.

Orchestrator: ties the daily MTM portfolio-return series (cnc_daily_returns) to the
recipe statistics (cnc_stats) into a single CNC confidence card. Exact math is
covered by the component test modules; here we verify correct WIRING and the
card's internal invariants on a seeded synthetic setup.
"""
import numpy as np
import pandas as pd
import pytest

from tools.methodology.confidence.cnc_stats import (
    expected_max_sharpe, effective_n_independent,
)
from tools.methodology.confidence.confidence_card_cnc import (
    compute_cnc_card, render_cnc_card,
)


def _synthetic_panel(n_days=80, symbols=("A", "B", "C", "D"), seed=20260615):
    """Seeded daily OHLC panel with a mild positive drift."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for sym in symbols:
        price = 100.0
        for dt in dates:
            ret = rng.normal(0.001, 0.02)
            opn = price
            close = price * (1 + ret)
            rows.append(dict(symbol=sym, date=dt, open=opn, close=close))
            price = close
    return pd.DataFrame(rows), dates


def _ledger(dates, symbols, seed=1):
    """A spread of overlapping signals across the period."""
    rng = np.random.default_rng(seed)
    rows = []
    for dt in dates[:-5]:
        for sym in symbols:
            if rng.random() < 0.4:
                rows.append({"signal_date": dt, "symbol": sym})
    return pd.DataFrame(rows)


def _card_kwargs():
    panel, dates = _synthetic_panel()
    syms = ("A", "B", "C", "D")
    return dict(
        setup_name="synthetic_revert_long",
        trades_disc=_ledger(dates, syms, seed=1),
        trades_oos=_ledger(dates, syms, seed=2),
        trades_ho=_ledger(dates, syms, seed=3),
        panel=panel,
        k_hold=2,
        cost=0.00547,
        trial_sharpes=[0.02, 0.05, 0.08, 0.11, 0.04, 0.07, 0.03, 0.09],
        m_trials=40,
        rho_bar=0.5,
    )


def test_compute_card_pf_ci_brackets_point_estimate():
    card = compute_cnc_card(**_card_kwargs())
    assert card.pf_ci.ci_lower <= card.pf_ci.point_estimate <= card.pf_ci.ci_upper
    assert card.pf_ci.method == "stationary_bootstrap"
    assert card.pf_ci.block_length >= 1.0


def test_compute_card_expectancy_ci_brackets_point_estimate():
    card = compute_cnc_card(**_card_kwargs())
    assert card.exp_ci.ci_lower <= card.exp_ci.point_estimate <= card.exp_ci.ci_upper


def test_compute_card_dsr_is_a_probability():
    card = compute_cnc_card(**_card_kwargs())
    assert 0.0 <= card.dsr <= 1.0


def test_compute_card_effective_n_matches_formula_and_caps_M():
    kw = _card_kwargs()
    card = compute_cnc_card(**kw)
    expected_n = effective_n_independent(kw["m_trials"], kw["rho_bar"])
    assert card.effective_n == pytest.approx(expected_n)
    assert card.effective_n <= kw["m_trials"]


def test_compute_card_sr0_uses_trial_sharpe_variance_and_effective_n():
    kw = _card_kwargs()
    card = compute_cnc_card(**kw)
    var_sr = float(np.var(kw["trial_sharpes"], ddof=1))
    n = effective_n_independent(kw["m_trials"], kw["rho_bar"])
    assert card.sr0 == pytest.approx(expected_max_sharpe(var_sr, n))


def test_compute_card_sharpe_se_positive_and_finite():
    card = compute_cnc_card(**_card_kwargs())
    assert card.sharpe_se > 0 and np.isfinite(card.sharpe_se)


def test_compute_card_annualized_sharpe_sign_matches_per_period():
    card = compute_cnc_card(**_card_kwargs())
    assert np.sign(card.sharpe_annualized) == np.sign(card.sharpe_per_period)
    assert np.isfinite(card.sharpe_annualized)


def test_compute_card_reports_oos_and_ho_pf():
    card = compute_cnc_card(**_card_kwargs())
    assert np.isfinite(card.oos_pf)
    assert np.isfinite(card.ho_pf)


def test_compute_card_uses_explicit_dsr_sharpe_when_provided():
    """When sr_hat_dsr / n_obs_dsr are supplied (locked cell's Sharpe on the SAME
    basis as trial_sharpes), the DSR is computed from those, NOT the MTM-daily
    headline Sharpe — so SR_hat and SR0 are basis-consistent (recipe F7/F8)."""
    from tools.methodology.confidence.cnc_stats import deflated_sharpe_ratio
    kw = _card_kwargs()
    card = compute_cnc_card(**kw, sr_hat_dsr=0.30, n_obs_dsr=120)
    var_sr = float(np.var(kw["trial_sharpes"], ddof=1))
    n = effective_n_independent(kw["m_trials"], kw["rho_bar"])
    sr0 = expected_max_sharpe(var_sr, n)
    expected_dsr = deflated_sharpe_ratio(0.30, sr0, 120, skew=card.skew, kurt=card.kurt)
    assert card.dsr == pytest.approx(expected_dsr)
    assert card.sr_hat_dsr == pytest.approx(0.30)


def test_compute_card_aggregate_sum_differs_from_mean():
    """The aggregate mode threads through to the daily-series construction:
    'sum' (fixed-Rs/slot book) yields a different PF than 'mean' (equal-weight)."""
    kw = _card_kwargs()
    mean_card = compute_cnc_card(**kw, aggregate="mean")
    sum_card = compute_cnc_card(**kw, aggregate="sum")
    assert mean_card.pf_ci.point_estimate != sum_card.pf_ci.point_estimate
    assert sum_card.aggregate == "sum"


def test_render_card_contains_key_sections():
    card = compute_cnc_card(**_card_kwargs())
    md = render_cnc_card(card)
    assert "synthetic_revert_long" in md
    assert "Deflated Sharpe" in md
    assert "block bootstrap" in md.lower()
    assert "OOS" in md and "Holdout" in md
