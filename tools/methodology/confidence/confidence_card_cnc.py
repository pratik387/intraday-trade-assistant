"""CNC/MTF confidence card — valid for multi-day-hold, daily-rebalanced,
cross-sectional reversion books whose per-trade observations are NOT independent.

This is the CNC counterpart to the intraday `confidence_card.py`. It replaces the
intraday card's invalid assumptions (iid per-trade BCa bootstrap; sqrt(252) Sharpe;
Harvey-Liu Bonferroni) with the deep-research recipe
(specs/2026-06-15-cnc-confidence-card-methodology-research.md):

  - PF / expectancy CI from a STATIONARY BLOCK BOOTSTRAP of the daily MTM
    portfolio-return series (Politis-Romano 1994; Politis-White 2004 block length).
  - Sharpe with Lo (2002) autocorrelation-corrected SE and eta(q) time-aggregation.
  - Deflated Sharpe Ratio selection-bias haircut (Bailey & Lopez de Prado 2014),
    fed the cross-sectional variance of the cell-mine trial Sharpes and the number
    of INDEPENDENT trials.

The card produces INTERVALS + a DSR probability, not a binary verdict — same
philosophy as the intraday card. The researcher applies judgment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _sps

from tools.methodology.confidence.cnc_daily_returns import build_daily_portfolio_returns
from tools.methodology.confidence.cnc_stats import (
    CNCBootstrapResult,
    deflated_sharpe_ratio,
    effective_n_independent,
    expected_max_sharpe,
    lo_eta,
    lo_sharpe_se,
    stat_mean,
    stat_pf,
    stationary_bootstrap_ci,
)

_TRADING_DAYS = 252


@dataclass(frozen=True)
class CNCCard:
    setup_name: str
    aggregate: str           # 'mean' (equal-weight) | 'sum' (fixed-Rs/slot book)
    n_signal_days: int
    n_trades_disc: int
    # Discovery aggregate (block-bootstrap CI on the daily portfolio-return series)
    pf_ci: CNCBootstrapResult
    exp_ci: CNCBootstrapResult
    # Sharpe (per-period daily, autocorrelation-corrected SE, eta(q) annualized)
    sharpe_per_period: float
    sharpe_se: float
    sharpe_annualized: float
    eta_q: float
    # Deflated Sharpe Ratio haircut
    dsr: float
    sr_hat_dsr: float        # locked-cell Sharpe on the trial-field basis (fed to DSR)
    sr0: float
    var_trial_sharpe: float
    m_trials: int
    rho_bar: float
    effective_n: float
    skew: float
    kurt: float
    # one-shot validators
    oos_pf: float
    ho_pf: float
    oos_n_days: int
    ho_n_days: int


def _sample_autocorrs(x: np.ndarray, max_lag: int) -> list:
    """Sample autocorrelations rho_1..rho_{max_lag} of a 1-D series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return [0.0] * max_lag
    dev = x - x.mean()
    denom = float(np.sum(dev ** 2))
    if denom == 0:
        return [0.0] * max_lag
    out = []
    for k in range(1, max_lag + 1):
        if k >= n:
            out.append(0.0)
        else:
            out.append(float(np.sum(dev[k:] * dev[:-k]) / denom))
    return out


def compute_cnc_card(
    *,
    setup_name: str,
    trades_disc: pd.DataFrame,
    trades_oos: pd.DataFrame,
    trades_ho: pd.DataFrame,
    panel: pd.DataFrame,
    k_hold: int,
    cost: float,
    trial_sharpes: Sequence[float],
    m_trials: int,
    rho_bar: float,
    sr_hat_dsr: Optional[float] = None,
    n_obs_dsr: Optional[int] = None,
    aggregate: str = "mean",
    n_resamples: int = 5000,
    seed: int = 20260615,
) -> CNCCard:
    """Compute a CNC confidence card from locked-cell ledgers + the daily panel.

    Args:
        trades_disc/oos/ho: per-trade ledgers (signal_date, symbol) per period.
        panel: clean daily OHLC panel (symbol, date, open, close, ...).
        k_hold: hold horizon K (entry T+1 open -> exit close T+1+K).
        cost: round-trip cost fraction charged on the entry day.
        trial_sharpes: per-cell Discovery Sharpes from the cell-mine sweep (for
            the DSR cross-sectional variance V[SR]).
        m_trials: raw number of trials swept (cell-mine M).
        rho_bar: mean pairwise correlation of trials -> effective independent N.
        sr_hat_dsr: locked-cell Sharpe on the SAME basis as `trial_sharpes`
            (basis-consistent SR_hat for the DSR). Defaults to the MTM-daily
            per-period Sharpe if not supplied.
        n_obs_dsr: number of observations behind `sr_hat_dsr` (e.g. signal-days).
            Defaults to the count of daily MTM observations.
    """
    daily = build_daily_portfolio_returns(trades_disc, panel, k_hold=k_hold, cost=cost, aggregate=aggregate)
    r = daily.to_numpy()

    pf_ci = stationary_bootstrap_ci(r, stat_pf, n_resamples=n_resamples, seed=seed)
    exp_ci = stationary_bootstrap_ci(r, stat_mean, n_resamples=n_resamples, seed=seed)

    # Sharpe (per-period daily) + Lo autocorrelation-corrected SE + eta(q) annualization.
    if len(r) >= 2 and r.std(ddof=1) > 0:
        sr_pp = float(r.mean() / r.std(ddof=1))
    else:
        sr_pp = 0.0
    sr_se = lo_sharpe_se(r)
    acf = _sample_autocorrs(r, _TRADING_DAYS - 1)
    try:
        eta_q = lo_eta(acf, _TRADING_DAYS)
    except ValueError:
        eta_q = float(np.sqrt(_TRADING_DAYS))
    sr_ann = sr_pp * eta_q

    # Deflated Sharpe Ratio haircut.
    var_sr = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1)) if len(trial_sharpes) > 1 else 0.0
    eff_n = effective_n_independent(m_trials, rho_bar)
    sr0 = expected_max_sharpe(var_sr, eff_n)
    skew = float(_sps.skew(r)) if len(r) > 2 else 0.0
    kurt = float(_sps.kurtosis(r, fisher=False)) if len(r) > 3 else 3.0
    # DSR SR_hat must share the basis of the trial-Sharpe field (recipe F7/F8).
    dsr_sr = sr_hat_dsr if sr_hat_dsr is not None else sr_pp
    dsr_n = n_obs_dsr if n_obs_dsr is not None else len(r)
    dsr = deflated_sharpe_ratio(dsr_sr, sr0, max(2, dsr_n), skew=skew, kurt=kurt) if dsr_n >= 2 else float("nan")

    # One-shot OOS / Holdout PF on their daily series.
    daily_oos = build_daily_portfolio_returns(trades_oos, panel, k_hold=k_hold, cost=cost, aggregate=aggregate)
    daily_ho = build_daily_portfolio_returns(trades_ho, panel, k_hold=k_hold, cost=cost, aggregate=aggregate)
    oos_pf = stat_pf(daily_oos.to_numpy()) if len(daily_oos) else float("nan")
    ho_pf = stat_pf(daily_ho.to_numpy()) if len(daily_ho) else float("nan")

    return CNCCard(
        setup_name=setup_name,
        aggregate=aggregate,
        n_signal_days=int(pd.to_datetime(trades_disc["signal_date"]).nunique()) if len(trades_disc) else 0,
        n_trades_disc=int(len(trades_disc)),
        pf_ci=pf_ci, exp_ci=exp_ci,
        sharpe_per_period=sr_pp, sharpe_se=sr_se, sharpe_annualized=sr_ann, eta_q=eta_q,
        dsr=dsr, sr_hat_dsr=dsr_sr, sr0=sr0, var_trial_sharpe=var_sr, m_trials=m_trials, rho_bar=rho_bar,
        effective_n=eff_n, skew=skew, kurt=kurt,
        oos_pf=oos_pf, ho_pf=ho_pf,
        oos_n_days=int(len(daily_oos)), ho_n_days=int(len(daily_ho)),
    )


def render_cnc_card(card: CNCCard) -> str:
    """Render a CNC confidence card as markdown."""
    c = card
    lines = [
        f"# CNC CONFIDENCE CARD: {c.setup_name}",
        "",
        f"**Discovery:** {c.n_trades_disc:,} trades over {c.n_signal_days:,} signal-days; "
        f"daily portfolio-return obs feed all statistics.",
        "",
        "## Aggregate — stationary block bootstrap of the daily portfolio-return series",
        "(Politis-Romano 1994; Politis-White 2004 adaptive block length)",
        "",
        f"- **Profit Factor:** {c.pf_ci.point_estimate:.3f}  "
        f"CI [{c.pf_ci.ci_lower:.3f}, {c.pf_ci.ci_upper:.3f}]  "
        f"(block≈{c.pf_ci.block_length:.1f}d, B={c.pf_ci.n_resamples})",
        f"- **Expectancy (daily ret):** {c.exp_ci.point_estimate:+.5f}  "
        f"CI [{c.exp_ci.ci_lower:+.5f}, {c.exp_ci.ci_upper:+.5f}]",
        "",
        "## Sharpe — Lo (2002) autocorrelation-corrected",
        "",
        f"- **Per-period (daily) Sharpe:** {c.sharpe_per_period:+.4f}  "
        f"(HAC SE {c.sharpe_se:.4f}; t≈{(c.sharpe_per_period / c.sharpe_se) if c.sharpe_se else float('nan'):+.2f})",
        f"- **Annualized (eta(q), NOT sqrt-q):** {c.sharpe_annualized:+.3f}  "
        f"(eta={c.eta_q:.2f} vs sqrt(252)={np.sqrt(_TRADING_DAYS):.2f}; skew {c.skew:+.2f}, kurt {c.kurt:.2f})",
        "",
        "## Selection-bias haircut — Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014)",
        "",
        f"- **Trials:** M={c.m_trials} swept, rho_bar={c.rho_bar:.2f} -> "
        f"**effective independent N={c.effective_n:.1f}**",
        f"- **SR0 (expected max of N unskilled):** {c.sr0:.4f}  "
        f"(V[trial SR]={c.var_trial_sharpe:.5f})",
        f"- **Locked-cell SR_hat (trial-field basis):** {c.sr_hat_dsr:+.4f}",
        f"- **DSR = P(true SR > SR0):** {c.dsr:.3f}",
        "",
        "## One-shot validators (cell locked on Discovery)",
        "",
        f"- **OOS PF:** {c.oos_pf:.3f}  (n_days={c.oos_n_days})",
        f"- **Holdout PF:** {c.ho_pf:.3f}  (n_days={c.ho_n_days})",
        "",
        "---",
        "_Recipe: specs/2026-06-15-cnc-confidence-card-methodology-research.md._",
        "_Stationary block bootstrap + Lo(2002) HAC Sharpe + Deflated Sharpe Ratio._",
    ]
    return "\n".join(lines)
