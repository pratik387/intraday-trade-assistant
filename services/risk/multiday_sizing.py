"""Volatility-targeted, correlation-aware position sizing for the multi-day book.

Replaces flat `qty = (margin_per_slot * leverage) // close`, which sized every
position at the same rupee notional regardless of the name's volatility or of how
many correlated positions the book already held.

Measured problem it solves (2026-08-12, 23 entry days / 121 deduped positions):
  per-position return SD   3.87%   (worst -10.20%)
  daily book P&L SD        Rs 33,035
  => at 6-8 slots that is 65-87% ANNUALISED vol, vs a 10-15% institutional norm.

Two standard results are used, both textbook (see spec
`specs/2026-08-12-multiday-capital-management-and-selection-plan.md` S10):

1. Volatility targeting — size inversely to the instrument's own volatility so
   each position contributes the same risk, holding book risk constant.
2. Equal-weight constant-correlation portfolio variance:

       sigma_book = r * sqrt( n * (1 + (n-1) * rho) )

   for n positions each contributing risk `r`. Inverting gives the per-position
   risk budget for a target book vol. At the measured rho=0.227, 8 positions carry
   1.61x the risk of 8 independent ones — which is why per-setup slot pools
   understate risk for this book (3 of its 4 setups run rho 0.38-0.68).

NOTHING here is fitted to returns. `rho` and the vol target are inputs; the
formulae are identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math


@dataclass(frozen=True)
class SizingResult:
    qty: int
    notional_inr: float
    margin_inr: float
    risk_budget_inr: float
    reason: str          # 'ok' | 'sigma_missing' | 'below_min_notional' | 'qty_zero'


def per_position_risk_inr(
    *,
    capital_inr: float,
    daily_vol_target_pct: float,
    n_planned: int,
    mean_pairwise_corr: float,
) -> float:
    """Rupee risk (1 SD of daily P&L) each position may contribute.

    Inverts sigma_book = r * sqrt(n * (1 + (n-1) * rho)) for r.

    `n_planned` is the concurrency the budget is spread across — use the planned
    cap, not today's count, or the first position of the day would be sized as if
    it were the only one.
    """
    if capital_inr <= 0:
        raise ValueError("capital_inr must be > 0")
    if daily_vol_target_pct <= 0:
        raise ValueError("daily_vol_target_pct must be > 0")
    if n_planned < 1:
        raise ValueError("n_planned must be >= 1")
    if not (-1.0 < mean_pairwise_corr < 1.0):
        raise ValueError("mean_pairwise_corr must be in (-1, 1)")

    target_inr = capital_inr * (daily_vol_target_pct / 100.0)
    breadth = math.sqrt(n_planned * (1.0 + (n_planned - 1) * mean_pairwise_corr))
    return target_inr / breadth


def size_position(
    *,
    risk_budget_inr: float,
    sigma_pct: Optional[float],
    close: float,
    leverage: float,
    min_notional_inr: float,
    max_notional_inr: float,
    fallback_sigma_pct: float,
) -> SizingResult:
    """Vol-targeted notional -> qty for one candidate.

    notional = risk_budget / (sigma/100)

    A high-vol name therefore gets a SMALLER notional for the same risk. Under the
    old flat sizing an illiquid Rs 919cr name and a large cap received identical
    Rs 1L margin despite very different vol.

    `sigma_pct` is the name's own recent daily-return SD (`sigma20_pct`, already
    carried on selector candidates). When absent we fall back to a configured
    value rather than skipping the trade, because a missing sigma is a data gap,
    not a signal — but the reason is reported so it can be logged and counted.

    `max_notional_inr` is a concentration cap: vol targeting alone would hand a
    very low-vol name an unbounded position.
    """
    if close <= 0:
        return SizingResult(0, 0.0, 0.0, risk_budget_inr, "qty_zero")

    reason = "ok"
    s = sigma_pct
    if s is None or not math.isfinite(float(s)) or float(s) <= 0:
        s = fallback_sigma_pct
        reason = "sigma_missing"

    notional = risk_budget_inr / (float(s) / 100.0)
    notional = min(notional, max_notional_inr)

    if notional < min_notional_inr:
        # Too small to trade economically: the Rs 20/order brokerage cap stops
        # helping below ~Rs 67k notional while delivery STT stays proportional,
        # so sub-minimum positions pay a worse effective cost for the same edge.
        return SizingResult(0, notional, 0.0, risk_budget_inr, "below_min_notional")

    qty = int(notional // close)
    if qty <= 0:
        return SizingResult(0, notional, 0.0, risk_budget_inr, "qty_zero")

    actual_notional = qty * close
    lev = max(float(leverage), 1.0)
    return SizingResult(qty, actual_notional, actual_notional / lev, risk_budget_inr, reason)


def implied_book_vol_pct(
    *,
    risk_budget_inr: float,
    n_positions: int,
    mean_pairwise_corr: float,
    capital_inr: float,
) -> float:
    """Forward check: book daily vol (% of capital) for n positions at that risk.

    Used to log what the sizing actually implies, so drift between the configured
    target and realised book vol is visible rather than assumed.
    """
    if n_positions < 1 or capital_inr <= 0:
        return 0.0
    breadth = math.sqrt(n_positions * (1.0 + (n_positions - 1) * mean_pairwise_corr))
    return 100.0 * (risk_budget_inr * breadth) / capital_inr
