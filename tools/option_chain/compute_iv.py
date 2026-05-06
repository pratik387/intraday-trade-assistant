"""Black-Scholes implied volatility solver via Brent's method.

Used to back out IV from observed Indian equity option LTPs (NSE bhavcopy
provides only LTP/OI/settlement, never IV). Risk-free rate defaults to 7%
which is the long-run 10Y G-sec average; the result is largely insensitive
to this choice for short-dated ATM options.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _bs_price(sigma: float, spot: float, strike: float, ttm_yrs: float,
              r: float, option_type: str) -> float:
    """Black-Scholes European option price (no dividends)."""
    if sigma <= 0 or ttm_yrs <= 0 or spot <= 0 or strike <= 0:
        return float("nan")
    sqrt_t = math.sqrt(ttm_yrs)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * ttm_yrs) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if option_type.upper() == "CE":
        return spot * norm.cdf(d1) - strike * math.exp(-r * ttm_yrs) * norm.cdf(d2)
    # PE
    return strike * math.exp(-r * ttm_yrs) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def implied_vol_bs(option_price: float, spot: float, strike: float,
                   ttm_days: float, risk_free_rate: float = 0.07,
                   option_type: str = "CE") -> float:
    """Newton-Raphson-style IV solver using Brent's bracketing method.

    Returns NaN on no convergence, on inputs that violate arbitrage bounds,
    or on degenerate inputs.
    """
    if option_price is None or not np.isfinite(option_price) or option_price <= 0:
        return float("nan")
    if spot <= 0 or strike <= 0 or ttm_days <= 0:
        return float("nan")

    ttm = ttm_days / 365.0
    disc_strike = strike * math.exp(-risk_free_rate * ttm)
    # Arbitrage bounds: CE >= max(S - Ke^-rT, 0); PE >= max(Ke^-rT - S, 0)
    if option_type.upper() == "CE":
        lower = max(spot - disc_strike, 0.0)
        upper = spot
    else:
        lower = max(disc_strike - spot, 0.0)
        upper = disc_strike
    if option_price <= lower + 1e-6 or option_price >= upper - 1e-6:
        return float("nan")

    def f(sigma: float) -> float:
        return _bs_price(sigma, spot, strike, ttm, risk_free_rate, option_type) - option_price

    try:
        return float(brentq(f, 0.01, 5.0, maxiter=100, xtol=1e-5))
    except (ValueError, RuntimeError):
        return float("nan")
