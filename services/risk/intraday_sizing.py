"""Position sizing for the intraday book — one path, one clamp, no fall-through.

Justified STRUCTURALLY, not by a P&L backtest. An earlier version of this
docstring cited an empirical case (corr(notional, return) = -0.136, "the two
largest setups are the two biggest losers") that was an artifact of a
measurement bug: P&L had been summed over final exit legs only, discarding
partial T1 exits, which are disproportionately winners. Corrected on the same
120 trades / 42 sessions:

    book net P&L        Rs +10,182  (not -1,643)      PF 1.18 (not 0.97)
    corr(notional, ret) -0.057      (not -0.136)
    above-median notional +0.385%/trade  vs below-median +0.200%

So size was NOT systematically landing on bad trades. What remains true, and
what this module exists for, is structural:

1. **Fall-through.** `sizing_mode` is REQUIRED per setup and validated; missing
   or unknown raises. or_window_failure_fade_short carried a Rs112,822 median
   notional — 3.8x the other setups — purely because it declared no mode and
   inherited `qty = risk / risk_per_share`, where a tight stop mints a huge
   position. Nobody chose that size, and that setup does lose money
   (-Rs3,406, -0.432%/trade). Sizing must be a decision (CLAUDE.md rule 1).

2. **One path.** Sizing happened in the orchestrator and was then silently
   re-done in the executor for notional-mode setups. Two paths, one plan.

3. **No notional ceiling.** capital_management.max_allocation_per_trade applies
   20% to MARGIN, so at 5x MIS it permits Rs500k of notional per trade and
   never bound. The largest observed position was Rs182,970 — 37% of capital
   in one intraday trade. Every mode now passes through the same
   [min, max] notional clamp; that is the tail control.

`vol_target` sizes inversely to ex-ante ATR so each position contributes equal
rupee risk. It is implemented and unit-tested but ACTIVE ON NO SETUP: it needs
sigma measured at SIGNAL time, and the unconditional ATR distribution is the
wrong population (all bars median 0.335%, opening hour 0.652%, yet the first
real signal through this path sized on 3.17%). See SIZING_OBS in
plan_orchestrator for how that distribution gets collected.

Nothing here is fitted to returns: ATR is an input and the formulae are
identities. Correct sizing does not manufacture an edge — the book's is
positive but not yet significant (mean +0.293%/trade, t=0.92, 95% CI
[-0.329%, +0.926%], n=120).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

VOL_TARGET = "vol_target"
NOTIONAL = "notional"
RISK = "risk"
VALID_MODES = (VOL_TARGET, NOTIONAL, RISK)


class SizingConfigError(ValueError):
    """Raised when a setup's sizing configuration is missing or unusable."""


@dataclass(frozen=True)
class IntradaySizingResult:
    qty: int
    notional_inr: float
    sigma_pct: Optional[float]
    mode: str
    reason: str  # 'ok' | 'atr_missing' | 'below_min_notional' | 'qty_zero'
    clamped: Optional[str] = None  # 'min' | 'max' | None


def sigma_pct_from_atr(*, atr: float, price: float) -> Optional[float]:
    """Ex-ante volatility as a percent of price. None when unusable.

    ATR here is the 5m-bar ATR carried on the plan (`indicators.atr`), which is
    the right horizon for an intraday hold: position risk over the hold scales
    with intraday range, not with daily close-to-close vol.
    """
    if atr is None or price is None:
        return None
    try:
        a, p = float(atr), float(price)
    except (TypeError, ValueError):
        return None
    if a <= 0 or p <= 0:
        return None
    return 100.0 * a / p


def resolve_sizing_mode(setup_cfg: dict, setup_name: str) -> str:
    """Read and validate `sizing_mode`. Missing/unknown is an error, not a default."""
    mode = setup_cfg.get("sizing_mode")
    if mode is None:
        raise SizingConfigError(
            f"setups.{setup_name}.sizing_mode is missing. Declare one of "
            f"{VALID_MODES}. It must never fall through to a default — an "
            f"undeclared mode is how or_window_failure_fade_short ended up "
            f"sized at Rs 112k/trade."
        )
    mode = str(mode)
    if mode not in VALID_MODES:
        raise SizingConfigError(
            f"setups.{setup_name}.sizing_mode={mode!r} is not one of {VALID_MODES}"
        )
    return mode


def size_intraday_position(
    *,
    setup_name: str,
    sizing_mode: str,
    entry_price: float,
    min_notional_inr: float,
    max_notional_inr: float,
    risk_per_share: Optional[float] = None,
    atr: Optional[float] = None,
    vol_risk_budget_inr: Optional[float] = None,
    stop_risk_budget_inr: Optional[float] = None,
    target_notional_pct: Optional[float] = None,
    total_capital_inr: Optional[float] = None,
    book_size_multiplier: float = 1.0,
) -> IntradaySizingResult:
    """Return the qty for one intraday entry under `sizing_mode`.

    vol_target : notional = vol_risk_budget / (sigma/100), sigma = ATR/price.
                 Equalises rupee risk across names of different volatility.
    notional   : notional = target_notional_pct * total_capital. Flat exposure,
                 vol-blind — kept for setups deliberately sized as a fixed slice.
    risk       : qty = stop_risk_budget / risk_per_share. Stop-distance sizing,
                 now an EXPLICIT choice rather than the silent fall-through.

    The two budgets are DIFFERENT quantities and must not be shared:
    `vol_risk_budget_inr` is the rupee 1-SD move per ATR bar (~Rs100 on Rs5L),
    `stop_risk_budget_inr` is the rupee loss if the stop is hit (~Rs1,000).
    Passing one where the other belongs mis-sizes by an order of magnitude.

    All three are clamped to [min_notional_inr, max_notional_inr].

    `book_size_multiplier` scales EVERY mode by one number, applied before the
    clamp. It exists so book-level risk is a single knob rather than an edit to
    each setup's own parameter, and so a paper run at Nx can be rescaled to any
    other size by dividing — the reason it is one multiplier and not per-setup
    tweaks. Rescaling is exact ONLY for trades the capital manager did not
    resize; check CAP_SCALE / CAP_REJECT before dividing.
    """
    if sizing_mode not in VALID_MODES:
        raise SizingConfigError(
            f"{setup_name}: sizing_mode={sizing_mode!r} not in {VALID_MODES}")
    if entry_price is None or float(entry_price) <= 0:
        return IntradaySizingResult(0, 0.0, None, sizing_mode, "qty_zero")
    entry = float(entry_price)
    if max_notional_inr < min_notional_inr:
        raise SizingConfigError(
            f"{setup_name}: max_notional_inr ({max_notional_inr}) < "
            f"min_notional_inr ({min_notional_inr})")

    sigma = sigma_pct_from_atr(atr=atr, price=entry)

    if sizing_mode == VOL_TARGET:
        if vol_risk_budget_inr is None:
            raise SizingConfigError(f"{setup_name}: vol_target needs vol_risk_budget_inr")
        if sigma is None:
            # No ex-ante vol -> we cannot equalise risk. Refuse rather than
            # silently fall back to a vol-blind size.
            return IntradaySizingResult(0, 0.0, None, sizing_mode, "atr_missing")
        notional = float(vol_risk_budget_inr) / (sigma / 100.0)
    elif sizing_mode == NOTIONAL:
        if target_notional_pct is None or total_capital_inr is None:
            raise SizingConfigError(
                f"{setup_name}: notional mode needs target_notional_pct + total_capital_inr")
        notional = float(target_notional_pct) * float(total_capital_inr)
    else:  # RISK
        if stop_risk_budget_inr is None:
            raise SizingConfigError(f"{setup_name}: risk mode needs stop_risk_budget_inr")
        rps = float(risk_per_share or 0.0)
        if rps <= 0:
            return IntradaySizingResult(0, 0.0, sigma, sizing_mode, "qty_zero")
        notional = (float(stop_risk_budget_inr) / rps) * entry

    notional *= float(book_size_multiplier)

    clamped = None
    if notional > max_notional_inr:
        notional, clamped = float(max_notional_inr), "max"
    elif notional < min_notional_inr:
        # Below the floor the position is not worth its costs. Report it so the
        # caller can shadow/skip; do NOT round it up into a size nobody chose.
        return IntradaySizingResult(0, 0.0, sigma, sizing_mode, "below_min_notional", "min")

    qty = int(notional // entry)
    if qty < 1:
        return IntradaySizingResult(0, 0.0, sigma, sizing_mode, "qty_zero", clamped)
    return IntradaySizingResult(qty, round(qty * entry, 2), sigma, sizing_mode, "ok", clamped)
