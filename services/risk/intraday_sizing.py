"""Position sizing for the intraday book — one path, one clamp, no fall-through.

Measured problem it solves (2026-08-13, 120 trades / 42 paper sessions on the
ACTIVE intraday setups):

    corr(notional, return%)             -0.136
    or_window_failure_fade_short   Rs 112,822 median  -0.432%/trade
    long_panic_gap_down            Rs  58,646 median  +0.215%/trade
    up_spike_fade_short            Rs  29,856 median  +0.841%/trade

The two largest-sized setups were the two biggest rupee losers, and they were
large BY ACCIDENT: neither declared `sizing_mode`, so both fell through to
`qty = risk_rupees / risk_per_share`, where a tight stop mints a huge position.
The best setup was pinned smallest by a flat 6%-of-capital rule. Same trades,
same total gross exposure, equal-notional instead: Rs +17,629 vs Rs -1,643.

Two defects, both fixed here:

1. **Fall-through.** `sizing_mode` is now REQUIRED per setup and validated; an
   unknown or missing mode raises instead of silently picking one. Sizing is a
   decision, never a default (CLAUDE.md rule 1).

2. **Vol-blindness.** `vol_target` sizes inversely to the name's own
   volatility, so a 2%-ATR name and a 6%-ATR name contribute the same rupee
   risk instead of 3x different risk at equal notional. Sigma is ex-ante — ATR
   as of the signal bar, already on the plan — never realised MAE, which is
   only knowable after the fact.

Every mode passes through the SAME [min, max] notional clamp. That is what
makes an accidental Rs 112k position impossible regardless of stop geometry.

Nothing here is fitted to returns: ATR is an input and the formulae are
identities. Sizing correctly is justified structurally; it does not manufacture
an edge, and the active book's edge is not yet distinguishable from zero
(mean +0.31%/trade, t=0.94, 95% CI [-0.32%, +0.95%], n=120).
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
