"""Plan-construction helpers shared by every detector's `_build_plan`.

These are the small geometry / validation primitives that every sub7+sub8
detector calls when turning a `StructureEvent` into a sized `TradePlan`:

  Entry-zone geometry
    - compute_entry_zone(entry, bias, zone_pct, zone_mode)
        Build a symmetric or directional fill window around the reference
        entry. Per-setup pct + mode are mandatory config keys (no defaults).

  Stop-loss validators (fail-fast — raise instead of returning bool)
    - assert_sl_outside_entry_zone(entry_zone, hard_sl, bias)
        Reject plans where the SL lands inside the entry zone — that means
        a fill at the wrong edge of the zone places the SL on the profit
        side of entry. The trade is structurally broken; do not trade.
    - enforce_min_stop_distance(entry, hard_sl, min_stop_pct)
        Reject plans where the stop is unreasonably tight (qty inflation
        guard — Discovery-Phase-1 hit qty=99,999 from pivot-anchored stops
        landing 0.01% from entry).

  Level/indicator lookups (NaN-safe)
    - safe_level_get(levels, key, fallback)
        dict.get returns NaN when the key exists with NaN value (e.g. ORH
        before the OR completes); this returns the fallback in that case.
    - calculate_structure_stop(entry, bias, atr, sl_atr_mult)
        ATR-based stop fallback. Detectors should normally compute their
        own structural stop; this is the legacy fallback.

History — relocated 2026-04-30 from `pipelines/base_pipeline.py` and
`pipelines/orchestrator.py` as part of the Phase C refactor that flattens
the pipelines/ folder. The validators previously lived inside the
category-pipeline `compute_eligibility()` (~line 1565); the entry-zone
geometry was inline in `_build_plan_from_sub7_detector` (~line 530).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from config.logging_config import get_agent_logger


logger = get_agent_logger()


class PlanRejected(Exception):
    """Raised when a plan fails a structural-validity check.

    Caught by orchestrator/detector to short-circuit plan emission with a
    structured rejection reason. The exception args carry:
        (reason: str, details: dict)
    so callers can log the reason without parsing free-form text.
    """

    def __init__(self, reason: str, **details):
        self.reason = reason
        self.details = details
        super().__init__(reason)


# ---------------------------------------------------------------------------
# Entry-zone geometry
# ---------------------------------------------------------------------------


def compute_entry_zone(
    entry: float,
    bias: str,
    zone_pct: float,
    zone_mode: str,
) -> Tuple[float, float]:
    """Build the entry fill window around `entry`.

    Args:
        entry:    Reference entry price (post any tick-snapping).
        bias:     "long" | "short"
        zone_pct: Half-width (symmetric) or full directional width as a
                  PERCENT, e.g. 0.10 means 0.10% (= 0.0010 fraction).
        zone_mode:
            "symmetric"  → [entry × (1−p), entry × (1+p)] regardless of bias.
                           Used by mean-reverting setups (gap_fade_short,
                           pdh_pdl_reject) to catch noise on both sides.
            "directional"→ Long: [entry, entry × (1+p)]
                           Short: [entry × (1−p), entry]
                           Used by continuation/breakout setups (orb_15) so
                           fills happen only as price moves further in the
                           trade direction.

    Returns:
        (low, high) tuple, rounded to 2 dp.

    Raises:
        ValueError on unknown zone_mode.

    Smoke-22 evidence: a hardcoded ±0.1% symmetric zone starved orb_15 to
    11% fill rate. Per-setup configuration is required — there is no
    sensible default that works across all setups.
    """
    if zone_pct < 0:
        raise ValueError(f"zone_pct must be >= 0, got {zone_pct}")
    pct_frac = zone_pct / 100.0

    if zone_mode == "symmetric":
        return (
            round(entry * (1.0 - pct_frac), 2),
            round(entry * (1.0 + pct_frac), 2),
        )
    if zone_mode == "directional":
        if bias == "long":
            return (round(entry, 2), round(entry * (1.0 + pct_frac), 2))
        return (round(entry * (1.0 - pct_frac), 2), round(entry, 2))

    raise ValueError(
        f"zone_mode={zone_mode!r} invalid — must be 'symmetric' or 'directional'"
    )


# ---------------------------------------------------------------------------
# Stop-loss validators
# ---------------------------------------------------------------------------


def assert_sl_outside_entry_zone(
    entry_zone: Sequence[float],
    hard_sl: float,
    bias: str,
) -> None:
    """Fail-fast guard against SL landing inside the entry zone.

    For longs: SL must be strictly < entry_zone[0] (low edge — below where we
    can possibly fill).
    For shorts: SL must be strictly > entry_zone[1] (high edge — above where
    we can possibly fill).

    If SL is inside the zone, a fill at the unfavourable edge places the SL
    on the WRONG side of entry — exits become profit-takes, losses become
    runaway. We reject the plan rather than auto-widening (auto-widen masks
    detector-geometry bugs).

    Raises:
        PlanRejected("sl_inside_entry_zone", ...) on violation.
    """
    zone_low, zone_high = float(entry_zone[0]), float(entry_zone[1])
    if bias == "long":
        if hard_sl >= zone_low:
            raise PlanRejected(
                "sl_inside_entry_zone",
                bias=bias,
                hard_sl=round(hard_sl, 2),
                zone_low=round(zone_low, 2),
            )
    elif bias == "short":
        if hard_sl <= zone_high:
            raise PlanRejected(
                "sl_inside_entry_zone",
                bias=bias,
                hard_sl=round(hard_sl, 2),
                zone_high=round(zone_high, 2),
            )
    else:
        raise ValueError(f"bias must be 'long' or 'short', got {bias!r}")


def enforce_min_stop_distance(
    entry: float,
    hard_sl: float,
    min_stop_pct: Optional[float],
) -> None:
    """Defence-in-depth qty-inflation guard.

    Detectors with pivot-anchored stops (cpr_mean_revert, vwap, narrow_cpr,
    pdh_pdl_reject) sometimes produce stops 0.01% from entry → qty math
    blows up to 99,999 / 19,999 against a configured Rs 1k risk and a
    single-trade loss is 5-10× the budget.

    `min_stop_pct` is per-setup config; pass None to disable for setups
    whose natural stop floor is high enough (e.g. gap_fade_short's
    1.0025-1.005× gap_high gives ~0.25-0.5% by construction).

    Raises:
        PlanRejected("stop_too_tight", ...) on violation.
    """
    if min_stop_pct is None or entry <= 0:
        return
    risk_per_share = abs(entry - hard_sl)
    stop_pct = (risk_per_share / entry) * 100.0
    if stop_pct < float(min_stop_pct):
        raise PlanRejected(
            "stop_too_tight",
            entry=round(entry, 2),
            hard_sl=round(hard_sl, 2),
            stop_pct=round(stop_pct, 4),
            min_stop_pct=float(min_stop_pct),
        )


# ---------------------------------------------------------------------------
# Level / ATR lookups
# ---------------------------------------------------------------------------


def safe_level_get(levels: Dict, key: str, fallback: float) -> float:
    """NaN-safe `levels.get(key, fallback)`.

    `dict.get(key, fallback)` returns the value when the key is present even
    if the value is NaN. This matters for late server starts where ORH/ORL
    haven't been computed yet and surface as NaN. We treat NaN as "missing"
    and substitute the fallback.
    """
    val = levels.get(key)
    if val is None:
        return fallback
    try:
        if pd.isna(val):
            return fallback
    except (TypeError, ValueError):
        # pd.isna on non-numeric raises — treat as present.
        return val
    return val


def calculate_structure_stop(
    entry_price: float,
    bias: str,
    atr: float,
    sl_atr_mult: float,
) -> float:
    """ATR-based stop fallback — entry ± atr × multiplier.

    Detectors should normally compute their own structural stop (swing
    high/low, prior pivot, gap edge etc.) and only call this when the
    structure has no natural anchor. The Indian retail standard is
    sl_atr_mult ≈ 1.5; per-setup overrides come from setup config.
    """
    if bias == "long":
        return entry_price - (atr * sl_atr_mult)
    return entry_price + (atr * sl_atr_mult)
