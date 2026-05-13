"""Re-anchor SL/targets to the ACTUAL fill price at trigger time.

Why this exists
---------------
A plan is built at decision time using `entry_ref_price` (the structure's
trigger reference). The actual fill almost always differs — momentum carries
price past the level, slippage moves it the other way. If we don't re-anchor,
two bugs creep in:

  1. R:R distortion. Plan says T1 = entry_ref + 1.5R; if we fill 0.4R late,
     T1 is now only 1.1R from actual fill — exit math thinks we're still
     hunting the original target.
  2. Inverted SL. A pivot-anchored stop computed from entry_ref can land on
     the wrong side of an actual fill that overshot the level (e.g. short
     trade where hard_sl < actual_entry).

Anchor types (set by the detector in plan metadata)
---------------------------------------------------
Each detector tags its plan with `target_anchor_type` so this module knows
what semantics to preserve:

  "structural"  — Targets are real levels (PDH/PDL, prior pivot, gap edge).
                  KEEP target levels unchanged; only update risk_per_share /
                  actual_entry. A better fill improves R:R naturally.
                  Used by: gap_fade_short, pdh_pdl_reject, vwap_pullback,
                  cpr_mean_revert, sweep_reclaim, ema5_alert_pullback,
                  expiry_pin_strike_reversal.

  "r_multiple"  — Targets were computed as entry_ref ± k×R. Recompute as
                  actual_entry ± k×actual_R using the planned k values
                  (the rr field on each target).
                  Used by: any detector that emits R-based targets without
                  a structural anchor (none currently in sub7+sub8).

  "or_range"    — ORB-style: targets are multiples of the opening range.
                  Recompute as actual_entry ± mult × OR_range and re-anchor
                  the stop to ORL/ORH ± buffer.
                  Used by: orb_15 (when enabled — currently DISABLED in
                  config because structural targets perform better in
                  Indian intraday).

History — relocated 2026-04-30 from
`services/execution/trigger_aware_executor.py::_recalculate_targets_for_actual_entry`
and `pipelines/breakout_pipeline.py::recalculate_orb_targets_at_trigger`.
The dispatch was previously a strategy-name string-match; we now dispatch on
the detector-set `target_anchor_type` so the executor doesn't need to know
which detector emitted the plan.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

from config.logging_config import get_agent_logger
from utils.price_utils import round_to_tick


logger = get_agent_logger()


_VALID_ANCHOR_TYPES = ("structural", "r_multiple", "or_range")


class InvertedSLError(Exception):
    """Actual entry crossed the planned hard_sl — the trade is structurally
    broken (entered on the wrong side of the stop). Caller should immediate-
    exit at market and skip target re-anchoring."""

    def __init__(self, side: str, actual_entry: float, hard_sl: float):
        self.side = side
        self.actual_entry = actual_entry
        self.hard_sl = hard_sl
        super().__init__(
            f"Inverted SL: {side} actual_entry={actual_entry:.2f} "
            f"vs hard_sl={hard_sl:.2f}"
        )


class UnknownAnchorTypeError(ValueError):
    """Raised when a detector emits a target_anchor_type the dispatcher
    doesn't recognize.

    History: until 2026-05-13 the dispatcher logged a warning and silently
    fell through to `_recalc_structural`, which KEPT detect-time T1/T2
    unchanged. Three detectors had set "arithmetic" (not in the dispatch
    table) and shipped broken for months — delivery_pct lost ~Rs.70K on
    Discovery alone because T2 was anchored to detect-bar close instead of
    actual entry. Fail-fast prevents that class of bug.
    """

    def __init__(self, anchor_type, symbol=None, strategy=None):
        self.anchor_type = anchor_type
        self.symbol = symbol
        self.strategy = strategy
        super().__init__(
            f"Unknown target_anchor_type={anchor_type!r} for "
            f"{symbol or '?'}/{strategy or '?'}. "
            f"Valid: {_VALID_ANCHOR_TYPES}"
        )


def _validate_sl_orientation(side: str, actual_entry: float, hard_sl: float) -> None:
    """Raise InvertedSLError if the actual fill placed the SL on the profit
    side of entry. For a long, SL must be < actual_entry; for a short, SL
    must be > actual_entry. Equality counts as inverted (zero-risk trade is
    a sizing-math hazard)."""
    if side.upper() == "BUY":
        if hard_sl >= actual_entry:
            raise InvertedSLError(side, actual_entry, hard_sl)
    else:  # SELL / SHORT
        if hard_sl <= actual_entry:
            raise InvertedSLError(side, actual_entry, hard_sl)


def recalculate_targets_for_actual_entry(
    plan: Dict[str, Any],
    actual_entry: float,
    side: str,
) -> Dict[str, Any]:
    """Re-anchor SL/targets/risk to `actual_entry`. Returns a deep-copied
    plan; the input is not mutated.

    Snapshots the decision-time SL and targets into `_decision_sl` /
    `_decision_targets` for downstream EXIT-diagnostic audit trails.

    Raises:
        InvertedSLError when the actual fill is on the wrong side of the
        planned hard_sl. Caller should immediate-exit at market — recalc
        cannot rescue a structurally-broken fill.
    """
    adjusted = copy.deepcopy(plan)

    stop_data = plan.get("stop") or {}
    hard_sl = stop_data.get("hard")
    original_entry = plan.get("entry_ref_price") or plan.get("price")
    original_targets = plan.get("targets", []) or []

    # Snapshot decision-time values for audit (idempotent — only set once)
    if hard_sl is not None and "_decision_sl" not in adjusted:
        adjusted["_decision_sl"] = hard_sl
    if original_targets and "_decision_targets" not in adjusted:
        adjusted["_decision_targets"] = [
            t.get("level") for t in original_targets if t.get("level") is not None
        ]

    if hard_sl is None or original_entry is None:
        logger.info(
            f"TARGET_RECALC_SKIP: missing hard_sl={hard_sl} "
            f"orig_entry={original_entry}"
        )
        return adjusted

    # Fail-fast: a short that filled below its SL is uninvestable, full stop.
    _validate_sl_orientation(side, actual_entry, hard_sl)

    # Treat a missing anchor_type as a plan-builder bug, not a "structural"
    # default. Until 2026-05-13, screener_live.py stripped the field off the
    # exec_item dict, so every plan defaulted to "structural" silently — which
    # nullified r_multiple recalc for delivery_pct / options_vol_iv_rank_revert.
    # Detectors that legitimately want structural preservation must say so
    # explicitly (TradePlan dataclass default already does this).
    anchor_type = plan.get("target_anchor_type")
    if anchor_type is None:
        raise UnknownAnchorTypeError(
            None,
            symbol=plan.get("symbol"),
            strategy=plan.get("strategy"),
        )

    if anchor_type == "structural":
        return _recalc_structural(adjusted, plan, actual_entry, side, hard_sl)
    if anchor_type == "r_multiple":
        return _recalc_r_multiple(
            adjusted, plan, actual_entry, side, hard_sl, original_entry,
            original_targets,
        )
    if anchor_type == "or_range":
        return _recalc_or_range(adjusted, plan, actual_entry, side)

    raise UnknownAnchorTypeError(
        anchor_type,
        symbol=plan.get("symbol"),
        strategy=plan.get("strategy"),
    )


# ---------------------------------------------------------------------------
# Anchor-specific re-calculators
# ---------------------------------------------------------------------------


def _recalc_structural(
    adjusted: Dict[str, Any],
    plan: Dict[str, Any],
    actual_entry: float,
    side: str,
    hard_sl: float,
) -> Dict[str, Any]:
    """Keep target levels; update only rps + actual_entry."""
    if side.upper() == "BUY":
        actual_rps = actual_entry - hard_sl
    else:
        actual_rps = hard_sl - actual_entry

    if actual_rps <= 0:
        # Should already be caught by _validate_sl_orientation; keep the
        # log-and-bail path for any edge case (e.g. integer rounding).
        logger.warning(
            f"STRUCTURAL_RECALC: invalid actual_rps={actual_rps:.4f} "
            f"({plan.get('symbol')}) — keeping decision-time targets"
        )
        return adjusted

    if "stop" in adjusted and isinstance(adjusted["stop"], dict):
        adjusted["stop"]["risk_per_share"] = round(actual_rps, 2)
    adjusted["risk_per_share"] = round(actual_rps, 2)
    adjusted["actual_entry"] = round_to_tick(actual_entry)

    logger.info(
        f"STRUCTURAL_TARGET_PRESERVED: {plan.get('symbol')} "
        f"{plan.get('strategy')} entry "
        f"{plan.get('entry_ref_price')}→{actual_entry}, targets unchanged"
    )
    return adjusted


def _recalc_r_multiple(
    adjusted: Dict[str, Any],
    plan: Dict[str, Any],
    actual_entry: float,
    side: str,
    hard_sl: float,
    original_entry: float,
    original_targets,
) -> Dict[str, Any]:
    """Recompute T1/T2 from actual_entry using planned R-multiples."""
    stop_data = plan.get("stop") or {}
    original_rps = stop_data.get("risk_per_share")
    if original_rps is None or original_rps <= 0:
        logger.info(
            f"R_MULT_RECALC_SKIP: missing original_rps for "
            f"{plan.get('symbol')} — preserving structural"
        )
        return _recalc_structural(adjusted, plan, actual_entry, side, hard_sl)

    if len(original_targets) < 2:
        logger.info(
            f"R_MULT_RECALC_SKIP: <2 targets for {plan.get('symbol')} — "
            f"preserving structural"
        )
        return _recalc_structural(adjusted, plan, actual_entry, side, hard_sl)

    if side.upper() == "BUY":
        actual_rps = actual_entry - hard_sl
    else:
        actual_rps = hard_sl - actual_entry
    if actual_rps <= 0:
        return _recalc_structural(adjusted, plan, actual_entry, side, hard_sl)

    def _r_for(target):
        # Prefer explicit `rr` (planner intent); derive from level only if missing.
        r = target.get("rr") or target.get("r_multiple")
        if r is not None:
            return r
        level = target.get("level")
        if level is None or original_rps <= 0:
            return None
        if side.upper() == "BUY":
            return (level - original_entry) / original_rps
        return (original_entry - level) / original_rps

    t1_r = _r_for(original_targets[0])
    t2_r = _r_for(original_targets[1])
    if t1_r is None or t2_r is None:
        logger.warning(
            f"R_MULT_RECALC: could not derive R-multiples for "
            f"{plan.get('symbol')} — preserving structural"
        )
        return _recalc_structural(adjusted, plan, actual_entry, side, hard_sl)

    if side.upper() == "BUY":
        new_t1 = actual_entry + t1_r * actual_rps
        new_t2 = actual_entry + t2_r * actual_rps
    else:
        new_t1 = actual_entry - t1_r * actual_rps
        new_t2 = actual_entry - t2_r * actual_rps

    # Plan-as-source-of-truth (2026-05-12): preserve per-target qty_pct and
    # action from the original detector emission. Re-anchor only changes the
    # numeric levels; the split/action contract is setup-authoritative.
    adjusted["targets"] = [
        {
            "level": round(new_t1, 2), "name": "T1", "rr": round(t1_r, 2),
            "qty_pct": original_targets[0].get("qty_pct", 0.5),
            "action": original_targets[0].get("action", "partial_exit"),
        },
        {
            "level": round(new_t2, 2), "name": "T2", "rr": round(t2_r, 2),
            "qty_pct": original_targets[1].get("qty_pct", 0.5),
            "action": original_targets[1].get("action", "exit_full"),
        },
    ]
    if "stop" in adjusted and isinstance(adjusted["stop"], dict):
        adjusted["stop"]["risk_per_share"] = round(actual_rps, 2)
    adjusted["risk_per_share"] = round(actual_rps, 2)
    adjusted["actual_entry"] = round_to_tick(actual_entry)

    logger.info(
        f"R_MULT_RECALCULATED: {plan.get('symbol')} {plan.get('strategy')} "
        f"entry {original_entry}→{actual_entry}, "
        f"T1→{new_t1:.2f} ({t1_r:.2f}R), T2→{new_t2:.2f} ({t2_r:.2f}R)"
    )
    return adjusted


def _recalc_or_range(
    adjusted: Dict[str, Any],
    plan: Dict[str, Any],
    actual_entry: float,
    side: str,
) -> Dict[str, Any]:
    """ORB-style recalc: targets are multiples of the OR range, SL is anchored
    to ORL/ORH ± buffer. Reads multipliers from the detector's plan
    metadata (`or_recalc.{t1_mult, t2_mult, sl_buffer_mult, qty_splits}`)
    rather than a category config — Phase C eliminates per-category configs.
    """
    or_cfg = plan.get("or_recalc") or {}
    if not or_cfg.get("enabled", False):
        logger.info(
            f"OR_RANGE_RECALC_DISABLED: {plan.get('symbol')} — "
            f"preserving structural targets"
        )
        return _recalc_structural(
            adjusted, plan, actual_entry, side, plan["stop"]["hard"]
        )

    levels = plan.get("levels") or {}
    orh = levels.get("ORH") or plan.get("orh")
    orl = levels.get("ORL") or plan.get("orl")
    if orh is None or orl is None:
        logger.warning(
            f"OR_RANGE_RECALC: missing ORH/ORL for {plan.get('symbol')} — "
            f"preserving structural"
        )
        return _recalc_structural(
            adjusted, plan, actual_entry, side, plan["stop"]["hard"]
        )

    or_range = orh - orl
    if or_range <= 0:
        logger.warning(
            f"OR_RANGE_RECALC: invalid or_range={or_range} for "
            f"{plan.get('symbol')} — preserving structural"
        )
        return _recalc_structural(
            adjusted, plan, actual_entry, side, plan["stop"]["hard"]
        )

    # Required multipliers — fail-fast if detector forgot to set them.
    t1_mult = or_cfg["t1_mult"]
    t2_mult = or_cfg["t2_mult"]
    sl_buffer_mult = or_cfg["sl_buffer_mult"]
    qty_splits = or_cfg.get("qty_splits") or {}

    if side.upper() == "BUY":
        new_t1 = actual_entry + or_range * t1_mult
        new_t2 = actual_entry + or_range * t2_mult
        new_sl = orl - or_range * sl_buffer_mult
    else:
        new_t1 = actual_entry - or_range * t1_mult
        new_t2 = actual_entry - or_range * t2_mult
        new_sl = orh + or_range * sl_buffer_mult

    new_rps = abs(actual_entry - new_sl)
    if new_rps <= 0:
        logger.warning(
            f"OR_RANGE_RECALC: zero rps for {plan.get('symbol')} — "
            f"preserving structural"
        )
        return _recalc_structural(
            adjusted, plan, actual_entry, side, plan["stop"]["hard"]
        )

    # Re-validate SL orientation against the freshly-anchored stop.
    _validate_sl_orientation(side, actual_entry, new_sl)

    t1_r = (or_range * t1_mult) / new_rps
    t2_r = (or_range * t2_mult) / new_rps

    adjusted["targets"] = [
        {
            "level": round(new_t1, 2), "name": "T1", "rr": round(t1_r, 2),
            "qty_pct": (qty_splits.get("t1") or 0) * 100,
        },
        {
            "level": round(new_t2, 2), "name": "T2", "rr": round(t2_r, 2),
            "qty_pct": (qty_splits.get("t2") or 0) * 100,
        },
    ]
    if "stop" in adjusted and isinstance(adjusted["stop"], dict):
        adjusted["stop"]["hard"] = round(new_sl, 2)
        adjusted["stop"]["risk_per_share"] = round(new_rps, 2)
    adjusted["risk_per_share"] = round(new_rps, 2)
    adjusted["actual_entry"] = round_to_tick(actual_entry)

    logger.info(
        f"OR_RANGE_RECALCULATED: {plan.get('symbol')} entry={actual_entry:.2f}, "
        f"OR={or_range:.2f}, T1={new_t1:.2f} ({t1_r:.2f}R), "
        f"T2={new_t2:.2f} ({t2_r:.2f}R), SL={new_sl:.2f}"
    )
    return adjusted
