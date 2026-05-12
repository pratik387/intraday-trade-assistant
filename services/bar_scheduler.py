"""Bar-level admission scheduling: priority-sorted capital allocation.

2026-05-12 architectural refactor: replaces the gate_chain's implicit
prioritization (rank_score-based) with explicit setup-priority * detector-
quality scoring. When multiple detectors fire on the same bar, this module
collects all decisions, sorts by plan.priority (descending), and asks
CapitalManager + SetupRiskTracker in order. First admit wins capital; later
plans see updated state.

Pure data-driven — no scanner instance state, idempotent given same inputs.
Caller invokes once per bar after structure detection produces plans.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def schedule_admits(
    plans: List[Dict[str, Any]],
    capital_manager,
    setup_risk_tracker,
    ts: pd.Timestamp,
) -> List[Dict[str, Any]]:
    """Sort plans by priority desc, admit in order subject to capital + risk.

    Args:
      plans: list of plan dicts (each with at minimum: symbol, strategy,
        priority, sizing.qty, entry_ref_price, stop.hard, stop.risk_per_share).
      capital_manager: services.capital_manager.CapitalManager instance.
      setup_risk_tracker: services.setup_risk.SetupRiskTracker instance.
      ts: bar timestamp (IST-naive) for risk-tracker time windows.

    Returns:
      List of plans that were admitted (a subset of input, in priority order).
      Plans not admitted are dropped silently — caller can compare lengths
      to detect rejections.

    Note: schedule_admits RESERVES the admission slot in setup_risk_tracker
    (via record_admit) but does NOT call CapitalManager.enter_position. The
    actual margin allocation happens when the executor fires the order
    (TRIGGER -> ENTRY), matching existing pipeline contract.
    """
    if not plans:
        return []

    # Sort by priority desc; stable for equal priorities (preserve emit order).
    sorted_plans = sorted(
        plans, key=lambda p: float(p.get("priority", 0.0)), reverse=True,
    )

    admitted: List[Dict[str, Any]] = []
    for plan in sorted_plans:
        sym = plan.get("symbol", "")
        setup_type = plan.get("strategy", "")

        # 1. Per-setup risk (concurrency + cooloff + rate-limit)
        ok, risk_reason = setup_risk_tracker.can_admit(sym, setup_type, ts)
        if not ok:
            logger.info(f"BAR_SCHED_BLOCK | {sym} | {setup_type} | risk:{risk_reason}")
            continue

        # 2. Capital (portfolio + per-setup budget).
        # CapitalManager.can_enter_position signature:
        #   (symbol, qty, price, cap_segment, mis_leverage, shadow, side, setup_type)
        # Returns (ok: bool, qty_adjusted: int, reason: str).
        sizing = plan.get("sizing") or {}
        qty = int(sizing.get("qty", 0))
        price = float(plan.get("entry_ref_price", 0.0))
        cap_seg = sizing.get("cap_segment", "unknown")
        mis_leverage = sizing.get("mis_leverage")
        bias = (plan.get("bias") or "short").upper()
        side = "BUY" if bias == "LONG" else "SELL"

        cap_ok, _qty_adjusted, cap_reason = capital_manager.can_enter_position(
            symbol=sym, qty=qty, price=price, cap_segment=cap_seg,
            mis_leverage=mis_leverage, side=side, setup_type=setup_type,
        )
        if not cap_ok:
            logger.info(f"BAR_SCHED_BLOCK | {sym} | {setup_type} | capital:{cap_reason}")
            continue

        # Admit
        admitted.append(plan)
        setup_risk_tracker.record_admit(sym, setup_type, ts)

    logger.info(
        f"BAR_SCHED | input={len(plans)} admitted={len(admitted)} "
        f"rejected={len(plans) - len(admitted)}"
    )
    return admitted
