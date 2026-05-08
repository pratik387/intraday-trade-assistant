"""Plan orchestrator — single entry point from screener/main.py to detectors.

This module replaces `pipelines/orchestrator.py` (Phase C, 2026-04-30).

What changed
------------
The legacy orchestrator carried a multi-strategy hedge-fund construct:
4 SMC categories (BREAKOUT/LEVEL/REVERSION/MOMENTUM), regime-based risk
budget allocation, ORB priority windows, HARD_BLOCKS, ranked slot
allocation across categories. Every active sub7+sub8 setup uses the SUB7
fast path that BYPASSES every one of those structures, leaving the legacy
machinery as ~1500 lines of dead code.

The new orchestrator is a thin dispatcher:

    candidate (StructureEvent) ──► sub7 detector.plan_*_strategy() ──► plan dict

For multi-symbol selection it ranks by `structural_rr` and applies two
config-driven caps: max_positions_total and skip_duplicate_symbols. Risk-
budget allocation is intentionally deferred — once the gauntlet identifies
which setups survive (Phase 1 pass criteria: PF≥1.10, n≥500, Sharpe>0),
budget allocation will be re-introduced based on real gauntlet confidence
rather than a-priori category guesses.

What's preserved (live-path concerns)
-------------------------------------
- Singleton pattern via `get_orchestrator()` — caller code in screener_live,
  parity_simulator, and main.py uses the same singleton.
- `process_single_candidate` / `process_candidates` / `process_candidates_multi`
  signatures kept identical to ease the import-rename in Commit-1 step 6.
- `return_all_eligible=True` short-circuit on process_candidates — used by
  LiveGateChain (parity path) to see every plan before gate selection.
- planning_log.log_accept / log_reject calls — these feed analytics.jsonl
  and are consumed by every downstream report tool.
- Lazy detector instantiation keyed by setup_type (config/configuration.json
  → `setups.{setup_type}` block).

What's removed (dead with sub7 fast path)
-----------------------------------------
- BasePipeline / Breakout/Level/Reversion/Momentum lazy imports
- regime_risk_budgets, category_constraints, selection_rules dispatch
- ORB priority window score boost
- HARD_BLOCKS regime-based blocking (deleted per project decision —
  surviving setups will absorb regime filters into their own detect() body)
- _is_category_blocked, _is_hard_blocked, _allocate_slots, _get_pipeline

The legacy `pipelines/orchestrator.py` will be removed in Commit 1 after
all importers are rewired.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.logging_config import get_agent_logger, get_planning_logger

# Active detector classes — every active setup is in this map.
# Sub-9 cleanup (2026-05-01): the 6 sub-7/sub-8 candidate detectors
# (orb_15, pdh_pdl_reject, pdh_pdl_sweep_reclaim, gap_and_go_continuation,
# ema5_alert_pullback, camarilla_l3_reversal) were deleted after Phase-1
# validation failure. See specs/2026-05-01-sub-project-9-microstructure-
# first-redesign.md.
from structures.gap_fade_short_structure import GapFadeShortStructure
from structures.expiry_pin_strike_reversal_structure import ExpiryPinStrikeReversalStructure
from structures.circuit_t1_fade_short_structure import CircuitT1FadeShortStructure
from structures.options_vol_iv_rank_revert_structure import OptionsVolIvRankRevertStructure
from structures.capitulation_long_morning_structure import CapitulationLongMorningStructure
from structures.delivery_pct_anomaly_short_structure import DeliveryPctAnomalyShortStructure
from structures.data_models import MarketContext

from services.symbol_metadata import get_cap_segment, get_mis_info
from services.plan_helpers import (
    PlanRejected,
    compute_entry_zone,
    enforce_min_stop_distance,
)


logger = get_agent_logger()


def _planning_logger():
    """Lazy fetch — must run after logging_config has been initialized."""
    return get_planning_logger()


# Setup → detector class. Adding a new setup = one entry here +
# config/configuration.json setups.* block + the detector file.
# Per sub-9 spec §3.3 a new setup requires a passing brief BEFORE code
# is written — don't add entries here ahead of that gate.
_DETECTOR_REGISTRY: Dict[str, Any] = {
    "gap_fade_short": GapFadeShortStructure,
    "expiry_pin_strike_reversal": ExpiryPinStrikeReversalStructure,
    "circuit_t1_fade_short": CircuitT1FadeShortStructure,
    "options_vol_iv_rank_revert": OptionsVolIvRankRevertStructure,
    "capitulation_long_morning": CapitulationLongMorningStructure,
    "delivery_pct_anomaly_short": DeliveryPctAnomalyShortStructure,
}

ACTIVE_SETUPS: frozenset = frozenset(_DETECTOR_REGISTRY.keys())


class OrchestratorConfigError(Exception):
    """Required config key missing from configuration.json — fail-fast."""


# ---------------------------------------------------------------------------
# Cached configuration.json read
# ---------------------------------------------------------------------------

_ROOT_CFG_CACHE: Optional[Dict[str, Any]] = None


def _load_root_config() -> Dict[str, Any]:
    """Module-level cached read of config/configuration.json.

    The orchestrator + detectors share this dict, so detectors instantiated
    here see the same `setups.*` blocks the orchestrator does. Cache is
    keyed by None — call sites that need a fresh read clear the global.
    """
    global _ROOT_CFG_CACHE
    if _ROOT_CFG_CACHE is not None:
        return _ROOT_CFG_CACHE
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "configuration.json",
    )
    with open(cfg_path, encoding="utf-8") as f:
        _ROOT_CFG_CACHE = json.load(f)
    return _ROOT_CFG_CACHE


def _get_selection_rules() -> Dict[str, Any]:
    """Return selection-control knobs from configuration.json (or defaults
    geared for the post-Phase-C single-bucket world).

    Required keys:
      max_positions_total       int
      skip_duplicate_symbols    bool
    Optional:
      min_score_to_select       float (defaults to 0)
    """
    cfg = _load_root_config()
    sel = (cfg.get("orchestrator") or {}).get("selection_rules") or {}
    if "max_positions_total" not in sel or "skip_duplicate_symbols" not in sel:
        # Fall back to legacy risk_budget_config so existing OCI / live
        # configs still drive the orchestrator while we migrate.
        legacy_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "pipelines", "risk_budget_config.json",
        )
        if os.path.exists(legacy_path):
            with open(legacy_path) as f:
                legacy = json.load(f)
            legacy_sel = legacy.get("selection_rules") or {}
            return {
                "max_positions_total": legacy_sel.get("max_positions_total", 5),
                "skip_duplicate_symbols": legacy_sel.get("skip_duplicate_symbols", True),
                "min_score_to_select": legacy_sel.get("min_score_to_select", 0.0),
            }
        raise OrchestratorConfigError(
            "orchestrator.selection_rules.{max_positions_total, "
            "skip_duplicate_symbols} required in configuration.json"
        )
    return {
        "max_positions_total": sel["max_positions_total"],
        "skip_duplicate_symbols": sel["skip_duplicate_symbols"],
        "min_score_to_select": sel.get("min_score_to_select", 0.0),
    }


# ---------------------------------------------------------------------------
# PlanOrchestrator — singleton
# ---------------------------------------------------------------------------


class PlanOrchestrator:
    """Routes candidates to their detectors, builds plans, ranks for multi-
    symbol selection. Every active setup goes through the same path."""

    def __init__(self) -> None:
        self._detectors: Dict[str, Any] = {}
        logger.debug("PlanOrchestrator initialized (sub7/sub8 fast path only)")

    # ---- Detector lookup ------------------------------------------------

    def _get_detector(self, setup_type: str) -> Optional[Any]:
        """Return a cached detector instance for `setup_type`. Detectors are
        instantiated once per process, with the same setup_cfg the
        MainDetector uses (so parameter parity is guaranteed).

        Returns None when the setup is unknown, disabled in config, or
        instantiation raises — all are logged so misconfig surfaces in the
        orchestrator log without breaking the pipeline.
        """
        if setup_type in self._detectors:
            return self._detectors[setup_type]

        cls = _DETECTOR_REGISTRY.get(setup_type)
        if cls is None:
            logger.warning(f"[ORCH] Unknown setup_type={setup_type!r}")
            return None

        try:
            full_cfg = _load_root_config()
            setups_cfg = full_cfg.get("setups") or {}
            setup_cfg = setups_cfg.get(setup_type) or {}
            if not setup_cfg.get("enabled", False):
                logger.debug(f"[ORCH] {setup_type} disabled in configuration.json — skipping")
                return None

            # Inject _setup_name so the detector identifies itself the same
            # way MainDetector does.
            setup_cfg = {**setup_cfg, "_setup_name": setup_type}
            det = cls(setup_cfg)
            self._detectors[setup_type] = det
            logger.debug(f"[ORCH] Instantiated {setup_type} detector")
            return det
        except Exception as exc:
            logger.exception(f"[ORCH] Failed to instantiate {setup_type}: {exc}")
            return None

    # ---- Plan construction ---------------------------------------------

    def build_plan(
        self,
        symbol: str,
        setup_type: str,
        df5m: pd.DataFrame,
        levels: Dict[str, float],
        regime: str,
        now: pd.Timestamp,
        cap_segment: Optional[str] = None,
        daily_df: Optional[pd.DataFrame] = None,
        structure_event: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build an orchestrator-shape plan dict from a sub7/sub8 detector.

        `structure_event` is the StructureEvent that produced this candidate
        in MainDetector. It MUST be provided — the orchestrator never
        re-calls detect() (architectural decision: detection runs exactly
        once per (symbol, bar, setup_type), latches in plan_*_strategy on
        success).
        """
        if structure_event is None:
            logger.warning(
                f"[ORCH] {symbol} {setup_type}: structure_event missing — caller "
                f"must thread it through from MainDetector"
            )
            return None

        detector = self._get_detector(setup_type)
        if detector is None:
            return None

        # ---- MarketContext (mirrors main_detector._build_market_context) ----
        try:
            if df5m is None or len(df5m) == 0:
                return None

            atr_val = (
                float(df5m["atr"].iloc[-1])
                if "atr" in df5m.columns and not pd.isna(df5m["atr"].iloc[-1])
                else float((df5m["high"] - df5m["low"]).tail(14).mean())
            )
            vol_z_val = (
                float(df5m["vol_z"].iloc[-1])
                if "vol_z" in df5m.columns and not pd.isna(df5m["vol_z"].iloc[-1])
                else 0.0
            )
            indicators: Dict[str, float] = {"atr": atr_val, "vol_z": vol_z_val}
            for col in ("vwap", "rsi", "adx"):
                if col in df5m.columns and not pd.isna(df5m[col].iloc[-1]):
                    indicators[col] = float(df5m[col].iloc[-1])

            bar_timestamp = pd.to_datetime(df5m.index[-1])

            # NIFTY 50 spot — needed by expiry_pin_strike_reversal for spot-vs-pin
            # distance. None for non-expiry sessions / non-heavyweight symbols
            # is fine; detector returns no-fire in that case.
            from services.index_spot_loader import get_nifty_spot
            nifty_spot = get_nifty_spot(bar_timestamp)
            if nifty_spot is not None:
                indicators["nifty_spot"] = nifty_spot

            current_price = float(df5m["close"].iloc[-1])
            if cap_segment is None:
                cap_segment = get_cap_segment(symbol)

            context = MarketContext(
                symbol=symbol,
                current_price=current_price,
                timestamp=bar_timestamp,
                df_5m=df5m,
                session_date=bar_timestamp.date(),
                df_daily=daily_df,
                orh=levels.get("ORH"),
                orl=levels.get("ORL"),
                pdh=levels.get("PDH"),
                pdl=levels.get("PDL"),
                pdc=levels.get("PDC"),
                regime=regime,
                cap_segment=cap_segment,
                indicators=indicators,
            )
        except Exception as exc:
            logger.exception(f"[ORCH] {symbol} {setup_type}: MarketContext build failed: {exc}")
            return None

        # ---- Direction (single source of truth: the StructureEvent) ----
        if setup_type.endswith("_long"):
            bias = "long"
        elif setup_type.endswith("_short"):
            bias = "short"
        else:
            evt_ctx = getattr(structure_event, "context", None)
            bias_from_event = (
                (evt_ctx.get("bias") if isinstance(evt_ctx, dict) else None)
                or getattr(structure_event, "side", None)
            )
            if bias_from_event not in ("long", "short"):
                logger.warning(
                    f"[ORCH] {symbol} {setup_type}: event.side / context['bias'] "
                    f"missing or invalid: {bias_from_event!r}"
                )
                return None
            bias = bias_from_event

        # ---- Detector emits the TradePlan ----
        try:
            if bias == "short":
                trade_plan = detector.plan_short_strategy(context, event=structure_event)
            else:
                trade_plan = detector.plan_long_strategy(context, event=structure_event)
        except Exception as exc:
            logger.exception(f"[ORCH] {symbol} {setup_type}: plan_{bias}_strategy raised: {exc}")
            return None

        if trade_plan is None:
            logger.debug(
                f"[ORCH] {symbol} {setup_type}: detector returned None — conditions not met"
            )
            return {
                "eligible": False,
                "reason": "detector_rejected",
                "strategy": setup_type,
                "bias": bias,
            }

        # ---- TradePlan → orchestrator-shape plan dict ----
        return self._assemble_plan_dict(
            symbol=symbol,
            setup_type=setup_type,
            bias=bias,
            regime=regime,
            cap_segment=cap_segment,
            trade_plan=trade_plan,
            df5m=df5m,
            current_price=current_price,
            indicators=indicators,
            atr_val=atr_val,
            vol_z_val=vol_z_val,
            levels=levels,
        )

    def _assemble_plan_dict(
        self,
        symbol: str,
        setup_type: str,
        bias: str,
        regime: str,
        cap_segment: Optional[str],
        trade_plan: Any,
        df5m: pd.DataFrame,
        current_price: float,
        indicators: Dict[str, float],
        atr_val: Optional[float],
        vol_z_val: float,
        levels: Dict[str, float],
    ) -> Dict[str, Any]:
        """Convert a detector TradePlan into the dict shape downstream code
        (gates, execution, analytics) expects. Encapsulates: entry-zone
        construction, min-stop-distance enforcement, position sizing,
        directional-bias multiplier, MIS / cap-segment metadata."""
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        rps = trade_plan.risk_params.risk_per_share
        atr_for_plan = trade_plan.risk_params.atr or atr_val

        # Targets — preserve detector intent, fill in computed rr if missing.
        raw_targets = trade_plan.exit_levels.targets if trade_plan.exit_levels else []
        targets: List[Dict[str, Any]] = []
        for t in raw_targets:
            t_level = t.get("level", 0.0)
            t_rr = t.get("rr", 0.0)
            if t_rr <= 0.0 and rps > 0.0:
                if bias == "short":
                    t_rr = round((entry - t_level) / rps, 2) if entry > t_level else 0.0
                else:
                    t_rr = round((t_level - entry) / rps, 2) if t_level > entry else 0.0
            targets.append({
                "level": round(t_level, 2),
                "rr": round(t_rr, 2),
                "qty_pct": t.get("qty_pct", 1.0),
                "action": t.get("action", "exit_full"),
                "name": t.get("name", "T1"),
            })
        structural_rr = targets[0]["rr"] if targets else 0.0

        # ---- Position sizing (Van Tharp CPR) ----
        root_cfg = _load_root_config()
        try:
            risk_per_trade_rupees = float(root_cfg["risk_per_trade_rupees"])
        except KeyError:
            raise OrchestratorConfigError(
                "risk_per_trade_rupees missing from configuration.json"
            )

        setup_cfg = (root_cfg.get("setups") or {}).get(setup_type) or {}
        if "entry_zone_pct" not in setup_cfg or "entry_zone_mode" not in setup_cfg:
            raise OrchestratorConfigError(
                f"setups.{setup_type}.entry_zone_pct / entry_zone_mode missing"
            )

        # Min-stop floor — defence against pivot-anchored stops landing too
        # tight (Discovery-Phase-1 saw qty=99,999 from rps≈0.01 / Rs100 stocks).
        try:
            enforce_min_stop_distance(
                entry=entry,
                hard_sl=hard_sl,
                min_stop_pct=setup_cfg.get("min_stop_distance_pct"),
            )
        except PlanRejected as e:
            logger.warning(
                f"[ORCH] {symbol} {setup_type}: stop_pct={e.details.get('stop_pct')} "
                f"< floor {e.details.get('min_stop_pct')} — rejecting plan"
            )
            return {
                "eligible": False,
                "reason": e.reason,
                "strategy": setup_type,
                "bias": bias,
                "details": [f"stop_pct={e.details.get('stop_pct')}",
                            f"min={e.details.get('min_stop_pct')}"],
            }

        qty = int(risk_per_trade_rupees / rps) if rps > 0 else 0
        # Sanity ceiling: notional ≤ 1Cr per trade. Anything more is a
        # detector-geometry bug (rps clamped to 1e-6 type), not a strategy choice.
        if qty * entry > 1e7:
            logger.warning(
                f"[ORCH] {symbol} {setup_type}: qty={qty} rps={rps} entry={entry} "
                f"exceeds 1Cr notional cap — likely detector geometry bug"
            )
            qty = 0
        notional = round(qty * entry, 2)

        mis_info = get_mis_info(symbol)
        try:
            entry_zone = list(compute_entry_zone(
                entry=entry,
                bias=bias,
                zone_pct=float(setup_cfg["entry_zone_pct"]),
                zone_mode=str(setup_cfg["entry_zone_mode"]),
            ))
        except ValueError as e:
            raise OrchestratorConfigError(
                f"setups.{setup_type}: entry_zone misconfigured: {e}"
            )

        # Directional-bias size multiplier (live-only; backtest tracker is no-op).
        dir_bias_mult = 1.0
        dir_bias_reason = "dir_bias:neutral"
        try:
            from services.gates.directional_bias import get_tracker
            db_tracker = get_tracker()
            if db_tracker is not None:
                dir_bias_mult, dir_bias_reason = db_tracker.get_size_mult(bias, category="sub7")
        except Exception:
            pass

        vwap_val = indicators.get("vwap")
        rsi_val = indicators.get("rsi")
        adx_val = indicators.get("adx")

        plan = {
            "symbol": symbol,
            "eligible": True,
            "strategy": setup_type,
            "bias": bias,
            "regime": regime,
            "category": "sub7",  # legacy tag — many tools filter by this

            # trade_id propagates StructureEvent → TradePlan → plan dict →
            # DECISION/TRIGGER/ENTRY/EXIT events.
            "trade_id": trade_plan.trade_id,

            "entry_ref_price": round(entry, 2),
            "entry_zone": entry_zone,
            "entry": {
                "reference": round(entry, 2),
                "zone": entry_zone,
                "trigger": "immediate",
                "mode": "immediate",
            },

            "stop": {
                "hard": round(hard_sl, 2),
                "risk_per_share": round(rps, 2),
                "target_risk": round(rps, 2),
            },

            "targets": targets,
            "trail": None,

            # `target_anchor_type` lets services/target_recalc.py decide how
            # to re-anchor at trigger. Detectors set this on the TradePlan;
            # default to "structural" (preserve target levels) for safety.
            "target_anchor_type": getattr(trade_plan, "target_anchor_type", "structural"),

            "quality": {
                "structural_rr": round(structural_rr, 2),
                "status": "good",
                "metrics": {
                    "entry": round(entry, 2),
                    "hard_sl": round(hard_sl, 2),
                    "rps": round(rps, 2),
                },
                "t1_feasible": structural_rr >= 1.0,
                "t2_feasible": len(targets) > 1,
                "rejection_reason": None,
            },

            "ranking": {"score": round(structural_rr, 3)},

            "sizing": {
                "qty": qty,
                "notional": notional,
                "risk_rupees": risk_per_trade_rupees,
                "risk_per_share": round(rps, 2),
                "size_mult": round(dir_bias_mult, 2),
                "base_mult": 1.0,
                "volatility_mult": 1.0,
                "cap_size_mult": 1.0,
                "dir_bias_mult": round(dir_bias_mult, 2),
                "dir_bias_reason": dir_bias_reason,
                "dir_bias_alignment": "neutral",
                "cap_segment": cap_segment,
                "cap_sl_mult": 1.0,
                "min_hold_bars": 0,
                "mis_enabled": mis_info.get("mis_enabled", False),
                "mis_leverage": mis_info.get("mis_leverage") or 1.0,
            },

            "indicators": {
                "atr": round(atr_for_plan, 2) if atr_for_plan else None,
                "adx": round(adx_val, 1) if adx_val else None,
                "rsi": round(rsi_val, 1) if rsi_val else None,
                "vwap": round(vwap_val, 2) if vwap_val else None,
            },

            "model_features": {
                "bb_width_proxy": 0.0,
                "volume5": float(df5m["volume"].iloc[-1]) if "volume" in df5m.columns else 0.0,
                "vol_z": vol_z_val,
                "vol_ratio": 0.0,
                "body_size_pct": 0.0,
                "wick_ratio": 0.0,
                "momentum_3bar_pct": 0.0,
                "momentum_1bar_pct": 0.0,
                "vwap_distance_pct": (
                    abs(current_price - vwap_val) / vwap_val * 100 if vwap_val else 0.0
                ),
            },

            "vc_reason": "sub7_fast_path",
            "levels": levels,
            "pipeline_reasons": ["sub7_fast_path"],
            "cautions": [],
        }

        logger.info(
            f"[ORCH] {symbol} {setup_type} APPROVED: entry={entry:.2f} sl={hard_sl:.2f} "
            f"rr={structural_rr:.2f} qty={qty}"
        )
        return plan

    # ---- Public entry points -------------------------------------------

    def process_single_candidate(
        self,
        symbol: str,
        setup_type: str,
        df5m: pd.DataFrame,
        levels: Dict[str, float],
        regime: str,
        now: pd.Timestamp,
        daily_df: Optional[pd.DataFrame] = None,
        htf_context: Optional[Dict[str, Any]] = None,
        regime_diagnostics: Optional[Dict[str, Any]] = None,
        daily_score: float = 0.0,
        cap_segment: Optional[str] = None,
        structure_event: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process one candidate. Returns plan dict (eligible=True/False) or
        None if the setup is unknown/disabled.

        Kept-but-unused: htf_context, regime_diagnostics, daily_score —
        these used to feed category-specific ranking; the new orchestrator
        ranks by structural_rr only. Signature preserved so callers don't
        need to change in Commit 1.
        """
        if setup_type not in ACTIVE_SETUPS:
            logger.warning(f"[ORCH] {symbol}: setup_type={setup_type} not in ACTIVE_SETUPS")
            return None

        plan = self.build_plan(
            symbol=symbol,
            setup_type=setup_type,
            df5m=df5m,
            levels=levels,
            regime=regime,
            now=now,
            cap_segment=cap_segment,
            daily_df=daily_df,
            structure_event=structure_event,
        )

        # Reject diagnostics
        timestamp = now.isoformat() if hasattr(now, "isoformat") else str(now)
        if plan is None or not plan.get("eligible", False):
            reason = (plan or {}).get("reason") or "no_plan"
            details = (plan or {}).get("details")
            logger.debug(f"[ORCH] {symbol} {setup_type} rejected: {reason}")
            plog = _planning_logger()
            if plog:
                plog.log_reject(
                    symbol,
                    reason=reason,
                    timestamp=timestamp,
                    strategy_type=setup_type,
                    category="sub7",
                    regime=regime,
                    gate_details=details,
                    bias=(plan or {}).get("bias"),
                )

        return plan

    def process_candidates(
        self,
        symbol: str,
        df5m: pd.DataFrame,
        levels: Dict[str, float],
        regime: str,
        now: pd.Timestamp,
        candidates: List[Any],
        daily_df: Optional[pd.DataFrame] = None,
        htf_context: Optional[Dict[str, Any]] = None,
        regime_diagnostics: Optional[Dict[str, Any]] = None,
        daily_score: float = 0.0,
        return_all_eligible: bool = False,
    ) -> Optional[Any]:
        """Process candidates for a single symbol.

        Default mode: pick the highest-`structural_rr` eligible plan.
        return_all_eligible=True: short-circuit and return EVERY eligible
            plan flat. Used by LiveGateChain (parity path) where the gate
            sees every plan and selects per its own logic.
        """
        if not candidates:
            return [] if return_all_eligible else None

        eligible: List[Tuple[Dict[str, Any], float]] = []
        for cand in candidates:
            setup_type = (
                str(cand.setup_type) if hasattr(cand, "setup_type") else str(cand)
            )
            cap_segment = getattr(cand, "cap_segment", None)
            detected_level = getattr(cand, "detected_level", None)
            extras = getattr(cand, "extras", None)
            structure_event = getattr(cand, "structure_event", None)

            cand_levels = dict(levels)
            if detected_level is not None:
                cand_levels["detected_level"] = detected_level

            plan = self.process_single_candidate(
                symbol=symbol,
                setup_type=setup_type,
                df5m=df5m,
                levels=cand_levels,
                regime=regime,
                now=now,
                daily_df=daily_df,
                structure_event=structure_event,
                cap_segment=cap_segment,
            )
            if plan and extras:
                plan["extras"] = extras
            if plan and plan.get("eligible", False):
                score = plan.get("ranking", {}).get("score", 0.0)
                eligible.append((plan, score))

        if not eligible:
            return [] if return_all_eligible else None

        if return_all_eligible:
            return [p for p, _s in eligible]

        # Single-best mode: highest structural_rr wins.
        eligible.sort(key=lambda x: x[1], reverse=True)
        best_plan, best_score = eligible[0]
        self._log_accept(best_plan, regime, now, competing=len(eligible))
        return best_plan

    def process_candidates_multi(
        self,
        symbols_data: List[Dict[str, Any]],
        regime: str,
        now: pd.Timestamp,
        max_positions: Optional[int] = None,
        regime_diagnostics: Optional[Dict[str, Any]] = None,
        daily_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Multi-symbol selection: rank globally by structural_rr, cap at
        max_positions_total, optionally skip duplicate symbols."""
        sel = _get_selection_rules()
        if max_positions is None:
            max_positions = sel["max_positions_total"]
        skip_duplicates = sel["skip_duplicate_symbols"]
        min_score = sel.get("min_score_to_select", 0.0)

        all_plans: List[Tuple[Dict[str, Any], float, str]] = []
        for data in symbols_data:
            symbol = data["symbol"]
            df5m = data["df5m"]
            levels = data["levels"]
            candidates = data.get("candidates") or []
            daily_df = data.get("daily_df")

            for cand in candidates:
                setup_type = (
                    str(cand.setup_type) if hasattr(cand, "setup_type") else str(cand)
                )
                detected_level = getattr(cand, "detected_level", None)
                extras = getattr(cand, "extras", None)
                structure_event = getattr(cand, "structure_event", None)

                cand_levels = dict(levels)
                if detected_level is not None:
                    cand_levels["detected_level"] = detected_level

                plan = self.process_single_candidate(
                    symbol=symbol,
                    setup_type=setup_type,
                    df5m=df5m,
                    levels=cand_levels,
                    regime=regime,
                    now=now,
                    daily_df=daily_df,
                    structure_event=structure_event,
                )
                if plan and extras:
                    plan["extras"] = extras
                if plan and plan.get("eligible", False):
                    score = plan.get("ranking", {}).get("score", 0.0)
                    if score >= min_score:
                        all_plans.append((plan, score, symbol))

        if not all_plans:
            logger.info("[ORCH] No eligible plans across symbols")
            return []

        # Global ranking — single bucket.
        all_plans.sort(key=lambda x: x[1], reverse=True)

        selected: List[Dict[str, Any]] = []
        seen_symbols: set = set()
        for plan, _score, symbol in all_plans:
            if len(selected) >= max_positions:
                break
            if skip_duplicates and symbol in seen_symbols:
                logger.debug(f"[ORCH] Skipping {symbol} — already selected")
                continue
            selected.append(plan)
            seen_symbols.add(symbol)

        logger.info(
            f"[ORCH] Selected {len(selected)}/{max_positions} plans for "
            f"regime={regime} from {len(all_plans)} eligible"
        )

        # Log accepts for analytics.jsonl
        for plan in selected:
            self._log_accept(plan, regime, now, competing=len(all_plans))

        return selected

    # ---- planning_log accept logger ------------------------------------

    def _log_accept(
        self,
        plan: Dict[str, Any],
        regime: str,
        now: pd.Timestamp,
        competing: int,
    ) -> None:
        """Write an ACCEPT line to analytics.jsonl. Mirrors the legacy
        orchestrator's payload so existing report tools (deep_edge_analysis,
        edge_optimizer, filter_simulation) keep working unchanged."""
        plog = _planning_logger()
        if not plog:
            return
        timestamp = now.isoformat() if hasattr(now, "isoformat") else str(now)
        targets = plan.get("targets") or []
        sizing = plan.get("sizing") or {}
        try:
            plog.log_accept(
                plan.get("symbol", "?"),
                timestamp=timestamp,
                strategy_type=plan.get("strategy"),
                category=plan.get("category"),
                bias=plan.get("bias"),
                entry_ref_price=plan.get("entry_ref_price"),
                structural_rr=(plan.get("quality") or {}).get("structural_rr"),
                t1_rr=targets[0]["rr"] if targets else None,
                rank_score=(plan.get("ranking") or {}).get("score"),
                quality_status=(plan.get("quality") or {}).get("status"),
                size_mult=sizing.get("size_mult"),
                regime=regime,
                selected=True,
                competing_plans=competing,
                stop_hard=(plan.get("stop") or {}).get("hard"),
                t2_rr=targets[1]["rr"] if len(targets) > 1 else None,
                entry_zone=plan.get("entry_zone"),
                qty=sizing.get("qty"),
                notional=sizing.get("notional"),
                mis_leverage=sizing.get("mis_leverage"),
                risk_per_share=sizing.get("risk_per_share"),
                cap_segment=sizing.get("cap_segment"),
                volatility_mult=sizing.get("volatility_mult"),
                cap_size_mult=sizing.get("cap_size_mult"),
                dir_bias_mult=sizing.get("dir_bias_mult"),
                indicators=plan.get("indicators"),
                vc_reason=plan.get("vc_reason"),
            )
        except Exception as e:
            logger.warning(f"[ORCH] log_accept failed for {plan.get('symbol')}: {e}")


# ---------------------------------------------------------------------------
# Singleton + convenience wrappers (preserve legacy import shape)
# ---------------------------------------------------------------------------


_orchestrator_instance: Optional[PlanOrchestrator] = None


def get_orchestrator() -> PlanOrchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = PlanOrchestrator()
    return _orchestrator_instance


def process_setup_candidates(
    symbol: str,
    df5m: pd.DataFrame,
    levels: Dict[str, float],
    regime: str,
    now: pd.Timestamp,
    candidates: List[Any],
    daily_df: Optional[pd.DataFrame] = None,
    htf_context: Optional[Dict[str, Any]] = None,
    regime_diagnostics: Optional[Dict[str, Any]] = None,
    daily_score: float = 0.0,
    return_all_eligible: bool = False,
) -> Optional[Any]:
    """Single-symbol convenience wrapper — same signature as the legacy fn."""
    return get_orchestrator().process_candidates(
        symbol=symbol,
        df5m=df5m,
        levels=levels,
        regime=regime,
        now=now,
        candidates=candidates,
        daily_df=daily_df,
        htf_context=htf_context,
        regime_diagnostics=regime_diagnostics,
        daily_score=daily_score,
        return_all_eligible=return_all_eligible,
    )


def process_multi_symbol_candidates(
    symbols_data: List[Dict[str, Any]],
    regime: str,
    now: pd.Timestamp,
    max_positions: Optional[int] = None,
    regime_diagnostics: Optional[Dict[str, Any]] = None,
    daily_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Multi-symbol convenience wrapper — same signature as the legacy fn."""
    return get_orchestrator().process_candidates_multi(
        symbols_data=symbols_data,
        regime=regime,
        now=now,
        max_positions=max_positions,
        regime_diagnostics=regime_diagnostics,
        daily_score=daily_score,
    )
