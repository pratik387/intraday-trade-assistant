# pipelines/orchestrator.py
"""
Pipeline Orchestrator - Professional Multi-Strategy Allocation System.

This orchestrator implements the professional approach used by multi-strategy
hedge funds (AQR, Bridgewater, Graham Capital):

1. RANK WITHIN CATEGORY - Each category (BREAKOUT, LEVEL, REVERSION, MOMENTUM)
   ranks its own candidates using category-specific factors
2. REGIME-BASED RISK BUDGET - Allocate capital/slots based on current regime
3. PROPORTIONAL SELECTION - Select top candidates from each category based on budget

Key insight: Comparing a breakout score of 2.5 with a reversion score of 2.5 is
meaningless - they measure DIFFERENT things. Professional funds don't compare
apples to oranges.

Reference: https://funds.aqr.com/Insights/Strategies/Multi-Strategy

Usage:
    orchestrator = PipelineOrchestrator()
    plans = orchestrator.process_candidates_multi(symbols_data, regime, now, max_positions=5)
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
import os
import pandas as pd

from config.logging_config import get_agent_logger, get_planning_logger
from config.setup_categories import get_category, get_base_setup_name, SetupCategory

# Import category pipelines
from pipelines.breakout_pipeline import BreakoutPipeline
from pipelines.level_pipeline import LevelPipeline
from pipelines.reversion_pipeline import ReversionPipeline
from pipelines.momentum_pipeline import MomentumPipeline
from pipelines.base_pipeline import BasePipeline, ConfigurationError, get_mis_info, get_cap_segment

# Sub-project #7: detector classes for fast-path routing (bypass SMC category pipelines)
from structures.gap_fade_short_structure import GapFadeShortStructure
from structures.mis_unwind_short_structure import MISUnwindShortStructure
from structures.cpr_mean_revert_structure import CPRMeanRevertStructure
from structures.orb_15_structure import ORB15Structure
from structures.narrow_cpr_breakout_structure import NarrowCPRBreakoutStructure
from structures.vwap_first_pullback_structure import VWAPFirstPullbackStructure
from structures.pdh_pdl_reject_structure import PDHPDLRejectStructure
from structures.data_models import MarketContext

# Setup types that use Sub7 fast path — detector emits complete TradePlan,
# so SMC category pipeline must NOT override entry/stop/target.
SUB7_SETUPS: frozenset = frozenset({
    "gap_fade_short",
    "mis_unwind_short",
    "cpr_mean_revert",
    "orb_15",
    "narrow_cpr_breakout",
    "vwap_first_pullback",
    "pdh_pdl_reject",
})

logger = get_agent_logger()


def _get_planning_logger():
    """Get planning logger lazily to ensure it's initialized after logging setup."""
    return get_planning_logger()


class RiskBudgetConfigError(Exception):
    """Raised when risk budget configuration is missing or invalid."""
    pass


def _load_risk_budget_config() -> Dict[str, Any]:
    """
    Load risk budget configuration from JSON file.

    All values must come from config - no defaults in code.

    Raises:
        RiskBudgetConfigError: If config file is missing or invalid
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "pipelines", "risk_budget_config.json"
    )

    if not os.path.exists(config_path):
        raise RiskBudgetConfigError(f"Risk budget config not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise RiskBudgetConfigError(f"Invalid JSON in risk budget config: {e}")

    # Validate required top-level keys
    required_keys = ["categories", "regime_risk_budgets", "selection_rules", "category_constraints", "logging"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise RiskBudgetConfigError(f"Missing required config keys: {missing}")

    # Validate selection_rules has all required fields
    selection_required = [
        "max_positions_per_symbol", "max_positions_total", "min_budget_threshold",
        "slot_allocation_method", "tie_breaker", "budget_boost_weight", "skip_duplicate_symbols"
    ]
    missing_selection = [k for k in selection_required if k not in config["selection_rules"]]
    if missing_selection:
        raise RiskBudgetConfigError(f"Missing selection_rules fields: {missing_selection}")

    # Validate logging section
    logging_required = ["log_slot_allocation", "log_category_ranking", "log_selection_details", "log_budget_summary"]
    missing_logging = [k for k in logging_required if k not in config["logging"]]
    if missing_logging:
        raise RiskBudgetConfigError(f"Missing logging fields: {missing_logging}")

    # Validate each category in regime_risk_budgets
    categories = config["categories"]
    for regime, budgets in config["regime_risk_budgets"].items():
        if regime.startswith("_"):
            continue
        for cat in categories:
            if cat not in budgets:
                raise RiskBudgetConfigError(f"Missing budget for {cat} in regime {regime}")

    # Validate category_constraints
    for cat in categories:
        if cat not in config["category_constraints"]:
            raise RiskBudgetConfigError(f"Missing category_constraints for {cat}")
        constraints = config["category_constraints"][cat]
        if "max_concurrent" not in constraints or "blocked_regimes" not in constraints or "enabled" not in constraints:
            raise RiskBudgetConfigError(f"Missing required fields in category_constraints for {cat}")

    return config


class PipelineOrchestrator:
    """
    Professional Multi-Strategy Orchestrator.

    Implements AQR/Bridgewater approach:
    - Rank within categories (not across)
    - Allocate by regime-based risk budget
    - Select proportionally from each category
    """

    def __init__(self):
        """
        Initialize the orchestrator with lazy pipeline loading.

        Raises:
            RiskBudgetConfigError: If config is missing or invalid
        """
        self._pipelines: Dict[SetupCategory, BasePipeline] = {}
        self._init_errors: Dict[SetupCategory, str] = {}
        self._config = _load_risk_budget_config()

        # Sub7 detector instances — instantiated lazily on first use.
        # Keyed by setup_type so _build_plan_from_sub7_detector can look them up.
        self._sub7_detectors: Dict[str, Any] = {}

        logger.debug("PipelineOrchestrator initialized with risk budget allocation")

    def _get_config(self, *keys: str) -> Any:
        """
        Get a config value by nested keys.

        Args:
            *keys: Path to config value (e.g., "selection_rules", "max_positions_total")

        Returns:
            Config value

        Raises:
            RiskBudgetConfigError: If key path not found
        """
        value = self._config
        path = []
        for key in keys:
            path.append(key)
            if not isinstance(value, dict) or key not in value:
                raise RiskBudgetConfigError(f"Config key not found: {'.'.join(path)}")
            value = value[key]
        return value

    def _should_log(self, log_type: str) -> bool:
        """Check if a log type is enabled in config."""
        return self._get_config("logging", log_type)

    def _get_pipeline(self, category: SetupCategory) -> Optional[BasePipeline]:
        """Get or lazily initialize a category pipeline."""
        if category in self._pipelines:
            return self._pipelines[category]

        if category in self._init_errors:
            logger.debug(f"[ORCHESTRATOR] Skipping {category.value} - previous init error")
            return None

        # Check if category is enabled
        cat_constraints = self._get_config("category_constraints", category.value)
        if not cat_constraints["enabled"]:
            logger.debug(f"[ORCHESTRATOR] Category {category.value} is disabled in config")
            return None

        try:
            if category == SetupCategory.BREAKOUT:
                self._pipelines[category] = BreakoutPipeline()
            elif category == SetupCategory.LEVEL:
                self._pipelines[category] = LevelPipeline()
            elif category == SetupCategory.REVERSION:
                self._pipelines[category] = ReversionPipeline()
            elif category == SetupCategory.MOMENTUM:
                self._pipelines[category] = MomentumPipeline()
            else:
                logger.warning(f"[ORCHESTRATOR] Unknown category: {category}")
                return None

            logger.info(f"[ORCHESTRATOR] Initialized {category.value} pipeline")
            return self._pipelines[category]

        except ConfigurationError as e:
            self._init_errors[category] = str(e)
            logger.error(f"[ORCHESTRATOR] Failed to initialize {category.value} pipeline: {e}")
            return None
        except Exception as e:
            self._init_errors[category] = str(e)
            logger.exception(f"[ORCHESTRATOR] Unexpected error initializing {category.value}: {e}")
            return None

    # ------------------------------------------------------------------
    # Sub-project #7 fast path
    # ------------------------------------------------------------------

    def _get_sub7_detector(self, setup_type: str) -> Optional[Any]:
        """Return the sub7 detector for *setup_type*, instantiating lazily.

        Config is loaded from the root configuration.json ``setups`` block (same
        source MainDetector uses, so parameter parity is guaranteed).
        """
        if setup_type in self._sub7_detectors:
            return self._sub7_detectors[setup_type]

        # Load full config once to get the setups sub-section
        try:
            import json as _json
            _cfg_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config", "configuration.json"
            )
            with open(_cfg_path) as _f:
                _full_cfg = _json.load(_f)
            setups_cfg = _full_cfg.get("setups", {})
            setup_cfg = setups_cfg.get(setup_type, {})
            if not setup_cfg.get("enabled", False):
                logger.warning(f"[SUB7] {setup_type} not enabled in configuration.json — skipping")
                return None

            # Inject _setup_name so detector can identify itself (same as MainDetector)
            setup_cfg = {**setup_cfg, "_setup_name": setup_type}

            _cls_map = {
                "gap_fade_short": GapFadeShortStructure,
                "mis_unwind_short": MISUnwindShortStructure,
                "cpr_mean_revert": CPRMeanRevertStructure,
                "orb_15": ORB15Structure,
                "narrow_cpr_breakout": NarrowCPRBreakoutStructure,
                "vwap_first_pullback": VWAPFirstPullbackStructure,
                "pdh_pdl_reject": PDHPDLRejectStructure,
            }
            cls = _cls_map.get(setup_type)
            if cls is None:
                logger.warning(f"[SUB7] No detector class mapped for {setup_type}")
                return None

            det = cls(setup_cfg)
            self._sub7_detectors[setup_type] = det
            logger.debug(f"[SUB7] Instantiated {setup_type} detector")
            return det

        except Exception as exc:
            logger.exception(f"[SUB7] Failed to instantiate {setup_type} detector: {exc}")
            return None

    def _build_plan_from_sub7_detector(
        self,
        symbol: str,
        setup_type: str,
        df5m: pd.DataFrame,
        levels: Dict[str, float],
        regime: str,
        now: pd.Timestamp,
        cap_segment: Optional[str] = None,
        daily_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a plan dict directly from a sub7 detector, bypassing category pipelines.

        Returns an orchestrator-compatible plan dict (same shape as base_pipeline.run_pipeline),
        or None if the detector rejects the setup.
        """
        detector = self._get_sub7_detector(setup_type)
        if detector is None:
            return None

        # Build MarketContext — mirrors main_detector._build_market_context()
        try:
            if df5m is None or len(df5m) == 0:
                return None

            atr_val = None
            if "atr" in df5m.columns and not pd.isna(df5m["atr"].iloc[-1]):
                atr_val = float(df5m["atr"].iloc[-1])
            else:
                atr_val = float((df5m["high"] - df5m["low"]).tail(14).mean())

            vol_z_val = 0.0
            if "vol_z" in df5m.columns and not pd.isna(df5m["vol_z"].iloc[-1]):
                vol_z_val = float(df5m["vol_z"].iloc[-1])

            indicators = {"atr": atr_val, "vol_z": vol_z_val}

            # Populate any vwap/rsi columns into indicators for detector use
            for col in ("vwap", "rsi", "adx"):
                if col in df5m.columns and not pd.isna(df5m[col].iloc[-1]):
                    indicators[col] = float(df5m[col].iloc[-1])

            bar_timestamp = pd.to_datetime(df5m.index[-1])
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
            logger.exception(f"[SUB7] {symbol} {setup_type}: failed to build MarketContext: {exc}")
            return None

        # Determine direction. Setups with explicit suffix (_long/_short) are unambiguous.
        # Bidirectional detectors (e.g. cpr_mean_revert) decide bias inside detect() based
        # on price geometry; calling the wrong plan_*_strategy() returns None and silently
        # drops the signal. So for those, run detect() to read the bias, then dispatch.
        if setup_type.endswith("_long"):
            bias = "long"
        elif setup_type.endswith("_short"):
            bias = "short"
        else:
            try:
                analysis = detector.detect(context)
            except Exception as exc:
                logger.exception(f"[SUB7] {symbol} {setup_type}: detect() raised: {exc}")
                return None
            evts = getattr(analysis, "events", []) or []
            bias_from_detect = evts[0].context.get("bias") if evts else None
            if bias_from_detect not in ("long", "short"):
                return None
            bias = bias_from_detect

        # Call appropriate plan method
        try:
            if bias == "short":
                trade_plan = detector.plan_short_strategy(context)
            else:
                trade_plan = detector.plan_long_strategy(context)
        except Exception as exc:
            logger.exception(f"[SUB7] {symbol} {setup_type}: plan_{bias}_strategy raised: {exc}")
            return None

        if trade_plan is None:
            logger.debug(f"[SUB7] {symbol} {setup_type}: detector returned None — conditions not met")
            return {"eligible": False, "reason": "sub7_detector_rejected", "strategy": setup_type, "bias": bias}

        # --- Convert TradePlan → plan dict (orchestrator-compatible shape) ---
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        rps = trade_plan.risk_params.risk_per_share
        atr_for_plan = trade_plan.risk_params.atr or atr_val

        # targets: list[dict] with at least "level" and "rr" keys
        raw_targets = trade_plan.exit_levels.targets if trade_plan.exit_levels else []
        targets = []
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

        # Structural RR from first target
        structural_rr = targets[0]["rr"] if targets else 0.0

        # Position sizing (Van Tharp CPR — same formula as base_pipeline)
        try:
            import json as _json2
            _cfg_path2 = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config", "configuration.json"
            )
            with open(_cfg_path2) as _f2:
                _root_cfg = _json2.load(_f2)
            risk_per_trade_rupees = float(_root_cfg["risk_per_trade_rupees"])
        except Exception:
            raise RiskBudgetConfigError(
                "risk_per_trade_rupees missing from config/configuration.json — cannot size sub7 trade"
            )

        qty = int(risk_per_trade_rupees / rps) if rps > 0 else 0
        notional = round(qty * entry, 2)

        # MIS info for leverage tracking
        mis_info = get_mis_info(symbol)

        # entry_zone: for sub7 immediate entries use a ±0.1% zone around entry
        entry_zone_half = entry * 0.001
        if bias == "short":
            entry_zone = [round(entry - entry_zone_half, 2), round(entry + entry_zone_half, 2)]
        else:
            entry_zone = [round(entry - entry_zone_half, 2), round(entry + entry_zone_half, 2)]

        # Directional bias multiplier (same as base_pipeline)
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
            "category": "sub7",

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

            "ranking": {
                "score": round(structural_rr, 3),
            },

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
            f"[SUB7] {symbol} {setup_type} APPROVED: entry={entry:.2f} sl={hard_sl:.2f} "
            f"rr={structural_rr:.2f} qty={qty}"
        )
        return plan

    def get_category_for_setup(self, setup_type: str) -> Optional[SetupCategory]:
        """Determine the category for a setup type."""
        return get_category(setup_type)

    def _get_risk_budget(self, regime: str) -> Dict[str, float]:
        """
        Get risk budget allocation for current regime.

        Args:
            regime: Current market regime

        Returns:
            Dict mapping category name to risk budget (0.0-1.0)
        """
        budgets = self._get_config("regime_risk_budgets")

        # Try exact regime match, then fallback to "neutral"
        if regime in budgets and not regime.startswith("_"):
            return {k: v for k, v in budgets[regime].items() if not k.startswith("_")}

        # Fallback to neutral
        if "neutral" in budgets:
            return {k: v for k, v in budgets["neutral"].items() if not k.startswith("_")}

        raise RiskBudgetConfigError(f"No budget defined for regime '{regime}' and no 'neutral' fallback")

    def _is_category_blocked(self, category: str, regime: str, setup_type: str = None, now: pd.Timestamp = None) -> bool:
        """
        Check if category is blocked for this regime.

        Special case: ORB Chop Exception
        From regime_gate.py - ORB setups are allowed in chop regime before 10:30 AM
        because opening range breakouts can still work in early session even
        if the broader market is choppy.
        """
        constraints = self._get_config("category_constraints", category)
        blocked_regimes = constraints["blocked_regimes"]

        if regime not in blocked_regimes:
            return False

        # Check ORB Chop Exception: ORB allowed in chop before 10:30 AM
        if regime == "chop" and setup_type and now:
            orb_chop_cfg = self._config.get("orb_chop_exception", {})
            if orb_chop_cfg.get("enabled", False):
                orb_setups = orb_chop_cfg.get("orb_setups", [
                    "orb_breakout_long", "orb_breakout_short",
                    "orb_pullback_long", "orb_pullback_short"
                ])
                end_hour = orb_chop_cfg.get("end_hour", 10)
                end_minute = orb_chop_cfg.get("end_minute", 30)

                if setup_type in orb_setups:
                    current_minutes = now.hour * 60 + now.minute
                    cutoff_minutes = end_hour * 60 + end_minute

                    if current_minutes <= cutoff_minutes:
                        logger.debug(
                            f"ORB_CHOP_EXCEPTION: {setup_type} allowed in chop before {end_hour}:{end_minute:02d}"
                        )
                        return False  # Allow ORB in chop before cutoff

        return True

    def _is_hard_blocked(self, setup_type: str, regime: str) -> bool:
        """
        Check if a specific setup type is hard blocked in this regime.

        Hard blocks are permanent blocks that CANNOT be bypassed by HCET or any other
        mechanism. They are based on evidence showing certain setups consistently
        lose money in specific regimes.

        Single source of truth: regime_gate.py HARD_BLOCKS constant.

        Args:
            setup_type: The setup type (e.g., "first_hour_momentum_long")
            regime: The current market regime

        Returns:
            True if this setup is permanently blocked in this regime
        """
        from pipelines.base_pipeline import HARD_BLOCKS, is_hard_blocked

        # Use centralized is_hard_blocked from base_pipeline
        return is_hard_blocked(setup_type, regime)

    def _is_orb_priority_window(self, now: pd.Timestamp) -> bool:
        """
        Check if we're in the ORB priority window.

        From trade_decision_gate.py analysis:
        - During 9:30-10:30 AM, ORB setups should be prioritized
        - Opening Range Breakout (ORB) has highest win rate in first hour
        """
        try:
            orb_cfg = self._config.get("orb_priority_window", {})
            if not orb_cfg.get("enabled", False):
                return False

            start_hour = orb_cfg.get("start_hour", 9)
            start_minute = orb_cfg.get("start_minute", 30)
            end_hour = orb_cfg.get("end_hour", 10)
            end_minute = orb_cfg.get("end_minute", 30)

            current_minutes = now.hour * 60 + now.minute
            start_minutes = start_hour * 60 + start_minute
            end_minutes = end_hour * 60 + end_minute

            return start_minutes <= current_minutes <= end_minutes
        except Exception:
            return False

    def _apply_orb_priority(
        self,
        category_plans: dict,
        now: pd.Timestamp
    ) -> dict:
        """
        Apply ORB priority window boost to ORB setup scores.

        During 9:30-10:30 AM:
        - ORB setups get a significant score boost
        - This prioritizes ORB over other setups regardless of raw score

        Args:
            category_plans: Dict of {category: [(plan, score, symbol), ...]}
            now: Current timestamp

        Returns:
            Modified category_plans with ORB scores boosted
        """
        if not self._is_orb_priority_window(now):
            return category_plans

        orb_cfg = self._config.get("orb_priority_window", {})
        score_boost = orb_cfg.get("score_boost", 5.0)  # Large boost to ensure ORB wins
        orb_setups = orb_cfg.get("orb_setups", ["orb_breakout_long", "orb_breakout_short", "orb_pullback_long", "orb_pullback_short"])

        # Apply boost to ORB setups
        for category in category_plans:
            boosted_plans = []
            for plan, score, symbol in category_plans[category]:
                strategy = plan.get("strategy", "")
                if strategy in orb_setups:
                    new_score = score + score_boost
                    plan["ranking"]["orb_priority_boost"] = score_boost
                    logger.debug(f"ORB_PRIORITY: {symbol} {strategy} score boosted {score:.2f} → {new_score:.2f}")
                    boosted_plans.append((plan, new_score, symbol))
                else:
                    boosted_plans.append((plan, score, symbol))

            # Re-sort after boosting
            boosted_plans.sort(key=lambda x: x[1], reverse=True)
            category_plans[category] = boosted_plans

        return category_plans

    def _allocate_slots(self, regime: str, max_positions: int) -> Dict[str, int]:
        """
        Allocate position slots to categories based on risk budget.

        Professional approach: Regime determines how many slots each category gets.

        Args:
            regime: Current market regime
            max_positions: Maximum total positions allowed

        Returns:
            Dict mapping category name to number of slots
        """
        budget = self._get_risk_budget(regime)
        min_threshold = self._get_config("selection_rules", "min_budget_threshold")
        categories = self._get_config("categories")

        # Filter categories below minimum threshold and blocked categories
        eligible_budget = {}
        for cat in categories:
            if budget.get(cat, 0) >= min_threshold and not self._is_category_blocked(cat, regime):
                eligible_budget[cat] = budget[cat]

        # Normalize remaining budgets
        total = sum(eligible_budget.values())
        if total > 0:
            normalized = {k: v / total for k, v in eligible_budget.items()}
        else:
            normalized = eligible_budget

        # Allocate slots proportionally
        slots = {}
        remaining = max_positions

        # Sort by budget descending for fair allocation
        sorted_cats = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

        for cat, pct in sorted_cats:
            if remaining <= 0:
                slots[cat] = 0
            else:
                # At least 1 slot if budget >= threshold, proportional otherwise
                cat_slots = max(1, round(pct * max_positions))
                # Respect max_concurrent from constraints
                max_concurrent = self._get_config("category_constraints", cat, "max_concurrent")
                cat_slots = min(cat_slots, max_concurrent, remaining)
                slots[cat] = cat_slots
                remaining -= cat_slots

        # Categories below threshold or blocked get 0
        for cat in categories:
            if cat not in slots:
                slots[cat] = 0

        if self._should_log("log_slot_allocation"):
            logger.info(
                f"[ORCHESTRATOR] Slot allocation for regime={regime}: "
                + ", ".join(f"{cat}={slots.get(cat, 0)}" for cat in categories)
            )

        return slots

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
        cap_segment: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single setup candidate through its category pipeline.

        Args:
            symbol: Trading symbol
            setup_type: Setup type (e.g., "orb_breakout_long")
            df5m: 5-minute OHLCV DataFrame
            levels: Key price levels
            regime: Current market regime
            now: Current timestamp
            daily_df: Daily OHLCV DataFrame (optional)
            htf_context: HTF (15m) context for category-specific ranking
            regime_diagnostics: Multi-TF regime info for universal ranking adjustments
                               {daily: {regime, confidence}, hourly: {session_bias, confidence}}
            daily_score: Daily timeframe score for score weighting

        Returns plan dict with category info, or None if rejected.
        """
        # --- Sub-project #7 fast path ---
        # Sub7 detectors emit complete TradePlans (entry/stop/target).
        # Routing through SMC category pipelines (especially ReversionPipeline)
        # overwrites those levels via pattern-matching, causing 93% expiry.
        # Fast path bypasses category pipelines entirely for sub7 setups.
        if setup_type in SUB7_SETUPS:
            logger.debug(f"[SUB7] {symbol} {setup_type}: fast-path (bypass SMC pipeline)")
            plan = self._build_plan_from_sub7_detector(
                symbol=symbol,
                setup_type=setup_type,
                df5m=df5m,
                levels=levels,
                regime=regime,
                now=now,
                cap_segment=cap_segment,
                daily_df=daily_df,
            )
            if plan and plan.get("eligible", False):
                plan["category"] = "sub7"
                plan["strategy"] = setup_type
            return plan
        # --- End sub7 fast path ---

        category = self.get_category_for_setup(setup_type)
        if category is None:
            logger.warning(f"[ORCHESTRATOR] No category mapping for: {setup_type}")
            return None

        # Check if category is blocked for this regime
        # Pass setup_type and now to enable ORB Chop Exception (ORB allowed in chop before 10:30)
        if self._is_category_blocked(category.value, regime, setup_type=setup_type, now=now):
            logger.debug(f"[ORCHESTRATOR] {category.value} blocked for regime {regime}")
            return None

        # Check HARD BLOCKS - setup types permanently blocked in specific regimes
        # This is the SINGLE source of truth from regime_gate.py - cannot be bypassed
        if self._is_hard_blocked(setup_type, regime):
            logger.debug(f"[ORCHESTRATOR] HARD_BLOCK: {setup_type} permanently blocked in {regime}")
            return None

        pipeline = self._get_pipeline(category)
        if pipeline is None:
            logger.debug(f"[ORCHESTRATOR] No pipeline for category: {category.value}")
            return None

        try:
            plan = pipeline.run_pipeline(
                symbol=symbol,
                setup_type=setup_type,
                df5m=df5m,
                levels=levels,
                regime=regime,
                now=now,
                daily_df=daily_df,
                htf_context=htf_context,
                regime_diagnostics=regime_diagnostics,
                daily_score=daily_score,
                cap_segment=cap_segment
            )

            timestamp = now.isoformat() if hasattr(now, 'isoformat') else str(now)

            if plan and plan.get("eligible", False):
                # Add category to plan for tracking
                plan["category"] = category.value
                # Store strategy type for later logging
                plan["strategy"] = setup_type

                logger.debug(f"[ORCHESTRATOR] {symbol} {setup_type} eligible via {category.value}")
                # Note: Logging of accepts happens in process_candidates/process_candidates_multi
                # to avoid duplicate entries when multiple candidates are evaluated
            else:
                reason = plan.get("reason", "unknown") if plan else "no_plan"
                rejection_reason = plan.get("quality", {}).get("rejection_reason") if plan else None
                # Get detailed gate_fail reasons if available
                gate_details = plan.get("details") if plan else None
                logger.debug(f"[ORCHESTRATOR] {symbol} {setup_type} rejected: {reason}")

                planning_log = _get_planning_logger()
                if planning_log:
                    planning_log.log_reject(
                        symbol,
                        reason=rejection_reason or reason,
                        timestamp=timestamp,
                        strategy_type=setup_type,
                        category=category.value,
                        structural_rr=plan.get("quality", {}).get("structural_rr") if plan else None,
                        regime=regime,
                        gate_details=gate_details,
                        # Indicators at rejection (when plan exists)
                        indicators=plan.get("indicators") if plan else None,
                        quality_status=plan.get("quality", {}).get("status") if plan else None,
                        rank_score=plan.get("ranking", {}).get("score") if plan else None,
                        rank_components=plan.get("ranking", {}).get("components") if plan else None,
                        bias=plan.get("bias") if plan else None,
                    )

            return plan

        except Exception as e:
            logger.exception(f"[ORCHESTRATOR] Error processing {symbol} {setup_type}: {e}")
            return None

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
    ) -> Optional[Dict[str, Any]]:
        """
        Process candidates for a SINGLE symbol - returns best plan (default)
        or List[plan] of all eligible plans across categories (return_all_eligible=True).

        For single-symbol processing, we still use category-specific ranking
        but select the best from the highest-budget category that has candidates.

        When return_all_eligible=True, the per-symbol-category dedupe is bypassed
        — caller (e.g., LiveGateChain) sees every eligible plan and applies its
        own selection logic. This matches gauntlet behavior where every executed
        trade is a separate row, not deduped per symbol-category.

        Args:
            symbol: Trading symbol
            df5m: 5-minute OHLCV DataFrame
            levels: Key price levels
            regime: Market regime
            now: Current timestamp
            candidates: List of SetupCandidate objects
            daily_df: Daily OHLCV DataFrame (optional)
            htf_context: HTF (15m) context for category-specific ranking
            regime_diagnostics: Multi-TF regime info for universal ranking adjustments
                               {daily: {regime, confidence}, hourly: {session_bias, confidence}}
            daily_score: Daily timeframe score for score weighting

        Returns:
            Best eligible plan or None
        """
        if not candidates:
            logger.debug(f"[ORCHESTRATOR] {symbol}: No candidates to process")
            return None

        # Group eligible plans by category
        category_plans: Dict[str, List[Tuple[Dict[str, Any], float]]] = defaultdict(list)

        for candidate in candidates:
            setup_type = str(candidate.setup_type) if hasattr(candidate, 'setup_type') else str(candidate)
            strength = getattr(candidate, 'strength', 0.5)
            cap_segment = getattr(candidate, 'cap_segment', None)
            detected_level = getattr(candidate, 'detected_level', None)
            extras = getattr(candidate, 'extras', None)  # Detector context for trade_report.csv

            # Merge detected_level into levels dict for quality calculation
            # Pro trader approach: use the actual detected level, not hardcoded PDH/PDL
            candidate_levels = dict(levels)  # Copy to avoid mutating original
            if detected_level is not None:
                candidate_levels["detected_level"] = detected_level

            plan = self.process_single_candidate(
                symbol=symbol,
                setup_type=setup_type,
                df5m=df5m,
                levels=candidate_levels,
                regime=regime,
                now=now,
                daily_df=daily_df,
                htf_context=htf_context,
                regime_diagnostics=regime_diagnostics,
                daily_score=daily_score,
                cap_segment=cap_segment
            )

            # Propagate detector extras into plan so they flow into events.jsonl DECISION
            # event and from there into trade_report.csv via diagnostics_report_builder.
            # Existing edge analysis tools (deep_edge_analysis, edge_optimizer, filter_simulation)
            # auto-discover any new columns in trade_report.csv.
            if plan and extras:
                plan["extras"] = extras

            if plan and plan.get("eligible", False):
                category = plan.get("category", "UNKNOWN")
                rank_score = plan.get("ranking", {}).get("score", strength)
                category_plans[category].append((plan, rank_score))

        if not category_plans:
            logger.debug(f"[ORCHESTRATOR] {symbol}: No eligible plans from {len(candidates)} candidates")
            return [] if return_all_eligible else None

        # Sort each category by rank score (within-category ranking)
        for cat in category_plans:
            category_plans[cat].sort(key=lambda x: x[1], reverse=True)

        # SHORT-CIRCUIT: when caller wants ALL eligible plans (LiveGateChain
        # parity path), skip the per-symbol-category dedupe and return every
        # eligible plan flat. The downstream gate sees the full candidate
        # population and decides — matches gauntlet's no-dedupe behavior.
        if return_all_eligible:
            all_plans = []
            for cat_plans in category_plans.values():
                for plan, _score in cat_plans:
                    all_plans.append(plan)
            return all_plans

        # Apply ORB priority window boost (if applicable)
        # Convert to (plan, score, symbol) format for _apply_orb_priority
        category_plans_with_symbol = {
            cat: [(plan, score, symbol) for plan, score in plans]
            for cat, plans in category_plans.items()
        }
        category_plans_with_symbol = self._apply_orb_priority(category_plans_with_symbol, now)
        # Convert back to (plan, score) format
        category_plans = {
            cat: [(plan, score) for plan, score, _ in plans]
            for cat, plans in category_plans_with_symbol.items()
        }

        if self._should_log("log_category_ranking"):
            for cat, plans in category_plans.items():
                logger.debug(f"[ORCHESTRATOR] {symbol} {cat}: {len(plans)} plans, top={plans[0][1]:.3f}")

        # Get risk budget for regime
        budget = self._get_risk_budget(regime)
        budget_boost = self._get_config("selection_rules", "budget_boost_weight")

        # Select from highest-budget category that has candidates
        sorted_categories = sorted(
            [(cat, budget.get(cat, 0)) for cat in category_plans.keys()],
            key=lambda x: x[1],
            reverse=True
        )

        best_plan = None
        best_score = -1
        best_category = None

        for cat, cat_budget in sorted_categories:
            if category_plans[cat]:
                top_plan, top_score = category_plans[cat][0]
                # Weight by budget for comparison
                weighted_score = top_score * (1 + cat_budget * budget_boost)

                if weighted_score > best_score:
                    best_score = weighted_score
                    best_plan = top_plan
                    best_category = cat

        if best_plan:
            actual_score = best_plan.get("ranking", {}).get("score", 0)

            if self._should_log("log_selection_details"):
                logger.info(
                    f"ORCHESTRATOR_BEST: {symbol} selected {best_plan.get('strategy')} "
                    f"from {best_category} (budget={budget.get(best_category, 0):.0%}) "
                    f"score={actual_score:.3f} from {sum(len(v) for v in category_plans.values())} eligible"
                )

            # Log final selection
            timestamp = now.isoformat() if hasattr(now, 'isoformat') else str(now)
            planning_log = _get_planning_logger()
            if planning_log:
                planning_log.log_accept(
                    symbol,
                    timestamp=timestamp,
                    strategy_type=best_plan.get("strategy"),
                    category=best_category,
                    bias=best_plan.get("bias"),
                    entry_ref_price=best_plan.get("entry_ref_price"),
                    structural_rr=best_plan.get("quality", {}).get("structural_rr"),
                    t1_rr=best_plan["targets"][0]["rr"] if best_plan.get("targets") else None,
                    rank_score=actual_score,
                    quality_status=best_plan.get("quality", {}).get("status"),
                    size_mult=best_plan.get("sizing", {}).get("size_mult"),
                    regime=regime,
                    selected=True,
                    competing_plans=sum(len(v) for v in category_plans.values()),
                    category_budget=budget.get(best_category, 0),
                    # Stop & targets
                    stop_hard=best_plan.get("stop", {}).get("hard"),
                    t2_rr=best_plan["targets"][1]["rr"] if len(best_plan.get("targets", [])) > 1 else None,
                    entry_zone=best_plan.get("entry_zone"),
                    # Position sizing
                    qty=best_plan.get("sizing", {}).get("qty"),
                    notional=best_plan.get("sizing", {}).get("notional"),
                    mis_leverage=best_plan.get("sizing", {}).get("mis_leverage"),
                    risk_per_share=best_plan.get("sizing", {}).get("risk_per_share"),
                    cap_segment=best_plan.get("sizing", {}).get("cap_segment"),
                    # Sizing multiplier breakdown
                    volatility_mult=best_plan.get("sizing", {}).get("volatility_mult"),
                    cap_size_mult=best_plan.get("sizing", {}).get("cap_size_mult"),
                    dir_bias_mult=best_plan.get("sizing", {}).get("dir_bias_mult"),
                    # Rank components
                    rank_components=best_plan.get("ranking", {}).get("components"),
                    # Indicators
                    indicators=best_plan.get("indicators"),
                    # Validated combination gate audit
                    vc_reason=best_plan.get("vc_reason"),
                )

        return best_plan

    def process_candidates_multi(
        self,
        symbols_data: List[Dict[str, Any]],
        regime: str,
        now: pd.Timestamp,
        max_positions: Optional[int] = None,
        regime_diagnostics: Optional[Dict[str, Any]] = None,
        daily_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Process candidates for MULTIPLE symbols with regime-based allocation.

        This is the PROFESSIONAL multi-strategy approach:
        1. Process all candidates and group by category
        2. Rank within each category
        3. Allocate slots based on regime risk budget
        4. Select top candidates from each category proportionally

        Args:
            symbols_data: List of dicts with keys:
                - symbol: str
                - df5m: DataFrame
                - levels: Dict
                - candidates: List
                - daily_df: DataFrame (optional)
                - htf_context: Dict (optional)
                - regime_diagnostics: Dict (optional, per-symbol override)
                - daily_score: float (optional, per-symbol override)
            regime: Current market regime
            now: Current timestamp
            max_positions: Maximum positions to return (uses config if None)
            regime_diagnostics: Default multi-TF regime info for universal ranking adjustments
                               {daily: {regime, confidence}, hourly: {session_bias, confidence}}
            daily_score: Default daily timeframe score for score weighting

        Returns:
            List of selected plans (up to max_positions)
        """
        if max_positions is None:
            max_positions = self._get_config("selection_rules", "max_positions_total")

        # Step 1: Process all candidates and group by category
        category_plans: Dict[str, List[Tuple[Dict[str, Any], float, str]]] = defaultdict(list)

        for data in symbols_data:
            symbol = data["symbol"]
            df5m = data["df5m"]
            levels = data["levels"]
            candidates = data.get("candidates", [])
            daily_df = data.get("daily_df")
            htf_context = data.get("htf_context")
            # Allow per-symbol override, fall back to method-level defaults
            symbol_regime_diag = data.get("regime_diagnostics", regime_diagnostics)
            symbol_daily_score = data.get("daily_score", daily_score)

            for candidate in candidates:
                setup_type = str(candidate.setup_type) if hasattr(candidate, 'setup_type') else str(candidate)
                detected_level = getattr(candidate, 'detected_level', None)
                extras = getattr(candidate, 'extras', None)  # Detector context for trade_report.csv

                # Merge detected_level into levels dict for quality calculation
                candidate_levels = dict(levels)  # Copy to avoid mutating original
                if detected_level is not None:
                    candidate_levels["detected_level"] = detected_level

                plan = self.process_single_candidate(
                    symbol=symbol,
                    setup_type=setup_type,
                    df5m=df5m,
                    levels=candidate_levels,
                    regime=regime,
                    now=now,
                    daily_df=daily_df,
                    htf_context=htf_context,
                    regime_diagnostics=symbol_regime_diag,
                    daily_score=symbol_daily_score
                )

                # Propagate detector extras into plan (see process_candidates for rationale)
                if plan and extras:
                    plan["extras"] = extras

                if plan and plan.get("eligible", False):
                    category = plan.get("category", "UNKNOWN")
                    rank_score = plan.get("ranking", {}).get("score", 0.5)
                    category_plans[category].append((plan, rank_score, symbol))

        if not category_plans:
            logger.info("[ORCHESTRATOR] No eligible plans from any symbols")
            return []

        # Step 2: Sort within each category (category-specific ranking)
        for cat in category_plans:
            category_plans[cat].sort(key=lambda x: x[1], reverse=True)
            if self._should_log("log_category_ranking"):
                logger.debug(
                    f"[ORCHESTRATOR] {cat} has {len(category_plans[cat])} candidates, "
                    f"top score={category_plans[cat][0][1]:.3f}"
                )

        # Step 2b: Apply ORB priority window boost (if applicable)
        category_plans = self._apply_orb_priority(category_plans, now)

        # Step 3: Allocate slots based on regime
        slots = self._allocate_slots(regime, max_positions)

        # Step 4: Select proportionally from each category
        selected_plans: List[Dict[str, Any]] = []
        selected_symbols = set()  # Track to avoid duplicate symbols
        skip_duplicates = self._get_config("selection_rules", "skip_duplicate_symbols")
        categories = self._get_config("categories")

        for category in sorted(categories, key=lambda c: slots.get(c, 0), reverse=True):
            num_slots = slots.get(category, 0)
            if num_slots <= 0:
                continue

            candidates_list = category_plans.get(category, [])
            selected_from_cat = 0

            for plan, score, symbol in candidates_list:
                if selected_from_cat >= num_slots:
                    break

                # Skip if we already have a plan for this symbol
                if skip_duplicates and symbol in selected_symbols:
                    logger.debug(f"[ORCHESTRATOR] Skipping {symbol} - already selected")
                    continue

                selected_plans.append(plan)
                selected_symbols.add(symbol)
                selected_from_cat += 1

                if self._should_log("log_selection_details"):
                    logger.info(
                        f"ORCHESTRATOR_SELECT: {symbol} {plan.get('strategy')} "
                        f"from {category} (slot {selected_from_cat}/{num_slots}) score={score:.3f}"
                    )

            if self._should_log("log_budget_summary"):
                logger.info(
                    f"[ORCHESTRATOR] {category}: selected {selected_from_cat}/{num_slots} slots "
                    f"from {len(candidates_list)} candidates"
                )

        # Step 4b: Sub7 fast-path plans get their own pass.
        # The SMC slot-allocation loop only iterates over the 4 SMC categories from config;
        # "sub7" is not in that list, so sub7 plans would be silently dropped.
        # Drain sub7 plans here, respecting max_positions cap and skip_duplicates.
        sub7_candidates = category_plans.get("sub7", [])
        sub7_selected = 0
        for plan, score, symbol in sub7_candidates:
            if len(selected_plans) >= max_positions:
                break
            if skip_duplicates and symbol in selected_symbols:
                logger.debug(f"[SUB7] Skipping {symbol} — already selected")
                continue
            selected_plans.append(plan)
            selected_symbols.add(symbol)
            sub7_selected += 1
            if self._should_log("log_selection_details"):
                logger.info(
                    f"ORCHESTRATOR_SELECT: {symbol} {plan.get('strategy')} "
                    f"from sub7 (slot {sub7_selected}) score={score:.3f}"
                )
        if sub7_selected and self._should_log("log_budget_summary"):
            logger.info(
                f"[ORCHESTRATOR] sub7: selected {sub7_selected} plans "
                f"from {len(sub7_candidates)} candidates"
            )

        # Log summary
        timestamp = now.isoformat() if hasattr(now, 'isoformat') else str(now)
        cat_counts = defaultdict(int)
        for p in selected_plans:
            cat_counts[p.get("category", "UNKNOWN")] += 1

        if self._should_log("log_budget_summary"):
            logger.info(
                f"ORCHESTRATOR_SUMMARY: Selected {len(selected_plans)}/{max_positions} plans "
                f"for regime={regime}: {dict(cat_counts)}"
            )

        # Log each selected plan
        planning_log = _get_planning_logger()
        if planning_log:
            for plan in selected_plans:
                planning_log.log_accept(
                    plan.get("symbol", "?"),
                    timestamp=timestamp,
                    strategy_type=plan.get("strategy"),
                    category=plan.get("category"),
                    bias=plan.get("bias"),
                    entry_ref_price=plan.get("entry_ref_price"),
                    structural_rr=plan.get("quality", {}).get("structural_rr"),
                    t1_rr=plan["targets"][0]["rr"] if plan.get("targets") else None,
                    rank_score=plan.get("ranking", {}).get("score"),
                    quality_status=plan.get("quality", {}).get("status"),
                    size_mult=plan.get("sizing", {}).get("size_mult"),
                    regime=regime,
                    selected=True,
                    allocation_method="regime_budget",
                    # Stop & targets
                    stop_hard=plan.get("stop", {}).get("hard"),
                    t2_rr=plan["targets"][1]["rr"] if len(plan.get("targets", [])) > 1 else None,
                    entry_zone=plan.get("entry_zone"),
                    # Position sizing
                    qty=plan.get("sizing", {}).get("qty"),
                    notional=plan.get("sizing", {}).get("notional"),
                    mis_leverage=plan.get("sizing", {}).get("mis_leverage"),
                    risk_per_share=plan.get("sizing", {}).get("risk_per_share"),
                    cap_segment=plan.get("sizing", {}).get("cap_segment"),
                    # Sizing multiplier breakdown
                    volatility_mult=plan.get("sizing", {}).get("volatility_mult"),
                    cap_size_mult=plan.get("sizing", {}).get("cap_size_mult"),
                    dir_bias_mult=plan.get("sizing", {}).get("dir_bias_mult"),
                    # Rank components
                    rank_components=plan.get("ranking", {}).get("components"),
                    # Indicators
                    indicators=plan.get("indicators"),
                    # Validated combination gate audit
                    vc_reason=plan.get("vc_reason"),
                )

        return selected_plans

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of all pipelines for diagnostics."""
        return {
            "initialized": [c.value for c in self._pipelines.keys()],
            "failed": self._init_errors,
            "available_categories": self._get_config("categories"),
            "config_loaded": True,
        }

    def get_current_budget(self, regime: str) -> Dict[str, Any]:
        """Get current risk budget allocation for diagnostics."""
        budget = self._get_risk_budget(regime)
        max_positions = self._get_config("selection_rules", "max_positions_total")
        slots = self._allocate_slots(regime, max_positions)
        return {
            "regime": regime,
            "budget_percentages": budget,
            "slots_for_max_positions": slots,
            "max_positions": max_positions
        }


# Singleton instance for easy access
_orchestrator_instance: Optional[PipelineOrchestrator] = None


def get_orchestrator() -> PipelineOrchestrator:
    """Get the singleton orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = PipelineOrchestrator()
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
) -> Optional[Dict[str, Any]]:
    """
    Convenience function for single-symbol processing.

    This is the main entry point for integrating with the existing system.

    Args:
        symbol: Trading symbol
        df5m: 5-minute OHLCV DataFrame
        levels: Key price levels
        regime: Current market regime
        now: Current timestamp
        candidates: List of setup candidates
        daily_df: Daily OHLCV DataFrame (optional)
        htf_context: HTF (15m) context for category-specific ranking
        regime_diagnostics: Multi-TF regime info for universal ranking adjustments
                           {daily: {regime, confidence}, hourly: {session_bias, confidence}}
        daily_score: Daily timeframe score for score weighting
    """
    orchestrator = get_orchestrator()
    return orchestrator.process_candidates(
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
    daily_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Convenience function for multi-symbol professional allocation.

    Use this when you have candidates from multiple symbols and want
    regime-based allocation across categories.

    Args:
        symbols_data: List of dicts with symbol data (may include per-symbol
                     regime_diagnostics and daily_score overrides)
        regime: Current market regime
        now: Current timestamp
        max_positions: Maximum positions to select (uses config if None)
        regime_diagnostics: Default multi-TF regime info for universal ranking adjustments
                           {daily: {regime, confidence}, hourly: {session_bias, confidence}}
        daily_score: Default daily timeframe score for score weighting

    Returns:
        List of selected plans
    """
    orchestrator = get_orchestrator()
    return orchestrator.process_candidates_multi(
        symbols_data=symbols_data,
        regime=regime,
        now=now,
        max_positions=max_positions,
        regime_diagnostics=regime_diagnostics,
        daily_score=daily_score
    )
