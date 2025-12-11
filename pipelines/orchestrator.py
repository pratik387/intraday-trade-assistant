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
from pipelines.base_pipeline import BasePipeline, ConfigurationError

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
        from services.gates.regime_gate import HARD_BLOCKS

        # Normalize regime key (choppy/range → chop)
        regime_key = regime
        if regime in {"choppy", "range"}:
            regime_key = "chop"

        blocked_setups = HARD_BLOCKS.get(regime_key, [])
        return setup_type in blocked_setups

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
                    logger.info(f"ORB_PRIORITY: {symbol} {strategy} score boosted {score:.2f} → {new_score:.2f}")
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
        df1m: Optional[pd.DataFrame],
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
            df1m: 1-minute OHLCV DataFrame (optional)
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
                df1m=df1m,
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
                        regime=regime
                    )

            return plan

        except Exception as e:
            logger.exception(f"[ORCHESTRATOR] Error processing {symbol} {setup_type}: {e}")
            return None

    def process_candidates(
        self,
        symbol: str,
        df5m: pd.DataFrame,
        df1m: Optional[pd.DataFrame],
        levels: Dict[str, float],
        regime: str,
        now: pd.Timestamp,
        candidates: List[Any],
        daily_df: Optional[pd.DataFrame] = None,
        htf_context: Optional[Dict[str, Any]] = None,
        regime_diagnostics: Optional[Dict[str, Any]] = None,
        daily_score: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Process candidates for a SINGLE symbol - returns best plan.

        For single-symbol processing, we still use category-specific ranking
        but select the best from the highest-budget category that has candidates.

        Args:
            symbol: Trading symbol
            df5m: 5-minute OHLCV DataFrame
            df1m: 1-minute OHLCV DataFrame (optional)
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

            plan = self.process_single_candidate(
                symbol=symbol,
                setup_type=setup_type,
                df5m=df5m,
                df1m=df1m,
                levels=levels,
                regime=regime,
                now=now,
                daily_df=daily_df,
                htf_context=htf_context,
                regime_diagnostics=regime_diagnostics,
                daily_score=daily_score,
                cap_segment=cap_segment
            )

            if plan and plan.get("eligible", False):
                category = plan.get("category", "UNKNOWN")
                rank_score = plan.get("ranking", {}).get("score", strength)
                category_plans[category].append((plan, rank_score))

        if not category_plans:
            logger.debug(f"[ORCHESTRATOR] {symbol}: No eligible plans from {len(candidates)} candidates")
            return None

        # Sort each category by rank score (within-category ranking)
        for cat in category_plans:
            category_plans[cat].sort(key=lambda x: x[1], reverse=True)

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
                    category_budget=budget.get(best_category, 0)
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
                - df1m: DataFrame (optional)
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
            df1m = data.get("df1m")
            levels = data["levels"]
            candidates = data.get("candidates", [])
            daily_df = data.get("daily_df")
            htf_context = data.get("htf_context")
            # Allow per-symbol override, fall back to method-level defaults
            symbol_regime_diag = data.get("regime_diagnostics", regime_diagnostics)
            symbol_daily_score = data.get("daily_score", daily_score)

            for candidate in candidates:
                setup_type = str(candidate.setup_type) if hasattr(candidate, 'setup_type') else str(candidate)

                plan = self.process_single_candidate(
                    symbol=symbol,
                    setup_type=setup_type,
                    df5m=df5m,
                    df1m=df1m,
                    levels=levels,
                    regime=regime,
                    now=now,
                    daily_df=daily_df,
                    htf_context=htf_context,
                    regime_diagnostics=symbol_regime_diag,
                    daily_score=symbol_daily_score
                )

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
                    allocation_method="regime_budget"
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
    df1m: Optional[pd.DataFrame],
    levels: Dict[str, float],
    regime: str,
    now: pd.Timestamp,
    candidates: List[Any],
    daily_df: Optional[pd.DataFrame] = None,
    htf_context: Optional[Dict[str, Any]] = None,
    regime_diagnostics: Optional[Dict[str, Any]] = None,
    daily_score: float = 0.0
) -> Optional[Dict[str, Any]]:
    """
    Convenience function for single-symbol processing.

    This is the main entry point for integrating with the existing system.

    Args:
        symbol: Trading symbol
        df5m: 5-minute OHLCV DataFrame
        df1m: 1-minute OHLCV DataFrame (optional)
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
        df1m=df1m,
        levels=levels,
        regime=regime,
        now=now,
        candidates=candidates,
        daily_df=daily_df,
        htf_context=htf_context,
        regime_diagnostics=regime_diagnostics,
        daily_score=daily_score
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
