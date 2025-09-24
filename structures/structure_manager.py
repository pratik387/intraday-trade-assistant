# structures/structure_manager.py
"""
StructureManager orchestrates all trading structures and coordinates their analysis.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Type
import pandas as pd
from datetime import datetime

from config.logging_config import get_agent_logger
from .base_structure import BaseStructure
from .data_models import (
    StructureEvent,
    TradePlan,
    MarketContext,
    StructureAnalysis
)

logger = get_agent_logger()


class StructureManager:
    """
    Orchestrates all trading structures and coordinates their analysis.

    This class manages the lifecycle of structure detection, strategy planning,
    and ranking across all implemented trading structures.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the StructureManager.

        Args:
            config: Global configuration for all structures
        """
        self.config = config
        self.structures: Dict[str, BaseStructure] = {}
        self.structure_priorities: List[str] = []

        # Manager configuration
        self.max_concurrent_structures = config.get("max_concurrent_structures", 5)
        self.quality_threshold = config.get("quality_threshold", 60.0)

        # Performance tracking
        self._detection_stats: Dict[str, Dict[str, int]] = {}

    def register_structure(self, structure_class: Type[BaseStructure],
                          structure_config: Dict[str, Any],
                          priority: int = 50) -> None:
        """
        Register a trading structure with the manager.

        Args:
            structure_class: Class of the structure to register
            structure_config: Configuration for this specific structure
            priority: Priority level (lower number = higher priority)
        """
        try:
            structure_name = structure_class.__name__.replace('Structure', '').lower()
            structure_instance = structure_class(structure_config)

            self.structures[structure_name] = structure_instance

            # Insert into priority list maintaining sort order
            inserted = False
            for i, (existing_name, existing_priority) in enumerate(self.structure_priorities):
                if priority < existing_priority:
                    self.structure_priorities.insert(i, (structure_name, priority))
                    inserted = True
                    break

            if not inserted:
                self.structure_priorities.append((structure_name, priority))

            # Initialize stats tracking
            self._detection_stats[structure_name] = {
                "detections": 0,
                "plans_generated": 0,
                "successful_rankings": 0,
                "errors": 0
            }

            logger.info(f"Registered structure: {structure_name} with priority {priority}")

        except Exception as e:
            logger.error(f"Failed to register structure {structure_class.__name__}: {e}")
            raise

    def detect_all_structures(self, market_context: MarketContext) -> Dict[str, StructureAnalysis]:
        """
        Run detection across all registered structures.

        Args:
            market_context: Market data and context for analysis

        Returns:
            Dictionary mapping structure names to their analysis results
        """
        results = {}

        for structure_name, _ in self.structure_priorities:
            try:
                structure = self.structures[structure_name]
                analysis = structure.detect(market_context)
                results[structure_name] = analysis

                # Update stats
                self._detection_stats[structure_name]["detections"] += 1
                if analysis.structure_detected:
                    logger.debug(f"Structure detected: {structure_name} for {market_context.symbol}")

            except Exception as e:
                logger.error(f"Structure detection failed for {structure_name} on {market_context.symbol}: {e}")
                self._detection_stats[structure_name]["errors"] += 1
                # Continue with other structures
                continue

        return results

    def generate_trade_plans(self, market_context: MarketContext,
                           detected_structures: Optional[Dict[str, StructureAnalysis]] = None) -> List[TradePlan]:
        """
        Generate trade plans from detected structures.

        Args:
            market_context: Market context for planning
            detected_structures: Pre-computed structure detections (optional)

        Returns:
            List of viable trade plans sorted by quality
        """
        if detected_structures is None:
            detected_structures = self.detect_all_structures(market_context)

        trade_plans = []

        for structure_name, analysis in detected_structures.items():
            if not analysis.structure_detected:
                continue

            try:
                structure = self.structures[structure_name]

                # Generate both long and short plans if applicable
                long_plan = structure.plan_long_strategy(market_context)
                if long_plan:
                    long_plan.structure_type = structure_name
                    trade_plans.append(long_plan)
                    self._detection_stats[structure_name]["plans_generated"] += 1

                short_plan = structure.plan_short_strategy(market_context)
                if short_plan:
                    short_plan.structure_type = structure_name
                    trade_plans.append(short_plan)
                    self._detection_stats[structure_name]["plans_generated"] += 1

            except Exception as e:
                logger.error(f"Trade plan generation failed for {structure_name} on {market_context.symbol}: {e}")
                self._detection_stats[structure_name]["errors"] += 1
                continue

        # Sort by confidence (highest first)
        trade_plans.sort(key=lambda plan: plan.confidence, reverse=True)

        logger.info(f"Generated {len(trade_plans)} trade plans for {market_context.symbol}")
        return trade_plans

    def rank_all_setups(self, market_context: MarketContext) -> Dict[str, float]:
        """
        Rank setup quality across all structures.

        Args:
            market_context: Market context for ranking

        Returns:
            Dictionary mapping structure names to quality scores (0-100)
        """
        rankings = {}

        for structure_name, _ in self.structure_priorities:
            try:
                structure = self.structures[structure_name]

                # Check timing first
                if not structure.validate_timing(market_context.timestamp):
                    rankings[structure_name] = 0.0
                    continue

                score = structure.rank_setup_quality(market_context)
                rankings[structure_name] = max(0.0, min(100.0, score))  # Clamp to 0-100

                self._detection_stats[structure_name]["successful_rankings"] += 1

            except Exception as e:
                logger.error(f"Setup ranking failed for {structure_name} on {market_context.symbol}: {e}")
                rankings[structure_name] = 0.0
                self._detection_stats[structure_name]["errors"] += 1

        return rankings

    def get_best_structure(self, market_context: MarketContext) -> Optional[str]:
        """
        Get the best structure for current market conditions.

        Args:
            market_context: Market context for evaluation

        Returns:
            Name of best structure or None if no good structures found
        """
        rankings = self.rank_all_setups(market_context)

        if not rankings:
            return None

        best_structure = max(rankings.items(), key=lambda x: x[1])

        # Only return if score is above minimum threshold
        min_score = self.config.get("min_structure_score", 25.0)
        if best_structure[1] >= min_score:
            return best_structure[0]

        return None

    def get_structure_by_name(self, name: str) -> Optional[BaseStructure]:
        """
        Get a registered structure by name.

        Args:
            name: Structure name

        Returns:
            Structure instance or None if not found
        """
        return self.structures.get(name.lower())

    def get_registered_structures(self) -> List[str]:
        """
        Get list of all registered structure names.

        Returns:
            List of structure names
        """
        return list(self.structures.keys())

    def get_performance_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get performance statistics for all structures.

        Returns:
            Dictionary with performance stats for each structure
        """
        return self._detection_stats.copy()

    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        for structure_name in self._detection_stats:
            self._detection_stats[structure_name] = {
                "detections": 0,
                "plans_generated": 0,
                "successful_rankings": 0,
                "errors": 0
            }

    def validate_all_structures(self) -> Dict[str, bool]:
        """
        Validate all registered structures have required methods.

        Returns:
            Dictionary mapping structure names to validation status
        """
        validation_results = {}

        required_methods = [
            'detect', 'plan_long_strategy', 'plan_short_strategy',
            'calculate_risk_params', 'get_exit_levels', 'rank_setup_quality',
            'validate_timing'
        ]

        for structure_name, structure in self.structures.items():
            try:
                valid = all(hasattr(structure, method) and callable(getattr(structure, method))
                           for method in required_methods)
                validation_results[structure_name] = valid

                if not valid:
                    logger.warning(f"Structure {structure_name} failed validation")

            except Exception as e:
                logger.error(f"Validation error for {structure_name}: {e}")
                validation_results[structure_name] = False

        return validation_results

    def __str__(self) -> str:
        """String representation of the manager."""
        return f"StructureManager(structures={len(self.structures)})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        structure_names = [name for name, _ in self.structure_priorities]
        return f"StructureManager(structures={structure_names})"