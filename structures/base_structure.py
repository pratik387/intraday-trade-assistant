# structures/base_structure.py
"""
Base structure interface for the structure-centric trading system.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd

from .data_models import (
    StructureEvent,
    TradePlan,
    RiskParams,
    ExitLevels,
    MarketContext,
    StructureAnalysis
)


class BaseStructure(ABC):
    """
    Abstract base class for all trading structures.

    Each trading structure (ORB, VWAP, Support/Resistance, etc.) should inherit
    from this class and implement all required methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the structure with configuration.

        Args:
            config: Structure-specific configuration parameters
        """
        self.config = config
        self.structure_name = self.__class__.__name__.replace('Structure', '').lower()

    @abstractmethod
    def detect(self, market_context: MarketContext) -> StructureAnalysis:
        """
        Detect structure patterns in market data.

        Args:
            market_context: Market data and context for analysis

        Returns:
            StructureAnalysis containing detected events and quality scores
        """
        pass

    @abstractmethod
    def plan_long_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        """
        Generate long trade plan for this structure.

        Args:
            market_context: Current market context

        Returns:
            TradePlan if viable long setup found, None otherwise
        """
        pass

    @abstractmethod
    def plan_short_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        """
        Generate short trade plan for this structure.

        Args:
            market_context: Current market context

        Returns:
            TradePlan if viable short setup found, None otherwise
        """
        pass

    @abstractmethod
    def calculate_risk_params(self, entry_price: float, market_context: MarketContext) -> RiskParams:
        """
        Calculate structure-specific risk parameters.

        Args:
            entry_price: Proposed entry price
            market_context: Current market context

        Returns:
            RiskParams with stop loss and position sizing info
        """
        pass

    @abstractmethod
    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """
        Get structure-specific exit levels and conditions.

        Args:
            trade_plan: Trade plan to generate exits for

        Returns:
            ExitLevels with targets and exit conditions
        """
        pass

    @abstractmethod
    def rank_setup_quality(self, market_context: MarketContext) -> float:
        """
        Score setup quality from 0-100.

        Args:
            market_context: Current market context

        Returns:
            Quality score (0-100, higher is better)
        """
        pass

    @abstractmethod
    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        """
        Check if timing is appropriate for this structure.

        Args:
            current_time: Current timestamp

        Returns:
            True if timing is appropriate, False otherwise
        """
        pass

    def should_detect_at_time(self, current_time: pd.Timestamp) -> bool:
        """
        Check if detector should run at the given time.

        This is called BEFORE detection to avoid wasting resources on
        time-expired setups (e.g., ORB after 10:30).

        Override this in structure implementations that have time constraints.
        Default implementation returns True (always detect).

        Args:
            current_time: Current timestamp

        Returns:
            True if detection should proceed, False to skip detection
        """
        return True

    # Helper methods that can be overridden by specific structures

    def get_structure_levels(self, market_context: MarketContext) -> Dict[str, float]:
        """
        Get structure-specific levels (can be overridden).

        Args:
            market_context: Current market context

        Returns:
            Dictionary of structure-specific levels
        """
        return {}

    def apply_filters(self, trade_plan: TradePlan) -> bool:
        """
        Apply structure-specific filters to trade plan.

        Args:
            trade_plan: Trade plan to validate

        Returns:
            True if trade plan passes all filters, False otherwise
        """
        return True

    def get_structure_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def calculate_confidence(self, market_context: MarketContext) -> float:
        """
        Calculate confidence in structure detection (0-1).

        Args:
            market_context: Current market context

        Returns:
            Confidence level (0.0 to 1.0)
        """
        # Default implementation - structures should override this
        return 0.5

    def __str__(self) -> str:
        """String representation of the structure."""
        return f"{self.__class__.__name__}(name={self.structure_name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name={self.structure_name}, config_keys={list(self.config.keys())})"