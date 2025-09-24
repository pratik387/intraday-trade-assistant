# structures/__init__.py
"""
Structure-centric trading system architecture.

This package contains the new structure-based trading system where each
trading pattern is encapsulated in its own comprehensive class.
"""

from .base_structure import BaseStructure
from .data_models import StructureEvent, TradePlan, RiskParams, ExitLevels, MarketContext

__all__ = [
    'BaseStructure',
    'StructureEvent',
    'TradePlan',
    'RiskParams',
    'ExitLevels',
    'MarketContext'
]