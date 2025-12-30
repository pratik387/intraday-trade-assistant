# pipelines/__init__.py
"""
Category-Based Pipeline Architecture

This module provides specialized pipelines for each setup category:
- BREAKOUT: Momentum breaks of levels (ORB, flag continuation, squeeze release, etc.)
- LEVEL: Bounce/rejection at levels (support bounce, resistance bounce, VWAP, premium/discount zones)
- REVERSION: Mean reversion plays (failure fade, volume spike reversal, gap fill)
- MOMENTUM: Trend continuation (trend pullback, trend continuation)

Each pipeline handles:
- Screening criteria (what filters to apply)
- Quality calculation (how to score the setup)
- Ranking weights (how to prioritize vs other setups)
- Gate logic (what validations to pass)
- Entry zone calculation (how tight/wide)
- Target calculation (R:R specific to category)

Detection and Execution remain COMMON across all pipelines.

Orchestrator:
- PipelineOrchestrator: Routes setup candidates to category-specific pipelines
- process_setup_candidates: Main entry point for processing candidates

Usage:
    from pipelines import process_setup_candidates

    plan = process_setup_candidates(
        symbol="RELIANCE",
        df5m=df5m,
        df1m=df1m,
        levels={"ORH": 2500, "ORL": 2480},
        regime="trend_up",
        now=pd.Timestamp.now(),
        candidates=setup_candidates,
    )
"""

from .base_pipeline import BasePipeline, ConfigurationError
from .breakout_pipeline import BreakoutPipeline
from .level_pipeline import LevelPipeline
from .reversion_pipeline import ReversionPipeline
from .momentum_pipeline import MomentumPipeline
from .orchestrator import (
    PipelineOrchestrator,
    get_orchestrator,
    process_setup_candidates,
)

__all__ = [
    'BasePipeline',
    'ConfigurationError',
    'BreakoutPipeline',
    'LevelPipeline',
    'ReversionPipeline',
    'MomentumPipeline',
    'PipelineOrchestrator',
    'get_orchestrator',
    'process_setup_candidates',
]
