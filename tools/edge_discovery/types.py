"""Core types for the Edge-First Discovery Framework.

All timestamps are IST-naive (no tzinfo). See utils/time_util.py for converters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd


@dataclass
class Event:
    """One event in a population. Pattern-specific labels go in metadata."""
    symbol: str
    event_time: pd.Timestamp  # IST-naive (no tzinfo)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_time.tzinfo is not None:
            raise ValueError(
                f"Event.event_time must be IST-naive (no tzinfo). "
                f"Got tz={self.event_time.tzinfo}. Use utils.time_util.to_naive_ist() to convert."
            )


@dataclass
class ConditionalOutcomeTable:
    """Output of the explorer. One row per event; columns = features + outcomes."""
    rows: pd.DataFrame

    def slice_by(self, feature: str, outcome: str) -> pd.DataFrame:
        """Aggregate stats of `outcome` bucketed by `feature` values."""
        if feature not in self.rows.columns:
            raise KeyError(f"Feature '{feature}' not in table columns")
        if outcome not in self.rows.columns:
            raise KeyError(f"Outcome '{outcome}' not in table columns")
        g = self.rows.groupby(feature, dropna=False)[outcome]
        return pd.DataFrame({
            "n": g.count(),
            "mean": g.mean(),
            "median": g.median(),
            "std": g.std(ddof=1),
        })

    def joint_slice(self, *features: str, outcome: str) -> pd.DataFrame:
        """2D / 3D slicing — same return shape as slice_by but on tuple keys."""
        for f in features:
            if f not in self.rows.columns:
                raise KeyError(f"Feature '{f}' not in table columns")
        if outcome not in self.rows.columns:
            raise KeyError(f"Outcome '{outcome}' not in table columns")
        g = self.rows.groupby(list(features), dropna=False)[outcome]
        return pd.DataFrame({
            "n": g.count(),
            "mean": g.mean(),
            "median": g.median(),
            "std": g.std(ddof=1),
        })
