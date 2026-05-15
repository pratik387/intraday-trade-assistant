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

    def top_edge_regions(
        self,
        outcome: str,
        feature_names: list,
        min_n: int,
        top_n: int,
        max_dims: int,
    ) -> list:
        """Rank candidate edge regions by |mean_return| * sqrt(n) (t-statistic proxy).

        Callers must source min_n, top_n, and max_dims from
        config/pipelines/base_config.json under edge_discovery.edge_region_scan.

        Scans 1D, 2D, 3D feature combinations up to max_dims. For each non-empty
        bucket meeting min_n, computes mean/std/n; ranks by abs(mean) * sqrt(n).
        Returns top_n regions as dicts.
        """
        from itertools import combinations
        import math
        if outcome not in self.rows.columns:
            raise KeyError(f"Outcome '{outcome}' not in table columns")
        missing = [f for f in feature_names if f not in self.rows.columns]
        if missing:
            raise KeyError(f"Features not in table columns: {missing}")
        regions: list = []
        for dim in range(1, min(max_dims, len(feature_names)) + 1):
            for combo in combinations(feature_names, dim):
                # Bucket continuous features into quantiles before grouping
                grouped = self.rows.copy()
                for f in combo:
                    if grouped[f].dtype.kind in "fc":
                        try:
                            grouped[f] = pd.qcut(grouped[f], q=5, duplicates="drop")
                        except ValueError:
                            pass
                g = grouped.groupby(list(combo), dropna=False)[outcome]
                stats = g.agg(["count", "mean", "std"]).reset_index()
                stats = stats[stats["count"] >= min_n]
                for _, row in stats.iterrows():
                    n = int(row["count"])
                    mean = float(row["mean"])
                    std = float(row["std"]) if not pd.isna(row["std"]) else 0.0
                    t_proxy = abs(mean) * math.sqrt(n) / max(std, 1e-9)
                    cut = {f: row[f] for f in combo}
                    regions.append({
                        "feature_cut": cut,
                        "n": n,
                        "mean_return": mean,
                        "std_return": std,
                        "t_proxy": t_proxy,
                    })
        regions.sort(key=lambda r: r["t_proxy"], reverse=True)
        return regions[:top_n]
