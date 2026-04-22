"""Calibration — decile curve + threshold derivation.

Given predicted vs realized R-multiple from a held-out fold, bucket predictions
into deciles, measure realized R per decile, derive the minimum predicted R
threshold above which realized R >= floor.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def build_decile_calibration(
    predicted: pd.Series, realized: pd.Series, n_buckets: int = 10
) -> pd.DataFrame:
    """Bucket predictions into quantiles, compute realized stats per bucket.

    Args:
        predicted: model predicted R-multiple per trade
        realized: actual realized R-multiple per trade
        n_buckets: number of quantile buckets (default 10 = deciles)

    Returns:
        DataFrame with columns: decile, predicted_lo, predicted_hi,
        realized_median, realized_mean, n
    """
    if len(predicted) == 0 or len(realized) == 0:
        raise ValueError("predicted and realized must be non-empty")
    if len(predicted) != len(realized):
        raise ValueError(
            f"predicted ({len(predicted)}) and realized ({len(realized)}) length mismatch"
        )

    df = pd.DataFrame({"predicted": predicted.values, "realized": realized.values})
    df["decile"] = pd.qcut(df["predicted"], q=n_buckets, labels=False, duplicates="drop")
    grouped = df.groupby("decile", observed=True).agg(
        predicted_lo=("predicted", "min"),
        predicted_hi=("predicted", "max"),
        realized_median=("realized", "median"),
        realized_mean=("realized", "mean"),
        n=("realized", "count"),
    ).reset_index()
    return grouped


def derive_threshold_from_calibration(
    curve: pd.DataFrame, floor: float = 0.3
) -> float:
    """Pick the first decile whose realized_median >= floor; return its predicted_lo.

    If no decile meets the floor, return +inf (gate rejects all predictions).
    """
    sorted_curve = curve.sort_values("decile").reset_index(drop=True)
    for _, row in sorted_curve.iterrows():
        if row["realized_median"] >= floor:
            return float(row["predicted_lo"])
    return float("inf")
