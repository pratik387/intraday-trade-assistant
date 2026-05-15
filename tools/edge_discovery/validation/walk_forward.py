"""Walk-forward simulation.

For each step: train on N months, test on next M months, walk forward by S months.
Records the validation PF series; stability = 1 - (std/mean).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    train_window_months: int
    test_window_months: int
    step_months: int


@dataclass
class WalkForwardResult:
    validation_pfs: List[float]
    stability_score: float
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)


def _pf(returns: pd.Series) -> float:
    pos_sum = float(returns[returns > 0].sum())
    neg_sum = float(-returns[returns < 0].sum())
    if neg_sum <= 0:
        return float("inf") if pos_sum > 0 else 0.0
    return pos_sum / neg_sum


def walk_forward(trades: pd.DataFrame, config: WalkForwardConfig) -> WalkForwardResult:
    """trades must have columns ['entry_time', 'net_return'] sorted by entry_time."""
    if "entry_time" not in trades.columns or "net_return" not in trades.columns:
        raise KeyError("trades DataFrame must have columns ['entry_time', 'net_return']")
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = df.sort_values("entry_time").reset_index(drop=True)
    start = df["entry_time"].min().to_period("M").to_timestamp()
    end = df["entry_time"].max().to_period("M").to_timestamp() + pd.DateOffset(months=1)

    validation_pfs: List[float] = []
    detail_rows = []
    cur_train_start = start
    while True:
        train_end = cur_train_start + pd.DateOffset(months=config.train_window_months)
        test_end = train_end + pd.DateOffset(months=config.test_window_months)
        if test_end > end:
            break
        test_mask = (df["entry_time"] >= train_end) & (df["entry_time"] < test_end)
        test_trades = df.loc[test_mask, "net_return"]
        if len(test_trades) >= 5:
            pf = _pf(test_trades)
            validation_pfs.append(pf if pf != float("inf") else 5.0)
            detail_rows.append({
                "train_start": cur_train_start, "train_end": train_end,
                "test_end": test_end, "test_n": int(len(test_trades)), "test_pf": pf,
            })
        cur_train_start = cur_train_start + pd.DateOffset(months=config.step_months)

    if len(validation_pfs) < 2:
        return WalkForwardResult(validation_pfs=validation_pfs, stability_score=0.0,
                                  detail=pd.DataFrame(detail_rows))
    arr = np.array(validation_pfs)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    stability = 1.0 - (std / mean) if mean > 0 else 0.0
    stability = max(0.0, min(1.0, stability))
    return WalkForwardResult(
        validation_pfs=validation_pfs,
        stability_score=stability,
        detail=pd.DataFrame(detail_rows),
    )
