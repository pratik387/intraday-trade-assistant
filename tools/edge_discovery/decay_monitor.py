"""Decay-and-replace governance for shipped setups.

For each shipped setup, tracks rolling N-month PF and emits status per config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class DecayConfig:
    rolling_window_months: int
    caution_pf_threshold: float
    pause_pf_threshold: float
    retire_pf_threshold: float
    retire_consecutive_months: int


@dataclass
class DecayStatus:
    status: str  # "ACTIVE" | "CAUTION" | "PAUSED" | "RETIRED"
    rolling_pf: float
    latest_month_pf: float
    consecutive_retire_months: int
    notes: str = ""


def compute_status(monthly_pf: pd.Series, config: DecayConfig) -> DecayStatus:
    """Compute decay status from a monthly-PF time-series."""
    if len(monthly_pf) == 0:
        return DecayStatus(
            status="ACTIVE",
            rolling_pf=float("nan"),
            latest_month_pf=float("nan"),
            consecutive_retire_months=0,
            notes="no data",
        )
    pf_sorted = monthly_pf.sort_index()
    latest_month_pf = float(pf_sorted.iloc[-1])
    window = pf_sorted.tail(config.rolling_window_months)
    rolling_pf = float(window.mean())

    # Count consecutive months below retire threshold from the tail
    consecutive_retire = 0
    for v in pf_sorted.iloc[::-1]:
        if v < config.retire_pf_threshold:
            consecutive_retire += 1
        else:
            break

    if consecutive_retire >= config.retire_consecutive_months:
        return DecayStatus(
            status="RETIRED",
            rolling_pf=rolling_pf,
            latest_month_pf=latest_month_pf,
            consecutive_retire_months=consecutive_retire,
            notes=f"{consecutive_retire} consecutive months below {config.retire_pf_threshold}",
        )
    if latest_month_pf < config.pause_pf_threshold:
        return DecayStatus(
            status="PAUSED",
            rolling_pf=rolling_pf,
            latest_month_pf=latest_month_pf,
            consecutive_retire_months=consecutive_retire,
            notes=f"latest_month_pf={latest_month_pf:.2f} below pause={config.pause_pf_threshold}",
        )
    if rolling_pf < config.caution_pf_threshold:
        return DecayStatus(
            status="CAUTION",
            rolling_pf=rolling_pf,
            latest_month_pf=latest_month_pf,
            consecutive_retire_months=consecutive_retire,
            notes=f"rolling_pf={rolling_pf:.2f} below caution={config.caution_pf_threshold}",
        )
    return DecayStatus(
        status="ACTIVE",
        rolling_pf=rolling_pf,
        latest_month_pf=latest_month_pf,
        consecutive_retire_months=consecutive_retire,
    )
