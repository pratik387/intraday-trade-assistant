"""Period assignment for Indian financial year + Discovery/Validation/Holdout split.

Indian FY = April 1 to March 31. FY2022-23 runs April 2022 through March 2023.

Discovery/Validation/Holdout split is locked at run config time and frozen
thereafter. The gauntlet refuses to run if these dates are modified after
Discovery results are generated.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Tuple


def assign_fy(d: date) -> str:
    """Assign a date to an Indian financial year label (FY2022-23 etc.)."""
    if d.month >= 4:
        start_year = d.year
    else:
        start_year = d.year - 1
    end_year = start_year + 1
    return f"FY{start_year}-{str(end_year)[-2:]}"


@dataclass(frozen=True)
class DiscoveryConfig:
    """Frozen period boundaries. Mutation raises FrozenInstanceError."""
    discovery_start: date
    discovery_end: date
    validation_start: date
    validation_end: date
    holdout_start: date
    holdout_end: date

    def __post_init__(self):
        # Disjoint and ordered
        if not (self.discovery_end < self.validation_start):
            raise ValueError("Discovery must end before Validation starts")
        if not (self.validation_end < self.holdout_start):
            raise ValueError("Validation must end before Holdout starts")


def get_discovery_subperiods(cfg: DiscoveryConfig) -> Tuple[Tuple[date, date], Tuple[date, date]]:
    """Split Discovery into two equal halves by calendar days.

    Returns ((h1_start, h1_end), (h2_start, h2_end)). h1_end + 1 day == h2_start.
    Sub-period PF check requires both halves to be individually positive.
    """
    total_days = (cfg.discovery_end - cfg.discovery_start).days
    mid_offset = total_days // 2
    h1_end = cfg.discovery_start + timedelta(days=mid_offset)
    h2_start = h1_end + timedelta(days=1)
    return ((cfg.discovery_start, h1_end), (h2_start, cfg.discovery_end))
