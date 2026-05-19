"""Event-calendar features: expiry, monthly expiry, rebalance, RBI, budget."""
from __future__ import annotations

from calendar import monthrange
from datetime import date
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from tools.edge_discovery.types import Event


def _is_thursday(d: date) -> bool:
    return d.weekday() == 3


def _is_last_thursday_of_month(d: date) -> bool:
    last_day = monthrange(d.year, d.month)[1]
    last_date = date(d.year, d.month, last_day)
    days_back = (last_date.weekday() - 3) % 7
    last_thu = date(d.year, d.month, last_day - days_back)
    return d == last_thu


class EventCalendarFeatures:
    name = "event_calendar_features"
    feature_names: List[str] = [
        "is_expiry_day",
        "is_expiry_week",
        "is_monthly_expiry_day",
        "days_to_next_earnings",
        "is_index_rebalance_day",
        "is_rbi_policy_day",
        "is_budget_day",
    ]

    def __init__(
        self,
        rbi_policy_dates: Optional[Set[date]] = None,
        index_rebalance_dates: Optional[Set[date]] = None,
        budget_dates: Optional[Set[date]] = None,
    ) -> None:
        self.rbi_policy_dates = rbi_policy_dates or set()
        self.index_rebalance_dates = index_rebalance_dates or set()
        self.budget_dates = budget_dates or set()

    def compute(
        self,
        event: Event,
        bars: pd.DataFrame,
        next_earnings_date: Optional[date] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        d = event.event_time.date()
        is_expiry_day = _is_thursday(d)
        # expiry week = the week (Mon-Fri) containing a Thursday
        days_to_thu = (3 - d.weekday()) % 7
        is_expiry_week = days_to_thu <= 4 and d.weekday() <= 4
        is_monthly_expiry = is_expiry_day and _is_last_thursday_of_month(d)
        days_to_earnings = (next_earnings_date - d).days if next_earnings_date is not None else None
        return {
            "is_expiry_day": is_expiry_day,
            "is_expiry_week": is_expiry_week,
            "is_monthly_expiry_day": is_monthly_expiry,
            "days_to_next_earnings": days_to_earnings,
            "is_index_rebalance_day": d in self.index_rebalance_dates,
            "is_rbi_policy_day": d in self.rbi_policy_dates,
            "is_budget_day": d in self.budget_dates,
        }
