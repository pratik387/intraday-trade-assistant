"""
EventsLoader - Unified loader for all event types (static and dynamic).

Provides fast lookups for:
- RBI MPC policy dates
- Union Budget dates
- F&O expiry dates (calculated)
- Stock-specific events (earnings, dividends, splits)
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

from config.logging_config import get_agent_logger

logger = get_agent_logger()


class EventsLoader:
    """
    Load and query market events from static config and dynamic cache.

    Usage:
        loader = EventsLoader()

        # Check if trading should be reduced
        if loader.is_rbi_day(date.today()):
            size_mult *= 0.5

        if loader.has_earnings_today("NSE:RELIANCE", date.today()):
            return False  # Skip trade
    """

    def __init__(
        self,
        config_dir: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize events loader.

        Args:
            config_dir: Path to config/events/ for static data
            cache_dir: Path to cache/events/ for dynamic data
        """
        base_path = Path(__file__).parent.parent.parent
        self._config_dir = Path(config_dir) if config_dir else base_path / "config" / "events"
        self._cache_dir = Path(cache_dir) if cache_dir else base_path / "cache" / "events"

        # Pre-load static events
        self._rbi_dates: Dict[str, List[dict]] = {}
        self._budget_dates: Dict[str, List[dict]] = {}
        self._load_static_events()

        # Cache for dynamic events
        self._stock_events: Dict[str, List[dict]] = {}

    def _load_static_events(self) -> None:
        """Load RBI and Budget dates from config files."""
        # Load RBI MPC dates
        rbi_file = self._config_dir / "rbi_policy_dates.json"
        if rbi_file.exists():
            try:
                with open(rbi_file) as f:
                    data = json.load(f)
                    self._rbi_dates = data.get("events", {})
                    logger.debug(f"Loaded RBI dates: {sum(len(v) for v in self._rbi_dates.values())} events")
            except Exception as e:
                logger.warning(f"Failed to load RBI dates: {e}")

        # Load Budget dates
        budget_file = self._config_dir / "budget_dates.json"
        if budget_file.exists():
            try:
                with open(budget_file) as f:
                    data = json.load(f)
                    self._budget_dates = data.get("events", {})
                    logger.debug(f"Loaded Budget dates: {sum(len(v) for v in self._budget_dates.values())} events")
            except Exception as e:
                logger.warning(f"Failed to load Budget dates: {e}")

    def load_stock_events(self, cache_date: Optional[date] = None) -> None:
        """
        Load stock-specific events from dynamic cache.

        Args:
            cache_date: Date for which to load events (default: today)
        """
        cache_date = cache_date or date.today()

        # Load board meetings
        meetings_file = self._cache_dir / "dynamic" / f"board_meetings_{cache_date}.json"
        if meetings_file.exists():
            try:
                with open(meetings_file) as f:
                    meetings = json.load(f)
                    for symbol, events in meetings.items():
                        if symbol not in self._stock_events:
                            self._stock_events[symbol] = []
                        self._stock_events[symbol].extend(events)
            except Exception as e:
                logger.warning(f"Failed to load board meetings: {e}")

        # Load corporate actions
        actions_file = self._cache_dir / "dynamic" / f"corporate_actions_{cache_date}.json"
        if actions_file.exists():
            try:
                with open(actions_file) as f:
                    actions = json.load(f)
                    for symbol, events in actions.items():
                        if symbol not in self._stock_events:
                            self._stock_events[symbol] = []
                        self._stock_events[symbol].extend(events)
            except Exception as e:
                logger.warning(f"Failed to load corporate actions: {e}")

    # ==================== RBI Policy ====================

    def is_rbi_day(self, check_date: date) -> bool:
        """Check if date is an RBI MPC policy announcement day."""
        year_str = str(check_date.year)
        date_str = check_date.strftime("%Y-%m-%d")

        if year_str not in self._rbi_dates:
            return False

        return any(event["date"] == date_str for event in self._rbi_dates[year_str])

    def get_next_rbi_date(self, from_date: date) -> Optional[date]:
        """Get next RBI MPC date from given date."""
        for year in range(from_date.year, from_date.year + 2):
            year_str = str(year)
            if year_str not in self._rbi_dates:
                continue
            for event in self._rbi_dates[year_str]:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
                if event_date >= from_date:
                    return event_date
        return None

    # ==================== Budget ====================

    def is_budget_day(self, check_date: date) -> bool:
        """Check if date is a Union Budget day."""
        year_str = str(check_date.year)
        date_str = check_date.strftime("%Y-%m-%d")

        if year_str not in self._budget_dates:
            return False

        return any(event["date"] == date_str for event in self._budget_dates[year_str])

    # ==================== F&O Expiry ====================

    @staticmethod
    @lru_cache(maxsize=64)
    def get_monthly_expiry(year: int, month: int) -> date:
        """
        Calculate monthly F&O expiry date (last Thursday of month).

        If Thursday is a holiday, expiry moves to Wednesday.
        Note: This doesn't account for exchange holidays - those need manual override.
        """
        # Find last day of month
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        # Find last Thursday
        days_until_thursday = (last_day.weekday() - 3) % 7
        last_thursday = last_day - timedelta(days=days_until_thursday)

        return last_thursday

    @staticmethod
    @lru_cache(maxsize=256)
    def get_weekly_expiry(year: int, month: int, week: int) -> date:
        """
        Calculate weekly F&O expiry date (Thursday of given week).

        Args:
            year: Year
            month: Month (1-12)
            week: Week number (1-5)
        """
        # Find first day of month
        first_day = date(year, month, 1)

        # Find first Thursday
        days_until_thursday = (3 - first_day.weekday()) % 7
        first_thursday = first_day + timedelta(days=days_until_thursday)

        # Get week'th Thursday
        return first_thursday + timedelta(weeks=week - 1)

    def is_expiry_day(self, check_date: date) -> bool:
        """Check if date is a monthly F&O expiry day."""
        monthly_expiry = self.get_monthly_expiry(check_date.year, check_date.month)
        return check_date == monthly_expiry

    def is_weekly_expiry_day(self, check_date: date) -> bool:
        """
        Check if date is a weekly expiry day.

        Note: NSE moved weekly expiry from Thursday to Tuesday starting Sept 2025.
        """
        # Before Sept 2025: Thursday (weekday=3)
        # After Sept 2025: Tuesday (weekday=1)
        cutoff = date(2025, 9, 1)
        if check_date >= cutoff:
            return check_date.weekday() == 1  # Tuesday
        return check_date.weekday() == 3  # Thursday

    def days_to_expiry(self, from_date: date) -> int:
        """Calculate days to monthly expiry."""
        monthly_expiry = self.get_monthly_expiry(from_date.year, from_date.month)
        if from_date > monthly_expiry:
            # Already past this month's expiry, calculate next month
            if from_date.month == 12:
                monthly_expiry = self.get_monthly_expiry(from_date.year + 1, 1)
            else:
                monthly_expiry = self.get_monthly_expiry(from_date.year, from_date.month + 1)
        return (monthly_expiry - from_date).days

    # ==================== Stock Events ====================

    def has_earnings_today(self, symbol: str, check_date: date) -> bool:
        """
        Check if stock has earnings announcement today.

        Matches board meetings that likely contain earnings:
        - "result" (Q1/Q2/Q3/Q4 Results)
        - "financial" (Financial Results)
        - "quarter" (Quarterly Results)
        - "board meeting" (conservative: any board meeting)
        """
        # Normalize symbol
        symbol = symbol.upper()
        if not symbol.startswith("NSE:"):
            symbol = f"NSE:{symbol}"

        if symbol not in self._stock_events:
            return False

        date_str = check_date.strftime("%Y-%m-%d")
        earnings_keywords = ["result", "financial", "quarter", "earning"]

        for event in self._stock_events[symbol]:
            if event.get("date") != date_str:
                continue
            purpose = event.get("purpose", "").lower()
            # Check if any earnings keyword present
            if any(kw in purpose for kw in earnings_keywords):
                return True
            # Board meeting type events are also likely earnings
            if event.get("type") == "board_meeting":
                return True
        return False

    def has_earnings_tomorrow(self, symbol: str, check_date: date) -> bool:
        """Check if stock has earnings announcement tomorrow."""
        return self.has_earnings_today(symbol, check_date + timedelta(days=1))

    def get_stock_events(self, symbol: str, check_date: date) -> List[dict]:
        """Get all events for a stock on a given date."""
        symbol = symbol.upper()
        if not symbol.startswith("NSE:"):
            symbol = f"NSE:{symbol}"

        if symbol not in self._stock_events:
            return []

        date_str = check_date.strftime("%Y-%m-%d")
        return [
            event for event in self._stock_events[symbol]
            if event.get("date") == date_str
        ]

    # ==================== Aggregate Queries ====================

    def get_macro_events(self, check_date: date) -> List[dict]:
        """Get all macro events (RBI, Budget, Expiry) for a date."""
        events = []
        date_str = check_date.strftime("%Y-%m-%d")

        if self.is_rbi_day(check_date):
            events.append({"date": date_str, "type": "rbi_mpc", "impact": "high"})

        if self.is_budget_day(check_date):
            events.append({"date": date_str, "type": "union_budget", "impact": "high"})

        if self.is_expiry_day(check_date):
            events.append({"date": date_str, "type": "monthly_expiry", "impact": "medium"})
        elif self.is_weekly_expiry_day(check_date):
            events.append({"date": date_str, "type": "weekly_expiry", "impact": "low"})

        return events

    def is_high_impact_day(self, check_date: date) -> bool:
        """Check if date has high-impact macro events (RBI or Budget)."""
        return self.is_rbi_day(check_date) or self.is_budget_day(check_date)

    def get_trading_multiplier(self, check_date: date, symbol: Optional[str] = None) -> Tuple[float, str]:
        """
        Get position sizing multiplier based on events.

        Returns:
            (multiplier, reason) - e.g., (0.5, "rbi_policy_day")
        """
        # Stock-specific events take priority
        if symbol:
            if self.has_earnings_today(symbol, check_date):
                return (0.0, "earnings_day")  # Don't trade
            if self.has_earnings_tomorrow(symbol, check_date):
                return (0.5, "earnings_tomorrow")

        # Macro events
        if self.is_budget_day(check_date):
            return (0.0, "budget_day")  # Don't trade

        if self.is_rbi_day(check_date):
            return (0.5, "rbi_policy_day")

        if self.is_expiry_day(check_date):
            return (0.75, "monthly_expiry")

        return (1.0, None)
