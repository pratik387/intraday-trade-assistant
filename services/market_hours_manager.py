"""
Market Hours Manager
Handles market hours validation, auto-squareoff, and graceful shutdown.

NSE Trading Hours:
- Pre-open: 09:00 - 09:15
- Normal: 09:15 - 15:30
- Closing: 15:30 - 15:40 (post-close auction)

Our Trading Hours:
- Start accepting signals: 09:15
- Stop new entries: 15:10 (20 min before close)
- Auto-squareoff all: 15:25 (5 min before close)
- Graceful shutdown: 15:35 (after market close)
"""

from datetime import datetime, time
import logging
from typing import Optional, Callable
import pandas as pd
from utils.time_util import _now_naive_ist, _to_naive_ist

logger = logging.getLogger(__name__)


class MarketHoursManager:
    """Manages market hours validation and auto-shutdown"""

    def __init__(
        self,
        market_open: str = "09:15",
        new_entry_cutoff: str = "15:10",
        squareoff_time: str = "15:25",
        shutdown_time: str = "15:35",
        on_squareoff: Optional[Callable] = None,
        on_shutdown: Optional[Callable] = None,
    ):
        """
        Initialize market hours manager.

        Args:
            market_open: Time when trading starts (HH:MM)
            new_entry_cutoff: Stop accepting new trades (HH:MM)
            squareoff_time: Auto-squareoff all positions (HH:MM)
            shutdown_time: Gracefully shutdown system (HH:MM)
            on_squareoff: Callback to trigger when squareoff time reached
            on_shutdown: Callback to trigger when shutdown time reached
        """
        self.market_open = self._parse_time(market_open)
        self.new_entry_cutoff = self._parse_time(new_entry_cutoff)
        self.squareoff_time = self._parse_time(squareoff_time)
        self.shutdown_time = self._parse_time(shutdown_time)

        self.on_squareoff = on_squareoff
        self.on_shutdown = on_shutdown

        # State tracking
        self._squareoff_triggered = False
        self._shutdown_triggered = False

        logger.info(
            f"Market hours configured: Open={market_open}, "
            f"Entry cutoff={new_entry_cutoff}, "
            f"Squareoff={squareoff_time}, "
            f"Shutdown={shutdown_time}"
        )

    @staticmethod
    def _parse_time(time_str: str) -> time:
        """Parse 'HH:MM' string to datetime.time object"""
        h, m = time_str.split(":")
        return time(int(h), int(m))

    def is_market_open(self, now: Optional[pd.Timestamp] = None) -> bool:
        """Check if market is currently open for trading"""
        if now is None:
            now = _now_naive_ist()
        else:
            now = _to_naive_ist(now)

        current_time = now.time()

        # Market is open between 09:15 and 15:30
        return self.market_open <= current_time < time(15, 30)

    def can_enter_new_trade(self, now: Optional[pd.Timestamp] = None) -> bool:
        """
        Check if new trades can be entered.
        Returns False after new_entry_cutoff (default 15:10)
        """
        if now is None:
            now = _now_naive_ist()
        else:
            now = _to_naive_ist(now)

        current_time = now.time()

        # Can enter between market open and entry cutoff
        return self.market_open <= current_time < self.new_entry_cutoff

    def should_squareoff(self, now: Optional[pd.Timestamp] = None) -> bool:
        """
        Check if it's time to squareoff all positions.
        Triggers once at squareoff_time (default 15:25)
        """
        if self._squareoff_triggered:
            return False

        if now is None:
            now = _now_naive_ist()
        else:
            now = _to_naive_ist(now)

        current_time = now.time()

        # Trigger squareoff at or after squareoff_time
        if current_time >= self.squareoff_time:
            self._squareoff_triggered = True
            logger.warning(
                f"[MARKET HOURS] Squareoff time reached ({self.squareoff_time}) - "
                "closing all positions"
            )
            if self.on_squareoff:
                self.on_squareoff()
            return True

        return False

    def should_shutdown(self, now: Optional[pd.Timestamp] = None) -> bool:
        """
        Check if it's time to shutdown the system.
        Triggers once at shutdown_time (default 15:35)
        """
        if self._shutdown_triggered:
            return False

        if now is None:
            now = _now_naive_ist()
        else:
            now = _to_naive_ist(now)

        current_time = now.time()

        # Trigger shutdown at or after shutdown_time
        if current_time >= self.shutdown_time:
            self._shutdown_triggered = True
            logger.warning(
                f"[MARKET HOURS] Shutdown time reached ({self.shutdown_time}) - "
                "stopping trading system"
            )
            if self.on_shutdown:
                self.on_shutdown()
            return True

        return False

    def check_and_trigger(self, now: Optional[pd.Timestamp] = None) -> dict:
        """
        Check all market hours conditions and trigger callbacks.
        Call this periodically (e.g., every minute) from main loop.

        Returns:
            dict with status: {
                'market_open': bool,
                'can_enter': bool,
                'should_squareoff': bool,
                'should_shutdown': bool
            }
        """
        if now is None:
            now = _now_naive_ist()

        status = {
            'market_open': self.is_market_open(now),
            'can_enter': self.can_enter_new_trade(now),
            'should_squareoff': self.should_squareoff(now),
            'should_shutdown': self.should_shutdown(now),
        }

        return status

    def get_time_until_event(self, now: Optional[pd.Timestamp] = None) -> dict:
        """
        Get time remaining until next market event.

        Returns:
            dict with minutes until each event
        """
        if now is None:
            now = _now_naive_ist()
        else:
            now = _to_naive_ist(now)

        current_time = now.time()

        def minutes_until(target_time: time) -> Optional[int]:
            """Calculate minutes until target time (same day only)"""
            target_dt = datetime.combine(now.date(), target_time)
            current_dt = datetime.combine(now.date(), current_time)

            if target_dt <= current_dt:
                return None  # Already passed

            delta = target_dt - current_dt
            return int(delta.total_seconds() / 60)

        return {
            'entry_cutoff': minutes_until(self.new_entry_cutoff),
            'squareoff': minutes_until(self.squareoff_time),
            'shutdown': minutes_until(self.shutdown_time),
        }

    def is_pre_market(self, now: Optional[pd.Timestamp] = None) -> bool:
        """Check if currently in pre-market period (before 09:15)"""
        if now is None:
            now = _now_naive_ist()
        else:
            now = _to_naive_ist(now)

        return now.time() < self.market_open

    def is_post_market(self, now: Optional[pd.Timestamp] = None) -> bool:
        """Check if currently in post-market period (after 15:30)"""
        if now is None:
            now = _now_naive_ist()
        else:
            now = _to_naive_ist(now)

        return now.time() >= time(15, 30)

    def reset_daily_state(self):
        """Reset state for new trading day (call this at start of day)"""
        self._squareoff_triggered = False
        self._shutdown_triggered = False
        logger.info("[MARKET HOURS] Daily state reset - ready for new trading day")


# Singleton instance for easy access
_market_hours_manager: Optional[MarketHoursManager] = None


def get_market_hours_manager() -> MarketHoursManager:
    """Get or create market hours manager singleton"""
    global _market_hours_manager
    if _market_hours_manager is None:
        _market_hours_manager = MarketHoursManager()
    return _market_hours_manager


def set_market_hours_manager(manager: MarketHoursManager):
    """Set custom market hours manager"""
    global _market_hours_manager
    _market_hours_manager = manager
