"""
fetch_daily_events.py - Pre-market job to fetch corporate events from NSE.

Run this job before market open (e.g., 8:30 AM IST) to fetch:
- Upcoming board meetings (earnings announcements)
- Corporate actions (dividends, splits, bonuses)

Usage:
    python -m jobs.fetch_daily_events
    python -m jobs.fetch_daily_events --days 14

The fetched data is saved to cache/events/dynamic/ and loaded by EventsLoader.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import get_agent_logger

logger = get_agent_logger()


def fetch_daily_events(days_ahead: int = 7) -> bool:
    """
    Fetch corporate events from NSE and save to cache.

    Args:
        days_ahead: Number of days to look ahead for events

    Returns:
        True if successful, False otherwise
    """
    try:
        from services.events.nse_fetcher import NSEEventsFetcher, NSE_AVAILABLE

        if not NSE_AVAILABLE:
            logger.warning(
                "NSE library not available. Install with: pip install nseindiaapi"
            )
            return False

        logger.info(f"Fetching NSE events for next {days_ahead} days...")

        fetcher = NSEEventsFetcher()
        fetcher.fetch_and_cache(days_ahead=days_ahead)

        logger.info("Daily events fetch completed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to fetch daily events: {e}")
        return False


def check_todays_events() -> None:
    """Log summary of today's events."""
    try:
        from services.events.events_loader import EventsLoader

        loader = EventsLoader()
        loader.load_stock_events()

        today = date.today()

        # Check macro events
        macro_events = loader.get_macro_events(today)
        if macro_events:
            logger.info(f"Today's macro events: {macro_events}")
        else:
            logger.info("No macro events today")

        # Check if high impact day
        if loader.is_high_impact_day(today):
            mult, reason = loader.get_trading_multiplier(today)
            logger.warning(f"HIGH IMPACT DAY: {reason}, trading multiplier: {mult}")

    except Exception as e:
        logger.error(f"Failed to check today's events: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch corporate events from NSE for event-based trading gates"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days ahead to fetch (default: 7)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check today's events without fetching"
    )

    args = parser.parse_args()

    if args.check_only:
        check_todays_events()
    else:
        success = fetch_daily_events(days_ahead=args.days)
        if success:
            check_todays_events()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
