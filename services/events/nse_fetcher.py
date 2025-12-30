"""
NSE Events Fetcher - Fetch corporate actions and board meetings from NSE.

Requires: pip install nseindiaapi

Rate limit: NSE allows 3 requests/second max
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time

from config.logging_config import get_agent_logger

logger = get_agent_logger()

# Try to import NSE library
# Package: https://pypi.org/project/nse/
try:
    from nse import NSE
    NSE_AVAILABLE = True
except ImportError:
    NSE_AVAILABLE = False
    logger.warning("nse package not installed. Run: pip install nse[local]")


class NSEEventsFetcher:
    """
    Fetch corporate actions and board meetings from NSE India.

    Usage:
        fetcher = NSEEventsFetcher()
        meetings = fetcher.fetch_upcoming_board_meetings(days_ahead=7)
        actions = fetcher.fetch_corporate_actions(days_ahead=7)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize NSE fetcher.

        Args:
            cache_dir: Directory to cache fetched data
        """
        base_path = Path(__file__).parent.parent.parent
        self._cache_dir = Path(cache_dir) if cache_dir else base_path / "cache" / "events" / "dynamic"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # NSE library needs a download folder for cookies/temp files
        self._nse_download_dir = self._cache_dir / "nse_temp"
        self._nse_download_dir.mkdir(parents=True, exist_ok=True)

        self._nse = None
        self._request_delay = 0.5  # 500ms between requests to respect rate limits

        if NSE_AVAILABLE:
            try:
                # NSE requires download_folder, server=False for local usage
                self._nse = NSE(download_folder=self._nse_download_dir, server=False)
                logger.info("NSE connection initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NSE connection: {e}")

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        time.sleep(self._request_delay)

    def _normalize_date(self, date_str: str) -> str:
        """
        Convert NSE date format to YYYY-MM-DD.

        NSE returns dates like "22-Dec-2025" or "2025-12-22"
        """
        if not date_str:
            return ""

        date_str = str(date_str).strip()

        # Already in YYYY-MM-DD format
        if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
            return date_str

        # Format: DD-Mon-YYYY (e.g., "22-Dec-2025")
        try:
            dt = datetime.strptime(date_str, "%d-%b-%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

        # Format: DD/MM/YYYY
        try:
            dt = datetime.strptime(date_str, "%d/%m/%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

        # Return as-is if can't parse
        return date_str

    def fetch_upcoming_board_meetings(
        self,
        days_ahead: int = 7,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, List[dict]]:
        """
        Fetch upcoming board meetings (earnings announcements).

        Args:
            days_ahead: Number of days ahead to fetch
            symbols: Optional list of symbols to filter (default: all)

        Returns:
            Dictionary mapping symbols to list of meetings:
            {
                "NSE:RELIANCE": [{"date": "2024-01-19", "purpose": "Q3 Results"}],
                "NSE:TCS": [{"date": "2024-01-11", "purpose": "Q3 Results"}]
            }
        """
        if not self._nse:
            logger.warning("NSE not available, returning empty results")
            return {}

        results: Dict[str, List[dict]] = {}
        # NSE API expects datetime objects, not date objects
        from_dt = datetime.combine(date.today(), datetime.min.time())
        to_dt = datetime.combine(date.today() + timedelta(days=days_ahead), datetime.min.time())

        try:
            self._rate_limit()

            # Fetch board meetings from NSE using boardMeetings() method
            # API: boardMeetings(index='equities', symbol=None, fno=False, from_date=None, to_date=None)
            meetings = self._nse.boardMeetings(
                index="equities",
                symbol=None,  # All symbols
                from_date=from_dt,
                to_date=to_dt
            )

            if not meetings:
                logger.debug("No board meetings found")
                return results

            # meetings is a List[Dict], not a DataFrame
            for meeting in meetings:
                symbol = meeting.get("symbol", "")
                if not symbol:
                    continue

                # Normalize symbol
                symbol_key = f"NSE:{symbol}"

                # Filter if symbols provided
                if symbols and symbol_key not in symbols:
                    continue

                meeting_date = meeting.get("bm_date") or meeting.get("date", "")
                purpose = meeting.get("bm_purpose") or meeting.get("purpose", "Board Meeting")

                if symbol_key not in results:
                    results[symbol_key] = []

                results[symbol_key].append({
                    "date": self._normalize_date(meeting_date),
                    "purpose": str(purpose),
                    "type": "board_meeting"
                })

            logger.info(f"Fetched board meetings for {len(results)} symbols")

        except Exception as e:
            logger.error(f"Failed to fetch board meetings: {e}")

        return results

    def fetch_corporate_actions(
        self,
        days_ahead: int = 7,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, List[dict]]:
        """
        Fetch corporate actions (dividends, splits, bonuses).

        Args:
            days_ahead: Number of days ahead to fetch
            symbols: Optional list of symbols to filter

        Returns:
            Dictionary mapping symbols to list of actions:
            {
                "NSE:RELIANCE": [{"date": "2024-01-19", "type": "dividend", "value": "8.0"}]
            }
        """
        if not self._nse:
            logger.warning("NSE not available, returning empty results")
            return {}

        results: Dict[str, List[dict]] = {}
        # NSE API expects datetime objects
        from_dt = datetime.combine(date.today(), datetime.min.time())
        to_dt = datetime.combine(date.today() + timedelta(days=days_ahead), datetime.min.time())

        try:
            self._rate_limit()

            # Fetch corporate actions from NSE using actions() method
            # API: actions(segment='equities', symbol=None, from_date=None, to_date=None)
            actions = self._nse.actions(
                segment="equities",
                symbol=None,
                from_date=from_dt,
                to_date=to_dt
            )

            if not actions:
                logger.debug("No corporate actions found")
                return results

            # actions is a List[Dict], not a DataFrame
            for action in actions:
                symbol = action.get("symbol", "")
                if not symbol:
                    continue

                symbol_key = f"NSE:{symbol}"

                if symbols and symbol_key not in symbols:
                    continue

                action_date = action.get("exDate") or action.get("ex_date") or action.get("date", "")
                action_type = action.get("subject", "")
                action_value = action.get("dividend", "") or action.get("subject", "")

                # Parse action type
                action_category = "other"
                action_lower = action_type.lower() if action_type else ""
                if "dividend" in action_lower:
                    action_category = "dividend"
                elif "split" in action_lower:
                    action_category = "split"
                elif "bonus" in action_lower:
                    action_category = "bonus"
                elif "rights" in action_lower:
                    action_category = "rights"

                if symbol_key not in results:
                    results[symbol_key] = []

                results[symbol_key].append({
                    "date": self._normalize_date(action_date),
                    "type": action_category,
                    "subject": str(action_type),
                    "value": str(action_value)
                })

            logger.info(f"Fetched corporate actions for {len(results)} symbols")

        except Exception as e:
            logger.error(f"Failed to fetch corporate actions: {e}")

        return results

    def save_to_cache(
        self,
        meetings: Dict[str, List[dict]],
        actions: Dict[str, List[dict]],
        cache_date: Optional[date] = None
    ) -> None:
        """
        Save fetched events to cache.

        Args:
            meetings: Board meetings data
            actions: Corporate actions data
            cache_date: Date for cache file naming
        """
        cache_date = cache_date or date.today()

        if meetings:
            meetings_file = self._cache_dir / f"board_meetings_{cache_date}.json"
            with open(meetings_file, "w") as f:
                json.dump(meetings, f, indent=2)
            logger.info(f"Saved board meetings to {meetings_file}")

        if actions:
            actions_file = self._cache_dir / f"corporate_actions_{cache_date}.json"
            with open(actions_file, "w") as f:
                json.dump(actions, f, indent=2)
            logger.info(f"Saved corporate actions to {actions_file}")

    def fetch_and_cache(self, days_ahead: int = 7) -> None:
        """
        Fetch all events and save to cache.

        This is the main method to call from daily pre-market job.
        """
        logger.info(f"Fetching events for next {days_ahead} days...")

        meetings = self.fetch_upcoming_board_meetings(days_ahead=days_ahead)
        actions = self.fetch_corporate_actions(days_ahead=days_ahead)

        self.save_to_cache(meetings, actions)

        total_events = sum(len(v) for v in meetings.values()) + sum(len(v) for v in actions.values())
        logger.info(f"Fetched and cached {total_events} total events")
