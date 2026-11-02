# @role: Helper utilities like date handling, price rounding etc.
# @used_by: tools/engine.py
# @filter_type: utility
# @tags: utility, helpers, tools
import pandas as pd
from pathlib import Path
import json
from jobs.refresh_holidays import download_nse_holidays
from config.logging_config import get_agent_logger

logger = get_agent_logger()

HOLIDAY_FILE = Path(__file__).resolve().parents[1] / "assets" / "nse_holidays.json"
_UNCOVERED_YEARS_LOGGED = set()


def is_trading_day(date):
    """
    Returns True if the given date is a valid NSE trading day (not weekend, not holiday).
    """
    try:
        dt = pd.Timestamp(date).normalize()

        # Weekend
        if dt.weekday() >= 5:
            return False

        # Holidays
        if not HOLIDAY_FILE.exists():
            download_nse_holidays()

        with open(HOLIDAY_FILE, "r", encoding="utf-8") as f:
            items = json.load(f)
            holidays = [
                pd.to_datetime(item.get("tradingDate") or item.get("holidayDate"), format="%d-%b-%Y", errors="coerce").normalize()
                for item in items
            ]
            holidays = [d for d in holidays if not pd.isna(d)]

        # Past the last listed year every weekday counts as a trading day (NSE
        # publishes each year's list in December; nothing refreshes the file).
        if holidays and dt.year not in _UNCOVERED_YEARS_LOGGED:
            last = max(h.year for h in holidays)
            if dt.year > last:
                _UNCOVERED_YEARS_LOGGED.add(dt.year)
                logger.error(
                    "NSE holiday calendar has no %d entries (last listed year %d): every %d weekday "
                    "counts as a trading day. Add NSE's %d cash-market holidays to %s.",
                    dt.year, last, dt.year, dt.year, HOLIDAY_FILE,
                )

        return dt not in holidays

    except Exception as e:
        logger.warning(f"is_trading_day fallback triggered: {e}")
        return True  # fallback to assume trading day
