# @role: Scheduler utility for updating holiday data
# @used_by: util.py
# @filter_type: utility
# @tags: schedule, holidays, nse
import requests
import json
import sys
from datetime import datetime
from pathlib import Path
from config.logging_config import get_agent_logger

logger = get_agent_logger()
# File path for storing downloaded holiday calendar as JSON
HOLIDAY_FILE = Path(__file__).resolve().parents[1] / "assets" / "nse_holidays.json"
# NSE API endpoint returning JSON holiday master
NSE_HOLIDAY_API = "https://www.nseindia.com/api/holiday-master?type=trading"
# The API returns one list per market segment (CM, FO, CD, COM, IRD, ...). Only the
# cash-market list is the equity calendar: the currency/interest-rate segments close on
# RBI bank holidays (New Year, Annual Bank Closing, Shivaji Jayanti, Id-E-Milad, ...)
# when equities trade. Merging every list put 10 such trading days into the file as
# holidays in 2025-26 (e.g. 2026-08-26: no paper session, overnight entry skipped).
EQUITY_SEGMENT = "CM"
DATE_FORMAT = "%d-%b-%Y"


def _holiday_date(row: dict):
    return datetime.strptime(str(row.get("tradingDate")), DATE_FORMAT).date()


def equity_holidays(payload) -> list:
    """Cash-market (equity) weekday holiday entries from the NSE holiday-master response.

    Raises ValueError (so nothing is written) when the CM segment is missing or any
    row's tradingDate does not parse: services/price_bands.load_nse_holidays parses
    every row strictly at daemon start, so one bad row would stop the daemons.
    Weekend closures and duplicate dates are dropped (they carry no trading-day
    information)."""
    if not isinstance(payload, dict) or EQUITY_SEGMENT not in payload:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"NSE holiday response has no '{EQUITY_SEGMENT}' segment (got {keys})")
    out, seen = [], set()
    for row in payload.get(EQUITY_SEGMENT) or []:
        if not isinstance(row, dict):
            continue
        d = _holiday_date(row)
        if d.weekday() >= 5 or d in seen:
            continue
        seen.add(d)
        out.append(row)
    return out


def download_nse_holidays():
    """
    Download and save the NSE holiday calendar using the JSON API.
    Saves the raw list of holiday entries to a JSON file for easier downstream manipulation.
    Returns a dict with status and count or error.
    """
    try:
        logger.info("🌐 Fetching NSE holiday calendar from JSON API...")
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com"
        })
        # Hit base page to set cookies
        session.get("https://www.nseindia.com", timeout=5)
        response = session.get(NSE_HOLIDAY_API, timeout=10)
        response.raise_for_status()

        items = equity_holidays(response.json())

        if not items:
            logger.error("❌ No holiday entries found in API response.")
            raise ValueError("Empty holiday data from API.")

        # MERGE with existing entries (don't overwrite). NSE API only returns
        # the current year's forward calendar; an overwrite would wipe
        # historical holidays needed for backtest date filtering. Dedupe by
        # `tradingDate`; entries already on disk take precedence over the API
        # response (preserves any manual hand-curated descriptions).
        # An existing file that cannot be read fails the refresh instead of being
        # replaced by the API's one-year list, which would wipe the history.
        existing = []
        if HOLIDAY_FILE.exists():
            with open(HOLIDAY_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f) or []
        existing_dates = {_holiday_date(h) for h in existing if isinstance(h, dict)}
        new_items = [h for h in items if _holiday_date(h) not in existing_dates]
        combined = sorted(existing + new_items, key=_holiday_date)

        HOLIDAY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = HOLIDAY_FILE.with_suffix(HOLIDAY_FILE.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        tmp.replace(HOLIDAY_FILE)

        added = len(new_items)
        total = len(combined)
        logger.info(
            "✅ NSE holidays merged into %s | added=%d new | total=%d entries",
            HOLIDAY_FILE, added, total,
        )
        return {"status": "success", "added": added, "total": total}

    except Exception as e:
        logger.exception("❌ Failed to download NSE holidays via API: %s", e)
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    result = download_nse_holidays()
    print(result)
    sys.exit(0 if result.get("status") == "success" else 1)