# @role: Scheduler utility for updating holiday data
# @used_by: util.py
# @filter_type: utility
# @tags: schedule, holidays, nse
import requests
import json
from pathlib import Path
from config.logging_config import get_agent_logger

logger = get_agent_logger()
# File path for storing downloaded holiday calendar as JSON
HOLIDAY_FILE = Path(__file__).resolve().parents[1] / "assets" / "nse_holidays.json"
# NSE API endpoint returning JSON holiday master
NSE_HOLIDAY_API = "https://www.nseindia.com/api/holiday-master?type=trading"


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

        payload = response.json()
        # Determine where holiday entries live under 'content' or any top-level list
        if isinstance(payload, dict) and "content" in payload:
            items = payload.get("content", [])
        elif isinstance(payload, dict):
            items = []
            for val in payload.values():
                if isinstance(val, list):
                    items.extend(val)
        else:
            raise ValueError("Unexpected JSON structure for holiday data")

        if not items:
            logger.error("❌ No holiday entries found in API response.")
            raise ValueError("Empty holiday data from API.")

        # MERGE with existing entries (don't overwrite). NSE API only returns
        # the current year's forward calendar; an overwrite would wipe
        # historical holidays needed for backtest date filtering. Dedupe by
        # `tradingDate`; entries already on disk take precedence over the API
        # response (preserves any manual hand-curated descriptions).
        existing = []
        if HOLIDAY_FILE.exists():
            try:
                with open(HOLIDAY_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f) or []
            except Exception as load_err:
                logger.warning(
                    "Existing holiday file unreadable, treating as empty: %s",
                    load_err,
                )
                existing = []
        existing_dates = {h.get("tradingDate") for h in existing if isinstance(h, dict)}
        new_items = [
            h for h in items
            if isinstance(h, dict) and h.get("tradingDate") not in existing_dates
        ]
        combined = existing + new_items

        HOLIDAY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HOLIDAY_FILE, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

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
    print(download_nse_holidays())