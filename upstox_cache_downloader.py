"""
Downloads 5-minute historical candles from Upstox for Jan 1, 2023 to Aug 13, 2025.
Assumes the Upstox Historical API is open and does not require authentication.
Uses instrument keys per symbol (filtered to match NSE list in cache) and saves to feather files for backtesting.

⚠️ Upstox Rate Limit Guidance:
- Max ~25 requests/second for historical endpoints.
- Recommended max_workers = 3 to 5
- Use exponential backoff on HTTP 429 responses (2s → 4s → 8s)
- Pause after every 20–25 requests to avoid hitting burst limits
"""
import os
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- Constants ---
START_DATE = "2025-07-01"
END_DATE = "2025-08-29"
INTERVAL = "1"
INTERVAL_UNIT = "days"  # "minutes", "days", "weeks", "hours"
MAX_WORKERS = 5

# --- Output Path ---
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = ROOT / "intraday-trade-assistant" / "cache" / "ohlcv_archive"
NSE_SYMBOLS_PATH = ROOT / "intraday-trade-assistant" / "nse_all.json"
UPSTOX_JSON_PATH = ROOT / "intraday-trade-assistant" / "upstox_instruments.json"
UPSTOX_MAP_SAVE_PATH = ROOT / "short-trade-assistant"/ "upstox_instrument_map.json"

# --- Headers ---
HEADERS = {
    "Accept": "application/json"
}

# --- Load instrument keys filtered from NDJSON based on NSE symbols ---
def load_instrument_map_ndjson(ndjson_path: str):
    if Path(UPSTOX_MAP_SAVE_PATH).exists():
        with open(UPSTOX_MAP_SAVE_PATH) as f:
            return json.load(f)

    with open(NSE_SYMBOLS_PATH) as f:
        allowed_data = json.load(f)
        allowed_symbols = set(item["symbol"] for item in allowed_data if "symbol" in item)

    instrument_map = {}
    with open(ndjson_path, "r") as f:
        items = json.load(f)

    for item in items:
        try:
            if item.get("segment") != "NSE_EQ":
                continue

            symbol = item.get("trading_symbol")
            if not symbol:
                continue

            if item.get("instrument_type") != "EQ" or item.get("security_type") != "NORMAL":
                continue

            name = item.get("name", "").upper()
            if any(x in name for x in ["BOND", "%", "SR", "TRC", "NCD", "PSU"]):
                continue

            symbol_ns = symbol + ".NS"
            if symbol_ns in allowed_symbols:
                instrument_map[symbol_ns] = item["instrument_key"]

        except Exception:
            continue

    # Save filtered map for reuse
    with open(UPSTOX_MAP_SAVE_PATH, "w") as f:
        json.dump(instrument_map, f, indent=2)

    return instrument_map

# --- API Call ---
def fetch_upstox_data(symbol: str, instrument_key: str, max_retries=3):
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
    all_chunks = []

    while start_date < end_date:
        chunk_start = start_date
        chunk_end = chunk_start + timedelta(days=28)
        if chunk_end > end_date:
            chunk_end = end_date

        chunk_start_str = chunk_start.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        url = (
            f"https://api.upstox.com/v3/historical-candle/"
            f"{instrument_key}/{INTERVAL_UNIT}/{INTERVAL}/{chunk_end_str}/{chunk_start_str}"
        )

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, headers=HEADERS)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"\u23f3 {symbol}: Rate limited. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 400:
                    return f"\u274c {symbol}: 400 Bad Request"

                resp.raise_for_status()
                candles = resp.json()["data"].get("candles", [])
                if candles:
                    all_chunks.extend(candles)

                break  # success

            except Exception as e:
                if attempt == max_retries:
                    return f"\u274c {symbol}: Failed on chunk {chunk_start_str}–{chunk_end_str}: {e}"

        start_date = chunk_end

    if not all_chunks:
        return f"\u274c {symbol}: No candle data retrieved."

    df = pd.DataFrame(all_chunks)
    df.columns = ["date", "open", "high", "low", "close", "volume", "_"]
    df = df.drop(columns=["_"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    out_dir = OUTPUT_BASE / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_{INTERVAL_UNIT}_{START_DATE}_{END_DATE}.feather"
    df.to_feather(out_path)
    return f"\u2705 {symbol}: {len(df)} rows"

# --- Runner ---
if __name__ == "__main__":
    instrument_map = load_instrument_map_ndjson(UPSTOX_JSON_PATH)
    print(f"🚀 Starting Upstox downloader for {len(instrument_map)} NSE symbols")
    results = []
    success_count = 0
    failure_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_upstox_data, sym, key): sym
            for sym, key in instrument_map.items()
        }
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if result.startswith("✅"):
                success_count += 1
            else:
                failure_count += 1

    print("\n".join(results))
    print(f"\n✅ Completed: {success_count} succeeded, {failure_count} failed.")
