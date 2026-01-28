"""
Download Oct 30-31, 2025 historical data for paper trading verification.
Adapted from upstox_cache_downloader.py for targeted symbol/date download.

Usage:
    python tools/download_verification_data.py
"""
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration
VERIFICATION_SYMBOLS_FILE = Path("verification_symbols_oct31.json")
DATE_RANGES = [
    {"name": "Oct30_PrevDay", "start_date": "2025-10-30", "end_date": "2025-10-30"},  # For PDH/PDL/PDC
    {"name": "Oct31_TradingDay", "start_date": "2025-10-31", "end_date": "2025-10-31"}  # For trade data
]

TIMEFRAMES = [
    {"interval": "1", "unit": "minutes"},  # 1-minute data
    {"interval": "1", "unit": "days"}       # Daily data
]

MAX_WORKERS = 4
OUTPUT_BASE = Path("cache/verification/oct31")
UPSTOX_JSON_PATH = Path("upstox_instruments.json")
UPSTOX_MAP_SAVE_PATH = Path("cache/upstox_instrument_map.json")

HEADERS = {"Accept": "application/json"}

# Load instrument map
def load_instrument_map():
    """Load cached Upstox instrument map"""
    if not UPSTOX_MAP_SAVE_PATH.exists():
        print(f"ERROR: {UPSTOX_MAP_SAVE_PATH} not found!")
        print("Please run upstox_cache_downloader.py first to generate the instrument map")
        exit(1)
    
    with open(UPSTOX_MAP_SAVE_PATH) as f:
        full_map = json.load(f)
    
    # Load verification symbols
    if not VERIFICATION_SYMBOLS_FILE.exists():
        print(f"ERROR: {VERIFICATION_SYMBOLS_FILE} not found!")
        exit(1)
    
    with open(VERIFICATION_SYMBOLS_FILE) as f:
        verification_symbols = set(json.load(f))
    
    # Filter map to only verification symbols
    filtered_map = {sym: key for sym, key in full_map.items() if sym in verification_symbols}
    
    missing = verification_symbols - set(filtered_map.keys())
    if missing:
        print(f"WARNING: {len(missing)} symbols not found in instrument map: {missing}")
    
    return filtered_map

# Fetch data from Upstox
def fetch_upstox_data(symbol, instrument_key, date_range, timeframe, max_retries=3):
    """Fetch historical candles from Upstox API"""
    start_date = datetime.strptime(date_range["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(date_range["end_date"], "%Y-%m-%d")
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    url = (
        f"https://api.upstox.com/v3/historical-candle/"
        f"{instrument_key}/{timeframe['unit']}/{timeframe['interval']}/{end_date_str}/{start_date_str}"
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"⏳ {symbol} {timeframe['interval']}{timeframe['unit']}: Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            
            if resp.status_code == 400:
                return f"❌ {symbol} {timeframe['interval']}{timeframe['unit']}: 400 Bad Request"
            
            resp.raise_for_status()
            candles = resp.json()["data"].get("candles", [])
            
            if not candles:
                return f"❌ {symbol} {timeframe['interval']}{timeframe['unit']}: No data"
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df.columns = ["date", "open", "high", "low", "close", "volume", "_"]
            df = df.drop(columns=["_"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            
            # Save to feather
            out_dir = OUTPUT_BASE / symbol.replace(".NS", "")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            data_path = out_dir / f"{symbol.replace('.NS', '')}_{timeframe['interval']}{timeframe['unit']}_{date_range['name']}.feather"
            df.to_feather(data_path)
            
            return f"✓ {symbol} {timeframe['interval']}{timeframe['unit']} {date_range['name']}: {len(df)} rows"
        
        except Exception as e:
            if attempt == max_retries:
                return f"❌ {symbol} {timeframe['interval']}{timeframe['unit']}: {e}"
            time.sleep(1)
    
    return f"❌ {symbol} {timeframe['interval']}{timeframe['unit']}: Failed after retries"

# Main
if __name__ == "__main__":
    print("Paper Trading Verification Data Downloader")
    print("=" * 60)
    
    instrument_map = load_instrument_map()
    
    total_tasks = len(instrument_map) * len(DATE_RANGES) * len(TIMEFRAMES)
    
    print(f"Symbols: {len(instrument_map)}")
    print(f"Date Ranges: {len(DATE_RANGES)} (Oct 30, Oct 31)")
    print(f"Timeframes: {len(TIMEFRAMES)} (1minute, 1day)")
    print(f"Total Tasks: {total_tasks}")
    print(f"Output: {OUTPUT_BASE}")
    print()
    
    results = []
    success_count = 0
    failure_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        
        for sym, key in instrument_map.items():
            for date_range in DATE_RANGES:
                for timeframe in TIMEFRAMES:
                    future = executor.submit(fetch_upstox_data, sym, key, date_range, timeframe)
                    futures[future] = f"{sym}_{timeframe['interval']}{timeframe['unit']}_{date_range['name']}"
        
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            
            if result.startswith("✓"):
                success_count += 1
            else:
                failure_count += 1
            
            print(result)
            
            # Progress update
            if len(results) % 10 == 0:
                print(f"Progress: {len(results)}/{total_tasks} ({len(results)/total_tasks*100:.1f}%)")
    
    print()
    print("=" * 60)
    print(f"Download Complete: {success_count} succeeded, {failure_count} failed")
    print(f"Data saved to: {OUTPUT_BASE}")
