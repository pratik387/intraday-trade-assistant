"""
Generic Upstox Historical Data Downloader

Downloads historical candles from Upstox for configurable date ranges and timeframes.
Supports both single date ranges and multiple discrete date ranges.
Downloads both minutes and daily data in separate files for backtesting.

Usage:
    python upstox_cache_downloader.py                    # Download both 1m and 1d data (default)
    python upstox_cache_downloader.py --data-type daily  # Download only daily data
    python upstox_cache_downloader.py --data-type minute # Download only minute data
    python upstox_cache_downloader.py --fix-problematic  # Redownload corrupted/incomplete symbols

⚠️ Upstox Rate Limit Guidance:
- Max ~25 requests/second for historical endpoints.
- Recommended max_workers = 3 to 5
- Use exponential backoff on HTTP 429 responses (2s → 4s → 8s)
- Pause after every 20–25 requests to avoid hitting burst limits
"""
import json
import requests
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- Configuration ---
# 6 distinct market regimes for structure evaluation
# DATE_RANGES = [
#     {"name": "Strong_Uptrend", "start_date": "2023-12-01", "end_date": "2023-12-31"},
#     {"name": "Shock_Down", "start_date": "2024-01-01", "end_date": "2024-01-31"},
#     {"name": "Event_Driven_HighVol", "start_date": "2024-06-01", "end_date": "2024-06-30"},
#     {"name": "Correction_RiskOff", "start_date": "2024-10-01", "end_date": "2024-10-31"},
#     {"name": "Prolonged_Drawdown", "start_date": "2025-02-01", "end_date": "2025-02-28"},
#     {"name": "Low_Vol_Range", "start_date": "2025-07-01", "end_date": "2025-07-31"}
# ]

DATE_RANGES = [{"name": "Strong_Uptrend", "start_date": "2023-01-01", "end_date": "2025-09-26"},
]

# Timeframes to download (must match Upstox API requirements)
# URL format: /historical-candle/{instrument_key}/{interval_unit}/{interval}/{to_date}/{from_date}
# Valid interval_units: "minutes", "days", "weeks", "months" (plural form - corrected based on API error)
# Valid intervals: 1, 3, 5, 10, 15, 30, 60 (for minutes), 1 (for days/weeks/months)

# Available timeframe options
TIMEFRAME_OPTIONS = {
    "minute": [{"interval": "1", "unit": "minutes"}],     # 1-minute candles -> 1minutes
    "daily": [{"interval": "1", "unit": "days"}],         # Daily candles -> 1days
    "both": [
        {"interval": "1", "unit": "minutes"},             # 1-minute candles -> 1minutes
        {"interval": "1", "unit": "days"}                 # Daily candles -> 1days
    ]
    # Other available options:
    # {"interval": "5", "unit": "minutes"},   # 5-minute candles -> 5minutes
    # {"interval": "15", "unit": "minutes"},  # 15-minute candles -> 15minutes
    # {"interval": "30", "unit": "minutes"},  # 30-minute candles -> 30minutes
    # {"interval": "60", "unit": "minutes"},  # 1-hour candles -> 60minutes
}

MAX_WORKERS = 8

# --- Output Path ---
ROOT = Path(__file__).resolve().parents[0]  # Changed to parents[0] to point to current directory
OUTPUT_BASE = ROOT / "cache" / "ohlcv_archive"
NSE_SYMBOLS_PATH = ROOT / "nse_all.json"
UPSTOX_JSON_PATH = ROOT / "upstox_instruments.json"
UPSTOX_MAP_SAVE_PATH = ROOT / "cache" / "upstox_instrument_map.json"  # Save in cache directory

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

        except Exception as e:
            print(f"Error processing item: {e}")
            continue

    # Save filtered map for reuse
    UPSTOX_MAP_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with open(UPSTOX_MAP_SAVE_PATH, "w") as f:
        json.dump(instrument_map, f, indent=2)

    return instrument_map

# --- Metadata Management ---
def load_metadata(metadata_path):
    """Load existing metadata or return empty structure"""
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            # Corrupted metadata file - remove and start fresh
            print(f"⚠️ Corrupted metadata file detected: {metadata_path}, resetting: {e}")
            metadata_path.unlink()  # Remove corrupted file
    return {"downloaded_ranges": [], "last_updated": None}

def save_metadata(metadata_path, metadata):
    """Save metadata to file"""
    metadata["last_updated"] = datetime.now().isoformat()
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def is_range_downloaded(metadata, date_range):
    """Check if date range already exists in metadata"""
    start_date = date_range["start_date"]
    end_date = date_range["end_date"]

    for existing_range in metadata["downloaded_ranges"]:
        if existing_range["start_date"] == start_date and existing_range["end_date"] == end_date:
            return True
    return False

# --- API Call ---
def fetch_upstox_data(symbol: str, instrument_key: str, date_range: dict, timeframe: dict, max_retries=3, force_redownload=False):
    start_date = datetime.strptime(date_range["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(date_range["end_date"], "%Y-%m-%d")
    all_chunks = []

    # For daily data, API supports up to decade in single call - no chunking needed
    if timeframe['unit'] == 'days':
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
                    print(f"\u23f3 {symbol} {timeframe['interval']}{timeframe['unit']}: Rate limited. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 400:
                    return f"\u274c {symbol} {timeframe['interval']}{timeframe['unit']}: 400 Bad Request"

                resp.raise_for_status()
                candles = resp.json()["data"].get("candles", [])
                if candles:
                    all_chunks.extend(candles)

                break  # success

            except Exception as e:
                if attempt == max_retries:
                    return f"\u274c {symbol} {timeframe['interval']}{timeframe['unit']}: Failed: {e}"

    else:
        # For minute data, use chunking to respect API limits
        while start_date < end_date:
            chunk_start = start_date
            chunk_end = chunk_start + timedelta(days=28)
            if chunk_end > end_date:
                chunk_end = end_date

            chunk_start_str = chunk_start.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            url = (
                f"https://api.upstox.com/v3/historical-candle/"
                f"{instrument_key}/{timeframe['unit']}/{timeframe['interval']}/{chunk_end_str}/{chunk_start_str}"
            )

            for attempt in range(1, max_retries + 1):
                try:
                    resp = requests.get(url, headers=HEADERS)
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        print(f"\u23f3 {symbol} {timeframe['interval']}{timeframe['unit']}: Rate limited. Retrying in {wait}s...")
                        time.sleep(wait)
                        continue

                    if resp.status_code == 400:
                        return f"\u274c {symbol} {timeframe['interval']}{timeframe['unit']}: 400 Bad Request"

                    resp.raise_for_status()
                    candles = resp.json()["data"].get("candles", [])
                    if candles:
                        all_chunks.extend(candles)

                    break  # success

                except Exception as e:
                    if attempt == max_retries:
                        return f"\u274c {symbol} {timeframe['interval']}{timeframe['unit']}: Failed on chunk {chunk_start_str}–{chunk_end_str}: {e}"

            start_date = chunk_end

    if not all_chunks:
        return f"\u274c {symbol} {timeframe['interval']}{timeframe['unit']}: No candle data retrieved."

    df = pd.DataFrame(all_chunks)
    df.columns = ["date", "open", "high", "low", "close", "volume", "_"]
    df = df.drop(columns=["_"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    out_dir = OUTPUT_BASE / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    data_path = out_dir / f"{symbol}_{timeframe['interval']}{timeframe['unit']}.feather"
    metadata_path = out_dir / f"{symbol}_{timeframe['interval']}{timeframe['unit']}_metadata.json"

    # Load existing metadata
    metadata = load_metadata(metadata_path)

    # Check if this range already downloaded (skip only if not forcing redownload)
    if not force_redownload and is_range_downloaded(metadata, date_range):
        return f"⏩ {symbol} {timeframe['interval']}{timeframe['unit']}: Range {date_range['start_date']} to {date_range['end_date']} already exists, skipped"

    # If forcing redownload, delete existing files and metadata
    if force_redownload:
        if data_path.exists():
            data_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        metadata = {"downloaded_ranges": [], "last_updated": None}
        combined_df = df
    else:
        # Load existing data if file exists
        if data_path.exists():
            try:
                existing_df = pd.read_feather(data_path)
                # Combine with new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.sort_values("date").reset_index(drop=True)
                # Remove duplicates if any
                combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
            except (OSError, Exception) as e:
                # Corrupted feather file - remove and use new data only
                print(f"⚠️ {symbol} {timeframe['interval']}{timeframe['unit']}: Corrupted file detected, replacing: {e}")
                data_path.unlink()  # Remove corrupted file
                combined_df = df
        else:
            combined_df = df

    # Save combined data
    combined_df.to_feather(data_path)

    # Update metadata
    metadata["downloaded_ranges"].append({
        "start_date": date_range["start_date"],
        "end_date": date_range["end_date"],
        "name": date_range["name"],
        "rows": len(df)
    })
    save_metadata(metadata_path, metadata)

    return f"success: {symbol} {timeframe['interval']}{timeframe['unit']}: Added {len(df)} rows (total: {len(combined_df)})"

# --- Command Line Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download historical data from Upstox API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upstox_cache_downloader.py                    # Download both 1m and 1d data (default)
  python upstox_cache_downloader.py --data-type daily  # Download only daily data for 3-year coverage
  python upstox_cache_downloader.py --data-type minute # Download only minute data for backtesting
        """
    )

    parser.add_argument(
        "--data-type",
        choices=["minute", "daily", "both"],
        default="both",
        help="Type of data to download (default: both)"
    )

    parser.add_argument(
        "--fix-problematic",
        action="store_true",
        help="Redownload only problematic symbols (from problematic_symbols.json)"
    )

    return parser.parse_args()

# --- Runner ---
if __name__ == "__main__":
    args = parse_arguments()

    # Select timeframes based on argument
    TIMEFRAMES = TIMEFRAME_OPTIONS[args.data_type]

    # Load full instrument map
    full_instrument_map = load_instrument_map_ndjson(UPSTOX_JSON_PATH)

    # Filter to problematic symbols if flag is set
    force_redownload = False
    if args.fix_problematic:
        problematic_file = Path("problematic_symbols.json")
        if not problematic_file.exists():
            print("ERROR: problematic_symbols.json not found!")
            print("Please run identify_problematic_symbols.py first")
            exit(1)

        with open(problematic_file) as f:
            problematic_data = json.load(f)

        problematic_symbols = set(problematic_data["symbols_to_redownload"])
        instrument_map = {sym: key for sym, key in full_instrument_map.items() if sym in problematic_symbols}
        force_redownload = True

        print(f"FIX MODE: Redownloading {len(instrument_map)} problematic symbols")
        print(f"  - {len(problematic_data['corrupted_files'])} corrupted files")
        print(f"  - {len(problematic_data['missing_morning_data'])} with missing morning data")
        print()
    else:
        instrument_map = full_instrument_map

    total_tasks = len(instrument_map) * len(DATE_RANGES) * len(TIMEFRAMES)

    # Format timeframes for display
    timeframe_names = [f"{tf['interval']}{tf['unit']}" for tf in TIMEFRAMES]

    print(f"Starting Upstox downloader")
    print(f"Data type: {args.data_type}")
    print(f"Symbols: {len(instrument_map)}")
    print(f"Date Ranges: {len(DATE_RANGES)}")
    print(f"Timeframes: {len(TIMEFRAMES)} ({timeframe_names})")
    print(f"Total Tasks: {total_tasks}")
    print()

    results = []
    success_count = 0
    failure_count = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}

        # Submit all combinations of symbol x date_range x timeframe
        for sym, key in instrument_map.items():
            for date_range in DATE_RANGES:
                for timeframe in TIMEFRAMES:
                    future = executor.submit(fetch_upstox_data, sym, key, date_range, timeframe, 3, force_redownload)
                    futures[future] = f"{sym}_{timeframe['interval']}{timeframe['unit']}_{date_range['name']}"

        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if "success" in result:
                success_count += 1
            elif "skipped" in result:
                skipped += 1
            else:
                failure_count += 1

            # Progress update every 50 completions
            if len(results) % 50 == 0:
                print(f"Progress: {len(results)}/{total_tasks} ({len(results)/total_tasks*100:.1f}%) | Success: {success_count} | Failed: {failure_count} | Skipped: {skipped}", flush=True)

    # Final progress update
    print(f"Final: {len(results)}/{total_tasks} (100.0%) | Success: {success_count} | Failed: {failure_count} | Skipped: {skipped}", flush=True)

    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Completed: {success_count} succeeded, {failure_count} failed out of {total_tasks} total tasks.")
    print(f"Data type downloaded: {args.data_type}")

    if args.fix_problematic:
        print(f"Fixed {success_count} problematic symbols")
    elif args.data_type == "daily":
        print(f"Daily data now supports 90+ day historic analysis for institutional features!")
    elif args.data_type == "minute":
        print(f"Minute data ready for high-resolution backtesting!")
    else:
        print(f"Complete dataset ready for both backtesting and institutional analysis!")
    if skipped > 0:
        print(f"Skipped: {skipped} ranges already existed.")
    if failure_count > 0:
        print("\nFailed downloads:")
        for result in results:
            if not result.startswith("success") and "skipped" not in result:
                print(f"   {result}")
