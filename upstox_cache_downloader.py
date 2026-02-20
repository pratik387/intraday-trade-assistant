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
    python upstox_cache_downloader.py --update-daily     # Download last 210 days daily (convenience)
    python upstox_cache_downloader.py --update-daily 300 # Download last 300 days daily
    python upstox_cache_downloader.py --intraday-today   # Fetch today's 1m + daily via Intraday API

⚠️ Upstox Rate Limit Guidance:
- Max ~25 requests/second for historical endpoints.
- Recommended max_workers = 3 to 5
- Use exponential backoff on HTTP 429 responses (2s → 4s → 8s)
- Pause after every 20–25 requests to avoid hitting burst limits

⚠️ Data Availability:
- Historical API: T+1 delay — today's data available ~7 AM IST next day
- Intraday API: Today's candles available during/after market hours (same day)
  Use --intraday-today to fetch today's data without waiting until tomorrow.
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

# --- Index Instruments (for --intraday-today and --update-daily) ---
INDEX_INSTRUMENTS = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "NIFTY BANK": "NSE_INDEX|Nifty Bank",
    "INDIA VIX": "NSE_INDEX|India VIX",
    "NIFTY IT": "NSE_INDEX|Nifty IT",
    "NIFTY FIN SERVICE": "NSE_INDEX|Nifty Fin Service",
    "NIFTY AUTO": "NSE_INDEX|Nifty Auto",
    "NIFTY PHARMA": "NSE_INDEX|Nifty Pharma",
    "NIFTY METAL": "NSE_INDEX|Nifty Metal",
    "NIFTY ENERGY": "NSE_INDEX|Nifty Energy",
    "NIFTY PSU BANK": "NSE_INDEX|Nifty PSU Bank",
}

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
    # Check metadata BEFORE making API calls to skip already-downloaded ranges
    out_dir = OUTPUT_BASE / symbol
    metadata_path = out_dir / f"{symbol}_{timeframe['interval']}{timeframe['unit']}_metadata.json"
    if not force_redownload and metadata_path.exists():
        metadata = load_metadata(metadata_path)
        if is_range_downloaded(metadata, date_range):
            return f"skipped: {symbol} {timeframe['interval']}{timeframe['unit']}: Range {date_range['start_date']} to {date_range['end_date']} already exists"

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
                    wait = 2 ** (attempt + 1)  # 4s, 8s, 16s
                    print(f"[rate-limit] {symbol} {timeframe['interval']}{timeframe['unit']}: 429, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 400:
                    # 400 = no data for this instrument/range — not fatal
                    return f"skipped: {symbol} {timeframe['interval']}{timeframe['unit']}: 400 (no data for range)"

                resp.raise_for_status()
                candles = resp.json()["data"].get("candles", [])
                if candles:
                    all_chunks.extend(candles)

                time.sleep(0.5)  # Pace requests to avoid rate limiting
                break  # success

            except Exception as e:
                if attempt == max_retries:
                    return f"FAIL {symbol} {timeframe['interval']}{timeframe['unit']}: Failed: {e}"

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
                        wait = 2 ** (attempt + 1)  # 4s, 8s, 16s
                        print(f"[rate-limit] {symbol} {timeframe['interval']}{timeframe['unit']}: 429, waiting {wait}s...")
                        time.sleep(wait)
                        continue

                    if resp.status_code == 400:
                        # 400 on a chunk = no data for this period (e.g. stock not yet listed)
                        # Skip this chunk, try next — don't fail the entire symbol
                        break

                    resp.raise_for_status()
                    candles = resp.json()["data"].get("candles", [])
                    if candles:
                        all_chunks.extend(candles)

                    time.sleep(0.5)  # Pace requests to avoid rate limiting
                    break  # success

                except Exception as e:
                    if attempt == max_retries:
                        print(f"[warn] {symbol} {timeframe['interval']}{timeframe['unit']}: chunk {chunk_start_str}-{chunk_end_str} failed: {e}")
                        break  # Skip failed chunk, try next

            start_date = chunk_end

    if not all_chunks:
        return f"FAIL {symbol} {timeframe['interval']}{timeframe['unit']}: No candle data retrieved."

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


# --- Intraday API (today's data, no T+1 delay) ---
def fetch_intraday_today(symbol: str, instrument_key: str, unit: str = "minutes", interval: str = "1", max_retries: int = 3):
    """Fetch today's candles via the Upstox V3 Intraday API.

    Uses: GET /v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}
    No date params needed — returns current day's data.
    No auth needed — same public endpoint as historical.

    Args:
        symbol: e.g. "RELIANCE.NS" or "NIFTY 50.NS"
        instrument_key: e.g. "NSE_EQ|INE002A01018" or "NSE_INDEX|Nifty 50"
        unit: "minutes" or "days"
        interval: "1" for 1-minute or 1-day candles
    """
    url = (
        f"https://api.upstox.com/v3/historical-candle/intraday/"
        f"{instrument_key}/{unit}/{interval}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"[rate-limit] {symbol} intraday {interval}{unit}: 429, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code == 400:
                return f"skipped: {symbol} intraday {interval}{unit}: 400 (no data)"

            resp.raise_for_status()
            candles = resp.json().get("data", {}).get("candles", [])

            if not candles:
                return f"skipped: {symbol} intraday {interval}{unit}: no candles returned"

            df = pd.DataFrame(candles)
            # Response has 7 columns: [date, open, high, low, close, volume, oi]
            col_names = ["date", "open", "high", "low", "close", "volume"]
            if len(df.columns) >= 7:
                df.columns = col_names + ["_"] + list(df.columns[7:])
                df = df[col_names]
            else:
                df.columns = col_names[:len(df.columns)]

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Merge with existing feather file
            out_dir = OUTPUT_BASE / symbol
            out_dir.mkdir(parents=True, exist_ok=True)
            data_path = out_dir / f"{symbol}_{interval}{unit}.feather"

            if data_path.exists():
                try:
                    existing_df = pd.read_feather(data_path)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df = combined_df.sort_values("date").reset_index(drop=True)
                    combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
                except (OSError, Exception) as e:
                    print(f"  [warn] {symbol}: corrupted feather, replacing: {e}")
                    combined_df = df
            else:
                combined_df = df

            combined_df.to_feather(data_path)

            time.sleep(0.3)
            return f"success: {symbol} intraday {interval}{unit}: {len(df)} candles (total: {len(combined_df)})"

        except Exception as e:
            if attempt == max_retries:
                return f"FAIL {symbol} intraday {interval}{unit}: {e}"
            time.sleep(1)

    return f"FAIL {symbol} intraday {interval}{unit}: exhausted retries"


def run_intraday_today(instrument_map: dict, workers: int):
    """Fetch today's 1m + daily candles for all symbols + indices via Intraday API."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching today's intraday data ({today_str}) via Intraday API")
    print(f"Symbols: {len(instrument_map)} equities + {len(INDEX_INSTRUMENTS)} indices")
    print()

    # Build combined task list: (symbol, instrument_key, unit, interval)
    tasks = []
    for sym, key in instrument_map.items():
        tasks.append((sym, key, "minutes", "1"))  # 1m candles
        tasks.append((sym, key, "days", "1"))      # daily bar

    for idx_name, idx_key in INDEX_INSTRUMENTS.items():
        idx_sym = f"{idx_name}.NS"
        tasks.append((idx_sym, idx_key, "minutes", "1"))
        tasks.append((idx_sym, idx_key, "days", "1"))

    total = len(tasks)
    results = []
    success_count = 0
    failure_count = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for sym, key, unit, interval in tasks:
            future = executor.submit(fetch_intraday_today, sym, key, unit, interval)
            futures[future] = f"{sym}_{interval}{unit}"

        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if "success" in result:
                success_count += 1
            elif "skipped" in result:
                skipped += 1
            else:
                failure_count += 1

            if len(results) % 100 == 0:
                print(f"  Progress: {len(results)}/{total} | OK: {success_count} | Skip: {skipped} | Fail: {failure_count}", flush=True)

    print(f"\n{'='*80}")
    print(f"INTRADAY-TODAY SUMMARY ({today_str})")
    print(f"{'='*80}")
    print(f"Success: {success_count} | Skipped: {skipped} | Failed: {failure_count} / {total} total")

    if failure_count > 0:
        print("\nFailed:")
        for r in results:
            if r.startswith("FAIL"):
                print(f"  {r}")


def run_update_daily(instrument_map: dict, days: int, workers: int):
    """Download last N days of daily OHLCV for all symbols + indices."""
    end_date = datetime.now() - timedelta(days=1)  # yesterday (today not on historical API)
    start_date = end_date - timedelta(days=days)
    date_range = {
        "name": f"update_daily_{days}d",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    timeframe = {"interval": "1", "unit": "days"}

    # Combine equities + indices
    combined_map = dict(instrument_map)
    for idx_name, idx_key in INDEX_INSTRUMENTS.items():
        combined_map[f"{idx_name}.NS"] = idx_key

    total = len(combined_map)
    print(f"Updating daily data: {date_range['start_date']} to {date_range['end_date']} ({days} days)")
    print(f"Symbols: {total} ({len(instrument_map)} equities + {len(INDEX_INSTRUMENTS)} indices)")
    print()

    results = []
    success_count = 0
    failure_count = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for sym, key in combined_map.items():
            future = executor.submit(fetch_upstox_data, sym, key, date_range, timeframe)
            futures[future] = sym

        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if "success" in result:
                success_count += 1
            elif "skipped" in result:
                skipped += 1
            else:
                failure_count += 1

            if len(results) % 100 == 0:
                print(f"  Progress: {len(results)}/{total} | OK: {success_count} | Skip: {skipped} | Fail: {failure_count}", flush=True)

    print(f"\n{'='*80}")
    print(f"UPDATE-DAILY SUMMARY ({days} days)")
    print(f"{'='*80}")
    print(f"Success: {success_count} | Skipped: {skipped} | Failed: {failure_count} / {total} total")

    if failure_count > 0:
        print("\nFailed:")
        for r in results:
            if r.startswith("FAIL"):
                print(f"  {r}")


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
  python upstox_cache_downloader.py --from 2025-12-01 --to 2026-02-09 --data-type minute
  python upstox_cache_downloader.py --update-daily     # Last 210 days daily data
  python upstox_cache_downloader.py --update-daily 300 # Last 300 days daily data
  python upstox_cache_downloader.py --intraday-today   # Today's 1m + daily via Intraday API
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

    parser.add_argument(
        "--from", dest="start_date",
        help="Start date YYYY-MM-DD (overrides hardcoded DATE_RANGES)"
    )

    parser.add_argument(
        "--to", dest="end_date",
        help="End date YYYY-MM-DD (overrides hardcoded DATE_RANGES)"
    )

    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Number of concurrent download workers (default: {MAX_WORKERS})"
    )

    parser.add_argument(
        "--update-daily",
        nargs="?", const=210, type=int, metavar="DAYS",
        help="Download last N days of daily OHLCV for all symbols + indices (default: 210)"
    )

    parser.add_argument(
        "--intraday-today",
        action="store_true",
        help="Fetch today's 1m + daily candles via Intraday API (no T+1 delay)"
    )

    return parser.parse_args()

# --- Runner ---
if __name__ == "__main__":
    args = parse_arguments()

    # Load full instrument map (needed for all modes)
    full_instrument_map = load_instrument_map_ndjson(UPSTOX_JSON_PATH)

    # --- Mode: --intraday-today ---
    if args.intraday_today:
        run_intraday_today(full_instrument_map, args.workers)
        exit(0)

    # --- Mode: --update-daily ---
    if args.update_daily is not None:
        run_update_daily(full_instrument_map, args.update_daily, args.workers)
        exit(0)

    # --- Default mode: historical range download ---

    # Override DATE_RANGES if --from/--to provided
    if args.start_date and args.end_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print("ERROR: Dates must be in YYYY-MM-DD format")
            exit(1)
        DATE_RANGES = [{"name": "custom", "start_date": args.start_date, "end_date": args.end_date}]
    elif args.start_date or args.end_date:
        print("ERROR: Both --from and --to must be provided together")
        exit(1)

    # Select timeframes based on argument
    TIMEFRAMES = TIMEFRAME_OPTIONS[args.data_type]

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

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
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
