"""
Identify symbols with corrupted files or consistently missing morning data
"""
import pyarrow.feather as feather
from pathlib import Path
import pandas as pd
import json
from datetime import datetime, timedelta

# Configuration
cache_dir = Path(r"D:\trading\intraday-trade-assistant\cache\ohlcv_archive")
minute_files = list(cache_dir.glob("**/*_1minutes.feather"))

# Test dates - check multiple dates to see if missing morning data is consistent
test_dates = [
    "2024-01-30",
    "2024-01-29",
    "2024-01-31",
    "2024-02-01",
    "2024-02-02"
]

print(f"Analyzing {len(minute_files)} symbols for data quality issues...")
print(f"Testing dates: {', '.join(test_dates)}")
print("=" * 80)

corrupted_files = []
missing_morning_data = {}  # symbol -> list of dates with missing morning data
symbols_to_redownload = []

for i, feather_file in enumerate(minute_files):
    symbol = feather_file.parent.name

    try:
        df = feather.read_feather(feather_file)

        if 'date' not in df.columns:
            corrupted_files.append({
                'symbol': symbol,
                'error': 'Missing date column',
                'file': str(feather_file)
            })
            continue

        df['date'] = pd.to_datetime(df['date'])

        # Check each test date for missing morning data
        dates_with_issues = []
        for test_date in test_dates:
            date_data = df[df['date'].dt.date == pd.to_datetime(test_date).date()]

            if len(date_data) > 0:
                # Check if data starts after 9:15 AM
                first_time = date_data['date'].min().time()
                market_open = pd.to_datetime('09:15').time()

                if first_time > market_open:
                    dates_with_issues.append(test_date)

        # If missing morning data on multiple dates, flag for redownload
        if len(dates_with_issues) >= 2:
            missing_morning_data[symbol] = {
                'dates_affected': dates_with_issues,
                'file': str(feather_file)
            }
            symbols_to_redownload.append(symbol)

    except Exception as e:
        corrupted_files.append({
            'symbol': symbol,
            'error': str(e),
            'file': str(feather_file)
        })

    # Progress indicator
    if (i + 1) % 200 == 0:
        print(f"  Processed {i+1}/{len(minute_files)} symbols...")

print(f"\n{'=' * 80}")
print("ANALYSIS RESULTS")
print("=" * 80)

print(f"\nCorrupted files: {len(corrupted_files)}")
if corrupted_files:
    print("\nCorrupted symbols (showing first 30):")
    for item in corrupted_files[:30]:
        print(f"  {item['symbol']:20} - {item['error'][:60]}")

print(f"\nSymbols with consistently missing morning data: {len(missing_morning_data)}")
if missing_morning_data:
    print("\nSymbols missing morning data on 2+ dates (showing first 30):")
    for symbol, info in list(missing_morning_data.items())[:30]:
        print(f"  {symbol:20} - Missing on {len(info['dates_affected'])} dates: {', '.join(info['dates_affected'][:3])}")

# Combine all symbols that need redownloading
all_problematic_symbols = list(set(
    [item['symbol'] for item in corrupted_files] +
    symbols_to_redownload
))

print(f"\nTotal unique symbols to redownload: {len(all_problematic_symbols)}")

# Save to JSON for use in downloader
output_file = Path("problematic_symbols.json")
output_data = {
    "corrupted_files": corrupted_files,
    "missing_morning_data": missing_morning_data,
    "symbols_to_redownload": sorted(all_problematic_symbols),
    "generated_at": datetime.now().isoformat()
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nAnalysis saved to: {output_file}")
print(f"\nNext steps:")
print(f"  1. Run updated upstox_cache_downloader.py with --fix-problematic flag")
print(f"  2. This will redownload data for {len(all_problematic_symbols)} symbols")
