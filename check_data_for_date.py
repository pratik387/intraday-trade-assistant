import pandas as pd
import pyarrow.feather as feather
from pathlib import Path
from datetime import datetime

# Date we're investigating
target_date = "2024-01-30"

# Get all 1-minute feather files
cache_dir = Path(r"D:\trading\intraday-trade-assistant\cache\ohlcv_archive")
minute_files = list(cache_dir.glob("**/*_1minutes.feather"))

print(f"Checking data for {target_date}")
print(f"Total symbols to check: {len(minute_files)}")
print("=" * 80)

symbols_with_data = []
symbols_missing_data = []
symbols_with_insufficient_bars = []

for feather_file in minute_files[:100]:  # Check first 100 symbols
    symbol = feather_file.parent.name
    try:
        df = feather.read_feather(feather_file)

        # Convert datetime column if needed
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        else:
            print(f"Warning: No datetime column found in {symbol}")
            continue

        # Filter for target date
        date_data = df[df['datetime'].dt.date == pd.to_datetime(target_date).date()]

        if len(date_data) > 0:
            # Check opening range period (9:15 to 9:30)
            or_data = date_data[
                (date_data['datetime'].dt.time >= pd.to_datetime('09:15').time()) &
                (date_data['datetime'].dt.time <= pd.to_datetime('09:30').time())
            ]

            if len(or_data) < 3:
                symbols_with_insufficient_bars.append({
                    'symbol': symbol,
                    'total_bars': len(date_data),
                    'or_bars': len(or_data),
                    'first_time': date_data['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'last_time': date_data['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
                })
            else:
                symbols_with_data.append(symbol)
        else:
            symbols_missing_data.append(symbol)

    except Exception as e:
        print(f"Error reading {symbol}: {e}")

print(f"\nSymbols with sufficient data: {len(symbols_with_data)}")
print(f"Symbols with INSUFFICIENT opening range bars (< 3): {len(symbols_with_insufficient_bars)}")
print(f"Symbols with NO data for {target_date}: {len(symbols_missing_data)}")
print()

if symbols_with_insufficient_bars:
    print(f"\nTop 20 symbols with insufficient opening range bars:")
    print("-" * 80)
    for info in symbols_with_insufficient_bars[:20]:
        print(f"Symbol: {info['symbol']}")
        print(f"  Total bars on {target_date}: {info['total_bars']}")
        print(f"  Opening range bars (9:15-9:30): {info['or_bars']}")
        print(f"  Data range: {info['first_time']} to {info['last_time']}")
        print()

# Now let's check if 2024-01-30 was a trading day
print("\n" + "=" * 80)
print("Checking if 2024-01-30 was a trading day...")
print("=" * 80)

# Check a major index like NIFTY
nifty_file = cache_dir / "^NSEI" / "^NSEI_1minutes.feather"
if nifty_file.exists():
    df = feather.read_feather(nifty_file)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])

    date_data = df[df['datetime'].dt.date == pd.to_datetime(target_date).date()]
    print(f"NIFTY bars on {target_date}: {len(date_data)}")

    if len(date_data) > 0:
        print(f"First bar: {date_data['datetime'].min()}")
        print(f"Last bar: {date_data['datetime'].max()}")
        print(f"\nFirst 10 timestamps:")
        print(date_data['datetime'].head(10).tolist())
    else:
        print(f"{target_date} appears to be a NON-TRADING DAY or data is missing!")
