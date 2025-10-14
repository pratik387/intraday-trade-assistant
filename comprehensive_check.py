import pyarrow.feather as feather
from pathlib import Path
import pandas as pd

# Date we're investigating
target_date = "2024-01-30"

# Get all 1-minute feather files
cache_dir = Path(r"D:\trading\intraday-trade-assistant\cache\ohlcv_archive")
minute_files = list(cache_dir.glob("**/*_1minutes.feather"))

print(f"Checking data for {target_date}")
print(f"Total symbols: {len(minute_files)}")
print("=" * 80)

# Find a liquid stock to verify the date
liquid_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "ABB.NS"]

for stock in liquid_stocks:
    stock_file = cache_dir / stock / f"{stock}_1minutes.feather"
    if stock_file.exists():
        print(f"\nChecking {stock}:")
        try:
            df = feather.read_feather(stock_file)
            df['date'] = pd.to_datetime(df['date'])

            # Get data for target date
            date_data = df[df['date'].dt.date == pd.to_datetime(target_date).date()]

            if len(date_data) > 0:
                print(f"  Total bars: {len(date_data)}")
                print(f"  First bar: {date_data['date'].min()}")
                print(f"  Last bar: {date_data['date'].max()}")

                # Check opening range
                or_data = date_data[
                    (date_data['date'].dt.time >= pd.to_datetime('09:15').time()) &
                    (date_data['date'].dt.time <= pd.to_datetime('09:30').time())
                ]
                print(f"  Opening range bars (9:15-9:30): {len(or_data)}")

                # Show first 15 timestamps
                print(f"\n  First 15 timestamps:")
                for i, ts in enumerate(date_data['date'].head(15)):
                    print(f"    {i+1}. {ts}")
                break
            else:
                print(f"  NO DATA for {target_date}")

                # Check surrounding dates
                print(f"\n  Checking surrounding dates:")
                for offset in [-3, -2, -1, 0, 1, 2, 3]:
                    check_date = pd.to_datetime(target_date) + pd.Timedelta(days=offset)
                    check_data = df[df['date'].dt.date == check_date.date()]
                    print(f"    {check_date.date()}: {len(check_data)} bars")
                break
        except Exception as e:
            print(f"  Error: {e}")

print("\n" + "=" * 80)
print("Summary of issues:")
print("=" * 80)

# Count problems
insufficient_symbols = []
missing_symbols = []
good_symbols = []
errors = []

print("Checking all symbols (this may take a moment)...")

for i, feather_file in enumerate(minute_files):
    symbol = feather_file.parent.name

    try:
        df = feather.read_feather(feather_file)

        if 'date' not in df.columns:
            continue

        df['date'] = pd.to_datetime(df['date'])
        date_data = df[df['date'].dt.date == pd.to_datetime(target_date).date()]

        if len(date_data) == 0:
            missing_symbols.append(symbol)
        else:
            # Check opening range
            or_data = date_data[
                (date_data['date'].dt.time >= pd.to_datetime('09:15').time()) &
                (date_data['date'].dt.time <= pd.to_datetime('09:30').time())
            ]

            if len(or_data) < 3:
                insufficient_symbols.append({
                    'symbol': symbol,
                    'total_bars': len(date_data),
                    'or_bars': len(or_data),
                    'first_time': date_data['date'].min()
                })
            else:
                good_symbols.append(symbol)

    except Exception as e:
        errors.append(f"{symbol}: {str(e)[:50]}")

    # Progress indicator
    if (i + 1) % 200 == 0:
        print(f"  Processed {i+1}/{len(minute_files)} symbols...")

print(f"\n\nResults:")
print(f"  Symbols with GOOD data: {len(good_symbols)}")
print(f"  Symbols with INSUFFICIENT opening range bars: {len(insufficient_symbols)}")
print(f"  Symbols with NO data for {target_date}: {len(missing_symbols)}")
print(f"  Errors: {len(errors)}")

if insufficient_symbols:
    print(f"\n\nSymbols with insufficient opening range data (showing first 30):")
    print("-" * 80)
    for info in insufficient_symbols[:30]:
        print(f"{info['symbol']:20} - Total: {info['total_bars']:3} bars, OR: {info['or_bars']}, Start: {info['first_time']}")

if errors[:10]:
    print(f"\n\nFirst 10 errors:")
    for err in errors[:10]:
        print(f"  {err}")

# Check if this was a holiday
print("\n" + "=" * 80)
print("Holiday check:")
print("=" * 80)

if len(missing_symbols) > len(minute_files) * 0.5:
    print(f"MORE THAN 50% of symbols have no data for {target_date}")
    print(f"This suggests {target_date} might be a market holiday or weekend!")
else:
    print(f"Most symbols have data, but {len(insufficient_symbols)} have incomplete opening range data.")
    print(f"This suggests a data collection issue starting from around 9:30 AM or later.")
