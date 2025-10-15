import pyarrow.feather as feather
from pathlib import Path
import pandas as pd

# Check a sample feather file
cache_dir = Path(r"D:\trading\intraday-trade-assistant\cache\ohlcv_archive")
sample_file = cache_dir / "ABB.NS" / "ABB.NS_1minutes.feather"

print(f"Inspecting: {sample_file}")
print("=" * 80)

df = feather.read_feather(sample_file)

print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 10 rows:")
print(df.head(10))
print(f"\nLast 10 rows:")
print(df.tail(10))

# Check if date column exists
print("\n" + "=" * 80)
print("Checking for date-related columns...")
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        print(f"\nFound: {col}")
        print(f"Sample values:")
        print(df[col].head(10))
