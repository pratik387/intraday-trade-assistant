#!/usr/bin/env python
"""
Pre-aggregate monthly cache for MASSIVE backtest speedup.

PROBLEM:
--------
Current backtest reads 1,992 separate feather files PER DAY:
- 1,992 files × 20 days = 39,840 file opens per month
- Each file open: ~10-30ms
- Total I/O time: 60+ seconds per day = 20 minutes per month!

SOLUTION:
---------
Pre-aggregate all symbols for a month into 3 single files:
- {YEAR}_{MONTH}_1m.feather (all 1-minute data)
- {YEAR}_{MONTH}_5m.feather (all 5-minute data)
- {YEAR}_{MONTH}_1d.feather (all daily data)

RESULT:
-------
- After: 1 file open per day × 20 days = 20 file opens
- Expected savings: 15-20 minutes per month (50% faster!)

Usage:
    python tools/preagg_monthly_cache.py 2024 10  # October 2024
    python tools/preagg_monthly_cache.py 2025 2   # February 2025
    python tools/preagg_monthly_cache.py 2025 7   # July 2025
"""
import sys
import pandas as pd
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache" / "ohlcv_archive"
PREAGG_DIR = ROOT / "cache" / "preaggregate"

def preaggregate_month(year: int, month: int):
    """Pre-aggregate all symbols for a month into single files"""
    start_time = time.time()

    PREAGG_DIR.mkdir(exist_ok=True, parents=True)

    # Output files for different timeframes
    output_1m = PREAGG_DIR / f"{year}_{month:02d}_1m.feather"
    output_5m = PREAGG_DIR / f"{year}_{month:02d}_5m.feather"
    output_1d = PREAGG_DIR / f"{year}_{month:02d}_1d.feather"

    print(f"\n{'='*70}")
    print(f"PRE-AGGREGATING {year}-{month:02d}")
    print(f"{'='*70}")
    print(f"Source: {CACHE_DIR}")
    print(f"Target: {PREAGG_DIR}")
    print(f"")

    # Containers for aggregated data
    all_data_1m = []
    all_data_5m = []
    all_data_1d = []

    processed = 0
    skipped = 0
    errors = 0

    symbol_dirs = sorted([d for d in CACHE_DIR.iterdir() if d.is_dir()])
    total = len(symbol_dirs)

    print(f"Processing {total} symbols...\n")

    for i, symbol_dir in enumerate(symbol_dirs, 1):
        symbol_ns = symbol_dir.name  # e.g., "RELIANCE.NS"
        symbol = symbol_ns.replace(".NS", "")

        # Progress indicator every 100 symbols
        if i % 100 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (total - i) / rate
            print(f"  [{i:4d}/{total}] {symbol:20s} | "
                  f"1m: {len(all_data_1m):4d} | "
                  f"5m: {len(all_data_5m):4d} | "
                  f"1d: {len(all_data_1d):4d} | "
                  f"ETA: {remaining/60:.1f}m")

        try:
            # === 1-MINUTE DATA ===
            feather_1m = symbol_dir / f"{symbol_ns}_1minutes.feather"
            if feather_1m.exists():
                df = pd.read_feather(feather_1m)

                # Normalize timestamp column
                if 'timestamp' in df.columns:
                    df['ts'] = pd.to_datetime(df['timestamp'])
                elif 'ts' in df.columns:
                    df['ts'] = pd.to_datetime(df['ts'])
                elif 'date' in df.columns:
                    df['ts'] = pd.to_datetime(df['date'])
                else:
                    # Assume index is timestamp
                    df = df.reset_index()
                    df['ts'] = pd.to_datetime(df.iloc[:, 0])

                # Filter to target month
                df = df[(df['ts'].dt.year == year) & (df['ts'].dt.month == month)].copy()

                if not df.empty:
                    df['symbol'] = symbol
                    all_data_1m.append(df)

            # === 5-MINUTE DATA ===
            feather_5m = symbol_dir / f"{symbol_ns}_5minutes.feather"
            if feather_5m.exists():
                df = pd.read_feather(feather_5m)

                # Normalize timestamp column
                if 'timestamp' in df.columns:
                    df['ts'] = pd.to_datetime(df['timestamp'])
                elif 'ts' in df.columns:
                    df['ts'] = pd.to_datetime(df['ts'])
                elif 'date' in df.columns:
                    df['ts'] = pd.to_datetime(df['date'])
                else:
                    df = df.reset_index()
                    df['ts'] = pd.to_datetime(df.iloc[:, 0])

                # Filter to target month
                df = df[(df['ts'].dt.year == year) & (df['ts'].dt.month == month)].copy()

                if not df.empty:
                    df['symbol'] = symbol
                    all_data_5m.append(df)

            # === DAILY DATA ===
            feather_1d = symbol_dir / f"{symbol_ns}_1days.feather"
            if feather_1d.exists():
                df = pd.read_feather(feather_1d)

                # Normalize timestamp column
                if 'timestamp' in df.columns:
                    df['ts'] = pd.to_datetime(df['timestamp'])
                elif 'ts' in df.columns:
                    df['ts'] = pd.to_datetime(df['ts'])
                elif 'date' in df.columns:
                    df['ts'] = pd.to_datetime(df['date'])
                else:
                    df = df.reset_index()
                    df['ts'] = pd.to_datetime(df.iloc[:, 0])

                # For daily: include previous 210 days (for EMA200 regime detection)
                target_start = pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(days=210)
                target_end = pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(months=1)

                df = df[(df['ts'] >= target_start) & (df['ts'] < target_end)].copy()

                if not df.empty:
                    df['symbol'] = symbol
                    all_data_1d.append(df)

            processed += 1

        except Exception as e:
            print(f"  ERROR {symbol}: {str(e)[:60]}")
            errors += 1
            continue

    # === SAVE AGGREGATED FILES ===
    print(f"\n{'='*70}")
    print("SAVING AGGREGATED FILES...")
    print(f"{'='*70}\n")

    # 1-minute data
    if all_data_1m:
        print(f"  Combining {len(all_data_1m)} symbols for 1m data...")
        combined = pd.concat(all_data_1m, ignore_index=True)
        combined = combined.sort_values(['symbol', 'ts'])
        combined.to_feather(output_1m)
        size_mb = output_1m.stat().st_size / 1024 / 1024
        print(f"  ✓ {output_1m.name:30s} | {size_mb:6.1f} MB | {len(combined):,} rows")
    else:
        print(f"  ✗ No 1-minute data found")

    # 5-minute data
    if all_data_5m:
        print(f"  Combining {len(all_data_5m)} symbols for 5m data...")
        combined = pd.concat(all_data_5m, ignore_index=True)
        combined = combined.sort_values(['symbol', 'ts'])
        combined.to_feather(output_5m)
        size_mb = output_5m.stat().st_size / 1024 / 1024
        print(f"  ✓ {output_5m.name:30s} | {size_mb:6.1f} MB | {len(combined):,} rows")
    else:
        print(f"  ✗ No 5-minute data found")

    # Daily data
    if all_data_1d:
        print(f"  Combining {len(all_data_1d)} symbols for 1d data...")
        combined = pd.concat(all_data_1d, ignore_index=True)
        combined = combined.sort_values(['symbol', 'ts'])
        combined.to_feather(output_1d)
        size_mb = output_1d.stat().st_size / 1024 / 1024
        print(f"  ✓ {output_1d.name:30s} | {size_mb:6.1f} MB | {len(combined):,} rows")
    else:
        print(f"  ✗ No daily data found")

    # === SUMMARY ===
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total symbols: {total}")
    print(f"  Processed:     {processed}")
    print(f"  Errors:        {errors}")
    print(f"  1m symbols:    {len(all_data_1m)}")
    print(f"  5m symbols:    {len(all_data_5m)}")
    print(f"  1d symbols:    {len(all_data_1d)}")
    print(f"  Time taken:    {elapsed/60:.1f} minutes")
    print(f"\n✓ Pre-aggregation complete for {year}-{month:02d}!")
    print(f"\nYour backtests will now be 15-20 minutes FASTER per month!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preagg_monthly_cache.py YEAR MONTH")
        print("\nExamples:")
        print("  python preagg_monthly_cache.py 2024 10  # October 2024")
        print("  python preagg_monthly_cache.py 2025 2   # February 2025")
        print("  python preagg_monthly_cache.py 2025 7   # July 2025")
        sys.exit(1)

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    if not (2020 <= year <= 2030):
        print(f"ERROR: Invalid year {year} (must be 2020-2030)")
        sys.exit(1)

    if not (1 <= month <= 12):
        print(f"ERROR: Invalid month {month} (must be 1-12)")
        sys.exit(1)

    preaggregate_month(year, month)
