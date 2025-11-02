#!/usr/bin/env python
"""
Create Consolidated Daily Cache for OCI
=======================================

Aggregates all individual *_1days.feather files into a single consolidated file.
This file will be uploaded to OCI and used instead of individual daily files.

Usage:
    python tools/create_consolidated_daily_cache.py
"""

import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache" / "ohlcv_archive"
OUTPUT_DIR = ROOT / "cache" / "preaggregate"

def create_consolidated_daily_cache():
    """Aggregate all daily feather files into one"""
    start_time = time.time()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    output_file = OUTPUT_DIR / "consolidated_daily.feather"

    print("=" * 70)
    print("Creating Consolidated Daily Cache")
    print("=" * 70)
    print(f"Source: {CACHE_DIR}")
    print(f"Output: {output_file}")
    print()

    all_daily_data = []
    processed = 0
    errors = 0
    skipped = 0

    symbol_dirs = sorted([d for d in CACHE_DIR.iterdir() if d.is_dir()])
    total = len(symbol_dirs)

    print(f"Processing {total} symbols...\n")

    for i, symbol_dir in enumerate(symbol_dirs, 1):
        symbol_ns = symbol_dir.name  # e.g., "AAPL.NS"
        symbol = symbol_ns.replace(".NS", "")  # e.g., "AAPL"

        if i % 200 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"  [{i:4d}/{total}] Processed: {processed:4d} | Errors: {errors:3d} | ETA: {remaining/60:.1f}m")

        try:
            # Look for daily feather file
            daily_file = symbol_dir / f"{symbol_ns}_1days.feather"

            if not daily_file.exists():
                skipped += 1
                continue

            # Read daily data
            df = pd.read_feather(daily_file)

            # Normalize timestamp column
            if 'timestamp' in df.columns:
                ts_col = 'timestamp'
            elif 'ts' in df.columns:
                ts_col = 'ts'
            elif 'date' in df.columns:
                ts_col = 'date'
            else:
                df = df.reset_index()
                ts_col = df.columns[0]

            # Rename to standard 'ts' and strip timezone
            df['ts'] = pd.to_datetime(df[ts_col])
            if df['ts'].dt.tz is not None:
                df['ts'] = df['ts'].dt.tz_localize(None)

            # Add symbol column
            df['symbol'] = symbol

            # Select required columns
            required_cols = ['ts', 'open', 'high', 'low', 'close', 'volume', 'symbol']
            df = df[required_cols]

            all_daily_data.append(df)
            processed += 1

        except Exception as e:
            errors += 1
            if errors <= 20:  # Print first 20 errors
                print(f"  ERROR {symbol}: {str(e)[:80]}")
            continue

    # Save consolidated file
    print(f"\n{'='*70}")
    print("SAVING CONSOLIDATED FILE...")
    print(f"{'='*70}\n")

    if all_daily_data:
        print(f"  Combining {len(all_daily_data)} symbols...")
        combined = pd.concat(all_daily_data, ignore_index=True)
        combined = combined.sort_values(['symbol', 'ts'])

        print(f"  Total rows: {len(combined):,}")
        print(f"  Date range: {combined['ts'].min()} to {combined['ts'].max()}")
        print(f"  Unique symbols: {combined['symbol'].nunique()}")
        print()

        combined.to_feather(output_file)
        size_mb = output_file.stat().st_size / (1024 * 1024)

        print(f"  [OK] {output_file.name}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Rows: {len(combined):,}")
    else:
        print("  [ERROR] No daily data found!")
        return

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total symbols checked:  {total}")
    print(f"  Successfully processed: {processed}")
    print(f"  Skipped (no file):      {skipped}")
    print(f"  Errors (corrupted):     {errors}")
    print(f"  Time taken:             {elapsed/60:.1f} minutes")
    print(f"\n[SUCCESS] Consolidated daily cache created!")
    print(f"\nNext steps:")
    print(f"  1. Upload to OCI: oci os object put --bucket backtest-cache \\")
    print(f"       --name consolidated_daily.feather --file {output_file}")
    print(f"  2. Update MockBroker to use consolidated cache in OCI")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    create_consolidated_daily_cache()
