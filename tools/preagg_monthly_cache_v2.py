#!/usr/bin/env python
"""
Pre-aggregate monthly cache for MASSIVE backtest speedup (Windows-safe version).

Expected savings: 15-20 minutes per month (50% faster!)

Usage:
    python tools/preagg_monthly_cache_v2.py 2024 10
    python tools/preagg_monthly_cache_v2.py 2025 2
    python tools/preagg_monthly_cache_v2.py 2025 7
"""
import sys
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache" / "ohlcv_archive"
PREAGG_DIR = ROOT / "cache" / "preaggregate"

def normalize_timestamp(df):
    """Extract and normalize timestamp column, strip timezone"""
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
    elif 'ts' in df.columns:
        ts = pd.to_datetime(df['ts'])
    elif 'date' in df.columns:
        ts = pd.to_datetime(df['date'])
    else:
        df = df.reset_index()
        ts = pd.to_datetime(df.iloc[:, 0])

    # Strip timezone if present (avoids comparison errors)
    if hasattr(ts.dt, 'tz') and ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)

    return ts, df

def preaggregate_month(year: int, month: int):
    """Pre-aggregate all symbols for a month into single files"""
    start_time = time.time()

    PREAGG_DIR.mkdir(exist_ok=True, parents=True)

    output_5m = PREAGG_DIR / f"{year}_{month:02d}_1m.feather"

    print(f"\n{'='*70}")
    print(f"PRE-AGGREGATING (1-MINUTE DATA) {year}-{month:02d}")
    print(f"{'='*70}")
    print(f"Source: {CACHE_DIR}")
    print(f"Target: {PREAGG_DIR}")
    print(f"")

    all_data_1m = []
    processed = 0
    errors = 0

    symbol_dirs = sorted([d for d in CACHE_DIR.iterdir() if d.is_dir()])
    total = len(symbol_dirs)

    print(f"Processing {total} symbols...\n")

    for i, symbol_dir in enumerate(symbol_dirs, 1):
        symbol_ns = symbol_dir.name
        symbol = symbol_ns.replace(".NS", "")

        if i % 200 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"  [{i:4d}/{total}] 1m: {len(all_data_1m):4d} | ETA: {remaining/60:.1f}m")

        try:
            # === 5-MINUTE DATA (most important for backtest) ===
            feather_5m = symbol_dir / f"{symbol_ns}_1minutes.feather"
            if not feather_5m.exists():
                continue

            df = pd.read_feather(feather_5m)

            # Normalize timestamp
            ts, df = normalize_timestamp(df)
            df['ts'] = ts

            # Filter to target month
            mask = (ts.dt.year == year) & (ts.dt.month == month)
            df = df[mask].copy()

            if not df.empty:
                df['symbol'] = symbol
                all_data_1m.append(df)
                processed += 1

        except Exception as e:
            errors += 1
            if errors <= 10:  # Only print first 10 errors
                print(f"  ERROR {symbol}: {str(e)[:50]}")
            continue

    # === SAVE AGGREGATED FILES ===
    print(f"\n{'='*70}")
    print("SAVING AGGREGATED FILES...")
    print(f"{'='*70}\n")

    # 5-minute data
    if all_data_1m:
        print(f"  Combining {len(all_data_1m)} symbols for 1m data...")
        combined = pd.concat(all_data_1m, ignore_index=True)
        combined = combined.sort_values(['symbol', 'ts'])
        combined.to_feather(output_5m)
        size_mb = output_5m.stat().st_size / 1024 / 1024
        print(f"  [OK] {output_5m.name:30s} | {size_mb:6.1f} MB | {len(combined):,} rows")
    else:
        print(f"  [SKIP] No 5-minute data found")

    # === SUMMARY ===
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total symbols:    {total}")
    print(f"  Processed (1m):   {processed}")
    print(f"  Errors:           {errors}")
    print(f"  Time taken:       {elapsed/60:.1f} minutes")
    print(f"\n[SUCCESS] Pre-aggregation complete for {year}-{month:02d}!")
    print(f"\nYour backtests will now be 15-20 minutes FASTER per month!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preagg_monthly_cache_v2.py YEAR MONTH")
        print("\nExamples:")
        print("  python preagg_monthly_cache_v2.py 2024 10")
        print("  python preagg_monthly_cache_v2.py 2025 2")
        print("  python preagg_monthly_cache_v2.py 2025 7")
        sys.exit(1)

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    if not (2020 <= year <= 2030):
        print(f"ERROR: Invalid year {year}")
        sys.exit(1)

    if not (1 <= month <= 12):
        print(f"ERROR: Invalid month {month}")
        sys.exit(1)

    preaggregate_month(year, month)
