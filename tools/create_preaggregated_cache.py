#!/usr/bin/env python3
"""
Create Pre-Aggregated Monthly Cache Files + Update Consolidated Daily
=====================================================================

Reads individual symbol feather files from cache/ohlcv_archive/ and combines
them into monthly pre-aggregated files for fast backtesting (50x speedup).

Also updates the consolidated daily feather used for PDH/PDL/PDC calculations.

Usage:
    python tools/create_preaggregated_cache.py --from 2025-12-01 --to 2026-02-09
    python tools/create_preaggregated_cache.py --from 2025-12-01 --to 2026-02-09 --skip-daily
    python tools/create_preaggregated_cache.py --from 2025-12-01 --to 2026-02-09 --output-dir backtest-cache-download/monthly

Output:
    backtest-cache-download/monthly/YYYY_MM_1m.feather  (one per month)
    cache/preaggregate/consolidated_daily.feather       (merged with existing)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import pytz

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "cache" / "ohlcv_archive"
DEFAULT_MONTHLY_DIR = ROOT / "backtest-cache-download" / "monthly"
# Canonical consolidated daily path — same one mock_broker.get_daily reads
# and oci/docker/entrypoint.py downloads into. Keep these in sync; a path
# divergence here silently broke circuit_t1_fade_short for 8 days because
# the broker read a stale copy at cache/preaggregate while this script
# wrote to backtest-cache-download/.
DEFAULT_DAILY_PATH = ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"

IST = pytz.FixedOffset(330)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create pre-aggregated monthly cache from individual symbol feather files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/create_preaggregated_cache.py --from 2025-12-01 --to 2026-02-09
  python tools/create_preaggregated_cache.py --from 2025-12-01 --to 2026-02-09 --skip-daily
        """
    )
    parser.add_argument("--from", dest="start_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="end_date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", default=str(DEFAULT_MONTHLY_DIR),
                        help=f"Output directory for monthly files (default: {DEFAULT_MONTHLY_DIR})")
    parser.add_argument("--skip-daily", action="store_true",
                        help="Skip updating consolidated_daily.feather")
    parser.add_argument("--source-dir", default=None,
                        help="Source archive directory (default: cache/ohlcv_archive)")
    return parser.parse_args()


def discover_symbol_files(suffix="_1minutes", source_dir=None):
    """
    Find all symbol directories and their feather files.

    Returns dict: bare_symbol -> Path to feather file
    e.g. {"RELIANCE": Path("cache/ohlcv_archive/RELIANCE.NS/RELIANCE.NS_1minutes.feather")}
    """
    search_dir = Path(source_dir) if source_dir else ARCHIVE_DIR
    files = {}
    if not search_dir.exists():
        return files

    for sym_dir in sorted(search_dir.iterdir()):
        if not sym_dir.is_dir():
            continue

        dir_name = sym_dir.name  # e.g. "RELIANCE.NS" or "NIFTY 50.NS"

        # Find feather files matching the suffix
        candidates = list(sym_dir.glob(f"*{suffix}.feather"))
        if not candidates:
            continue

        # Pick newest by mtime
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        feather_path = candidates[0]

        # Bare symbol name (strip .NS suffix from dir name)
        bare = dir_name.replace(".NS", "").replace(".BSE", "")
        files[bare] = feather_path

    return files


def load_and_filter(feather_path, from_dt_naive, to_dt_naive):
    """Load feather file, filter to date range, return DataFrame with naive timestamps."""
    df = pd.read_feather(feather_path)

    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])

    # Create naive version for filtering
    if df["date"].dt.tz is not None:
        date_naive = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        date_naive = df["date"]

    mask = (date_naive >= from_dt_naive) & (date_naive <= to_dt_naive)
    df = df[mask].copy()

    if df.empty:
        return None

    return df


def create_monthly_files(symbol_files, from_dt, to_dt, output_dir):
    """Create monthly pre-aggregated 1m feather files.

    Processes ONE MONTH at a time to avoid OOM with large symbol universes.
    For each month, loads only that month's data from each symbol file.
    """
    print(f"\n{'='*60}")
    print("Creating monthly 1-minute pre-aggregated files")
    print(f"{'='*60}")
    print(f"Source: {ARCHIVE_DIR}")
    print(f"Output: {output_dir}")
    print(f"Range:  {from_dt.date()} to {to_dt.date()}")
    print(f"Symbols found: {len(symbol_files)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate list of (year, month) tuples to process
    months_to_process = []
    current = from_dt.replace(day=1)
    while current <= to_dt:
        months_to_process.append((current.year, current.month))
        # Advance to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    print(f"  Months to process: {len(months_to_process)}")

    for year, month in months_to_process:
        # Define month boundaries
        month_start = pd.Timestamp(year=year, month=month, day=1)
        if month == 12:
            month_end = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(seconds=1)
        else:
            month_end = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(seconds=1)

        # Clip to overall range
        eff_start = max(month_start, from_dt)
        eff_end = min(month_end, to_dt)

        month_frames = []
        loaded = 0

        for symbol, path in symbol_files.items():
            df = load_and_filter(path, eff_start, eff_end)
            if df is None:
                continue

            # Ensure tz-aware date column (IST)
            if df["date"].dt.tz is None:
                df["date"] = df["date"].dt.tz_localize(IST)
            else:
                df["date"] = df["date"].dt.tz_convert(IST)

            # Create naive ts column
            df["ts"] = df["date"].dt.tz_localize(None)
            df["symbol"] = symbol
            df = df[["date", "open", "high", "low", "close", "volume", "ts", "symbol"]]
            month_frames.append(df)
            loaded += 1

        if not month_frames:
            print(f"  {year}_{month:02d}: no data, skipping")
            continue

        combined = pd.concat(month_frames, ignore_index=True)
        del month_frames  # Free memory

        out_df = (combined
                  .sort_values(["symbol", "ts"])
                  .reset_index(drop=True))
        del combined

        out_df["volume"] = out_df["volume"].astype("int64")

        filename = f"{year}_{month:02d}_1m.feather"
        output_path = output_dir / filename
        out_df.to_feather(output_path)

        n_symbols = out_df["symbol"].nunique()
        n_rows = len(out_df)
        n_days = out_df["ts"].dt.date.nunique()
        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"  Written: {filename} ({n_rows:,} rows, {n_symbols} symbols, {n_days} trading days, {size_mb:.1f} MB)")
        del out_df  # Free memory

    print(f"Monthly files created successfully!")


def update_consolidated_daily(symbol_files_daily, from_dt, to_dt):
    """Update consolidated daily feather with new daily data."""
    print(f"\n{'='*60}")
    print("Updating consolidated daily cache")
    print(f"{'='*60}")
    print(f"Output: {DEFAULT_DAILY_PATH}")

    # Collect new daily data
    all_frames = []
    loaded = 0

    for i, (symbol, path) in enumerate(symbol_files_daily.items()):
        df = load_and_filter(path, from_dt, to_dt)
        if df is None:
            continue

        # Create naive ts column
        if df["date"].dt.tz is not None:
            df["ts"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        else:
            df["ts"] = df["date"]

        df["symbol"] = symbol

        # Match consolidated daily schema: ts, open, high, low, close, volume, symbol
        df = df[["ts", "open", "high", "low", "close", "volume", "symbol"]]
        all_frames.append(df)
        loaded += 1

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(symbol_files_daily)} symbols ({loaded} with data)...")

    print(f"  Processed {len(symbol_files_daily)}/{len(symbol_files_daily)} symbols ({loaded} with data)")

    if not all_frames:
        print("WARNING: No daily data found in requested range")
        return

    new_daily = pd.concat(all_frames, ignore_index=True)
    print(f"  New daily data: {len(new_daily):,} rows, {new_daily['symbol'].nunique()} symbols")

    # Merge with existing consolidated daily
    if DEFAULT_DAILY_PATH.exists():
        print(f"  Loading existing consolidated daily...")
        existing = pd.read_feather(DEFAULT_DAILY_PATH)
        print(f"  Existing: {len(existing):,} rows ({existing['ts'].min().date()} to {existing['ts'].max().date()})")

        combined = pd.concat([existing, new_daily], ignore_index=True)
        # Deduplicate on (ts, symbol), keeping latest
        combined = combined.drop_duplicates(subset=["ts", "symbol"], keep="last")
    else:
        combined = new_daily

    combined = combined.sort_values(["symbol", "ts"]).reset_index(drop=True)
    combined["volume"] = combined["volume"].astype("int64")

    DEFAULT_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_feather(DEFAULT_DAILY_PATH)

    size_mb = DEFAULT_DAILY_PATH.stat().st_size / (1024 ** 2)
    print(f"  Written: consolidated_daily.feather ({len(combined):,} rows, {combined['symbol'].nunique()} symbols, {size_mb:.1f} MB)")
    print(f"  Range: {combined['ts'].min().date()} to {combined['ts'].max().date()}")


def create_monthly_5m_enriched(symbol_files, from_dt, to_dt, output_dir):
    """Create monthly pre-aggregated 5m enriched feather files.

    Same pattern as 1m monthly files but for precomputed enriched 5m bars.
    Columns: symbol, date, open, high, low, close, volume, vwap, bb_width_proxy, adx, rsi
    """
    print(f"\n{'='*60}")
    print("Creating monthly 5-minute enriched pre-aggregated files")
    print(f"{'='*60}")
    print(f"Symbols found: {len(symbol_files)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate list of (year, month) tuples
    months_to_process = []
    current = from_dt.replace(day=1)
    while current <= to_dt:
        months_to_process.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    for year, month in months_to_process:
        month_start = pd.Timestamp(year=year, month=month, day=1)
        if month == 12:
            month_end = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(seconds=1)
        else:
            month_end = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(seconds=1)

        eff_start = max(month_start, from_dt)
        eff_end = min(month_end, to_dt)

        month_frames = []
        loaded = 0

        for symbol, path in symbol_files.items():
            df = load_and_filter(path, eff_start, eff_end)
            if df is None:
                continue

            # Ensure naive timestamps
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)

            df["symbol"] = symbol
            keep_cols = ["date", "symbol", "open", "high", "low", "close", "volume",
                         "vwap", "bb_width_proxy", "adx", "rsi"]
            df = df[[c for c in keep_cols if c in df.columns]]
            month_frames.append(df)
            loaded += 1

        if not month_frames:
            print(f"  {year}_{month:02d}: no 5m enriched data, skipping")
            continue

        combined = pd.concat(month_frames, ignore_index=True)
        del month_frames

        out_df = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
        del combined

        filename = f"{year}_{month:02d}_5m_enriched.feather"
        output_path = output_dir / filename
        out_df.to_feather(output_path)

        n_symbols = out_df["symbol"].nunique()
        n_rows = len(out_df)
        n_days = out_df["date"].dt.date.nunique()
        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"  Written: {filename} ({n_rows:,} rows, {n_symbols} symbols, {n_days} trading days, {size_mb:.1f} MB)")
        del out_df

    print("Monthly 5m enriched files created successfully!")


def main():
    args = parse_args()

    try:
        from_dt = pd.to_datetime(args.start_date)
        to_dt = pd.to_datetime(args.end_date)
    except ValueError:
        print("ERROR: Dates must be in YYYY-MM-DD format")
        sys.exit(1)

    # End of day for to_date (inclusive)
    to_dt = to_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)

    # Discover symbol files
    src = args.source_dir
    minute_files = discover_symbol_files("_1minutes", source_dir=src)
    src_label = src or str(ARCHIVE_DIR)
    print(f"Found {len(minute_files)} symbols with minute data in {src_label}")

    if not minute_files:
        print(f"ERROR: No symbol files found in {src_label}")
        print("Run upstox_cache_downloader.py or tools/compare_data_sources.py first")
        sys.exit(1)

    # Create monthly pre-aggregated 1m files
    create_monthly_files(minute_files, from_dt, to_dt, args.output_dir)

    # Create monthly pre-aggregated 5m enriched files
    enriched_5m_files = discover_symbol_files("_5minutes_enriched", source_dir=src)
    if enriched_5m_files:
        print(f"\nFound {len(enriched_5m_files)} symbols with enriched 5m data")
        create_monthly_5m_enriched(enriched_5m_files, from_dt, to_dt, args.output_dir)
    else:
        print("\nWARNING: No enriched 5m files found. Run tools/precompute_5m_cache.py first.")

    # Update consolidated daily
    if not args.skip_daily:
        daily_files = discover_symbol_files("_1days", source_dir=src)
        print(f"\nFound {len(daily_files)} symbols with daily data in {src_label}")
        if daily_files:
            update_consolidated_daily(daily_files, from_dt, to_dt)
        else:
            print("WARNING: No daily data files found, skipping consolidated daily update")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
