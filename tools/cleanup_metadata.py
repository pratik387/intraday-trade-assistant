#!/usr/bin/env python
"""
Cleanup metadata files to match actual feather file contents.

Reads each feather file to get actual date range, then rewrites metadata
with a single consolidated entry.

Usage:
    python tools/cleanup_metadata.py
    python tools/cleanup_metadata.py --dry-run  # Preview only
"""
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache" / "ohlcv_archive"

def cleanup_metadata(dry_run: bool = False):
    """Clean up all metadata files to match actual feather contents."""

    symbol_dirs = sorted([d for d in CACHE_DIR.iterdir() if d.is_dir()])
    total = len(symbol_dirs)

    print(f"{'='*60}")
    print(f"METADATA CLEANUP {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*60}")
    print(f"Found {total} symbol directories\n")

    fixed = 0
    skipped = 0
    errors = 0

    for i, symbol_dir in enumerate(symbol_dirs, 1):
        symbol_ns = symbol_dir.name
        feather_file = symbol_dir / f"{symbol_ns}_1minutes.feather"
        metadata_file = symbol_dir / f"{symbol_ns}_1minutes_metadata.json"

        if not feather_file.exists():
            skipped += 1
            continue

        try:
            # Read actual date range from feather
            df = pd.read_feather(feather_file)

            if 'date' not in df.columns:
                skipped += 1
                continue

            min_date = pd.to_datetime(df['date'].min())
            max_date = pd.to_datetime(df['date'].max())
            rows = len(df)

            # Strip timezone for clean date strings
            start_str = min_date.strftime('%Y-%m-%d')
            end_str = max_date.strftime('%Y-%m-%d')

            # Create clean metadata
            new_metadata = {
                "downloaded_ranges": [
                    {
                        "start_date": start_str,
                        "end_date": end_str,
                        "name": "full_3yr_download",
                        "rows": rows
                    }
                ],
                "last_updated": datetime.now().isoformat()
            }

            if not dry_run:
                with open(metadata_file, 'w') as f:
                    json.dump(new_metadata, f, indent=2)

            fixed += 1

            if i % 200 == 0:
                print(f"  [{i:4d}/{total}] Fixed: {fixed}, Skipped: {skipped}, Errors: {errors}")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {symbol_ns}: {str(e)[:50]}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Fixed:   {fixed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")

    if dry_run:
        print(f"\n  [DRY RUN] No files were modified. Run without --dry-run to apply changes.")
    else:
        print(f"\n  [DONE] All metadata files cleaned up!")

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    cleanup_metadata(dry_run=dry_run)
