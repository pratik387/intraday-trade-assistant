#!/usr/bin/env python3
"""
Upload Index OHLCV Data to OCI Object Storage
==============================================

Uploads index 1-minute data files for risk modulation backtesting.
These files are downloaded by the OCI backtest worker when risk_modulator is enabled.

The files are organized by:
- index_ohlcv/{symbol}/{symbol}_{year}_{month}_1minutes.feather

Usage:
    python oci_cloud/tools/upload_index_data.py                    # Upload all index data
    python oci_cloud/tools/upload_index_data.py --symbol "NSE:NIFTY 50"  # Upload specific index
    python oci_cloud/tools/upload_index_data.py --dry-run          # Show what would be uploaded
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Project root
ROOT = Path(__file__).resolve().parents[2]
INDEX_DATA_DIR = ROOT / "backtest-cache-download" / "index_ohlcv"

# OCI bucket for cache files
BUCKET_NAME = "backtest-cache"


def get_namespace():
    """Get OCI namespace."""
    try:
        result = subprocess.run(
            ['oci', 'os', 'ns', 'get', '--query', 'data', '--raw-output'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"ERROR: Could not get OCI namespace: {e}")
        sys.exit(1)


def upload_file(namespace: str, local_path: Path, object_name: str, dry_run: bool = False) -> bool:
    """Upload a file to OCI Object Storage."""
    if dry_run:
        print(f"  [DRY-RUN] Would upload: {local_path.name} -> {object_name}")
        return True

    try:
        result = subprocess.run(
            ['oci', 'os', 'object', 'put',
             '--namespace', namespace,
             '--bucket-name', BUCKET_NAME,
             '--name', object_name,
             '--file', str(local_path),
             '--force'],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR uploading {local_path.name}: {e.stderr}")
        return False


def get_object_exists(namespace: str, object_name: str) -> bool:
    """Check if object exists in OCI."""
    try:
        result = subprocess.run(
            ['oci', 'os', 'object', 'head',
             '--namespace', namespace,
             '--bucket-name', BUCKET_NAME,
             '--name', object_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload index OHLCV data to OCI for risk modulation backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python oci_cloud/tools/upload_index_data.py                    # Upload all
  python oci_cloud/tools/upload_index_data.py --symbol "NSE:NIFTY 50"  # Specific index
  python oci_cloud/tools/upload_index_data.py --dry-run          # Preview
  python oci_cloud/tools/upload_index_data.py --force            # Re-upload existing
        """
    )

    parser.add_argument(
        "--symbol",
        type=str,
        help="Specific index symbol to upload (e.g., 'NSE:NIFTY 50')"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload files even if they already exist in OCI"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("INDEX DATA UPLOAD TO OCI")
    print("=" * 70)

    # Check if index data directory exists
    if not INDEX_DATA_DIR.exists():
        print(f"ERROR: Index data directory not found: {INDEX_DATA_DIR}")
        print("\nRun 'python download_index_data.py' first to download index data.")
        sys.exit(1)

    # Get OCI namespace
    if not args.dry_run:
        namespace = get_namespace()
        print(f"OCI Namespace: {namespace}")
    else:
        namespace = "dry-run-namespace"
        print("DRY-RUN MODE - No files will be uploaded")

    print(f"Bucket: {BUCKET_NAME}")
    print(f"Source: {INDEX_DATA_DIR}")
    print()

    # Find all feather files
    if args.symbol:
        # Convert symbol to safe directory name
        safe_symbol = args.symbol.replace(":", "_").replace(" ", "_")
        symbol_dirs = [INDEX_DATA_DIR / safe_symbol]
        if not symbol_dirs[0].exists():
            print(f"ERROR: No data found for symbol: {args.symbol}")
            print(f"Expected directory: {symbol_dirs[0]}")
            sys.exit(1)
    else:
        symbol_dirs = [d for d in INDEX_DATA_DIR.iterdir() if d.is_dir()]

    if not symbol_dirs:
        print("ERROR: No index data directories found")
        sys.exit(1)

    print(f"Found {len(symbol_dirs)} index symbol(s)")
    print()

    total_files = 0
    uploaded_files = 0
    skipped_files = 0
    failed_files = 0

    for symbol_dir in sorted(symbol_dirs):
        symbol_name = symbol_dir.name
        feather_files = list(symbol_dir.glob("*_1minutes.feather"))

        if not feather_files:
            print(f"  {symbol_name}: No 1-minute files found")
            continue

        print(f"{symbol_name}:")

        for feather_file in sorted(feather_files):
            total_files += 1

            # Object path in OCI
            object_name = f"index_ohlcv/{symbol_name}/{feather_file.name}"

            # Check if exists (skip if not force)
            if not args.force and not args.dry_run:
                if get_object_exists(namespace, object_name):
                    print(f"  [SKIP] {feather_file.name} (already exists)")
                    skipped_files += 1
                    continue

            # Upload
            size_mb = feather_file.stat().st_size / (1024 * 1024)
            if upload_file(namespace, feather_file, object_name, args.dry_run):
                print(f"  [OK] {feather_file.name} ({size_mb:.1f} MB)")
                uploaded_files += 1
            else:
                failed_files += 1

    print()
    print("=" * 70)
    print("UPLOAD SUMMARY")
    print("=" * 70)
    print(f"Total files:    {total_files}")
    print(f"Uploaded:       {uploaded_files}")
    print(f"Skipped:        {skipped_files}")
    print(f"Failed:         {failed_files}")

    if args.dry_run:
        print("\n[DRY-RUN] No files were actually uploaded.")
        print("Run without --dry-run to upload files.")


if __name__ == "__main__":
    main()
