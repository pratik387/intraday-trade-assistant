#!/usr/bin/env python3
"""
Upload Monthly Cache Files to OCI Object Storage
=================================================

Uploads pre-aggregated monthly feather files (1m bars + 5m enriched) to
the backtest-cache bucket for OCI backtest pods to download.

Usage:
    python oci/tools/upload_monthly_cache.py                    # Upload all
    python oci/tools/upload_monthly_cache.py --months 2026_03   # Specific month
    python oci/tools/upload_monthly_cache.py --5m-only          # Only 5m enriched
"""

import argparse
import sys
from pathlib import Path

# Use oci CLI from the current Python environment
_OCI_CLI = str(Path(sys.executable).parent / "oci")

MONTHLY_DIR = Path(__file__).resolve().parents[2] / "backtest-cache-download" / "monthly"
BUCKET = "backtest-cache"


def upload_file(bucket, local_file, object_name):
    """Upload file to OCI Object Storage using OCI CLI."""
    import subprocess
    size_mb = local_file.stat().st_size / (1024 ** 2)
    print(f"  Uploading {object_name} ({size_mb:.1f} MB)...", end=" ", flush=True)
    try:
        subprocess.run(
            [_OCI_CLI, "os", "object", "put",
             "--bucket-name", bucket,
             "--name", object_name,
             "--file", str(local_file),
             "--force"],
            capture_output=True,
            check=True,
        )
        print("OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {e.stderr.decode() if e.stderr else e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload monthly cache to OCI")
    parser.add_argument("--months", type=str, default=None,
                        help="Comma-separated month prefixes (e.g., 2026_03,2026_02). Default: all")
    parser.add_argument("--5m-only", action="store_true", dest="five_m_only",
                        help="Only upload 5m enriched files")
    parser.add_argument("--1m-only", action="store_true", dest="one_m_only",
                        help="Only upload 1m files")
    args = parser.parse_args()

    if not MONTHLY_DIR.exists():
        print(f"ERROR: {MONTHLY_DIR} does not exist")
        print("Run: python tools/create_preaggregated_cache.py --from 2023-01-01 --to 2026-03-30")
        sys.exit(1)

    # Discover files
    all_files = sorted(MONTHLY_DIR.glob("*.feather"))
    if args.months:
        prefixes = [m.strip() for m in args.months.split(",")]
        all_files = [f for f in all_files if any(f.name.startswith(p) for p in prefixes)]

    if args.five_m_only:
        all_files = [f for f in all_files if "5m_enriched" in f.name]
    elif args.one_m_only:
        all_files = [f for f in all_files if "1m.feather" in f.name]

    if not all_files:
        print("No files to upload")
        sys.exit(0)

    total_mb = sum(f.stat().st_size for f in all_files) / (1024 ** 2)
    print(f"Uploading {len(all_files)} files ({total_mb:.0f} MB total) to {BUCKET}/monthly/")
    print()

    ok, fail = 0, 0
    for f in all_files:
        object_name = f"monthly/{f.name}"
        if upload_file(BUCKET, f, object_name):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} uploaded, {fail} failed")


if __name__ == "__main__":
    main()
