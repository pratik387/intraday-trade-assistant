#!/usr/bin/env python3
"""
Upload Regime Cache Files from Local Directory
==============================================

This script is meant to be run from OCI Cloud Shell after you've
extracted the cache files there.

Usage:
  1. Extract cache files in Cloud Shell: tar -xzf regime_caches.tar.gz
  2. Run this script: python3 upload_regime_caches_from_local.py
"""

import subprocess
import sys
from pathlib import Path


def get_config_hash():
    """Get configuration hash"""
    # Read from metadata or hardcode
    return "d0635bb8"  # Current config hash


def upload_file(bucket, local_file, object_name):
    """Upload file to OCI Object Storage"""
    print(f"Uploading {local_file.name} ({local_file.stat().st_size / (1024**2):.1f} MB)...", end=' ', flush=True)

    try:
        subprocess.run(
            ['oci', 'os', 'object', 'put',
             '--bucket-name', bucket,
             '--name', object_name,
             '--file', str(local_file),
             '--force'],
            capture_output=True,
            check=True
        )
        print("✅")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Upload failed: {e.stderr.decode() if e.stderr else e}")
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("Upload Regime Cache Files to OCI")
    print("=" * 60)
    print()

    config_hash = get_config_hash()
    bucket = 'backtest-cache'

    print(f"Config hash: {config_hash}")
    print(f"Bucket: {bucket}")
    print()

    # Files in current directory
    files_to_upload = [
        '2023_12_1m.feather',
        '2024_01_1m.feather',
        '2025_02_1m.feather',
        '2025_07_1m.feather'
    ]

    uploaded = 0
    failed = 0

    for filename in files_to_upload:
        file_path = Path(filename)

        if not file_path.exists():
            print(f"⚠️  File not found: {filename} (skipping)")
            continue

        object_name = f"monthly/{config_hash}/{filename}"

        if upload_file(bucket, file_path, object_name):
            uploaded += 1
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"✅ Uploaded: {uploaded}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    print("=" * 60)
    print()

    # Verify all files
    print("Verifying all cache files in OCI...")
    try:
        result = subprocess.run(
            ['oci', 'os', 'object', 'list',
             '--bucket-name', bucket,
             '--prefix', f'monthly/{config_hash}/',
             '--query', 'data[].name',
             '--output', 'table'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"⚠️  Could not verify: {e}")


if __name__ == '__main__':
    main()
