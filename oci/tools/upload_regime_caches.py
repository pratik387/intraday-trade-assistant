#!/usr/bin/env python3
"""
Upload Regime Cache Files to OCI Object Storage
===============================================

Uploads the 6 regime month cache files to OCI for parallel backtesting.
"""

import subprocess
import sys
import hashlib
import json
from pathlib import Path


def get_config_hash():
    """Get configuration hash"""
    config_file = Path('config/configuration.json')
    with open(config_file, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def get_namespace():
    """Get OCI Object Storage namespace"""
    try:
        result = subprocess.run(
            ['oci', 'os', 'ns', 'get', '--query', 'data', '--raw-output'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Error getting namespace: {e}")
        print("Make sure OCI CLI is configured: oci setup config")
        sys.exit(1)


def check_file_exists(namespace, bucket, object_name):
    """Check if object exists in OCI"""
    try:
        subprocess.run(
            ['oci', 'os', 'object', 'head',
             '--namespace', namespace,
             '--bucket-name', bucket,
             '--name', object_name],
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def upload_file(namespace, bucket, local_path, object_name):
    """Upload file to OCI Object Storage"""
    try:
        subprocess.run(
            ['oci', 'os', 'object', 'put',
             '--namespace', namespace,
             '--bucket-name', bucket,
             '--name', object_name,
             '--file', str(local_path),
             '--force'],
            capture_output=True,
            check=True
        )
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

    # Get config hash and namespace
    config_hash = get_config_hash()
    namespace = get_namespace()
    bucket = 'backtest-cache'

    print(f"Config hash: {config_hash}")
    print(f"Namespace: {namespace}")
    print(f"Bucket: {bucket}")
    print()

    # Files to upload (all 6 regime months)
    cache_dir = Path('cache/preaggregate')
    files_to_upload = [
        '2023_12_1m.feather',  # Strong_Uptrend
        '2024_01_1m.feather',  # Shock_Down
        '2024_06_1m.feather',  # Event_Driven_HighVol
        '2024_10_1m.feather',  # Correction_RiskOff
        '2025_02_1m.feather',  # Prolonged_Drawdown
        '2025_07_1m.feather'   # Low_Vol_Range
    ]

    uploaded = 0
    skipped = 0
    failed = 0

    for filename in files_to_upload:
        file_path = cache_dir / filename

        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            failed += 1
            continue

        object_name = f"monthly/{config_hash}/{filename}"
        size_mb = file_path.stat().st_size / (1024 ** 2)

        # Check if already exists
        if check_file_exists(namespace, bucket, object_name):
            print(f"⚠️  Already exists: {filename} ({size_mb:.1f} MB)")
            skipped += 1
            continue

        # Upload
        print(f"Uploading {filename} ({size_mb:.1f} MB)...", end=' ', flush=True)

        if upload_file(namespace, bucket, file_path, object_name):
            print("✅")
            uploaded += 1
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"✅ Uploaded: {uploaded}")
    print(f"⚠️  Skipped: {skipped}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
