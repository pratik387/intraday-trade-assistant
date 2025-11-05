#!/usr/bin/env python3
"""
Verify Cache Files - Compare Local vs OCI
==========================================

Checks all regime month cache files and reports:
1. Local file size and symbol count
2. OCI file size (if exists)
3. Whether they match or need re-upload
"""

import subprocess
import sys
from pathlib import Path


def get_namespace():
    """Get OCI namespace (works only if OCI CLI is available)"""
    try:
        result = subprocess.run(
            ['oci', 'os', 'ns', 'get', '--query', 'data', '--raw-output'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return None


def get_oci_object_size(namespace, bucket, object_name):
    """Get size of object in OCI (returns None if not found)"""
    try:
        result = subprocess.run(
            ['oci', 'os', 'object', 'head',
             '--namespace', namespace,
             '--bucket-name', bucket,
             '--name', object_name,
             '--query', 'content-length',
             '--raw-output'],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except:
        return None


def get_symbol_count_from_cache(cache_file):
    """Get symbol count from feather file"""
    try:
        import pandas as pd
        df = pd.read_feather(cache_file)
        return df['symbol'].nunique()
    except Exception as e:
        return None


def format_size(bytes_size):
    """Format bytes to human readable"""
    if bytes_size is None:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def main():
    """Main verification"""
    print("=" * 80)
    print("Cache File Verification - Local vs OCI")
    print("=" * 80)
    print()

    print("Monthly cache is config-independent (raw 1m OHLC bars)")
    print()

    # Check if OCI CLI is available
    namespace = get_namespace()
    oci_available = namespace is not None

    if oci_available:
        print(f"OCI namespace: {namespace}")
        print()
    else:
        print("WARNING: OCI CLI not available - showing local files only")
        print("   (Run this from OCI Cloud Shell to check OCI files)")
        print()

    bucket = 'backtest-cache'
    cache_dir = Path('cache/preaggregate')

    # All regime month cache files
    cache_files = [
        ('2023_12_1m.feather', 'Strong_Uptrend (Dec 2023)'),
        ('2024_01_1m.feather', 'Shock_Down (Jan 2024)'),
        ('2024_06_1m.feather', 'Event_Driven_HighVol (Jun 2024)'),
        ('2024_10_1m.feather', 'Correction_RiskOff (Oct 2024)'),
        ('2025_02_1m.feather', 'Prolonged_Drawdown (Feb 2025)'),
        ('2025_07_1m.feather', 'Low_Vol_Range (Jul 2025)')
    ]

    print(f"{'File':<22} {'Regime':<35} {'Local':<12} {'Symbols':<8} {'OCI':<12} {'Status':<15}")
    print("=" * 110)

    results = []
    for filename, regime in cache_files:
        local_file = cache_dir / filename

        # Local file info
        if local_file.exists():
            local_size = local_file.stat().st_size
            local_size_str = format_size(local_size)

            # Try to get symbol count
            symbol_count = get_symbol_count_from_cache(local_file)
            symbol_str = str(symbol_count) if symbol_count else "?"
        else:
            local_size = None
            local_size_str = "MISSING"
            symbol_str = "N/A"

        # OCI file info
        if oci_available:
            object_name = f"monthly/{filename}"  # Config-independent
            oci_size = get_oci_object_size(namespace, bucket, object_name)
            oci_size_str = format_size(oci_size)

            # Determine status
            if local_size is None:
                status = "NO LOCAL"
            elif oci_size is None:
                status = "NEED UPLOAD"
            elif local_size == oci_size:
                status = "OK - MATCH"
            else:
                diff = local_size - oci_size
                if diff > 0:
                    status = "OCI SMALLER"
                else:
                    status = "LOCAL NEWER"
        else:
            oci_size = None
            oci_size_str = "N/A"
            status = "CHECK OCI"

        print(f"{filename:<22} {regime:<35} {local_size_str:<12} {symbol_str:<8} {oci_size_str:<12} {status:<15}")

        results.append({
            'filename': filename,
            'regime': regime,
            'local_size': local_size,
            'oci_size': oci_size,
            'symbol_count': symbol_count,
            'status': status
        })

    print("=" * 110)
    print()

    # Summary
    if oci_available:
        match_count = sum(1 for r in results if 'MATCH' in r['status'])
        need_upload = sum(1 for r in results if 'UPLOAD' in r['status'])
        mismatch = sum(1 for r in results if 'SMALLER' in r['status'] or 'NEWER' in r['status'])

        print("Summary:")
        print(f"  OK: Files matching: {match_count}")
        print(f"  WARNING: Need upload: {need_upload}")
        print(f"  WARNING: Size mismatch: {mismatch}")
        print()

        if need_upload > 0 or mismatch > 0:
            print("Action Required:")
            print("  1. Files that need upload or have mismatches should be re-uploaded")
            print("  2. Use oci/tools/upload_regime_caches.py to upload missing files")
            print("  3. Or manually upload via OCI Console:")
            print(f"     Bucket: {bucket}")
            print(f"     Prefix: monthly/")
            print()
    else:
        local_count = sum(1 for r in results if r['local_size'] is not None)
        print(f"Found {local_count}/6 cache files locally")
        print()
        print("To check OCI files:")
        print("  Run this script from OCI Cloud Shell where OCI CLI is available")
        print()

    print("=" * 80)


if __name__ == '__main__':
    main()
