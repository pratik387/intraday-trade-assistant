#!/usr/bin/env python3
"""
OCI Kubernetes Backtest Worker - Entrypoint
============================================

This script runs inside each Kubernetes pod.
Each pod processes exactly ONE trading day.

Flow:
1. Get assigned date from JOB_COMPLETION_INDEX
2. Download code from OCI Object Storage
3. Download monthly cache for that date
4. Run backtest for that date
5. Upload results to OCI Object Storage
6. Exit

Environment Variables:
    JOB_COMPLETION_INDEX: Pod index (0-119)
    DATES_LIST: Comma-separated list of dates
    RUN_ID: Run identifier
    OCI_BUCKET_CACHE: Cache bucket name
    OCI_BUCKET_CODE: Code bucket name
    OCI_BUCKET_RESULTS: Results bucket name
    OCI_REGION: OCI region
    CONFIG_HASH: Configuration hash
"""

import os
import sys
import subprocess
import tarfile
import json
from datetime import datetime
from pathlib import Path
import oci

# Force unbuffered output for all Python print/logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(message):
    """Print with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)


def get_assigned_date():
    """Get the date this pod should process"""
    index = int(os.environ.get('JOB_COMPLETION_INDEX', '0'))
    dates_list = os.environ.get('DATES_LIST', '')

    if not dates_list:
        log("ERROR: DATES_LIST environment variable not set")
        sys.exit(1)

    dates = dates_list.split(',')

    if index >= len(dates):
        log(f"ERROR: Index {index} out of range (only {len(dates)} dates)")
        sys.exit(1)

    return dates[index]


def download_code():
    """Download code tarball from OCI Object Storage"""
    log("Downloading code from OCI Object Storage...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CODE', 'backtest-code')
    run_id = os.environ.get('RUN_ID')

    object_name = f"{run_id}/code.tar.gz"

    try:
        # Download code tarball
        get_obj = os_client.get_object(
            namespace_name=namespace,
            bucket_name=bucket,
            object_name=object_name
        )

        # Save to /tmp
        tarball_path = Path('/tmp/code.tar.gz')
        with open(tarball_path, 'wb') as f:
            for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)

        log(f"Downloaded code: {tarball_path.stat().st_size / (1024*1024):.1f} MB")

        # Extract to /app
        app_dir = Path('/app')
        app_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(path=app_dir)

        log(f"Extracted code to: {app_dir}")

        # Cleanup
        tarball_path.unlink()

    except Exception as e:
        log(f"ERROR downloading code: {e}")
        sys.exit(1)


def download_consolidated_daily_cache():
    """
    Download consolidated daily cache (contains all symbols' daily data).
    This is required for PDH/PDL/PDC calculations and daily indicators.
    """
    log("Downloading consolidated daily cache...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')

    object_name = "consolidated_daily.feather"

    try:
        # Download consolidated daily cache
        get_obj = os_client.get_object(
            namespace_name=namespace,
            bucket_name=bucket,
            object_name=object_name
        )

        # Save to /app/cache/preaggregate/
        cache_dir = Path('/app/cache/preaggregate')
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / 'consolidated_daily.feather'

        with open(cache_file, 'wb') as f:
            for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)

        size_mb = cache_file.stat().st_size / (1024 * 1024)
        log(f"Downloaded consolidated daily cache: {size_mb:.1f} MB")

    except oci.exceptions.ServiceError as e:
        if e.status == 404:
            log(f"WARNING: Consolidated daily cache not found in OCI")
            log("Daily indicators (PDH/PDL/PDC, EMA200, ADX) will not work!")
        else:
            log(f"ERROR downloading consolidated daily cache: {e}")
            raise

    except Exception as e:
        log(f"ERROR downloading consolidated daily cache: {e}")
        raise


def download_monthly_cache(date_str):
    """
    Download monthly cache file for the given date.

    Args:
        date_str: Date in YYYY-MM-DD format
    """
    log(f"Downloading monthly cache for {date_str}...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')

    # Parse date to get year and month
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    monthly_filename = f"{date_obj.year}_{date_obj.month:02d}_1m.feather"

    # Download path in OCI (monthly cache is config-independent - just raw 1m bars)
    object_name = f"monthly/{monthly_filename}"

    try:
        # Download monthly cache
        get_obj = os_client.get_object(
            namespace_name=namespace,
            bucket_name=bucket,
            object_name=object_name
        )

        # Save to /app/cache/preaggregate/
        cache_dir = Path('/app/cache/preaggregate')
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / monthly_filename

        with open(cache_file, 'wb') as f:
            for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)

        size_mb = cache_file.stat().st_size / (1024 * 1024)
        log(f"Downloaded cache: {monthly_filename} ({size_mb:.1f} MB)")

    except oci.exceptions.ServiceError as e:
        if e.status == 404:
            log(f"WARNING: Monthly cache not found: {object_name}")
            log("Attempting to download consolidated cache...")

            # Fallback: try consolidated cache
            try:
                object_name_fallback = f"{config_hash}/preagg_5m.feather"
                get_obj = os_client.get_object(
                    namespace_name=namespace,
                    bucket_name=bucket,
                    object_name=object_name_fallback
                )

                cache_file = cache_dir / 'preagg_5m.feather'
                with open(cache_file, 'wb') as f:
                    for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                        f.write(chunk)

                log(f"Downloaded consolidated cache: {cache_file.stat().st_size / (1024**2):.1f} MB")

            except Exception as e2:
                log(f"ERROR: Could not download cache: {e2}")
                sys.exit(1)
        else:
            log(f"ERROR downloading cache: {e}")
            sys.exit(1)

    except Exception as e:
        log(f"ERROR downloading cache: {e}")
        sys.exit(1)


def download_index_data(date_str):
    """
    Download index OHLCV data for risk modulation (if enabled in config).

    Downloads full 1-minute data files for NIFTY indices used by IndexSectorRiskModulator.
    Only downloads if risk_modulator.enabled is True in config.

    Args:
        date_str: Date in YYYY-MM-DD format (unused - downloads full file)
    """
    # Check if risk modulator is enabled in config
    # Try JSON config first (configuration.json), then fallback to TOML
    config_path = Path('/app/config/configuration.json')
    log(f"Checking for config at: {config_path} (exists: {config_path.exists()})")

    if not config_path.exists():
        config_path = Path('/app/config/default.toml')
        log(f"Fallback to TOML: {config_path} (exists: {config_path.exists()})")

    if not config_path.exists():
        # List what's actually in /app/config for debugging
        config_dir = Path('/app/config')
        if config_dir.exists():
            files = list(config_dir.rglob('*'))
            log(f"Config dir exists but target file missing. Contents: {[str(f.relative_to(config_dir)) for f in files[:20]]}")
        else:
            log(f"Config directory does not exist: {config_dir}")
        log("No config file found, skipping index data download")
        return

    log(f"Using config file: {config_path}")

    try:
        if config_path.suffix == '.json':
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            import tomllib
            with open(config_path, 'rb') as f:
                config = tomllib.load(f)

        risk_mod_cfg = config.get('risk_modulator', {})
        if not risk_mod_cfg.get('enabled', False):
            log("Risk modulator disabled, skipping index data download")
            return

        primary_indices = risk_mod_cfg.get('primary_indices', [])
        if not primary_indices:
            log("No primary indices configured, skipping index data download")
            return

    except Exception as e:
        log(f"Could not read config for risk modulator check: {e}")
        return

    log(f"Downloading index data for risk modulation ({len(primary_indices)} indices)...")

    oci_config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(oci_config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')

    # Create output directory
    index_dir = Path('/app/backtest-cache-download/index_ohlcv')
    index_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for index_symbol in primary_indices:
        # Convert symbol to safe filename: "NSE:NIFTY 50" -> "NSE_NIFTY_50"
        safe_symbol = index_symbol.replace(":", "_").replace(" ", "_")

        # Object path in OCI: index_ohlcv/{symbol}/{symbol}_1minutes.feather (full file)
        object_name = f"index_ohlcv/{safe_symbol}/{safe_symbol}_1minutes.feather"

        # Local path
        symbol_dir = index_dir / safe_symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        local_file = symbol_dir / f"{safe_symbol}_1minutes.feather"

        try:
            get_obj = os_client.get_object(
                namespace_name=namespace,
                bucket_name=bucket,
                object_name=object_name
            )

            with open(local_file, 'wb') as f:
                for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                    f.write(chunk)

            size_mb = local_file.stat().st_size / (1024 * 1024)
            log(f"  Downloaded {index_symbol}: {size_mb:.1f} MB")
            downloaded += 1

        except oci.exceptions.ServiceError as e:
            if e.status == 404:
                log(f"  WARNING: Index data not found for {index_symbol}")
            else:
                log(f"  ERROR downloading {index_symbol}: {e}")

        except Exception as e:
            log(f"  ERROR downloading {index_symbol}: {e}")

    log(f"Downloaded {downloaded}/{len(primary_indices)} index data files")


def run_backtest(date_str):
    """
    Run backtest for the given date.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        dict: Execution results
    """
    log(f"Running backtest for {date_str}...")

    # Change to /app directory
    os.chdir('/app')

    # Run main.py with run-prefix (like engine.py does)
    # This ensures proper logger initialization with file handlers
    run_prefix = f"bt_{date_str}_"
    cmd = [
        sys.executable,
        'main.py',
        '--dry-run',
        '--session-date', date_str,
        '--from-hhmm', '09:15',
        '--to-hhmm', '15:15',
        '--run-prefix', run_prefix
    ]

    log(f"Command: {' '.join(cmd)}")

    start_time = datetime.now()

    try:
        # Run WITHOUT capture_output so logs appear in real-time
        result = subprocess.run(
            cmd,
            timeout=5400  # 90 minutes timeout per day (increased from 60min due to cloud overhead)
        )

        runtime_sec = (datetime.now() - start_time).total_seconds()

        log(f"Backtest complete: {runtime_sec:.1f}s")

        if result.returncode != 0:
            log(f"WARNING: Backtest exited with code {result.returncode}")

        return {
            'date': date_str,
            'runtime_sec': runtime_sec,
            'exit_code': result.returncode
        }

    except subprocess.TimeoutExpired:
        runtime_sec = (datetime.now() - start_time).total_seconds()
        log(f"ERROR: Backtest timeout after {runtime_sec:.1f}s")
        return {
            'date': date_str,
            'runtime_sec': 3600,
            'exit_code': -1,
            'error': 'Timeout'
        }
    except Exception as e:
        log(f"ERROR running backtest: {e}")
        return {
            'date': date_str,
            'runtime_sec': 0,
            'exit_code': -1,
            'error': str(e)
        }


def upload_results(date_str):
    """
    Upload results to OCI Object Storage.

    Args:
        date_str: Date in YYYY-MM-DD format
    """
    log(f"Uploading results for {date_str}...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_RESULTS', 'backtest-results')
    run_id = os.environ.get('RUN_ID')

    # Find session logs directory
    logs_dir = Path('/app/logs')

    if not logs_dir.exists():
        log("WARNING: No logs directory found")
        return

    # Find latest session for this date
    # Match the new run_prefix pattern: bt_{date}_*
    session_dirs = sorted(logs_dir.glob(f'bt_{date_str}_*'))

    if not session_dirs:
        log(f"WARNING: No session logs found for {date_str}")
        return

    session_dir = session_dirs[-1]  # Latest session
    log(f"Found session: {session_dir.name}")

    uploaded = 0

    # Upload all files in session directory
    for file_path in session_dir.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(session_dir)
            object_name = f"{run_id}/{date_str}/{relative_path}"

            try:
                with open(file_path, 'rb') as f:
                    os_client.put_object(
                        namespace_name=namespace,
                        bucket_name=bucket,
                        object_name=object_name,
                        put_object_body=f
                    )

                uploaded += 1

            except Exception as e:
                log(f"WARNING: Failed to upload {file_path.name}: {e}")

    log(f"Uploaded {uploaded} result files")


def main():
    """Main entrypoint"""
    log("=" * 60)
    log("OCI Kubernetes Backtest Worker")
    log("=" * 60)

    # Get assigned date
    date_str = get_assigned_date()
    index = os.environ.get('JOB_COMPLETION_INDEX', '0')

    log(f"Pod Index: {index}")
    log(f"Assigned Date: {date_str}")
    log(f"Run ID: {os.environ.get('RUN_ID')}")

    # Download code
    download_code()

    # Download consolidated daily cache (shared across all dates)
    download_consolidated_daily_cache()

    # Download monthly cache for this specific date
    download_monthly_cache(date_str)

    # Download index data for risk modulation (if enabled)
    download_index_data(date_str)

    # Run backtest
    result = run_backtest(date_str)

    # Upload results
    upload_results(date_str)

    # Exit with backtest exit code
    exit_code = result.get('exit_code', 0)

    if exit_code == 0:
        log(f"✅ SUCCESS: {date_str}")
    else:
        log(f"❌ FAILED: {date_str} (exit code: {exit_code})")

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
