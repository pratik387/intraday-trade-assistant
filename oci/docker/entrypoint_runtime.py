#!/usr/bin/env python3
"""
OCI Kubernetes Backtest Worker - Runtime Installation Entrypoint
================================================================

This version installs dependencies at runtime, eliminating the need
to build a custom Docker image. Uses public python:3.11-slim image.

Flow:
1. Install system dependencies (TA-Lib)
2. Install Python dependencies
3. Get assigned date from JOB_COMPLETION_INDEX
4. Download code from OCI Object Storage
5. Download monthly cache for that date
6. Run backtest for that date
7. Upload results to OCI Object Storage
8. Exit
"""

import os
import sys
import subprocess
import tarfile
import json
from datetime import datetime
from pathlib import Path


def log(message):
    """Print with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)


def install_dependencies():
    """Install system and Python dependencies at runtime"""
    log("Installing dependencies...")

    # Install TA-Lib
    talib_installed = Path('/usr/lib/libta_lib.so').exists()

    if not talib_installed:
        log("Installing TA-Lib...")
        commands = [
            "apt-get update",
            "apt-get install -y build-essential wget",
            "wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz",
            "tar -xzf ta-lib-0.4.0-src.tar.gz",
            "cd ta-lib && ./configure --prefix=/usr && make && make install",
            "cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz"
        ]

        for cmd in commands:
            result = subprocess.run(cmd, shell=True, capture_output=True)
            if result.returncode != 0:
                log(f"Warning: Command failed: {cmd}")

    # Install Python packages
    log("Installing Python packages...")
    packages = [
        "oci",
        "pandas",
        "numpy",
        "pyarrow",
        "TA-Lib",
        "python-dateutil",
        "pytz"
    ]

    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet"] + packages)
    log("Dependencies installed!")


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

    import oci
    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CODE', 'backtest-code')
    run_id = os.environ.get('RUN_ID')

    object_name = f"{run_id}/code.tar.gz"

    try:
        get_obj = os_client.get_object(
            namespace_name=namespace,
            bucket_name=bucket,
            object_name=object_name
        )

        tarball_path = Path('/tmp/code.tar.gz')
        with open(tarball_path, 'wb') as f:
            for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)

        log(f"Downloaded code: {tarball_path.stat().st_size / (1024*1024):.1f} MB")

        app_dir = Path('/app')
        app_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(path=app_dir)

        log(f"Extracted code to: {app_dir}")
        tarball_path.unlink()

    except Exception as e:
        log(f"ERROR downloading code: {e}")
        sys.exit(1)


def download_monthly_cache(date_str):
    """Download monthly cache file for the given date"""
    log(f"Downloading monthly cache for {date_str}...")

    import oci
    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')

    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    monthly_filename = f"{date_obj.year}_{date_obj.month:02d}_1m.feather"

    config_hash = os.environ.get('CONFIG_HASH', 'default')
    object_name = f"monthly/{config_hash}/{monthly_filename}"

    try:
        get_obj = os_client.get_object(
            namespace_name=namespace,
            bucket_name=bucket,
            object_name=object_name
        )

        cache_dir = Path('/app/cache/preaggregate')
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / monthly_filename

        with open(cache_file, 'wb') as f:
            for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)

        log(f"Downloaded cache: {monthly_filename} ({cache_file.stat().st_size / (1024 * 1024):.1f} MB)")

    except Exception as e:
        log(f"ERROR downloading cache: {e}")
        sys.exit(1)


def run_backtest(date_str):
    """Run backtest for the given date"""
    log(f"Running backtest for {date_str}...")

    os.chdir('/app')

    cmd = [
        sys.executable,
        'main.py',
        '--dry-run',
        '--session-date', date_str,
        '--from-hhmm', '09:25',
        '--to-hhmm', '15:15'
    ]

    log(f"Command: {' '.join(cmd)}")
    start_time = datetime.now()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        runtime_sec = (datetime.now() - start_time).total_seconds()
        log(f"Backtest complete: {runtime_sec:.1f}s")

        if result.returncode != 0:
            log(f"WARNING: Backtest exited with code {result.returncode}")

        return {
            'date': date_str,
            'runtime_sec': runtime_sec,
            'exit_code': result.returncode
        }

    except Exception as e:
        log(f"ERROR running backtest: {e}")
        return {'date': date_str, 'runtime_sec': 0, 'exit_code': -1, 'error': str(e)}


def upload_results(date_str):
    """Upload results to OCI Object Storage"""
    log(f"Uploading results for {date_str}...")

    import oci
    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_RESULTS', 'backtest-results')
    run_id = os.environ.get('RUN_ID')

    logs_dir = Path('/app/logs')
    if not logs_dir.exists():
        log("WARNING: No logs directory found")
        return

    date_prefix = date_str.replace('-', '')
    session_dirs = sorted(logs_dir.glob(f'{date_prefix}_*'))

    if not session_dirs:
        log(f"WARNING: No session logs found for {date_str}")
        return

    session_dir = session_dirs[-1]
    log(f"Found session: {session_dir.name}")

    uploaded = 0
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
    log("OCI Kubernetes Backtest Worker (Runtime Install)")
    log("=" * 60)

    # Install dependencies first
    install_dependencies()

    # Get assigned date
    date_str = get_assigned_date()
    index = os.environ.get('JOB_COMPLETION_INDEX', '0')

    log(f"Pod Index: {index}")
    log(f"Assigned Date: {date_str}")
    log(f"Run ID: {os.environ.get('RUN_ID')}")

    # Download code
    download_code()

    # Download monthly cache
    download_monthly_cache(date_str)

    # Run backtest
    result = run_backtest(date_str)

    # Upload results
    upload_results(date_str)

    exit_code = result.get('exit_code', 0)

    if exit_code == 0:
        log(f"✅ SUCCESS: {date_str}")
    else:
        log(f"❌ FAILED: {date_str} (exit code: {exit_code})")

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
