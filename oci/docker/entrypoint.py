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


def apply_oci_config_override():
    """Fold sub8_oci_overrides.json into configuration.json (in place).

    Bundled in the code tarball; missing override is a hard failure rather
    than silent skip — running OCI without the override gives a misleading
    "everything works" result while actually capturing only gap_fade_short.

    Delegates to tools/apply_oci_override.py which has unit tests and is
    usable for local wide-open smoke runs too. Same merge logic both ways.
    """
    base_path = Path('/app/config/configuration.json')
    override_path = Path('/app/config/sub8_oci_overrides.json')

    if not base_path.exists():
        log(f"ERROR: base config not found: {base_path}")
        sys.exit(1)
    if not override_path.exists():
        log(f"ERROR: OCI override not found: {override_path}")
        log("       (Should be bundled in the code tarball — check submit_oci_backtest.py include_patterns)")
        sys.exit(1)

    log(f"Applying OCI config override: {override_path} -> {base_path}")
    cmd = [
        sys.executable,
        '/app/tools/apply_oci_override.py',
        '--base', str(base_path),
        '--override', str(override_path),
    ]
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True,
            env={**os.environ, 'PYTHONPATH': '/app'},
        )
        # Surface the utility's one-line summary in OCI logs.
        for line in (result.stdout or '').splitlines():
            log(f"  {line}")
    except subprocess.CalledProcessError as e:
        log(f"ERROR applying OCI override:")
        log(f"  stdout: {e.stdout}")
        log(f"  stderr: {e.stderr}")
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


def _download_oci_file(os_client, namespace, bucket, object_name, local_path):
    """Download a single file from OCI Object Storage. Returns True on success."""
    try:
        get_obj = os_client.get_object(
            namespace_name=namespace,
            bucket_name=bucket,
            object_name=object_name
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'wb') as f:
            for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)
        size_mb = local_path.stat().st_size / (1024 * 1024)
        log(f"Downloaded: {object_name} ({size_mb:.1f} MB)")
        return True
    except oci.exceptions.ServiceError as e:
        if e.status == 404:
            log(f"WARNING: Not found in OCI: {object_name}")
            return False
        raise


def download_monthly_cache(date_str):
    """
    Download monthly cache files for the given date:
    - 1m bars (for execution replay)
    - 5m enriched bars (for structure detection — precomputed indicators)
    """
    log(f"Downloading monthly cache for {date_str}...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')

    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    year_month = f"{date_obj.year}_{date_obj.month:02d}"

    # Primary: download 1m bars (execution replay) and 5m enriched (structure detection)
    cache_dir = Path('/app/cache/preaggregate')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Also save to backtest-cache-download/monthly for FeatherTickLoader fast path
    monthly_dir = Path('/app/backtest-cache-download/monthly')
    monthly_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        (f"monthly/{year_month}_1m.feather", cache_dir / f"{year_month}_1m.feather"),
        (f"monthly/{year_month}_1m.feather", monthly_dir / f"{year_month}_1m.feather"),
        (f"monthly/{year_month}_5m_enriched.feather", monthly_dir / f"{year_month}_5m_enriched.feather"),
    ]

    success_count = 0
    for object_name, local_path in files_to_download:
        try:
            if _download_oci_file(os_client, namespace, bucket, object_name, local_path):
                success_count += 1
        except Exception as e:
            log(f"ERROR downloading {object_name}: {e}")

    if success_count == 0:
        log("ERROR: No cache files downloaded — backtest may fail")
        sys.exit(1)


def download_index_ohlcv():
    """
    Download index OHLCV feather from OCI Object Storage.
    Required by DirectionalBiasTracker for backtest Nifty price lookups.

    Bucket path:  index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1minutes.feather
    Local path:   /app/backtest-cache-download/index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1minutes.feather
    """
    log("Downloading index OHLCV for directional bias...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')

    object_name = "index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1minutes.feather"

    try:
        get_obj = os_client.get_object(
            namespace_name=namespace,
            bucket_name=bucket,
            object_name=object_name
        )

        local_dir = Path('/app/backtest-cache-download/index_ohlcv/NSE_NIFTY_50')
        local_dir.mkdir(parents=True, exist_ok=True)

        local_file = local_dir / 'NSE_NIFTY_50_1minutes.feather'

        with open(local_file, 'wb') as f:
            for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)

        size_mb = local_file.stat().st_size / (1024 * 1024)
        log(f"Downloaded index OHLCV: {size_mb:.1f} MB")

    except oci.exceptions.ServiceError as e:
        if e.status == 404:
            log(f"WARNING: Index OHLCV not found in OCI: {object_name}")
            log("Directional bias will be disabled for this run")
        else:
            log(f"ERROR downloading index OHLCV: {e}")
            raise

    except Exception as e:
        log(f"ERROR downloading index OHLCV: {e}")
        raise


def download_option_chain(date_str):
    """Download NSE F&O OI parquet snapshots for the session date and the
    prior ~31 days from OCI Object Storage.

    Required by structures/expiry_pin_strike_reversal_structure: the detector
    calls services.option_chain_loader.find_max_oi_strike() which reads
    `data/option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet`. The detector's D-1
    lookup also walks backward up to 7 calendar days when D-1 is missing
    (weekends + NSE holidays), so we need the prior month available too —
    walking 7 days back from a Monday-after-long-weekend can cross a month
    boundary.

    Bucket layout:
        option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet

    Local layout (mirrors repo):
        /app/data/option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet

    A 404 on the entire prefix is logged as WARNING (not fatal) — the
    detector returns `pin strike unavailable from OI snapshot` on missing
    data, which is recoverable. Only the OTHER detectors stay functional.
    """
    log(f"Downloading option_chain OI snapshots for {date_str} (+ prior month)...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)

    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')

    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    # Two month buckets: the session's month + the prior month (handles
    # walk-back across month boundaries).
    months_to_fetch = {
        f"{date_obj.year:04d}/{date_obj.month:02d}",
    }
    # Prior month (handle Jan → previous Dec)
    if date_obj.month == 1:
        months_to_fetch.add(f"{date_obj.year - 1:04d}/12")
    else:
        months_to_fetch.add(f"{date_obj.year:04d}/{date_obj.month - 1:02d}")

    total_downloaded = 0
    total_bytes = 0
    for ym in sorted(months_to_fetch):
        prefix = f"option_chain/{ym}/"
        local_dir = Path(f"/app/data/option_chain/{ym}")
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            # List all objects under the month prefix (~22 parquets per month)
            list_response = os_client.list_objects(
                namespace_name=namespace,
                bucket_name=bucket,
                prefix=prefix,
            )
            objects = list_response.data.objects or []
            if not objects:
                log(f"  (no OI snapshots found under {prefix} — skipping)")
                continue
            for obj in objects:
                object_name = obj.name
                # Filename only (last path segment) — preserves YYYY-MM-DD.parquet
                local_file = local_dir / Path(object_name).name
                if local_file.exists():
                    continue
                try:
                    get_obj = os_client.get_object(
                        namespace_name=namespace,
                        bucket_name=bucket,
                        object_name=object_name,
                    )
                    with open(local_file, 'wb') as f:
                        for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                            f.write(chunk)
                    total_downloaded += 1
                    total_bytes += local_file.stat().st_size
                except Exception as e:
                    log(f"  WARNING: failed to download {object_name}: {e}")
        except oci.exceptions.ServiceError as e:
            if e.status == 404:
                log(f"  WARNING: prefix not found in bucket: {prefix}")
            else:
                log(f"  ERROR listing {prefix}: {e}")
        except Exception as e:
            log(f"  ERROR listing {prefix}: {e}")

    size_mb = total_bytes / (1024 * 1024)
    log(f"Downloaded {total_downloaded} OI snapshots ({size_mb:.1f} MB)")


def generate_analytics(date_str):
    """Regenerate analytics.jsonl from events.jsonl for the given date.

    main.py --dry-run emits events.jsonl but does NOT call populate_analytics_from_events
    at EOD. Without this step, analytics.jsonl contains only inline real-time writes
    which miss every multi-exit trade (T1-partial → trailing-stop), dropping ~50% of
    real trades — specifically, the half that includes winners with aggregated PnL.
    """
    log(f"Generating analytics from events.jsonl for {date_str}...")

    logs_dir = Path('/app/logs')
    session_dirs = sorted(logs_dir.glob(f'bt_{date_str}_*'))
    if not session_dirs:
        log(f"WARNING: No session logs found for analytics generation: {date_str}")
        return

    session_dir = session_dirs[-1]
    session_id = session_dir.name
    events_file = session_dir / 'events.jsonl'

    if not events_file.exists() or events_file.stat().st_size == 0:
        log(f"WARNING: No events.jsonl for analytics generation: {date_str}")
        return

    try:
        sys.path.insert(0, '/app')
        from services.logging.trading_logger import TradingLogger
        logger = TradingLogger(session_id, str(session_dir))
        logger.populate_analytics_from_events()
        analytics_file = session_dir / 'analytics.jsonl'
        size_kb = analytics_file.stat().st_size // 1024 if analytics_file.exists() else 0
        log(f"Analytics generated: {analytics_file.name} ({size_kb} KB)")
    except Exception as e:
        log(f"ERROR generating analytics for {date_str}: {e}")
        import traceback
        traceback.print_exc()


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
            timeout=28800  # 8 hours timeout per day (structure detection ~60-120s/bar × 360 bars)
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


# Whitelist of FILENAMES safe to gzip before upload. Restricted to:
#  1. NEW detector logs that no existing post-processing reads
#  2. agent.log (only consumed for human reading, not as input)
#
# Explicitly NOT gzipped (existing post-processing reads these as plain
# text via process_backtest_run.py + trading_logger.py + comprehensive_run_analyzer.py):
#  - events.jsonl       (consumed by trading_logger to build analytics.jsonl)
#  - analytics.jsonl    (consumed by comprehensive_run_analyzer)
#  - screening.jsonl    (consumed by analysis tools)
#  - planning.jsonl, scanning.jsonl, ranking.jsonl, events_decisions.jsonl
#  - trade_logs.log, timing.jsonl
#  - any .csv (potential analyzer input)
#  - any .json (config dumps; potential metadata)
#
# DuckDB and pandas read .gz files natively, so the gzipped files cause
# no friction for downstream analysis when consumed via those tools.
_GZIP_FILENAMES = {
    "detector_rejections.jsonl",
    "detector_accepts.jsonl",
    "agent.log",
}


def _gzip_file_in_place(file_path: Path) -> Path:
    """Compress a file with gzip and remove the original.

    Returns the new .gz path on success, or the original path if compression
    fails or the file is empty (gzipping empty files is wasteful).
    """
    import gzip
    import shutil

    try:
        if file_path.stat().st_size == 0:
            return file_path
        gz_path = file_path.with_suffix(file_path.suffix + ".gz")
        with open(file_path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out, length=64 * 1024)
        file_path.unlink()
        return gz_path
    except Exception as e:
        log(f"WARNING: gzip failed for {file_path.name}: {e}")
        return file_path


def upload_results(date_str):
    """
    Upload results to OCI Object Storage.

    Text-format files (.jsonl/.log/.csv/.json/.txt) are gzipped in-place
    before upload to cut transfer time + storage cost ~10x for the
    detector_rejections.jsonl + detector_accepts.jsonl files (which can
    be 30-100MB per pod uncompressed). Total bandwidth saved across 120
    pods × 3-yr backtest: ~10-12GB → ~1-2GB.

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

    # Pre-pass: gzip whitelisted files in place. Whitelist limited to NEW
    # detector logs + agent.log to avoid breaking existing post-processing
    # (process_backtest_run.py + trading_logger.py read events.jsonl /
    # analytics.jsonl / screening.jsonl as plain text).
    _t_gzip_start = datetime.now()
    bytes_before = 0
    bytes_after = 0
    gzipped_count = 0
    file_paths_to_upload = []
    for file_path in list(session_dir.rglob('*')):
        if not file_path.is_file():
            continue
        size_before = file_path.stat().st_size
        bytes_before += size_before
        if file_path.name in _GZIP_FILENAMES:
            new_path = _gzip_file_in_place(file_path)
            if new_path != file_path:
                gzipped_count += 1
            file_paths_to_upload.append(new_path)
            bytes_after += new_path.stat().st_size if new_path.exists() else 0
        else:
            file_paths_to_upload.append(file_path)
            bytes_after += size_before
    gzip_secs = (datetime.now() - _t_gzip_start).total_seconds()
    if bytes_before > 0:
        ratio = bytes_after / bytes_before
        log(
            f"Gzipped {gzipped_count} whitelisted files in {gzip_secs:.1f}s | "
            f"{bytes_before / (1024 * 1024):.1f}MB -> {bytes_after / (1024 * 1024):.1f}MB "
            f"(ratio {ratio:.2f})"
        )

    uploaded = 0

    # Upload all files in session directory
    for file_path in file_paths_to_upload:
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

    # Apply OCI config override (sub8_oci_overrides.json -> configuration.json).
    # The override file flips wide_open_mode + 8 sub7+sub8 detector enables on
    # so the OCI capture sees all candidate signals. Without this step, the
    # container would read the production configuration.json as-is (wide_open=
    # false, only gap_fade_short enabled) and the gauntlet would see no signal
    # from the new sub8 detectors. Implemented as a separate utility so the
    # same merge logic is used by local wide-open smoke runs too.
    apply_oci_config_override()

    # Download consolidated daily cache (shared across all dates)
    download_consolidated_daily_cache()

    # Download monthly cache for this specific date
    download_monthly_cache(date_str)

    # Download index OHLCV for directional bias
    download_index_ohlcv()

    # Download option_chain OI snapshots for expiry_pin_strike_reversal.
    # Pulls the session's month + the prior month (~44 parquets, ~65 MB total)
    # so the detector's D-1 walk-back lookup can find the latest available
    # bhavcopy even on Monday-after-long-weekend.
    download_option_chain(date_str)

    # Run backtest
    result = run_backtest(date_str)

    # Regenerate analytics.jsonl from events.jsonl.
    # main.py --dry-run writes events.jsonl but does NOT call populate_analytics_from_events
    # at EOD. Without this step, analytics.jsonl misses every multi-exit trade
    # (T1-partial → trailing-stop), biasing the dataset by ~50% (~7K / 13K trades/session).
    generate_analytics(date_str)

    # Free disk BEFORE upload — cache files are no longer needed after backtest
    # completes. Each pod downloads ~200-500MB of feather cache; with 16-39
    # concurrent pods per node this causes DiskPressure evictions (~60% first-
    # attempt failure rate historically). Deleting cache here reclaims the
    # dominant disk consumer before upload adds more I/O.
    import shutil
    for cache_dir in ['/app/cache', '/app/backtest-cache-download']:
        try:
            if Path(cache_dir).exists():
                size_mb = sum(f.stat().st_size for f in Path(cache_dir).rglob('*') if f.is_file()) / (1024 * 1024)
                shutil.rmtree(cache_dir)
                log(f"DISK_CLEANUP: Deleted {cache_dir} ({size_mb:.0f} MB freed)")
        except Exception as e:
            log(f"WARNING: Failed to clean {cache_dir}: {e}")

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
