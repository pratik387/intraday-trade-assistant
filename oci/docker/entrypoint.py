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


# download_iv_rank() removed 2026-05-14 with options_vol_iv_rank_revert retire
# (see docs/retired_setups.md).


def download_delivery_pct():
    """Download NSE delivery-percentage history parquet for the sub-9
    delivery_pct_anomaly_short detector.

    Single file (~39 MB) at `delivery_pct/delivery_history.parquet`.
    Loaded once at screener daily-cache seed by
    `services/delivery_pct_enrichment.py` and merged into each
    symbol's daily_df as the `delivery_pct` column.

    Built locally by tools/delivery_pct/fetch_delivery.py from NSE
    archive (MTO + PR daily files). Uploaded via
    oci/tools/upload_delivery_pct.py.

    Bucket layout:
        delivery_pct/delivery_history.parquet

    Per-date 404 (file missing) is logged at warning — the detector
    tolerates absence (silently no-fires every signal) but the run
    won't generate any delivery_pct_anomaly_short fires.
    """
    log("Downloading delivery_pct history parquet for delivery_pct_anomaly_short...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)
    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')

    object_name = "delivery_pct/delivery_history.parquet"
    local_dir = Path('/app/data/delivery_pct')
    local_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_dir / Path(object_name).name

    try:
        get_obj = os_client.get_object(
            namespace_name=namespace,
            bucket_name=bucket,
            object_name=object_name,
        )
        with open(local_file, 'wb') as f:
            for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)
        size_mb = local_file.stat().st_size / (1024 * 1024)
        log(f"Downloaded {object_name}: {size_mb:.1f} MB")
    except oci.exceptions.ServiceError as e:
        if e.status == 404:
            log(f"WARNING: {object_name} not found in OCI cache; "
                "delivery_pct_anomaly_short detector will silently skip every bar. "
                "Run oci/tools/upload_delivery_pct.py to upload.")
        else:
            log(f"ERROR downloading {object_name}: {e}")


def download_mis_short_eligibility():
    """Download the MIS-short eligibility map (broker intraday leverage>1) for
    up_spike_fade_short's universe gate.

    Small JSON (~30 KB) at `mis_short_eligibility/latest.json`. Built locally by
    jobs/refresh_mis_short_eligibility.py (daily Kite order_margins sweep) and
    uploaded via oci/tools/upload_mis_short_eligibility.py. Consumed by
    services/setup_universe.py::up_spike_fade_short_universe (admits only
    symbols with broker intraday SELL/MIS leverage > 1).

    If absent, up_spike_fade_short_universe fail-closes to an EMPTY universe
    (require_short_eligibility_map=true) -> zero up_spike fires. Logged at
    warning; panic_crash_revert_long is unaffected (no eligibility dependency).
    """
    log("Downloading MIS-short eligibility map for up_spike_fade_short...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)
    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')

    object_name = "mis_short_eligibility/latest.json"
    local_dir = Path('/app/data/mis_short_eligibility')
    local_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_dir / "latest.json"

    try:
        get_obj = os_client.get_object(
            namespace_name=namespace,
            bucket_name=bucket,
            object_name=object_name,
        )
        with open(local_file, 'wb') as f:
            for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)
        size_kb = local_file.stat().st_size / 1024
        log(f"Downloaded {object_name}: {size_kb:.1f} KB")
    except oci.exceptions.ServiceError as e:
        if e.status == 404:
            log(f"WARNING: {object_name} not found in OCI cache; "
                "up_spike_fade_short universe fail-closes to empty (zero fires). "
                "Run oci/tools/upload_mis_short_eligibility.py to upload.")
        else:
            log(f"ERROR downloading {object_name}: {e}")


def download_cross_day_rvol():
    """Look up the precomputed cross-day RVOL baseline parquet from OCI cache.

    Single static file (~77 MB) at `cross_day_rvol/rvol_baseline.parquet`.
    Built once locally by tools/cross_day_rvol/build_baseline.py and
    uploaded via oci/tools/upload_cross_day_rvol.py — same data for
    every backtest run since the baseline is a function of historical
    5m bars, which never change.

    Loaded by `services/cross_day_rvol_enrichment.py` and queried
    bar-by-bar by the detector to compute RVOL = today_bar_vol /
    cached_baseline_at_(symbol, date, hhmm).

    Why this exists: structures/delivery_pct_anomaly_short_structure.py's
    _cross_day_rvol() needs 20 prior-session same-tod 5m bars. The
    screener caps df_5m at screener_store_5m_max=120 (~1.5 days), so
    the in-detector calculation always returned None. Verified in OCI
    run 20260508-230433_full: 0 fires across 501 Discovery days vs
    sanity's 261 expected.

    Uses the canonical _download_oci_file helper + broad except so a
    transient bucket / network error never crashes the pod. The detector
    auto-skips when the file is absent, so the worst case is "no
    delivery_pct fires for this run" not "pod failure".
    """
    log("Looking up cross_day_rvol baseline from OCI cache...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)
    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')

    object_name = "cross_day_rvol/rvol_baseline.parquet"
    local_path = Path('/app/data/cross_day_rvol') / Path(object_name).name

    try:
        ok = _download_oci_file(os_client, namespace, bucket, object_name, local_path)
        if not ok:
            log(f"WARNING: {object_name} not in OCI cache; "
                "delivery_pct_anomaly_short will reject every signal "
                "with 'cross-day RVOL unavailable'. Run "
                "oci/tools/upload_cross_day_rvol.py to upload.")
    except Exception as e:
        log(f"ERROR looking up {object_name} (continuing without it): {e}")


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
    Download NIFTY 50 index OHLCV feathers (1m + daily) from OCI Object Storage.

    Required by:
      - DirectionalBiasTracker (1m): backtest Nifty price lookups
      - MarketRegimeGate.compute_regime_multi_tf (daily): broad-market regime
        — used by EVERY active setup via ctx.regime in detect()

    Bucket paths:
      index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1minutes.feather
      index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1days.feather
    """
    log("Downloading NIFTY 50 OHLCV (1m + daily) for directional bias + regime gate...")

    config = oci.config.from_file()
    os_client = oci.object_storage.ObjectStorageClient(config)
    namespace = os_client.get_namespace().data
    bucket = os.environ.get('OCI_BUCKET_CACHE', 'backtest-cache')
    local_dir = Path('/app/backtest-cache-download/index_ohlcv/NSE_NIFTY_50')
    local_dir.mkdir(parents=True, exist_ok=True)

    for object_name, local_name in [
        ("index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1minutes.feather", "NSE_NIFTY_50_1minutes.feather"),
        ("index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1days.feather",    "NSE_NIFTY_50_1days.feather"),
    ]:
        local_file = local_dir / local_name
        try:
            get_obj = os_client.get_object(
                namespace_name=namespace,
                bucket_name=bucket,
                object_name=object_name,
            )
            with open(local_file, 'wb') as f:
                for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                    f.write(chunk)
            size_mb = local_file.stat().st_size / (1024 * 1024)
            log(f"Downloaded {object_name}: {size_mb:.2f} MB")
        except oci.exceptions.ServiceError as e:
            if e.status == 404:
                log(f"WARNING: {object_name} not found in OCI; regime/directional features degraded")
            else:
                log(f"ERROR downloading {object_name}: {e}")
                raise
        except Exception as e:
            log(f"ERROR downloading {object_name}: {e}")
            raise


# download_earnings_calendar() and download_option_chain() removed 2026-05-14
# with earnings_day_intraday_fade + expiry_pin_strike_reversal retire
# (see docs/retired_setups.md). No active setup consumes either feed.


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

    # Switch to OCI research mode: enables wide_open_mode + all candidate detectors
    # via the mode_profiles.oci_research block in configuration.json.
    os.environ["RUN_MODE"] = "oci_research"
    log("OCI mode: set RUN_MODE=oci_research")

    # Download consolidated daily cache (shared across all dates)
    download_consolidated_daily_cache()

    # Download monthly cache for this specific date
    download_monthly_cache(date_str)

    # Download index OHLCV for directional bias
    download_index_ohlcv()

    # download_option_chain / download_iv_rank / download_earnings_calendar
    # removed 2026-05-14 — no active setup consumes those feeds. The
    # corresponding setups (expiry_pin_strike_reversal, options_vol_iv_rank_revert,
    # earnings_day_intraday_fade) are retired. See docs/retired_setups.md.

    # Download NSE delivery-percentage history parquet for
    # delivery_pct_anomaly_short (~39 MB, single file). Detector auto-skips
    # if absent — sub9 round-8 ship.
    download_delivery_pct()

    # Download precomputed cross-day RVOL baseline parquet for
    # delivery_pct_anomaly_short (~77 MB, single file). Required because
    # screener_store_5m_max=120 means df_5m never has 20 prior same-tod
    # bars for the in-detector RVOL calculation. Detector rejects with
    # "cross-day RVOL unavailable" if absent — every signal blocked.
    download_cross_day_rvol()

    # Download MIS-short eligibility map (broker leverage>1) for
    # up_spike_fade_short's universe gate (~30 KB). Universe fail-closes to
    # empty (zero up_spike fires) if absent.
    download_mis_short_eligibility()

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
