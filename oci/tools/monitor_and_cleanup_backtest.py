#!/usr/bin/env python3
"""
Monitor and Auto-Cleanup OCI Backtest
======================================

Monitors a running backtest job and automatically triggers cleanup
and download when the job completes.

This is a convenience wrapper that combines:
1. monitor_oci_backtest.py - Monitor job progress
2. cleanup_and_download_backtest.py - Download and cleanup

Usage:
    python oci/tools/monitor_and_cleanup_backtest.py <run_id>
    python oci/tools/monitor_and_cleanup_backtest.py 20251121-084341 --parallel 20
    python oci/tools/monitor_and_cleanup_backtest.py 20251121-084341 --skip-nodepool

Arguments:
    run_id: The backtest run ID (e.g., 20251121-084341)

Options:
    --parallel N: Number of parallel downloads (default: 10)
    --skip-nodepool: Skip scaling down node pool
    --keep-oci-files: Don't delete files from OCI bucket after download
    --keep-extracted: Don't delete local extracted directory after zipping
    --monitor-only: Only monitor, don't run cleanup

Example:
    python oci/tools/monitor_and_cleanup_backtest.py 20251121-084341
"""

import argparse
import subprocess
import json
import time
import sys
from datetime import datetime
from pathlib import Path


def monitor_job(run_id):
    """
    Monitor Kubernetes Job progress in real-time.

    Args:
        run_id: Backtest run ID

    Returns:
        True if job completed successfully, False otherwise
    """
    job_name = f"backtest-{run_id}"

    print()
    print("=" * 80)
    print("MONITORING BACKTEST JOB")
    print("=" * 80)
    print(f"Job: {job_name}")
    print(f"Run ID: {run_id}")
    print("=" * 80)
    print()
    print("Note: kubectl connection may timeout after ~1 hour")
    print(f"   For long jobs, use: python oci/tools/check_job_status.py {run_id} --watch")
    print()
    print(" Time    Running  Complete  Failed  Progress")
    print("━" * 60)

    start_time = None
    total_days = None

    while True:
        try:
            # Get job status
            result = subprocess.run(
                ['kubectl', 'get', 'job', job_name, '-o', 'json'],
                capture_output=True,
                text=True,
                check=True
            )

            job_status = json.loads(result.stdout)

            # Get job start time
            if start_time is None:
                start_timestamp = job_status.get('status', {}).get('startTime')
                if not start_timestamp:
                    start_timestamp = job_status.get('metadata', {}).get('creationTimestamp')

                if start_timestamp:
                    job_start = datetime.strptime(
                        start_timestamp.replace('Z', '+00:00').split('+')[0],
                        '%Y-%m-%dT%H:%M:%S'
                    )
                    start_time = job_start.timestamp()
                else:
                    start_time = time.time()

            # Get totals
            if total_days is None:
                total_days = job_status.get('spec', {}).get('completions', 120)

            succeeded = job_status.get('status', {}).get('succeeded', 0)
            failed = job_status.get('status', {}).get('failed', 0)
            active = job_status.get('status', {}).get('active', 0)

            elapsed = int(time.time() - start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60

            progress_pct = (succeeded / total_days) * 100 if total_days > 0 else 0
            progress_bar = '▰' * int(progress_pct / 10) + '▱' * (10 - int(progress_pct / 10))

            print(
                f" {minutes:2d}m {seconds:02d}s    {active:3d}      {succeeded:3d}      {failed:2d}    "
                f"{progress_bar} {progress_pct:3.0f}%",
                end='\r'
            )

            # Check if complete
            if succeeded + failed >= total_days:
                print()
                break

            time.sleep(5)

        except subprocess.CalledProcessError as e:
            if 'NotFound' in e.stderr:
                print(f"\n❌ Job not found: {job_name}")
                print(f"   Check running jobs: kubectl get jobs")
                return False
            else:
                print(f"\n⚠️  Error getting job status: {e.stderr}")
                time.sleep(10)

        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoring interrupted (job still running)")
            print(f"\nResume monitoring:")
            print(f"  python oci/tools/monitor_and_cleanup_backtest.py {run_id}")
            print(f"\nOr download manually when job completes:")
            print(f"  python oci/tools/cleanup_and_download_backtest.py {run_id}")
            sys.exit(0)

        except json.JSONDecodeError:
            print(f"\n⚠️  Invalid JSON response, retrying...")
            time.sleep(10)

    print()

    # Final summary
    if failed > 0:
        print()
        print(f"⚠️  {failed}/{total_days} days failed")
        print()
        print("Check failed pods:")
        print(f"  kubectl get pods -l run-id={run_id} --field-selector=status.phase=Failed")
        print()
        print("View logs:")
        print(f"  kubectl logs -l run-id={run_id},status=Failed --tail=100")
        print()

    if succeeded == total_days:
        elapsed_total = int(time.time() - start_time)
        minutes = elapsed_total // 60
        seconds = elapsed_total % 60

        # Calculate cost
        ocpu_hours = (240 * elapsed_total) / 3600
        cost = ocpu_hours * 0.0015

        print()
        print(f"✅ All {total_days} days completed!")
        print(f"⏱️  Duration: {minutes}m {seconds}s")
        print(f"💰 Estimated cost: ${cost:.2f}")
        print()

        return True

    return False


def run_cleanup(run_id, args):
    """
    Run cleanup and download automation.

    Args:
        run_id: Backtest run ID
        args: Command-line arguments
    """
    print()
    print("=" * 80)
    print("STARTING CLEANUP & DOWNLOAD")
    print("=" * 80)
    print()

    # Build command
    script_path = Path(__file__).parent / "cleanup_and_download_backtest.py"
    cmd = [sys.executable, str(script_path), run_id]

    if args.parallel:
        cmd.extend(['--parallel', str(args.parallel)])
    if args.skip_nodepool:
        cmd.append('--skip-nodepool')
    if args.keep_oci_files:
        cmd.append('--keep-oci-files')
    if args.keep_extracted:
        cmd.append('--keep-extracted')

    print(f"Running: {' '.join(cmd)}")
    print()

    # Run cleanup script
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print()
        print("❌ Cleanup failed")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor backtest job and auto-cleanup when complete',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor and auto-cleanup
  python oci/tools/monitor_and_cleanup_backtest.py 20251121-084341

  # With custom parallel downloads
  python oci/tools/monitor_and_cleanup_backtest.py 20251121-084341 --parallel 20

  # Skip node pool scaling
  python oci/tools/monitor_and_cleanup_backtest.py 20251121-084341 --skip-nodepool

  # Only monitor (no cleanup)
  python oci/tools/monitor_and_cleanup_backtest.py 20251121-084341 --monitor-only
        """
    )

    parser.add_argument('run_id', help='Backtest run ID (e.g., 20251121-084341)')
    parser.add_argument('--parallel', type=int, default=10,
                        help='Number of parallel downloads (default: 10)')
    parser.add_argument('--skip-nodepool', action='store_true',
                        help='Skip scaling down node pool')
    parser.add_argument('--keep-oci-files', action='store_true',
                        help="Don't delete files from OCI bucket after download")
    parser.add_argument('--keep-extracted', action='store_true',
                        help="Don't delete local extracted directory after zipping")
    parser.add_argument('--monitor-only', action='store_true',
                        help='Only monitor, do not run cleanup')

    args = parser.parse_args()

    # Step 1: Monitor job
    success = monitor_job(args.run_id)

    if not success:
        # monitor_job already printed appropriate message (interrupted or failed)
        # Just exit without additional error message
        sys.exit(1)

    # Step 2: Run cleanup (unless --monitor-only)
    if args.monitor_only:
        print()
        print("SKIPPED: Cleanup (--monitor-only)")
        print()
        print("To download results manually:")
        print(f"  python oci/tools/cleanup_and_download_backtest.py {args.run_id}")
        print()
        sys.exit(0)

    run_cleanup(args.run_id, args)

    print()
    print("=" * 80)
    print("ALL DONE!")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
