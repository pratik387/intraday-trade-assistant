#!/usr/bin/env python3
"""
Check OCI Backtest Job Status
==============================

Quick status check for a backtest job without continuous monitoring.
Useful when kubectl connection might timeout.

Usage:
    python oci_cloud/tools/check_job_status.py <run_id>
    python oci_cloud/tools/check_job_status.py 20251124-023241 --watch

Options:
    --watch: Continuously check status (like monitor, but reconnects)
    --interval N: Check interval in seconds (default: 30)
"""

import argparse
import subprocess
import json
import time
import sys
from datetime import datetime


def get_job_status(run_id):
    """
    Get current job status.

    Args:
        run_id: Backtest run ID

    Returns:
        dict with status info or None if failed
    """
    job_name = f"backtest-{run_id}"

    try:
        result = subprocess.run(
            ['kubectl', 'get', 'job', job_name, '-o', 'json'],
            capture_output=True,
            text=True,
            check=True,
            timeout=30  # Timeout after 30 seconds
        )

        job_status = json.loads(result.stdout)

        # Extract relevant info
        total_days = job_status.get('spec', {}).get('completions', 120)
        succeeded = job_status.get('status', {}).get('succeeded', 0)
        failed = job_status.get('status', {}).get('failed', 0)
        active = job_status.get('status', {}).get('active', 0)

        # Get start time
        start_timestamp = job_status.get('status', {}).get('startTime')
        if not start_timestamp:
            start_timestamp = job_status.get('metadata', {}).get('creationTimestamp')

        return {
            'job_name': job_name,
            'total': total_days,
            'succeeded': succeeded,
            'failed': failed,
            'active': active,
            'start_time': start_timestamp,
            'completed': (succeeded + failed >= total_days)
        }

    except subprocess.TimeoutExpired:
        print(f"Warning: kubectl command timed out (connection issue?)")
        return None
    except subprocess.CalledProcessError as e:
        if 'NotFound' in e.stderr:
            print(f"Error: Job not found: {job_name}")
            print(f"   Check running jobs: kubectl get jobs")
        else:
            print(f"Warning: Error getting job status: {e.stderr}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON response from kubectl")
        return None


def print_status(status):
    """Print formatted status."""
    if not status:
        return

    progress_pct = (status['succeeded'] / status['total']) * 100 if status['total'] > 0 else 0
    progress_bar = '#' * int(progress_pct / 10) + '-' * (10 - int(progress_pct / 10))

    print()
    print(f"Job: {status['job_name']}")
    print(f"Status: {'COMPLETED' if status['completed'] else 'RUNNING'}")
    print()
    print(f"  Running:  {status['active']}")
    print(f"  Complete: {status['succeeded']}/{status['total']}")
    print(f"  Failed:   {status['failed']}")
    print(f"  Progress: {progress_bar} {progress_pct:.1f}%")

    if status['start_time']:
        print(f"  Started:  {status['start_time']}")

    print()


def check_once(run_id):
    """Check status once and exit."""
    print("Checking job status...")
    status = get_job_status(run_id)

    if not status:
        sys.exit(1)

    print_status(status)

    if status['completed']:
        if status['failed'] > 0:
            print(f"Warning: Job completed with {status['failed']} failures")
            print()
            print("Check failed pods:")
            print(f"  kubectl get pods -l run-id={run_id} --field-selector=status.phase=Failed")
        else:
            print("OK: Job completed successfully!")
            print()
            print("Download results:")
            print(f"  python oci_cloud/tools/cleanup_and_download_backtest.py {run_id}")
        print()
    else:
        remaining = status['total'] - status['succeeded'] - status['failed']
        print(f"Job still running ({remaining} days remaining)")
        print()
        print("Continue monitoring:")
        print(f"  python oci_cloud/tools/check_job_status.py {run_id} --watch")
        print()


def watch_status(run_id, interval=30):
    """
    Continuously check status with reconnection handling.

    Args:
        run_id: Backtest run ID
        interval: Check interval in seconds
    """
    print(f"Watching job status (checking every {interval}s)...")
    print("Press Ctrl+C to stop")
    print()

    consecutive_failures = 0
    max_failures = 3

    try:
        while True:
            status = get_job_status(run_id)

            if status:
                consecutive_failures = 0  # Reset on success
                print(f"[{datetime.now().strftime('%H:%M:%S')}]", end=" ")

                if status['completed']:
                    print_status(status)

                    if status['failed'] > 0:
                        print(f"Warning: Job completed with {status['failed']} failures")
                    else:
                        print("OK: Job completed successfully!")
                        print()
                        print("Download results:")
                        print(f"  python oci_cloud/tools/cleanup_and_download_backtest.py {run_id}")

                    break
                else:
                    # Show inline status
                    progress_pct = (status['succeeded'] / status['total']) * 100
                    progress_bar = '#' * int(progress_pct / 10) + '-' * (10 - int(progress_pct / 10))
                    print(f"{status['active']} running, {status['succeeded']}/{status['total']} done  {progress_bar} {progress_pct:.1f}%")

            else:
                consecutive_failures += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Failed to get status (attempt {consecutive_failures}/{max_failures})")

                if consecutive_failures >= max_failures:
                    print()
                    print(f"Error: Failed to get status {max_failures} times in a row")
                    print()
                    print("Possible issues:")
                    print("  - kubectl connection lost (console timeout)")
                    print("  - Job was deleted")
                    print("  - Network issues")
                    print()
                    print("Try reconnecting:")
                    print("  1. Check kubectl config: kubectl get jobs")
                    print("  2. Reconnect to OCI if needed")
                    print(f"  3. Resume watching: python oci_cloud/tools/check_job_status.py {run_id} --watch")
                    sys.exit(1)

            # Wait before next check
            time.sleep(interval)

    except KeyboardInterrupt:
        print()
        print()
        print("Warning: Monitoring stopped")
        print()
        print("Resume watching:")
        print(f"  python oci_cloud/tools/check_job_status.py {run_id} --watch")
        print()
        print("Or check status once:")
        print(f"  python oci_cloud/tools/check_job_status.py {run_id}")
        print()
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Check OCI backtest job status',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status once
  python oci_cloud/tools/check_job_status.py 20251124-023241

  # Watch continuously (reconnects on timeout)
  python oci_cloud/tools/check_job_status.py 20251124-023241 --watch

  # Watch with custom interval
  python oci_cloud/tools/check_job_status.py 20251124-023241 --watch --interval 60
        """
    )

    parser.add_argument('run_id', help='Backtest run ID (e.g., 20251124-023241)')
    parser.add_argument('--watch', action='store_true',
                        help='Continuously check status')
    parser.add_argument('--interval', type=int, default=30,
                        help='Check interval in seconds for --watch mode (default: 30)')

    args = parser.parse_args()

    if args.watch:
        watch_status(args.run_id, args.interval)
    else:
        check_once(args.run_id)


if __name__ == '__main__':
    main()
