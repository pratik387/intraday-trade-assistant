"""
Monitor OCI Kubernetes Backtest Progress
=========================================

Real-time monitoring of running backtest job.

Usage:
    python tools/monitor_oci_backtest.py 20251027_154230
"""

import argparse
import subprocess
import json
import time
import sys
from datetime import datetime


def monitor_job(run_id):
    """Monitor Kubernetes Job progress in real-time"""

    job_name = f"backtest-{run_id}"

    print(f"Monitoring job: {job_name}")
    print()
    print(" Time    Running  Complete  Failed  Progress")
    print("━" * 60)

    start_time = None  # Will get from job metadata
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

            # Get actual job start time from status or metadata (only once)
            if start_time is None:
                # Try status.startTime first (when job controller started processing)
                start_timestamp = job_status.get('status', {}).get('startTime')

                # Fallback to metadata.creationTimestamp (when job object was created)
                if not start_timestamp:
                    start_timestamp = job_status.get('metadata', {}).get('creationTimestamp')

                if start_timestamp:
                    # Parse ISO 8601 timestamp (e.g., "2024-10-27T15:42:30Z")
                    job_start = datetime.strptime(start_timestamp.replace('Z', '+00:00').split('+')[0], '%Y-%m-%dT%H:%M:%S')
                    start_time = job_start.timestamp()
                    print(f"Job started at: {start_timestamp}")
                else:
                    # Fallback to now if no timestamp available
                    start_time = time.time()
                    print(f"Warning: Could not get job start time, using current time")

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

            print(f" {minutes:2d}m {seconds:02d}s    {active:3d}      {succeeded:3d}      {failed:2d}    {progress_bar} {progress_pct:3.0f}%", end='\r')

            # Check if complete
            if succeeded + failed >= total_days:
                print()  # New line
                break

            time.sleep(5)

        except subprocess.CalledProcessError as e:
            if 'NotFound' in e.stderr:
                print(f"\n❌ Job not found: {job_name}")
                print(f"   Check running jobs: kubectl get jobs")
                sys.exit(1)
            else:
                print(f"\n⚠️  Error getting job status: {e.stderr}")
                time.sleep(10)

        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoring interrupted (job still running)")
            print(f"Resume: python tools/monitor_oci_backtest.py {run_id}")
            sys.exit(0)

        except json.JSONDecodeError:
            print(f"\n⚠️  Invalid JSON response, retrying...")
            time.sleep(10)

    print()

    # Final summary
    if failed > 0:
        print(f"\n⚠️  {failed}/{total_days} days failed")
        print(f"\nCheck failed pods:")
        print(f"  kubectl get pods -l run-id={run_id} --field-selector=status.phase=Failed")
        print(f"\nView logs:")
        print(f"  kubectl logs -l run-id={run_id},status=Failed --tail=100")

    if succeeded == total_days:
        elapsed_total = int(time.time() - start_time)
        minutes = elapsed_total // 60
        seconds = elapsed_total % 60

        # Calculate cost
        ocpu_hours = (240 * elapsed_total) / 3600
        cost = ocpu_hours * 0.0015  # $0.0015 per OCPU-hour (Spot)

        print(f"\n✅ All {total_days} days completed!")
        print(f"⏱️  Duration: {minutes}m {seconds}s")
        print(f"💰 Estimated cost: ${cost:.2f}")
        print()
        print(f"Download results:")
        print(f"  python tools/download_oci_results.py {run_id}")


def main():
    parser = argparse.ArgumentParser(description='Monitor OCI Kubernetes backtest')
    parser.add_argument('run_id', help='Run ID (e.g., 20251027_154230)')

    args = parser.parse_args()

    monitor_job(args.run_id)


if __name__ == '__main__':
    main()
