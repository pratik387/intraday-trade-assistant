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


def get_next_retry_number(base_run_id, project_root):
    """
    Find the next retry number for a given base run ID.

    Mirrors OCIBacktestSubmitter._get_next_retry_number logic.

    Args:
        base_run_id: The original run ID (without -retry suffix)
        project_root: Project root path

    Returns:
        Next retry number (1 if no retries exist)
    """
    cloud_results = project_root / 'cloud_results'
    if not cloud_results.exists():
        return 1

    existing_retries = []
    for d in cloud_results.iterdir():
        if d.is_dir() and d.name.startswith(base_run_id):
            if '-retry' in d.name:
                try:
                    retry_part = d.name.split('-retry')[-1]
                    retry_num = int(retry_part)
                    existing_retries.append(retry_num)
                except ValueError:
                    pass

    if not existing_retries:
        return 1

    return max(existing_retries) + 1


def submit_retry_job(failed_dates_file, project_root):
    """
    Submit a retry job for failed dates using submit_oci_backtest.py --no-wait.

    Nodes are assumed to be already running (we haven't scaled down yet).

    Args:
        failed_dates_file: Path to failed_dates.json
        project_root: Project root path

    Returns:
        True if submission succeeded, False otherwise
    """
    print()
    print("=" * 80)
    print("AUTO-RETRY: Re-running failed dates")
    print("=" * 80)
    print()

    submit_script = project_root / 'oci' / 'tools' / 'submit_oci_backtest.py'
    cmd = [sys.executable, str(submit_script),
           '--failed-dates', str(failed_dates_file),
           '--no-wait']

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print()
        print("❌ Auto-retry submission failed")
        return False

    return True


def get_failed_dates_from_pods(run_id, dates_list):
    """
    Get the list of dates that failed by mapping failed pod indices to dates.

    Per Kubernetes docs for Indexed Jobs:
    - Each pod gets annotation: batch.kubernetes.io/job-completion-index
    - With restartPolicy: Never and backoffLimit > 0, multiple failed pods
      may exist for the same index (one per retry attempt)
    - We deduplicate to get unique failed indices

    Args:
        run_id: Backtest run ID
        dates_list: List of dates (ordered by index)

    Returns:
        List of unique failed date strings (sorted)
    """
    failed_indices = set()  # Use set to deduplicate (multiple pods per index due to retries)

    try:
        # Get all failed pods for this job
        # Note: With backoffLimit=3, each index can have up to 3 failed pods
        result = subprocess.run(
            ['kubectl', 'get', 'pods', '-l', f'run-id={run_id}',
             '--field-selector=status.phase=Failed', '-o', 'json'],
            capture_output=True,
            text=True,
            check=True
        )

        pods = json.loads(result.stdout).get('items', [])

        for pod in pods:
            # Get the pod's job completion index from annotation
            # Per K8s docs: batch.kubernetes.io/job-completion-index is set by Job controller
            annotations = pod.get('metadata', {}).get('annotations', {})
            index_str = annotations.get('batch.kubernetes.io/job-completion-index')

            if index_str is not None:
                try:
                    index = int(index_str)
                    if 0 <= index < len(dates_list):
                        failed_indices.add(index)  # Set handles duplicates
                except ValueError:
                    pass

    except subprocess.CalledProcessError:
        pass
    except json.JSONDecodeError:
        pass

    # Convert indices to dates and sort
    failed_dates = [dates_list[i] for i in sorted(failed_indices)]
    return failed_dates


def get_dates_list_from_job(run_id):
    """
    Extract the DATES_LIST from the job spec.

    Args:
        run_id: Backtest run ID

    Returns:
        List of date strings or empty list
    """
    try:
        result = subprocess.run(
            ['kubectl', 'get', 'job', f'backtest-{run_id}', '-o', 'json'],
            capture_output=True,
            text=True,
            check=True
        )

        job = json.loads(result.stdout)
        containers = job.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])

        for container in containers:
            for env in container.get('env', []):
                if env.get('name') == 'DATES_LIST':
                    dates_csv = env.get('value', '')
                    return dates_csv.split(',') if dates_csv else []

        return []

    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return []


def get_missing_dates_from_bucket(run_id, dates_list):
    """
    Find dates that are missing from the results bucket.

    This is more reliable than checking failed pods because:
    - Pods that never ran won't show up as 'Failed'
    - The bucket shows what actually completed successfully

    Args:
        run_id: Backtest run ID
        dates_list: List of expected dates (ordered by index)

    Returns:
        List of missing date strings (sorted)
    """
    try:
        import oci
        config = oci.config.from_file()
        os_client = oci.object_storage.ObjectStorageClient(config)
        namespace = os_client.get_namespace().data

        # List ALL objects with pagination
        all_objects = []
        next_start = None
        while True:
            response = os_client.list_objects(
                namespace_name=namespace,
                bucket_name='backtest-results',
                prefix=f'{run_id}/',
                start=next_start,
                limit=1000
            )
            all_objects.extend(response.data.objects)
            next_start = response.data.next_start_with
            if not next_start:
                break

        # Get completed dates from bucket
        completed_dates = set()
        for obj in all_objects:
            parts = obj.name.split('/')
            if len(parts) >= 2 and parts[1] and '-' in parts[1]:
                completed_dates.add(parts[1])

        # Find missing dates
        expected_set = set(dates_list)
        missing = sorted([d for d in expected_set if d not in completed_dates])

        return missing

    except Exception as e:
        print(f"⚠️  Could not check bucket for missing dates: {e}")
        return []


def save_failed_dates(run_id, failed_dates, project_root):
    """
    Save failed dates to a file for later re-run.

    Args:
        run_id: Backtest run ID
        failed_dates: List of failed date strings
        project_root: Project root path
    """
    if not failed_dates:
        return None

    failed_dates_file = project_root / 'cloud_results' / run_id / 'failed_dates.json'
    failed_dates_file.parent.mkdir(parents=True, exist_ok=True)

    with open(failed_dates_file, 'w') as f:
        json.dump({
            'run_id': run_id,
            'failed_dates': failed_dates,
            'count': len(failed_dates),
            'saved_at': datetime.now().isoformat()
        }, f, indent=2)

    return failed_dates_file


def monitor_job(run_id):
    """
    Monitor Kubernetes Job progress in real-time.

    Args:
        run_id: Backtest run ID

    Returns:
        dict with keys:
            - completed: True if job finished (success or failure)
            - succeeded: Number of successful pods
            - failed: Number of failed pods
            - total: Total expected completions
            - failed_dates: List of dates that failed (empty if all succeeded)
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
    dates_list = []

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
                    # Parse UTC timestamp from Kubernetes
                    job_start = datetime.strptime(
                        start_timestamp.replace('Z', '+00:00').split('+')[0],
                        '%Y-%m-%dT%H:%M:%S'
                    )
                    # Convert to Unix timestamp (UTC) - add timezone info
                    from datetime import timezone
                    start_time = job_start.replace(tzinfo=timezone.utc).timestamp()
                else:
                    start_time = time.time()

            # Get totals and dates list
            if total_days is None:
                total_days = job_status.get('spec', {}).get('completions', 120)
                dates_list = get_dates_list_from_job(run_id)

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

            # Check if complete - use job conditions, not failed pod count
            # failed count includes retries, so succeeded + failed can exceed total_days
            conditions = job_status.get('status', {}).get('conditions', [])
            is_complete = any(
                c.get('type') == 'Complete' and c.get('status') == 'True'
                for c in conditions
            )
            is_failed = any(
                c.get('type') == 'Failed' and c.get('status') == 'True'
                for c in conditions
            )

            if is_complete or is_failed:
                print()
                break

            # Fallback: if no active pods and all indices accounted for
            # Note: failed count includes retries, so we need to check unique failed indices
            if active == 0 and succeeded > 0:
                # Get unique failed indices count
                failed_dates = get_failed_dates_from_pods(run_id, dates_list)
                unique_failed = len(failed_dates)
                if succeeded + unique_failed >= total_days:
                    # All indices are accounted for
                    time.sleep(5)  # Wait to confirm no new pods scheduled
                    print()
                    break

            time.sleep(5)

        except subprocess.CalledProcessError as e:
            if 'NotFound' in e.stderr:
                print(f"\n❌ Job not found: {job_name}")
                print(f"   Check running jobs: kubectl get jobs")
                return {'completed': False, 'succeeded': 0, 'failed': 0, 'total': 0, 'failed_dates': []}
            else:
                print(f"\n⚠️  Error getting job status: {e.stderr}")
                time.sleep(10)

        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoring interrupted (job still running)")
            print(f"\nResume monitoring:")
            print(f"  python oci/tools/monitor_and_cleanup_backtest.py {run_id}")
            print(f"\nOr download manually when job completes:")
            print(f"  python oci/tools/cleanup_and_download_backtest.py {run_id}")
            # Return partial status - important: still trigger cleanup!
            return {'completed': False, 'interrupted': True, 'succeeded': 0, 'failed': 0, 'total': 0, 'failed_dates': []}

        except json.JSONDecodeError:
            print(f"\n⚠️  Invalid JSON response, retrying...")
            time.sleep(10)

    print()

    # Get missing dates by checking the bucket (more reliable than pod status)
    # This catches both failed pods AND pods that never ran
    print("Checking bucket for missing dates...")
    missing_dates = get_missing_dates_from_bucket(run_id, dates_list)

    # Final summary
    elapsed_total = int(time.time() - start_time) if start_time else 0
    minutes = elapsed_total // 60
    seconds = elapsed_total % 60

    # Calculate cost
    ocpu_hours = (240 * elapsed_total) / 3600
    cost = ocpu_hours * 0.0015

    completed_count = total_days - len(missing_dates)

    print()
    if missing_dates:
        print(f"⚠️  {len(missing_dates)}/{total_days} days missing/failed")
        print(f"✅ {completed_count}/{total_days} days completed")

        print()
        print(f"Missing dates ({len(missing_dates)}):")
        for date in missing_dates[:10]:  # Show first 10
            print(f"  - {date}")
        if len(missing_dates) > 10:
            print(f"  ... and {len(missing_dates) - 10} more")
        print()
        print("Check failed pods:")
        print(f"  kubectl get pods -l run-id={run_id} --field-selector=status.phase=Failed")
        print()
        print("View logs:")
        print(f"  kubectl logs -l run-id={run_id},status=Failed --tail=100")
    else:
        print(f"✅ All {total_days} days completed!")

    print(f"⏱️  Duration: {minutes}m {seconds}s")
    print(f"💰 Estimated cost: ${cost:.2f}")
    print()

    return {
        'completed': True,
        'succeeded': completed_count,
        'failed': len(missing_dates),
        'total': total_days,
        'failed_dates': missing_dates  # Now contains ALL missing dates, not just failed pods
    }


def get_base_run_id(run_id):
    """
    Extract base run ID from a potentially retry-suffixed run ID.

    Args:
        run_id: Run ID (e.g., '20260120-194012' or '20260120-194012-retry1')

    Returns:
        Base run ID without -retry suffix (e.g., '20260120-194012')
    """
    if '-retry' in run_id:
        return run_id.split('-retry')[0]
    return run_id


def run_cleanup(run_id, args):
    """
    Run cleanup and download automation.

    Always uses base run ID to download ALL related runs (original + retries).

    Args:
        run_id: Backtest run ID (can include -retry suffix)
        args: Command-line arguments
    """
    print()
    print("=" * 80)
    print("STARTING CLEANUP & DOWNLOAD")
    print("=" * 80)
    print()

    # Always use base run ID to download all related runs (original + retries)
    base_run_id = get_base_run_id(run_id)
    if base_run_id != run_id:
        print(f"Note: Using base run ID '{base_run_id}' to download all related runs")
        print()

    # Build command
    script_path = Path(__file__).parent / "cleanup_and_download_backtest.py"
    cmd = [sys.executable, str(script_path), base_run_id]

    if args.parallel:
        cmd.extend(['--parallel', str(args.parallel)])
    if args.skip_nodepool:
        cmd.append('--skip-nodepool')
    if args.keep_oci_files:
        cmd.append('--keep-oci-files')
    if args.keep_extracted:
        cmd.append('--keep-extracted')
    if hasattr(args, 'local') and args.local:
        cmd.append('--local')

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

  # Force cleanup even on failure (default behavior now)
  python oci/tools/monitor_and_cleanup_backtest.py 20251121-084341 --force-cleanup
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
    parser.add_argument('--force-cleanup', action='store_true', default=True,
                        help='Run cleanup even if job has failures (default: True)')
    parser.add_argument('--skip-download-on-failure', action='store_true',
                        help='Skip downloading results if job has failures (still scales down nodes)')
    parser.add_argument('--no-auto-retry', action='store_true',
                        help='Disable automatic retry of failed dates (default: auto-retries once)')
    parser.add_argument('--local', action='store_true',
                        help='Local workflow: download all (original + retries), process, generate report (no zip)')

    args = parser.parse_args()

    # Step 1: Monitor job
    result = monitor_job(args.run_id)

    # Get project root for saving failed dates
    project_root = Path(__file__).parent.parent.parent

    # Handle interrupted monitoring
    if result.get('interrupted'):
        print()
        print("⚠️  Monitoring was interrupted but job may still be running")
        print("   Nodes will NOT be scaled down to avoid disrupting running job")
        print()
        sys.exit(1)

    # Extract base run ID for file paths and related-run operations
    base_run_id = get_base_run_id(args.run_id)

    # Save failed dates if any
    if result.get('failed_dates'):
        failed_dates_file = save_failed_dates(base_run_id, result['failed_dates'], project_root)
        if failed_dates_file:
            print()
            print(f"📝 Failed dates saved to: {failed_dates_file}")
            print(f"   Re-run failed dates: python oci/tools/submit_oci_backtest.py --failed-dates {failed_dates_file}")
            print()

    # Step 2: Cleanup (ALWAYS run to avoid cost overruns)
    # The key change: cleanup happens even on failure!
    if args.monitor_only:
        print()
        print("SKIPPED: Cleanup (--monitor-only)")
        print()
        print("⚠️  WARNING: Nodes are still running and incurring costs!")
        print("   Scale down manually: oci ce node-pool update --node-pool-id <id> --size 0 --force")
        print()
        print("To download results manually (includes original + retries):")
        print(f"  python oci/tools/cleanup_and_download_backtest.py {base_run_id}")
        print()
        sys.exit(0)

    # Check if job has failures
    has_failures = result.get('failed', 0) > 0

    # If job has failures, attempt auto-retry once before giving up
    if has_failures and result.get('completed') and not args.no_auto_retry:
        failed_dates_file = project_root / 'cloud_results' / base_run_id / 'failed_dates.json'

        # Compute retry run ID before submitting (must match what submit script generates)
        retry_num = get_next_retry_number(base_run_id, project_root)
        retry_run_id = f"{base_run_id}-retry{retry_num}"

        print()
        print(f"⚠️  {result.get('failed', 0)} dates failed — auto-retrying once before scaling down...")
        print(f"   Retry run ID: {retry_run_id}")

        retry_submitted = submit_retry_job(failed_dates_file, project_root)

        if retry_submitted:
            # Monitor the retry job
            print()
            retry_result = monitor_job(retry_run_id)

            # Save retry failed dates if any
            if retry_result.get('failed_dates'):
                retry_failed_file = save_failed_dates(retry_run_id, retry_result['failed_dates'], project_root)
                if retry_failed_file:
                    print(f"📝 Retry failed dates saved to: {retry_failed_file}")
                # Also update base run's failed_dates.json so manual suggestions point to remaining failures
                save_failed_dates(base_run_id, retry_result['failed_dates'], project_root)

            retry_has_failures = retry_result.get('failed', 0) > 0

            if not retry_has_failures and retry_result.get('completed'):
                # Retry fixed everything — run full cleanup (downloads original + retry)
                print()
                print("✅ Auto-retry succeeded — all failed dates recovered!")
                run_cleanup(args.run_id, args)

                print()
                print("=" * 80)
                print("ALL DONE!")
                print("=" * 80)
                print()
                sys.exit(0)
            else:
                # Retry still has failures — update result for the scale-down path below
                print()
                print(f"⚠️  Auto-retry still has {retry_result.get('failed', 0)} missing dates")
                result = retry_result
                has_failures = True

    # If job has failures OR didn't complete properly, scale down nodes
    if has_failures or not result.get('completed'):
        print()
        if has_failures:
            print(f"⚠️  Job has {result.get('failed', 0)} missing/failed dates")
        else:
            print("⚠️  Job did not complete properly")
        print("   Scaling down nodes to save costs (skipping download)...")
        print()

        # Scale down nodes only
        try:
            import oci
            config = oci.config.from_file()
            container_engine_client = oci.container_engine.ContainerEngineClient(config)
            node_pool_id = "ocid1.nodepool.oc1.ap-mumbai-1.aaaaaaaaqs7a4f5jyyhcy3dsmedknnzbmhpdmdj6dqkastv5cnaehilq5g3q"

            update_details = oci.container_engine.models.UpdateNodePoolDetails(
                node_config_details=oci.container_engine.models.UpdateNodePoolNodeConfigDetails(
                    size=0
                )
            )
            container_engine_client.update_node_pool(
                node_pool_id=node_pool_id,
                update_node_pool_details=update_details
            )
            print("✅ Node pool scaled down to 0")
        except Exception as e:
            print(f"❌ Failed to scale down: {e}")
            print("   Please scale down manually!")

        print()
        print("=" * 80)
        print(f"JOB INCOMPLETE - {result.get('failed', 0)} DATES MISSING")
        print("=" * 80)
        print()
        print("To download completed results (original + retries):")
        print(f"  python oci/tools/cleanup_and_download_backtest.py {base_run_id} --skip-nodepool")
        print()
        print("To re-run missing dates:")
        failed_dates_file = project_root / 'cloud_results' / base_run_id / 'failed_dates.json'
        print(f"  python oci/tools/submit_oci_backtest.py --failed-dates {failed_dates_file}")
        print()
        sys.exit(1)

    # Run full cleanup (includes scale down + download) - only if ALL succeeded
    run_cleanup(args.run_id, args)

    print()
    print("=" * 80)
    print("ALL DONE!")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
