"""
Submit Parallel Backtest to Oracle Cloud Kubernetes
====================================================

Workflow:
1. Package current code (config + Python files)
2. Upload to OCI Object Storage
3. Submit Kubernetes Job with 120 parallel pods
4. Monitor progress in real-time
5. Download results when complete

Usage:
    python tools/submit_oci_backtest.py --start 2024-05-01 --end 2024-10-31
    python tools/submit_oci_backtest.py --start 2024-05-01 --end 2024-10-31 --description "Test variant A"
    python tools/submit_oci_backtest.py --start 2024-05-01 --end 2024-10-31 --no-wait
"""

import argparse
import subprocess
import tarfile
import time
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import sys
import oci
from oci.object_storage import ObjectStorageClient
import pandas as pd

# Project root
ROOT = Path(__file__).parent.parent.parent
HOLIDAY_FILE = ROOT / "assets" / "nse_holidays.json"


def is_trading_day(date):
    """
    Returns True if the given date is a valid NSE trading day (not weekend, not holiday).
    Self-contained version for OCI tools to avoid circular imports.
    """
    try:
        dt = pd.Timestamp(date).normalize()

        # Weekend check
        if dt.weekday() >= 5:
            return False

        # Load holidays
        if not HOLIDAY_FILE.exists():
            print(f"Warning: Holiday file not found at {HOLIDAY_FILE}, assuming all weekdays are trading days")
            return True

        with open(HOLIDAY_FILE, "r", encoding="utf-8") as f:
            items = json.load(f)
            holidays = [
                pd.to_datetime(item.get("tradingDate") or item.get("holidayDate"), format="%d-%b-%Y", errors="coerce").normalize()
                for item in items
            ]
            holidays = [d for d in holidays if not pd.isna(d)]

        return dt not in holidays

    except Exception as e:
        print(f"Warning: is_trading_day error: {e}, assuming trading day")
        return True


class OCIBacktestSubmitter:
    def __init__(self):
        """Initialize OCI clients"""
        self.config = oci.config.from_file()
        self.namespace = self._get_namespace()
        self.os_client = ObjectStorageClient(self.config)
        self.container_engine_client = oci.container_engine.ContainerEngineClient(self.config)
        self.region = self.config['region']

        # Bucket names
        self.code_bucket = 'backtest-code'
        self.cache_bucket = 'backtest-cache'
        self.results_bucket = 'backtest-results'

        # Node pool configuration (Basic cluster - free control plane)
        self.node_pool_id = "ocid1.nodepool.oc1.ap-mumbai-1.aaaaaaaaqs7a4f5jyyhcy3dsmedknnzbmhpdmdj6dqkastv5cnaehilq5g3q"

        # Project root (oci_cloud/tools/submit.py -> parent.parent.parent = project root)
        self.root = Path(__file__).parent.parent.parent

    def _get_namespace(self):
        """Get OCI Object Storage namespace using OCI SDK"""
        try:
            config = oci.config.from_file()
            client = ObjectStorageClient(config)
            return client.get_namespace().data
        except Exception as e:
            print(f"❌ Error getting namespace: {e}")
            print("Make sure OCI CLI is configured: oci setup config")
            sys.exit(1)

    def _get_next_retry_number(self, base_run_id):
        """
        Find the next retry number for a given base run ID.

        Checks cloud_results directory for existing retries and returns the next number.

        Args:
            base_run_id: The original run ID (without -retry suffix)

        Returns:
            Next retry number (1 if no retries exist)
        """
        cloud_results = self.root / 'cloud_results'
        if not cloud_results.exists():
            return 1

        # Find all existing retry directories for this base run
        existing_retries = []
        for d in cloud_results.iterdir():
            if d.is_dir() and d.name.startswith(base_run_id):
                if '-retry' in d.name:
                    try:
                        # Extract retry number from name like "20251121-084341-retry2"
                        retry_part = d.name.split('-retry')[-1]
                        retry_num = int(retry_part)
                        existing_retries.append(retry_num)
                    except ValueError:
                        pass

        if not existing_retries:
            return 1

        return max(existing_retries) + 1

    def generate_trading_dates(self, start_date, end_date):
        """
        Generate list of trading dates (exclude weekends and NSE holidays).

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of date strings
        """
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        dates = []
        while current <= end:
            # Use is_trading_day to exclude weekends AND NSE holidays
            if is_trading_day(current):
                dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        return dates

    def package_code(self, run_id):
        """
        Package current code into tarball.

        Returns:
            Path to tarball
        """
        print("\n[1/5] 📦 Packaging code...")

        # Create temp directory
        temp_dir = self.root / 'temp' / run_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Files and directories to include
        include_patterns = [
            'main.py',
            'config/**/*.json',
            'config/**/*.py',
            'config/**/*.toml',
            'services/**/*.py',
            'structures/**/*.py',
            'utils/**/*.py',
            'broker/**/*.py',
            'diagnostics/**/*.py',
            'pipelines/**/*.py',
            'pipelines/**/*.json',
            'api/**/*.py',
            'assets/**/*.json',
            'nse_all.json',
            'requirements.txt',
            '.env.development',
            '.env.production'
        ]

        tarball_path = temp_dir / 'code.tar.gz'

        file_count = 0
        with tarfile.open(tarball_path, 'w:gz') as tar:
            for pattern in include_patterns:
                # Handle dotfiles separately (glob doesn't match them by default)
                if pattern.startswith('.'):
                    file_path = self.root / pattern
                    if file_path.is_file():
                        arcname = file_path.relative_to(self.root)
                        tar.add(file_path, arcname=arcname)
                        file_count += 1
                else:
                    for file_path in self.root.glob(pattern):
                        if file_path.is_file():
                            arcname = file_path.relative_to(self.root)
                            tar.add(file_path, arcname=arcname)
                            file_count += 1
                            if file_count % 10 == 0:
                                print(f"  Adding files... {file_count}", end='\r')

        size_mb = tarball_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Packaged {file_count} files ({size_mb:.1f} MB)")

        return tarball_path

    def upload_code(self, tarball_path, run_id):
        """Upload code tarball to OCI Object Storage"""
        print("\n[2/5] ☁️  Uploading code to OCI...")

        object_name = f"{run_id}/code.tar.gz"

        try:
            with open(tarball_path, 'rb') as file_data:
                self.os_client.put_object(
                    namespace_name=self.namespace,
                    bucket_name=self.code_bucket,
                    object_name=object_name,
                    put_object_body=file_data
                )

            print(f"  ✓ Uploaded to: oci://{self.code_bucket}/{object_name}")
            return object_name

        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
            sys.exit(1)

    def get_config_hash(self):
        """Get configuration hash (using SHA256 for FIPS compliance)"""
        config_file = self.root / 'config' / 'configuration.json'
        with open(config_file, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:8]

    def submit_kubernetes_job(self, run_id, dates, description, max_parallel=None):
        """Submit Kubernetes Job"""
        print("\n[3/5] 🚀 Submitting Kubernetes Job...")

        # Read template
        template_path = self.root / 'oci_cloud' / 'k8s' / 'backtest-job-template.yaml'

        if not template_path.exists():
            print(f"  ❌ Template not found: {template_path}")
            print("  Please ensure oci_cloud/k8s/backtest-job-template.yaml exists")
            sys.exit(1)

        with open(template_path) as f:
            template = f.read()

        # Replace placeholders
        dates_csv = ','.join(dates)
        config_hash = self.get_config_hash()
        num_dates = len(dates)

        # Limit parallelism based on max_parallel (default: all dates)
        parallelism = min(max_parallel, num_dates) if max_parallel else num_dates

        job_yaml = template.replace('{{RUN_ID}}', run_id)
        job_yaml = job_yaml.replace('{{DATES_LIST}}', dates_csv)
        job_yaml = job_yaml.replace('{{TENANCY_NAMESPACE}}', self.namespace)
        job_yaml = job_yaml.replace('{{CONFIG_HASH}}', config_hash)
        job_yaml = job_yaml.replace('{{PARALLELISM}}', str(parallelism))
        job_yaml = job_yaml.replace('{{COMPLETIONS}}', str(num_dates))

        # Write temporary job file
        temp_job_path = self.root / 'temp' / f'job-{run_id}.yaml'
        temp_job_path.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_job_path, 'w') as f:
            f.write(job_yaml)

        # Apply job
        try:
            result = subprocess.run(
                ['kubectl', 'apply', '-f', str(temp_job_path)],
                capture_output=True,
                text=True,
                check=True
            )

            print(f"  ✓ Job created: backtest-{run_id}")
            print(f"  ✓ Total days: {len(dates)}")
            print(f"  ✓ Parallel pods: {parallelism} (runs {parallelism} at a time)")

            # Save metadata
            metadata = {
                'run_id': run_id,
                'start_date': dates[0],
                'end_date': dates[-1],
                'total_days': len(dates),
                'parallelism': parallelism,
                'description': description,
                'config_hash': config_hash,
                'submitted_at': datetime.now().isoformat(),
                'namespace': self.namespace,
                'region': self.region
            }

            metadata_path = self.root / 'cloud_results' / run_id / 'metadata.json'
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return parallelism

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Job submission failed: {e.stderr}")
            return None
        except FileNotFoundError:
            print(f"  ❌ kubectl not found. Please install kubectl and configure access to OKE cluster")
            return None

    def monitor_job(self, run_id, total_days, parallelism):
        """Monitor Kubernetes Job progress"""
        print(f"\n[4/5] 📊 Monitoring progress...")
        print()
        print(" Time    Running  Complete  Failed  Progress")
        print("━" * 60)

        start_time = time.time()

        while True:
            try:
                # Get job status
                result = subprocess.run(
                    ['kubectl', 'get', 'job', f'backtest-{run_id}', '-o', 'json'],
                    capture_output=True,
                    text=True,
                    check=True
                )

                job_status = json.loads(result.stdout)

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

            except subprocess.CalledProcessError:
                print(f"\n  ⚠️  Job not found, waiting...")
                time.sleep(10)
            except KeyboardInterrupt:
                print("\n\n  ⚠️  Monitoring interrupted (job still running in background)")
                print(f"  Resume monitoring: python tools/monitor_oci_backtest.py {run_id}")
                return False

        print()

        if failed > 0:
            print(f"\n  ⚠️  {failed} days failed")
            print(f"  Check logs: kubectl logs -l run-id={run_id},status=Failed")

        if succeeded == total_days:
            print(f"\n  ✅ All {total_days} days completed successfully!")

            elapsed_total = int(time.time() - start_time)
            minutes = elapsed_total // 60
            seconds = elapsed_total % 60

            # Calculate cost (2 OCPU per pod)
            total_ocpu = parallelism * 2
            ocpu_hours = (total_ocpu * elapsed_total) / 3600
            cost = ocpu_hours * 0.0015  # $0.0015 per OCPU-hour (Spot)

            print(f"  ⏱️  Duration: {minutes}m {seconds}s")
            print(f"  💰 Cost: ${cost:.2f} ({total_ocpu} OCPU × {minutes}m)")

        return succeeded == total_days

    def print_summary(self, run_id, start_date, end_date, total_days, description):
        """Print submission summary"""
        print()
        print("━" * 60)
        print("Oracle Cloud Parallel Backtest")
        print("━" * 60)
        print()
        print(f"Run ID: {run_id}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Total Days: {total_days}")
        if description:
            print(f"Description: {description}")
        print()
        print("━" * 60)

    def scale_up_nodepool(self, num_nodes, wait_seconds=180):
        """
        Scale up the node pool to the specified number of nodes.

        Args:
            num_nodes: Number of nodes to scale to
            wait_seconds: Seconds to wait after scaling (default: 180 = 3 minutes)

        Returns:
            True if successful, False otherwise
        """
        print()
        print("━" * 60)
        print(f"Scaling Up Node Pool to {num_nodes} nodes")
        print("━" * 60)
        print()

        try:
            print(f"Node Pool ID: {self.node_pool_id}")
            print(f"Setting size to {num_nodes}...")
            print()

            update_details = oci.container_engine.models.UpdateNodePoolDetails(
                node_config_details=oci.container_engine.models.UpdateNodePoolNodeConfigDetails(
                    size=num_nodes
                )
            )

            self.container_engine_client.update_node_pool(
                node_pool_id=self.node_pool_id,
                update_node_pool_details=update_details
            )

            print(f"✅ Node pool scaling initiated to {num_nodes} nodes")
            print()

            # Wait for nodes to become ready (smart polling)
            if wait_seconds > 0:
                print(f"⏳ Waiting for {num_nodes} nodes to become Ready (max {wait_seconds}s)...")
                print()

                start_time = time.time()
                check_interval = 10  # Check every 10 seconds
                last_check = 0

                while True:
                    elapsed = time.time() - start_time
                    remaining = int(wait_seconds - elapsed)

                    if remaining <= 0:
                        print()
                        print("⚠️  Timeout waiting for nodes, proceeding anyway...")
                        break

                    # Check node status periodically
                    if elapsed - last_check >= check_interval or last_check == 0:
                        last_check = elapsed
                        try:
                            result = subprocess.run(
                                ['kubectl', 'get', 'nodes', '--no-headers'],
                                capture_output=True, text=True, timeout=15
                            )
                            if result.returncode == 0:
                                lines = [l for l in result.stdout.strip().split('\n') if l]
                                ready_count = sum(1 for l in lines if 'Ready' in l and 'NotReady' not in l)

                                if ready_count >= num_nodes:
                                    print()
                                    print(f"✅ All {ready_count} nodes are Ready!")
                                    break
                                else:
                                    mins, secs = divmod(remaining, 60)
                                    print(f"   {ready_count}/{num_nodes} nodes Ready, waiting... ({mins:02d}:{secs:02d} remaining)", end='\r')
                        except Exception:
                            pass  # Ignore kubectl errors, keep waiting

                    time.sleep(1)

                print()
                print("✅ Proceeding with job submission")
            else:
                print("⏳ Nodes will take ~2-3 minutes to become ready")
                print("   Job will start once nodes are available")

            print()
            print("━" * 60)

            return True

        except Exception as e:
            print(f"❌ Failed to scale up node pool: {e}")
            print()
            print("You can scale up manually with:")
            print(f"  oci ce node-pool update --node-pool-id {self.node_pool_id} --size {num_nodes} --force")
            print()
            print("━" * 60)
            return False

    def run(self, start_date=None, end_date=None, description=None, no_wait=False,
            max_parallel=None, num_nodes=None, wait_after_scale=180, failed_dates_file=None):
        """Main workflow

        Args:
            start_date: Start date (YYYY-MM-DD) - required if not using failed_dates_file
            end_date: End date (YYYY-MM-DD) - required if not using failed_dates_file
            description: Optional description
            no_wait: If True, submit and exit without waiting
            max_parallel: Max parallel pods
            num_nodes: Number of nodes to scale to
            wait_after_scale: Seconds to wait after scaling (default: 180)
            failed_dates_file: Path to JSON file containing failed dates to re-run
        """
        # Handle failed dates re-run
        if failed_dates_file:
            with open(failed_dates_file, 'r') as f:
                failed_data = json.load(f)

            dates = failed_data.get('failed_dates', [])
            original_run_id = failed_data.get('run_id', 'unknown')

            if not dates:
                print("❌ No failed dates found in the file")
                return

            # Generate run ID based on original with retry suffix
            # Check for existing retries to increment the counter
            base_run_id = original_run_id.split('-retry')[0]  # Strip any existing retry suffix
            retry_num = self._get_next_retry_number(base_run_id)
            run_id = f"{base_run_id}-retry{retry_num}"

            start_date = min(dates)
            end_date = max(dates)

            print()
            print("━" * 60)
            print(f"RE-RUNNING FAILED DATES (Retry #{retry_num})")
            print("━" * 60)
            print()
            print(f"Original Run: {original_run_id}")
            print(f"Retry Run ID: {run_id}")
            print(f"Failed Dates: {len(dates)}")
            print(f"Date Range: {start_date} to {end_date}")
            print()
            print("━" * 60)
        else:
            if not start_date or not end_date:
                print("❌ Either --start/--end or --failed-dates is required")
                return

            # Generate run ID (use hyphens for Kubernetes DNS compliance)
            run_id = datetime.now().strftime('%Y%m%d-%H%M%S')

            # Generate trading dates
            dates = self.generate_trading_dates(start_date, end_date)

            # Print summary
            self.print_summary(run_id, start_date, end_date, len(dates), description)

        # Scale up node pool if requested
        if num_nodes is not None:
            if not self.scale_up_nodepool(num_nodes, wait_seconds=wait_after_scale):
                print("⚠️  Node pool scaling failed, but continuing anyway...")
                print()

        # Package code
        tarball_path = self.package_code(run_id)

        # Upload code
        self.upload_code(tarball_path, run_id)

        # Submit Kubernetes job
        parallelism = self.submit_kubernetes_job(run_id, dates, description, max_parallel)

        if parallelism is None:
            print("\n❌ Job submission failed")
            return

        if no_wait:
            print(f"\n✅ Job submitted (not waiting for completion)")
            print(f"\nMonitor & cleanup: python oci_cloud/tools/monitor_and_cleanup_backtest.py {run_id}")
            return

        # Hand off to monitor_and_cleanup_backtest.py for monitoring, download, and cleanup
        print(f"\n[4/5] 📊 Handing off to monitor_and_cleanup_backtest.py...")
        monitor_script = self.root / 'oci_cloud' / 'tools' / 'monitor_and_cleanup_backtest.py'

        cmd = [sys.executable, str(monitor_script), run_id]
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\n❌ Monitor/cleanup failed with exit code {result.returncode}")
            return


def main():
    parser = argparse.ArgumentParser(
        description='Submit parallel backtest to Oracle Cloud Kubernetes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic submission with auto node scaling
  python oci_cloud/tools/submit_oci_backtest.py --start 2024-01-01 --end 2024-06-30 --nodes 4

  # Without node scaling (assumes nodes already running)
  python oci_cloud/tools/submit_oci_backtest.py --start 2024-01-01 --end 2024-06-30

  # With custom parallelism limit
  python oci_cloud/tools/submit_oci_backtest.py --start 2024-01-01 --end 2024-06-30 --nodes 4 --max-parallel 50

  # Skip wait after node scaling (for already-running nodes)
  python oci_cloud/tools/submit_oci_backtest.py --start 2024-01-01 --end 2024-06-30 --nodes 4 --wait-after-scale 0

  # Re-run only failed dates from a previous run
  python oci_cloud/tools/submit_oci_backtest.py --failed-dates cloud_results/20251121-084341/failed_dates.json --nodes 4
        """
    )

    parser.add_argument('--start', help='Start date (YYYY-MM-DD). Required unless using --failed-dates')
    parser.add_argument('--end', help='End date (YYYY-MM-DD). Required unless using --failed-dates')
    parser.add_argument('--description', help='Description of this backtest run')
    parser.add_argument('--no-wait', action='store_true', help='Submit and exit without waiting')
    parser.add_argument('--max-parallel', type=int, default=None,
                        help='Max parallel pods (default: unlimited). Use 50 for 100 OCPU limit.')
    parser.add_argument('--nodes', type=int, default=None,
                        help='Number of nodes to scale node pool to before starting (e.g., 4). '
                             'If not specified, assumes nodes are already running.')
    parser.add_argument('--wait-after-scale', type=int, default=180,
                        help='Seconds to wait after scaling up nodes (default: 180 = 3 minutes). '
                             'Set to 0 to skip wait.')
    parser.add_argument('--failed-dates', type=str, default=None,
                        help='Path to failed_dates.json file to re-run only failed dates. '
                             'If specified, --start and --end are ignored.')

    args = parser.parse_args()

    # Validate arguments
    if not args.failed_dates and (not args.start or not args.end):
        parser.error("Either --start/--end or --failed-dates is required")

    submitter = OCIBacktestSubmitter()
    submitter.run(
        start_date=args.start,
        end_date=args.end,
        description=args.description,
        no_wait=args.no_wait,
        max_parallel=args.max_parallel,
        num_nodes=args.nodes,
        wait_after_scale=args.wait_after_scale,
        failed_dates_file=args.failed_dates
    )


if __name__ == '__main__':
    main()
