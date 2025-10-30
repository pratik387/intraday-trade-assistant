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


class OCIBacktestSubmitter:
    def __init__(self):
        """Initialize OCI clients"""
        self.config = oci.config.from_file()
        self.namespace = self._get_namespace()
        self.os_client = ObjectStorageClient(self.config)
        self.region = self.config['region']

        # Bucket names
        self.code_bucket = 'backtest-code'
        self.cache_bucket = 'backtest-cache'
        self.results_bucket = 'backtest-results'

        # Project root (oci/tools/submit.py -> parent.parent.parent = project root)
        self.root = Path(__file__).parent.parent.parent

    def _get_namespace(self):
        """Get OCI Object Storage namespace"""
        try:
            result = subprocess.run(
                ['oci', 'os', 'ns', 'get', '--query', 'data', '--raw-output'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception as e:
            print(f"❌ Error getting namespace: {e}")
            print("Make sure OCI CLI is configured: oci setup config")
            sys.exit(1)

    def generate_trading_dates(self, start_date, end_date):
        """
        Generate list of trading dates (exclude weekends).

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
            # Exclude weekends (Saturday=5, Sunday=6)
            if current.weekday() < 5:
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
            'services/**/*.py',
            'structures/**/*.py',
            'utils/**/*.py',
            'broker/**/*.py',
            'diagnostics/**/*.py',
            'nse_all.json',
            'oci/docker/entrypoint_runtime.py',
            'requirements.txt',
            '.env.development',
            '.env.production'
        ]

        tarball_path = temp_dir / 'code.tar.gz'

        file_count = 0
        with tarfile.open(tarball_path, 'w:gz') as tar:
            for pattern in include_patterns:
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

    def submit_kubernetes_job(self, run_id, dates, description):
        """Submit Kubernetes Job"""
        print("\n[3/5] 🚀 Submitting Kubernetes Job...")

        # Read template
        template_path = self.root / 'oci' / 'k8s' / 'backtest-job-template.yaml'

        if not template_path.exists():
            print(f"  ❌ Template not found: {template_path}")
            print("  Please ensure oci/k8s/backtest-job-template.yaml exists")
            sys.exit(1)

        with open(template_path) as f:
            template = f.read()

        # Replace placeholders
        dates_csv = ','.join(dates)
        config_hash = self.get_config_hash()
        num_dates = len(dates)

        job_yaml = template.replace('{{RUN_ID}}', run_id)
        job_yaml = job_yaml.replace('{{DATES_LIST}}', dates_csv)
        job_yaml = job_yaml.replace('{{TENANCY_NAMESPACE}}', self.namespace)
        job_yaml = job_yaml.replace('{{CONFIG_HASH}}', config_hash)
        job_yaml = job_yaml.replace('{{PARALLELISM}}', str(num_dates))
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
            print(f"  ✓ Pods: {len(dates)} (parallel execution)")

            # Save metadata
            metadata = {
                'run_id': run_id,
                'start_date': dates[0],
                'end_date': dates[-1],
                'total_days': len(dates),
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

            return True

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Job submission failed: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"  ❌ kubectl not found. Please install kubectl and configure access to OKE cluster")
            return False

    def monitor_job(self, run_id, total_days):
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

            # Calculate cost
            ocpu_hours = (240 * elapsed_total) / 3600
            cost = ocpu_hours * 0.0015  # $0.0015 per OCPU-hour (Spot)

            print(f"  ⏱️  Duration: {minutes}m {seconds}s")
            print(f"  💰 Cost: ${cost:.2f}")

        return succeeded == total_days

    def download_results(self, run_id):
        """Download results from OCI Object Storage"""
        print(f"\n[5/5] 📥 Downloading results...")

        results_dir = self.root / 'cloud_results' / run_id
        results_dir.mkdir(parents=True, exist_ok=True)

        try:
            # List objects in results bucket for this run
            list_objects_response = self.os_client.list_objects(
                namespace_name=self.namespace,
                bucket_name=self.results_bucket,
                prefix=f"{run_id}/"
            )

            objects = list_objects_response.data.objects

            if not objects:
                print(f"  ⚠️  No results found in oci://{self.results_bucket}/{run_id}/")
                return

            downloaded = 0
            for obj in objects:
                object_name = obj.name

                # Skip directory markers
                if object_name.endswith('/'):
                    continue

                # Local file path
                relative_path = object_name.replace(f"{run_id}/", "")
                local_path = results_dir / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Download object
                get_obj = self.os_client.get_object(
                    namespace_name=self.namespace,
                    bucket_name=self.results_bucket,
                    object_name=object_name
                )

                with open(local_path, 'wb') as f:
                    for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                        f.write(chunk)

                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"  Downloaded {downloaded} files...", end='\r')

            print(f"  ✓ Downloaded {downloaded} files to: {results_dir}")

        except Exception as e:
            print(f"  ❌ Download failed: {e}")

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

    def run(self, start_date, end_date, description=None, no_wait=False):
        """Main workflow"""
        # Generate run ID (use hyphens for Kubernetes DNS compliance)
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S')

        # Generate trading dates
        dates = self.generate_trading_dates(start_date, end_date)

        # Print summary
        self.print_summary(run_id, start_date, end_date, len(dates), description)

        # Package code
        tarball_path = self.package_code(run_id)

        # Upload code
        self.upload_code(tarball_path, run_id)

        # Submit Kubernetes job
        success = self.submit_kubernetes_job(run_id, dates, description)

        if not success:
            print("\n❌ Job submission failed")
            return

        if no_wait:
            print(f"\n✅ Job submitted (not waiting for completion)")
            print(f"\nMonitor: python tools/monitor_oci_backtest.py {run_id}")
            print(f"Download: python tools/download_oci_results.py {run_id}")
            return

        # Monitor progress
        success = self.monitor_job(run_id, len(dates))

        if not success:
            return

        # Download results
        self.download_results(run_id)

        print()
        print("━" * 60)
        print("✅ Backtest Complete!")
        print("━" * 60)
        print()
        print(f"Results: ./cloud_results/{run_id}/")
        print()
        print(f"Next: python tools/analyze_6month_backtest.py cloud_results/{run_id}/")
        print("━" * 60)
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Submit parallel backtest to Oracle Cloud Kubernetes',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--description', help='Description of this backtest run')
    parser.add_argument('--no-wait', action='store_true', help='Submit and exit without waiting')

    args = parser.parse_args()

    submitter = OCIBacktestSubmitter()
    submitter.run(
        start_date=args.start,
        end_date=args.end,
        description=args.description,
        no_wait=args.no_wait
    )


if __name__ == '__main__':
    main()
