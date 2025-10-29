"""
Download OCI Backtest Results
==============================

Download results from OCI Object Storage to local machine.

Usage:
    python tools/download_oci_results.py 20251027_154230
    python tools/download_oci_results.py 20251027_154230 --parallel 10
"""

import argparse
import oci
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


class OCIResultsDownloader:
    def __init__(self):
        """Initialize OCI client"""
        self.config = oci.config.from_file()
        self.os_client = oci.object_storage.ObjectStorageClient(self.config)
        self.namespace = self._get_namespace()
        self.results_bucket = 'backtest-results'
        self.root = Path(__file__).parent.parent

    def _get_namespace(self):
        """Get OCI namespace"""
        return self.os_client.get_namespace().data

    def download_file(self, object_name, local_path):
        """Download single file from OCI"""
        try:
            get_obj = self.os_client.get_object(
                namespace_name=self.namespace,
                bucket_name=self.results_bucket,
                object_name=object_name
            )

            local_path.parent.mkdir(parents=True, exist_ok=True)

            with open(local_path, 'wb') as f:
                for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                    f.write(chunk)

            return True

        except Exception as e:
            print(f"  ⚠️  Failed to download {object_name}: {e}")
            return False

    def download_results(self, run_id, parallel=5):
        """
        Download all results for a run.

        Args:
            run_id: Run identifier
            parallel: Number of parallel downloads
        """
        print(f"Downloading results for run: {run_id}")
        print()

        results_dir = self.root / 'cloud_results' / run_id
        results_dir.mkdir(parents=True, exist_ok=True)

        # List objects
        print(f"Listing objects in oci://{self.results_bucket}/{run_id}/...")

        try:
            list_response = self.os_client.list_objects(
                namespace_name=self.namespace,
                bucket_name=self.results_bucket,
                prefix=f"{run_id}/"
            )

            objects = list_response.data.objects

            if not objects:
                print(f"❌ No results found for run: {run_id}")
                return

            # Filter out directory markers
            files_to_download = [obj for obj in objects if not obj.name.endswith('/')]

            print(f"Found {len(files_to_download)} files to download")
            print()

            # Download files in parallel
            downloaded = 0
            failed = 0

            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {}

                for obj in files_to_download:
                    object_name = obj.name
                    relative_path = object_name.replace(f"{run_id}/", "")
                    local_path = results_dir / relative_path

                    future = executor.submit(self.download_file, object_name, local_path)
                    futures[future] = object_name

                for future in as_completed(futures):
                    object_name = futures[future]

                    if future.result():
                        downloaded += 1
                    else:
                        failed += 1

                    if (downloaded + failed) % 10 == 0:
                        print(f"  Progress: {downloaded}/{len(files_to_download)} files...", end='\r')

            print(f"\n\n✅ Downloaded {downloaded} files to: {results_dir}")

            if failed > 0:
                print(f"⚠️  {failed} files failed to download")

            print()
            print(f"Next: python tools/analyze_6month_backtest.py {results_dir}")

        except Exception as e:
            print(f"❌ Error listing objects: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Download OCI backtest results')
    parser.add_argument('run_id', help='Run ID (e.g., 20251027_154230)')
    parser.add_argument('--parallel', type=int, default=5, help='Number of parallel downloads (default: 5)')

    args = parser.parse_args()

    downloader = OCIResultsDownloader()
    downloader.download_results(args.run_id, args.parallel)


if __name__ == '__main__':
    main()
