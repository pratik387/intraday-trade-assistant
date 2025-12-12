#!/usr/bin/env python3
"""
OCI Backtest Cleanup and Download Automation
=============================================

Automates the complete post-backtest workflow:
1. Scale down node pool to 0
2. Download all results from OCI Object Storage (handles >1000 files)
3. Create local zip archive
4. Clean up OCI bucket
5. Clean up local extracted directory

This script bypasses the manual confirmation issue with bulk-download
by using the OCI Python SDK directly.

Usage:
    python oci/tools/cleanup_and_download_backtest.py <run_id>
    python oci/tools/cleanup_and_download_backtest.py 20251121-084341 --parallel 20
    python oci/tools/cleanup_and_download_backtest.py 20251121-084341 --skip-nodepool
    python oci/tools/cleanup_and_download_backtest.py 20251121-084341 --keep-oci-files

Arguments:
    run_id: The backtest run ID (e.g., 20251121-084341)

Options:
    --parallel N: Number of parallel downloads (default: 10)
    --skip-nodepool: Skip scaling down node pool
    --keep-oci-files: Don't delete files from OCI bucket after download
    --keep-extracted: Don't delete local extracted directory after zipping

Example:
    python oci/tools/cleanup_and_download_backtest.py 20251121-084341
"""

import argparse
import oci
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import time


class BacktestCleanupAutomation:
    def __init__(self, run_id, parallel=10, skip_nodepool=False, keep_oci_files=False, keep_extracted=False):
        """
        Initialize automation.

        Args:
            run_id: Backtest run ID
            parallel: Number of parallel downloads
            skip_nodepool: Skip node pool scaling
            keep_oci_files: Don't delete OCI files
            keep_extracted: Don't delete local extracted directory
        """
        self.run_id = run_id
        self.parallel = parallel
        self.skip_nodepool = skip_nodepool
        self.keep_oci_files = keep_oci_files
        self.keep_extracted = keep_extracted

        # OCI configuration (Basic cluster - free control plane)
        self.node_pool_id = "ocid1.nodepool.oc1.ap-mumbai-1.aaaaaaaaqs7a4f5jyyhcy3dsmedknnzbmhpdmdj6dqkastv5cnaehilq5g3q"
        self.bucket_name = "backtest-results"

        # Local paths
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent
        self.download_dir = self.project_root / f"{run_id}_full"
        self.zip_file = self.project_root / f"backtest_{run_id}.zip"

        # Initialize OCI clients
        print("Initializing OCI clients...")
        self.config = oci.config.from_file()
        self.os_client = oci.object_storage.ObjectStorageClient(self.config)
        self.container_engine_client = oci.container_engine.ContainerEngineClient(self.config)
        self.namespace = self.os_client.get_namespace().data
        print(f"OK: Connected to OCI (namespace: {self.namespace})")
        print()

    def scale_down_nodepool(self):
        """Scale down node pool to 0 nodes."""
        if self.skip_nodepool:
            print("SKIPPED: Node pool scaling (--skip-nodepool)")
            print()
            return

        print("=" * 80)
        print("STEP 1: Scaling Down Node Pool")
        print("=" * 80)

        try:
            print(f"Node Pool ID: {self.node_pool_id}")
            print("Setting size to 0...")

            update_details = oci.container_engine.models.UpdateNodePoolDetails(
                node_config_details=oci.container_engine.models.UpdateNodePoolNodeConfigDetails(
                    size=0
                )
            )

            self.container_engine_client.update_node_pool(
                node_pool_id=self.node_pool_id,
                update_node_pool_details=update_details
            )

            print("OK: Node pool scaled down to 0")
            print()

        except Exception as e:
            print(f"ERROR: Failed to scale down node pool: {e}")
            print("You can scale down manually with:")
            print(f"  oci ce node-pool update --node-pool-id {self.node_pool_id} --size 0 --force")
            print()
            # Continue anyway - this is not critical

    def list_all_objects(self):
        """
        List all objects in the bucket with the run_id prefix.

        Returns:
            List of object names
        """
        print("=" * 80)
        print("STEP 2: Listing Objects in OCI Bucket")
        print("=" * 80)

        print(f"Bucket: {self.bucket_name}")
        print(f"Prefix: {self.run_id}/")
        print()

        all_objects = []
        next_start = None

        try:
            while True:
                # List objects (paginated)
                list_response = self.os_client.list_objects(
                    namespace_name=self.namespace,
                    bucket_name=self.bucket_name,
                    prefix=f"{self.run_id}/",
                    start=next_start,
                    limit=1000  # Max per page
                )

                objects = list_response.data.objects

                # Filter out directory markers
                files = [obj.name for obj in objects if not obj.name.endswith('/')]
                all_objects.extend(files)

                # Check if there are more pages
                next_start = list_response.data.next_start_with
                if not next_start:
                    break

                print(f"Listed {len(all_objects)} objects so far...", end='\r')

            print(f"\nOK: Found {len(all_objects)} files to download")
            print()

            return all_objects

        except Exception as e:
            print(f"ERROR: Failed to list objects: {e}")
            sys.exit(1)

    def download_file(self, object_name):
        """
        Download a single file from OCI Object Storage.

        Args:
            object_name: Full object name (including prefix)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get object
            get_obj = self.os_client.get_object(
                namespace_name=self.namespace,
                bucket_name=self.bucket_name,
                object_name=object_name
            )

            # Determine local path (remove run_id prefix)
            relative_path = object_name.replace(f"{self.run_id}/", "")
            local_path = self.download_dir / relative_path

            # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(local_path, 'wb') as f:
                for chunk in get_obj.data.raw.stream(1024 * 1024, decode_content=False):
                    f.write(chunk)

            return True

        except Exception as e:
            print(f"\nWARNING: Failed to download {object_name}: {e}")
            return False

    def download_all_files(self, object_names):
        """
        Download all files in parallel.

        Args:
            object_names: List of object names to download

        Returns:
            Number of successfully downloaded files
        """
        print("=" * 80)
        print("STEP 3: Downloading Files")
        print("=" * 80)

        print(f"Download directory: {self.download_dir}")
        print(f"Parallel downloads: {self.parallel}")
        print()

        # Create download directory
        self.download_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        failed = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = {executor.submit(self.download_file, obj): obj for obj in object_names}

            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
                else:
                    failed += 1

                # Progress update every 10 files
                if (downloaded + failed) % 10 == 0 or (downloaded + failed) == len(object_names):
                    elapsed = time.time() - start_time
                    rate = downloaded / elapsed if elapsed > 0 else 0
                    pct = ((downloaded + failed) / len(object_names)) * 100
                    print(f"Progress: {downloaded}/{len(object_names)} files ({pct:.1f}%) - {rate:.1f} files/sec", end='\r')

        print()  # New line after progress
        print()
        print(f"OK: Downloaded {downloaded} files in {time.time() - start_time:.1f}s")

        if failed > 0:
            print(f"WARNING: {failed} files failed to download")

        print()
        return downloaded

    def create_zip_archive(self):
        """Create zip archive of downloaded files."""
        print("=" * 80)
        print("STEP 4: Creating Zip Archive")
        print("=" * 80)

        print(f"Source: {self.download_dir}")
        print(f"Archive: {self.zip_file}")
        print()

        try:
            # Remove existing zip if present
            if self.zip_file.exists():
                print(f"Removing existing archive: {self.zip_file}")
                self.zip_file.unlink()

            # Create zip using shutil (cross-platform)
            print("Creating archive...")
            start_time = time.time()

            # Use shutil.make_archive for cross-platform support
            base_name = str(self.zip_file.with_suffix(''))
            shutil.make_archive(base_name, 'zip', self.download_dir)

            elapsed = time.time() - start_time
            size_mb = self.zip_file.stat().st_size / (1024 * 1024)

            print(f"OK: Archive created successfully")
            print(f"    Size: {size_mb:.1f} MB")
            print(f"    Time: {elapsed:.1f}s")
            print()

        except Exception as e:
            print(f"ERROR: Failed to create zip archive: {e}")
            sys.exit(1)

    def cleanup_oci_files(self, object_names):
        """
        Delete files from OCI Object Storage.

        Args:
            object_names: List of object names to delete
        """
        if self.keep_oci_files:
            print("SKIPPED: OCI file cleanup (--keep-oci-files)")
            print()
            return

        print("=" * 80)
        print("STEP 5: Cleaning Up OCI Files")
        print("=" * 80)

        print(f"Deleting {len(object_names)} files from bucket...")
        print()

        deleted = 0
        failed = 0

        # Delete in batches (OCI has bulk delete limits)
        batch_size = 100

        try:
            for i in range(0, len(object_names), batch_size):
                batch = object_names[i:i + batch_size]

                # Create delete details
                delete_details = oci.object_storage.models.CreateMultipartUploadDetails()

                # Delete each object
                for obj_name in batch:
                    try:
                        self.os_client.delete_object(
                            namespace_name=self.namespace,
                            bucket_name=self.bucket_name,
                            object_name=obj_name
                        )
                        deleted += 1
                    except Exception as e:
                        print(f"\nWARNING: Failed to delete {obj_name}: {e}")
                        failed += 1

                    # Progress update
                    if (deleted + failed) % 50 == 0 or (deleted + failed) == len(object_names):
                        pct = ((deleted + failed) / len(object_names)) * 100
                        print(f"Progress: {deleted}/{len(object_names)} files deleted ({pct:.1f}%)", end='\r')

            print()  # New line after progress
            print()
            print(f"OK: Deleted {deleted} files from OCI bucket")

            if failed > 0:
                print(f"WARNING: {failed} files failed to delete")

            print()

        except Exception as e:
            print(f"ERROR: Failed to delete OCI files: {e}")
            print("You can clean up manually with:")
            print(f"  oci os object bulk-delete --bucket-name {self.bucket_name} --prefix {self.run_id}/ --force")
            print()

    def cleanup_local_directory(self):
        """Delete local extracted directory."""
        if self.keep_extracted:
            print("SKIPPED: Local directory cleanup (--keep-extracted)")
            print()
            return

        print("=" * 80)
        print("STEP 6: Cleaning Up Local Directory")
        print("=" * 80)

        print(f"Deleting: {self.download_dir}")

        try:
            if self.download_dir.exists():
                shutil.rmtree(self.download_dir)
                print("OK: Local directory deleted")
            else:
                print("SKIPPED: Directory does not exist")

            print()

        except Exception as e:
            print(f"ERROR: Failed to delete local directory: {e}")
            print(f"You can delete manually: rmdir /s /q {self.download_dir}")
            print()

    def run(self):
        """Execute the complete automation workflow."""
        print()
        print("=" * 80)
        print("OCI BACKTEST CLEANUP & DOWNLOAD AUTOMATION")
        print("=" * 80)
        print(f"Run ID: {self.run_id}")
        print(f"Parallel downloads: {self.parallel}")
        print("=" * 80)
        print()

        try:
            # Step 1: Scale down node pool
            self.scale_down_nodepool()

            # Step 2: List all objects
            object_names = self.list_all_objects()

            if not object_names:
                print("ERROR: No files found to download")
                sys.exit(1)

            # Step 3: Download all files
            downloaded = self.download_all_files(object_names)

            if downloaded == 0:
                print("ERROR: No files were downloaded successfully")
                sys.exit(1)

            # Step 4: Create zip archive
            self.create_zip_archive()

            # Step 5: Cleanup OCI files
            self.cleanup_oci_files(object_names)

            # Step 6: Cleanup local directory
            self.cleanup_local_directory()

            # Success!
            print()
            print("=" * 80)
            print("AUTOMATION COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print()
            print(f"Archive: {self.zip_file}")
            print(f"Size: {self.zip_file.stat().st_size / (1024 * 1024):.1f} MB")
            print()
            print("Next steps:")
            print(f"  1. Process the backtest: python oci/process_backtest_run.py {self.zip_file}")
            print()
            print("=" * 80)
            print()

        except KeyboardInterrupt:
            print()
            print("=" * 80)
            print("AUTOMATION INTERRUPTED")
            print("=" * 80)
            print()
            print("Partial progress may have been made. Check:")
            print(f"  - Download directory: {self.download_dir}")
            print(f"  - Zip archive: {self.zip_file}")
            print()
            sys.exit(1)

        except Exception as e:
            print()
            print("=" * 80)
            print("AUTOMATION FAILED")
            print("=" * 80)
            print(f"Error: {e}")
            print()
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Automate OCI backtest cleanup and download',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python oci/tools/cleanup_and_download_backtest.py 20251121-084341

  # With custom parallel downloads
  python oci/tools/cleanup_and_download_backtest.py 20251121-084341 --parallel 20

  # Skip node pool scaling
  python oci/tools/cleanup_and_download_backtest.py 20251121-084341 --skip-nodepool

  # Keep OCI files after download
  python oci/tools/cleanup_and_download_backtest.py 20251121-084341 --keep-oci-files

  # Keep local extracted directory
  python oci/tools/cleanup_and_download_backtest.py 20251121-084341 --keep-extracted
        """
    )

    parser.add_argument('run_id', help='Backtest run ID (e.g., 20251121-084341)')
    parser.add_argument('--parallel', type=int, default=10,
                        help='Number of parallel downloads (default: 10)')
    parser.add_argument('--skip-nodepool', action='store_true',
                        help='Skip scaling down node pool')
    parser.add_argument('--keep-oci-files', action='store_true', default=True,
                        help="Don't delete files from OCI bucket after download (default: True)")
    parser.add_argument('--delete-oci-files', action='store_true',
                        help="Delete files from OCI bucket after download (must be explicit)")
    parser.add_argument('--keep-extracted', action='store_true',
                        help="Don't delete local extracted directory after zipping")

    args = parser.parse_args()

    # Default is to KEEP OCI files unless --delete-oci-files is explicitly passed
    keep_oci = not args.delete_oci_files

    automation = BacktestCleanupAutomation(
        run_id=args.run_id,
        parallel=args.parallel,
        skip_nodepool=args.skip_nodepool,
        keep_oci_files=keep_oci,
        keep_extracted=args.keep_extracted
    )

    automation.run()


if __name__ == '__main__':
    main()
