#!/usr/bin/env python3
"""
Auto-scale node pool to 0 after Kubernetes job completes.

Usage:
    python oci_cloud/tools/auto_scale_after_job.py <job-name> <node-pool-id>
"""
import subprocess
import time
import sys
import json

def get_job_status(job_name):
    """Check if Kubernetes job is complete"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "job", job_name, "-o", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        job = json.loads(result.stdout)

        completions = job.get("spec", {}).get("completions", 0)
        succeeded = job.get("status", {}).get("succeeded", 0)
        active = job.get("status", {}).get("active", 0)
        failed = job.get("status", {}).get("failed", 0)

        return {
            "completions": completions,
            "succeeded": succeeded,
            "active": active,
            "failed": failed,
            "complete": (succeeded + failed >= completions) and active == 0
        }
    except subprocess.CalledProcessError:
        return None

def scale_node_pool(node_pool_id, size=0):
    """Scale OCI node pool to specified size"""
    try:
        subprocess.run(
            [
                "oci", "ce", "node-pool", "update",
                "--node-pool-id", node_pool_id,
                "--size", str(size),
                "--force"
            ],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to scale node pool: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python auto_scale_after_job.py <job-name> <node-pool-id>")
        sys.exit(1)

    job_name = sys.argv[1]
    node_pool_id = sys.argv[2]

    print(f"Monitoring job: {job_name}")
    print(f"Will scale down node pool: {node_pool_id}")
    print("=" * 80)

    check_interval = 60  # Check every 60 seconds

    while True:
        status = get_job_status(job_name)

        if status is None:
            print(f"Job {job_name} not found. Exiting.")
            sys.exit(1)

        print(f"Job status: {status['succeeded']}/{status['completions']} completed, "
              f"{status['active']} active, {status['failed']} failed")

        if status['complete']:
            print("\n" + "=" * 80)
            print("Job completed! Scaling node pool to 0...")
            print("=" * 80)

            if scale_node_pool(node_pool_id, 0):
                print("\n✓ Node pool scaled to 0")
                print("✓ Nodes will terminate in ~5 minutes")
                print("✓ Cost savings activated!")
                sys.exit(0)
            else:
                print("\n✗ Failed to scale node pool")
                print("Please scale down manually in OCI Console")
                sys.exit(1)

        time.sleep(check_interval)

if __name__ == "__main__":
    main()
