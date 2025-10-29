"""
List OCI Backtests
==================

List all backtest runs in OCI Object Storage.

Usage:
    python tools/list_oci_backtests.py
    python tools/list_oci_backtests.py --limit 20
"""

import argparse
import oci
import json
from pathlib import Path
from datetime import datetime


class OCIBacktestLister:
    def __init__(self):
        """Initialize OCI client"""
        self.config = oci.config.from_file()
        self.os_client = oci.object_storage.ObjectStorageClient(self.config)
        self.namespace = self.os_client.get_namespace().data
        self.results_bucket = 'backtest-results'
        self.root = Path(__file__).parent.parent

    def list_runs(self, limit=50):
        """List all backtest runs"""
        print(f"Listing backtest runs from oci://{self.results_bucket}/")
        print()

        try:
            # List all objects with prefix matching run_id pattern
            list_response = self.os_client.list_objects(
                namespace_name=self.namespace,
                bucket_name=self.results_bucket,
                delimiter='/',
                limit=1000
            )

            # Extract run IDs from prefixes
            prefixes = list_response.data.prefixes or []

            if not prefixes:
                print("No backtest runs found")
                return

            # Get metadata for each run
            runs = []

            for prefix in prefixes:
                run_id = prefix.rstrip('/')

                # Try to find metadata file
                metadata_key = f"{run_id}/metadata.json"

                try:
                    get_obj = self.os_client.get_object(
                        namespace_name=self.namespace,
                        bucket_name=self.results_bucket,
                        object_name=metadata_key
                    )

                    metadata = json.loads(get_obj.data.content.decode('utf-8'))

                    runs.append({
                        'run_id': run_id,
                        'date': metadata.get('submitted_at', 'Unknown'),
                        'start_date': metadata.get('start_date', 'Unknown'),
                        'end_date': metadata.get('end_date', 'Unknown'),
                        'days': metadata.get('total_days', 0),
                        'description': metadata.get('description', '')
                    })

                except:
                    # No metadata, just add run_id
                    runs.append({
                        'run_id': run_id,
                        'date': 'Unknown',
                        'start_date': 'Unknown',
                        'end_date': 'Unknown',
                        'days': 0,
                        'description': ''
                    })

            # Sort by run_id (descending, most recent first)
            runs.sort(key=lambda x: x['run_id'], reverse=True)

            # Limit results
            runs = runs[:limit]

            # Print table
            print(f"{'Run ID':<20} {'Submitted':<20} {'Date Range':<25} {'Days':<6} {'Description':<30}")
            print("━" * 120)

            for run in runs:
                run_id = run['run_id']
                date = run['date']
                if date != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(date)
                        date = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass

                date_range = f"{run['start_date']} to {run['end_date']}"
                days = run['days']
                description = run['description'][:28] + '..' if len(run['description']) > 30 else run['description']

                print(f"{run_id:<20} {date:<20} {date_range:<25} {days:<6} {description:<30}")

            print()
            print(f"Total: {len(runs)} runs (showing {limit} most recent)")
            print()
            print("Download results: python tools/download_oci_results.py <run_id>")
            print("Monitor job:      python tools/monitor_oci_backtest.py <run_id>")

        except Exception as e:
            print(f"❌ Error listing runs: {e}")


def main():
    parser = argparse.ArgumentParser(description='List OCI backtest runs')
    parser.add_argument('--limit', type=int, default=50, help='Maximum number of runs to display (default: 50)')

    args = parser.parse_args()

    lister = OCIBacktestLister()
    lister.list_runs(args.limit)


if __name__ == '__main__':
    main()
