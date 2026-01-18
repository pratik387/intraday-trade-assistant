#!/usr/bin/env python3
"""
Submit Regime-Based Backtest to Oracle Cloud Kubernetes
=======================================================

Runs backtests for specific regime periods (non-sequential months).

Usage:
    python oci_cloud/tools/submit_regime_backtest.py --max-parallel 50
    python oci_cloud/tools/submit_regime_backtest.py --regimes Strong_Uptrend Shock_Down --max-parallel 50
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import submit_oci_backtest
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from submit_oci_backtest import OCIBacktestSubmitter
from datetime import datetime, timedelta
from utils.util import is_trading_day


# Regime configurations (from regime_orchestrator.py)
REGIME_CONFIGS = [
    {"name": "Strong_Uptrend", "start": "2023-12-01", "end": "2023-12-31"},
    {"name": "Shock_Down", "start": "2024-01-01", "end": "2024-01-31"},
    {"name": "Event_Driven_HighVol", "start": "2024-06-01", "end": "2024-06-30"},
    {"name": "Correction_RiskOff", "start": "2024-10-01", "end": "2024-10-31"},
    {"name": "Prolonged_Drawdown", "start": "2025-02-01", "end": "2025-02-28"},
    {"name": "Low_Vol_Range", "start": "2025-07-01", "end": "2025-07-31"}
]


def generate_regime_dates(regime_configs):
    """
    Generate list of trading dates for specified regimes (exclude weekends and NSE holidays).

    Args:
        regime_configs: List of regime dicts with 'name', 'start', 'end'

    Returns:
        List of (date_str, regime_name) tuples
    """
    dates = []

    for regime in regime_configs:
        current = datetime.strptime(regime['start'], '%Y-%m-%d')
        end = datetime.strptime(regime['end'], '%Y-%m-%d')

        while current <= end:
            # Use is_trading_day to exclude weekends AND NSE holidays
            if is_trading_day(current):
                dates.append((current.strftime('%Y-%m-%d'), regime['name']))
            current += timedelta(days=1)

    return dates


def print_regime_summary(regime_configs):
    """Print summary of regimes to be tested"""
    print()
    print("=" * 70)
    print("Regime-Based Parallel Backtest")
    print("=" * 70)
    print()
    print("Regimes:")

    total_days = 0
    for regime in regime_configs:
        dates = generate_regime_dates([regime])
        num_days = len(dates)
        total_days += num_days
        print(f"  • {regime['name']:25s} {regime['start']} to {regime['end']}  ({num_days:2d} days)")

    print()
    print(f"Total trading days: {total_days}")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Submit regime-based backtest to Oracle Cloud Kubernetes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 6 regimes with auto node scaling
  python oci_cloud/tools/submit_regime_backtest.py --nodes 4 --max-parallel 50

  # Run specific regimes only
  python oci_cloud/tools/submit_regime_backtest.py --regimes Strong_Uptrend Shock_Down --nodes 4 --max-parallel 50

  # Run without waiting for completion
  python oci_cloud/tools/submit_regime_backtest.py --nodes 4 --max-parallel 50 --no-wait

  # Run without node scaling (assumes nodes already running)
  python oci_cloud/tools/submit_regime_backtest.py --max-parallel 50

Available regimes:
  Strong_Uptrend (Dec 2023)
  Shock_Down (Jan 2024)
  Event_Driven_HighVol (Jun 2024)
  Correction_RiskOff (Oct 2024)
  Prolonged_Drawdown (Feb 2025)
  Low_Vol_Range (Jul 2025)
        """
    )

    parser.add_argument('--regimes', nargs='+', help='Specific regimes to run (default: all)')
    parser.add_argument('--description', help='Description of this backtest run')
    parser.add_argument('--no-wait', action='store_true', help='Submit and exit without waiting')
    parser.add_argument('--max-parallel', type=int, default=50,
                        help='Max parallel pods (default: 50 for 100 OCPU limit)')
    parser.add_argument('--nodes', type=int, default=None,
                        help='Number of nodes to scale node pool to before starting (e.g., 4). '
                             'If not specified, assumes nodes are already running.')

    args = parser.parse_args()

    # Filter regimes if specified
    if args.regimes:
        selected_regimes = [r for r in REGIME_CONFIGS if r['name'] in args.regimes]

        if not selected_regimes:
            print(f"❌ No valid regimes found. Available: {[r['name'] for r in REGIME_CONFIGS]}")
            sys.exit(1)

        if len(selected_regimes) < len(args.regimes):
            found = {r['name'] for r in selected_regimes}
            missing = set(args.regimes) - found
            print(f"⚠️  Warning: Regimes not found: {missing}")
    else:
        selected_regimes = REGIME_CONFIGS

    # Print summary
    print_regime_summary(selected_regimes)

    # Generate dates for selected regimes
    regime_dates = generate_regime_dates(selected_regimes)
    dates_only = [date for date, _ in regime_dates]

    # Build description
    regime_names = [r['name'] for r in selected_regimes]
    auto_description = f"Regime backtest: {', '.join(regime_names)}"
    description = args.description or auto_description

    # Create submitter and run
    submitter = OCIBacktestSubmitter()

    # Generate run ID
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Print summary
    submitter.print_summary(run_id, dates_only[0], dates_only[-1], len(dates_only), description)

    # Scale up node pool if requested
    if args.nodes is not None:
        if not submitter.scale_up_nodepool(args.nodes):
            print("Warning: Node pool scaling failed, but continuing anyway...")
            print()

    # Package code
    tarball_path = submitter.package_code(run_id)

    # Upload code
    submitter.upload_code(tarball_path, run_id)

    # Submit Kubernetes job
    parallelism = submitter.submit_kubernetes_job(run_id, dates_only, description, args.max_parallel)

    if parallelism is None:
        print("\n❌ Job submission failed")
        return

    if args.no_wait:
        print(f"\n✅ Job submitted (not waiting for completion)")
        print(f"\nMonitor: python oci_cloud/tools/monitor_oci_backtest.py {run_id}")
        print(f"Download: python oci_cloud/tools/download_oci_results.py {run_id}")
        return

    # Monitor progress
    success = submitter.monitor_job(run_id, len(dates_only), parallelism)

    if not success:
        return

    # Download results
    submitter.download_results(run_id)

    print()
    print("=" * 70)
    print("✅ Regime Backtest Complete!")
    print("=" * 70)
    print()
    print(f"Results: ./cloud_results/{run_id}/")
    print()
    print("Next steps:")
    print(f"  1. Analyze results: python tools/analyze_6month_backtest.py cloud_results/{run_id}/")
    print(f"  2. Compare regimes: python tools/regime_orchestrator.py --analyze cloud_results/{run_id}/")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
