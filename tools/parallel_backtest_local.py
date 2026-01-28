#!/usr/bin/env python
"""
Run parallel backtests locally using multiprocessing.
This is the SIMPLEST solution - no AWS, no Docker, just pure Python.

Usage:
    python tools/parallel_backtest_local.py --date-list aws/date_lists/quick_test.txt --workers 4
    python tools/parallel_backtest_local.py --date-list aws/date_lists/june_2024.txt --workers 8
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_single_backtest(date):
    """Run backtest for a single date"""
    print(f'[{date}] Starting backtest...')
    start_time = datetime.now()

    cmd = [
        sys.executable,
        'main.py',
        '--dry-run',
        '--session-date', date,
        '--from-hhmm', '09:25',
        '--to-hhmm', '15:15'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per day
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            print(f'[{date}] SUCCESS in {elapsed:.1f}s ({elapsed/60:.1f}min)')
            return {'date': date, 'status': 'success', 'elapsed': elapsed}
        else:
            print(f'[{date}] FAILED in {elapsed:.1f}s')
            print(f'[{date}] Error: {result.stderr[:200]}')
            return {'date': date, 'status': 'failed', 'elapsed': elapsed, 'error': result.stderr}

    except subprocess.TimeoutExpired:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f'[{date}] TIMEOUT after {elapsed:.1f}s')
        return {'date': date, 'status': 'timeout', 'elapsed': elapsed}

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f'[{date}] ERROR: {str(e)}')
        return {'date': date, 'status': 'error', 'elapsed': elapsed, 'error': str(e)}

def load_date_list(file_path):
    """Load dates from text file"""
    dates = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                dates.append(line)
    return dates

def main():
    parser = argparse.ArgumentParser(
        description='Run parallel backtests locally',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Quick test with 4 parallel workers
  python tools/parallel_backtest_local.py --date-list aws/date_lists/quick_test.txt --workers 4

  # Full month with 8 parallel workers (faster!)
  python tools/parallel_backtest_local.py --date-list aws/date_lists/june_2024.txt --workers 8

Notes:
  - Each worker runs one backtest at a time
  - More workers = faster overall, but uses more CPU/memory
  - Recommended: workers = number of CPU cores (or less)
  - Results saved in logs/{session_id}/ as usual
'''
    )

    parser.add_argument('--date-list', required=True, help='Path to date list file')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')

    args = parser.parse_args()

    # Load dates
    dates = load_date_list(args.date_list)

    print('='*60)
    print('Parallel Backtest Runner (Local)')
    print('='*60)
    print(f'Date list: {args.date_list}')
    print(f'Total dates: {len(dates)}')
    print(f'Workers: {args.workers}')
    print(f'Date range: {dates[0]} to {dates[-1]}')
    print()

    # Confirm
    response = input(f'Run {len(dates)} backtests with {args.workers} parallel workers? (y/n): ')
    if response.lower() != 'y':
        print('Cancelled')
        return

    print()
    print('Starting parallel backtests...')
    print()

    start_time = datetime.now()
    results = []

    # Run in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all jobs
        futures = {executor.submit(run_single_backtest, date): date for date in dates}

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Summary
    elapsed_total = (datetime.now() - start_time).total_seconds()

    print()
    print('='*60)
    print('RESULTS SUMMARY')
    print('='*60)
    print(f'Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)')
    print(f'Total dates: {len(dates)}')
    print()

    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    timeout = [r for r in results if r['status'] == 'timeout']
    errors = [r for r in results if r['status'] == 'error']

    print(f'Success: {len(success)}')
    print(f'Failed: {len(failed)}')
    print(f'Timeout: {len(timeout)}')
    print(f'Errors: {len(errors)}')
    print()

    if success:
        avg_time = sum(r['elapsed'] for r in success) / len(success)
        print(f'Average time per backtest: {avg_time:.1f}s ({avg_time/60:.1f}min)')
        print()

    if failed:
        print('Failed dates:')
        for r in failed:
            print(f'  - {r["date"]}')
        print()

    if timeout:
        print('Timed out dates:')
        for r in timeout:
            print(f'  - {r["date"]}')
        print()

    if errors:
        print('Error dates:')
        for r in errors:
            print(f'  - {r["date"]}: {r.get("error", "Unknown error")}')
        print()

    print('Results saved in logs/ directory')
    print()

if __name__ == '__main__':
    main()
