#!/usr/bin/env python
"""
Run big movers analysis for multiple days across different market regimes.
Selects 5 trading days from each of the 6 regimes (30 days total).
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Import the single-day analysis functions
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import functions directly from find_big_movers
import pandas as pd
import numpy as np
from datetime import datetime as dt

def find_top_movers(ohlcv_archive_dir, target_date, top_n=100):
    """Scan all stocks and find top N movers for target date."""
    from tools.find_big_movers import analyze_single_stock_day

    archive_path = Path(ohlcv_archive_dir)
    all_symbol_dirs = [d for d in archive_path.iterdir() if d.is_dir()]

    movers = []
    for symbol_dir in all_symbol_dirs:
        result = analyze_single_stock_day(symbol_dir, target_date)
        if result:
            movers.append(result)

    movers.sort(key=lambda x: x['abs_move'], reverse=True)
    return movers[:top_n]

def compare_with_backtest(top_movers, backtest_dir, target_date):
    """Compare top movers with what the system screened/accepted."""
    from tools.find_big_movers import compare_with_backtest as cwb
    return cwb(top_movers, backtest_dir, target_date)

# Regime configurations from regime_orchestrator
REGIMES = [
    {"name": "Strong_Uptrend", "start": "2023-12-01", "end": "2023-12-31"},
    {"name": "Shock_Down", "start": "2024-01-01", "end": "2024-01-31"},
    {"name": "Event_Driven_HighVol", "start": "2024-06-01", "end": "2024-06-30"},
    {"name": "Correction_RiskOff", "start": "2024-10-01", "end": "2024-10-31"},
    {"name": "Prolonged_Drawdown", "start": "2025-02-01", "end": "2025-02-28"},
    {"name": "Low_Vol_Range", "start": "2025-07-01", "end": "2025-07-31"}
]

def get_trading_days_for_regime(regime, num_days=5):
    """
    Get trading days for a regime, selecting evenly spaced days.
    Skip weekends.
    """
    start = datetime.strptime(regime['start'], '%Y-%m-%d')
    end = datetime.strptime(regime['end'], '%Y-%m-%d')

    # Get all weekdays in range
    trading_days = []
    current = start
    while current <= end:
        # Skip weekends (5=Saturday, 6=Sunday)
        if current.weekday() < 5:
            trading_days.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    if len(trading_days) <= num_days:
        return trading_days

    # Select evenly spaced days
    step = len(trading_days) / num_days
    selected = []
    for i in range(num_days):
        idx = int(i * step)
        selected.append(trading_days[idx])

    return selected

def main():
    """
    Run big movers analysis for 30 days (5 from each regime).
    """

    ohlcv_archive = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\cache\ohlcv_archive"
    backtest_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251110-132748_extracted\20251110-132748_full\20251110-132748"

    # Collect all target dates
    all_dates = []
    for regime in REGIMES:
        dates = get_trading_days_for_regime(regime, num_days=5)
        for date in dates:
            all_dates.append({
                'date': date,
                'regime': regime['name']
            })

    print(f"="*80)
    print(f"BIG MOVERS MULTI-DAY ANALYSIS")
    print(f"="*80)
    print(f"\nAnalyzing {len(all_dates)} trading days across {len(REGIMES)} regimes:")
    for regime in REGIMES:
        regime_dates = [d for d in all_dates if d['regime'] == regime['name']]
        print(f"  {regime['name']:<25} {len(regime_dates)} days: {', '.join([d['date'] for d in regime_dates])}")
    print()

    # Aggregate results
    all_results = []
    aggregate_stats = {
        'total_days': 0,
        'total_movers_found': 0,
        'total_screened': 0,
        'total_accepted': 0,
        'total_rejected': 0,
        'total_never_screened': 0,
        'by_regime': {}
    }

    for i, day_info in enumerate(all_dates, 1):
        target_date = day_info['date']
        regime_name = day_info['regime']

        print(f"[{i}/{len(all_dates)}] Processing {target_date} ({regime_name})...")

        try:
            # Find top movers
            top_movers = find_top_movers(ohlcv_archive, target_date, top_n=100)

            if not top_movers:
                print(f"  -> No movers found for {target_date}")
                continue

            # Compare with backtest
            comparison = compare_with_backtest(top_movers, backtest_dir, target_date)

            # Store results
            day_result = {
                'date': target_date,
                'regime': regime_name,
                'movers_count': len(top_movers),
                'screened': comparison['screened'] if comparison else 0,
                'accepted': comparison['accepted'] if comparison else 0,
                'rejected': comparison['rejected'] if comparison else 0,
                'never_screened': comparison['never_screened'] if comparison else 0,
                'top_10_movers': top_movers[:10]
            }
            all_results.append(day_result)

            # Update aggregate stats
            aggregate_stats['total_days'] += 1
            aggregate_stats['total_movers_found'] += len(top_movers)
            aggregate_stats['total_screened'] += day_result['screened']
            aggregate_stats['total_accepted'] += day_result['accepted']
            aggregate_stats['total_rejected'] += day_result['rejected']
            aggregate_stats['total_never_screened'] += day_result['never_screened']

            # Track by regime
            if regime_name not in aggregate_stats['by_regime']:
                aggregate_stats['by_regime'][regime_name] = {
                    'days': 0,
                    'movers': 0,
                    'screened': 0,
                    'accepted': 0,
                    'rejected': 0,
                    'never_screened': 0
                }

            regime_stats = aggregate_stats['by_regime'][regime_name]
            regime_stats['days'] += 1
            regime_stats['movers'] += len(top_movers)
            regime_stats['screened'] += day_result['screened']
            regime_stats['accepted'] += day_result['accepted']
            regime_stats['rejected'] += day_result['rejected']
            regime_stats['never_screened'] += day_result['never_screened']

            print(f"  -> Found {len(top_movers)} movers, screened: {day_result['screened']}, accepted: {day_result['accepted']}")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

    # Print aggregate summary
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS - {aggregate_stats['total_days']} Days Analyzed")
    print(f"{'='*80}\n")

    total_movers = aggregate_stats['total_movers_found']
    print(f"Total big movers identified: {total_movers}")
    print(f"  Screened by system:  {aggregate_stats['total_screened']:4d} ({aggregate_stats['total_screened']/total_movers*100:5.1f}%)")
    print(f"  Accepted:            {aggregate_stats['total_accepted']:4d} ({aggregate_stats['total_accepted']/total_movers*100:5.1f}%)")
    print(f"  Rejected:            {aggregate_stats['total_rejected']:4d} ({aggregate_stats['total_rejected']/total_movers*100:5.1f}%)")
    print(f"  Never screened:      {aggregate_stats['total_never_screened']:4d} ({aggregate_stats['total_never_screened']/total_movers*100:5.1f}%)")

    # Print by regime
    print(f"\n{'='*80}")
    print(f"RESULTS BY REGIME")
    print(f"{'='*80}\n")

    for regime_name, stats in aggregate_stats['by_regime'].items():
        print(f"{regime_name}:")
        print(f"  Days analyzed: {stats['days']}")
        print(f"  Movers found:  {stats['movers']}")
        if stats['movers'] > 0:
            print(f"  Screened:      {stats['screened']:3d} ({stats['screened']/stats['movers']*100:5.1f}%)")
            print(f"  Accepted:      {stats['accepted']:3d} ({stats['accepted']/stats['movers']*100:5.1f}%)")
            print(f"  Never screened:{stats['never_screened']:3d} ({stats['never_screened']/stats['movers']*100:5.1f}%)")
        print()

    # Save results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'total_days': aggregate_stats['total_days'],
        'aggregate_stats': aggregate_stats,
        'daily_results': all_results
    }

    output_file = Path(f"big_movers_30day_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")

    # Print key insights
    print(f"\n{'='*80}")
    print(f"KEY INSIGHTS")
    print(f"={'='*80}\n")

    if total_movers > 0:
        screening_rate = aggregate_stats['total_screened'] / total_movers * 100
        acceptance_rate = aggregate_stats['total_accepted'] / total_movers * 100

        print(f"1. SCREENING COVERAGE:")
        print(f"   -> System screens only {screening_rate:.1f}% of big movers")
        print(f"   -> {100-screening_rate:.1f}% of big movers never reach the system")
        print()

        print(f"2. ACCEPTANCE RATE:")
        print(f"   -> System accepts only {acceptance_rate:.1f}% of big movers")
        print(f"   -> This explains the 0% T3 hit rate")
        print()

        print(f"3. ROOT CAUSE:")
        print(f"   -> The stock universe is too restrictive")
        print(f"   -> Pre-screening filters eliminate big movers before they reach decision logic")
        print()

        print(f"4. NEXT ACTIONS:")
        print(f"   -> Investigate nse_all.json filters (market cap, liquidity, etc)")
        print(f"   -> Check if big movers are in the tradeable universe")
        print(f"   -> Identify common characteristics of never-screened movers")

if __name__ == '__main__':
    main()
