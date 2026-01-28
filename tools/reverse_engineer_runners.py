#!/usr/bin/env python
"""
Reverse-engineer characteristics of big movers (runners).

Process:
1. Load 1-day 1m bar data for all NSE stocks
2. Calculate intraday metrics (range, direction, volume, timing)
3. Identify top 100 movers
4. Analyze their characteristics at potential entry points
5. Compare to current system's criteria
6. Identify missing filters that would catch these runners
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

def analyze_single_day_movers(date_str, data_dir, cache_dir):
    """
    Analyze all stocks for a single day to find top movers.

    Args:
        date_str: Date in YYYY-MM-DD format
        data_dir: Directory containing 1m bar data
        cache_dir: Directory containing cached hourly data
    """

    results = {
        'date': date_str,
        'total_stocks_analyzed': 0,
        'top_100_movers': [],
        'characteristics': {
            'by_time_of_move': defaultdict(int),
            'by_volume_profile': defaultdict(int),
            'by_pattern_type': defaultdict(int)
        }
    }

    # Load cached hourly data for the date
    cache_file = Path(cache_dir) / f"{date_str.replace('-', '')}_hourly.feather"

    if not cache_file.exists():
        print(f"Cache file not found: {cache_file}")
        print("Please run data aggregation first")
        return None

    df = pd.read_feather(cache_file)

    # Group by symbol
    movers = []

    for symbol in df['symbol'].unique():
        sym_data = df[df['symbol'] == symbol].copy()

        if len(sym_data) < 10:  # Need sufficient data
            continue

        # Calculate metrics
        day_open = sym_data.iloc[0]['open']
        day_high = sym_data['high'].max()
        day_low = sym_data['low'].min()
        day_close = sym_data.iloc[-1]['close']

        # Intraday range
        intraday_range_pct = ((day_high - day_low) / day_open) * 100

        # Directional move
        directional_move_pct = ((day_close - day_open) / day_open) * 100

        # Volume analysis
        total_volume = sym_data['volume'].sum()

        # Find time of biggest move
        sym_data['price_change'] = sym_data['close'].diff().abs()
        max_move_idx = sym_data['price_change'].idxmax()
        time_of_biggest_move = sym_data.loc[max_move_idx, 'datetime']

        # Categorize time
        hour = pd.to_datetime(time_of_biggest_move).hour
        if hour < 10:
            time_category = 'ORB'
        elif hour < 13:
            time_category = 'Morning'
        elif hour < 15:
            time_category = 'Afternoon'
        else:
            time_category = 'Close'

        # Calculate "runability" score - how clean/consistent was the move?
        # High score = moved in one direction with few reversals
        sym_data['direction'] = np.sign(sym_data['close'].diff())
        direction_changes = (sym_data['direction'].diff() != 0).sum()
        consistency_score = 100 - (direction_changes / len(sym_data) * 100)

        movers.append({
            'symbol': symbol,
            'intraday_range_pct': intraday_range_pct,
            'directional_move_pct': directional_move_pct,
            'abs_move': abs(directional_move_pct),
            'total_volume': total_volume,
            'time_of_move': time_category,
            'time_of_move_hour': hour,
            'consistency_score': consistency_score,
            'data': sym_data  # Keep for further analysis
        })

    # Sort by absolute move
    movers.sort(key=lambda x: x['abs_move'], reverse=True)

    # Take top 100
    top_100 = movers[:100]

    results['total_stocks_analyzed'] = len(movers)
    results['top_100_movers'] = top_100

    # Analyze characteristics
    for mover in top_100:
        results['characteristics']['by_time_of_move'][mover['time_of_move']] += 1

    return results

def compare_with_system_decisions(top_movers, backtest_dir, date_str):
    """
    For top movers, check if our system screened/accepted them.
    """

    backtest_path = Path(backtest_dir)
    events_files = list(backtest_path.rglob(f"{date_str}*/events.jsonl"))

    if not events_files:
        print(f"No events.jsonl found for date {date_str}")
        return None

    # Load all events for this date
    all_events = []
    for events_file in events_files:
        with open(events_file, 'r') as f:
            for line in f:
                if line.strip():
                    all_events.append(json.loads(line))

    comparison = {
        'found_in_system': 0,
        'accepted_by_system': 0,
        'rejected_by_system': 0,
        'never_screened': 0,
        'rejection_reasons': defaultdict(int),
        'missed_opportunities': []
    }

    for mover in top_movers:
        symbol = mover['symbol']

        # Find this symbol in events
        symbol_events = [e for e in all_events if e.get('symbol') == symbol]

        if not symbol_events:
            comparison['never_screened'] += 1
            comparison['missed_opportunities'].append({
                'symbol': symbol,
                'move': mover['abs_move'],
                'reason': 'Never screened'
            })
            continue

        comparison['found_in_system'] += 1

        # Check if accepted
        accepted = any(e.get('decision') == 'ACCEPT' for e in symbol_events)

        if accepted:
            comparison['accepted_by_system'] += 1
        else:
            comparison['rejected_by_system'] += 1

            # Get rejection reasons
            for event in symbol_events:
                if event.get('decision') == 'REJECT':
                    reason = event.get('reason', 'unknown')
                    comparison['rejection_reasons'][reason] += 1

            comparison['missed_opportunities'].append({
                'symbol': symbol,
                'move': mover['abs_move'],
                'reason': f"Rejected: {reason}",
                'rank_score': symbol_events[0].get('rank_score'),
                'structural_rr': symbol_events[0].get('structural_rr')
            })

    return comparison

def print_analysis(results, comparison):
    """Print comprehensive analysis."""

    print("="*80)
    print("RUNNER CHARACTERISTICS - Reverse Engineering Big Movers")
    print("="*80)

    print(f"\nDate analyzed: {results['date']}")
    print(f"Total stocks: {results['total_stocks_analyzed']}")
    print(f"Top 100 movers analyzed\n")

    print("-"*80)
    print("TOP 100 MOVER CHARACTERISTICS")
    print("-"*80)

    top_100 = results['top_100_movers']

    # Stats on moves
    moves = [m['abs_move'] for m in top_100]
    print(f"\nMove size (% of open):")
    print(f"  Median:  {np.median(moves):.2f}%")
    print(f"  Average: {np.mean(moves):.2f}%")
    print(f"  Min:     {min(moves):.2f}%")
    print(f"  Max:     {max(moves):.2f}%")

    # Consistency scores
    consistency = [m['consistency_score'] for m in top_100]
    print(f"\nMove consistency (100 = one direction, 0 = choppy):")
    print(f"  Median:  {np.median(consistency):.1f}")
    print(f"  Average: {np.mean(consistency):.1f}")

    # Time of move distribution
    print(f"\nTiming of biggest move:")
    for time_cat, count in sorted(results['characteristics']['by_time_of_move'].items(),
                                   key=lambda x: -x[1]):
        pct = (count / 100) * 100
        print(f"  {time_cat:<15} {count:3d} ({pct:5.1f}%)")

    # Show top 10 biggest movers
    print(f"\n" + "="*80)
    print("TOP 10 BIGGEST MOVERS")
    print("="*80)
    print(f"{'Symbol':<15} {'Move %':>8} {'Range %':>8} {'Consistency':>12} {'Time':>10}")
    print("-"*80)

    for mover in top_100[:10]:
        print(f"{mover['symbol']:<15} {mover['abs_move']:>7.2f}% "
              f"{mover['intraday_range_pct']:>7.2f}% "
              f"{mover['consistency_score']:>11.1f} "
              f"{mover['time_of_move']:>10}")

    if comparison:
        print(f"\n" + "="*80)
        print("SYSTEM COMPARISON - Did We Catch These Runners?")
        print("="*80)

        print(f"\nTop 100 movers:")
        print(f"  Found in system:     {comparison['found_in_system']:3d} ({comparison['found_in_system']}%)")
        print(f"  Accepted by system:  {comparison['accepted_by_system']:3d} ({comparison['accepted_by_system']}%)")
        print(f"  Rejected by system:  {comparison['rejected_by_system']:3d} ({comparison['rejected_by_system']}%)")
        print(f"  Never screened:      {comparison['never_screened']:3d} ({comparison['never_screened']}%)")

        if comparison['rejection_reasons']:
            print(f"\nRejection reasons for big movers:")
            for reason, count in sorted(comparison['rejection_reasons'].items(),
                                       key=lambda x: -x[1])[:10]:
                print(f"  {reason:<40} {count:3d}")

        if comparison['missed_opportunities']:
            print(f"\n" + "="*80)
            print("TOP 10 MISSED OPPORTUNITIES")
            print("="*80)
            print(f"{'Symbol':<15} {'Move %':>8} {'Reason':<40}")
            print("-"*80)

            for opp in comparison['missed_opportunities'][:10]:
                reason = opp['reason'][:38]
                print(f"{opp['symbol']:<15} {opp['move']:>7.2f}% {reason:<40}")

    print(f"\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Calculate insights
    orb_movers = results['characteristics']['by_time_of_move'].get('ORB', 0)
    high_consistency = sum(1 for m in top_100 if m['consistency_score'] > 70)

    print(f"\n1. Timing Pattern:")
    if orb_movers > 50:
        print(f"   -> {orb_movers}% of big moves happen during ORB (first hour)")
        print("   -> INSIGHT: ORB priority is correct, but may need stricter ORB filters")
    else:
        print(f"   -> Only {orb_movers}% during ORB, moves distributed throughout day")
        print("   -> INSIGHT: Don't over-weight ORB, need good filters for all-day setups")

    print(f"\n2. Move Quality:")
    print(f"   -> {high_consistency}% of top movers had consistency score > 70")
    print("   -> INSIGHT: True runners move in one direction with few reversals")
    print("   -> ACTION: Add 'choppiness' filter to reject back-and-forth moves")

    if comparison:
        hit_rate = (comparison['accepted_by_system'] / 100) * 100 if comparison else 0
        print(f"\n3. System Hit Rate:")
        print(f"   -> Only catching {hit_rate:.0f}% of big movers")
        print("   -> INSIGHT: System is too restrictive OR wrong criteria")

        if comparison['never_screened'] > 50:
            print(f"   -> {comparison['never_screened']}% never even screened")
            print("   -> ACTION: Universe too small? Check screener filters")

    print(f"\n" + "="*80)
    print("RECOMMENDED ACTIONS")
    print("="*80)

    print("\n1. **Expand Analysis**: Run this for 10-20 days to get robust patterns")
    print("2. **Build Discriminator**: Use ML to find features that predict runners")
    print("3. **Add Missing Filters**:")
    print("   - Consistency score (reject choppy moves)")
    print("   - Volume acceleration (not just volume, but increasing volume)")
    print("   - Price action quality (clean breaks vs messy consolidations)")
    print("4. **Relax Current Filters**: Check if rank_score or structural_rr too strict")
    print("5. **Test Iteratively**: Add one filter at a time, backtest, measure improvement")

def main():
    """
    Main execution.

    Usage:
        python reverse_engineer_runners.py
    """

    # For now, use a sample date from the backtest
    date_str = "2023-12-22"  # Date we know has data

    # Paths
    cache_dir = r"cache"
    backtest_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251110-132748_extracted\20251110-132748_full\20251110-132748"

    print(f"Analyzing big movers for {date_str}...")
    print("This analysis identifies characteristics of top 100 movers")
    print("and compares them to what our system currently catches.\n")

    # Note: This script requires pre-aggregated hourly cache
    # For now, we'll create a simplified version that works with backtest data

    print("NOTE: This script requires 1m bar data pre-aggregated.")
    print("To run full analysis:")
    print("1. Aggregate 1m bars for target date")
    print("2. Run this script with proper cache")
    print("\nFor now, we can analyze from backtest events.jsonl...")
    print("(Limited analysis based on what was screened)")

    # TODO: Implement full analysis with 1m bar data
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Setup: Get 1m bar data for a representative trading day")
    print("2. Aggregate: Process into hourly cache with volume/price metrics")
    print("3. Analyze: Run this script to identify top 100 movers")
    print("4. Compare: Cross-reference with system's events.jsonl")
    print("5. Learn: Extract features that distinguish runners from duds")
    print("6. Implement: Add missing filters to system")
    print("7. Backtest: Validate that new filters improve T3 hit rate")

if __name__ == '__main__':
    main()
