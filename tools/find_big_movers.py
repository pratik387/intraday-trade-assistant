#!/usr/bin/env python
"""
Find top 100 movers for a given date using cached 1m data.
Analyze their characteristics and compare to system decisions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict

def analyze_single_stock_day(symbol_dir, target_date):
    """
    Analyze one stock for one day.

    Returns metrics if data exists for target_date, else None.
    """

    ohlcv_1m_file = symbol_dir / f"{symbol_dir.name}_1minutes.feather"

    if not ohlcv_1m_file.exists():
        return None

    try:
        df = pd.read_feather(ohlcv_1m_file)

        # Filter to target date
        df['day'] = pd.to_datetime(df['date']).dt.date
        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()

        day_data = df[df['day'] == target_date_obj].copy()

        if len(day_data) < 10:  # Need meaningful data
            return None

        # Calculate intraday metrics
        day_open = day_data.iloc[0]['open']
        day_high = day_data['high'].max()
        day_low = day_data['low'].min()
        day_close = day_data.iloc[-1]['close']

        # Range and directional move
        intraday_range_pct = ((day_high - day_low) / day_open) * 100
        directional_move_pct = ((day_close - day_open) / day_open) * 100

        # Movement consistency
        day_data['price_change'] = day_data['close'].diff()
        day_data['direction'] = np.sign(day_data['price_change'])
        direction_changes = (day_data['direction'].diff() != 0).sum()
        consistency_score = 100 - (direction_changes / len(day_data) * 100)

        # Time of biggest move
        day_data['abs_change'] = day_data['price_change'].abs()
        max_move_idx = day_data['abs_change'].idxmax()
        time_of_move = pd.to_datetime(day_data.loc[max_move_idx, 'date']).hour

        if time_of_move < 10:
            time_category = 'ORB'
        elif time_of_move < 13:
            time_category = 'Morning'
        elif time_of_move < 15:
            time_category = 'Afternoon'
        else:
            time_category = 'Close'

        # Volume (if available)
        total_volume = day_data['volume'].sum() if 'volume' in day_data.columns else 0

        return {
            'symbol': symbol_dir.name,
            'intraday_range_pct': intraday_range_pct,
            'directional_move_pct': directional_move_pct,
            'abs_move': abs(directional_move_pct),
            'consistency_score': consistency_score,
            'time_of_move': time_category,
            'time_of_move_hour': time_of_move,
            'total_volume': total_volume
        }

    except Exception as e:
        print(f"Error processing {symbol_dir.name}: {e}")
        return None

def find_top_movers(ohlcv_archive_dir, target_date, top_n=100):
    """
    Scan all stocks and find top N movers for target date.
    """

    archive_path = Path(ohlcv_archive_dir)
    all_symbol_dirs = [d for d in archive_path.iterdir() if d.is_dir()]

    print(f"Scanning {len(all_symbol_dirs)} stocks for {target_date}...")

    movers = []

    for i, symbol_dir in enumerate(all_symbol_dirs):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(all_symbol_dirs)}...")

        result = analyze_single_stock_day(symbol_dir, target_date)
        if result:
            movers.append(result)

    # Sort by absolute move
    movers.sort(key=lambda x: x['abs_move'], reverse=True)

    print(f"\nFound {len(movers)} stocks with data for {target_date}")
    print(f"Returning top {top_n} movers\n")

    return movers[:top_n]

def compare_with_backtest(top_movers, backtest_dir, target_date):
    """
    Compare top movers with what the system screened/accepted.
    """

    # Find events.jsonl for target date
    backtest_path = Path(backtest_dir)

    # Date folder might be YYYY-MM-DD format
    date_folder = target_date
    events_files = list(backtest_path.rglob(f"{date_folder}*/events.jsonl"))

    if not events_files:
        print(f"No events.jsonl found for {target_date} in backtest")
        return None

    print(f"Loading events from: {events_files[0]}")

    # Load events
    all_events = []
    with open(events_files[0], 'r') as f:
        for line in f:
            if line.strip():
                all_events.append(json.loads(line))

    comparison = {
        'screened': 0,
        'accepted': 0,
        'rejected': 0,
        'never_screened': 0,
        'rejection_reasons': defaultdict(int),
        'missed_opportunities': []
    }

    for mover in top_movers:
        # Convert symbol format (AAVAS.NS -> NSE:AAVAS)
        symbol_nse = f"NSE:{mover['symbol'].replace('.NS', '')}"

        # Find this symbol in events
        symbol_events = [e for e in all_events if e.get('symbol') == symbol_nse]

        if not symbol_events:
            comparison['never_screened'] += 1
            comparison['missed_opportunities'].append({
                'symbol': symbol_nse,
                'move': mover['abs_move'],
                'reason': 'Never screened',
                'consistency': mover['consistency_score'],
                'time': mover['time_of_move']
            })
            continue

        comparison['screened'] += 1

        # Check if accepted
        accepted = any(e.get('decision') == 'ACCEPT' for e in symbol_events)

        if accepted:
            comparison['accepted'] += 1
        else:
            comparison['rejected'] += 1

            # Get rejection reasons
            for event in symbol_events:
                if event.get('decision') == 'REJECT':
                    reason = event.get('reason', 'unknown')
                    comparison['rejection_reasons'][reason] += 1

                    comparison['missed_opportunities'].append({
                        'symbol': symbol_nse,
                        'move': mover['abs_move'],
                        'reason': f"Rejected: {reason}",
                        'rank_score': event.get('rank_score'),
                        'structural_rr': event.get('structural_rr'),
                        'consistency': mover['consistency_score'],
                        'time': mover['time_of_move']
                    })
                    break

    return comparison

def print_analysis(top_movers, comparison):
    """Print analysis results."""

    print("="*80)
    print("BIG MOVER ANALYSIS - Runner Characteristics")
    print("="*80)

    print(f"\nTop {len(top_movers)} movers:")
    print(f"  Median move:       {np.median([m['abs_move'] for m in top_movers]):.2f}%")
    print(f"  Average move:      {np.mean([m['abs_move'] for m in top_movers]):.2f}%")
    print(f"  Largest move:      {max(m['abs_move'] for m in top_movers):.2f}%")

    # Consistency
    consistency = [m['consistency_score'] for m in top_movers]
    print(f"\nMovement consistency (100 = clean, 0 = choppy):")
    print(f"  Median:  {np.median(consistency):.1f}")
    print(f"  Average: {np.mean(consistency):.1f}")

    # Time distribution
    time_dist = defaultdict(int)
    for m in top_movers:
        time_dist[m['time_of_move']] += 1

    print(f"\nTiming of big moves:")
    for time_cat in ['ORB', 'Morning', 'Afternoon', 'Close']:
        count = time_dist[time_cat]
        pct = (count / len(top_movers)) * 100
        print(f"  {time_cat:<12} {count:3d} ({pct:5.1f}%)")

    # Top 10
    print("\n" + "="*80)
    print("TOP 10 BIGGEST MOVERS")
    print("="*80)
    print(f"{'Symbol':<20} {'Move %':>8} {'Consistency':>12} {'Time':>10}")
    print("-"*80)

    for mover in top_movers[:10]:
        print(f"{mover['symbol']:<20} {mover['abs_move']:>7.2f}% "
              f"{mover['consistency_score']:>11.1f} {mover['time_of_move']:>10}")

    if comparison:
        print("\n" + "="*80)
        print("SYSTEM COMPARISON - Did We Catch These?")
        print("="*80)

        total = len(top_movers)
        print(f"\nTop {total} movers:")
        print(f"  Screened by system:  {comparison['screened']:3d} ({comparison['screened']/total*100:.1f}%)")
        print(f"  Accepted:            {comparison['accepted']:3d} ({comparison['accepted']/total*100:.1f}%)")
        print(f"  Rejected:            {comparison['rejected']:3d} ({comparison['rejected']/total*100:.1f}%)")
        print(f"  Never screened:      {comparison['never_screened']:3d} ({comparison['never_screened']/total*100:.1f}%)")

        if comparison['rejection_reasons']:
            print(f"\nTop rejection reasons:")
            for reason, count in sorted(comparison['rejection_reasons'].items(),
                                       key=lambda x: -x[1])[:5]:
                print(f"  {reason:<50} {count:3d}")

        if comparison['missed_opportunities']:
            print("\n" + "="*80)
            print("TOP 10 MISSED OPPORTUNITIES")
            print("="*80)
            print(f"{'Symbol':<15} {'Move %':>8} {'Consistency':>12} {'Reason':<40}")
            print("-"*80)

            for opp in comparison['missed_opportunities'][:10]:
                reason = opp['reason'][:38]
                print(f"{opp['symbol']:<15} {opp['move']:>7.2f}% "
                      f"{opp.get('consistency', 0):>11.1f} {reason:<40}")

def main():
    # Configuration
    target_date = "2024-01-02"  # Use a date from our backtest
    ohlcv_archive = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\cache\ohlcv_archive"
    backtest_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251112-023811_extracted\20251112-023811_full\20251112-023811"

    print(f"Analyzing big movers for {target_date}")
    print(f"Using OHLCV archive: {ohlcv_archive}")
    print()

    # Find top movers
    top_movers = find_top_movers(ohlcv_archive, target_date, top_n=100)

    if not top_movers:
        print("No movers found. Check date and data availability.")
        return

    # Compare with backtest
    comparison = compare_with_backtest(top_movers, backtest_dir, target_date)

    # Print analysis
    print_analysis(top_movers, comparison)

    # Save results - convert numpy types to native Python
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    output = {
        'date': target_date,
        'top_movers': convert_to_native(top_movers),
        'comparison': {
            'screened': comparison['screened'] if comparison else 0,
            'accepted': comparison['accepted'] if comparison else 0,
            'rejected': comparison['rejected'] if comparison else 0,
            'never_screened': comparison['never_screened'] if comparison else 0,
            'rejection_reasons': dict(comparison['rejection_reasons']) if comparison else {},
            'missed_opportunities': convert_to_native(comparison['missed_opportunities']) if comparison else []
        }
    }

    output_file = Path(f"big_movers_{target_date}.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Run this for multiple dates to find patterns")
    print("2. Analyze rejected movers - what filters blocked them?")
    print("3. Check if consistency_score correlates with T3 potential")
    print("4. Add volume/ADX analysis at entry point")
    print("5. Build discriminator: runners vs duds")

if __name__ == '__main__':
    main()
