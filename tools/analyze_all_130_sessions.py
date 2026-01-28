#!/usr/bin/env python
"""
Comprehensive analysis of ALL 130+ trading sessions.

For each session:
1. Find top 100 big movers
2. Compare with system decisions (screened/accepted/rejected)
3. For big movers, analyze entry characteristics
4. Aggregate patterns across all 130 days

This provides statistical confidence before making system changes.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Import big mover analysis
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.find_big_movers import analyze_single_stock_day, compare_with_backtest

def get_all_trading_days_from_backtest(backtest_dir):
    """Get all trading days from backtest directory."""
    backtest_path = Path(backtest_dir)

    # Find all date directories (format: YYYY-MM-DD)
    date_dirs = []
    for item in backtest_path.iterdir():
        if item.is_dir() and item.name.startswith('202'):
            # Extract date from directory name
            try:
                datetime.strptime(item.name, '%Y-%m-%d')
                date_dirs.append(item.name)
            except:
                continue

    date_dirs.sort()
    return date_dirs

def analyze_runner_at_entry(symbol, target_date, ohlcv_archive_dir):
    """
    For a big mover, analyze entry characteristics at multiple time points.
    Returns characteristics at optimal entry point.
    """

    symbol_clean = symbol.replace('NSE:', '').replace('.NS', '')
    symbol_dir = Path(ohlcv_archive_dir) / f"{symbol_clean}.NS"

    if not symbol_dir.exists():
        return None

    ohlcv_1m_file = symbol_dir / f"{symbol_clean}.NS_1minutes.feather"

    if not ohlcv_1m_file.exists():
        return None

    try:
        df = pd.read_feather(ohlcv_1m_file)
        df['datetime'] = pd.to_datetime(df['date'])
        df['day'] = df['datetime'].dt.date

        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
        day_data = df[df['day'] == target_date_obj].copy()

        if len(day_data) < 50:
            return None

        day_data = day_data.sort_values('datetime').reset_index(drop=True)

        # Find day open and final metrics
        day_open = day_data.iloc[0]['open']
        day_close = day_data.iloc[-1]['close']
        final_move_pct = ((day_close - day_open) / day_open) * 100

        # Key time points to check
        check_times = [
            ('09:20', 'ORB_START'),
            ('09:30', 'ORB_10MIN'),
            ('09:45', 'ORB_25MIN'),
            ('10:00', 'ORB_END'),
        ]

        best_entry = None

        for time_str, label in check_times:
            # Get data up to this time
            cutoff = pd.Timestamp(f"{target_date} {time_str}", tz=day_data['datetime'].iloc[0].tz)
            data_so_far = day_data[day_data['datetime'] <= cutoff].copy()

            if len(data_so_far) < 5:
                continue

            # Calculate metrics
            current_price = data_so_far.iloc[-1]['close']
            high_so_far = data_so_far['high'].max()
            low_so_far = data_so_far['low'].min()

            range_pct = ((high_so_far - low_so_far) / day_open) * 100
            move_from_open_pct = ((current_price - day_open) / day_open) * 100

            # Volume acceleration
            avg_volume = data_so_far['volume'].mean() if 'volume' in data_so_far.columns else 0
            recent_volume = data_so_far.tail(5)['volume'].mean() if 'volume' in data_so_far.columns else 0
            volume_acceleration = (recent_volume / avg_volume) if avg_volume > 0 else 1.0

            # Momentum (last 5 bars)
            if len(data_so_far) >= 5:
                price_5bars_ago = data_so_far.iloc[-5]['close']
                momentum_5bar = ((current_price - price_5bars_ago) / price_5bars_ago) * 100
            else:
                momentum_5bar = 0

            # Consistency score
            data_so_far['direction'] = np.sign(data_so_far['close'].diff())
            direction_changes = (data_so_far['direction'].diff() != 0).sum()
            consistency = 100 - (direction_changes / len(data_so_far) * 100) if len(data_so_far) > 0 else 0

            # Setup type
            setup_type = 'orb_forming'
            if abs(move_from_open_pct) > 0.5 and volume_acceleration > 1.2:
                setup_type = 'orb_breakout'

            # Remaining move
            remaining_move_pct = final_move_pct - move_from_open_pct

            entry_point = {
                'time': time_str,
                'label': label,
                'move_from_open_pct': move_from_open_pct,
                'range_pct': range_pct,
                'momentum_5bar': momentum_5bar,
                'consistency': consistency,
                'volume_acceleration': volume_acceleration,
                'setup_type': setup_type,
                'remaining_move_pct': remaining_move_pct
            }

            # Find best entry (earliest with signal + >70% move remaining)
            if (abs(momentum_5bar) > 0.3 and
                volume_acceleration > 1.2 and
                (abs(remaining_move_pct) / abs(final_move_pct) * 100) > 70 if final_move_pct != 0 else True):
                best_entry = entry_point
                break

        # If no optimal entry found, use ORB_START
        if not best_entry and len(check_times) > 0:
            for time_str, label in check_times:
                cutoff = pd.Timestamp(f"{target_date} {time_str}", tz=day_data['datetime'].iloc[0].tz)
                data_so_far = day_data[day_data['datetime'] <= cutoff].copy()
                if len(data_so_far) >= 5:
                    # Use simplified metrics
                    current_price = data_so_far.iloc[-1]['close']
                    move_from_open_pct = ((current_price - day_open) / day_open) * 100
                    best_entry = {
                        'time': time_str,
                        'label': label,
                        'move_from_open_pct': move_from_open_pct,
                        'range_pct': 0,
                        'momentum_5bar': 0,
                        'consistency': 0,
                        'volume_acceleration': 1.0,
                        'setup_type': 'orb_forming',
                        'remaining_move_pct': final_move_pct - move_from_open_pct
                    }
                    break

        if not best_entry:
            return None

        return {
            'symbol': symbol,
            'date': target_date,
            'final_move_pct': final_move_pct,
            'optimal_entry': best_entry
        }

    except Exception as e:
        return None

def main():
    """
    Main execution for all 130+ sessions.
    """

    ohlcv_archive = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\cache\ohlcv_archive"
    backtest_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251110-132748_extracted\20251110-132748_full\20251110-132748"

    print(f"="*80)
    print(f"COMPREHENSIVE 130+ SESSION ANALYSIS")
    print(f"="*80)
    print()

    # Get all trading days
    all_trading_days = get_all_trading_days_from_backtest(backtest_dir)

    # Exclude already analyzed dates from big_movers_30day_analysis.json
    already_analyzed = {
        '2023-12-01', '2023-12-07', '2023-12-13', '2023-12-19',
        '2024-01-01', '2024-01-05', '2024-01-12', '2024-01-18', '2024-01-25',
        '2024-06-03', '2024-06-07', '2024-06-13', '2024-06-19', '2024-06-25',
        '2024-10-01', '2024-10-07', '2024-10-14', '2024-10-18', '2024-10-25',
        '2025-02-03', '2025-02-07', '2025-02-13', '2025-02-19', '2025-02-25',
        '2025-07-01', '2025-07-07', '2025-07-14', '2025-07-18', '2025-07-25'
    }

    all_trading_days = [d for d in all_trading_days if d not in already_analyzed]

    print(f"Found {len(all_trading_days)} NEW trading days to analyze (excluding 29 already done)")
    print(f"Date range: {all_trading_days[0]} to {all_trading_days[-1]}")
    print()

    # Aggregate stats
    aggregate = {
        'total_days': 0,
        'total_movers': 0,
        'total_screened': 0,
        'total_accepted': 0,
        'total_rejected': 0,
        'total_never_screened': 0,
        'runner_characteristics': {
            'setup_types': defaultdict(int),
            'entry_times': defaultdict(int),
            'momentum_values': [],
            'volume_accel_values': [],
            'consistency_values': [],
            'range_values': []
        }
    }

    daily_results = []

    # Process each day
    for i, target_date in enumerate(all_trading_days, 1):
        print(f"[{i}/{len(all_trading_days)}] Processing {target_date}...", flush=True)

        try:
            # Find top 100 movers for this day
            archive_path = Path(ohlcv_archive)
            all_symbol_dirs = [d for d in archive_path.iterdir() if d.is_dir()]

            movers = []
            for symbol_dir in all_symbol_dirs:
                result = analyze_single_stock_day(symbol_dir, target_date)
                if result:
                    movers.append(result)

            movers.sort(key=lambda x: x['abs_move'], reverse=True)
            top_movers = movers[:100]

            if not top_movers:
                print(f"  -> No movers found")
                continue

            # Compare with backtest
            comparison = compare_with_backtest(top_movers, backtest_dir, target_date)

            if not comparison:
                print(f"  -> No backtest events found")
                continue

            # Analyze entry characteristics for top 10 movers
            runner_analyses = []
            for mover in top_movers[:10]:
                symbol = f"NSE:{mover['symbol'].replace('.NS', '')}"
                analysis = analyze_runner_at_entry(symbol, target_date, ohlcv_archive)
                if analysis:
                    runner_analyses.append(analysis)

            # Update aggregates
            aggregate['total_days'] += 1
            aggregate['total_movers'] += len(top_movers)
            aggregate['total_screened'] += comparison['screened']
            aggregate['total_accepted'] += comparison['accepted']
            aggregate['total_rejected'] += comparison['rejected']
            aggregate['total_never_screened'] += comparison['never_screened']

            # Aggregate runner characteristics
            for analysis in runner_analyses:
                entry = analysis['optimal_entry']
                aggregate['runner_characteristics']['setup_types'][entry['setup_type']] += 1
                aggregate['runner_characteristics']['entry_times'][entry['label']] += 1
                aggregate['runner_characteristics']['momentum_values'].append(entry['momentum_5bar'])
                aggregate['runner_characteristics']['volume_accel_values'].append(entry['volume_acceleration'])
                aggregate['runner_characteristics']['consistency_values'].append(entry['consistency'])
                aggregate['runner_characteristics']['range_values'].append(entry['range_pct'])

            # Store daily result
            daily_results.append({
                'date': target_date,
                'movers_count': len(top_movers),
                'screened': comparison['screened'],
                'accepted': comparison['accepted'],
                'rejected': comparison['rejected'],
                'never_screened': comparison['never_screened'],
                'runner_analyses_count': len(runner_analyses)
            })

            print(f"  -> {len(top_movers)} movers, {comparison['screened']} screened, {comparison['accepted']} accepted, {len(runner_analyses)} analyzed")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

    # Print aggregate results
    print()
    print("="*80)
    print(f"AGGREGATE RESULTS - {aggregate['total_days']} Days")
    print("="*80)
    print()

    print(f"Total big movers: {aggregate['total_movers']}")
    print(f"  Screened:       {aggregate['total_screened']:4d} ({aggregate['total_screened']/aggregate['total_movers']*100:5.1f}%)")
    print(f"  Accepted:       {aggregate['total_accepted']:4d} ({aggregate['total_accepted']/aggregate['total_movers']*100:5.1f}%)")
    print(f"  Rejected:       {aggregate['total_rejected']:4d} ({aggregate['total_rejected']/aggregate['total_movers']*100:5.1f}%)")
    print(f"  Never screened: {aggregate['total_never_screened']:4d} ({aggregate['total_never_screened']/aggregate['total_movers']*100:5.1f}%)")
    print()

    # Runner characteristics
    rc = aggregate['runner_characteristics']

    print("="*80)
    print("RUNNER ENTRY CHARACTERISTICS")
    print("="*80)
    print()

    print("Setup Type Distribution:")
    total_setups = sum(rc['setup_types'].values())
    for setup, count in sorted(rc['setup_types'].items(), key=lambda x: -x[1]):
        pct = (count / total_setups * 100) if total_setups > 0 else 0
        print(f"  {setup:<20} {count:4d} ({pct:5.1f}%)")
    print()

    print("Entry Time Distribution:")
    total_entries = sum(rc['entry_times'].values())
    for time_label, count in sorted(rc['entry_times'].items(), key=lambda x: -x[1]):
        pct = (count / total_entries * 100) if total_entries > 0 else 0
        print(f"  {time_label:<15} {count:4d} ({pct:5.1f}%)")
    print()

    print("Parameter Statistics:")
    print(f"{'Parameter':<25} {'Median':>10} {'Mean':>10} {'P25':>10} {'P75':>10}")
    print("-"*80)

    params = [
        ('momentum_5bar', rc['momentum_values']),
        ('volume_acceleration', rc['volume_accel_values']),
        ('consistency', rc['consistency_values']),
        ('range_at_entry', rc['range_values'])
    ]

    for param_name, values in params:
        if values:
            print(f"{param_name:<25} {np.median(values):>10.2f} {np.mean(values):>10.2f} "
                  f"{np.percentile(values, 25):>10.2f} {np.percentile(values, 75):>10.2f}")

    print()
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()

    # Calculate key metrics
    screening_rate = (aggregate['total_screened'] / aggregate['total_movers'] * 100) if aggregate['total_movers'] > 0 else 0

    print(f"1. SCREENING FAILURE:")
    print(f"   -> System screens only {screening_rate:.1f}% of big movers")
    print(f"   -> {100-screening_rate:.1f}% never reach the system")
    print()

    if rc['setup_types']:
        dominant_setup = max(rc['setup_types'].items(), key=lambda x: x[1])[0]
        dominant_pct = (rc['setup_types'][dominant_setup] / total_setups * 100)
        print(f"2. DOMINANT SETUP:")
        print(f"   -> {dominant_setup} accounts for {dominant_pct:.1f}% of runners")
        print()

    if rc['entry_times']:
        orb_count = sum(count for label, count in rc['entry_times'].items() if 'ORB' in label)
        orb_pct = (orb_count / total_entries * 100) if total_entries > 0 else 0
        print(f"3. ENTRY TIMING:")
        print(f"   -> {orb_pct:.1f}% of optimal entries during ORB")
        print()

    if rc['momentum_values']:
        median_momentum = np.median(rc['momentum_values'])
        print(f"4. MOMENTUM THRESHOLD:")
        print(f"   -> Median momentum at entry: {median_momentum:.2f}%")
        print()

    # Save results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'total_days_analyzed': aggregate['total_days'],
        'aggregate_stats': {
            'total_movers': aggregate['total_movers'],
            'total_screened': aggregate['total_screened'],
            'total_accepted': aggregate['total_accepted'],
            'total_rejected': aggregate['total_rejected'],
            'total_never_screened': aggregate['total_never_screened'],
            'screening_rate_pct': screening_rate
        },
        'runner_characteristics': {
            'setup_types': dict(rc['setup_types']),
            'entry_times': dict(rc['entry_times']),
            'parameter_stats': {
                param: {
                    'median': float(np.median(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'p25': float(np.percentile(values, 25)),
                    'p75': float(np.percentile(values, 75))
                }
                for param, values in params if values
            }
        },
        'daily_results': daily_results
    }

    output_file = Path("runner_analysis_130_sessions.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")
    print()
    print("="*80)

if __name__ == '__main__':
    main()
