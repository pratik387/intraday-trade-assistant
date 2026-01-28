#!/usr/bin/env python
"""
Analyze entry characteristics of big movers (runners).

For each big mover, find:
1. When did the structure form? (ORB, mid-day breakout, etc)
2. What were the parameters at entry? (ADX, volume, momentum, rank_score)
3. What patterns are common across all runners?
4. How do runners differ from rejected/failed setups?

This will help build a discriminator for true runners.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

def load_big_movers_data(json_file):
    """Load the 30-day big movers analysis."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def analyze_runner_at_entry(symbol, target_date, ohlcv_archive_dir):
    """
    For a big mover, analyze what the structure looked like at optimal entry.

    Returns entry characteristics at different times:
    - 9:20 (ORB start)
    - 10:00 (ORB end)
    - 11:00 (mid-morning)
    - Each hour through the day
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

        # Find day open (first bar)
        day_open = day_data.iloc[0]['open']
        day_high = day_data['high'].max()
        day_low = day_data['low'].min()

        # Calculate final move
        day_close = day_data.iloc[-1]['close']
        final_move_pct = ((day_close - day_open) / day_open) * 100

        # Analyze at different time points
        entry_points = []

        # Key time points to check
        check_times = [
            ('09:20', 'ORB_START'),
            ('09:30', 'ORB_10MIN'),
            ('09:45', 'ORB_25MIN'),
            ('10:00', 'ORB_END'),
            ('10:30', 'MID_MORNING'),
            ('11:00', 'LATE_MORNING'),
            ('12:00', 'NOON'),
            ('13:00', 'AFTERNOON'),
            ('14:00', 'LATE_DAY'),
        ]

        for time_str, label in check_times:
            # Get data up to this time - use tz-aware timestamp
            cutoff = pd.Timestamp(f"{target_date} {time_str}", tz=day_data['datetime'].iloc[0].tz)
            data_so_far = day_data[day_data['datetime'] <= cutoff].copy()

            if len(data_so_far) < 5:
                continue

            # Calculate structure parameters at this point
            current_price = data_so_far.iloc[-1]['close']
            high_so_far = data_so_far['high'].max()
            low_so_far = data_so_far['low'].min()

            # Range
            range_pct = ((high_so_far - low_so_far) / day_open) * 100

            # Current move from open
            move_from_open_pct = ((current_price - day_open) / day_open) * 100

            # Volume profile
            avg_volume = data_so_far['volume'].mean() if 'volume' in data_so_far.columns else 0
            recent_volume = data_so_far.tail(5)['volume'].mean() if 'volume' in data_so_far.columns else 0
            volume_acceleration = (recent_volume / avg_volume) if avg_volume > 0 else 1.0

            # Momentum (last 5 bars)
            if len(data_so_far) >= 5:
                price_5bars_ago = data_so_far.iloc[-5]['close']
                momentum_5bar = ((current_price - price_5bars_ago) / price_5bars_ago) * 100
            else:
                momentum_5bar = 0

            # Price consistency (how clean is the move?)
            data_so_far['direction'] = np.sign(data_so_far['close'].diff())
            direction_changes = (data_so_far['direction'].diff() != 0).sum()
            consistency = 100 - (direction_changes / len(data_so_far) * 100) if len(data_so_far) > 0 else 0

            # Distance from high/low
            dist_from_high_pct = ((high_so_far - current_price) / current_price) * 100
            dist_from_low_pct = ((current_price - low_so_far) / current_price) * 100

            # Setup type detection
            setup_type = 'unknown'
            if label in ['ORB_START', 'ORB_10MIN', 'ORB_25MIN', 'ORB_END']:
                if abs(move_from_open_pct) > 0.5:
                    setup_type = 'orb_breakout'
                else:
                    setup_type = 'orb_forming'
            elif range_pct > 2.0 and abs(move_from_open_pct) > 1.0:
                setup_type = 'mid_day_breakout'
            elif consistency > 70:
                setup_type = 'trending_move'
            else:
                setup_type = 'consolidation'

            entry_point = {
                'time': time_str,
                'label': label,
                'current_price': current_price,
                'move_from_open_pct': move_from_open_pct,
                'range_pct': range_pct,
                'momentum_5bar': momentum_5bar,
                'consistency': consistency,
                'volume_acceleration': volume_acceleration,
                'dist_from_high_pct': dist_from_high_pct,
                'dist_from_low_pct': dist_from_low_pct,
                'setup_type': setup_type,
                'bars_traded': len(data_so_far),
                # How much room left to move?
                'remaining_move_pct': final_move_pct - move_from_open_pct
            }

            entry_points.append(entry_point)

        return {
            'symbol': symbol,
            'date': target_date,
            'day_open': day_open,
            'day_high': day_high,
            'day_low': day_low,
            'day_close': day_close,
            'final_move_pct': final_move_pct,
            'final_range_pct': ((day_high - day_low) / day_open) * 100,
            'entry_points': entry_points
        }

    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def find_optimal_entry_time(runner_analysis):
    """
    Find the best entry time for a runner.

    Best = earliest time with clear signal that still captures majority of move.
    """

    if not runner_analysis or not runner_analysis.get('entry_points'):
        return None

    final_move = abs(runner_analysis['final_move_pct'])

    # Find earliest entry that:
    # 1. Has strong momentum (>0.3%)
    # 2. Has volume acceleration (>1.2x)
    # 3. Still has >70% of move remaining

    for ep in runner_analysis['entry_points']:
        move_so_far = abs(ep['move_from_open_pct'])
        remaining_pct = (abs(ep['remaining_move_pct']) / final_move * 100) if final_move > 0 else 0

        if (abs(ep['momentum_5bar']) > 0.3 and
            ep['volume_acceleration'] > 1.2 and
            remaining_pct > 70):
            return ep

    # If no clear early entry, return first entry with momentum
    for ep in runner_analysis['entry_points']:
        if abs(ep['momentum_5bar']) > 0.2:
            return ep

    return runner_analysis['entry_points'][0] if runner_analysis['entry_points'] else None

def aggregate_runner_patterns(all_runner_analyses):
    """
    Aggregate common patterns across all runners.
    """

    patterns = {
        'by_setup_type': defaultdict(int),
        'by_entry_time': defaultdict(int),
        'optimal_entry_characteristics': {
            'momentum_5bar': [],
            'volume_acceleration': [],
            'consistency': [],
            'range_at_entry': [],
            'move_captured': []
        }
    }

    for analysis in all_runner_analyses:
        if not analysis:
            continue

        optimal = find_optimal_entry_time(analysis)
        if not optimal:
            continue

        # Track setup types
        patterns['by_setup_type'][optimal['setup_type']] += 1

        # Track entry times
        patterns['by_entry_time'][optimal['label']] += 1

        # Collect parameters
        patterns['optimal_entry_characteristics']['momentum_5bar'].append(optimal['momentum_5bar'])
        patterns['optimal_entry_characteristics']['volume_acceleration'].append(optimal['volume_acceleration'])
        patterns['optimal_entry_characteristics']['consistency'].append(optimal['consistency'])
        patterns['optimal_entry_characteristics']['range_at_entry'].append(optimal['range_pct'])

        # Calculate what % of move was captured
        final_move = abs(analysis['final_move_pct'])
        remaining = abs(optimal['remaining_move_pct'])
        captured = (remaining / final_move * 100) if final_move > 0 else 0
        patterns['optimal_entry_characteristics']['move_captured'].append(captured)

    return patterns

def print_analysis(patterns):
    """Print comprehensive runner entry analysis."""

    print("="*80)
    print("RUNNER ENTRY CHARACTERISTICS ANALYSIS")
    print("="*80)

    print("\n" + "="*80)
    print("SETUP TYPE DISTRIBUTION")
    print("="*80)

    total_setups = sum(patterns['by_setup_type'].values())
    print(f"\nTotal runners analyzed: {total_setups}\n")

    for setup_type, count in sorted(patterns['by_setup_type'].items(), key=lambda x: -x[1]):
        pct = (count / total_setups * 100) if total_setups > 0 else 0
        print(f"  {setup_type:<25} {count:3d} ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("OPTIMAL ENTRY TIME DISTRIBUTION")
    print("="*80)

    total_entries = sum(patterns['by_entry_time'].values())
    print(f"\nWhen should we enter runners?\n")

    for time_label, count in sorted(patterns['by_entry_time'].items(),
                                   key=lambda x: -x[1]):
        pct = (count / total_entries * 100) if total_entries > 0 else 0
        print(f"  {time_label:<20} {count:3d} ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("ENTRY CHARACTERISTICS - Parameter Ranges")
    print("="*80)

    chars = patterns['optimal_entry_characteristics']

    print(f"\n{'Parameter':<25} {'Median':>10} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("-"*80)

    for param, values in chars.items():
        if values:
            median = np.median(values)
            mean = np.mean(values)
            min_val = min(values)
            max_val = max(values)

            print(f"{param:<25} {median:>10.2f} {mean:>10.2f} {min_val:>10.2f} {max_val:>10.2f}")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Insight 1: Setup types
    if patterns['by_setup_type']:
        dominant_setup = max(patterns['by_setup_type'].items(), key=lambda x: x[1])[0]
        dominant_pct = (patterns['by_setup_type'][dominant_setup] / total_setups * 100)
    else:
        print("\n⚠ WARNING: No setup patterns found")
        print("   This means all stocks failed analysis (timezone or data issues)")
        return

    print(f"\n1. DOMINANT SETUP TYPE:")
    print(f"   -> {dominant_setup} accounts for {dominant_pct:.1f}% of runners")
    print(f"   -> ACTION: Prioritize this setup type in screening")

    # Insight 2: Entry timing
    if patterns['by_entry_time']:
        orb_entries = sum(count for label, count in patterns['by_entry_time'].items()
                         if 'ORB' in label)
        orb_pct = (orb_entries / total_entries * 100) if total_entries > 0 else 0

        print(f"\n2. ENTRY TIMING:")
        print(f"   -> {orb_pct:.1f}% of optimal entries are during ORB period")
        if orb_pct > 60:
            print(f"   -> ACTION: Keep ORB priority high")
        else:
            print(f"   -> ACTION: Don't over-weight ORB, watch for mid-day setups")

    # Insight 3: Parameter thresholds
    if chars['momentum_5bar']:
        median_momentum = np.median(chars['momentum_5bar'])
        print(f"\n3. MOMENTUM THRESHOLD:")
        print(f"   -> Median 5-bar momentum at entry: {median_momentum:.2f}%")
        print(f"   -> ACTION: Require minimum {median_momentum*0.7:.2f}% momentum for entry")

    if chars['volume_acceleration']:
        median_vol_accel = np.median(chars['volume_acceleration'])
        print(f"\n4. VOLUME ACCELERATION:")
        print(f"   -> Median volume acceleration at entry: {median_vol_accel:.2f}x")
        print(f"   -> ACTION: Require minimum {median_vol_accel*0.8:.2f}x volume increase")

    if chars['move_captured']:
        median_captured = np.median(chars['move_captured'])
        print(f"\n5. MOVE CAPTURE POTENTIAL:")
        print(f"   -> Median % of move remaining at entry: {median_captured:.1f}%")
        print(f"   -> If we enter at these optimal points, we capture {median_captured:.1f}% of the move")

    print("\n" + "="*80)
    print("RECOMMENDED FILTERS")
    print("="*80)

    print("\nBased on runner characteristics, add these filters:\n")

    if chars['momentum_5bar']:
        threshold = np.percentile(chars['momentum_5bar'], 30)
        print(f"1. Momentum Filter:")
        print(f"   - Require 5-bar momentum > {threshold:.2f}%")

    if chars['volume_acceleration']:
        threshold = np.percentile(chars['volume_acceleration'], 30)
        print(f"\n2. Volume Acceleration Filter:")
        print(f"   - Require volume > {threshold:.2f}x recent average")

    if chars['consistency']:
        threshold = np.percentile(chars['consistency'], 30)
        print(f"\n3. Price Action Quality Filter:")
        print(f"   - Require consistency score > {threshold:.1f}")
        print(f"   - (Rejects choppy, back-and-forth moves)")

    if chars['range_at_entry']:
        median_range = np.median(chars['range_at_entry'])
        print(f"\n4. Range Filter:")
        print(f"   - Typical range at entry: {median_range:.2f}%")
        print(f"   - Don't enter if already extended > {median_range*1.5:.2f}%")

    print("\n" + "="*80)

def main():
    """
    Main execution.

    Process:
    1. Load big movers from 30-day analysis
    2. For each big mover, analyze entry characteristics at different times
    3. Find optimal entry point for each
    4. Aggregate patterns across all runners
    5. Output recommended filters
    """

    big_movers_file = "big_movers_30day_analysis.json"
    ohlcv_archive = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\cache\ohlcv_archive"

    print(f"Loading big movers data from {big_movers_file}...")
    data = load_big_movers_data(big_movers_file)

    # Get all big movers from all days
    all_movers = []
    for day_result in data['daily_results']:
        for mover in day_result['top_10_movers']:  # Top 10 per day for speed
            all_movers.append({
                'symbol': f"NSE:{mover['symbol'].replace('.NS', '')}",
                'date': day_result['date'],
                'move': mover['abs_move']
            })

    print(f"\nAnalyzing {len(all_movers)} big movers...")
    print("This will take a few minutes...\n")

    # Analyze each mover
    runner_analyses = []
    for i, mover in enumerate(all_movers, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(all_movers)}...")

        analysis = analyze_runner_at_entry(
            mover['symbol'],
            mover['date'],
            ohlcv_archive
        )

        if analysis:
            runner_analyses.append(analysis)

    print(f"\nSuccessfully analyzed {len(runner_analyses)} runners")

    # Aggregate patterns
    print("\nAggregating patterns...")
    patterns = aggregate_runner_patterns(runner_analyses)

    # Print analysis
    print_analysis(patterns)

    # Save results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'total_runners_analyzed': len(runner_analyses),
        'patterns': {
            'by_setup_type': dict(patterns['by_setup_type']),
            'by_entry_time': dict(patterns['by_entry_time']),
            'parameter_stats': {
                param: {
                    'median': float(np.median(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'p25': float(np.percentile(values, 25)),
                    'p75': float(np.percentile(values, 75))
                }
                for param, values in patterns['optimal_entry_characteristics'].items()
                if values
            }
        },
        'runner_analyses': runner_analyses
    }

    output_file = Path("runner_entry_characteristics.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)

    print("\n1. Implement the recommended filters in your screening logic")
    print("2. Backtest with these new filters to measure improvement")
    print("3. Compare: what % of big movers are now screened vs 0.3% currently?")
    print("4. Iterate: adjust thresholds based on backtest results")
    print("5. Build ML discriminator using these features")

if __name__ == '__main__':
    main()
