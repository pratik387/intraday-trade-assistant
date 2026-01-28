#!/usr/bin/env python
"""
Analyze big mover rejection patterns across ALL dates in backtest.

Uses the 30day analysis file to get big mover lists, then checks
screening.jsonl in the backtest for actual rejection reasons.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def load_big_movers_from_file(file_path: Path) -> List[str]:
    """Load big movers from individual JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    if 'top_movers' in data:
        # Convert to NSE symbols
        nse_symbols = [
            f"NSE:{m['symbol'].replace('.NS', '')}"
            for m in data['top_movers']
        ]
        return nse_symbols
    return []

def extract_rejections(screening_file: Path, symbols: List[str]) -> Dict[str, List[Dict]]:
    """Extract all rejections for specified symbols from screening.jsonl."""
    rejections = defaultdict(list)

    if not screening_file.exists():
        return rejections

    symbol_set = set(symbols)

    with open(screening_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    symbol = entry.get('symbol')

                    if symbol in symbol_set and entry.get('action') == 'reject':
                        rejections[symbol].append({
                            'timestamp': entry.get('timestamp'),
                            'reason': entry.get('reason'),
                            'all_reasons': entry.get('all_reasons', []),
                            'setup_type': entry.get('setup_type', 'unknown'),
                            'price': entry.get('current_price')
                        })
                except:
                    pass

    return rejections

def categorize_rejection_reason(reason: str) -> str:
    """Categorize rejection reason into high-level categories."""
    if 'momentum_consolidation_fail' in reason:
        return 'Stage-0: Momentum/Consolidation Filter'
    elif 'range_compression' in reason:
        return 'Stage-0: Range Compression Filter'
    elif 'no_structure_event' in reason:
        return 'Stage-1: No Structure Detected'
    elif 'blocked_by_daily_regime' in reason:
        return 'Stage-1: Regime Block'
    elif 'blocked_by_mcap' in reason:
        return 'Stage-1: Market Cap Filter'
    elif 'timing' in reason or 'session' in reason:
        return 'Stage-1: Timing Filter'
    elif 'volume' in reason:
        return 'Stage-1: Volume Filter'
    elif 'adx' in reason:
        return 'Stage-1: ADX Filter'
    else:
        return f'Other: {reason[:50]}'

def analyze_date(date_str: str, backtest_dir: Path, big_mover_symbols: List[str]) -> Dict:
    """Analyze big movers for a specific date."""

    if not big_mover_symbols:
        return {'error': 'No big movers found', 'date': date_str}

    # Load rejections from screening.jsonl
    screening_file = backtest_dir / date_str / 'screening.jsonl'

    if not screening_file.exists():
        return {'error': 'screening.jsonl not found', 'date': date_str}

    rejections = extract_rejections(screening_file, big_mover_symbols)

    # Categorize rejections
    rejection_categories = defaultdict(int)

    for symbol, symbol_rejections in rejections.items():
        for rej in symbol_rejections:
            category = categorize_rejection_reason(rej['reason'])
            rejection_categories[category] += 1

    # Find movers that never reached screening
    screened_symbols = set(rejections.keys())
    never_screened = [s for s in big_mover_symbols if s not in screened_symbols]

    return {
        'date': date_str,
        'total_big_movers': len(big_mover_symbols),
        'screened': len(screened_symbols),
        'never_screened': len(never_screened),
        'rejection_categories': dict(rejection_categories)
    }

def main():
    """Main analysis function."""

    backtest_base = Path('backtest_20251112-023811_extracted/20251112-023811_full/20251112-023811')

    # Get all available dates in backtest
    available_dates = sorted([
        d.name for d in backtest_base.iterdir()
        if d.is_dir() and d.name.startswith('202')
    ])
    print(f"Found {len(available_dates)} dates in backtest")

    # Find all big mover JSON files
    big_mover_files = list(Path('.').glob('big_movers_*.json'))
    print(f"Found {len(big_mover_files)} big mover data files")

    # Create mapping of date to big mover file
    big_movers_by_date = {}
    for file in big_mover_files:
        # Extract date from filename (big_movers_YYYY-MM-DD.json)
        date_str = file.stem.replace('big_movers_', '')
        if date_str in available_dates:
            big_movers = load_big_movers_from_file(file)
            if big_movers:
                big_movers_by_date[date_str] = big_movers

    print(f"Matched {len(big_movers_by_date)} dates with both backtest and big mover data")

    # Analyze all dates that have both big mover data and backtest logs
    all_results = []
    dates_to_analyze = sorted(big_movers_by_date.keys())

    print(f"\nAnalyzing {len(dates_to_analyze)} dates...")

    for i, date in enumerate(dates_to_analyze, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(dates_to_analyze)} dates...")

        result = analyze_date(
            date,
            backtest_base,
            big_movers_by_date[date]
        )
        all_results.append(result)

    # Calculate aggregate statistics
    print("\n" + "="*80)
    print("BIG MOVER REJECTION ANALYSIS - ALL DATES")
    print("="*80)

    valid_results = [r for r in all_results if 'error' not in r]

    total_movers = sum(r['total_big_movers'] for r in valid_results)
    total_screened = sum(r['screened'] for r in valid_results)
    total_never_screened = sum(r['never_screened'] for r in valid_results)

    print(f"\nDates analyzed: {len(valid_results)}")
    print(f"Total big movers: {total_movers}")
    print(f"Reached screening: {total_screened} ({total_screened/total_movers*100:.1f}%)")
    print(f"Never screened: {total_never_screened} ({total_never_screened/total_movers*100:.1f}%)")

    # Aggregate rejection categories
    all_categories = defaultdict(int)
    for result in valid_results:
        for category, count in result.get('rejection_categories', {}).items():
            all_categories[category] += count

    total_rejections = sum(all_categories.values())

    print(f"\nTotal rejections recorded: {total_rejections}")
    print(f"\nTop Rejection Categories:")
    for category, count in sorted(all_categories.items(), key=lambda x: -x[1])[:15]:
        pct = count / total_rejections * 100 if total_rejections > 0 else 0
        print(f"  {category:<50} {count:5d} ({pct:5.1f}%)")

    # Show per-date breakdown sample
    print(f"\n" + "="*80)
    print("SAMPLE PER-DATE BREAKDOWN (First 10 dates)")
    print("="*80)
    print(f"{'Date':<12} {'Total':>6} {'Screened':>9} {'Never':>9} {'Screen %':>10}")
    print("-"*80)

    for result in valid_results[:10]:
        if 'error' not in result:
            screen_pct = result['screened'] / result['total_big_movers'] * 100
            print(f"{result['date']:<12} {result['total_big_movers']:>6} "
                  f"{result['screened']:>9} {result['never_screened']:>9} "
                  f"{screen_pct:>9.1f}%")

    # Save detailed results
    output_file = Path('big_mover_rejection_analysis_all_dates.json')
    with open(output_file, 'w') as f:
        json.dump({
            'total_dates': len(valid_results),
            'total_big_movers': total_movers,
            'total_screened': total_screened,
            'total_never_screened': total_never_screened,
            'aggregate_rejection_categories': dict(all_categories),
            'daily_results': all_results
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
