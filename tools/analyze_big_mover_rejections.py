#!/usr/bin/env python
"""
Analyze big mover rejection patterns across multiple dates.

Consolidates rejection reasons from screening.jsonl to understand
why stocks with significant intraday moves are not being traded.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def load_big_movers(file_path: Path) -> List[Dict]:
    """Load big movers from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'top_movers' in data:
        return data['top_movers']
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

def analyze_date(date_str: str, backtest_dir: Path, big_movers_file: Path) -> Dict:
    """Analyze big movers for a specific date."""

    # Load big movers
    big_movers = load_big_movers(big_movers_file)
    if not big_movers:
        return {'error': 'No big movers found'}

    # Convert to NSE symbols
    nse_symbols = [f"NSE:{m['symbol'].replace('.NS', '')}" for m in big_movers]

    # Load rejections
    screening_file = backtest_dir / date_str / 'screening.jsonl'
    rejections = extract_rejections(screening_file, nse_symbols)

    # Categorize rejections
    rejection_categories = defaultdict(int)
    detailed_rejections = []

    for symbol, symbol_rejections in rejections.items():
        for rej in symbol_rejections:
            category = categorize_rejection_reason(rej['reason'])
            rejection_categories[category] += 1

            detailed_rejections.append({
                'symbol': symbol,
                'timestamp': rej['timestamp'],
                'category': category,
                'reason': rej['reason'],
                'setup_type': rej['setup_type'],
                'price': rej['price']
            })

    # Find movers that never reached screening
    screened_symbols = set(rejections.keys())
    never_screened = [s for s in nse_symbols if s not in screened_symbols]

    return {
        'date': date_str,
        'total_big_movers': len(big_movers),
        'screened': len(screened_symbols),
        'never_screened': len(never_screened),
        'never_screened_symbols': never_screened[:10],  # Sample
        'rejection_categories': dict(rejection_categories),
        'detailed_rejections': detailed_rejections[:50]  # Sample
    }

def main():
    """Main analysis function."""

    # Define analysis targets - using correct extracted backtest directory
    backtest_base = Path('backtest_20251112-023811_extracted/20251112-023811_full/20251112-023811')

    analysis_configs = [
        {
            'date': '2024-01-02',
            'big_movers_file': Path('big_movers_2024-01-02.json'),
            'backtest_dir': backtest_base
        },
        {
            'date': '2023-12-22',
            'big_movers_file': Path('big_movers_2023-12-22.json'),
            'backtest_dir': backtest_base
        }
    ]

    all_results = []

    for config in analysis_configs:
        if config['big_movers_file'].exists():
            print(f"\nAnalyzing {config['date']}...")
            result = analyze_date(
                config['date'],
                config['backtest_dir'],
                config['big_movers_file']
            )
            all_results.append(result)
        else:
            print(f"Skipping {config['date']}: file not found")

    # Print consolidated results
    print("\n" + "="*80)
    print("BIG MOVER REJECTION ANALYSIS - CONSOLIDATED")
    print("="*80)

    for result in all_results:
        if 'error' in result:
            print(f"\n{result.get('date', 'Unknown')}: {result['error']}")
            continue

        print(f"\nDate: {result['date']}")
        print(f"  Total big movers: {result['total_big_movers']}")
        print(f"  Reached screening: {result['screened']} ({result['screened']/result['total_big_movers']*100:.1f}%)")
        print(f"  Never screened: {result['never_screened']} ({result['never_screened']/result['total_big_movers']*100:.1f}%)")

        if result['rejection_categories']:
            print(f"\n  Rejection Categories:")
            for category, count in sorted(result['rejection_categories'].items(), key=lambda x: -x[1]):
                print(f"    {category:<50} {count:3d}")

        if result['never_screened_symbols']:
            print(f"\n  Sample never-screened stocks: {', '.join(result['never_screened_symbols'][:5])}")

    # Aggregate statistics across all dates
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)

    total_movers = sum(r.get('total_big_movers', 0) for r in all_results if 'error' not in r)
    total_never_screened = sum(r.get('never_screened', 0) for r in all_results if 'error' not in r)

    all_categories = defaultdict(int)
    for result in all_results:
        if 'error' not in result:
            for category, count in result.get('rejection_categories', {}).items():
                all_categories[category] += count

    print(f"\nTotal big movers analyzed: {total_movers}")
    print(f"Never reached screening: {total_never_screened} ({total_never_screened/total_movers*100:.1f}%)")

    print(f"\nTop Rejection Categories (across all dates):")
    for category, count in sorted(all_categories.items(), key=lambda x: -x[1])[:10]:
        print(f"  {category:<50} {count:3d}")

    # Save detailed results
    output_file = Path('big_mover_rejection_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
