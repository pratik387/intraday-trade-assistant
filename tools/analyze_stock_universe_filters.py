#!/usr/bin/env python
"""
Analyze why 99.6% of big movers are never screened.
Compare big movers against nse_all.json universe to identify filter impact.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_nse_universe(nse_file):
    """Load stock universe from nse_all.json"""
    with open(nse_file, 'r') as f:
        stocks = json.load(f)

    # Create lookup by symbol
    universe = {}
    for stock in stocks:
        symbol = stock['symbol']
        universe[symbol] = stock

    return universe

def analyze_big_movers_against_universe(big_movers_file, nse_file):
    """
    Compare big movers with nse_all.json to find why they're filtered out.
    """

    # Load data
    print("Loading data...")
    with open(big_movers_file, 'r') as f:
        big_movers_data = json.load(f)

    universe = load_nse_universe(nse_file)

    print(f"Universe size: {len(universe)} stocks")
    print(f"Analyzing {len(big_movers_data['daily_results'])} days of big movers\n")

    # Statistics
    stats = {
        'total_movers': 0,
        'in_universe': 0,
        'not_in_universe': 0,
        'screened': 0,
        'never_screened': 0,
        'cap_segments': defaultdict(int),
        'mis_enabled': {'yes': 0, 'no': 0},
        'not_in_universe_examples': []
    }

    # Analyze each day
    for day_result in big_movers_data['daily_results']:
        date = day_result['date']
        top_movers = day_result.get('top_10_movers', [])

        stats['total_movers'] += len(top_movers)
        stats['screened'] += day_result.get('screened', 0)
        stats['never_screened'] += day_result.get('never_screened', 0)

        for mover in top_movers:
            symbol_raw = mover['symbol']  # AAVAS.NS format
            symbol_nse = f"NSE:{symbol_raw.replace('.NS', '')}"  # Convert to NSE:AAVAS

            # Check if in universe
            if symbol_raw in universe:
                stats['in_universe'] += 1

                stock_info = universe[symbol_raw]

                # Track cap segment
                cap_segment = stock_info.get('cap_segment', 'unknown')
                stats['cap_segments'][cap_segment] += 1

                # Track MIS enabled
                mis_enabled = stock_info.get('mis_enabled', False)
                stats['mis_enabled']['yes' if mis_enabled else 'no'] += 1

            else:
                stats['not_in_universe'] += 1

                # Save examples
                if len(stats['not_in_universe_examples']) < 20:
                    stats['not_in_universe_examples'].append({
                        'symbol': symbol_raw,
                        'date': date,
                        'move': mover.get('abs_move', 0),
                        'consistency': mover.get('consistency_score', 0)
                    })

    return stats

def print_analysis(stats):
    """Print detailed analysis of filter impact."""

    print("="*80)
    print("STOCK UNIVERSE FILTER IMPACT ANALYSIS")
    print("="*80)
    print()

    total = stats['total_movers']

    print(f"UNIVERSE COVERAGE:")
    print(f"  Total big movers analyzed:     {total:4d}")
    print(f"  In nse_all.json universe:      {stats['in_universe']:4d} ({stats['in_universe']/total*100:5.1f}%)")
    print(f"  NOT in universe:               {stats['not_in_universe']:4d} ({stats['not_in_universe']/total*100:5.1f}%)")
    print()

    if stats['not_in_universe'] > 0:
        print(f"⚠ {stats['not_in_universe']/total*100:.1f}% of big movers are NOT in the stock universe!")
        print(f"  This explains why they never reach screening logic.")
        print()

    print(f"SCREENING RATE:")
    print(f"  Screened:      {stats['screened']:4d} ({stats['screened']/total*100:5.1f}%)")
    print(f"  Never screened:{stats['never_screened']:4d} ({stats['never_screened']/total*100:5.1f}%)")
    print()

    # Cap segment breakdown (for movers IN universe)
    in_universe = stats['in_universe']
    if in_universe > 0:
        print("="*80)
        print(f"CAP SEGMENT BREAKDOWN (for {in_universe} movers IN universe)")
        print("="*80)
        for segment, count in sorted(stats['cap_segments'].items(), key=lambda x: -x[1]):
            print(f"  {segment:<15} {count:4d} ({count/in_universe*100:5.1f}%)")
        print()

        print("MIS ENABLED:")
        for mis_status, count in stats['mis_enabled'].items():
            print(f"  MIS {mis_status:<3} {count:4d} ({count/in_universe*100:5.1f}%)")
        print()

    # Examples of stocks NOT in universe
    if stats['not_in_universe_examples']:
        print("="*80)
        print(f"EXAMPLES OF BIG MOVERS NOT IN UNIVERSE (showing {len(stats['not_in_universe_examples'])})")
        print("="*80)
        print(f"{'Symbol':<15} {'Date':<12} {'Move %':>8} {'Consistency':>12}")
        print("-"*80)
        for ex in stats['not_in_universe_examples']:
            print(f"{ex['symbol']:<15} {ex['date']:<12} {ex['move']:>7.2f}% {ex['consistency']:>11.1f}")
        print()

    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()

    universe_miss_pct = stats['not_in_universe'] / total * 100

    if universe_miss_pct > 50:
        print(f"1. CRITICAL: {universe_miss_pct:.1f}% of big movers are NOT in nse_all.json")
        print(f"   -> The stock universe itself is the primary bottleneck")
        print(f"   -> These stocks never reach any decision logic")
        print()

    if stats['in_universe'] > 0:
        screened_pct_of_universe = stats['screened'] / stats['in_universe'] * 100
        print(f"2. SECONDARY FILTER: Of stocks IN universe, only {screened_pct_of_universe:.1f}% get screened")
        print(f"   -> Additional filters after universe lookup are also blocking movers")
        print()

    print("3. NEXT ACTIONS:")
    print("   a) Expand nse_all.json to include more stocks")
    print("   b) Investigate what criteria are used to build nse_all.json")
    print("   c) Find and analyze the screening/filtering code that runs BEFORE events.jsonl")
    print("   d) Check if mid_cap/small_cap stocks are being filtered more than large_cap")

def main():
    """Main analysis."""

    # Use the combined analysis from all sessions
    big_movers_file = Path("runner_analysis_130_sessions.json")

    if not big_movers_file.exists():
        print(f"Using 30-day analysis instead...")
        big_movers_file = Path("big_movers_30day_analysis.json")

    nse_file = Path("nse_all.json")

    if not big_movers_file.exists():
        print(f"Error: {big_movers_file} not found")
        return

    if not nse_file.exists():
        print(f"Error: {nse_file} not found")
        return

    print(f"Analyzing: {big_movers_file}")
    print()

    stats = analyze_big_movers_against_universe(big_movers_file, nse_file)
    print_analysis(stats)

    # Save results
    output_file = Path("stock_universe_filter_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
