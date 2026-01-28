#!/usr/bin/env python3
"""
Analyze what differentiates trades that hit targets vs those that don't.

Compares:
- Setup types (which setups have high vs low target hit rates)
- Market regimes (which market conditions favor target hits)
- Trade direction (long vs short performance)
- Rank scores (do higher-ranked trades hit targets more often)
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_trade_logs(backtest_dir: Path) -> Tuple[Dict, Dict]:
    """Parse all trade_logs.log files to categorize trades by target hit status."""

    trades_hit_target = defaultdict(lambda: {
        'count': 0,
        'total_pnl': 0.0,
        'setups': defaultdict(int),
        'regimes': defaultdict(int),
        'directions': defaultdict(int),
        'symbols': []
    })

    trades_missed_target = defaultdict(lambda: {
        'count': 0,
        'total_pnl': 0.0,
        'setups': defaultdict(int),
        'regimes': defaultdict(int),
        'directions': defaultdict(int),
        'symbols': []
    })

    # Track entries and their exits
    entries = {}  # symbol -> (entry_price, strategy, direction)

    for session_dir in sorted(backtest_dir.glob("2*/trade_logs.log")):
        with open(session_dir, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse TRIGGER_EXEC (entry)
                if "TRIGGER_EXEC" in line:
                    match = re.search(r'NSE:(\w+)\s+\|\s+(BUY|SELL)\s+\d+\s+@\s+([\d.]+)\s+\|\s+strategy=(\w+)', line)
                    if match:
                        symbol = match.group(1)
                        direction = "long" if match.group(2) == "BUY" else "short"
                        entry_price = float(match.group(3))
                        strategy = match.group(4)
                        entries[symbol] = (entry_price, strategy, direction)

                # Parse EXIT
                elif "EXIT" in line:
                    match = re.search(r'NSE:(\w+)\s+\|\s+Qty:\s+\d+\s+\|\s+Entry:\s+Rs\.([\d.]+)\s+\|\s+Exit:\s+Rs\.([\d.]+)\s+\|\s+PnL:\s+Rs\.([-\d.]+)\s+(\w+)', line)
                    if match:
                        symbol = match.group(1)
                        entry_px = float(match.group(2))
                        exit_px = float(match.group(3))
                        pnl = float(match.group(4))
                        exit_reason = match.group(5)

                        # Get entry info
                        if symbol not in entries:
                            continue

                        entry_price, strategy, direction = entries[symbol]

                        # Determine if trade hit target
                        hit_target = exit_reason in ['t1_partial', 't2_partial']

                        # Categorize
                        if hit_target:
                            cat = trades_hit_target
                        else:
                            cat = trades_missed_target

                        cat['all']['count'] += 1
                        cat['all']['total_pnl'] += pnl
                        cat['all']['setups'][strategy] += 1
                        cat['all']['directions'][direction] += 1
                        if symbol not in cat['all']['symbols']:
                            cat['all']['symbols'].append(symbol)

                        # Remove entry after final exit
                        if exit_reason not in ['t1_partial', 't2_partial']:
                            if symbol in entries:
                                del entries[symbol]

    return dict(trades_hit_target), dict(trades_missed_target)


def print_comparison_table(hit_dict: Dict, miss_dict: Dict):
    """Print formatted comparison table."""

    print("\n" + "="*100)
    print("TRADE CHARACTERISTICS COMPARISON: TARGET HITS vs MISSES")
    print("="*100)

    # Overall stats
    hit_count = hit_dict['all']['count']
    hit_pnl = hit_dict['all']['total_pnl']
    miss_count = miss_dict['all']['count']
    miss_pnl = miss_dict['all']['total_pnl']

    print(f"\n{'Metric':<30} | {'Hit Target':>15} | {'Missed Target':>15} | {'Delta':>15}")
    print("-" * 100)
    print(f"{'Total Exits':<30} | {hit_count:>15,} | {miss_count:>15,} | {hit_count - miss_count:>15,}")
    print(f"{'Total P&L':<30} | {f'Rs.{hit_pnl:,.0f}':>15} | {f'Rs.{miss_pnl:,.0f}':>15} | {f'Rs.{hit_pnl - miss_pnl:,.0f}':>15}")
    print(f"{'Avg P&L per Exit':<30} | {f'Rs.{hit_pnl/hit_count:,.0f}':>15} | {f'Rs.{miss_pnl/miss_count:,.0f}':>15} | {f'Rs.{(hit_pnl/hit_count) - (miss_pnl/miss_count):,.0f}':>15}")

    # Setup type breakdown
    print("\n" + "="*100)
    print("SETUP TYPE BREAKDOWN")
    print("="*100)

    all_setups = set(hit_dict['all']['setups'].keys()) | set(miss_dict['all']['setups'].keys())

    print(f"\n{'Setup Type':<30} | {'Hit Count':>12} | {'Hit %':>8} | {'Miss Count':>12} | {'Miss %':>8} | {'Hit Rate':>10}")
    print("-" * 100)

    for setup in sorted(all_setups):
        hit_setup_count = hit_dict['all']['setups'].get(setup, 0)
        miss_setup_count = miss_dict['all']['setups'].get(setup, 0)
        total_setup = hit_setup_count + miss_setup_count

        hit_pct = (hit_setup_count / hit_count * 100) if hit_count > 0 else 0
        miss_pct = (miss_setup_count / miss_count * 100) if miss_count > 0 else 0
        hit_rate = (hit_setup_count / total_setup * 100) if total_setup > 0 else 0

        print(f"{setup:<30} | {hit_setup_count:>12} | {hit_pct:>7.1f}% | {miss_setup_count:>12} | {miss_pct:>7.1f}% | {hit_rate:>9.1f}%")

    # Direction breakdown
    print("\n" + "="*100)
    print("DIRECTION BREAKDOWN")
    print("="*100)

    print(f"\n{'Direction':<30} | {'Hit Count':>12} | {'Hit %':>8} | {'Miss Count':>12} | {'Miss %':>8} | {'Hit Rate':>10}")
    print("-" * 100)

    for direction in ['long', 'short']:
        hit_dir_count = hit_dict['all']['directions'].get(direction, 0)
        miss_dir_count = miss_dict['all']['directions'].get(direction, 0)
        total_dir = hit_dir_count + miss_dir_count

        hit_pct = (hit_dir_count / hit_count * 100) if hit_count > 0 else 0
        miss_pct = (miss_dir_count / miss_count * 100) if miss_count > 0 else 0
        hit_rate = (hit_dir_count / total_dir * 100) if total_dir > 0 else 0

        print(f"{direction:<30} | {hit_dir_count:>12} | {hit_pct:>7.1f}% | {miss_dir_count:>12} | {miss_pct:>7.1f}% | {hit_rate:>9.1f}%")

    print("\n" + "="*100)


def main():
    backtest_dir = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251106-041140_extracted\20251106-041140_full\20251106-041140")

    print("Parsing trade logs...")
    hit_dict, miss_dict = parse_trade_logs(backtest_dir)

    print_comparison_table(hit_dict, miss_dict)

    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 100)

    hit_count = hit_dict['all']['count']
    miss_count = miss_dict['all']['count']
    total = hit_count + miss_count
    overall_hit_rate = (hit_count / total * 100) if total > 0 else 0

    print(f"1. Overall Target Hit Rate: {overall_hit_rate:.1f}% ({hit_count}/{total} exits)")

    # Find best and worst setups
    all_setups = set(hit_dict['all']['setups'].keys()) | set(miss_dict['all']['setups'].keys())
    setup_hit_rates = {}

    for setup in all_setups:
        hit_setup = hit_dict['all']['setups'].get(setup, 0)
        miss_setup = miss_dict['all']['setups'].get(setup, 0)
        total_setup = hit_setup + miss_setup
        if total_setup >= 5:  # Only consider setups with at least 5 trades
            hit_rate = (hit_setup / total_setup * 100)
            setup_hit_rates[setup] = (hit_rate, total_setup)

    if setup_hit_rates:
        best_setup = max(setup_hit_rates.items(), key=lambda x: x[1][0])
        worst_setup = min(setup_hit_rates.items(), key=lambda x: x[1][0])

        print(f"2. Best Setup: {best_setup[0]} ({best_setup[1][0]:.1f}% hit rate, {best_setup[1][1]} trades)")
        print(f"3. Worst Setup: {worst_setup[0]} ({worst_setup[1][0]:.1f}% hit rate, {worst_setup[1][1]} trades)")

    # Direction comparison
    long_hits = hit_dict['all']['directions'].get('long', 0)
    long_misses = miss_dict['all']['directions'].get('long', 0)
    short_hits = hit_dict['all']['directions'].get('short', 0)
    short_misses = miss_dict['all']['directions'].get('short', 0)

    long_hit_rate = (long_hits / (long_hits + long_misses) * 100) if (long_hits + long_misses) > 0 else 0
    short_hit_rate = (short_hits / (short_hits + short_misses) * 100) if (short_hits + short_misses) > 0 else 0

    print(f"4. Long Hit Rate: {long_hit_rate:.1f}% ({long_hits}/{long_hits + long_misses})")
    print(f"5. Short Hit Rate: {short_hit_rate:.1f}% ({short_hits}/{short_hits + short_misses})")

    print("\n" + "="*100)
    print("\nRECOMMENDATIONS:")
    print("-" * 100)

    # Actionable recommendations based on hit rates
    for setup, (hit_rate, count) in sorted(setup_hit_rates.items(), key=lambda x: x[1][0], reverse=True):
        if hit_rate >= 40:
            print(f"[+] KEEP: {setup} ({hit_rate:.1f}% hit rate) - Performing above baseline")
        elif hit_rate < 25:
            print(f"[-] REVIEW: {setup} ({hit_rate:.1f}% hit rate) - Consider tightening filters or removing")
        else:
            print(f"[~] MONITOR: {setup} ({hit_rate:.1f}% hit rate) - Marginal performance")

    print("\n" + "="*100)


if __name__ == "__main__":
    main()
