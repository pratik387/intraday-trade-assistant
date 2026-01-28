#!/usr/bin/env python3
"""
Exit Split Strategy Analyzer

Compares different exit split strategies across all trades:
- 40-40-20 (current)
- 30-70, 70-30
- 60-40, 50-50
- 60-20-20, 50-30-20
- 33-33-33
- 80-20, 20-80

Usage:
    python tools/analyze_exit_splits.py <backtest_dir>
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_analytics(backtest_dir):
    """Load analytics.jsonl from all sessions."""
    trades = []
    sessions_dir = Path(backtest_dir)

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        analytics_file = session_dir / "analytics.jsonl"
        if not analytics_file.exists():
            continue

        with open(analytics_file, 'r') as f:
            for line in f:
                if line.strip():
                    trade = json.loads(line)
                    trades.append(trade)

    return trades


def calculate_split_pnl(trade, split_config):
    """
    Calculate PnL for a given exit split configuration.

    split_config: list of (pct, target) tuples
    Example: [(40, 't1'), (40, 't2'), (20, 't3')]
    """
    entry = trade.get('entry_price', 0)
    qty_total = trade.get('qty', 0)
    side = trade.get('side', '').upper()

    if entry == 0 or qty_total == 0:
        return 0.0

    # Get exit prices
    exits = trade.get('exits', [])
    exit_prices = {}

    for exit_event in exits:
        reason = exit_event.get('reason', '')
        price = exit_event.get('price', 0)

        if 't1' in reason.lower():
            exit_prices['t1'] = price
        elif 't2' in reason.lower():
            exit_prices['t2'] = price
        elif 'sl' in reason.lower() or 'hard' in reason.lower():
            exit_prices['sl'] = price

    # Calculate T3 (3.0R target)
    # Need to estimate R from T1 or T2 exits
    r_per_share = 0
    if 't1' in exit_prices:
        # Assume T1 is ~1.0R
        price_move_t1 = abs(exit_prices['t1'] - entry)
        r_per_share = price_move_t1
    elif 't2' in exit_prices:
        # Assume T2 is ~2.0R
        price_move_t2 = abs(exit_prices['t2'] - entry)
        r_per_share = price_move_t2 / 2.0

    if r_per_share > 0:
        if side == 'BUY':
            exit_prices['t3'] = entry + (r_per_share * 3.0)
        else:
            exit_prices['t3'] = entry - (r_per_share * 3.0)

    # Calculate PnL for split
    total_pnl = 0.0

    for pct, target in split_config:
        qty = int(qty_total * (pct / 100.0))

        # Default to SL if target not hit
        exit_price = exit_prices.get(target, exit_prices.get('sl', entry))

        # Calculate P&L
        if side == 'BUY':
            pnl = qty * (exit_price - entry)
        else:
            pnl = qty * (entry - exit_price)

        total_pnl += pnl

    return total_pnl


def analyze_all_splits(trades):
    """Analyze all exit split strategies."""

    # Define split strategies
    strategies = {
        '40-40-20': [(40, 't1'), (40, 't2'), (20, 't3')],
        '30-70': [(30, 't1'), (70, 't2')],
        '70-30': [(70, 't1'), (30, 't2')],
        '60-40': [(60, 't1'), (40, 't2')],
        '50-50': [(50, 't1'), (50, 't2')],
        '60-20-20': [(60, 't1'), (20, 't2'), (20, 't3')],
        '50-30-20': [(50, 't1'), (30, 't2'), (20, 't3')],
        '33-33-33': [(33, 't1'), (33, 't2'), (34, 't3')],
        '80-20': [(80, 't1'), (20, 't2')],
        '20-80': [(20, 't1'), (80, 't2')],
    }

    results = defaultdict(float)
    actual_pnl = 0.0
    trade_count = 0

    for trade in trades:
        # Get actual PnL
        actual = trade.get('pnl', 0)
        actual_pnl += actual
        trade_count += 1

        # Calculate PnL for each strategy
        for strategy_name, split_config in strategies.items():
            pnl = calculate_split_pnl(trade, split_config)
            results[strategy_name] += pnl

    # Print results
    print(f"\n{'='*80}")
    print(f"Exit Split Strategy Analysis - {trade_count} Trades")
    print(f"{'='*80}\n")

    print(f"Actual PnL (40-40-20 with SL): Rs. {actual_pnl:,.2f}")
    print()

    # Sort by total PnL
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Strategy':<15} {'Total PnL':>15} {'vs Actual':>12} {'Improvement':>12}")
    print(f"{'-'*15} {'-'*15} {'-'*12} {'-'*12}")

    for strategy, pnl in sorted_results:
        vs_actual = pnl - actual_pnl
        pct_change = (vs_actual / actual_pnl * 100) if actual_pnl != 0 else 0

        print(f"{strategy:<15} Rs. {pnl:>11,.2f} Rs. {vs_actual:>9,.2f} {pct_change:>10.1f}%")

    print(f"\n{'='*80}")

    return results, actual_pnl


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_exit_splits.py <backtest_dir>")
        print("\nExample:")
        print("  python tools/analyze_exit_splits.py backtest_20251110-062142_extracted/20251110-062142_full/20251110-062142")
        sys.exit(1)

    backtest_dir = sys.argv[1]

    print(f"Loading trades from: {backtest_dir}")
    trades = load_analytics(backtest_dir)

    if not trades:
        print(f"ERROR: No analytics data found in {backtest_dir}")
        sys.exit(1)

    print(f"Found {len(trades)} trades\n")

    results, actual_pnl = analyze_all_splits(trades)

    # Save results to JSON
    output_file = Path(backtest_dir).parent / "exit_split_analysis.json"
    output_data = {
        'total_trades': len(trades),
        'actual_pnl': actual_pnl,
        'strategies': {k: float(v) for k, v in results.items()}
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
