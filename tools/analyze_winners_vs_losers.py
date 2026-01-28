#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze winning vs losing trades to identify stock selection quality issues.

Compares characteristics of:
- 13 trades that hit targets (winners)
- 85 trades that didn't hit targets or hit SL (losers)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Path to extracted backtest
BACKTEST_DIR = Path(r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251112-104346_extracted\20251112-104346_full\20251112-104346")

def load_all_analytics():
    """Load all analytics.jsonl files from all sessions."""
    all_trades = []

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        analytics_file = session_dir / "analytics.jsonl"
        if not analytics_file.exists():
            continue

        with open(analytics_file, 'r') as f:
            for line in f:
                if line.strip():
                    trade = json.loads(line)
                    if trade.get('stage') == 'EXIT':
                        all_trades.append(trade)

    return all_trades

def categorize_trades(trades):
    """Categorize trades into winners and losers based on exit reason."""
    winners = []
    losers = []

    for trade in trades:
        pnl = trade.get('pnl', 0)
        exit_reason = trade.get('reason', '')

        # Winner = any positive PnL
        if pnl > 0:
            winners.append(trade)
        else:
            losers.append(trade)

    return winners, losers

def analyze_exit_patterns(trades):
    """Analyze exit patterns to understand quality."""
    exit_patterns = defaultdict(lambda: {'count': 0, 'total_pnl': 0, 'trades': []})

    for trade in trades:
        reason = trade.get('reason', '')

        # Categorize exit reasons
        if 'hard_sl' in reason:
            category = 'hard_sl'
        elif 'sl_post_t1' in reason:
            category = 'sl_post_t1'
        elif 'sl_post_t2' in reason:
            category = 'sl_post_t2'
        elif 'sl_post_t3' in reason:
            category = 'sl_post_t3'
        elif 'eod' in reason:
            category = 'eod_squareoff'
        elif 't1_hit' in reason or 't2_hit' in reason or 't3_hit' in reason:
            category = 'target_hit'
        else:
            category = 'other'

        exit_patterns[category]['count'] += 1
        exit_patterns[category]['total_pnl'] += trade.get('pnl', 0)
        exit_patterns[category]['trades'].append(trade)

    return exit_patterns

def analyze_characteristics(trades, label):
    """Analyze characteristics of a set of trades."""
    if not trades:
        return {}

    # Setup distribution
    setups = defaultdict(int)
    regimes = defaultdict(int)
    symbols = defaultdict(int)

    total_pnl = 0

    for trade in trades:
        setups[trade.get('setup_type', 'unknown')] += 1
        regimes[trade.get('regime', 'unknown')] += 1
        symbols[trade.get('symbol', 'unknown')] += 1
        total_pnl += trade.get('pnl', 0)

    return {
        'count': len(trades),
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / len(trades),
        'setups': dict(setups),
        'regimes': dict(regimes),
        'top_symbols': dict(sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:10])
    }

def main():
    print("Loading all trades from analytics.jsonl files...")
    all_trades = load_all_analytics()
    print(f"Total trades loaded: {len(all_trades)}")

    # Categorize
    winners, losers = categorize_trades(all_trades)
    print(f"\nWinners: {len(winners)} ({len(winners)/len(all_trades)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(all_trades)*100:.1f}%)")

    # Analyze exit patterns
    print("\n" + "="*80)
    print("EXIT PATTERN DISTRIBUTION")
    print("="*80)

    exit_patterns = analyze_exit_patterns(all_trades)

    for category in ['hard_sl', 'eod_squareoff', 'sl_post_t1', 'sl_post_t2', 'sl_post_t3', 'target_hit', 'other']:
        if category in exit_patterns:
            data = exit_patterns[category]
            print(f"\n{category.upper()}:")
            print(f"  Count: {data['count']} ({data['count']/len(all_trades)*100:.1f}%)")
            print(f"  Total PnL: ₹{data['total_pnl']:.2f}")
            print(f"  Avg PnL: ₹{data['total_pnl']/data['count']:.2f}")

    # Analyze characteristics
    print("\n" + "="*80)
    print("WINNERS ANALYSIS")
    print("="*80)

    winner_stats = analyze_characteristics(winners, "Winners")
    print(f"\nCount: {winner_stats['count']}")
    print(f"Total PnL: ₹{winner_stats['total_pnl']:.2f}")
    print(f"Avg PnL: ₹{winner_stats['avg_pnl']:.2f}")

    print("\nSetup Distribution:")
    for setup, count in sorted(winner_stats['setups'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {setup}: {count} ({count/winner_stats['count']*100:.1f}%)")

    print("\nRegime Distribution:")
    for regime, count in sorted(winner_stats['regimes'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {regime}: {count} ({count/winner_stats['count']*100:.1f}%)")

    print("\nTop Symbols:")
    for symbol, count in list(winner_stats['top_symbols'].items())[:5]:
        print(f"  {symbol}: {count} trades")

    # Losers analysis
    print("\n" + "="*80)
    print("LOSERS ANALYSIS")
    print("="*80)

    loser_stats = analyze_characteristics(losers, "Losers")
    print(f"\nCount: {loser_stats['count']}")
    print(f"Total PnL: ₹{loser_stats['total_pnl']:.2f}")
    print(f"Avg PnL: ₹{loser_stats['avg_pnl']:.2f}")

    print("\nSetup Distribution:")
    for setup, count in sorted(loser_stats['setups'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {setup}: {count} ({count/loser_stats['count']*100:.1f}%)")

    print("\nRegime Distribution:")
    for regime, count in sorted(loser_stats['regimes'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {regime}: {count} ({count/loser_stats['count']*100:.1f}%)")

    print("\nTop Symbols:")
    for symbol, count in list(loser_stats['top_symbols'].items())[:5]:
        print(f"  {symbol}: {count} trades")

    # Key Insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    target_hit_count = exit_patterns.get('target_hit', {}).get('count', 0)
    hard_sl_count = exit_patterns.get('hard_sl', {}).get('count', 0)
    eod_count = exit_patterns.get('eod_squareoff', {}).get('count', 0)

    print(f"\n1. TARGET HIT RATE: {target_hit_count}/{len(all_trades)} = {target_hit_count/len(all_trades)*100:.1f}%")
    print(f"   - This is HALF of expected 25-30% baseline")

    print(f"\n2. HARD SL RATE: {hard_sl_count}/{len(all_trades)} = {hard_sl_count/len(all_trades)*100:.1f}%")
    print(f"   - Stopped out before hitting T1")

    print(f"\n3. EOD SQUAREOFF RATE: {eod_count}/{len(all_trades)} = {eod_count/len(all_trades)*100:.1f}%")
    print(f"   - Stocks that went nowhere (bad selection)")

    print(f"\n4. COMBINED BAD TRADES: {hard_sl_count + eod_count}/{len(all_trades)} = {(hard_sl_count + eod_count)/len(all_trades)*100:.1f}%")
    print(f"   - This is the stock selection quality problem!")

if __name__ == '__main__':
    main()
