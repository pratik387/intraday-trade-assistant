#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Analysis by DETECTED Regime

Analyzes how setups perform based on the regime the system ACTUALLY detected,
not the historical regime label.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251114-125524_extracted\20251114-125524_full\20251114-125524")

def main():
    print("=" * 80)
    print("PERFORMANCE ANALYSIS BY DETECTED REGIME")
    print("=" * 80)

    # Collect trades with their detected regimes from analytics
    trades_by_regime = defaultdict(list)

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        analytics_file = session_dir / "analytics.jsonl"
        if analytics_file.exists():
            with open(analytics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        trade = json.loads(line)
                        regime = trade.get('regime', 'unknown')
                        trades_by_regime[regime].append(trade)

    print(f"\nLoaded {sum(len(t) for t in trades_by_regime.values())} total trades")
    print(f"Regimes detected: {list(trades_by_regime.keys())}")

    # Overall performance by regime
    print(f"\n" + "=" * 80)
    print("OVERALL PERFORMANCE BY DETECTED REGIME")
    print("=" * 80)

    regime_stats = {}
    for regime in sorted(trades_by_regime.keys()):
        trades = trades_by_regime[regime]
        total = len(trades)
        winners = sum(1 for t in trades if t.get('pnl', 0) > 0)
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        win_rate = (winners / total * 100) if total > 0 else 0
        avg_pnl = total_pnl / total if total > 0 else 0

        regime_stats[regime] = {
            'trades': total,
            'winners': winners,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        }

        print(f"\n{regime.upper()}:")
        print(f"  Trades: {total}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total PnL: Rs.{total_pnl:,.0f}")
        print(f"  Avg PnL/Trade: Rs.{avg_pnl:,.0f}")

    # Setup performance by regime
    print(f"\n" + "=" * 80)
    print("SETUP PERFORMANCE BY DETECTED REGIME")
    print("=" * 80)

    for regime in sorted(trades_by_regime.keys()):
        print(f"\n{regime.upper()}:")
        print(f"{'  Setup Type':<30} {'Trades':>8} {'Win%':>8} {'PnL':>12} {'Avg/Trade':>12}")
        print("  " + "-" * 72)

        # Group by setup type
        setup_stats = defaultdict(lambda: {
            'trades': 0,
            'winners': 0,
            'pnl': 0.0
        })

        for trade in trades_by_regime[regime]:
            setup = trade.get('setup_type', 'unknown')
            pnl = trade.get('pnl', 0)

            setup_stats[setup]['trades'] += 1
            setup_stats[setup]['pnl'] += pnl
            if pnl > 0:
                setup_stats[setup]['winners'] += 1

        # Sort by PnL
        for setup, stats in sorted(setup_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            trades = stats['trades']
            winners = stats['winners']
            win_rate = (winners / trades * 100) if trades > 0 else 0
            pnl = stats['pnl']
            avg_pnl = pnl / trades if trades > 0 else 0

            marker = ""
            if win_rate >= 60:
                marker = " [STRONG]"
            elif win_rate < 40:
                marker = " [WEAK]"

            print(f"  {setup:<30} {trades:>8} {win_rate:>7.1f}% Rs.{pnl:>9,.0f} Rs.{avg_pnl:>9,.0f}{marker}")

    # Regime fitness matrix
    print(f"\n" + "=" * 80)
    print("REGIME-SETUP FITNESS MATRIX")
    print("=" * 80)

    print(f"\nSetup performance across regimes (Win Rate %):")
    print(f"{'Setup Type':<25} {'chop':>10} {'trend_up':>10} {'trend_down':>12}")
    print("-" * 60)

    # Get all unique setup types
    all_setups = set()
    for trades in trades_by_regime.values():
        for trade in trades:
            all_setups.add(trade.get('setup_type', 'unknown'))

    # Build matrix
    for setup in sorted(all_setups):
        row = [setup]
        for regime in ['chop', 'trend_up', 'trend_down']:
            if regime in trades_by_regime:
                regime_trades = [t for t in trades_by_regime[regime] if t.get('setup_type') == setup]
                if regime_trades:
                    winners = sum(1 for t in regime_trades if t.get('pnl', 0) > 0)
                    wr = (winners / len(regime_trades) * 100)

                    # Color code
                    if wr >= 60:
                        marker = f"{wr:5.1f}% [GOOD]"
                    elif wr < 40:
                        marker = f"{wr:5.1f}% [POOR]"
                    else:
                        marker = f"{wr:5.1f}%"

                    row.append(f"{marker:>10}")
                else:
                    row.append("    -     ")
            else:
                row.append("    -     ")

        print(f"{row[0]:<25} {row[1] if len(row) > 1 else '-':>10} {row[2] if len(row) > 2 else '-':>10} {row[3] if len(row) > 3 else '-':>12}")

    # Critical insights
    print(f"\n" + "=" * 80)
    print("CRITICAL INSIGHTS & RECOMMENDATIONS")
    print("=" * 80)

    # Find best/worst combinations
    combos = []
    for regime in trades_by_regime.keys():
        setup_stats = defaultdict(lambda: {'trades': 0, 'winners': 0, 'pnl': 0.0})
        for trade in trades_by_regime[regime]:
            setup = trade.get('setup_type', 'unknown')
            pnl = trade.get('pnl', 0)
            setup_stats[setup]['trades'] += 1
            setup_stats[setup]['pnl'] += pnl
            if pnl > 0:
                setup_stats[setup]['winners'] += 1

        for setup, stats in setup_stats.items():
            if stats['trades'] >= 3:  # Minimum sample size
                wr = (stats['winners'] / stats['trades'] * 100)
                combos.append((regime, setup, wr, stats['trades'], stats['pnl']))

    print(f"\nBEST Regime-Setup Combinations (Win Rate >= 60%, min 3 trades):")
    best = [c for c in combos if c[2] >= 60]
    best.sort(key=lambda x: x[2], reverse=True)
    for regime, setup, wr, trades, pnl in best[:10]:
        print(f"  - {setup:<30} in {regime:<12}: {wr:5.1f}% WR ({trades} trades, Rs.{pnl:,.0f})")

    print(f"\nWORST Regime-Setup Combinations (Win Rate < 40%, min 3 trades):")
    worst = [c for c in combos if c[2] < 40]
    worst.sort(key=lambda x: x[2])
    for regime, setup, wr, trades, pnl in worst[:10]:
        print(f"  - {setup:<30} in {regime:<12}: {wr:5.1f}% WR ({trades} trades, Rs.{pnl:,.0f}) [AVOID]")

if __name__ == '__main__':
    main()
