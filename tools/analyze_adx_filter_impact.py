#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze ADX filter impact on ALL trades (winners + losers)

Questions:
1. When should ADX be checked? (Answer: At decision time on 5m bar)
2. How many winning trades would we lose with ADX > 20 filter?
3. What's the net P&L impact?
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json
from pathlib import Path
from collections import defaultdict

BACKTEST_DIR = Path("backtest_20251106-165927_extracted/20251106-165927_full/20251106-165927")

def load_all_trades():
    """Load ALL trades (winners and losers) from backtest"""

    all_trades = []

    session_dirs = sorted(BACKTEST_DIR.glob('20*'))

    for session_dir in session_dirs:
        events_file = session_dir / 'events.jsonl'

        if not events_file.exists():
            continue

        # Read all DECISION events to get plan details
        decisions = {}

        with open(events_file, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)

                        if event.get('type') == 'DECISION':
                            symbol = event.get('symbol')
                            plan = event.get('plan', {})
                            indicators = plan.get('indicators', {})

                            decisions[symbol] = {
                                'adx': indicators.get('adx14', 0) or 0,
                                'rsi': indicators.get('rsi14', 50) or 50,
                                'strategy': plan.get('strategy', 'unknown'),
                                'regime': plan.get('regime', 'unknown'),
                                'bias': plan.get('bias', 'unknown'),
                            }
                    except:
                        continue

        # Now read EXIT events and match with decisions
        analytics_file = session_dir / 'analytics.jsonl'

        if analytics_file.exists():
            with open(analytics_file, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)

                            if data.get('stage') == 'EXIT':
                                symbol = data.get('symbol')
                                reason = data.get('reason')
                                pnl = float(data.get('pnl', 0))

                                # Get decision data
                                decision_data = decisions.get(symbol, {})

                                all_trades.append({
                                    'date': session_dir.name,
                                    'symbol': symbol,
                                    'exit_reason': reason,
                                    'pnl': pnl,
                                    'adx': decision_data.get('adx', 0),
                                    'rsi': decision_data.get('rsi', 50),
                                    'strategy': decision_data.get('strategy', 'unknown'),
                                    'regime': decision_data.get('regime', 'unknown'),
                                    'bias': decision_data.get('bias', 'unknown'),
                                    'is_winner': pnl > 0,
                                    'is_hard_sl': reason == 'hard_sl',
                                })
                        except:
                            continue

    return all_trades

def main():
    print("="*120)
    print("ADX FILTER IMPACT ANALYSIS")
    print("="*120)
    print()

    # Load all trades
    all_trades = load_all_trades()

    print(f"Total trades: {len(all_trades)}")
    print()

    # Separate winners and losers
    winners = [t for t in all_trades if t['is_winner']]
    losers = [t for t in all_trades if not t['is_winner']]
    hard_sl = [t for t in all_trades if t['is_hard_sl']]

    print("="*120)
    print("BASELINE (NO ADX FILTER)")
    print("="*120)
    print()
    print(f"Total trades: {len(all_trades)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(all_trades)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(all_trades)*100:.1f}%)")
    print(f"Hard SL: {len(hard_sl)} ({len(hard_sl)/len(all_trades)*100:.1f}%)")
    print()

    total_pnl = sum(t['pnl'] for t in all_trades)
    winner_pnl = sum(t['pnl'] for t in winners)
    loser_pnl = sum(t['pnl'] for t in losers)

    print(f"Total P&L: Rs.{total_pnl:.2f}")
    print(f"  Winners: Rs.{winner_pnl:.2f}")
    print(f"  Losers: Rs.{loser_pnl:.2f}")
    print()

    # Test different ADX thresholds
    print("="*120)
    print("ADX FILTER IMPACT ANALYSIS")
    print("="*120)
    print()

    thresholds = [15, 18, 20, 22, 25, 30]

    print(f"{'ADX >':>8} {'Trades':>8} {'Winners':>8} {'Losers':>8} {'Hard SL':>8} "
          f"{'Win %':>8} {'Lost Winners':>14} {'Lost Winner $':>16} "
          f"{'Avoided SL':>12} {'Avoided SL $':>14} {'Net P&L':>12} {'Net Impact':>12}")
    print("-"*120)

    for threshold in thresholds:
        # Filter trades with ADX > threshold
        filtered = [t for t in all_trades if t['adx'] > threshold]

        # What did we lose?
        rejected = [t for t in all_trades if t['adx'] <= threshold]
        rejected_winners = [t for t in rejected if t['is_winner']]
        rejected_losers = [t for t in rejected if not t['is_winner']]
        rejected_hard_sl = [t for t in rejected if t['is_hard_sl']]

        # What's left?
        filtered_winners = [t for t in filtered if t['is_winner']]
        filtered_losers = [t for t in filtered if not t['is_winner']]
        filtered_hard_sl = [t for t in filtered if t['is_hard_sl']]

        # P&L calculations
        filtered_pnl = sum(t['pnl'] for t in filtered)
        rejected_winner_pnl = sum(t['pnl'] for t in rejected_winners)
        rejected_loser_pnl = sum(t['pnl'] for t in rejected_losers)
        net_impact = filtered_pnl - total_pnl

        win_rate = len(filtered_winners) / len(filtered) * 100 if len(filtered) > 0 else 0

        print(f"{threshold:>8} {len(filtered):>8} {len(filtered_winners):>8} {len(filtered_losers):>8} "
              f"{len(filtered_hard_sl):>8} {win_rate:>7.1f}% "
              f"{len(rejected_winners):>14} Rs.{rejected_winner_pnl:>12.2f} "
              f"{len(rejected_hard_sl):>12} Rs.{abs(rejected_loser_pnl):>12.2f} "
              f"Rs.{filtered_pnl:>9.2f} Rs.{net_impact:>9.2f}")

    print()

    # Detailed analysis for ADX > 20
    print("="*120)
    print("DETAILED ANALYSIS: ADX > 20 FILTER")
    print("="*120)
    print()

    threshold = 20
    filtered = [t for t in all_trades if t['adx'] > threshold]
    rejected = [t for t in all_trades if t['adx'] <= threshold]

    rejected_winners = [t for t in rejected if t['is_winner']]
    rejected_hard_sl = [t for t in rejected if t['is_hard_sl']]

    print(f"REJECTED TRADES (ADX <= {threshold}):")
    print(f"  Total rejected: {len(rejected)}")
    print(f"  Winners lost: {len(rejected_winners)}")
    print(f"  Hard SL avoided: {len(rejected_hard_sl)}")
    print()

    # Show rejected winners
    if len(rejected_winners) > 0:
        print("REJECTED WINNERS (Profit we'd lose):")
        print(f"{'Date':<12} {'Symbol':<20} {'ADX':>6} {'Exit Reason':<20} {'P&L':>12}")
        print("-"*120)

        for t in sorted(rejected_winners, key=lambda x: x['pnl'], reverse=True)[:15]:
            print(f"{t['date']:<12} {t['symbol']:<20} {t['adx']:>6.1f} {t['exit_reason']:<20} Rs.{t['pnl']:>9.2f}")

        print()

    # Show rejected hard_sl
    if len(rejected_hard_sl) > 0:
        print("REJECTED HARD_SL (Losses we'd avoid):")
        print(f"{'Date':<12} {'Symbol':<20} {'ADX':>6} {'Strategy':<20} {'P&L':>12}")
        print("-"*120)

        for t in sorted(rejected_hard_sl, key=lambda x: x['pnl'])[:15]:
            print(f"{t['date']:<12} {t['symbol']:<20} {t['adx']:>6.1f} {t['strategy']:<20} Rs.{t['pnl']:>9.2f}")

        print()

    # Net analysis
    rejected_winner_pnl = sum(t['pnl'] for t in rejected_winners)
    rejected_loser_pnl = sum(t['pnl'] for t in rejected if not t['is_winner'])
    net_pnl_impact = rejected_loser_pnl + rejected_winner_pnl  # Negative loser + positive winner

    print("="*120)
    print("NET IMPACT OF ADX > 20 FILTER")
    print("="*120)
    print()
    print(f"Winners we'd lose: {len(rejected_winners)} trades → Rs.{rejected_winner_pnl:.2f} (COST)")
    print(f"Losers we'd avoid: {len(rejected) - len(rejected_winners)} trades → Rs.{rejected_loser_pnl:.2f} (BENEFIT)")
    print()
    print(f"NET BENEFIT: Rs.{net_pnl_impact:.2f}")
    print()

    if net_pnl_impact > 0:
        print("✓ FILTER IS BENEFICIAL - Avoiding losers outweighs lost winners")
    else:
        print("✗ FILTER IS HARMFUL - Lost winners exceed avoided losers")

    print()

    # When should ADX be checked?
    print("="*120)
    print("ADX MEASUREMENT TIMING")
    print("="*120)
    print()
    print("ANSWER: ADX should be checked at DECISION time on the 5-minute bar")
    print()
    print("Professional standard:")
    print("  1. Scanner identifies potential setup (any timeframe)")
    print("  2. At decision time (5m bar close), calculate ADX_14 on 5m timeframe")
    print("  3. If ADX_14 > 20: Proceed with entry validation")
    print("  4. If ADX_14 <= 20: Reject trade (insufficient trend strength)")
    print()
    print("This is already what your system does - the ADX value in 'plan.indicators.adx14'")
    print("is calculated at decision time. You just need to add the filter.")
    print()

    # Strategy-specific breakdown
    print("="*120)
    print("ADX IMPACT BY STRATEGY")
    print("="*120)
    print()

    strategies = defaultdict(lambda: {'total': 0, 'winners': 0, 'losers': 0, 'hard_sl': 0,
                                     'rejected': 0, 'rejected_winners': 0, 'rejected_hard_sl': 0})

    for t in all_trades:
        strategy = t['strategy']
        strategies[strategy]['total'] += 1

        if t['is_winner']:
            strategies[strategy]['winners'] += 1
        else:
            strategies[strategy]['losers'] += 1

        if t['is_hard_sl']:
            strategies[strategy]['hard_sl'] += 1

        if t['adx'] <= threshold:
            strategies[strategy]['rejected'] += 1
            if t['is_winner']:
                strategies[strategy]['rejected_winners'] += 1
            if t['is_hard_sl']:
                strategies[strategy]['rejected_hard_sl'] += 1

    print(f"{'Strategy':<25} {'Total':>8} {'Winners':>8} {'Hard SL':>8} {'Rejected':>10} "
          f"{'Lost W':>8} {'Avoid SL':>10} {'Impact':>10}")
    print("-"*120)

    for strategy, stats in sorted(strategies.items()):
        total = stats['total']
        rejected = stats['rejected']
        lost_winners = stats['rejected_winners']
        avoided_sl = stats['rejected_hard_sl']

        impact = "✓ GOOD" if avoided_sl >= lost_winners else "✗ BAD" if lost_winners > avoided_sl else "~ NEUTRAL"

        print(f"{strategy:<25} {total:>8} {stats['winners']:>8} {stats['hard_sl']:>8} "
              f"{rejected:>10} {lost_winners:>8} {avoided_sl:>10} {impact:>10}")

    print()
    print("="*120)

if __name__ == "__main__":
    main()
