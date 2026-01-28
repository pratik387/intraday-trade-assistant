#!/usr/bin/env python3
"""
Analyze failure_fade trades by regime to understand which market conditions work best.
Uses the 6-month backtest data to determine optimal regimes for failure_fade setups.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def parse_failure_fade_by_regime(backtest_dir: Path) -> Dict:
    """Parse all failure_fade trades and group by regime."""

    regime_stats = defaultdict(lambda: {
        'trades': [],
        'decisions': 0,
        'triggered': 0,
        'wins': 0,
        'losses': 0,
        'total_pnl': 0.0,
        'by_setup': defaultdict(lambda: {
            'count': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0
        })
    })

    for session_dir in sorted(backtest_dir.glob("2*")):
        if not session_dir.is_dir():
            continue

        events_file = session_dir / "events.jsonl"
        if not events_file.exists():
            continue

        session_date = session_dir.name

        # Track trades by trade_id
        trades = {}

        with open(events_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line)
                    event_type = event.get('type')

                    if event_type == 'DECISION':
                        setup = event.get('decision', {}).get('setup_type', '')

                        if 'failure_fade' in setup:
                            trade_id = event.get('trade_id')
                            regime = event.get('decision', {}).get('regime', 'unknown')
                            symbol = event.get('symbol', '').replace('NSE:', '')

                            # Get detailed plan info
                            plan = event.get('plan', {})
                            quality = plan.get('quality', {})

                            trades[trade_id] = {
                                'session': session_date,
                                'symbol': symbol,
                                'setup': setup,
                                'regime': regime,
                                'trade_id': trade_id,
                                'decision_ts': event.get('ts'),
                                'structural_rr': quality.get('structural_rr'),
                                'triggered': False,
                                'pnl': 0.0,
                                'exit_reason': None
                            }

                            regime_stats[regime]['decisions'] += 1

                    elif event_type == 'TRIGGER':
                        trade_id = event.get('trade_id')
                        if trade_id in trades:
                            trades[trade_id]['triggered'] = True

                    elif event_type == 'EXIT':
                        trade_id = event.get('trade_id')
                        if trade_id in trades:
                            exit_info = event.get('exit', {})
                            pnl = exit_info.get('pnl', 0.0)

                            trades[trade_id]['pnl'] = pnl
                            trades[trade_id]['exit_reason'] = exit_info.get('reason', 'unknown')

                except Exception as e:
                    continue

        # Aggregate triggered trades by regime
        for trade_id, trade in trades.items():
            if trade['triggered']:
                regime = trade['regime']
                setup = trade['setup']
                pnl = trade['pnl']

                regime_stats[regime]['triggered'] += 1
                regime_stats[regime]['total_pnl'] += pnl
                regime_stats[regime]['trades'].append(trade)

                # Track by setup type within regime
                regime_stats[regime]['by_setup'][setup]['count'] += 1
                regime_stats[regime]['by_setup'][setup]['pnl'] += pnl

                if pnl > 0:
                    regime_stats[regime]['wins'] += 1
                    regime_stats[regime]['by_setup'][setup]['wins'] += 1
                elif pnl < 0:
                    regime_stats[regime]['losses'] += 1
                    regime_stats[regime]['by_setup'][setup]['losses'] += 1

    return regime_stats


def analyze_regime_performance(regime_stats: Dict):
    """Analyze and display regime performance for failure_fade trades."""

    print("=" * 100)
    print("FAILURE_FADE TRADES BY REGIME ANALYSIS")
    print("=" * 100)
    print()

    # Calculate overall stats
    total_decisions = sum(stats['decisions'] for stats in regime_stats.values())
    total_triggered = sum(stats['triggered'] for stats in regime_stats.values())
    total_wins = sum(stats['wins'] for stats in regime_stats.values())
    total_losses = sum(stats['losses'] for stats in regime_stats.values())
    total_pnl = sum(stats['total_pnl'] for stats in regime_stats.values())

    print(f"Overall Summary:")
    print(f"  Total failure_fade decisions: {total_decisions}")
    print(f"  Total triggered: {total_triggered}")
    print(f"  Total wins: {total_wins}")
    print(f"  Total losses: {total_losses}")
    print(f"  Overall WR: {total_wins/total_triggered*100:.1f}%" if total_triggered > 0 else "  Overall WR: N/A")
    print(f"  Total PnL: Rs.{total_pnl:.2f}")
    print()

    # Sort regimes by PnL
    sorted_regimes = sorted(regime_stats.items(),
                           key=lambda x: x[1]['total_pnl'],
                           reverse=True)

    print("=" * 100)
    print("REGIME BREAKDOWN")
    print("=" * 100)
    print()

    for regime, stats in sorted_regimes:
        decisions = stats['decisions']
        triggered = stats['triggered']
        wins = stats['wins']
        losses = stats['losses']
        pnl = stats['total_pnl']

        if triggered == 0:
            continue

        wr = wins / triggered * 100 if triggered > 0 else 0
        avg_pnl = pnl / triggered if triggered > 0 else 0
        trigger_rate = triggered / decisions * 100 if decisions > 0 else 0

        print(f"Regime: {regime.upper()}")
        print(f"  Decisions: {decisions}")
        print(f"  Triggered: {triggered} ({trigger_rate:.1f}% trigger rate)")
        print(f"  Wins: {wins} | Losses: {losses}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  Total PnL: Rs.{pnl:.2f}")
        print(f"  Avg PnL/trade: Rs.{avg_pnl:.2f}")
        print()

        # Breakdown by setup type within regime
        if len(stats['by_setup']) > 1:
            print(f"  Setup breakdown:")
            for setup, setup_stats in sorted(stats['by_setup'].items(),
                                            key=lambda x: x[1]['pnl'],
                                            reverse=True):
                count = setup_stats['count']
                setup_wins = setup_stats['wins']
                setup_losses = setup_stats['losses']
                setup_pnl = setup_stats['pnl']
                setup_wr = setup_wins / count * 100 if count > 0 else 0

                print(f"    {setup}: {count} trades, WR:{setup_wr:.1f}%, PnL:Rs.{setup_pnl:.2f}")
            print()

    # Key insights
    print("=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    print()

    # Find best regime
    best_regime = max(sorted_regimes, key=lambda x: x[1]['total_pnl'])
    worst_regime = min(sorted_regimes, key=lambda x: x[1]['total_pnl'])

    best_name, best_stats = best_regime
    worst_name, worst_stats = worst_regime

    if best_stats['triggered'] > 0:
        best_wr = best_stats['wins'] / best_stats['triggered'] * 100
        print(f"1. BEST REGIME: {best_name.upper()}")
        print(f"   - {best_stats['triggered']} trades, {best_wr:.1f}% WR, Rs.{best_stats['total_pnl']:.2f} PnL")
        print()

    if worst_stats['triggered'] > 0:
        worst_wr = worst_stats['wins'] / worst_stats['triggered'] * 100
        print(f"2. WORST REGIME: {worst_name.upper()}")
        print(f"   - {worst_stats['triggered']} trades, {worst_wr:.1f}% WR, Rs.{worst_stats['total_pnl']:.2f} PnL")
        print()

    # Analyze win rate by regime
    wr_by_regime = []
    for regime, stats in regime_stats.items():
        if stats['triggered'] > 5:  # Only consider regimes with 5+ trades
            wr = stats['wins'] / stats['triggered'] * 100
            wr_by_regime.append((regime, wr, stats['triggered'], stats['total_pnl']))

    if wr_by_regime:
        wr_by_regime.sort(key=lambda x: x[1], reverse=True)

        print("3. WIN RATE RANKING (regimes with 5+ trades):")
        for regime, wr, count, pnl in wr_by_regime:
            print(f"   {regime:10s}: {wr:5.1f}% WR ({count} trades, Rs.{pnl:.2f})")
        print()

    # Recommendations
    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print()

    # Find profitable regimes
    profitable_regimes = [r for r, s in regime_stats.items()
                         if s['total_pnl'] > 0 and s['triggered'] >= 3]
    loss_making_regimes = [r for r, s in regime_stats.items()
                          if s['total_pnl'] < 0 and s['triggered'] >= 3]

    if profitable_regimes:
        print("ALLOW failure_fade in these regimes:")
        for regime in profitable_regimes:
            stats = regime_stats[regime]
            wr = stats['wins'] / stats['triggered'] * 100 if stats['triggered'] > 0 else 0
            print(f"  - {regime}: {stats['triggered']} trades, {wr:.1f}% WR, Rs.{stats['total_pnl']:.2f}")
        print()

    if loss_making_regimes:
        print("BLOCK failure_fade in these regimes:")
        for regime in loss_making_regimes:
            stats = regime_stats[regime]
            wr = stats['wins'] / stats['triggered'] * 100 if stats['triggered'] > 0 else 0
            print(f"  - {regime}: {stats['triggered']} trades, {wr:.1f}% WR, Rs.{stats['total_pnl']:.2f}")
        print()

    # Detailed trade examples
    print("=" * 100)
    print("SAMPLE TRADES BY REGIME (Best and Worst)")
    print("=" * 100)
    print()

    for regime_name, regime_stats_data in sorted_regimes[:2]:  # Top 2 regimes
        trades = regime_stats_data['trades']
        if not trades:
            continue

        print(f"{regime_name.upper()} - Sample trades:")
        for i, trade in enumerate(sorted(trades, key=lambda x: x['pnl'], reverse=True)[:5], 1):
            print(f"  {i}. {trade['session']} {trade['symbol']:12s} | {trade['setup']:25s} | "
                  f"PnL:Rs.{trade['pnl']:7.2f} | RR:{trade.get('structural_rr', 0):.2f} | "
                  f"Exit:{trade.get('exit_reason', 'unknown')}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_failure_fade_regimes.py <backtest_dir>")
        print("Example: python tools/analyze_failure_fade_regimes.py backtest_20251119-082113_extracted/20251119-082113_full/20251119-082113")
        sys.exit(1)

    backtest_dir = Path(sys.argv[1])

    if not backtest_dir.exists():
        print(f"Error: Directory {backtest_dir} does not exist")
        sys.exit(1)

    print(f"Analyzing failure_fade trades from: {backtest_dir}")
    print()

    regime_stats = parse_failure_fade_by_regime(backtest_dir)
    analyze_regime_performance(regime_stats)


if __name__ == '__main__':
    main()
