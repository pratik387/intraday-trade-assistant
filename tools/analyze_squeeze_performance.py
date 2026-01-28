"""
Squeeze Regime Performance Analysis

Analyzes which setups are trading in squeeze regime and their performance.
"""

import json
import os
from collections import defaultdict

BACKTEST_DIR = 'backtest_20251211-032449_extracted'


def analyze_squeeze_trades():
    """Load and analyze all squeeze regime trades from analytics.jsonl (final exits only)."""
    squeeze_trades = []
    all_trades_by_regime = defaultdict(list)
    seen_trades = set()

    for date_folder in sorted(os.listdir(BACKTEST_DIR)):
        analytics_file = f'{BACKTEST_DIR}/{date_folder}/analytics.jsonl'
        if not os.path.exists(analytics_file):
            continue

        with open(analytics_file) as f:
            for line in f:
                record = json.loads(line)

                # Only count final exits to avoid double counting
                if not record.get('is_final_exit', False):
                    continue

                trade_id = record.get('trade_id')
                if trade_id in seen_trades:
                    continue
                seen_trades.add(trade_id)

                regime = record.get('regime', 'unknown')
                setup_type = record.get('setup_type', '')

                trade = {
                    'date': date_folder,
                    'symbol': record.get('symbol'),
                    'setup_type': setup_type,
                    'regime': regime,
                    'pnl': record.get('total_trade_pnl', record.get('pnl', 0)),
                    'exit_reason': record.get('reason', ''),
                    'bias': record.get('bias'),
                }

                all_trades_by_regime[regime].append(trade)

                if regime == 'squeeze':
                    squeeze_trades.append(trade)

    return squeeze_trades, all_trades_by_regime


def main():
    squeeze_trades, all_by_regime = analyze_squeeze_trades()

    print('=' * 80)
    print('SQUEEZE REGIME PERFORMANCE ANALYSIS')
    print('=' * 80)
    print()

    print('REGIME DETECTION CRITERIA:')
    print('-' * 80)
    print('  ADX < 15 AND BB width in lowest 30% quantile')
    print('  = Very low volatility + weak trend = compression/consolidation')
    print()

    print('REGIME TRADE COUNTS:')
    print('-' * 80)
    for regime, trades in sorted(all_by_regime.items()):
        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        wr = wins / len(trades) * 100 if trades else 0
        avg_pnl = total_pnl / len(trades) if trades else 0
        print(f"  {regime:12}: {len(trades):3} trades | {wr:5.1f}% WR | Rs {total_pnl:>8,.0f} | Rs {avg_pnl:>6,.0f}/trade")
    print()

    print(f'SQUEEZE REGIME: {len(squeeze_trades)} trades')
    print('=' * 80)

    # Group by setup type
    setup_perf = defaultdict(lambda: {'trades': [], 'pnl': 0, 'wins': 0})
    for t in squeeze_trades:
        setup = t['setup_type']
        setup_perf[setup]['trades'].append(t)
        setup_perf[setup]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            setup_perf[setup]['wins'] += 1

    print()
    print('SQUEEZE SETUP PERFORMANCE (sorted by PnL):')
    print('-' * 80)
    print(f"{'Setup Type':<40} {'Trades':>6} {'WR%':>7} {'Total PnL':>12} {'Avg PnL':>10}")
    print('-' * 80)

    sorted_setups = sorted(setup_perf.items(), key=lambda x: x[1]['pnl'], reverse=True)
    for setup, data in sorted_setups:
        count = len(data['trades'])
        wr = data['wins'] / count * 100 if count > 0 else 0
        avg = data['pnl'] / count if count > 0 else 0
        marker = '***' if data['pnl'] < 0 else ''
        print(f"  {setup:<38} {count:>6} {wr:>6.1f}% Rs {data['pnl']:>10,.0f} Rs {avg:>8,.0f} {marker}")

    print()
    print('EXIT REASON BREAKDOWN FOR SQUEEZE:')
    print('-' * 80)
    exit_reasons = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for t in squeeze_trades:
        reason = t['exit_reason'] or 'unknown'
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl'] += t['pnl']

    for reason, data in sorted(exit_reasons.items(), key=lambda x: x[1]['pnl'], reverse=True):
        print(f"  {reason:<20}: {data['count']:3} exits | Rs {data['pnl']:>8,.0f}")

    # Identify worst performers
    print()
    print('SQUEEZE - WORST PERFORMING SETUPS (negative PnL):')
    print('-' * 80)
    for setup, data in sorted_setups:
        if data['pnl'] < 0:
            print(f"  {setup}: Rs {data['pnl']:,.0f} ({len(data['trades'])} trades)")
            # Show individual trades
            for t in data['trades'][:5]:
                print(f"    - {t['date']} {t['symbol']}: Rs {t['pnl']:,.0f} ({t['exit_reason']})")

    # Bias distribution in squeeze
    print()
    print('BIAS DISTRIBUTION IN SQUEEZE TRADES:')
    print('-' * 80)
    bias_perf = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
    for t in squeeze_trades:
        bias = t['bias'] or 'unknown'
        bias_perf[bias]['count'] += 1
        bias_perf[bias]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            bias_perf[bias]['wins'] += 1

    for bias, data in sorted(bias_perf.items(), key=lambda x: x[1]['pnl'], reverse=True):
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"  {bias}: {data['count']} trades | {wr:.1f}% WR | Rs {data['pnl']:,.0f}")

    # Compare with allowed setups config
    print()
    print('=' * 80)
    print('ALLOWED SETUPS IN SQUEEZE (from config):')
    print('=' * 80)
    allowed = [
        'squeeze_release_long', 'squeeze_release_short',
        'orb_breakout_long', 'orb_breakout_short',
        'order_block_long', 'order_block_short',
        'fair_value_gap_long', 'fair_value_gap_short',
        'liquidity_sweep_long', 'liquidity_sweep_short',
        'break_of_structure_long', 'break_of_structure_short',
        'range_bounce_long', 'range_bounce_short',
        'first_hour_momentum_short'
    ]

    print()
    print('Configured as allowed:')
    for setup in allowed:
        if setup in setup_perf:
            data = setup_perf[setup]
            count = len(data['trades'])
            wr = data['wins'] / count * 100 if count > 0 else 0
            print(f"  {setup:<35}: {count:3} trades | {wr:5.1f}% WR | Rs {data['pnl']:>8,.0f}")
        else:
            print(f"  {setup:<35}: 0 trades (not triggered)")

    # Check for setups trading in squeeze that AREN'T in allowed list
    print()
    print('Setups trading in squeeze NOT in allowed list:')
    for setup in setup_perf.keys():
        if setup not in allowed:
            data = setup_perf[setup]
            count = len(data['trades'])
            wr = data['wins'] / count * 100 if count > 0 else 0
            print(f"  {setup:<35}: {count:3} trades | {wr:5.1f}% WR | Rs {data['pnl']:>8,.0f} **UNEXPECTED**")


if __name__ == '__main__':
    main()
