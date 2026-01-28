#!/usr/bin/env python3
"""
Comprehensive Exit Scenario Analyzer

Analyzes ALL exit patterns:
1. SL before T1 (no targets hit)
2. T1 hit -> then SL (common pattern)
3. T2 hit -> then SL (reversal pattern)
4. T3 hit (perfect scenario)

Simulates realistic 33-33-33 PnL accounting for all scenarios.

Usage:
    python tools/analyze_all_exit_scenarios.py <backtest_dir>
"""

import re
import sys
from pathlib import Path
from collections import defaultdict


def parse_trade_logs(backtest_dir):
    """Parse all trade_logs.log files to extract trade data with exit sequence."""
    trades = {}
    sessions_dir = Path(backtest_dir)

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        trade_log_file = session_dir / "trade_logs.log"
        if not trade_log_file.exists():
            continue

        with open(trade_log_file, 'r') as f:
            for line in f:
                if 'TRIGGER_EXEC' in line:
                    match = re.search(r'TRIGGER_EXEC \| ([^\|]+) \| (\w+) (\d+) @ ([\d.]+)', line)
                    if match:
                        symbol, side, qty, price = match.groups()
                        symbol = symbol.strip()

                        if symbol not in trades:
                            trades[symbol] = {
                                'symbol': symbol,
                                'side': side,
                                'entry_qty': int(qty),
                                'entry_price': float(price),
                                'exits': [],
                                'exit_sequence': []
                            }

                elif 'EXIT |' in line:
                    match_symbol = re.search(r'EXIT \| ([^\|]+) \|', line)
                    match_data = re.search(r'Qty: (\d+) \| Entry: Rs\.([\d.]+) \| Exit: Rs\.([\d.]+) \| PnL: Rs\.([-\d.]+) (.+)$', line)

                    if match_symbol and match_data:
                        symbol = match_symbol.group(1).strip()
                        qty, entry, exit_price, pnl, reason = match_data.groups()

                        if symbol in trades:
                            exit_event = {
                                'qty': int(qty),
                                'exit_price': float(exit_price),
                                'pnl': float(pnl),
                                'reason': reason.strip()
                            }
                            trades[symbol]['exits'].append(exit_event)
                            trades[symbol]['exit_sequence'].append(reason.strip())

    return list(trades.values())


def classify_trade_pattern(trade):
    """
    Classify trade into one of 4 patterns:
    1. sl_only: SL hit before any targets
    2. t1_then_sl: T1 hit, then SL
    3. t2_then_sl: T2 hit, then SL (reversal)
    4. t3_or_better: T3 hit or better exit
    """
    exit_sequence = trade['exit_sequence']

    has_t1 = any('t1' in exit.lower() for exit in exit_sequence)
    has_t2 = any('t2' in exit.lower() for exit in exit_sequence)
    has_sl = any('sl' in exit.lower() for exit in exit_sequence)

    if has_t2:
        # Check if SL came after T2
        t2_index = next(i for i, exit in enumerate(exit_sequence) if 't2' in exit.lower())
        sl_after_t2 = any('sl' in exit_sequence[i].lower() for i in range(t2_index + 1, len(exit_sequence)))

        if sl_after_t2:
            return 't2_then_sl'
        else:
            return 't3_or_better'

    elif has_t1:
        # Check if SL came after T1
        t1_index = next(i for i, exit in enumerate(exit_sequence) if 't1' in exit.lower())
        sl_after_t1 = any('sl' in exit_sequence[i].lower() for i in range(t1_index + 1, len(exit_sequence)))

        if sl_after_t1:
            return 't1_then_sl'
        else:
            # T1 hit but no SL after (maybe EOD or other exit)
            return 't1_then_other'

    elif has_sl:
        return 'sl_only'

    else:
        # No clear pattern (maybe EOD, time stop, etc.)
        return 'other'


def simulate_33_33_33_all_scenarios(trades):
    """
    Simulate 33-33-33 split with ALL exit scenarios.

    Realistic scenarios:
    1. SL only: All 100% exit at SL (no partials taken)
    2. T1 -> SL: 33% at T1, 67% at SL
    3. T2 -> SL: 33% at T1, 33% at T2, 34% at SL
    4. T3 or better: 33% at T1, 33% at T2, 34% at T3
    """

    total_pnl_perfect = 0.0  # If all T3 hit
    total_pnl_realistic = 0.0  # Accounting for all scenarios
    total_loss_from_scenarios = 0.0

    pattern_stats = defaultdict(lambda: {'count': 0, 'pnl_perfect': 0.0, 'pnl_realistic': 0.0, 'loss': 0.0})
    detailed_trades = defaultdict(list)

    for trade in trades:
        entry = trade['entry_price']
        qty_total = trade['entry_qty']
        side = trade['side'].upper()
        exits = trade['exits']
        pattern = classify_trade_pattern(trade)

        # Build exit price map
        exit_prices = {}
        for exit_event in exits:
            reason = exit_event['reason'].lower()
            price = exit_event['exit_price']

            if 't1' in reason:
                exit_prices['t1'] = price
            elif 't2' in reason:
                exit_prices['t2'] = price
            elif 'sl' in reason:
                exit_prices['sl'] = price

        # Calculate R
        r_per_share = 0
        if 't1' in exit_prices:
            price_move_t1 = abs(exit_prices['t1'] - entry)
            r_per_share = price_move_t1
        elif 't2' in exit_prices:
            price_move_t2 = abs(exit_prices['t2'] - entry)
            r_per_share = price_move_t2 / 2.0
        elif 'sl' in exit_prices:
            # Estimate R from SL
            price_move_sl = abs(exit_prices['sl'] - entry)
            r_per_share = price_move_sl

        # Calculate T3 (3.0R)
        if r_per_share > 0:
            if side == 'BUY':
                exit_prices['t3'] = entry + (r_per_share * 3.0)
            else:
                exit_prices['t3'] = entry - (r_per_share * 3.0)

        # Simulate 33-33-33 split
        qty_t1 = int(qty_total * 0.33)
        qty_t2 = int(qty_total * 0.33)
        qty_remaining = qty_total - qty_t1 - qty_t2  # ~34%

        # PERFECT: All hit T3
        t1_price = exit_prices.get('t1', exit_prices.get('sl', entry))
        t2_price = exit_prices.get('t2', exit_prices.get('sl', entry))
        t3_price = exit_prices.get('t3', entry)

        if side == 'BUY':
            pnl_t1_perfect = qty_t1 * (t1_price - entry)
            pnl_t2_perfect = qty_t2 * (t2_price - entry)
            pnl_t3_perfect = qty_remaining * (t3_price - entry)
        else:
            pnl_t1_perfect = qty_t1 * (entry - t1_price)
            pnl_t2_perfect = qty_t2 * (entry - t2_price)
            pnl_t3_perfect = qty_remaining * (entry - t3_price)

        pnl_perfect = pnl_t1_perfect + pnl_t2_perfect + pnl_t3_perfect
        total_pnl_perfect += pnl_perfect

        # REALISTIC: Based on actual pattern
        if pattern == 'sl_only':
            # All exit at SL (no partials taken)
            sl_price = exit_prices.get('sl', entry)
            if side == 'BUY':
                pnl_realistic = qty_total * (sl_price - entry)
            else:
                pnl_realistic = qty_total * (entry - sl_price)

        elif pattern == 't1_then_sl':
            # 33% at T1, 67% at SL
            sl_price = exit_prices.get('sl', entry)
            if side == 'BUY':
                pnl_t1_real = qty_t1 * (t1_price - entry)
                pnl_remaining_real = (qty_t2 + qty_remaining) * (sl_price - entry)
            else:
                pnl_t1_real = qty_t1 * (entry - t1_price)
                pnl_remaining_real = (qty_t2 + qty_remaining) * (entry - sl_price)

            pnl_realistic = pnl_t1_real + pnl_remaining_real

        elif pattern == 't2_then_sl':
            # 33% at T1, 33% at T2, 34% at SL
            sl_price = exit_prices.get('sl', entry)
            if side == 'BUY':
                pnl_t1_real = qty_t1 * (t1_price - entry)
                pnl_t2_real = qty_t2 * (t2_price - entry)
                pnl_remaining_real = qty_remaining * (sl_price - entry)
            else:
                pnl_t1_real = qty_t1 * (entry - t1_price)
                pnl_t2_real = qty_t2 * (entry - t2_price)
                pnl_remaining_real = qty_remaining * (entry - sl_price)

            pnl_realistic = pnl_t1_real + pnl_t2_real + pnl_remaining_real

        else:  # t3_or_better or other
            # Use perfect scenario (T3 hit)
            pnl_realistic = pnl_perfect

        total_pnl_realistic += pnl_realistic
        loss = pnl_perfect - pnl_realistic
        total_loss_from_scenarios += loss

        # Track by pattern
        pattern_stats[pattern]['count'] += 1
        pattern_stats[pattern]['pnl_perfect'] += pnl_perfect
        pattern_stats[pattern]['pnl_realistic'] += pnl_realistic
        pattern_stats[pattern]['loss'] += loss

        detailed_trades[pattern].append({
            'symbol': trade['symbol'],
            'pnl_perfect': pnl_perfect,
            'pnl_realistic': pnl_realistic,
            'loss': loss
        })

    return {
        'total_pnl_perfect': total_pnl_perfect,
        'total_pnl_realistic': total_pnl_realistic,
        'total_loss': total_loss_from_scenarios,
        'pattern_stats': dict(pattern_stats),
        'detailed_trades': dict(detailed_trades)
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_all_exit_scenarios.py <backtest_dir>")
        sys.exit(1)

    backtest_dir = sys.argv[1]

    print(f"Loading trades from: {backtest_dir}")
    trades = parse_trade_logs(backtest_dir)

    if not trades:
        print(f"ERROR: No trade data found")
        sys.exit(1)

    print(f"Found {len(trades)} trades\n")

    # Classify all trades by pattern
    pattern_counts = defaultdict(int)
    for trade in trades:
        pattern = classify_trade_pattern(trade)
        pattern_counts[pattern] += 1

    print(f"{'='*80}")
    print(f"Exit Pattern Classification")
    print(f"{'='*80}\n")

    print(f"{'Pattern':<20} {'Count':>10} {'Percentage':>12}")
    print(f"{'-'*20} {'-'*10} {'-'*12}")

    pattern_names = {
        'sl_only': 'SL only (no targets)',
        't1_then_sl': 'T1 -> SL',
        't2_then_sl': 'T2 -> SL (reversal)',
        't3_or_better': 'T3 or better',
        't1_then_other': 'T1 -> other exit',
        'other': 'Other pattern'
    }

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        name = pattern_names.get(pattern, pattern)
        pct = (count / len(trades)) * 100
        print(f"{name:<20} {count:>10} {pct:>11.1f}%")

    print()

    # Simulate 33-33-33 with all scenarios
    simulation = simulate_33_33_33_all_scenarios(trades)

    print(f"{'='*80}")
    print(f"33-33-33 Split Simulation - ALL Exit Scenarios")
    print(f"{'='*80}\n")

    print(f"PERFECT SCENARIO (all runners hit T3):")
    print(f"  Total PnL: Rs. {simulation['total_pnl_perfect']:,.2f}")
    print()

    print(f"REALISTIC SCENARIO (accounting for ALL exit patterns):")
    print(f"  Total PnL: Rs. {simulation['total_pnl_realistic']:,.2f}")
    print(f"  Total loss from bad exits: Rs. {simulation['total_loss']:,.2f}")
    print(f"  Loss impact: {(simulation['total_loss']/simulation['total_pnl_perfect']*100):.1f}% of perfect PnL")
    print()

    print(f"COMPARISON:")
    print(f"  Perfect 33-33-33: Rs. {simulation['total_pnl_perfect']:,.2f}")
    print(f"  Realistic 33-33-33: Rs. {simulation['total_pnl_realistic']:,.2f}")
    print(f"  Actual 40-40-20: Rs. 4,096.82")
    print(f"  Improvement: Rs. {(simulation['total_pnl_realistic'] - 4096.82):,.2f} ({((simulation['total_pnl_realistic'] - 4096.82)/4096.82*100):+.1f}%)")
    print()

    # Breakdown by pattern
    print(f"{'='*80}")
    print(f"Loss Breakdown by Exit Pattern")
    print(f"{'='*80}\n")

    print(f"{'Pattern':<20} {'Trades':>8} {'Perfect PnL':>15} {'Realistic PnL':>15} {'Loss':>12}")
    print(f"{'-'*20} {'-'*8} {'-'*15} {'-'*15} {'-'*12}")

    for pattern in ['sl_only', 't1_then_sl', 't2_then_sl', 't3_or_better', 't1_then_other', 'other']:
        if pattern in simulation['pattern_stats']:
            stats = simulation['pattern_stats'][pattern]
            name = pattern_names.get(pattern, pattern)
            print(f"{name:<20} {stats['count']:>8} Rs. {stats['pnl_perfect']:>11,.2f} Rs. {stats['pnl_realistic']:>11,.2f} Rs. {stats['loss']:>8,.2f}")

    print()

    # Show worst trades from each pattern
    print(f"{'='*80}")
    print(f"Top 5 Worst Trades by Pattern")
    print(f"{'='*80}\n")

    for pattern in ['sl_only', 't1_then_sl', 't2_then_sl']:
        if pattern in simulation['detailed_trades']:
            trades_list = simulation['detailed_trades'][pattern]
            trades_list.sort(key=lambda x: x['loss'], reverse=True)

            name = pattern_names.get(pattern, pattern)
            print(f"\n{name}:")
            print(f"{'Symbol':<15} {'Perfect PnL':>12} {'Realistic PnL':>15} {'Loss':>12}")
            print(f"{'-'*15} {'-'*12} {'-'*15} {'-'*12}")

            for trade in trades_list[:5]:
                print(f"{trade['symbol']:<15} Rs. {trade['pnl_perfect']:>8,.2f} Rs. {trade['pnl_realistic']:>11,.2f} Rs. {trade['loss']:>8,.2f}")

    print(f"\n{'='*80}")
    print(f"FINAL CONCLUSION")
    print(f"{'='*80}\n")

    print(f"Realistic 33-33-33 PnL: Rs. {simulation['total_pnl_realistic']:,.2f}")
    print(f"Current 40-40-20 PnL: Rs. 4,096.82")
    print(f"Improvement: {((simulation['total_pnl_realistic'] - 4096.82)/4096.82*100):+.1f}%")
    print()

    print(f"Loss Breakdown:")
    for pattern in ['sl_only', 't1_then_sl', 't2_then_sl']:
        if pattern in simulation['pattern_stats']:
            stats = simulation['pattern_stats'][pattern]
            name = pattern_names.get(pattern, pattern)
            pct_of_total_loss = (stats['loss'] / simulation['total_loss'] * 100) if simulation['total_loss'] > 0 else 0
            print(f"  {name}: Rs. {stats['loss']:,.2f} ({pct_of_total_loss:.1f}% of total loss)")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
