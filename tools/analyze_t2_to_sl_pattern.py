#!/usr/bin/env python3
"""
T2-to-SL Pattern Analyzer

Analyzes trades that hit T2 but then the runner got stopped out.
Simulates realistic 33-33-33 split PnL accounting for T2→SL reversals.

Usage:
    python tools/analyze_t2_to_sl_pattern.py <backtest_dir>
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
                                'exit_sequence': []  # Track order of exits
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


def analyze_t2_to_sl_pattern(trades):
    """Analyze trades that hit T2 then SL (the reversal problem)."""

    t2_hit_count = 0
    t2_then_sl_count = 0
    t2_then_t3_count = 0
    no_t2_count = 0

    t2_sl_trades = []

    for trade in trades:
        exit_sequence = trade['exit_sequence']

        # Check if T2 was hit
        has_t2 = any('t2' in exit.lower() for exit in exit_sequence)
        has_t3 = any('t3' in exit.lower() for exit in exit_sequence)
        has_sl_after_t2 = False

        if has_t2:
            t2_hit_count += 1

            # Check if SL came after T2
            t2_index = next(i for i, exit in enumerate(exit_sequence) if 't2' in exit.lower())
            sl_after_t2 = any('sl' in exit_sequence[i].lower() for i in range(t2_index + 1, len(exit_sequence)))

            if sl_after_t2:
                t2_then_sl_count += 1
                has_sl_after_t2 = True
                t2_sl_trades.append(trade)
            elif has_t3 or len(exit_sequence) > t2_index + 1:
                # Either hit T3 or got some other exit (not SL)
                t2_then_t3_count += 1
        else:
            no_t2_count += 1

    return {
        't2_hit_count': t2_hit_count,
        't2_then_sl_count': t2_then_sl_count,
        't2_then_t3_count': t2_then_t3_count,
        'no_t2_count': no_t2_count,
        't2_sl_trades': t2_sl_trades
    }


def simulate_33_33_33_realistic(trades, pattern_analysis):
    """
    Simulate 33-33-33 split with realistic T2→SL reversals.

    Scenarios:
    1. T1 hit: Exit 33%
    2. T2 hit: Exit 33%
    3. T3 hit: Exit 34% (best case)
    4. SL after T2: Exit remaining 34% at SL (reversal case)
    """

    total_pnl_perfect = 0.0  # If all T3 hit
    total_pnl_realistic = 0.0  # Accounting for T2→SL reversals
    reversal_loss = 0.0

    trades_by_pattern = {
        't2_then_t3': [],
        't2_then_sl': [],
        'no_t2': []
    }

    for trade in trades:
        entry = trade['entry_price']
        qty_total = trade['entry_qty']
        side = trade['side'].upper()
        exits = trade['exits']
        exit_sequence = trade['exit_sequence']

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

        # T1 exit
        t1_price = exit_prices.get('t1', exit_prices.get('sl', entry))
        if side == 'BUY':
            pnl_t1 = qty_t1 * (t1_price - entry)
        else:
            pnl_t1 = qty_t1 * (entry - t1_price)

        # T2 exit
        t2_price = exit_prices.get('t2', exit_prices.get('sl', entry))
        if side == 'BUY':
            pnl_t2 = qty_t2 * (t2_price - entry)
        else:
            pnl_t2 = qty_t2 * (entry - t2_price)

        # Remaining runner (34%)
        # PERFECT: Assumes T3 hit
        t3_price = exit_prices.get('t3', entry)
        if side == 'BUY':
            pnl_t3_perfect = qty_remaining * (t3_price - entry)
        else:
            pnl_t3_perfect = qty_remaining * (entry - t3_price)

        total_pnl_perfect += (pnl_t1 + pnl_t2 + pnl_t3_perfect)

        # REALISTIC: Check if T2→SL reversal
        has_t2 = any('t2' in exit.lower() for exit in exit_sequence)

        if has_t2:
            t2_index = next(i for i, exit in enumerate(exit_sequence) if 't2' in exit.lower())
            sl_after_t2 = any('sl' in exit_sequence[i].lower() for i in range(t2_index + 1, len(exit_sequence)))

            if sl_after_t2:
                # Reversal: Exit runner at SL
                sl_price = exit_prices.get('sl', entry)
                if side == 'BUY':
                    pnl_runner_realistic = qty_remaining * (sl_price - entry)
                else:
                    pnl_runner_realistic = qty_remaining * (entry - sl_price)

                total_pnl_realistic += (pnl_t1 + pnl_t2 + pnl_runner_realistic)
                reversal_loss += (pnl_t3_perfect - pnl_runner_realistic)

                trades_by_pattern['t2_then_sl'].append({
                    'symbol': trade['symbol'],
                    'pnl_perfect': pnl_t1 + pnl_t2 + pnl_t3_perfect,
                    'pnl_realistic': pnl_t1 + pnl_t2 + pnl_runner_realistic,
                    'reversal_loss': pnl_t3_perfect - pnl_runner_realistic
                })
            else:
                # No reversal: Either hit T3 or got better exit
                total_pnl_realistic += (pnl_t1 + pnl_t2 + pnl_t3_perfect)

                trades_by_pattern['t2_then_t3'].append({
                    'symbol': trade['symbol'],
                    'pnl': pnl_t1 + pnl_t2 + pnl_t3_perfect
                })
        else:
            # No T2 hit: Exit at SL or T1 only
            total_pnl_realistic += (pnl_t1 + pnl_t2 + pnl_t3_perfect)
            trades_by_pattern['no_t2'].append({
                'symbol': trade['symbol'],
                'pnl': pnl_t1 + pnl_t2 + pnl_t3_perfect
            })

    return {
        'total_pnl_perfect': total_pnl_perfect,
        'total_pnl_realistic': total_pnl_realistic,
        'reversal_loss': reversal_loss,
        'trades_by_pattern': trades_by_pattern
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_t2_to_sl_pattern.py <backtest_dir>")
        sys.exit(1)

    backtest_dir = sys.argv[1]

    print(f"Loading trades from: {backtest_dir}")
    trades = parse_trade_logs(backtest_dir)

    if not trades:
        print(f"ERROR: No trade data found")
        sys.exit(1)

    print(f"Found {len(trades)} trades\n")

    # Analyze T2→SL pattern
    pattern_analysis = analyze_t2_to_sl_pattern(trades)

    print(f"{'='*80}")
    print(f"T2 -> SL Reversal Pattern Analysis")
    print(f"{'='*80}\n")

    print(f"Total trades: {len(trades)}")
    print(f"Trades that hit T2: {pattern_analysis['t2_hit_count']} ({pattern_analysis['t2_hit_count']/len(trades)*100:.1f}%)")
    print(f"  -> Then hit T3 or better: {pattern_analysis['t2_then_t3_count']} ({pattern_analysis['t2_then_t3_count']/pattern_analysis['t2_hit_count']*100:.1f}% of T2 trades)")
    print(f"  -> Then hit SL (REVERSAL): {pattern_analysis['t2_then_sl_count']} ({pattern_analysis['t2_then_sl_count']/pattern_analysis['t2_hit_count']*100:.1f}% of T2 trades)")
    print(f"Trades that didn't hit T2: {pattern_analysis['no_t2_count']}")
    print()

    # Simulate 33-33-33 realistic
    simulation = simulate_33_33_33_realistic(trades, pattern_analysis)

    print(f"{'='*80}")
    print(f"33-33-33 Split Simulation")
    print(f"{'='*80}\n")

    print(f"PERFECT SCENARIO (all runners hit T3):")
    print(f"  Total PnL: Rs. {simulation['total_pnl_perfect']:,.2f}")
    print()

    print(f"REALISTIC SCENARIO (T2->SL reversals at actual SL price):")
    print(f"  Total PnL: Rs. {simulation['total_pnl_realistic']:,.2f}")
    print(f"  Loss from reversals: Rs. {simulation['reversal_loss']:,.2f}")
    print(f"  Reversal impact: {(simulation['reversal_loss']/simulation['total_pnl_perfect']*100):.1f}% of perfect PnL")
    print()

    print(f"COMPARISON:")
    print(f"  Perfect 33-33-33: Rs. {simulation['total_pnl_perfect']:,.2f}")
    print(f"  Realistic 33-33-33: Rs. {simulation['total_pnl_realistic']:,.2f}")
    print(f"  Difference: Rs. {(simulation['total_pnl_perfect'] - simulation['total_pnl_realistic']):,.2f}")
    print()

    # Show worst reversal trades
    t2_sl_trades = simulation['trades_by_pattern']['t2_then_sl']
    if t2_sl_trades:
        t2_sl_trades.sort(key=lambda x: x['reversal_loss'], reverse=True)

        print(f"{'='*80}")
        print(f"Top 10 Worst Reversal Trades (T2->SL)")
        print(f"{'='*80}\n")

        print(f"{'Symbol':<15} {'Perfect PnL':>12} {'Realistic PnL':>15} {'Loss':>12}")
        print(f"{'-'*15} {'-'*12} {'-'*15} {'-'*12}")

        for trade in t2_sl_trades[:10]:
            print(f"{trade['symbol']:<15} Rs. {trade['pnl_perfect']:>8,.2f} Rs. {trade['pnl_realistic']:>11,.2f} Rs. {trade['reversal_loss']:>8,.2f}")

    print(f"\n{'='*80}")
    print(f"CONCLUSION:")
    print(f"{'='*80}")

    reversal_rate = pattern_analysis['t2_then_sl_count'] / pattern_analysis['t2_hit_count'] * 100 if pattern_analysis['t2_hit_count'] > 0 else 0

    print(f"\nReversal Rate: {reversal_rate:.1f}% of T2 trades reverse to SL")
    print(f"Reversal Cost: Rs. {simulation['reversal_loss']:,.2f} ({(simulation['reversal_loss']/simulation['total_pnl_perfect']*100):.1f}% of potential)")
    print(f"\nRealistic 33-33-33 PnL: Rs. {simulation['total_pnl_realistic']:,.2f}")
    print(f"Still {((simulation['total_pnl_realistic'] - 4096.82) / 4096.82 * 100):+.1f}% better than actual Rs. 4,096.82")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
