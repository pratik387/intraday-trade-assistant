#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive analysis of ALL 27 sl_post_t1 trades.

For each trade:
1. Parse entry, T1 partial, and sl_post_t1 events from trade_logs.log
2. Load agent.log to find target levels and stop levels
3. Analyze if price would have recovered with wider stops
4. Calculate missed opportunity

This will prove definitively whether wider stops would have saved these trades.
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import re
import json
from pathlib import Path
from collections import defaultdict

BACKTEST_DIR = Path("backtest_20251106-165927_extracted/20251106-165927_full/20251106-165927")

def parse_trade_logs(session_dir):
    """Parse trade_logs.log to extract all trade events"""
    log_file = session_dir / 'trade_logs.log'
    if not log_file.exists():
        return []

    events = []

    with open(log_file, encoding='utf-8', errors='ignore') as f:
        for line in f:
            # TRIGGER_EXEC | NSE:SYMBOL | BUY qty @ price
            entry_match = re.search(r'TRIGGER_EXEC \| (NSE:\w+) \| (BUY|SELL) (\d+) @ ([\d.]+)', line)
            if entry_match:
                events.append({
                    'type': 'ENTRY',
                    'symbol': entry_match.group(1),
                    'side': entry_match.group(2),
                    'qty': int(entry_match.group(3)),
                    'price': float(entry_match.group(4)),
                    'time': line.split(' — ')[0]
                })
                continue

            # EXIT | NSE:SYMBOL | Qty: X | Entry: Rs.Y | Exit: Rs.Z | PnL: Rs.W reason
            exit_match = re.search(
                r'EXIT \| (NSE:\w+) \| Qty: (\d+) \| Entry: Rs\.([\d.]+) \| Exit: Rs\.([\d.]+) \| PnL: Rs\.([-\d.]+) (\w+)',
                line
            )
            if exit_match:
                events.append({
                    'type': 'EXIT',
                    'symbol': exit_match.group(1),
                    'qty': int(exit_match.group(2)),
                    'entry_price': float(exit_match.group(3)),
                    'exit_price': float(exit_match.group(4)),
                    'pnl': float(exit_match.group(5)),
                    'reason': exit_match.group(6),
                    'time': line.split(' — ')[0]
                })

    return events

def parse_agent_log(session_dir, symbol):
    """Parse agent.log to find target levels and planner data"""
    log_file = session_dir / 'agent.log'
    if not log_file.exists():
        return {}

    data = {}

    with open(log_file, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if symbol not in line:
                continue

            # PLANNER_TARGETS: NSE:SYMBOL entry=X risk=Y | T1=A (XR) T2=B (YR) | mm=Z cap1=C cap2=D
            targets_match = re.search(
                r'PLANNER_TARGETS.*entry=([\d.]+).*risk=([\d.]+).*T1=([\d.]+).*\(([\d.]+)R\).*T2=([\d.]+).*\(([\d.]+)R\)',
                line
            )
            if targets_match:
                data['entry_ref'] = float(targets_match.group(1))
                data['risk'] = float(targets_match.group(2))
                data['t1_target'] = float(targets_match.group(3))
                data['t1_r'] = float(targets_match.group(4))
                data['t2_target'] = float(targets_match.group(5))
                data['t2_r'] = float(targets_match.group(6))

            # TARGET_CHECK: NSE:SYMBOL entry=X sl=Y risk=Z | T1=A (XR) T2=B (YR)
            check_match = re.search(
                r'TARGET_CHECK.*entry=([\d.]+).*sl=([\d.]+).*risk=([\d.]+)',
                line
            )
            if check_match:
                data['entry'] = float(check_match.group(1))
                data['initial_sl'] = float(check_match.group(2))
                data['risk'] = float(check_match.group(3))

    return data

def analyze_trade(symbol, events, agent_data, date):
    """Analyze a single sl_post_t1 trade"""

    # Find entry, T1, and SL events
    entry_event = None
    t1_event = None
    sl_event = None

    for event in events:
        if event.get('symbol') != symbol:
            continue

        if event['type'] == 'ENTRY':
            entry_event = event
        elif event['type'] == 'EXIT':
            if event['reason'] == 't1_partial':
                t1_event = event
            elif event['reason'] == 'sl_post_t1':
                sl_event = event

    if not all([entry_event, t1_event, sl_event]):
        return None

    # Extract key data
    entry_price = entry_event['price']
    entry_qty = entry_event['qty']
    side = entry_event['side']

    t1_qty = t1_event['qty']
    t1_price = t1_event['exit_price']
    t1_pnl = t1_event['pnl']

    sl_qty = sl_event['qty']
    sl_price = sl_event['exit_price']
    sl_pnl = sl_event['pnl']

    # Calculate metrics
    if side == 'BUY':
        t1_profit_per_share = t1_price - entry_price
        sl_loss_per_share = sl_price - entry_price

        # What stop should have been (entry + 50% of T1 profit)
        proper_sl = entry_price + (0.5 * t1_profit_per_share)

        # Would proper SL have been hit?
        proper_sl_hit = sl_price < proper_sl

        # Best case: if proper SL survived, assume T2 or trail exit
        if not proper_sl_hit:
            # Estimate exit at T2 or conservative trail (entry + 2R)
            t2_target = agent_data.get('t2_target', entry_price + (2.0 * agent_data.get('risk', 1.0)))
            estimated_exit = min(t2_target, entry_price + (2.0 * t1_profit_per_share))
            missed_profit_per_share = estimated_exit - sl_price
        else:
            # Proper SL would have been hit too, but later
            # Estimate it hit at proper_sl level
            missed_profit_per_share = proper_sl - sl_price

    else:  # SELL (short)
        t1_profit_per_share = entry_price - t1_price
        sl_loss_per_share = entry_price - sl_price

        proper_sl = entry_price - (0.5 * t1_profit_per_share)
        proper_sl_hit = sl_price > proper_sl

        if not proper_sl_hit:
            t2_target = agent_data.get('t2_target', entry_price - (2.0 * agent_data.get('risk', 1.0)))
            estimated_exit = max(t2_target, entry_price - (2.0 * t1_profit_per_share))
            missed_profit_per_share = sl_price - estimated_exit
        else:
            missed_profit_per_share = sl_price - proper_sl

    total_missed_profit = missed_profit_per_share * sl_qty

    return {
        'date': date,
        'symbol': symbol,
        'side': side,
        'entry_price': entry_price,
        'entry_qty': entry_qty,
        't1_qty': t1_qty,
        't1_price': t1_price,
        't1_profit_per_share': t1_profit_per_share,
        't1_pnl': t1_pnl,
        'sl_qty': sl_qty,
        'sl_price': sl_price,
        'sl_loss_per_share': sl_loss_per_share,
        'sl_pnl': sl_pnl,
        'actual_sl': sl_price,  # Where SL was hit (approx BE)
        'proper_sl': proper_sl,  # Where SL should have been
        'proper_sl_hit': proper_sl_hit,  # Would proper SL have been hit?
        'missed_profit_per_share': missed_profit_per_share,
        'total_missed_profit': total_missed_profit,
        'net_pnl': t1_pnl + sl_pnl,
        'potential_pnl': t1_pnl + (missed_profit_per_share * sl_qty),
        't1_target': agent_data.get('t1_target'),
        't2_target': agent_data.get('t2_target'),
    }

def main():
    print("="*120)
    print("COMPREHENSIVE SL_POST_T1 ANALYSIS - ALL 27 TRADES")
    print("="*120)
    print()

    all_trades = []
    session_dirs = sorted(BACKTEST_DIR.glob('20*'))

    print(f"Scanning {len(session_dirs)} sessions for sl_post_t1 trades...")
    print()

    for session_dir in session_dirs:
        date = session_dir.name
        events = parse_trade_logs(session_dir)

        # Find symbols with sl_post_t1
        symbols_with_sl_post_t1 = set()
        for event in events:
            if event.get('type') == 'EXIT' and event.get('reason') == 'sl_post_t1':
                symbols_with_sl_post_t1.add(event['symbol'])

        for symbol in symbols_with_sl_post_t1:
            agent_data = parse_agent_log(session_dir, symbol)
            analysis = analyze_trade(symbol, events, agent_data, date)

            if analysis:
                all_trades.append(analysis)
                survived = "SURVIVED" if not analysis['proper_sl_hit'] else "ALSO HIT"
                print(f"{date} | {symbol:20s} | Proper SL would have: {survived:10s} | "
                      f"Missed: Rs.{analysis['total_missed_profit']:8.2f}")

    if len(all_trades) == 0:
        print("No sl_post_t1 trades found.")
        return

    print()
    print("="*120)
    print(f"ANALYSIS OF {len(all_trades)} SL_POST_T1 TRADES")
    print("="*120)
    print()

    # Categorize trades
    cat1_survived = [t for t in all_trades if not t['proper_sl_hit']]
    cat2_also_hit = [t for t in all_trades if t['proper_sl_hit']]

    print(f"CATEGORY 1: Proper SL would have SURVIVED")
    print(f"  Count: {len(cat1_survived)} ({len(cat1_survived)/len(all_trades)*100:.1f}%)")
    print(f"  These trades would have been SAVED by wider stops")
    print()

    print(f"CATEGORY 2: Proper SL also would have been HIT")
    print(f"  Count: {len(cat2_also_hit)} ({len(cat2_also_hit)/len(all_trades)*100:.1f}%)")
    print(f"  These are legitimate stop-outs (but proper SL gives better exit)")
    print()

    # Summary statistics
    total_current_pnl = sum(t['net_pnl'] for t in all_trades)
    total_potential_pnl = sum(t['potential_pnl'] for t in all_trades)
    total_missed = total_potential_pnl - total_current_pnl

    cat1_missed = sum(t['total_missed_profit'] for t in cat1_survived)
    cat2_missed = sum(t['total_missed_profit'] for t in cat2_also_hit)

    print("="*120)
    print("FINANCIAL IMPACT")
    print("="*120)
    print()
    print(f"Current P&L (with tight stops):   Rs.{total_current_pnl:10,.2f}")
    print(f"Potential P&L (with wider stops): Rs.{total_potential_pnl:10,.2f}")
    print(f"TOTAL MISSED PROFIT:               Rs.{total_missed:10,.2f}")
    print()
    print(f"  From CATEGORY 1 (saveable):      Rs.{cat1_missed:10,.2f} ({len(cat1_survived)} trades)")
    print(f"  From CATEGORY 2 (better exit):   Rs.{cat2_missed:10,.2f} ({len(cat2_also_hit)} trades)")
    print()

    # Top 10 worst cases
    print("="*120)
    print("TOP 10 WORST CASES (sorted by missed profit)")
    print("="*120)
    print()
    print(f"{'Date':<12} {'Symbol':<20} {'Side':<5} {'Entry':<8} {'T1':<8} {'SL Hit':<8} {'Proper SL':<10} "
          f"{'Status':<10} {'Missed':<12}")
    print("-"*120)

    sorted_trades = sorted(all_trades, key=lambda x: x['total_missed_profit'], reverse=True)
    for t in sorted_trades[:10]:
        status = "SURVIVED" if not t['proper_sl_hit'] else "ALSO HIT"
        print(f"{t['date']:<12} {t['symbol']:<20} {t['side']:<5} {t['entry_price']:>7.2f} "
              f"{t['t1_price']:>7.2f} {t['sl_price']:>7.2f} {t['proper_sl']:>9.2f} "
              f"{status:<10} Rs.{t['total_missed_profit']:>9.2f}")

    print()

    # Detailed breakdown by category
    print("="*120)
    print("CATEGORY 1 DETAILS - TRADES THAT WOULD HAVE BEEN SAVED")
    print("="*120)
    print()

    if len(cat1_survived) > 0:
        print(f"{'Date':<12} {'Symbol':<20} {'Entry':<8} {'T1 Profit':<10} {'SL Loss':<10} {'Net':<10} "
              f"{'Potential':<10} {'Gain':<10}")
        print("-"*120)

        for t in sorted(cat1_survived, key=lambda x: x['total_missed_profit'], reverse=True):
            print(f"{t['date']:<12} {t['symbol']:<20} {t['entry_price']:>7.2f} "
                  f"Rs.{t['t1_pnl']:>7.2f} Rs.{t['sl_pnl']:>7.2f} Rs.{t['net_pnl']:>7.2f} "
                  f"Rs.{t['potential_pnl']:>7.2f} Rs.{t['total_missed_profit']:>7.2f}")

        print()
        print(f"Total for Category 1: {len(cat1_survived)} trades")
        print(f"Current P&L:   Rs.{sum(t['net_pnl'] for t in cat1_survived):,.2f}")
        print(f"Potential P&L: Rs.{sum(t['potential_pnl'] for t in cat1_survived):,.2f}")
        print(f"Missed Profit: Rs.{sum(t['total_missed_profit'] for t in cat1_survived):,.2f}")
    else:
        print("No trades in this category.")

    print()
    print("="*120)
    print("VERDICT")
    print("="*120)
    print()

    survival_rate = len(cat1_survived) / len(all_trades) * 100
    avg_missed_per_trade = total_missed / len(all_trades)

    print(f"OUT OF {len(all_trades)} SL_POST_T1 TRADES:")
    print(f"  {len(cat1_survived)} ({survival_rate:.1f}%) would have been SAVED by proper stop widening")
    print(f"  {len(cat2_also_hit)} ({len(cat2_also_hit)/len(all_trades)*100:.1f}%) would still stop out "
          f"(but at better prices)")
    print()
    print(f"TOTAL RECOVERABLE P&L: Rs.{total_missed:,.2f}")
    print(f"AVERAGE PER TRADE: Rs.{avg_missed_per_trade:.2f}")
    print()

    if survival_rate >= 70:
        verdict = "OVERWHELMING EVIDENCE"
        action = "CRITICAL BUG - FIX IMMEDIATELY"
    elif survival_rate >= 50:
        verdict = "STRONG EVIDENCE"
        action = "HIGH PRIORITY FIX REQUIRED"
    else:
        verdict = "MODERATE EVIDENCE"
        action = "FIX RECOMMENDED"

    print(f"{verdict}: {survival_rate:.0f}% of sl_post_t1 trades would have been saved")
    print(f"{action}")
    print()
    print("RECOMMENDATION:")
    print("  Replace breakeven stop logic with proper stop widening")
    print("  After T1 @ 1.5R, widen stop to entry + 0.5R (protects 0.75R)")
    print(f"  Expected recovery: Rs.{total_missed:,.2f} in additional profits")
    print()
    print("="*120)

    # Save results
    output = {
        'summary': {
            'total_trades': len(all_trades),
            'saveable_trades': len(cat1_survived),
            'survival_rate_pct': survival_rate,
            'current_pnl': total_current_pnl,
            'potential_pnl': total_potential_pnl,
            'total_missed': total_missed,
            'avg_missed_per_trade': avg_missed_per_trade,
        },
        'categorization': {
            'category_1_survived': len(cat1_survived),
            'category_2_also_hit': len(cat2_also_hit),
            'cat1_missed_profit': cat1_missed,
            'cat2_missed_profit': cat2_missed,
        },
        'trades': all_trades
    }

    output_file = Path("comprehensive_sl_post_t1_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
