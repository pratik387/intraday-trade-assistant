#!/usr/bin/env python
"""
Calculate optimal exit split based on actual trade patterns.

Given that 0% of trades hit T3, analyze what split maximizes PnL:
- 50-50 (T1=50%, T2=50%)
- 60-40 (T1=60%, T2=40%)
- 40-60 (T1=40%, T2=60%)
- 40-40-20 (current)
- 33-33-33 (tested, failed)
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def parse_trade_details(backtest_dir):
    """Parse trade logs to get entry/exit prices and quantities."""

    backtest_path = Path(backtest_dir)
    log_files = list(backtest_path.rglob("trade_logs.log"))

    trades = []

    for log_file in log_files:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        current_trade = {}

        for line in lines:
            if "TRIGGER_EXEC" in line:
                # New trade
                match = re.search(r'\| (NSE:\w+) \| \w+ (\d+) @ ([\d\.]+)', line)
                if match:
                    if current_trade:
                        trades.append(current_trade)

                    current_trade = {
                        'symbol': match.group(1),
                        'qty': int(match.group(2)),
                        'entry': float(match.group(3)),
                        'exits': []
                    }

            elif "EXIT" in line and current_trade:
                # Exit
                match = re.search(r'Qty: (\d+) \| Entry: Rs\.([\d\.]+) \| Exit: Rs\.([\d\.]+) \| PnL: Rs\.([\d\.\-]+) (\w+)', line)
                if match:
                    current_trade['exits'].append({
                        'qty': int(match.group(1)),
                        'entry': float(match.group(2)),
                        'exit': float(match.group(3)),
                        'pnl': float(match.group(4)),
                        'reason': match.group(5)
                    })

        if current_trade:
            trades.append(current_trade)

    return trades

def calculate_pnl_for_split(trades, t1_pct, t2_pct):
    """Calculate what PnL would be if we used a different split."""

    total_pnl = 0

    for trade in trades:
        exits = trade['exits']
        if not exits:
            continue

        original_qty = trade['qty']
        entry_price = trade['entry']

        # Categorize the trade
        has_t1 = any('t1_partial' in e['reason'] for e in exits)
        has_t2 = any('t2_partial' in e['reason'] for e in exits)
        has_hard_sl = any('hard_sl' in e['reason'] for e in exits)

        if has_hard_sl and not has_t1:
            # Hit hard SL before T1 - no change
            total_pnl += sum(e['pnl'] for e in exits)

        elif not has_t1:
            # No targets, just EOD - no change
            total_pnl += sum(e['pnl'] for e in exits)

        elif has_t1 and not has_t2:
            # Hit T1 only, then reversed or EOD
            t1_exit = next(e for e in exits if 't1_partial' in e['reason'])
            remaining_exits = [e for e in exits if 't1_partial' not in e['reason']]

            # Recalculate T1 exit with new percentage
            new_t1_qty = int(max(1, round(original_qty * (t1_pct / 100.0))))
            new_t1_pnl = new_t1_qty * (t1_exit['exit'] - entry_price)

            # Remaining exits use original exit price but adjusted quantity
            remaining_qty = original_qty - new_t1_qty
            remaining_pnl = sum(e['pnl'] for e in remaining_exits)

            # Scale remaining PnL by new quantity ratio
            original_remaining_qty = sum(e['qty'] for e in remaining_exits)
            if original_remaining_qty > 0:
                remaining_pnl = remaining_pnl * (remaining_qty / original_remaining_qty)

            total_pnl += new_t1_pnl + remaining_pnl

        elif has_t1 and has_t2:
            # Hit T1 and T2
            t1_exit = next(e for e in exits if 't1_partial' in e['reason'])
            t2_exit = next(e for e in exits if 't2_partial' in e['reason'])
            remaining_exits = [e for e in exits if 't1_partial' not in e['reason'] and 't2_partial' not in e['reason']]

            # Recalculate T1 and T2 with new percentages
            new_t1_qty = int(max(1, round(original_qty * (t1_pct / 100.0))))
            new_t1_pnl = new_t1_qty * (t1_exit['exit'] - entry_price)

            after_t1 = original_qty - new_t1_qty
            new_t2_qty = int(max(1, round(after_t1 * (t2_pct / 100.0))))
            new_t2_pnl = new_t2_qty * (t2_exit['exit'] - entry_price)

            # Remaining after T2
            remaining_qty = after_t1 - new_t2_qty
            remaining_pnl = sum(e['pnl'] for e in remaining_exits)

            # Scale remaining PnL
            original_remaining_qty = sum(e['qty'] for e in remaining_exits)
            if original_remaining_qty > 0:
                remaining_pnl = remaining_pnl * (remaining_qty / original_remaining_qty)

            total_pnl += new_t1_pnl + new_t2_pnl + remaining_pnl

    return total_pnl

def main():
    before_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251110-132748_extracted\20251110-132748_full\20251110-132748"

    print("Parsing trade details from BEFORE (40-40-20) backtest...")
    trades = parse_trade_details(before_dir)
    print(f"Parsed {len(trades)} trades\n")

    # Test different splits
    splits = [
        ("50-50-0", 50, 50),
        ("60-40-0", 60, 40),
        ("40-60-0", 40, 60),
        ("70-30-0", 70, 30),
        ("30-70-0", 30, 70),
        ("40-40-20", 40, 40),  # Current
        ("33-33-33", 33, 33),  # Tested
    ]

    print("="*80)
    print("OPTIMAL SPLIT ANALYSIS")
    print("="*80)
    print("Given that 0% of trades hit T3, testing 2-target splits:\n")

    results = []
    for name, t1_pct, t2_pct in splits:
        pnl = calculate_pnl_for_split(trades, t1_pct, t2_pct)
        results.append((name, pnl, t1_pct, t2_pct))

    # Sort by PnL
    results.sort(key=lambda x: x[1], reverse=True)

    baseline_pnl = 8550.78  # Actual 40-40-20 result

    print(f"{'Split':<12} {'Est. PnL':>12} {'vs Baseline':>12} {'T1%':>6} {'T2%':>6} {'T3%':>6}")
    print("-"*80)

    for name, pnl, t1_pct, t2_pct in results:
        diff = pnl - baseline_pnl
        diff_pct = (diff / baseline_pnl * 100) if baseline_pnl != 0 else 0
        t3_pct = 100 - t1_pct - t2_pct

        marker = " *BEST*" if pnl == max(r[1] for r in results) else ""
        marker = " (CURRENT)" if name == "40-40-20" else marker

        print(f"{name:<12} Rs.{pnl:>9,.2f}  {diff:>+9,.2f} ({diff_pct:>+5.1f}%)  {t1_pct:>4}%  {t2_pct:>4}%  {t3_pct:>4}%{marker}")

    print("\n" + "="*80)
    print("FINDINGS")
    print("="*80)

    best = results[0]
    print(f"\nOptimal split: {best[0]}")
    print(f"Expected PnL: Rs. {best[1]:,.2f}")
    print(f"Improvement over 40-40-20: Rs. {best[1] - baseline_pnl:+,.2f} ({(best[1] - baseline_pnl)/baseline_pnl*100:+.1f}%)")

    print("\nKEY INSIGHTS:")
    print("- Since 0% of trades hit T3, T3 allocation is wasted capital")
    print("- Higher T1% captures profit before BE stop on reversals")
    print("- Need to balance T1 profit taking vs letting winners run to T2")
    print("- EOD exits (50% of trades) benefit from having more shares in play")

if __name__ == '__main__':
    main()
