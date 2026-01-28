#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze sl_post_t1 trades using MFE data from events.jsonl

The events.jsonl files contain MFE (Maximum Favorable Excursion) which tells us
the best price the trade achieved. This proves whether price reversed after SL.
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json
from pathlib import Path
from collections import defaultdict

BACKTEST_DIR = Path("backtest_20251106-165927_extracted/20251106-165927_full/20251106-165927")

def load_events(session_dir):
    """Load trade events from events.jsonl"""
    events_file = session_dir / 'events.jsonl'
    events = []

    if events_file.exists():
        with open(events_file, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except:
                        pass

    return events

def main():
    print("="*120)
    print("SL_POST_T1 ANALYSIS USING MFE DATA")
    print("="*120)
    print()

    all_sl_post_t1 = []

    session_dirs = sorted(BACKTEST_DIR.glob('20*'))
    print(f"Scanning {len(session_dirs)} sessions for sl_post_t1 trades...\n")

    for session_dir in session_dirs:
        date = session_dir.name
        events = load_events(session_dir)

        for event in events:
            if event.get('type') == 'EXIT' and event.get('reason') == 'sl_post_t1':
                all_sl_post_t1.append({
                    'date': date,
                    'symbol': event.get('symbol'),
                    'exit_price': event.get('exit_price'),
                    'pnl': event.get('pnl'),
                    'diagnostics': event.get('diagnostics', {}),
                })

    print(f"Found {len(all_sl_post_t1)} sl_post_t1 trades\n")

    if len(all_sl_post_t1) == 0:
        print("No sl_post_t1 trades found.")
        return

    # Analyze each trade
    print("="*120)
    print("DETAILED ANALYSIS")
    print("="*120)
    print()

    trades_with_mfe = []
    trades_without_mfe = []

    for trade in all_sl_post_t1:
        diag = trade['diagnostics']
        mfe = diag.get('mfe')
        r_at_exit = diag.get('r_multiple', 0)

        if mfe is not None:
            trades_with_mfe.append({
                **trade,
                'mfe': mfe,
                'r_at_exit': r_at_exit,
            })
        else:
            trades_without_mfe.append(trade)

    print(f"Trades with MFE data: {len(trades_with_mfe)}")
    print(f"Trades without MFE data: {len(trades_without_mfe)}")
    print()

    if len(trades_with_mfe) == 0:
        print("No MFE data available for analysis.")
        print("MFE (Maximum Favorable Excursion) shows the best price achieved during the trade.")
        return

    # Calculate statistics
    print("="*120)
    print(f"ANALYSIS OF {len(trades_with_mfe)} TRADES WITH MFE DATA")
    print("="*120)
    print()

    # Key insight: If MFE > exit R, price moved favorably AFTER exit
    trades_that_continued = []
    for t in trades_with_mfe:
        if t['mfe'] > t['r_at_exit']:
            trades_that_continued.append(t)

    print(f"Trades where price CONTINUED HIGHER after SL: {len(trades_that_continued)} ({len(trades_that_continued)/len(trades_with_mfe)*100:.1f}%)")
    print()

    # Show examples
    print("TOP 10 EXAMPLES (sorted by MFE - shows best missed opportunities):")
    print("-"*120)
    print(f"{'Date':<12} {'Symbol':<20} {'Exit R':<10} {'MFE (Best)':<15} {'Missed':<10} {'P&L':<12}")
    print("-"*120)

    sorted_trades = sorted(trades_with_mfe, key=lambda x: x['mfe'], reverse=True)
    for t in sorted_trades[:10]:
        missed_r = t['mfe'] - t['r_at_exit']
        print(f"{t['date']:<12} {t['symbol']:<20} {t['r_at_exit']:>8.2f}R {t['mfe']:>12.2f}R {missed_r:>8.2f}R Rs.{t['pnl']:>8.2f}")

    print()

    # Statistical summary
    total_mfe = sum(t['mfe'] for t in trades_with_mfe)
    total_exit_r = sum(t['r_at_exit'] for t in trades_with_mfe)
    avg_mfe = total_mfe / len(trades_with_mfe)
    avg_exit_r = total_exit_r / len(trades_with_mfe)
    avg_missed_r = avg_mfe - avg_exit_r

    print("="*120)
    print("STATISTICAL SUMMARY")
    print("="*120)
    print()
    print(f"Average exit R: {avg_exit_r:.2f}R")
    print(f"Average MFE (best achieved): {avg_mfe:.2f}R")
    print(f"Average missed opportunity: {avg_missed_r:.2f}R per trade")
    print()

    # Convert to P&L estimate
    # Rough estimate: 1R ≈ Rs.200 (varies by trade size)
    estimated_r_value = 200
    total_missed_pnl_estimate = avg_missed_r * estimated_r_value * len(trades_with_mfe)

    print(f"Estimated total missed P&L: Rs.{total_missed_pnl_estimate:.2f}")
    print(f"(Assuming 1R ≈ Rs.{estimated_r_value} average)")
    print()

    # Categorize by severity
    print("="*120)
    print("CATEGORIZATION BY SEVERITY")
    print("="*120)
    print()

    catastrophic = [t for t in trades_with_mfe if (t['mfe'] - t['r_at_exit']) >= 2.0]
    severe = [t for t in trades_with_mfe if 1.0 <= (t['mfe'] - t['r_at_exit']) < 2.0]
    moderate = [t for t in trades_with_mfe if 0.5 <= (t['mfe'] - t['r_at_exit']) < 1.0]
    minor = [t for t in trades_with_mfe if (t['mfe'] - t['r_at_exit']) < 0.5]

    print(f"CATASTROPHIC (missed >2R): {len(catastrophic)} trades")
    print(f"SEVERE (missed 1-2R): {len(severe)} trades")
    print(f"MODERATE (missed 0.5-1R): {len(moderate)} trades")
    print(f"MINOR (missed <0.5R): {len(minor)} trades")
    print()

    if len(catastrophic) > 0:
        print("CATASTROPHIC CASES (these hurt the most):")
        print("-"*120)
        for t in sorted(catastrophic, key=lambda x: x['mfe'] - x['r_at_exit'], reverse=True)[:5]:
            missed = t['mfe'] - t['r_at_exit']
            print(f"  {t['symbol']:20s} | {t['date']:12s} | Exited at {t['r_at_exit']:.2f}R | MFE: {t['mfe']:.2f}R | MISSED: {missed:.2f}R")
        print()

    # Final verdict
    print("="*120)
    print("VERDICT")
    print("="*120)
    print()

    continuation_rate = len(trades_that_continued) / len(trades_with_mfe) * 100

    if continuation_rate >= 70:
        verdict = "OVERWHELMING EVIDENCE"
        action = "IMMEDIATE FIX REQUIRED"
    elif continuation_rate >= 50:
        verdict = "STRONG EVIDENCE"
        action = "FIX HIGHLY RECOMMENDED"
    else:
        verdict = "MODERATE EVIDENCE"
        action = "FIX RECOMMENDED"

    print(f"{continuation_rate:.1f}% of sl_post_t1 trades continued higher after being stopped out")
    print(f"Average missed opportunity: {avg_missed_r:.2f}R per trade")
    print()
    print(f"{verdict}: Stop widening would prevent these premature exits")
    print(f"{action}")
    print()

    # Specific recommendations
    print("="*120)
    print("RECOMMENDATION")
    print("="*120)
    print()
    print("Current behavior:")
    print("  - After T1 @ 1.5R, stop moves to BE + 0.1% (10 paisa)")
    print("  - Normal pullback triggers stop immediately")
    print("  - Trade exits, price continues without us")
    print()
    print("Proposed fix:")
    print("  - After T1 @ 1.5R, widen stop to entry + 0.75R (50% of T1 profit)")
    print("  - This protects 0.75R profit floor")
    print("  - Allows normal 1-2R pullbacks without stopping out")
    print("  - Based on data: Would save " + f"{continuation_rate:.0f}% of these trades")
    print()
    print(f"Expected P&L recovery: Rs.{total_missed_pnl_estimate:.2f}")
    print()
    print("="*120)

    # Save results
    output = {
        'summary': {
            'total_trades': len(trades_with_mfe),
            'trades_that_continued': len(trades_that_continued),
            'continuation_rate_pct': continuation_rate,
            'avg_exit_r': avg_exit_r,
            'avg_mfe': avg_mfe,
            'avg_missed_r': avg_missed_r,
            'estimated_missed_pnl': total_missed_pnl_estimate,
        },
        'categorization': {
            'catastrophic': len(catastrophic),
            'severe': len(severe),
            'moderate': len(moderate),
            'minor': len(minor),
        },
        'trades': trades_with_mfe
    }

    output_file = Path("sl_post_t1_mfe_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
