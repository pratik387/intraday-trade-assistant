#!/usr/bin/env python
"""
Analyze why 0% of trades hit T3.

Investigate:
1. Are T3 targets set too aggressively?
2. Do trades hit T2 but can't reach T3?
3. How close do trades get to T3 before reversing?
4. What's the MFE (Maximum Favorable Excursion) vs T3 distance?
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def analyze_t3_potential(backtest_dir):
    """Analyze how close trades get to T3 and why they don't hit it."""

    backtest_path = Path(backtest_dir)

    # Parse events.jsonl to get target levels
    events_files = list(backtest_path.rglob("events.jsonl"))

    # Parse analytics.jsonl to get MFE data
    analytics_files = list(backtest_path.rglob("analytics.jsonl"))

    results = {
        "total_trades": 0,
        "reached_t1": 0,
        "reached_t2": 0,
        "reached_t3": 0,
        "t2_to_t3_distance": [],
        "mfe_vs_t3": [],
        "exit_reasons_after_t2": defaultdict(int)
    }

    # Read analytics data
    for analytics_file in analytics_files:
        with open(analytics_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                trade = json.loads(line)
                results["total_trades"] += 1

                # Get target levels if available
                targets = trade.get('targets', {})
                t1 = targets.get('t1')
                t2 = targets.get('t2')
                t3 = targets.get('t3')

                entry = trade.get('entry_price')
                mfe = trade.get('mfe')  # Maximum Favorable Excursion

                exits = trade.get('exits', [])

                # Check what targets were hit
                has_t1 = any('t1_partial' in e.get('reason', '') for e in exits)
                has_t2 = any('t2_partial' in e.get('reason', '') for e in exits)
                has_t3 = any('target_t3' in e.get('reason', '') for e in exits)

                if has_t1:
                    results["reached_t1"] += 1
                if has_t2:
                    results["reached_t2"] += 1
                if has_t3:
                    results["reached_t3"] += 1

                # For trades that hit T2 but not T3, analyze why
                if has_t2 and not has_t3 and t2 and t3 and entry:
                    # Calculate distance from T2 to T3
                    side = trade.get('side', 'BUY')

                    if side == 'BUY':
                        # Long trade
                        t2_to_t3_pct = ((t3 - t2) / entry) * 100
                    else:
                        # Short trade
                        t2_to_t3_pct = ((t2 - t3) / entry) * 100

                    results["t2_to_t3_distance"].append(t2_to_t3_pct)

                    # If we have MFE, check if trade got close to T3
                    if mfe and mfe > 0:
                        if side == 'BUY':
                            mfe_vs_t3_pct = ((mfe - entry) / (t3 - entry)) * 100
                        else:
                            mfe_vs_t3_pct = ((entry - mfe) / (entry - t3)) * 100

                        results["mfe_vs_t3"].append({
                            'symbol': trade.get('symbol'),
                            'mfe_pct_of_t3': mfe_vs_t3_pct,
                            'reached_t3': mfe_vs_t3_pct >= 100,
                            'exits': [e.get('reason') for e in exits]
                        })

                    # Track exit reason after T2
                    for e in exits:
                        if e.get('reason') not in ['t1_partial', 't2_partial']:
                            results["exit_reasons_after_t2"][e.get('reason')] += 1

    return results

def print_analysis(results):
    """Print comprehensive T3 blocker analysis."""

    print("="*80)
    print("T3 BLOCKER ANALYSIS - Why 0% of Trades Hit T3")
    print("="*80)

    total = results["total_trades"]
    print(f"\nTotal trades: {total}")
    print(f"Reached T1:   {results['reached_t1']} ({results['reached_t1']/total*100:.1f}%)")
    print(f"Reached T2:   {results['reached_t2']} ({results['reached_t2']/total*100:.1f}%)")
    print(f"Reached T3:   {results['reached_t3']} ({results['reached_t3']/total*100:.1f}%)")

    print("\n" + "="*80)
    print("T2 to T3 Distance Analysis")
    print("="*80)

    if results["t2_to_t3_distance"]:
        avg_distance = sum(results["t2_to_t3_distance"]) / len(results["t2_to_t3_distance"])
        min_distance = min(results["t2_to_t3_distance"])
        max_distance = max(results["t2_to_t3_distance"])

        print(f"\nTrades that reached T2: {len(results['t2_to_t3_distance'])}")
        print(f"Average T2→T3 distance: {avg_distance:.2f}% of entry price")
        print(f"Min T2→T3 distance:     {min_distance:.2f}%")
        print(f"Max T2→T3 distance:     {max_distance:.2f}%")

        if avg_distance > 3.0:
            print(f"\n⚠ WARNING: T3 targets are {avg_distance:.1f}% beyond T2")
            print("   This is aggressive - most intraday moves don't have this much room")
    else:
        print("\nNo trades reached T2 - cannot analyze T2→T3 distance")

    print("\n" + "="*80)
    print("MFE (Maximum Favorable Excursion) vs T3")
    print("="*80)

    if results["mfe_vs_t3"]:
        print(f"\nTrades with MFE data: {len(results['mfe_vs_t3'])}")

        # Count how many got close to T3
        reached_90pct = sum(1 for t in results["mfe_vs_t3"] if t['mfe_pct_of_t3'] >= 90)
        reached_75pct = sum(1 for t in results["mfe_vs_t3"] if t['mfe_pct_of_t3'] >= 75)
        reached_50pct = sum(1 for t in results["mfe_vs_t3"] if t['mfe_pct_of_t3'] >= 50)
        actually_hit = sum(1 for t in results["mfe_vs_t3"] if t['reached_t3'])

        print(f"  Got to 90%+ of T3: {reached_90pct} ({reached_90pct/len(results['mfe_vs_t3'])*100:.1f}%)")
        print(f"  Got to 75%+ of T3: {reached_75pct} ({reached_75pct/len(results['mfe_vs_t3'])*100:.1f}%)")
        print(f"  Got to 50%+ of T3: {reached_50pct} ({reached_50pct/len(results['mfe_vs_t3'])*100:.1f}%)")
        print(f"  Actually hit T3:   {actually_hit} ({actually_hit/len(results['mfe_vs_t3'])*100:.1f}%)")

        # Show trades that got close but missed
        almost_made_it = [t for t in results["mfe_vs_t3"] if 85 <= t['mfe_pct_of_t3'] < 100]
        if almost_made_it:
            print(f"\n  {len(almost_made_it)} trades got 85-99% to T3 but reversed:")
            for t in almost_made_it[:5]:
                print(f"    {t['symbol']}: {t['mfe_pct_of_t3']:.1f}% of T3, exits: {t['exits']}")

    print("\n" + "="*80)
    print("Exit Reasons After T2")
    print("="*80)

    if results["exit_reasons_after_t2"]:
        print(f"\nFor {sum(results['exit_reasons_after_t2'].values())} exits after T2:")
        for reason, count in sorted(results["exit_reasons_after_t2"].items(), key=lambda x: -x[1]):
            print(f"  {reason:<25} {count:3d} exits")

    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)

    # Diagnose the main problem
    if results["reached_t2"] == 0:
        print("\n❌ PROBLEM: No trades reach T2 at all")
        print("   → T1 targets might be too aggressive")
        print("   → OR trades reverse immediately after entry")
        print("   → Need to analyze T1 distance and entry quality")

    elif results["reached_t2"] < total * 0.2:
        pct = results["reached_t2"] / total * 100
        print(f"\n❌ PROBLEM: Only {pct:.1f}% of trades reach T2")
        print("   → Most trades exit before T2")
        print("   → T2 targets are too aggressive for this trade population")
        print("   → Consider: Are we taking low-momentum setups?")

    else:
        pct = results["reached_t2"] / total * 100
        print(f"\n⚠ {pct:.1f}% of trades reach T2, but 0% reach T3")

        if results["t2_to_t3_distance"]:
            avg_dist = sum(results["t2_to_t3_distance"]) / len(results["t2_to_t3_distance"])
            if avg_dist > 3.0:
                print(f"   → T3 is {avg_dist:.1f}% beyond T2 (too aggressive)")
                print("   → Intraday moves rarely have this much continuation")
                print(f"   → RECOMMENDATION: Reduce T3 distance to 2-2.5% beyond T2")
            else:
                print(f"   → T3 distance ({avg_dist:.1f}%) seems reasonable")
                print("   → Problem is likely trail stop management after T2")
                print("   → Trail stop may be too tight, choking off runners")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\n1. **Target Calibration**:")
    print("   - Run analysis to check if T1/T2/T3 distances match typical intraday R:R")
    print("   - Compare target distances to average intraday ranges")

    print("\n2. **Trail Stop Management**:")
    print("   - After T2, trail stop should be wider to let runners breathe")
    print("   - Consider: Don't move to BE+buffer, stay at T1 level")

    print("\n3. **Trade Selection**:")
    print("   - System may not be identifying true momentum breakouts")
    print("   - Add filters for: volume expansion, momentum, institutional interest")

    print("\n" + "="*80)

def main():
    before_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251110-132748_extracted\20251110-132748_full\20251110-132748"

    print("Analyzing T3 blockers in BEFORE (40-40-20) backtest...")
    print("(Same pattern appears in AFTER backtest - 0% T3 hits in both)\n")

    results = analyze_t3_potential(before_dir)
    print_analysis(results)

if __name__ == '__main__':
    main()
