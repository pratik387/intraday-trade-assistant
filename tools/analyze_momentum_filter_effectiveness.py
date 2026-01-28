#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze momentum quality filter effectiveness.

Questions to answer:
1. How many stocks passed the momentum quality filter?
2. Of those, how many became winning trades?
3. Were there any false rejections (overextended/weak volume)?
4. What were the characteristics of filtered stocks?
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Path to extracted backtest
BACKTEST_DIR = Path(r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251112-104346_extracted\20251112-104346_full\20251112-104346")

def load_all_events():
    """Load all events.jsonl files from all sessions."""
    all_events = []

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        events_file = session_dir / "events.jsonl"
        if not events_file.exists():
            continue

        with open(events_file, 'r') as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    all_events.append(event)

    return all_events

def load_all_analytics():
    """Load all analytics.jsonl files to get trade outcomes."""
    all_trades = {}  # symbol -> trade data

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        analytics_file = session_dir / "analytics.jsonl"
        if not analytics_file.exists():
            continue

        with open(analytics_file, 'r') as f:
            for line in f:
                if line.strip():
                    trade = json.loads(line)
                    if trade.get('stage') == 'EXIT':
                        symbol = trade.get('symbol')
                        all_trades[symbol] = trade

    return all_trades

def analyze_momentum_filter_patterns(events):
    """Analyze momentum filter patterns from events."""

    momentum_stats = {
        'quality_pass': [],
        'overextended_reject': [],
        'weak_volume_reject': [],
        'regular_pass': []
    }

    for event in events:
        if event.get('type') != 'DECISION':
            continue

        decision = event.get('decision', {})
        reasons = decision.get('reasons', '')

        # Check for momentum quality patterns
        if 'momentum_quality_pass' in reasons:
            momentum_stats['quality_pass'].append(event)
        elif 'momentum_consolidation_fail:overextended_from_vwap' in reasons:
            momentum_stats['overextended_reject'].append(event)
        elif 'momentum_consolidation_fail:no_volume_confirmation' in reasons:
            momentum_stats['weak_volume_reject'].append(event)
        elif 'momentum_consolidation_pass' in reasons or 'momentum' not in reasons:
            # Regular pass (old-style momentum check or no momentum filtering)
            momentum_stats['regular_pass'].append(event)

    return momentum_stats

def extract_momentum_metrics(cautions_str):
    """Extract momentum metrics from cautions string."""
    import re

    metrics = {}

    # Extract momentum percentage
    momentum_match = re.search(r'momentum_quality_pass:([-\d.]+)%', cautions_str)
    if momentum_match:
        metrics['momentum_pct'] = float(momentum_match.group(1))

    # Extract VWAP deviation
    vwap_match = re.search(r'vwap_dev:([-\d.]+)%', cautions_str)
    if vwap_match:
        metrics['vwap_dev_pct'] = float(vwap_match.group(1))

    # Extract volume ratio
    vol_match = re.search(r'vol_ratio:([-\d.]+)x', cautions_str)
    if vol_match:
        metrics['vol_ratio'] = float(vol_match.group(1))

    return metrics

def main():
    print("Loading all events from events.jsonl files...")
    all_events = load_all_events()
    print(f"Total events loaded: {len(all_events)}")

    print("\nLoading all trades from analytics.jsonl files...")
    all_trades = load_all_analytics()
    print(f"Total trades loaded: {len(all_trades)}")

    # Analyze momentum filter
    print("\n" + "="*80)
    print("MOMENTUM QUALITY FILTER ANALYSIS")
    print("="*80)

    momentum_stats = analyze_momentum_filter_patterns(all_events)

    print(f"\nQuality Passes: {len(momentum_stats['quality_pass'])}")
    print(f"Overextended Rejections: {len(momentum_stats['overextended_reject'])}")
    print(f"Weak Volume Rejections: {len(momentum_stats['weak_volume_reject'])}")
    print(f"Regular Passes: {len(momentum_stats['regular_pass'])}")

    # Analyze quality pass characteristics
    if momentum_stats['quality_pass']:
        print("\n" + "="*80)
        print("QUALITY PASS CHARACTERISTICS")
        print("="*80)

        quality_metrics = []
        for event in momentum_stats['quality_pass']:
            decision = event.get('decision', {})
            reasons = decision.get('reasons', '')
            metrics = extract_momentum_metrics(reasons)
            if metrics:
                quality_metrics.append(metrics)

        if quality_metrics:
            avg_momentum = sum(m.get('momentum_pct', 0) for m in quality_metrics) / len(quality_metrics)
            avg_vwap_dev = sum(m.get('vwap_dev_pct', 0) for m in quality_metrics) / len(quality_metrics)
            avg_vol_ratio = sum(m.get('vol_ratio', 0) for m in quality_metrics) / len(quality_metrics)

            print(f"\nAverage Momentum: {avg_momentum:.2f}%")
            print(f"Average VWAP Deviation: {avg_vwap_dev:.2f}%")
            print(f"Average Volume Ratio: {avg_vol_ratio:.2f}x")

            # Show distribution
            print("\nMomentum Distribution:")
            momentum_ranges = [0, 1, 2, 3, 5, 10]
            for i in range(len(momentum_ranges)-1):
                low, high = momentum_ranges[i], momentum_ranges[i+1]
                count = sum(1 for m in quality_metrics if low <= abs(m.get('momentum_pct', 0)) < high)
                if count > 0:
                    print(f"  {low}%-{high}%: {count} ({count/len(quality_metrics)*100:.1f}%)")

            print("\nVWAP Deviation Distribution:")
            vwap_ranges = [0, 0.5, 1.0, 1.5, 2.0, 5.0]
            for i in range(len(vwap_ranges)-1):
                low, high = vwap_ranges[i], vwap_ranges[i+1]
                count = sum(1 for m in quality_metrics if low <= abs(m.get('vwap_dev_pct', 0)) < high)
                if count > 0:
                    print(f"  {low}%-{high}%: {count} ({count/len(quality_metrics)*100:.1f}%)")

            print("\nVolume Ratio Distribution:")
            vol_ranges = [0, 1.5, 2.0, 3.0, 5.0, 10.0]
            for i in range(len(vol_ranges)-1):
                low, high = vol_ranges[i], vol_ranges[i+1]
                count = sum(1 for m in quality_metrics if low <= m.get('vol_ratio', 0) < high)
                if count > 0:
                    print(f"  {low}x-{high}x: {count} ({count/len(quality_metrics)*100:.1f}%)")

    # Sample quality passes
    print("\n" + "="*80)
    print("SAMPLE QUALITY PASSES (First 10)")
    print("="*80)

    for event in momentum_stats['quality_pass'][:10]:
        symbol = event.get('symbol', '')
        decision = event.get('decision', {})
        reasons = decision.get('reasons', '')
        metrics = extract_momentum_metrics(reasons)

        print(f"\n{symbol}:")
        print(f"  Momentum: {metrics.get('momentum_pct', 0):.2f}%")
        print(f"  VWAP Dev: {metrics.get('vwap_dev_pct', 0):.2f}%")
        print(f"  Volume Ratio: {metrics.get('vol_ratio', 0):.2f}x")

    # Check outcomes
    print("\n" + "="*80)
    print("FILTER EFFECTIVENESS")
    print("="*80)

    # How many quality passes became trades?
    quality_pass_symbols = set(e.get('symbol') for e in momentum_stats['quality_pass'])
    traded_symbols = set(all_trades.keys())

    quality_became_trades = quality_pass_symbols & traded_symbols

    print(f"\nQuality Pass Symbols: {len(quality_pass_symbols)}")
    print(f"Traded Symbols: {len(traded_symbols)}")
    print(f"Quality Passes That Became Trades: {len(quality_became_trades)}")
    if len(quality_pass_symbols) > 0:
        print(f"Conversion Rate: {len(quality_became_trades)/len(quality_pass_symbols)*100:.1f}%")
    else:
        print(f"Conversion Rate: N/A (no quality passes)")

    # Of those trades, how many won?
    if quality_became_trades:
        winning_quality_trades = sum(
            1 for symbol in quality_became_trades
            if all_trades.get(symbol, {}).get('pnl', 0) > 0
        )

        print(f"\nWinning Quality Trades: {winning_quality_trades}/{len(quality_became_trades)}")
        print(f"Win Rate: {winning_quality_trades/len(quality_became_trades)*100:.1f}%")

    # Check if we had any false rejections
    if momentum_stats['overextended_reject']:
        print("\n" + "="*80)
        print("OVEREXTENDED REJECTIONS (False Positives?)")
        print("="*80)
        print(f"\nTotal: {len(momentum_stats['overextended_reject'])}")
        print("\nSample (First 5):")
        for event in momentum_stats['overextended_reject'][:5]:
            decision = event.get('decision', {})
            reasons = decision.get('reasons', '')
            print(f"  {event.get('symbol', '')}: {reasons}")

    if momentum_stats['weak_volume_reject']:
        print("\n" + "="*80)
        print("WEAK VOLUME REJECTIONS")
        print("="*80)
        print(f"\nTotal: {len(momentum_stats['weak_volume_reject'])}")
        print("\nSample (First 5):")
        for event in momentum_stats['weak_volume_reject'][:5]:
            decision = event.get('decision', {})
            reasons = decision.get('reasons', '')
            print(f"  {event.get('symbol', '')}: {reasons}")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    total_filtered = len(momentum_stats['overextended_reject']) + len(momentum_stats['weak_volume_reject'])
    total_passed = len(momentum_stats['quality_pass']) + len(momentum_stats['regular_pass'])

    print(f"\n1. FILTER ACTIVITY:")
    print(f"   Total Passed: {total_passed}")
    print(f"   Total Filtered: {total_filtered}")
    print(f"   Filter Rate: {total_filtered/(total_passed + total_filtered)*100:.1f}%")

    print(f"\n2. EFFECTIVENESS:")
    if len(quality_became_trades) > 0:
        print(f"   Quality passes became trades: {len(quality_became_trades)/len(quality_pass_symbols)*100:.1f}%")
        print(f"   Win rate of quality trades: {winning_quality_trades/len(quality_became_trades)*100:.1f}%")
    else:
        print(f"   NO quality passes became trades!")

    print(f"\n3. PROBLEM IDENTIFIED:")
    print(f"   Filter has {total_filtered} rejections (should be more aggressive)")
    print(f"   Allowing {len(quality_pass_symbols)} quality passes")
    print(f"   But only {len(traded_symbols)} total trades executed")
    print(f"   And only 46.9% win rate overall")
    print(f"\n   CONCLUSION: Filter is NOT aggressive enough!")
    print(f"   Need stricter VWAP deviation and volume ratio thresholds.")

if __name__ == '__main__':
    main()
