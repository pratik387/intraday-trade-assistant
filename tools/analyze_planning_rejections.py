#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Planning Phase Rejection Analysis

Analyzes why accepted decisions don't convert to enqueued trades:
1. How many decisions get rejected in planning phase?
2. What are the rejection reasons? (structural_rr, dedup, etc)
3. Which detectors are most affected?
"""

import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251114-125524_extracted\20251114-125524_full\20251114-125524")

def extract_detector_from_reasons(reasons):
    """Extract detector name from reasons string"""
    if 'structure:detector:' in reasons:
        for part in reasons.split(';'):
            if 'structure:detector:' in part:
                return part.split('structure:detector:')[1].strip()
    return 'unknown'

def main():
    print("=" * 80)
    print("PLANNING PHASE REJECTION ANALYSIS")
    print("=" * 80)

    decisions_by_detector = defaultdict(int)
    enqueued_by_detector = defaultdict(int)
    skipped_by_detector = defaultdict(lambda: defaultdict(int))

    # Track symbols for dedup analysis
    decision_symbols = []
    enqueued_symbols = []
    skipped_symbols = []

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        agent_log = session_dir / "agent.log"
        if not agent_log.exists():
            continue

        with open(agent_log, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Track DECISION:ACCEPT events
                if 'DECISION:ACCEPT' in line:
                    match = re.search(r'sym=([^\s]+).*setup=([^\s]+)', line)
                    if match:
                        symbol = match.group(1)
                        setup = match.group(2)

                        # Extract detector from reasons
                        if '|' in line:
                            reasons = line.split('|')[1].strip()
                            detector = extract_detector_from_reasons(reasons)
                            decisions_by_detector[detector] += 1
                            decision_symbols.append((symbol, detector, setup))

                # Track ENQUEUE events (successful planning)
                elif 'ENQUEUE' in line:
                    match = re.search(r'ENQUEUE\s+([^\s]+)', line)
                    if match:
                        symbol = match.group(1)
                        enqueued_symbols.append(symbol)

                        # Match back to detector (use most recent decision for this symbol)
                        for sym, det, setup in reversed(decision_symbols):
                            if sym == symbol:
                                enqueued_by_detector[det] += 1
                                break

                # Track SKIP events (planning rejection)
                elif 'SKIP' in line and 'sym=' in line:
                    match = re.search(r'sym=([^\s]+)', line)
                    if match:
                        symbol = match.group(1)

                        # Extract rejection reason
                        rejection_reason = 'unknown'
                        if 'rejection_reason=' in line:
                            reason_match = re.search(r'rejection_reason=([^\s]+)', line)
                            if reason_match:
                                rejection_reason = reason_match.group(1)
                        elif 'reason=' in line:
                            reason_match = re.search(r'reason=([^\s]+)', line)
                            if reason_match:
                                rejection_reason = reason_match.group(1)

                        skipped_symbols.append((symbol, rejection_reason))

                        # Match back to detector
                        for sym, det, setup in reversed(decision_symbols):
                            if sym == symbol:
                                skipped_by_detector[det][rejection_reason] += 1
                                break

    total_decisions = sum(decisions_by_detector.values())
    total_enqueued = sum(enqueued_by_detector.values())
    total_skipped = sum(sum(reasons.values()) for reasons in skipped_by_detector.values())

    print(f"\nTotal decisions: {total_decisions}")
    print(f"  Enqueued: {total_enqueued} ({total_enqueued/total_decisions*100:.1f}%)")
    print(f"  Skipped in planning: {total_skipped} ({total_skipped/total_decisions*100:.1f}%)")
    print(f"  Unknown fate: {total_decisions - total_enqueued - total_skipped}")

    # Planning phase conversion rate by detector
    print(f"\n" + "=" * 80)
    print("PLANNING PHASE CONVERSION RATE BY DETECTOR")
    print("=" * 80)

    print(f"\n{'Detector':<35} {'Decisions':>10} {'Enqueued':>10} {'Skipped':>10} {'Conv%':>8}")
    print("-" * 75)

    for detector in sorted(decisions_by_detector.keys(), key=lambda d: decisions_by_detector[d], reverse=True):
        decisions = decisions_by_detector[detector]
        enqueued = enqueued_by_detector.get(detector, 0)
        skipped = sum(skipped_by_detector[detector].values())
        conv_rate = (enqueued / decisions * 100) if decisions > 0 else 0

        marker = ""
        if conv_rate == 0 and decisions >= 5:
            marker = " [PLANNING BLOCKED]"
        elif conv_rate < 20 and decisions >= 5:
            marker = " [LOW PLANNING CONV]"

        print(f"{detector:<35} {decisions:>10} {enqueued:>10} {skipped:>10} {conv_rate:>7.1f}%{marker}")

    # Rejection reasons by detector
    print(f"\n" + "=" * 80)
    print("PLANNING REJECTION REASONS BY DETECTOR")
    print("=" * 80)

    for detector in sorted(skipped_by_detector.keys(), key=lambda d: sum(skipped_by_detector[d].values()), reverse=True):
        total_skipped = sum(skipped_by_detector[detector].values())
        if total_skipped == 0:
            continue

        print(f"\n{detector} ({total_skipped} skipped):")
        print(f"{'  Rejection Reason':<50} {'Count':>10} {'%':>8}")
        print("  " + "-" * 70)

        for reason, count in sorted(skipped_by_detector[detector].items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_skipped * 100) if total_skipped > 0 else 0
            print(f"  {reason:<50} {count:>10} {pct:>7.1f}%")

    # Aggregate rejection reasons
    print(f"\n" + "=" * 80)
    print("TOP PLANNING REJECTION REASONS (ALL DETECTORS)")
    print("=" * 80)

    all_rejection_reasons = Counter()
    for detector_reasons in skipped_by_detector.values():
        for reason, count in detector_reasons.items():
            all_rejection_reasons[reason] += count

    print(f"\n{'Rejection Reason':<50} {'Count':>10} {'%':>8}")
    print("-" * 70)

    total_rejections = sum(all_rejection_reasons.values())
    for reason, count in all_rejection_reasons.most_common():
        pct = (count / total_rejections * 100) if total_rejections > 0 else 0
        print(f"{reason:<50} {count:>10} {pct:>7.1f}%")

    # Critical insights
    print(f"\n" + "=" * 80)
    print("CRITICAL INSIGHTS")
    print("=" * 80)

    # Detectors with low planning conversion
    low_conv_detectors = []
    for detector, decisions in decisions_by_detector.items():
        if decisions >= 5:
            enqueued = enqueued_by_detector.get(detector, 0)
            conv_rate = (enqueued / decisions * 100) if decisions > 0 else 0
            if conv_rate < 20:
                low_conv_detectors.append((detector, decisions, enqueued, conv_rate))

    if low_conv_detectors:
        print(f"\nDetectors with LOW planning conversion rate (<20%):")
        for detector, decisions, enqueued, conv_rate in sorted(low_conv_detectors, key=lambda x: x[3]):
            print(f"  - {detector}: {decisions} decisions → {enqueued} enqueued ({conv_rate:.1f}% conversion)")
            print(f"    Top rejection reasons:")
            for reason, count in sorted(skipped_by_detector[detector].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"      • {reason}: {count} occurrences")

    # Most common blocker
    if all_rejection_reasons:
        top_blocker, top_count = all_rejection_reasons.most_common(1)[0]
        print(f"\n[CRITICAL] Most common planning blocker: {top_blocker}")
        print(f"  Affects {top_count} decisions ({top_count/total_decisions*100:.1f}% of all decisions)")

if __name__ == '__main__':
    main()
