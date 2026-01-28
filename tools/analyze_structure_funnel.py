#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Structure Detection Funnel Analysis

Analyzes the complete funnel from structure detection to trade execution:
1. How many structures are detected?
2. Which detectors are active?
3. Why are structures being rejected?
4. Filter error analysis
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251115-080014_extracted\20251115-080014_full\20251115-080014")

def extract_detector_from_reasons(reasons):
    """Extract detector name from reasons string"""
    if 'structure:detector:' in reasons:
        for part in reasons.split(';'):
            if 'structure:detector:' in part:
                return part.split('structure:detector:')[1].strip()
    return 'unknown'

def extract_rejection_reasons(reasons):
    """Extract rejection reasons from DECISION:REJECT"""
    rejection_tags = []
    for part in reasons.split(';'):
        part = part.strip()
        if part.startswith('breakout_'):
            rejection_tags.append(part)
        elif '_fail' in part or '_error' in part:
            rejection_tags.append(part)
    return rejection_tags if rejection_tags else ['unknown_rejection']

def main():
    print("=" * 80)
    print("STRUCTURE DETECTION FUNNEL ANALYSIS")
    print("=" * 80)

    # Collect all DECISION events
    all_decisions = []
    accepted_decisions = []
    rejected_decisions = []
    errors_by_type = defaultdict(int)
    detectors_by_regime = defaultdict(lambda: defaultdict(int))
    rejections_by_detector = defaultdict(lambda: defaultdict(int))

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        agent_log = session_dir / "agent.log"
        if agent_log.exists():
            with open(agent_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'DECISION:ACCEPT' in line:
                        # Extract decision details
                        parts = line.split('|')
                        if len(parts) >= 2:
                            reasons = parts[1].strip()
                            detector = extract_detector_from_reasons(reasons)

                            # Extract setup and regime
                            setup = ''
                            regime = ''
                            if 'setup=' in parts[0]:
                                setup = parts[0].split('setup=')[1].split()[0]
                            if 'regime=' in parts[0]:
                                regime = parts[0].split('regime=')[1].split()[0]

                            accepted_decisions.append({
                                'detector': detector,
                                'setup': setup,
                                'regime': regime,
                                'reasons': reasons
                            })

                            detectors_by_regime[regime][detector] += 1

                            # Check for errors in reasons
                            if '_error' in reasons:
                                for part in reasons.split(';'):
                                    if '_error' in part:
                                        error_type = part.split(':')[0]
                                        errors_by_type[error_type] += 1

                    elif 'DECISION:REJECT' in line:
                        # Extract rejection details
                        parts = line.split('|')
                        if len(parts) >= 2:
                            reasons = parts[1].strip()
                            detector = extract_detector_from_reasons(reasons)

                            rejection_reasons = extract_rejection_reasons(reasons)

                            rejected_decisions.append({
                                'detector': detector,
                                'reasons': reasons,
                                'rejection_tags': rejection_reasons
                            })

                            for tag in rejection_reasons:
                                rejections_by_detector[detector][tag] += 1

    total_decisions = len(accepted_decisions) + len(rejected_decisions)

    print(f"\nTotal decisions found: {total_decisions}")
    print(f"  Accepted: {len(accepted_decisions)} ({len(accepted_decisions)/total_decisions*100:.1f}%)")
    print(f"  Rejected: {len(rejected_decisions)} ({len(rejected_decisions)/total_decisions*100:.1f}%)")

    # Detector activity summary
    print(f"\n" + "=" * 80)
    print("DETECTOR ACTIVITY (Accepted Decisions)")
    print("=" * 80)

    detector_counts = Counter([d['detector'] for d in accepted_decisions])
    print(f"\n{'Detector':<35} {'Decisions':>10} {'%':>8}")
    print("-" * 55)
    for detector, count in detector_counts.most_common():
        pct = (count / len(accepted_decisions) * 100) if accepted_decisions else 0
        print(f"{detector:<35} {count:>10} {pct:>7.1f}%")

    # Errors analysis
    if errors_by_type:
        print(f"\n" + "=" * 80)
        print("FILTER ERRORS DETECTED (CRITICAL BUGS)")
        print("=" * 80)

        print(f"\n{'Error Type':<50} {'Count':>10}")
        print("-" * 62)
        for error_type, count in sorted(errors_by_type.items(), key=lambda x: x[1], reverse=True):
            print(f"{error_type:<50} {count:>10} [BUG]")

    # Rejections by detector
    if rejections_by_detector:
        print(f"\n" + "=" * 80)
        print("REJECTION REASONS BY DETECTOR")
        print("=" * 80)

        for detector in sorted(rejections_by_detector.keys()):
            total_rejections = sum(rejections_by_detector[detector].values())
            print(f"\n{detector} ({total_rejections} rejections):")
            print(f"{'  Rejection Reason':<50} {'Count':>10} {'%':>8}")
            print("  " + "-" * 70)

            for reason, count in sorted(rejections_by_detector[detector].items(),
                                       key=lambda x: x[1], reverse=True):
                pct = (count / total_rejections * 100) if total_rejections > 0 else 0
                print(f"  {reason:<50} {count:>10} {pct:>7.1f}%")

    # Detector activity by regime
    print(f"\n" + "=" * 80)
    print("DETECTOR ACTIVITY BY REGIME")
    print("=" * 80)

    for regime in sorted(detectors_by_regime.keys()):
        total = sum(detectors_by_regime[regime].values())
        print(f"\n{regime.upper()} ({total} decisions):")
        print(f"{'  Detector':<35} {'Decisions':>10} {'%':>8}")
        print("  " + "-" * 55)

        for detector, count in sorted(detectors_by_regime[regime].items(),
                                      key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {detector:<35} {count:>10} {pct:>7.1f}%")

    # Critical insights
    print(f"\n" + "=" * 80)
    print("CRITICAL INSIGHTS")
    print("=" * 80)

    if errors_by_type:
        print(f"\n[CRITICAL] {sum(errors_by_type.values())} decisions have FILTER ERRORS!")
        print("These are bugs that need immediate fixing:")
        for error_type, count in sorted(errors_by_type.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count} occurrences")

    # Detector diversity
    unique_detectors = len(detector_counts)
    print(f"\nDetector diversity: {unique_detectors} unique detectors active")
    if unique_detectors < 5:
        print(f"  [WARNING] Only {unique_detectors} detectors active out of 30+ enabled!")
        print(f"  Missing detectors likely blocked by regime or quality filters")

    # Setup diversity
    setup_counts = Counter([d['setup'] for d in accepted_decisions])
    print(f"\nSetup diversity: {len(setup_counts)} unique setup types")
    for setup, count in setup_counts.most_common():
        pct = (count / len(accepted_decisions) * 100) if accepted_decisions else 0
        print(f"  - {setup}: {count} decisions ({pct:.1f}%)")

if __name__ == '__main__':
    main()
