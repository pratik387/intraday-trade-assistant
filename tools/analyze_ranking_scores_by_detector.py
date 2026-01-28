#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ranking Score Analysis by Detector

Analyzes why only 4 detectors produce trades by examining:
1. Ranking score distribution by detector type
2. Which detectors are scoring too low to pass ranking
3. Comparison of top-ranked vs dropped detectors
"""

import sys
from pathlib import Path
from collections import defaultdict
import re

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

def main():
    print("=" * 80)
    print("RANKING SCORE ANALYSIS BY DETECTOR TYPE")
    print("=" * 80)

    # Track decisions and their ranking scores
    decisions_by_detector = defaultdict(list)
    ranked_symbols = defaultdict(list)
    enqueued_symbols = set()

    for session_dir in sorted(BACKTEST_DIR.iterdir()):
        if not session_dir.is_dir():
            continue

        agent_log = session_dir / "agent.log"
        if not agent_log.exists():
            continue

        with open(agent_log, 'r', encoding='utf-8', errors='ignore') as f:
            current_decision_detector = None
            current_decision_symbol = None

            for line in f:
                # Track DECISION:ACCEPT events
                if 'DECISION:ACCEPT' in line:
                    # Extract symbol
                    sym_match = re.search(r'sym=([^\s]+)', line)
                    if sym_match:
                        current_decision_symbol = sym_match.group(1)

                        # Extract detector from reasons
                        if '|' in line:
                            reasons = line.split('|')[1].strip()
                            current_decision_detector = extract_detector_from_reasons(reasons)

                            decisions_by_detector[current_decision_detector].append({
                                'symbol': current_decision_symbol,
                                'session': session_dir.name,
                                'rank_score': None,
                                'ranked': False,
                                'enqueued': False
                            })

                # Track ranker output (shows rank_score for symbols that made it through ranking)
                elif 'ranker: top50' in line:
                    # Parse ranked symbols with scores
                    # Format: "ranker: top50 -> NSE:SYMBOL:0.15, NSE:SYMBOL2:1.10"
                    if '->' in line:
                        ranked_part = line.split('->')[1].strip()
                        for symbol_score in ranked_part.split(','):
                            symbol_score = symbol_score.strip()
                            if ':' in symbol_score:
                                parts = symbol_score.rsplit(':', 1)
                                if len(parts) == 2:
                                    symbol = parts[0].strip()
                                    try:
                                        score = float(parts[1].strip())
                                        ranked_symbols[symbol].append(score)

                                        # Mark decision as ranked
                                        for detector, decisions in decisions_by_detector.items():
                                            for dec in decisions:
                                                if dec['symbol'] == symbol and dec['session'] == session_dir.name and dec['rank_score'] is None:
                                                    dec['rank_score'] = score
                                                    dec['ranked'] = True
                                    except ValueError:
                                        pass

                # Track ENQUEUE events
                elif 'ENQUEUE' in line:
                    enq_match = re.search(r'ENQUEUE\s+([^\s]+)', line)
                    if enq_match:
                        symbol = enq_match.group(1)
                        enqueued_symbols.add(symbol)

                        # Mark decision as enqueued
                        for detector, decisions in decisions_by_detector.items():
                            for dec in decisions:
                                if dec['symbol'] == symbol and dec['session'] == session_dir.name:
                                    dec['enqueued'] = True

    # Calculate statistics by detector
    print(f"\n{'Detector':<35} {'Decisions':>10} {'Ranked':>10} {'RankRate':>10} {'Enqueued':>10} {'Trigger%':>10}")
    print("-" * 90)

    for detector in sorted(decisions_by_detector.keys(), key=lambda d: len(decisions_by_detector[d]), reverse=True):
        decisions = decisions_by_detector[detector]
        total_decisions = len(decisions)

        ranked_count = sum(1 for d in decisions if d['ranked'])
        enqueued_count = sum(1 for d in decisions if d['enqueued'])

        rank_rate = (ranked_count / total_decisions * 100) if total_decisions > 0 else 0
        trigger_rate = (enqueued_count / total_decisions * 100) if total_decisions > 0 else 0

        marker = ""
        if ranked_count == 0 and total_decisions >= 5:
            marker = " [RANKING BLOCKED]"
        elif rank_rate < 20 and total_decisions >= 10:
            marker = " [LOW RANKING]"

        print(f"{detector:<35} {total_decisions:>10} {ranked_count:>10} {rank_rate:>9.1f}% {enqueued_count:>10} {trigger_rate:>9.1f}%{marker}")

    # Ranking score distribution by detector
    print(f"\n" + "=" * 80)
    print("RANKING SCORE DISTRIBUTION BY DETECTOR")
    print("=" * 80)

    for detector in sorted(decisions_by_detector.keys(), key=lambda d: len(decisions_by_detector[d]), reverse=True):
        decisions = decisions_by_detector[detector]
        scores = [d['rank_score'] for d in decisions if d['rank_score'] is not None]

        if scores:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            print(f"\n{detector}:")
            print(f"  Ranked decisions: {len(scores)}/{len(decisions)} ({len(scores)/len(decisions)*100:.1f}%)")
            print(f"  Score range: {min_score:.3f} to {max_score:.3f} (avg: {avg_score:.3f})")

            # Show score distribution
            low_scores = sum(1 for s in scores if s < 0.5)
            med_scores = sum(1 for s in scores if 0.5 <= s < 1.0)
            high_scores = sum(1 for s in scores if s >= 1.0)

            print(f"  Score distribution:")
            print(f"    Low (<0.5): {low_scores} ({low_scores/len(scores)*100:.1f}%)")
            print(f"    Medium (0.5-1.0): {med_scores} ({med_scores/len(scores)*100:.1f}%)")
            print(f"    High (>=1.0): {high_scores} ({high_scores/len(scores)*100:.1f}%)")
        else:
            print(f"\n{detector}:")
            print(f"  No ranked decisions (all {len(decisions)} decisions dropped by ranking!)")

    # Critical insights
    print(f"\n" + "=" * 80)
    print("CRITICAL INSIGHTS")
    print("=" * 80)

    # Detectors completely blocked by ranking
    blocked_detectors = []
    for detector, decisions in decisions_by_detector.items():
        if len(decisions) >= 5:
            ranked_count = sum(1 for d in decisions if d['ranked'])
            if ranked_count == 0:
                blocked_detectors.append((detector, len(decisions)))

    if blocked_detectors:
        print(f"\nDetectors 100% BLOCKED by ranking (never passing top50 filter):")
        for detector, count in sorted(blocked_detectors, key=lambda x: x[1], reverse=True):
            print(f"  - {detector}: {count} decisions, 0 ranked (0.0%)")
        print(f"\n[CRITICAL] {len(blocked_detectors)} detector types completely blocked by ranking!")

    # Detectors with low ranking conversion
    low_ranking_detectors = []
    for detector, decisions in decisions_by_detector.items():
        if len(decisions) >= 10:
            ranked_count = sum(1 for d in decisions if d['ranked'])
            rank_rate = (ranked_count / len(decisions) * 100) if len(decisions) > 0 else 0
            if 0 < rank_rate < 20:
                low_ranking_detectors.append((detector, len(decisions), ranked_count, rank_rate))

    if low_ranking_detectors:
        print(f"\nDetectors with LOW ranking conversion (<20%):")
        for detector, decisions, ranked, rank_rate in sorted(low_ranking_detectors, key=lambda x: x[3]):
            print(f"  - {detector}: {decisions} decisions → {ranked} ranked ({rank_rate:.1f}%)")

    # Total ranking loss
    total_decisions = sum(len(d) for d in decisions_by_detector.values())
    total_ranked = sum(sum(1 for dec in d if dec['ranked']) for d in decisions_by_detector.values())
    total_enqueued = sum(sum(1 for dec in d if dec['enqueued']) for d in decisions_by_detector.values())

    ranking_loss = total_decisions - total_ranked

    print(f"\n[CRITICAL] Ranking bottleneck summary:")
    print(f"  Total decisions: {total_decisions}")
    print(f"  Passed ranking: {total_ranked} ({total_ranked/total_decisions*100:.1f}%)")
    print(f"  Lost to ranking: {ranking_loss} ({ranking_loss/total_decisions*100:.1f}%)")
    print(f"  Finally enqueued: {total_enqueued} ({total_enqueued/total_decisions*100:.1f}%)")

if __name__ == '__main__':
    main()
