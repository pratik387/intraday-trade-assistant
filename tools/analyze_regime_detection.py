#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regime Detection Analysis - Verify System's Real-Time Regime Detection

Analyzes screening.jsonl to see what regimes the system detected
during backtest and compares with expected regime labels.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251114-125524_extracted\20251114-125524_full\20251114-125524")

# Expected regime mapping from regime_orchestrator.py
EXPECTED_REGIMES = {
    "2023-12": "Strong_Uptrend",
    "2024-01": "Shock_Down",
    "2024-06": "Event_Driven_HighVol",
    "2024-10": "Correction_RiskOff",
    "2025-02": "Prolonged_Drawdown",
    "2025-07": "Low_Vol_Range"
}

def parse_session_date(session_name):
    """Parse session directory name to get date."""
    try:
        return datetime.strptime(session_name, "%Y-%m-%d").date()
    except:
        return None

def get_year_month(date_obj):
    """Get YYYY-MM string from date."""
    return f"{date_obj.year}-{date_obj.month:02d}"

def main():
    print("=" * 80)
    print("REGIME DETECTION ANALYSIS - ACTUAL vs EXPECTED")
    print("=" * 80)

    # Collect detected regimes from screening.jsonl
    regime_by_session = {}
    regime_counts_by_month = defaultdict(lambda: defaultdict(int))

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        session_date = parse_session_date(session_dir.name)
        if session_date is None:
            continue

        ym = get_year_month(session_date)

        # Read events.jsonl to get detected regime from first DECISION event
        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            try:
                with open(events_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            event = json.loads(line)
                            if event.get('type') == 'DECISION':
                                # Get regime from decision field
                                decision = event.get('decision', {})
                                detected_regime = decision.get('regime', 'unknown')
                                regime_by_session[session_dir.name] = detected_regime
                                regime_counts_by_month[ym][detected_regime] += 1
                                break  # Use first decision event regime
            except Exception as e:
                regime_by_session[session_dir.name] = 'error'
                regime_counts_by_month[ym]['error'] += 1
        else:
            regime_by_session[session_dir.name] = 'no_events'
            regime_counts_by_month[ym]['no_events'] += 1

    print(f"\nAnalyzed {len(regime_by_session)} sessions")

    # Monthly regime detection summary
    print(f"\n" + "=" * 80)
    print("REGIME DETECTION BY MONTH")
    print("=" * 80)

    for ym in sorted(regime_counts_by_month.keys()):
        expected_regime = EXPECTED_REGIMES.get(ym, "Unknown")
        print(f"\n{ym} (Expected: {expected_regime}):")
        print(f"{'  Detected Regime':<30} {'Sessions':>10} {'%':>8}")
        print("  " + "-" * 50)

        total_sessions = sum(regime_counts_by_month[ym].values())
        for regime, count in sorted(regime_counts_by_month[ym].items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_sessions * 100) if total_sessions > 0 else 0
            marker = " [MATCH]" if regime == expected_regime else ""
            print(f"  {regime:<30} {count:>10} {pct:>7.1f}%{marker}")

    # Regime consistency analysis
    print(f"\n" + "=" * 80)
    print("REGIME CONSISTENCY ANALYSIS")
    print("=" * 80)

    for ym in sorted(regime_counts_by_month.keys()):
        expected_regime = EXPECTED_REGIMES.get(ym, "Unknown")
        total_sessions = sum(regime_counts_by_month[ym].values())

        # Check if system consistently detected expected regime
        if expected_regime != "Unknown":
            detected_count = regime_counts_by_month[ym].get(expected_regime, 0)
            consistency_pct = (detected_count / total_sessions * 100) if total_sessions > 0 else 0

            if consistency_pct >= 80:
                status = "[EXCELLENT]"
            elif consistency_pct >= 50:
                status = "[GOOD]"
            elif consistency_pct >= 20:
                status = "[POOR]"
            else:
                status = "[FAILED]"

            print(f"{ym}: {consistency_pct:5.1f}% consistency {status}")

            # Show what was detected instead
            if consistency_pct < 100:
                other_regimes = [(r, c) for r, c in regime_counts_by_month[ym].items() if r != expected_regime]
                if other_regimes:
                    print(f"  Also detected: {', '.join([f'{r} ({c})' for r, c in other_regimes])}")

    # Detailed session-by-session for problem months
    print(f"\n" + "=" * 80)
    print("PROBLEM MONTHS - DETAILED SESSION ANALYSIS")
    print("=" * 80)

    problem_months = []
    for ym in sorted(regime_counts_by_month.keys()):
        expected_regime = EXPECTED_REGIMES.get(ym, "Unknown")
        if expected_regime != "Unknown":
            total_sessions = sum(regime_counts_by_month[ym].values())
            detected_count = regime_counts_by_month[ym].get(expected_regime, 0)
            consistency_pct = (detected_count / total_sessions * 100) if total_sessions > 0 else 0

            if consistency_pct < 80:
                problem_months.append(ym)

    if problem_months:
        for ym in problem_months:
            expected_regime = EXPECTED_REGIMES.get(ym)
            print(f"\n{ym} (Expected: {expected_regime}):")
            print(f"{'  Session':<15} {'Detected Regime':<25} {'Match':>10}")
            print("  " + "-" * 50)

            # Show each session
            for session_name in sorted(regime_by_session.keys()):
                session_date = parse_session_date(session_name)
                if session_date and get_year_month(session_date) == ym:
                    detected = regime_by_session[session_name]
                    match = "YES" if detected == expected_regime else "NO"
                    print(f"  {session_name:<15} {detected:<25} {match:>10}")
    else:
        print("\nNo problem months - all regimes detected with >80% consistency!")

    # Summary statistics
    print(f"\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    all_detected_regimes = set()
    for regimes in regime_counts_by_month.values():
        all_detected_regimes.update(regimes.keys())

    print(f"\nUnique regimes detected by system: {len(all_detected_regimes)}")
    for regime in sorted(all_detected_regimes):
        total_count = sum(counts.get(regime, 0) for counts in regime_counts_by_month.values())
        print(f"  - {regime}: {total_count} sessions")

if __name__ == '__main__':
    main()
