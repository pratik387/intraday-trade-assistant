#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze trigger rates by setup type to understand why only 90/909 decisions triggered.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251113-115738_extracted\20251113-115738_full\20251113-115738")

def main():
    print("=" * 80)
    print("TRIGGER RATE ANALYSIS BY SETUP TYPE")
    print("=" * 80)

    # Load all decisions and triggers
    all_decisions = []
    all_triggers = []

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get('type') == 'DECISION':
                            all_decisions.append(event)
                        elif event.get('type') == 'TRIGGER':
                            all_triggers.append(event)

    print(f"\nTotal decisions: {len(all_decisions)}")
    print(f"Total triggers: {len(all_triggers)}")

    # Build decision and trigger maps by trade_id
    decision_by_id = {d.get('trade_id'): d for d in all_decisions}
    trigger_ids = set([t.get('trade_id') for t in all_triggers])

    # Analyze by setup type
    setup_stats = defaultdict(lambda: {'decisions': 0, 'triggers': 0})

    for decision in all_decisions:
        setup = decision.get('decision', {}).get('setup_type', 'unknown')
        trade_id = decision.get('trade_id')

        setup_stats[setup]['decisions'] += 1
        if trade_id in trigger_ids:
            setup_stats[setup]['triggers'] += 1

    # Display results
    print("\n" + "=" * 80)
    print("TRIGGER RATES BY SETUP TYPE")
    print("=" * 80)
    print(f"\n{'Setup Type':<25} {'Decisions':>10} {'Triggers':>10} {'Rate':>10}")
    print("-" * 80)

    for setup, stats in sorted(setup_stats.items(), key=lambda x: x[1]['decisions'], reverse=True):
        decisions = stats['decisions']
        triggers = stats['triggers']
        rate = (triggers / decisions * 100) if decisions > 0 else 0

        print(f"{setup:<25} {decisions:>10} {triggers:>10} {rate:>9.1f}%")

    # Also check by ORB level
    print("\n" + "=" * 80)
    print("TRIGGER RATES BY ORB LEVEL")
    print("=" * 80)

    orb_stats = defaultdict(lambda: {'decisions': 0, 'triggers': 0})

    for decision in all_decisions:
        reasons = decision.get('decision', {}).get('reasons', '')
        trade_id = decision.get('trade_id')

        if 'structure:level:ORH' in reasons:
            level = 'ORH'
        elif 'structure:level:ORL' in reasons:
            level = 'ORL'
        else:
            level = 'Other (PDH/PDL/etc)'

        orb_stats[level]['decisions'] += 1
        if trade_id in trigger_ids:
            orb_stats[level]['triggers'] += 1

    print(f"\n{'Level':<25} {'Decisions':>10} {'Triggers':>10} {'Rate':>10}")
    print("-" * 80)

    for level, stats in sorted(orb_stats.items(), key=lambda x: x[1]['decisions'], reverse=True):
        decisions = stats['decisions']
        triggers = stats['triggers']
        rate = (triggers / decisions * 100) if decisions > 0 else 0

        print(f"{level:<25} {decisions:>10} {triggers:>10} {rate:>9.1f}%")

    # Cross-tabulation: Setup x Level
    print("\n" + "=" * 80)
    print("TRIGGER RATES BY SETUP TYPE AND LEVEL")
    print("=" * 80)

    cross_stats = defaultdict(lambda: {'decisions': 0, 'triggers': 0})

    for decision in all_decisions:
        setup = decision.get('decision', {}).get('setup_type', 'unknown')
        reasons = decision.get('decision', {}).get('reasons', '')
        trade_id = decision.get('trade_id')

        if 'structure:level:ORH' in reasons:
            level = 'ORH'
        elif 'structure:level:ORL' in reasons:
            level = 'ORL'
        else:
            level = 'Other'

        key = f"{setup} @ {level}"
        cross_stats[key]['decisions'] += 1
        if trade_id in trigger_ids:
            cross_stats[key]['triggers'] += 1

    print(f"\n{'Setup @ Level':<35} {'Decisions':>10} {'Triggers':>10} {'Rate':>10}")
    print("-" * 80)

    for key, stats in sorted(cross_stats.items(), key=lambda x: x[1]['decisions'], reverse=True):
        decisions = stats['decisions']
        triggers = stats['triggers']
        rate = (triggers / decisions * 100) if decisions > 0 else 0

        print(f"{key:<35} {decisions:>10} {triggers:>10} {rate:>9.1f}%")

if __name__ == '__main__':
    main()
