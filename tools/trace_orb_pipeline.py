#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trace ORB decisions through the entire pipeline to find where they're being blocked.

Pipeline stages:
1. Scanning (ORB priority candidates identified)
2. Structure Detection (ORB decisions made)
3. Trigger Logic (decisions → triggers)
4. Execution (triggers → trades)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Path to extracted backtest
BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251113-115738_extracted\20251113-115738_full\20251113-115738")

def is_orb_decision(event):
    """Check if decision is ORB-related."""
    decision = event.get('decision', {})
    reasons = decision.get('reasons', '')
    return 'structure:level:ORH' in reasons or 'structure:level:ORL' in reasons

def main():
    print("=" * 80)
    print("ORB PIPELINE TRACE - Finding Where ORB Decisions Go")
    print("=" * 80)

    # Load all events from all sessions
    all_scanning = []
    all_decisions = []
    all_triggers = []
    all_analytics = []

    print("\nLoading events from all sessions...")
    session_count = 0
    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        session_count += 1

        # Load scanning
        scanning_file = session_dir / "scanning.jsonl"
        if scanning_file.exists():
            with open(scanning_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        event['session'] = session_dir.name
                        all_scanning.append(event)

        # Load events
        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        event['session'] = session_dir.name
                        if event.get('type') == 'DECISION':
                            all_decisions.append(event)
                        elif event.get('type') == 'TRIGGER':
                            all_triggers.append(event)

        # Load analytics
        analytics_file = session_dir / "analytics.jsonl"
        if analytics_file.exists():
            with open(analytics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        trade = json.loads(line)
                        trade['session'] = session_dir.name
                        all_analytics.append(trade)

    print(f"Loaded data from {session_count} sessions")

    # Stage 1: Scanning
    print("\n" + "=" * 80)
    print("STAGE 1: SCANNING")
    print("=" * 80)

    orb_scans = [s for s in all_scanning if s.get('category') == 'orb_priority']
    orb_symbols = set([s['symbol'] for s in orb_scans])

    print(f"\nTotal scans: {len(all_scanning)}")
    print(f"ORB priority scans: {len(orb_scans)} ({len(orb_scans)/len(all_scanning)*100:.1f}%)")
    print(f"Unique ORB symbols: {len(orb_symbols)}")

    # Stage 2: Decisions
    print("\n" + "=" * 80)
    print("STAGE 2: DECISIONS")
    print("=" * 80)

    orb_decisions = [d for d in all_decisions if is_orb_decision(d)]
    non_orb_decisions = [d for d in all_decisions if not is_orb_decision(d)]

    print(f"\nTotal decisions: {len(all_decisions)}")
    print(f"ORB decisions: {len(orb_decisions)} ({len(orb_decisions)/len(all_decisions)*100:.1f}%)")
    print(f"Non-ORB decisions: {len(non_orb_decisions)} ({len(non_orb_decisions)/len(all_decisions)*100:.1f}%)")

    # Check decision eligibility
    orb_eligible = [d for d in orb_decisions if d.get('plan', {}).get('eligible', False)]
    orb_ineligible = [d for d in orb_decisions if not d.get('plan', {}).get('eligible', False)]

    print(f"\nORB Decision Eligibility:")
    print(f"  Eligible: {len(orb_eligible)}")
    print(f"  Ineligible: {len(orb_ineligible)}")

    if orb_ineligible:
        print(f"\nSample ineligible reasons:")
        rejection_reasons = defaultdict(int)
        for d in orb_ineligible:
            reason = d.get('plan', {}).get('quality', {}).get('rejection_reason', 'Unknown')
            rejection_reasons[reason] += 1
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {reason}: {count}")

    # Stage 3: Triggers
    print("\n" + "=" * 80)
    print("STAGE 3: TRIGGERS")
    print("=" * 80)

    print(f"\nTotal triggers: {len(all_triggers)}")

    # Match triggers to decisions by trade_id
    decision_by_trade_id = {d.get('trade_id'): d for d in all_decisions}
    trigger_by_trade_id = {t.get('trade_id'): t for t in all_triggers}

    orb_decision_trade_ids = set([d.get('trade_id') for d in orb_decisions])
    orb_trigger_trade_ids = set([t.get('trade_id') for t in all_triggers if t.get('trade_id') in orb_decision_trade_ids])

    print(f"\nORB decisions with trade_id: {len(orb_decision_trade_ids)}")
    print(f"ORB triggers (matched by trade_id): {len(orb_trigger_trade_ids)}")

    non_orb_decision_trade_ids = set([d.get('trade_id') for d in non_orb_decisions])
    non_orb_trigger_trade_ids = set([t.get('trade_id') for t in all_triggers if t.get('trade_id') in non_orb_decision_trade_ids])

    print(f"\nNon-ORB decisions with trade_id: {len(non_orb_decision_trade_ids)}")
    print(f"Non-ORB triggers (matched by trade_id): {len(non_orb_trigger_trade_ids)}")

    # Stage 4: Trades
    print("\n" + "=" * 80)
    print("STAGE 4: TRADES (Analytics)")
    print("=" * 80)

    print(f"\nTotal trades in analytics: {len(all_analytics)}")

    # Match trades to decisions
    analytics_trade_ids = set([t.get('trade_id') for t in all_analytics])
    orb_trade_ids = orb_decision_trade_ids & analytics_trade_ids
    non_orb_trade_ids = non_orb_decision_trade_ids & analytics_trade_ids

    print(f"ORB trades (matched by trade_id): {len(orb_trade_ids)}")
    print(f"Non-ORB trades (matched by trade_id): {len(non_orb_trade_ids)}")

    # Conversion funnel
    print("\n" + "=" * 80)
    print("CONVERSION FUNNEL")
    print("=" * 80)

    print(f"\nORB PIPELINE:")
    print(f"  1. Scanned: {len(orb_symbols)} symbols")
    print(f"  2. Decisions: {len(orb_decisions)}")
    print(f"     - Eligible: {len(orb_eligible)}")
    print(f"     - Ineligible: {len(orb_ineligible)}")
    print(f"  3. Triggers: {len(orb_trigger_trade_ids)}")
    print(f"  4. Trades: {len(orb_trade_ids)}")

    if len(orb_eligible) > 0:
        trigger_conversion = len(orb_trigger_trade_ids) / len(orb_eligible) * 100
    else:
        trigger_conversion = 0

    print(f"\nORB Conversion Rates:")
    print(f"  Eligible → Triggers: {trigger_conversion:.1f}% ({len(orb_trigger_trade_ids)}/{len(orb_eligible)})")

    print(f"\nNON-ORB PIPELINE:")
    print(f"  2. Decisions: {len(non_orb_decisions)}")
    print(f"  3. Triggers: {len(non_orb_trigger_trade_ids)}")
    print(f"  4. Trades: {len(non_orb_trade_ids)}")

    if len(non_orb_decisions) > 0:
        non_orb_trigger_conversion = len(non_orb_trigger_trade_ids) / len(non_orb_decisions) * 100
    else:
        non_orb_trigger_conversion = 0

    print(f"\nNon-ORB Conversion Rates:")
    print(f"  Decisions → Triggers: {non_orb_trigger_conversion:.1f}% ({len(non_orb_trigger_trade_ids)}/{len(non_orb_decisions)})")

    # Find the bottleneck
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)

    orb_decisions_not_triggered = orb_decision_trade_ids - orb_trigger_trade_ids
    print(f"\nORB decisions that did NOT trigger: {len(orb_decisions_not_triggered)}")

    if orb_decisions_not_triggered:
        print(f"\nSample ORB decisions that didn't trigger (first 10):")
        for trade_id in list(orb_decisions_not_triggered)[:10]:
            decision = decision_by_trade_id.get(trade_id)
            if decision:
                symbol = decision.get('symbol', '')
                session = decision.get('session', '')
                setup = decision.get('decision', {}).get('setup_type', '')
                eligible = decision.get('plan', {}).get('eligible', False)
                print(f"  {session} - {symbol} - {setup} - eligible={eligible}")

        print(f"\nChecking why they didn't trigger...")
        print(f"\nWere they eligible?")
        not_triggered_decisions = [decision_by_trade_id[tid] for tid in orb_decisions_not_triggered if tid in decision_by_trade_id]
        not_triggered_eligible = [d for d in not_triggered_decisions if d.get('plan', {}).get('eligible', False)]
        not_triggered_ineligible = [d for d in not_triggered_decisions if not d.get('plan', {}).get('eligible', False)]

        print(f"  Eligible but didn't trigger: {len(not_triggered_eligible)}")
        print(f"  Ineligible (expected): {len(not_triggered_ineligible)}")

        if not_triggered_eligible:
            print(f"\n⚠️ CRITICAL: {len(not_triggered_eligible)} eligible ORB decisions did NOT trigger!")
            print(f"\nThis is the bottleneck. These decisions passed all filters but failed to trigger.")
            print(f"\nSample eligible but not triggered (first 5):")
            for d in not_triggered_eligible[:5]:
                symbol = d.get('symbol', '')
                session = d.get('session', '')
                setup = d.get('decision', {}).get('setup_type', '')
                entry_zone = d.get('plan', {}).get('entry', {}).get('zone', [])
                print(f"  {session} - {symbol} - {setup}")
                print(f"    Entry zone: {entry_zone}")

if __name__ == '__main__':
    main()
