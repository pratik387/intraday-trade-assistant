#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze ORB (Opening Range Breakout) trades.

Questions to answer:
1. How many ORB DECISIONS were made (ORH/ORL levels)?
2. How many of those actually TRIGGERED?
3. What time were they detected vs triggered?
4. Are there any patterns preventing ORB trades from triggering?
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

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

        with open(events_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    event['session_date'] = session_dir.name
                    all_events.append(event)

    return all_events

def is_orb_trade(event):
    """Check if event is related to ORB (ORH/ORL levels)."""
    decision = event.get('decision', {})

    # Check for ORH/ORL in level_name or reasons
    level_name = decision.get('level_name', '')
    reasons = decision.get('reasons', '')

    return 'ORH' in level_name or 'ORL' in level_name or 'ORH' in reasons or 'ORL' in reasons

def extract_time(event):
    """Extract time from event timestamp."""
    timestamp_str = event.get('timestamp', '')
    if timestamp_str:
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime('%H:%M:%S')
        except:
            return 'N/A'
    return 'N/A'

def main():
    print("Loading all events from events.jsonl files...")
    all_events = load_all_events()
    print(f"Total events loaded: {len(all_events)}")

    # Categorize events
    orb_decisions = []
    orb_triggers = []
    orb_rejections = []
    all_decisions = []
    all_triggers = []

    for event in all_events:
        event_type = event.get('type')

        if event_type == 'DECISION':
            all_decisions.append(event)
            if is_orb_trade(event):
                orb_decisions.append(event)
                # Check if rejected
                decision = event.get('decision', {})
                if not decision.get('eligible', True):
                    orb_rejections.append(event)
        elif event_type == 'TRIGGER':
            all_triggers.append(event)
            if is_orb_trade(event):
                orb_triggers.append(event)

    print("\n" + "="*80)
    print("ORB TRADE ANALYSIS")
    print("="*80)

    print(f"\nTotal DECISIONS: {len(all_decisions)}")
    print(f"Total TRIGGERS: {len(all_triggers)}")

    print(f"\nORB DECISIONS (ORH/ORL): {len(orb_decisions)}")
    print(f"ORB TRIGGERS (ORH/ORL): {len(orb_triggers)}")
    print(f"ORB REJECTIONS: {len(orb_rejections)}")

    trigger_rate = (len(orb_triggers) / len(orb_decisions) * 100) if orb_decisions else 0
    print(f"\nORB TRIGGER RATE: {trigger_rate:.1f}% ({len(orb_triggers)}/{len(orb_decisions)})")

    # Show sample ORB decisions
    print("\n" + "="*80)
    print("SAMPLE ORB DECISIONS (First 20)")
    print("="*80)

    for event in orb_decisions[:20]:
        symbol = event.get('symbol', '')
        session = event.get('session_date', '')
        time = extract_time(event)
        decision = event.get('decision', {})
        level_name = decision.get('level_name', 'N/A')
        setup_type = decision.get('setup_type', 'N/A')
        eligible = decision.get('eligible', False)
        reasons = decision.get('reasons', '')

        print(f"\n{session} {time} - {symbol}:")
        print(f"  Level: {level_name}")
        print(f"  Setup: {setup_type}")
        print(f"  Eligible: {eligible}")
        if not eligible:
            rejection_reason = decision.get('quality', {}).get('rejection_reason', 'Unknown')
            print(f"  REJECTED: {rejection_reason}")
        else:
            print(f"  Reasons: {reasons[:150]}...")

    # Show sample ORB triggers
    if orb_triggers:
        print("\n" + "="*80)
        print("SAMPLE ORB TRIGGERS (All)")
        print("="*80)

        for event in orb_triggers:
            symbol = event.get('symbol', '')
            session = event.get('session_date', '')
            time = extract_time(event)
            decision = event.get('decision', {})
            level_name = decision.get('level_name', 'N/A')
            setup_type = decision.get('setup_type', 'N/A')

            print(f"\n{session} {time} - {symbol}:")
            print(f"  Level: {level_name}")
            print(f"  Setup: {setup_type}")
    else:
        print("\n" + "="*80)
        print("NO ORB TRIGGERS FOUND!")
        print("="*80)

    # Show ORB rejections
    if orb_rejections:
        print("\n" + "="*80)
        print("ORB REJECTIONS ANALYSIS")
        print("="*80)

        rejection_reasons = defaultdict(int)

        for event in orb_rejections:
            decision = event.get('decision', {})
            rejection_reason = decision.get('quality', {}).get('rejection_reason', 'Unknown')
            rejection_reasons[rejection_reason] += 1

        print(f"\nTotal ORB Rejections: {len(orb_rejections)}")
        print("\nRejection Reasons:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} ({count/len(orb_rejections)*100:.1f}%)")

        print("\nSample Rejections (First 10):")
        for event in orb_rejections[:10]:
            symbol = event.get('symbol', '')
            session = event.get('session_date', '')
            time = extract_time(event)
            decision = event.get('decision', {})
            level_name = decision.get('level_name', 'N/A')
            rejection_reason = decision.get('quality', {}).get('rejection_reason', 'Unknown')

            print(f"\n{session} {time} - {symbol}:")
            print(f"  Level: {level_name}")
            print(f"  REJECTED: {rejection_reason}")

    # Time distribution
    print("\n" + "="*80)
    print("ORB DECISION TIME DISTRIBUTION")
    print("="*80)

    time_buckets = {
        '09:15-09:30': 0,
        '09:30-10:00': 0,
        '10:00-10:30': 0,
        '10:30-11:00': 0,
        '11:00-12:00': 0,
        '12:00+': 0
    }

    for event in orb_decisions:
        time_str = extract_time(event)
        if time_str != 'N/A':
            hour, minute, _ = map(int, time_str.split(':'))
            time_minutes = hour * 60 + minute

            if 555 <= time_minutes < 570:  # 9:15-9:30
                time_buckets['09:15-09:30'] += 1
            elif 570 <= time_minutes < 600:  # 9:30-10:00
                time_buckets['09:30-10:00'] += 1
            elif 600 <= time_minutes < 630:  # 10:00-10:30
                time_buckets['10:00-10:30'] += 1
            elif 630 <= time_minutes < 660:  # 10:30-11:00
                time_buckets['10:30-11:00'] += 1
            elif 660 <= time_minutes < 720:  # 11:00-12:00
                time_buckets['11:00-12:00'] += 1
            else:
                time_buckets['12:00+'] += 1

    for bucket, count in time_buckets.items():
        pct = (count / len(orb_decisions) * 100) if orb_decisions else 0
        print(f"{bucket}: {count} ({pct:.1f}%)")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    print(f"\n1. ORB DECISION RATE:")
    print(f"   ORB decisions: {len(orb_decisions)}/{len(all_decisions)} = {len(orb_decisions)/len(all_decisions)*100:.1f}%")

    print(f"\n2. ORB TRIGGER RATE:")
    print(f"   ORB triggers: {len(orb_triggers)}/{len(orb_decisions)} = {trigger_rate:.1f}%")

    if len(orb_triggers) == 0:
        print(f"\n3. PROBLEM IDENTIFIED:")
        print(f"   NO ORB TRADES TRIGGERED!")
        print(f"   ORB decisions are being made ({len(orb_decisions)})")
        print(f"   But NONE are triggering")

        if orb_rejections:
            print(f"\n4. POSSIBLE CAUSE:")
            print(f"   {len(orb_rejections)}/{len(orb_decisions)} ORB decisions were REJECTED")
            print(f"   Rejection rate: {len(orb_rejections)/len(orb_decisions)*100:.1f}%")
            print(f"\n   Top rejection reasons:")
            for reason, count in list(sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True))[:5]:
                print(f"   - {reason}: {count}")
        else:
            eligible_orb = len(orb_decisions) - len(orb_rejections)
            print(f"\n4. ALTERNATIVE CAUSE:")
            print(f"   {eligible_orb} ORB decisions were ELIGIBLE but didn't trigger")
            print(f"   This suggests a trigger logic problem, not a filter problem")

if __name__ == '__main__':
    main()
