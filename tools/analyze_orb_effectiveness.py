#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze ORB (Opening Range Breakout) trade effectiveness in backtest run.

Checks:
1. How many ORB priority candidates were scanned
2. How many ORB decisions were made (ORH/ORL levels)
3. How many ORB triggers executed
4. ORB trade performance vs non-ORB trades
5. Timing distribution of ORB events
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
BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251113-115738_extracted\20251113-115738_full\20251113-115738")

def load_all_scanning_events():
    """Load all scanning.jsonl files from all sessions."""
    all_events = []
    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        scanning_file = session_dir / "scanning.jsonl"
        if not scanning_file.exists():
            continue
        with open(scanning_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    event['session_date'] = session_dir.name
                    all_events.append(event)
    return all_events

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

def load_all_analytics():
    """Load all analytics.jsonl files from all sessions."""
    all_trades = []
    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        analytics_file = session_dir / "analytics.jsonl"
        if not analytics_file.exists():
            continue
        with open(analytics_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    trade = json.loads(line)
                    trade['session_date'] = session_dir.name
                    all_trades.append(trade)
    return all_trades

def extract_time(timestamp_str):
    """Extract HH:MM from timestamp."""
    try:
        if 'T' in timestamp_str:
            return timestamp_str.split('T')[1][:5]
        return timestamp_str.split(' ')[1][:5] if ' ' in timestamp_str else 'N/A'
    except:
        return 'N/A'

def is_orb_event(event):
    """Check if event is ORB-related (has ORH/ORL level)."""
    decision = event.get('decision', {})

    # Check decision.level_name (old format)
    level_name = decision.get('level_name', '')
    if level_name in ('ORH', 'ORL'):
        return True

    # Check reasons string for "structure:level:ORH" or "structure:level:ORL" (new format)
    reasons = decision.get('reasons', '')
    if 'structure:level:ORH' in reasons or 'structure:level:ORL' in reasons:
        return True

    return False

def extract_orb_level(event):
    """Extract ORB level (ORH or ORL) from event."""
    decision = event.get('decision', {})

    # Check decision.level_name first
    level_name = decision.get('level_name', '')
    if level_name in ('ORH', 'ORL'):
        return level_name

    # Parse from reasons string
    reasons = decision.get('reasons', '')
    if 'structure:level:ORH' in reasons:
        return 'ORH'
    elif 'structure:level:ORL' in reasons:
        return 'ORL'

    return 'Unknown'

def main():
    print("=" * 80)
    print("ORB EFFECTIVENESS ANALYSIS")
    print("=" * 80)

    print("\nLoading scanning events...")
    scanning_events = load_all_scanning_events()
    print(f"Loaded {len(scanning_events)} scanning events")

    print("\nLoading decision/trigger events...")
    all_events = load_all_events()
    print(f"Loaded {len(all_events)} events")

    print("\nLoading analytics...")
    all_trades = load_all_analytics()
    print(f"Loaded {len(all_trades)} trades")

    # Analyze scanning events
    print("\n" + "=" * 80)
    print("PART 1: ORB PRIORITY SCANNER ACTIVITY")
    print("=" * 80)

    orb_scans = [e for e in scanning_events if e.get('category') == 'orb_priority']
    momentum_scans = [e for e in scanning_events if e.get('category') == 'momentum']
    mr_scans = [e for e in scanning_events if e.get('category') == 'mean_reversion']

    print(f"\nTotal scanning events: {len(scanning_events)}")
    print(f"  ORB priority: {len(orb_scans)} ({len(orb_scans)/len(scanning_events)*100:.1f}%)")
    print(f"  Momentum: {len(momentum_scans)} ({len(momentum_scans)/len(scanning_events)*100:.1f}%)")
    print(f"  Mean reversion: {len(mr_scans)} ({len(mr_scans)/len(scanning_events)*100:.1f}%)")

    orb_symbols = set([e['symbol'] for e in orb_scans])
    print(f"\nUnique ORB priority symbols: {len(orb_symbols)}")

    # Time distribution of ORB scans
    orb_by_hour = defaultdict(int)
    for e in orb_scans:
        time_str = extract_time(e.get('timestamp', ''))
        if time_str != 'N/A':
            hour = time_str[:2]
            orb_by_hour[hour] += 1

    print(f"\nORB priority scanning by hour:")
    for hour in sorted(orb_by_hour.keys()):
        print(f"  {hour}:00 - {orb_by_hour[hour]} scans")

    # Sample ORB priority events
    print(f"\nSample ORB priority scans (first 5):")
    for e in orb_scans[:5]:
        symbol = e.get('symbol', '')
        time = extract_time(e.get('timestamp', ''))
        bias = e.get('bias', '')
        rank = e.get('rank_long' if bias == 'long' else 'rank_short', 'N/A')
        print(f"  {time} - {symbol} ({bias}) rank={rank}")

    # Analyze decisions
    print("\n" + "=" * 80)
    print("PART 2: ORB DECISIONS")
    print("=" * 80)

    decisions = [e for e in all_events if e.get('type') == 'DECISION']
    orb_decisions = [d for d in decisions if is_orb_event(d)]

    print(f"\nTotal decisions: {len(decisions)}")
    print(f"ORB decisions (ORH/ORL): {len(orb_decisions)} ({len(orb_decisions)/len(decisions)*100:.1f}%)")

    # ORB decision breakdown
    orh_decisions = [d for d in orb_decisions if extract_orb_level(d) == 'ORH']
    orl_decisions = [d for d in orb_decisions if extract_orb_level(d) == 'ORL']

    print(f"  ORH breakouts: {len(orh_decisions)}")
    print(f"  ORL breakdowns: {len(orl_decisions)}")

    # Time distribution of ORB decisions
    orb_decision_by_hour = defaultdict(int)
    for d in orb_decisions:
        time_str = extract_time(d.get('timestamp', ''))
        if time_str != 'N/A':
            hour = time_str[:2]
            orb_decision_by_hour[hour] += 1

    print(f"\nORB decisions by hour:")
    for hour in sorted(orb_decision_by_hour.keys()):
        print(f"  {hour}:00 - {orb_decision_by_hour[hour]} decisions")

    # Sample ORB decisions
    print(f"\nSample ORB decisions (first 10):")
    for d in orb_decisions[:10]:
        symbol = d.get('symbol', '')
        time = extract_time(d.get('timestamp', ''))
        level = extract_orb_level(d)
        setup = d.get('decision', {}).get('setup_type', '')
        print(f"  {time} - {symbol} - {setup} at {level}")

    # Analyze triggers
    print("\n" + "=" * 80)
    print("PART 3: ORB TRIGGERS")
    print("=" * 80)

    triggers = [e for e in all_events if e.get('type') == 'TRIGGER']
    orb_triggers = [t for t in triggers if is_orb_event(t)]

    print(f"\nTotal triggers: {len(triggers)}")
    print(f"ORB triggers: {len(orb_triggers)} ({len(orb_triggers)/len(triggers)*100:.1f}%)")

    orb_trigger_rate = (len(orb_triggers) / len(orb_decisions) * 100) if orb_decisions else 0
    print(f"\nORB trigger rate: {orb_trigger_rate:.1f}% ({len(orb_triggers)}/{len(orb_decisions)})")

    # Sample ORB triggers
    if orb_triggers:
        print(f"\nSample ORB triggers (first 10):")
        for t in orb_triggers[:10]:
            symbol = t.get('symbol', '')
            time = extract_time(t.get('timestamp', ''))
            level = extract_orb_level(t)
            setup = t.get('decision', {}).get('setup_type', '')
            print(f"  {time} - {symbol} - {setup} at {level}")
    else:
        print("\n⚠️ NO ORB TRIGGERS FOUND")

    # Analyze trades
    print("\n" + "=" * 80)
    print("PART 4: ORB TRADE PERFORMANCE")
    print("=" * 80)

    # Match trades with ORB triggers
    orb_trigger_symbols = set([t['symbol'] for t in orb_triggers])
    orb_trades = [t for t in all_trades if t.get('symbol') in orb_trigger_symbols]

    print(f"\nTotal trades: {len(all_trades)}")
    print(f"ORB trades: {len(orb_trades)}")

    if orb_trades:
        orb_pnl = sum([t.get('pnl', 0) for t in orb_trades])
        orb_winners = len([t for t in orb_trades if t.get('pnl', 0) > 0])
        orb_wr = (orb_winners / len(orb_trades) * 100) if orb_trades else 0

        print(f"\nORB Trade Performance:")
        print(f"  Total PnL: Rs.{orb_pnl:.2f}")
        print(f"  Win Rate: {orb_wr:.1f}% ({orb_winners}/{len(orb_trades)})")
        print(f"  Avg PnL/Trade: Rs.{orb_pnl/len(orb_trades):.2f}")

        # Non-ORB trades
        non_orb_trades = [t for t in all_trades if t.get('symbol') not in orb_trigger_symbols]
        if non_orb_trades:
            non_orb_pnl = sum([t.get('pnl', 0) for t in non_orb_trades])
            non_orb_winners = len([t for t in non_orb_trades if t.get('pnl', 0) > 0])
            non_orb_wr = (non_orb_winners / len(non_orb_trades) * 100) if non_orb_trades else 0

            print(f"\nNon-ORB Trade Performance:")
            print(f"  Total PnL: Rs.{non_orb_pnl:.2f}")
            print(f"  Win Rate: {non_orb_wr:.1f}% ({non_orb_winners}/{len(non_orb_trades)})")
            print(f"  Avg PnL/Trade: Rs.{non_orb_pnl/len(non_orb_trades):.2f}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print(f"\n1. ORB PRIORITY SCANNER:")
    print(f"   - Scanned {len(orb_symbols)} unique symbols as ORB priority")
    print(f"   - {len(orb_scans)/len(scanning_events)*100:.1f}% of all scans were ORB priority")
    print(f"   - Peak activity: {max(orb_by_hour.items(), key=lambda x: x[1])[0]}:00 with {max(orb_by_hour.values())} scans")

    print(f"\n2. ORB DECISIONS:")
    print(f"   - {len(orb_decisions)} ORB decisions made ({len(orb_decisions)/len(decisions)*100:.1f}% of all decisions)")
    print(f"   - ORH: {len(orh_decisions)}, ORL: {len(orl_decisions)}")

    print(f"\n3. ORB TRIGGER RATE:")
    print(f"   - {len(orb_triggers)}/{len(orb_decisions)} = {orb_trigger_rate:.1f}%")
    if orb_trigger_rate == 0:
        print(f"   ⚠️ WARNING: NO ORB TRADES TRIGGERED!")
        print(f"   - ORB candidates ARE being scanned ({len(orb_symbols)} symbols)")
        print(f"   - ORB decisions ARE being made ({len(orb_decisions)} decisions)")
        print(f"   - But NONE are triggering - investigate trigger logic")

    if orb_trades:
        print(f"\n4. ORB PERFORMANCE:")
        print(f"   - {len(orb_trades)} ORB trades executed")
        print(f"   - Win rate: {orb_wr:.1f}% vs {non_orb_wr:.1f}% (non-ORB)")
        print(f"   - Total PnL: Rs.{orb_pnl:.2f} vs Rs.{non_orb_pnl:.2f} (non-ORB)")

if __name__ == '__main__':
    main()
