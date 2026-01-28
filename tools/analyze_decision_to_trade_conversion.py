"""
Analyze why DECISION events don't convert to completed trades.

CRITICAL QUESTION: Why do we have 89 vwap_reclaim_long decisions but only 44 trades?
And why do order_block_long have 55 decisions but only 9 trades?

This could indicate:
1. Trigger price not met (trade planned but never triggered)
2. Rejection by other gates (risk, regime, filters)
3. Trade triggered but not completed yet (unlikely in historical backtest)
4. Detection happening but trades being rejected somewhere
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def analyze_decision_to_trade_funnel(backtest_dir: str):
    """Analyze the funnel from DECISION to completed trade."""

    base_path = Path(backtest_dir)

    setups_to_analyze = [
        'vwap_reclaim_long',
        'order_block_long',
        'fair_value_gap_long',
        'order_block_short',
        'fair_value_gap_short',
    ]

    print("="*80)
    print("DECISION -> TRADE CONVERSION FUNNEL ANALYSIS")
    print("="*80)

    for setup_name in setups_to_analyze:
        print(f"\n{'='*80}")
        print(f"{setup_name.upper()}")
        print(f"{'='*80}")

        # Step 1: Count DECISION events
        decisions = []
        for session_dir in sorted(base_path.iterdir()):
            if not session_dir.is_dir():
                continue

            events_file = session_dir / 'events.jsonl'
            if not events_file.exists():
                continue

            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if (event.get('type') == 'DECISION' and
                            event.get('decision', {}).get('setup_type') == setup_name):
                            decisions.append({
                                'session': session_dir.name,
                                'symbol': event.get('symbol'),
                                'regime': event.get('decision', {}).get('regime'),
                                'timestamp': event.get('ts'),
                                'trade_id': event.get('trade_id'),
                            })
                    except:
                        continue

        # Step 2: Count TRIGGER events
        triggers = []
        for session_dir in sorted(base_path.iterdir()):
            if not session_dir.is_dir():
                continue

            events_file = session_dir / 'events.jsonl'
            if not events_file.exists():
                continue

            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('type') == 'TRIGGER':
                            # Check if this trigger is for our setup
                            # Need to match by trade_id or symbol+timestamp
                            triggers.append({
                                'session': session_dir.name,
                                'symbol': event.get('symbol'),
                                'trade_id': event.get('trade_id'),
                                'timestamp': event.get('ts'),
                            })
                    except:
                        continue

        # Step 3: Count completed trades
        completed_trades = []
        for session_dir in sorted(base_path.iterdir()):
            if not session_dir.is_dir():
                continue

            analytics_file = session_dir / 'analytics.jsonl'
            if not analytics_file.exists():
                continue

            with open(analytics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        trade = json.loads(line.strip())
                        if trade.get('setup_type') == setup_name:
                            completed_trades.append({
                                'session': session_dir.name,
                                'symbol': trade.get('symbol'),
                                'regime': trade.get('regime'),
                                'pnl': trade.get('pnl', 0),
                                'exit_reason': trade.get('exit_reason'),
                            })
                    except:
                        continue

        # Print funnel
        print(f"\nFUNNEL:")
        print(f"  1. DECISION events:     {len(decisions)}")
        print(f"  2. Completed trades:    {len(completed_trades)}")
        print(f"  Conversion rate:        {len(completed_trades)/len(decisions)*100:.1f}%")
        print(f"  Loss:                   {len(decisions) - len(completed_trades)} decisions didn't convert")

        # Regime breakdown for decisions
        decisions_by_regime = defaultdict(int)
        for d in decisions:
            decisions_by_regime[d['regime']] += 1

        print(f"\n  DECISIONS BY REGIME:")
        for regime, count in sorted(decisions_by_regime.items(), key=lambda x: -x[1]):
            print(f"    {regime:<15} {count:>3} decisions")

        # Regime breakdown for completed trades
        trades_by_regime = defaultdict(int)
        for t in completed_trades:
            trades_by_regime[t['regime']] += 1

        print(f"\n  COMPLETED TRADES BY REGIME:")
        for regime, count in sorted(trades_by_regime.items(), key=lambda x: -x[1]):
            pct = count / len(completed_trades) * 100 if len(completed_trades) > 0 else 0
            print(f"    {regime:<15} {count:>3} trades ({pct:.1f}%)")

        # Check if chop decisions are being rejected
        chop_decisions = decisions_by_regime.get('chop', 0)
        chop_trades = trades_by_regime.get('chop', 0)

        if chop_decisions > 0:
            chop_conversion = chop_trades / chop_decisions * 100
            print(f"\n  CHOP REGIME CONVERSION:")
            print(f"    CHOP decisions: {chop_decisions}")
            print(f"    CHOP trades: {chop_trades}")
            print(f"    Conversion rate: {chop_conversion:.1f}%")

            if chop_conversion < 100:
                print(f"    >>> {chop_decisions - chop_trades} CHOP decisions were REJECTED somewhere!")

        # Sample a few non-converted decisions
        if len(decisions) > len(completed_trades):
            print(f"\n  SAMPLE NON-CONVERTED DECISIONS:")
            # Find decisions that didn't convert
            trade_ids = {t.get('trade_id') for t in completed_trades if 'trade_id' in t}
            decision_trade_ids = {d['trade_id'] for d in decisions}

            # Show first 3 sessions with decisions
            sessions_with_decisions = {}
            for d in decisions:
                if d['session'] not in sessions_with_decisions:
                    sessions_with_decisions[d['session']] = []
                sessions_with_decisions[d['session']].append(d)

            for i, (session, session_decisions) in enumerate(sorted(sessions_with_decisions.items())[:3]):
                session_trades = [t for t in completed_trades if t['session'] == session]
                print(f"\n    Session {session}:")
                print(f"      Decisions: {len(session_decisions)}")
                print(f"      Trades: {len(session_trades)}")
                print(f"      Conversion: {len(session_trades)/len(session_decisions)*100:.1f}%")

    # CRITICAL CHECK: Are decisions being made but trades not completing?
    print(f"\n\n{'='*80}")
    print("CRITICAL ANALYSIS: WHY ARE DECISIONS NOT CONVERTING?")
    print(f"{'='*80}")

    print(f"\nPossible reasons for low conversion rates:")
    print(f"1. Trigger price not met (setup detected, but price never reached entry)")
    print(f"2. Rejected by regime gate (decision made, but regime changed before trigger)")
    print(f"3. Rejected by risk gate (position sizing, max positions, etc.)")
    print(f"4. Rejected by filters (ADX, volume, etc. at trigger time)")
    print(f"5. Trade timeout (45min guard rail - cancel if trigger not met)")

    print(f"\nTo investigate, we need to check:")
    print(f"- TRIGGER events (shows if entry price was met)")
    print(f"- REJECTION events (shows why trades were rejected)")
    print(f"- TIMEOUT events (shows if trades expired)")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    backtest_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251117-021749_extracted\20251117-021749_full\20251117-021749"
    analyze_decision_to_trade_funnel(backtest_dir)
