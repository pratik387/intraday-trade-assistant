#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze failing breakout trades to find common patterns

From backtest_20251114-031403:
- Breakout Long: Rs.69 (45.5% WR, 11 trades)
- Breakout Short: -Rs.1,581 (25.0% WR, 16 trades) - CATASTROPHIC

Need to find:
1. What are the common characteristics of LOSING breakout trades?
2. What differentiates winners from losers?
3. Are there specific entry timing issues?
4. Is there a pattern in exit reasons (SL hit too early, etc.)?
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Use the extracted backtest directory
BACKTEST_DIR = Path(r"backtest_20251114-031403_extracted\20251114-031403_full\20251114-031403")

def main():
    print("=" * 80)
    print("FAILING BREAKOUT TRADES ANALYSIS")
    print("Finding patterns in LOSING trades to fix root cause")
    print("=" * 80)

    if not BACKTEST_DIR.exists():
        print(f"\nERROR: Backtest directory not found: {BACKTEST_DIR}")
        print("Please extract backtest_20251114-031403.zip first")
        return

    # Collect all breakout trades
    breakout_trades = []

    print("\nLoading trades...")
    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        events_file = session_dir / "events.jsonl"
        if not events_file.exists():
            continue

        # Load all events for this session
        session_events = []
        with open(events_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    session_events.append(event)

        # Find breakout trades (TRIGGER + corresponding EXIT/FINAL)
        triggered_trades = {}
        for event in session_events:
            if event.get('type') == 'TRIGGER':
                trigger = event.get('trigger', {})
                strategy = trigger.get('strategy', '')
                if 'breakout' in strategy.lower():
                    trade_id = event.get('trade_id')
                    triggered_trades[trade_id] = {
                        'session': session_dir.name,
                        'symbol': event.get('symbol'),
                        'strategy': strategy,
                        'trigger_ts': event.get('ts'),
                        'trigger_event': event,
                        'exit_events': [],
                        'final_event': None
                    }

            elif event.get('type') in ['EXIT', 'FINAL']:
                trade_id = event.get('trade_id')
                if trade_id in triggered_trades:
                    if event.get('type') == 'FINAL':
                        triggered_trades[trade_id]['final_event'] = event
                    else:
                        triggered_trades[trade_id]['exit_events'].append(event)

        breakout_trades.extend(triggered_trades.values())

    print(f"Loaded {len(breakout_trades)} breakout trades")

    # Analyze by strategy and PnL
    winners = [t for t in breakout_trades if t.get('final_event') and t['final_event'].get('pnl_net', 0) > 0]
    losers = [t for t in breakout_trades if t.get('final_event') and t['final_event'].get('pnl_net', 0) <= 0]

    print(f"\n1. WIN/LOSS BREAKDOWN:")
    print(f"   Winners: {len(winners)} trades")
    print(f"   Losers: {len(losers)} trades")
    print(f"   Win Rate: {len(winners)/(len(winners)+len(losers))*100:.1f}%")

    # Breakdown by long/short
    long_trades = [t for t in breakout_trades if 'long' in t['strategy'].lower()]
    short_trades = [t for t in breakout_trades if 'short' in t['strategy'].lower()]

    long_winners = [t for t in long_trades if t.get('final_event') and t['final_event'].get('pnl_net', 0) > 0]
    long_losers = [t for t in long_trades if t.get('final_event') and t['final_event'].get('pnl_net', 0) <= 0]

    short_winners = [t for t in short_trades if t.get('final_event') and t['final_event'].get('pnl_net', 0) > 0]
    short_losers = [t for t in short_trades if t.get('final_event') and t['final_event'].get('pnl_net', 0) <= 0]

    print(f"\n   LONG Breakouts: {len(long_trades)} trades")
    print(f"     Winners: {len(long_winners)} ({len(long_winners)/len(long_trades)*100:.1f}% WR)")
    print(f"     Losers: {len(long_losers)}")
    if long_trades:
        long_pnl = sum(t['final_event'].get('pnl_net', 0) for t in long_trades if t.get('final_event'))
        print(f"     Total PnL: Rs.{long_pnl:.2f} (Rs.{long_pnl/len(long_trades):.2f}/trade)")

    print(f"\n   SHORT Breakouts: {len(short_trades)} trades")
    print(f"     Winners: {len(short_winners)} ({len(short_winners)/len(short_trades)*100:.1f}% WR)")
    print(f"     Losers: {len(short_losers)}")
    if short_trades:
        short_pnl = sum(t['final_event'].get('pnl_net', 0) for t in short_trades if t.get('final_event'))
        print(f"     Total PnL: Rs.{short_pnl:.2f} (Rs.{short_pnl/len(short_trades):.2f}/trade)")

    # Analyze EXIT REASONS for losers
    print(f"\n" + "=" * 80)
    print("2. EXIT REASON ANALYSIS - Why did losers lose?")
    print("=" * 80)

    loser_exit_reasons = defaultdict(int)
    loser_exit_details = defaultdict(list)

    for trade in losers:
        final = trade.get('final_event', {})
        exit_reason = final.get('exit_reason', 'unknown')
        loser_exit_reasons[exit_reason] += 1

        pnl = final.get('pnl_net', 0)
        duration_bars = len(trade.get('exit_events', []))

        loser_exit_details[exit_reason].append({
            'symbol': trade['symbol'],
            'strategy': trade['strategy'],
            'pnl': pnl,
            'duration_bars': duration_bars,
            'session': trade['session']
        })

    print(f"\nLOSING TRADES EXIT REASONS:")
    for reason, count in sorted(loser_exit_reasons.items(), key=lambda x: -x[1]):
        pct = count / len(losers) * 100 if losers else 0
        avg_loss = sum(t['pnl'] for t in loser_exit_details[reason]) / count
        print(f"  {reason:<20}: {count:>3} trades ({pct:>5.1f}%) | Avg loss: Rs.{avg_loss:.2f}")

    # Compare with WINNERS exit reasons
    winner_exit_reasons = defaultdict(int)
    for trade in winners:
        final = trade.get('final_event', {})
        exit_reason = final.get('exit_reason', 'unknown')
        winner_exit_reasons[exit_reason] += 1

    print(f"\nWINNING TRADES EXIT REASONS (for comparison):")
    for reason, count in sorted(winner_exit_reasons.items(), key=lambda x: -x[1]):
        pct = count / len(winners) * 100 if winners else 0
        print(f"  {reason:<20}: {count:>3} trades ({pct:>5.1f}%)")

    # Analyze ENTRY TIMING - time from decision to trigger
    print(f"\n" + "=" * 80)
    print("3. ENTRY TIMING ANALYSIS - Decision to Trigger Delay")
    print("=" * 80)

    # Need to match decisions to triggers
    all_decisions = {}
    all_triggers = {}

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

                    if event.get('type') == 'DECISION':
                        decision = event.get('decision', {})
                        setup_type = decision.get('setup_type', '')
                        if 'breakout' in setup_type.lower():
                            trade_id = event.get('trade_id')
                            all_decisions[trade_id] = {
                                'ts': event.get('ts'),
                                'symbol': event.get('symbol'),
                                'setup_type': setup_type,
                                'entry_ref': decision.get('entry_reference', 0)
                            }

                    elif event.get('type') == 'TRIGGER':
                        trigger = event.get('trigger', {})
                        strategy = trigger.get('strategy', '')
                        if 'breakout' in strategy.lower():
                            trade_id = event.get('trade_id')
                            all_triggers[trade_id] = {
                                'ts': event.get('ts'),
                                'symbol': event.get('symbol'),
                                'strategy': strategy,
                                'entry_price': trigger.get('entry', 0)
                            }

    # Match decisions to triggers and calculate delays
    winner_delays = []
    loser_delays = []

    winner_trade_ids = set(t['trigger_event'].get('trade_id') for t in winners if t.get('trigger_event'))
    loser_trade_ids = set(t['trigger_event'].get('trade_id') for t in losers if t.get('trigger_event'))

    for trade_id in all_decisions:
        if trade_id in all_triggers:
            decision_ts = pd.Timestamp(all_decisions[trade_id]['ts'])
            trigger_ts = pd.Timestamp(all_triggers[trade_id]['ts'])
            delay_minutes = (trigger_ts - decision_ts).total_seconds() / 60

            if trade_id in winner_trade_ids:
                winner_delays.append(delay_minutes)
            elif trade_id in loser_trade_ids:
                loser_delays.append(delay_minutes)

    if winner_delays and loser_delays:
        print(f"\nDECISION → TRIGGER DELAY:")
        print(f"  Winners: {len(winner_delays)} trades")
        print(f"    Avg delay: {sum(winner_delays)/len(winner_delays):.1f} minutes")
        print(f"    Median delay: {sorted(winner_delays)[len(winner_delays)//2]:.1f} minutes")
        print(f"    Min/Max: {min(winner_delays):.1f} / {max(winner_delays):.1f} minutes")

        print(f"\n  Losers: {len(loser_delays)} trades")
        print(f"    Avg delay: {sum(loser_delays)/len(loser_delays):.1f} minutes")
        print(f"    Median delay: {sorted(loser_delays)[len(loser_delays)//2]:.1f} minutes")
        print(f"    Min/Max: {min(loser_delays):.1f} / {max(loser_delays):.1f} minutes")

        avg_winner_delay = sum(winner_delays) / len(winner_delays)
        avg_loser_delay = sum(loser_delays) / len(loser_delays)

        print(f"\n  KEY FINDING:")
        if avg_loser_delay > avg_winner_delay:
            print(f"    Losers trigger {avg_loser_delay - avg_winner_delay:.1f} min SLOWER than winners")
            print(f"    => Late entries are catching reversals, not momentum!")
        else:
            print(f"    Winners trigger {avg_winner_delay - avg_loser_delay:.1f} min SLOWER than losers")
            print(f"    => Entry timing is NOT the primary issue")

    # Analyze COMMON PATTERNS in losers
    print(f"\n" + "=" * 80)
    print("4. COMMON PATTERNS IN LOSING TRADES")
    print("=" * 80)

    # Most frequently losing symbols
    loser_symbols = defaultdict(int)
    loser_symbol_pnl = defaultdict(float)

    for trade in losers:
        symbol = trade['symbol']
        loser_symbols[symbol] += 1
        if trade.get('final_event'):
            loser_symbol_pnl[symbol] += trade['final_event'].get('pnl_net', 0)

    print(f"\nMOST FREQUENTLY LOSING SYMBOLS (top 10):")
    sorted_symbols = sorted(loser_symbols.items(), key=lambda x: -x[1])
    for symbol, count in sorted_symbols[:10]:
        avg_loss = loser_symbol_pnl[symbol] / count
        print(f"  {symbol:<15}: {count} losses | Avg loss: Rs.{avg_loss:.2f}")

    # Losing trades by session (time-based patterns)
    loser_sessions = defaultdict(int)
    for trade in losers:
        loser_sessions[trade['session']] += 1

    print(f"\nSESSIONS WITH MOST LOSSES (top 10):")
    sorted_sessions = sorted(loser_sessions.items(), key=lambda x: -x[1])
    for session, count in sorted_sessions[:10]:
        print(f"  {session}: {count} losses")

    # DETAILED SAMPLE OF WORST LOSERS
    print(f"\n" + "=" * 80)
    print("5. WORST LOSING TRADES (bottom 10)")
    print("=" * 80)

    worst_losers = sorted(losers, key=lambda t: t.get('final_event', {}).get('pnl_net', 0))[:10]

    print(f"\n{'Symbol':<12} {'Strategy':<25} {'PnL':>10} {'Exit Reason':<15} {'Session':<15}")
    print("-" * 90)

    for trade in worst_losers:
        symbol = trade['symbol']
        strategy = trade['strategy']
        final = trade.get('final_event', {})
        pnl = final.get('pnl_net', 0)
        exit_reason = final.get('exit_reason', 'unknown')
        session = trade['session']

        print(f"{symbol:<12} {strategy:<25} Rs.{pnl:>7.2f} {exit_reason:<15} {session:<15}")

    # SUMMARY
    print(f"\n" + "=" * 80)
    print("SUMMARY & PATTERNS")
    print("=" * 80)

    print(f"\n1. PRIMARY ISSUE: Breakout SHORT trades are failing (25% WR)")
    print(f"   - SHORT losers dominate the losses")
    print(f"   - Suggests catching REVERSALS instead of true breakdowns")

    if avg_loser_delay > avg_winner_delay:
        print(f"\n2. TIMING ISSUE: Late entries")
        print(f"   - Losers trigger {avg_loser_delay - avg_winner_delay:.1f} min after decision")
        print(f"   - By the time we enter, momentum is already fading")
        print(f"   - Need IMMEDIATE execution or tighter trigger timeout")

    print(f"\n3. EXIT REASON PATTERN:")
    top_exit = max(loser_exit_reasons.items(), key=lambda x: x[1])
    print(f"   - Most common exit: {top_exit[0]} ({top_exit[1]} trades)")
    if top_exit[0] in ['hard_sl', 'HARD_SL']:
        print(f"   - Stop loss hit too early = catching false breakouts")
        print(f"   - Need better QUALITY filters BEFORE decision")

    print(f"\n4. RECOMMENDED FIXES:")
    print(f"   A. IMMEDIATE TRIGGER: Remove entry zone wait, enter on decision")
    print(f"   B. BREAKOUT CONFIRMATION: Add volume surge + momentum filters")
    print(f"   C. SHORT BIAS FIX: Stricter filters for breakout_short (lower win rate)")
    print(f"   D. STOP PLACEMENT: Review if stops are too tight for breakout volatility")

if __name__ == '__main__':
    main()
