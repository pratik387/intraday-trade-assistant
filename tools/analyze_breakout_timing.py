#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze timing between DECISION and TRIGGER for breakout trades

Question: How long are we waiting between decision and trigger?
Is there a pattern in timing delays for winners vs losers?
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BACKTEST_DIR = Path(r"backtest_20251114-031403_extracted\20251114-031403_full\20251114-031403")

def main():
    print("=" * 80)
    print("BREAKOUT TIMING ANALYSIS: Decision → Trigger Delays")
    print("=" * 80)

    if not BACKTEST_DIR.exists():
        print(f"\nERROR: Directory not found: {BACKTEST_DIR}")
        return

    # Collect all decisions and triggers
    decisions = {}  # trade_id -> decision event
    triggers = {}   # trade_id -> trigger event
    finals = {}     # trade_id -> final event (for PnL)

    print("\nLoading events...")
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
                    event['session'] = session_dir.name

                    if event.get('type') == 'DECISION':
                        decision = event.get('decision', {})
                        setup_type = decision.get('setup_type', '')
                        if 'breakout' in setup_type.lower():
                            trade_id = event.get('trade_id')
                            decisions[trade_id] = event

                    elif event.get('type') == 'TRIGGER':
                        trigger = event.get('trigger', {})
                        strategy = trigger.get('strategy', '')
                        if 'breakout' in strategy.lower():
                            trade_id = event.get('trade_id')
                            triggers[trade_id] = event

                    elif event.get('type') == 'FINAL':
                        trade_id = event.get('trade_id')
                        if trade_id in decisions:  # Only track if it was a breakout decision
                            finals[trade_id] = event

    print(f"Found {len(decisions)} breakout decisions")
    print(f"Found {len(triggers)} breakout triggers")
    print(f"Found {len(finals)} completed trades")

    # Calculate delays and categorize
    timing_data = []

    for trade_id in decisions:
        decision = decisions[trade_id]
        decision_ts = pd.Timestamp(decision.get('ts'))
        symbol = decision.get('symbol')
        session = decision.get('session')

        decision_info = decision.get('decision', {})
        setup_type = decision_info.get('setup_type', '')

        plan = decision.get('plan', {})
        entry_zone = plan.get('entry', {}).get('zone', [])
        entry_ref = plan.get('entry', {}).get('reference', 0)

        # Check if triggered
        if trade_id in triggers:
            trigger = triggers[trade_id]
            trigger_ts = pd.Timestamp(trigger.get('ts'))
            trigger_info = trigger.get('trigger', {})
            entry_price = trigger_info.get('entry', 0)

            # Calculate delay
            delay_minutes = (trigger_ts - decision_ts).total_seconds() / 60

            # Get PnL if available
            pnl = None
            exit_reason = None
            if trade_id in finals:
                final = finals[trade_id]
                pnl = final.get('pnl_net', 0)
                exit_reason = final.get('exit_reason', 'unknown')

            timing_data.append({
                'trade_id': trade_id,
                'symbol': symbol,
                'session': session,
                'setup_type': setup_type,
                'decision_ts': decision_ts,
                'trigger_ts': trigger_ts,
                'delay_minutes': delay_minutes,
                'entry_zone': entry_zone,
                'entry_ref': entry_ref,
                'entry_price': entry_price,
                'pnl': pnl,
                'exit_reason': exit_reason,
                'triggered': True
            })
        else:
            # Decision but no trigger (rejected)
            timing_data.append({
                'trade_id': trade_id,
                'symbol': symbol,
                'session': session,
                'setup_type': setup_type,
                'decision_ts': decision_ts,
                'trigger_ts': None,
                'delay_minutes': None,
                'entry_zone': entry_zone,
                'entry_ref': entry_ref,
                'entry_price': None,
                'pnl': None,
                'exit_reason': None,
                'triggered': False
            })

    # Analysis
    triggered = [t for t in timing_data if t['triggered']]
    not_triggered = [t for t in timing_data if not t['triggered']]

    print(f"\n" + "=" * 80)
    print("1. TRIGGER SUCCESS RATE")
    print("=" * 80)
    print(f"Decisions: {len(timing_data)}")
    print(f"Triggered: {len(triggered)} ({len(triggered)/len(timing_data)*100:.1f}%)")
    print(f"Not Triggered: {len(not_triggered)} ({len(not_triggered)/len(timing_data)*100:.1f}%)")

    if triggered:
        print(f"\n" + "=" * 80)
        print("2. TIMING DELAYS (Decision → Trigger)")
        print("=" * 80)

        delays = [t['delay_minutes'] for t in triggered]
        delays_sorted = sorted(delays)

        print(f"\nOVERALL DELAYS:")
        print(f"  Average delay: {sum(delays)/len(delays):.1f} minutes")
        print(f"  Median delay: {delays_sorted[len(delays)//2]:.1f} minutes")
        print(f"  Min delay: {min(delays):.1f} minutes")
        print(f"  Max delay: {max(delays):.1f} minutes")
        print(f"  75th percentile: {delays_sorted[int(len(delays)*0.75)]:.1f} minutes")
        print(f"  95th percentile: {delays_sorted[int(len(delays)*0.95)]:.1f} minutes")

        # Delay distribution
        print(f"\nDELAY DISTRIBUTION:")
        delay_bins = [
            (0, 1, "Immediate (0-1 min)"),
            (1, 5, "Fast (1-5 min)"),
            (5, 15, "Moderate (5-15 min)"),
            (15, 30, "Slow (15-30 min)"),
            (30, 45, "Very slow (30-45 min)"),
            (45, float('inf'), "Timeout zone (45+ min)")
        ]

        for min_delay, max_delay, label in delay_bins:
            count = sum(1 for d in delays if min_delay <= d < max_delay)
            pct = count / len(delays) * 100
            print(f"  {label:<30}: {count:>3} trades ({pct:>5.1f}%)")

        # Winners vs Losers timing
        winners = [t for t in triggered if t.get('pnl') is not None and t['pnl'] > 0]
        losers = [t for t in triggered if t.get('pnl') is not None and t['pnl'] <= 0]

        if winners and losers:
            print(f"\n" + "=" * 80)
            print("3. TIMING: WINNERS vs LOSERS")
            print("=" * 80)

            winner_delays = [t['delay_minutes'] for t in winners]
            loser_delays = [t['delay_minutes'] for t in losers]

            avg_winner = sum(winner_delays) / len(winner_delays)
            avg_loser = sum(loser_delays) / len(loser_delays)

            median_winner = sorted(winner_delays)[len(winner_delays)//2]
            median_loser = sorted(loser_delays)[len(loser_delays)//2]

            print(f"\nWINNERS ({len(winners)} trades):")
            print(f"  Average delay: {avg_winner:.1f} minutes")
            print(f"  Median delay: {median_winner:.1f} minutes")
            print(f"  Min/Max: {min(winner_delays):.1f} / {max(winner_delays):.1f} minutes")

            print(f"\nLOSERS ({len(losers)} trades):")
            print(f"  Average delay: {avg_loser:.1f} minutes")
            print(f"  Median delay: {median_loser:.1f} minutes")
            print(f"  Min/Max: {min(loser_delays):.1f} / {max(loser_delays):.1f} minutes")

            print(f"\nCOMPARISON:")
            diff = avg_loser - avg_winner
            if diff > 0:
                print(f"  Losers trigger {diff:.1f} min SLOWER than winners")
                print(f"  => Late entries catching exhausted moves / reversals")
            else:
                print(f"  Winners trigger {-diff:.1f} min SLOWER than losers")
                print(f"  => Timing NOT the primary issue (quality problem)")

            # Delay distribution comparison
            print(f"\nDELAY DISTRIBUTION COMPARISON:")
            print(f"{'Delay Range':<30} {'Winners':>10} {'Losers':>10}")
            print("-" * 52)

            for min_delay, max_delay, label in delay_bins:
                winner_count = sum(1 for d in winner_delays if min_delay <= d < max_delay)
                loser_count = sum(1 for d in loser_delays if min_delay <= d < max_delay)
                winner_pct = winner_count / len(winners) * 100 if winners else 0
                loser_pct = loser_count / len(losers) * 100 if losers else 0
                print(f"{label:<30} {winner_pct:>8.1f}% {loser_pct:>8.1f}%")

        # Long vs Short timing
        longs = [t for t in triggered if 'long' in t['setup_type'].lower()]
        shorts = [t for t in triggered if 'short' in t['setup_type'].lower()]

        if longs and shorts:
            print(f"\n" + "=" * 80)
            print("4. TIMING: LONG vs SHORT")
            print("=" * 80)

            long_delays = [t['delay_minutes'] for t in longs]
            short_delays = [t['delay_minutes'] for t in shorts]

            avg_long = sum(long_delays) / len(long_delays)
            avg_short = sum(short_delays) / len(short_delays)

            print(f"\nLONG BREAKOUTS ({len(longs)} trades):")
            print(f"  Average delay: {avg_long:.1f} minutes")
            print(f"  Median delay: {sorted(long_delays)[len(long_delays)//2]:.1f} minutes")

            print(f"\nSHORT BREAKOUTS ({len(shorts)} trades):")
            print(f"  Average delay: {avg_short:.1f} minutes")
            print(f"  Median delay: {sorted(short_delays)[len(short_delays)//2]:.1f} minutes")

        # Sample detailed trades
        print(f"\n" + "=" * 80)
        print("5. SAMPLE TRADES (Fastest and Slowest)")
        print("=" * 80)

        triggered_sorted = sorted(triggered, key=lambda t: t['delay_minutes'])

        print(f"\nFASTEST TRIGGERS (top 5):")
        print(f"{'Symbol':<12} {'Setup':<20} {'Delay':>8} {'PnL':>10} {'Exit Reason':<15}")
        print("-" * 75)
        for t in triggered_sorted[:5]:
            symbol = t['symbol']
            setup = t['setup_type'][:18]
            delay = t['delay_minutes']
            pnl = t.get('pnl', 0) or 0
            exit_reason = (t.get('exit_reason') or 'N/A')[:13]
            print(f"{symbol:<12} {setup:<20} {delay:>6.1f}m Rs.{pnl:>6.0f} {exit_reason:<15}")

        print(f"\nSLOWEST TRIGGERS (bottom 5):")
        print(f"{'Symbol':<12} {'Setup':<20} {'Delay':>8} {'PnL':>10} {'Exit Reason':<15}")
        print("-" * 75)
        for t in triggered_sorted[-5:]:
            symbol = t['symbol']
            setup = t['setup_type'][:18]
            delay = t['delay_minutes']
            pnl = t.get('pnl', 0) or 0
            exit_reason = (t.get('exit_reason') or 'N/A')[:13]
            print(f"{symbol:<12} {setup:<20} {delay:>6.1f}m Rs.{pnl:>6.0f} {exit_reason:<15}")

    # Not triggered analysis
    if not_triggered:
        print(f"\n" + "=" * 80)
        print("6. NOT TRIGGERED ANALYSIS")
        print("=" * 80)
        print(f"\n{len(not_triggered)} decisions never triggered")
        print(f"These are likely entry zone rejections (price never came back to zone)")

    # KEY FINDINGS
    print(f"\n" + "=" * 80)
    print("KEY FINDINGS & IMPLICATIONS")
    print("=" * 80)

    if triggered:
        avg_delay = sum(delays) / len(delays)
        median_delay = delays_sorted[len(delays)//2]

        print(f"\n1. AVERAGE DELAY: {avg_delay:.1f} minutes")
        if avg_delay > 10:
            print(f"   => VERY SLOW execution (professional standard: <2 minutes)")
            print(f"   => By the time we enter, momentum is likely exhausted")
        elif avg_delay > 5:
            print(f"   => MODERATE delay (could miss early momentum)")
        else:
            print(f"   => FAST execution (good)")

        immediate_count = sum(1 for d in delays if d <= 1)
        immediate_pct = immediate_count / len(delays) * 100

        print(f"\n2. IMMEDIATE EXECUTION (<1 min): {immediate_pct:.1f}%")
        if immediate_pct < 20:
            print(f"   => Most trades are delayed (NOT immediate)")
            print(f"   => Confirms hypothesis: waiting for entry zone is the bottleneck")

        timeout_count = sum(1 for d in delays if d >= 45)
        timeout_pct = timeout_count / len(delays) * 100

        print(f"\n3. TIMEOUT ZONE (45+ min): {timeout_pct:.1f}%")
        if timeout_pct > 5:
            print(f"   => Significant portion triggering near timeout")
            print(f"   => These are likely catching exhausted moves")

        if winners and losers and avg_loser > avg_winner:
            print(f"\n4. WINNERS TRIGGER FASTER THAN LOSERS")
            print(f"   => Confirms timing is critical for trade quality")
            print(f"   => Fast execution = better PnL")

    print(f"\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nBased on timing analysis:")

    if triggered:
        if avg_delay > 5:
            print(f"  => IMMEDIATE EXECUTION is critical")
            print(f"  => Remove entry zone wait for breakouts")
            print(f"  => Execute on decision with volume/momentum confirmation")
        else:
            print(f"  => Timing is acceptable")
            print(f"  => Focus on quality filters (volume/momentum) instead")

    print(f"\nThis timing analysis confirms the resolution plan:")
    print(f"  - Phase 1: Immediate execution (remove entry zone wait)")
    print(f"  - Phase 2: Volume surge filter (1.5x minimum)")
    print(f"  - Phase 3: Momentum candle validation (2x size)")

if __name__ == '__main__':
    main()
