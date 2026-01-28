#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detector Coverage Analysis by Regime

Analyzes which detectors are firing decisions vs which are actually executing trades,
broken down by detected regime.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251114-125524_extracted\20251114-125524_full\20251114-125524")

def main():
    print("=" * 80)
    print("DETECTOR COVERAGE ANALYSIS BY REGIME")
    print("=" * 80)

    # Collect DECISIONS by regime and detector
    decisions_by_regime = defaultdict(lambda: defaultdict(int))
    trades_by_regime = defaultdict(lambda: defaultdict(lambda: {
        'trades': 0,
        'winners': 0,
        'pnl': 0.0
    }))

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        # Read events.jsonl for DECISION events
        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get('type') == 'DECISION':
                            decision = event.get('decision', {})
                            regime = decision.get('regime', 'unknown')
                            setup_type = decision.get('setup_type', 'unknown')

                            # Extract detector from reasons
                            reasons = decision.get('reasons', '')
                            detector = 'unknown'
                            if 'structure:detector:' in reasons:
                                for part in reasons.split(';'):
                                    if 'structure:detector:' in part:
                                        detector = part.split('structure:detector:')[1].strip()
                                        break

                            decisions_by_regime[regime][detector] += 1

        # Read analytics.jsonl for EXECUTED trades
        analytics_file = session_dir / "analytics.jsonl"
        if analytics_file.exists():
            with open(analytics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        trade = json.loads(line)
                        regime = trade.get('regime', 'unknown')
                        setup_type = trade.get('setup_type', 'unknown')
                        pnl = trade.get('pnl', 0)

                        # Infer detector from setup type
                        if 'breakout' in setup_type:
                            detector = 'level_breakout_long'
                        elif 'failure_fade' in setup_type:
                            detector = 'failure_fade_long'
                        elif 'trend_pullback' in setup_type:
                            detector = 'trend_pullback_long'
                        elif 'trend_continuation' in setup_type:
                            detector = 'trend_continuation_long'
                        else:
                            detector = 'unknown'

                        trades_by_regime[regime][detector]['trades'] += 1
                        trades_by_regime[regime][detector]['pnl'] += pnl
                        if pnl > 0:
                            trades_by_regime[regime][detector]['winners'] += 1

    print(f"\nAnalyzed {sum(sum(d.values()) for d in decisions_by_regime.values())} total decisions")
    print(f"Resulted in {sum(sum(t['trades'] for t in d.values()) for d in trades_by_regime.values())} trades")

    # Analysis by regime
    print(f"\n" + "=" * 80)
    print("DECISIONS vs TRADES BY REGIME AND DETECTOR")
    print("=" * 80)

    for regime in sorted(decisions_by_regime.keys()):
        print(f"\n{regime.upper()} REGIME:")
        print(f"{'  Detector':<35} {'Decisions':>10} {'Trades':>8} {'Trigger%':>10} {'Win%':>8} {'PnL':>12}")
        print("  " + "-" * 88)

        total_decisions = sum(decisions_by_regime[regime].values())
        total_trades = sum(t['trades'] for t in trades_by_regime[regime].values())

        # Sort by decisions (most active detectors first)
        for detector in sorted(decisions_by_regime[regime].keys(),
                              key=lambda d: decisions_by_regime[regime][d],
                              reverse=True):
            decisions = decisions_by_regime[regime][detector]

            if detector in trades_by_regime[regime]:
                trade_stats = trades_by_regime[regime][detector]
                trades = trade_stats['trades']
                winners = trade_stats['winners']
                pnl = trade_stats['pnl']
                trigger_rate = (trades / decisions * 100) if decisions > 0 else 0
                win_rate = (winners / trades * 100) if trades > 0 else 0
            else:
                trades = 0
                trigger_rate = 0
                win_rate = 0
                pnl = 0

            # Mark poor trigger rates
            marker = ""
            if trigger_rate == 0 and decisions >= 10:
                marker = " [NO TRADES!]"
            elif trigger_rate < 5 and decisions >= 10:
                marker = " [LOW TRIGGER]"
            elif trigger_rate >= 30:
                marker = " [HIGH TRIGGER]"

            print(f"  {detector:<35} {decisions:>10} {trades:>8} {trigger_rate:>9.1f}% {win_rate:>7.1f}% Rs.{pnl:>9,.0f}{marker}")

        print(f"  {'-'*35} {total_decisions:>10} {total_trades:>8}")

    # Summary statistics
    print(f"\n" + "=" * 80)
    print("SUMMARY: MISSING DETECTORS BY REGIME")
    print("=" * 80)

    for regime in sorted(decisions_by_regime.keys()):
        print(f"\n{regime.upper()}:")

        # Find detectors with decisions but no trades
        no_trades = []
        low_trigger = []

        for detector, decisions in decisions_by_regime[regime].items():
            if decisions >= 5:  # Only consider detectors with meaningful decision volume
                trades = trades_by_regime[regime][detector]['trades'] if detector in trades_by_regime[regime] else 0
                trigger_rate = (trades / decisions * 100) if decisions > 0 else 0

                if trades == 0:
                    no_trades.append((detector, decisions))
                elif trigger_rate < 5:
                    low_trigger.append((detector, decisions, trades, trigger_rate))

        if no_trades:
            print(f"  Detectors with ZERO trades (despite decisions):")
            for detector, decisions in sorted(no_trades, key=lambda x: x[1], reverse=True):
                print(f"    - {detector}: {decisions} decisions, 0 trades (0.0% trigger)")

        if low_trigger:
            print(f"  Detectors with LOW trigger rate (<5%):")
            for detector, decisions, trades, trigger_rate in sorted(low_trigger, key=lambda x: x[3]):
                print(f"    - {detector}: {decisions} decisions, {trades} trades ({trigger_rate:.1f}% trigger)")

    # Regime diversity analysis
    print(f"\n" + "=" * 80)
    print("REGIME DIVERSITY ANALYSIS")
    print("=" * 80)

    for regime in sorted(decisions_by_regime.keys()):
        total_decisions = sum(decisions_by_regime[regime].values())
        total_trades = sum(t['trades'] for t in trades_by_regime[regime].values())

        unique_detectors_deciding = len(decisions_by_regime[regime])
        unique_detectors_trading = len([d for d in trades_by_regime[regime] if trades_by_regime[regime][d]['trades'] > 0])

        print(f"\n{regime.upper()}:")
        print(f"  Unique detectors making decisions: {unique_detectors_deciding}")
        print(f"  Unique detectors executing trades: {unique_detectors_trading}")
        print(f"  Decision diversity: {unique_detectors_deciding} detectors")
        print(f"  Trade diversity: {unique_detectors_trading} detectors")

        if unique_detectors_deciding > unique_detectors_trading:
            print(f"  [WARNING] {unique_detectors_deciding - unique_detectors_trading} detectors not producing trades!")

    # Top rejection reasons by regime
    print(f"\n" + "=" * 80)
    print("CRITICAL INSIGHTS")
    print("=" * 80)

    print(f"\nMost Active Detectors (by decisions):")
    all_detector_decisions = defaultdict(int)
    all_detector_trades = defaultdict(int)

    for regime in decisions_by_regime:
        for detector, count in decisions_by_regime[regime].items():
            all_detector_decisions[detector] += count

    for regime in trades_by_regime:
        for detector, stats in trades_by_regime[regime].items():
            all_detector_trades[detector] += stats['trades']

    for detector in sorted(all_detector_decisions.keys(),
                          key=lambda d: all_detector_decisions[d],
                          reverse=True):
        decisions = all_detector_decisions[detector]
        trades = all_detector_trades.get(detector, 0)
        trigger_rate = (trades / decisions * 100) if decisions > 0 else 0

        marker = ""
        if trades == 0:
            marker = " [BLOCKED]"
        elif trigger_rate < 5:
            marker = " [MOSTLY BLOCKED]"

        print(f"  {detector:<35}: {decisions:>4} decisions → {trades:>3} trades ({trigger_rate:5.1f}% trigger){marker}")

if __name__ == '__main__':
    main()
