#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze Decision → Trigger conversion rate

Simple analysis:
1. Count DECISION events by setup_type
2. Count TRIGGER events by setup_type
3. Calculate trigger rate
4. Find why breakouts have 2.8% rate vs failure_fade 35% rate
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
    print("DECISION → TRIGGER CONVERSION RATE ANALYSIS")
    print("=" * 80)

    # Track by setup_type
    decisions_by_setup = defaultdict(int)
    triggers_by_setup = defaultdict(int)
    triggered_trade_ids = set()

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

                    if event.get('type') == 'DECISION':
                        setup_type = event.get('decision', {}).get('setup_type', 'unknown')
                        decisions_by_setup[setup_type] += 1

                    elif event.get('type') == 'TRIGGER':
                        trade_id = event.get('trade_id')
                        triggered_trade_ids.add(trade_id)
                        setup_type = event.get('trigger', {}).get('strategy', 'unknown')
                        triggers_by_setup[setup_type] += 1

    print(f"Loaded {sum(decisions_by_setup.values())} decisions, {sum(triggers_by_setup.values())} triggers")

    # Calculate trigger rates
    print(f"\n" + "=" * 80)
    print("TRIGGER RATE BY SETUP TYPE")
    print("=" * 80)

    print(f"\n{'Setup Type':<30} {'Decisions':>10} {'Triggers':>10} {'Rate':>8} {'Lost':>8}")
    print("-" * 80)

    all_setups = set(list(decisions_by_setup.keys()) + list(triggers_by_setup.keys()))

    # Sort by decision count
    setup_list = sorted(all_setups, key=lambda x: decisions_by_setup.get(x, 0), reverse=True)

    for setup in setup_list:
        decisions = decisions_by_setup.get(setup, 0)
        triggers = triggers_by_setup.get(setup, 0)
        rate = (triggers / decisions * 100) if decisions > 0 else 0
        lost = decisions - triggers

        # Highlight breakouts vs fades
        marker = ""
        if 'breakout' in setup:
            marker = " ⚠️ LOW" if rate < 10 else ""
        elif 'fade' in setup:
            marker = " ✓ GOOD" if rate > 20 else ""

        print(f"{setup:<30} {decisions:>10} {triggers:>10} {rate:>7.1f}% {lost:>8}{marker}")

    # Summary stats
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    breakout_decisions = sum(v for k, v in decisions_by_setup.items() if 'breakout' in k)
    breakout_triggers = sum(v for k, v in triggers_by_setup.items() if 'breakout' in k)
    breakout_rate = (breakout_triggers / breakout_decisions * 100) if breakout_decisions > 0 else 0

    fade_decisions = sum(v for k, v in decisions_by_setup.items() if 'fade' in k)
    fade_triggers = sum(v for k, v in triggers_by_setup.items() if 'fade' in k)
    fade_rate = (fade_triggers / fade_decisions * 100) if fade_decisions > 0 else 0

    print(f"\nBREAKOUT strategies:")
    print(f"  Decisions: {breakout_decisions}")
    print(f"  Triggers: {breakout_triggers}")
    print(f"  Rate: {breakout_rate:.1f}%")
    print(f"  Lost: {breakout_decisions - breakout_triggers}")

    print(f"\nFAILURE_FADE strategies:")
    print(f"  Decisions: {fade_decisions}")
    print(f"  Triggers: {fade_triggers}")
    print(f"  Rate: {fade_rate:.1f}%")
    print(f"  Lost: {fade_decisions - fade_triggers}")

    print(f"\nGAP ANALYSIS:")
    print(f"  Fade is {fade_rate / breakout_rate:.1f}x better than breakout")
    print(f"  If breakout matched fade rate ({fade_rate:.1f}%):")
    print(f"    Expected triggers: {int(breakout_decisions * fade_rate / 100)}")
    print(f"    Potential gain: +{int(breakout_decisions * fade_rate / 100) - breakout_triggers} trades")

if __name__ == '__main__':
    main()
