"""
Check if ICT patterns are appearing in trade decisions.

Analyzes the most recent run to see if ICT setup types are making it
through structure detection -> gates -> decisions pipeline.
"""

import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]


def find_latest_run():
    """Find the most recent run directory."""
    logs_dir = ROOT / "logs"
    run_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

    if not run_dirs:
        return None

    # Sort by modification time
    latest = max(run_dirs, key=lambda d: d.stat().st_mtime)
    return latest


def analyze_decisions(run_dir):
    """Analyze decisions for ICT setup types."""
    decisions_file = run_dir / "events_decisions.jsonl"

    if not decisions_file.exists():
        print(f"[!] No events_decisions.jsonl found in {run_dir.name}")
        return

    ict_setup_types = {
        'order_block_long', 'order_block_short',
        'fair_value_gap_long', 'fair_value_gap_short',
        'liquidity_sweep_long', 'liquidity_sweep_short',
        'premium_zone_short', 'discount_zone_long',
        'break_of_structure_long', 'break_of_structure_short',
        'change_of_character_long', 'change_of_character_short',
    }

    total_decisions = 0
    ict_decisions = 0
    setup_type_counts = defaultdict(int)
    ict_examples = []

    with open(decisions_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            decision = json.loads(line)

            if decision.get('stage') != 'events_decision':
                continue

            total_decisions += 1
            strategy_type = decision.get('strategy_type', '')

            setup_type_counts[strategy_type] += 1

            if strategy_type in ict_setup_types:
                ict_decisions += 1
                if len(ict_examples) < 5:
                    ict_examples.append({
                        'symbol': decision.get('symbol'),
                        'strategy_type': strategy_type,
                        'action': decision.get('action'),
                        'timestamp': decision.get('timestamp'),
                        'score': decision.get('score')
                    })

    # Print results
    print("\n" + "="*80)
    print(f"DECISION ANALYSIS: {run_dir.name}")
    print("="*80 + "\n")

    print(f"Total decisions: {total_decisions}")
    print(f"ICT decisions: {ict_decisions} ({ict_decisions/total_decisions*100:.1f}%)" if total_decisions > 0 else "ICT decisions: 0")

    print("\n" + "="*80)
    print("SETUP TYPE BREAKDOWN")
    print("="*80 + "\n")

    # Sort by count descending
    sorted_types = sorted(setup_type_counts.items(), key=lambda x: x[1], reverse=True)

    for setup_type, count in sorted_types:
        pct = count / total_decisions * 100 if total_decisions > 0 else 0
        is_ict = " [ICT]" if setup_type in ict_setup_types else ""
        print(f"  {setup_type:40s} {count:4d} ({pct:5.1f}%){is_ict}")

    if ict_examples:
        print("\n" + "="*80)
        print("ICT DECISION EXAMPLES")
        print("="*80 + "\n")

        for ex in ict_examples:
            print(f"  {ex['symbol']:20s} {ex['strategy_type']:30s} {ex['action']:10s} score={ex.get('score', 'N/A')}")

    print("\n" + "="*80 + "\n")

    return {
        'total_decisions': total_decisions,
        'ict_decisions': ict_decisions,
        'setup_type_counts': dict(setup_type_counts)
    }


def analyze_events(run_dir):
    """Analyze DECISION events from events.jsonl for ICT patterns."""
    events_file = run_dir / "events.jsonl"

    if not events_file.exists():
        print(f"[!] No events.jsonl found in {run_dir.name}")
        return

    ict_setup_types = {
        'order_block_long', 'order_block_short',
        'fair_value_gap_long', 'fair_value_gap_short',
        'liquidity_sweep_long', 'liquidity_sweep_short',
        'premium_zone_short', 'discount_zone_long',
        'break_of_structure_long', 'break_of_structure_short',
        'change_of_character_long', 'change_of_character_short',
    }

    total_events = 0
    ict_events = 0
    setup_type_counts = defaultdict(int)

    with open(events_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            event = json.loads(line)

            if event.get('type') != 'DECISION':
                continue

            total_events += 1
            setup_type = event.get('decision', {}).get('setup_type', '')

            setup_type_counts[setup_type] += 1

            if setup_type in ict_setup_types:
                ict_events += 1

    print("\n" + "="*80)
    print(f"EVENTS.JSONL ANALYSIS: {run_dir.name}")
    print("="*80 + "\n")

    print(f"Total DECISION events: {total_events}")
    print(f"ICT DECISION events: {ict_events} ({ict_events/total_events*100:.1f}%)" if total_events > 0 else "ICT DECISION events: 0")

    if setup_type_counts:
        print("\nSetup types in DECISION events:")
        sorted_types = sorted(setup_type_counts.items(), key=lambda x: x[1], reverse=True)

        for setup_type, count in sorted_types[:10]:  # Top 10
            pct = count / total_events * 100 if total_events > 0 else 0
            is_ict = " [ICT]" if setup_type in ict_setup_types else ""
            print(f"  {setup_type:40s} {count:4d} ({pct:5.1f}%){is_ict}")

    print("\n" + "="*80 + "\n")


def main():
    """Main analysis."""
    print("\n" + "="*80)
    print("ICT PATTERN DECISION CHECKER")
    print("="*80 + "\n")

    latest_run = find_latest_run()

    if not latest_run:
        print("[!] No run directories found in logs/")
        return

    print(f"Analyzing: {latest_run.name}")
    print(f"Location: {latest_run}\n")

    # Check events_decisions.jsonl
    analyze_decisions(latest_run)

    # Check events.jsonl
    analyze_events(latest_run)


if __name__ == '__main__':
    main()
