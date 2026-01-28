#!/usr/bin/env python
"""
Experiment Registry Manager

Simple tool to view, update, and compare experiments tracked in experiments_registry.json

Usage:
    python tools/experiments.py list                    # List all experiments
    python tools/experiments.py show baseline-001       # Show details of experiment
    python tools/experiments.py compare baseline-001 exit-opt-001   # Compare two experiments
    python tools/experiments.py update exit-opt-001 --pnl 15234 --win-rate 0.52   # Update results
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

REGISTRY_FILE = Path(__file__).parent.parent / "experiments_registry.json"


def load_registry() -> Dict[str, Any]:
    """Load experiments registry"""
    if not REGISTRY_FILE.exists():
        print(f"Error: Registry file not found: {REGISTRY_FILE}")
        sys.exit(1)

    with open(REGISTRY_FILE) as f:
        return json.load(f)


def save_registry(registry: Dict[str, Any]) -> None:
    """Save experiments registry"""
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"✓ Registry updated: {REGISTRY_FILE}")


def list_experiments() -> None:
    """List all experiments"""
    registry = load_registry()

    print("\n" + "=" * 80)
    print("EXPERIMENTS REGISTRY")
    print("=" * 80)

    for exp in registry['experiments']:
        exp_id = exp['experiment_id']
        status = exp['status']
        date = exp.get('date', 'N/A')
        pnl = exp.get('results', {}).get('total_pnl', 'TBD')

        status_icon = "[OK]" if status == "completed" else "[PLANNED]" if status == "planned" else "[RUNNING]"

        print(f"\n{status_icon} {exp_id} ({status})")
        print(f"   Date: {date}")
        print(f"   Run ID: {exp.get('run_id', 'N/A')}")
        print(f"   P&L: Rs.{pnl if isinstance(pnl, (int, float)) else pnl}")
        print(f"   Description: {exp.get('description', 'N/A')}")

    print("\n" + "=" * 80 + "\n")


def show_experiment(exp_id: str) -> None:
    """Show details of a specific experiment"""
    registry = load_registry()

    exp = None
    for e in registry['experiments']:
        if e['experiment_id'] == exp_id:
            exp = e
            break

    if not exp:
        print(f"Error: Experiment '{exp_id}' not found")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {exp_id}")
    print("=" * 80)

    print(f"\nStatus: {exp['status']}")
    print(f"Date: {exp.get('date', 'N/A')}")
    print(f"Run ID: {exp.get('run_id', 'N/A')}")
    print(f"Type: {exp.get('type', 'N/A')}")
    print(f"Archive: {exp.get('backtest_archive', 'N/A')}")

    print(f"\nDescription:")
    print(f"  {exp.get('description', 'N/A')}")

    if 'hypothesis' in exp:
        print(f"\nHypothesis:")
        print(f"  {exp['hypothesis']}")

    # Config
    if 'config' in exp:
        print(f"\nConfiguration:")
        for key, value in exp['config'].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    # Results
    if 'results' in exp:
        print(f"\nResults:")
        for key, value in exp['results'].items():
            if key == 'note':
                continue
            print(f"  {key}: {value}")

        if 'note' in exp['results']:
            print(f"  Note: {exp['results']['note']}")

    # Key findings
    if 'key_findings' in exp:
        print(f"\nKey Findings:")
        for finding in exp['key_findings']:
            print(f"  • {finding}")

    # Analysis reports
    if 'analysis_reports' in exp:
        print(f"\nAnalysis Reports:")
        for report in exp['analysis_reports']:
            print(f"  • {report}")

    print("\n" + "=" * 80 + "\n")


def compare_experiments(exp_id1: str, exp_id2: str) -> None:
    """Compare two experiments side-by-side"""
    registry = load_registry()

    exp1 = None
    exp2 = None
    for e in registry['experiments']:
        if e['experiment_id'] == exp_id1:
            exp1 = e
        if e['experiment_id'] == exp_id2:
            exp2 = e

    if not exp1:
        print(f"Error: Experiment '{exp_id1}' not found")
        sys.exit(1)
    if not exp2:
        print(f"Error: Experiment '{exp_id2}' not found")
        sys.exit(1)

    print("\n" + "=" * 100)
    print(f"COMPARISON: {exp_id1} vs {exp_id2}")
    print("=" * 100)

    # Results comparison
    metrics = [
        ('total_trades', 'Total Trades'),
        ('total_pnl', 'Total P&L (Rs.)'),
        ('win_rate', 'Win Rate'),
        ('stop_loss_hit_rate', 'SL Hit Rate'),
        ('stop_loss_pnl', 'SL P&L (Rs.)'),
    ]

    print(f"\n{'Metric':<25} {exp_id1:<25} {exp_id2:<25} {'Diff':<20}")
    print("-" * 100)

    for metric_key, metric_name in metrics:
        val1 = exp1.get('results', {}).get(metric_key, 'N/A')
        val2 = exp2.get('results', {}).get(metric_key, 'N/A')

        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            diff_pct = (diff / val1 * 100) if val1 != 0 else 0
            diff_str = f"{diff:+.2f} ({diff_pct:+.1f}%)"
        else:
            diff_str = "N/A"

        print(f"{metric_name:<25} {str(val1):<25} {str(val2):<25} {diff_str:<20}")

    print("\n" + "=" * 100 + "\n")


def update_experiment(exp_id: str, updates: Dict[str, Any]) -> None:
    """Update experiment results"""
    registry = load_registry()

    exp_idx = None
    for i, e in enumerate(registry['experiments']):
        if e['experiment_id'] == exp_id:
            exp_idx = i
            break

    if exp_idx is None:
        print(f"Error: Experiment '{exp_id}' not found")
        sys.exit(1)

    # Apply updates
    if 'results' not in registry['experiments'][exp_idx]:
        registry['experiments'][exp_idx]['results'] = {}

    for key, value in updates.items():
        registry['experiments'][exp_idx]['results'][key] = value
        print(f"✓ Updated {key} = {value}")

    # Update status if results are complete
    if all(k in registry['experiments'][exp_idx]['results'] for k in ['total_trades', 'total_pnl', 'win_rate']):
        registry['experiments'][exp_idx]['status'] = 'completed'
        print(f"✓ Status updated to 'completed'")

    save_registry(registry)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == 'list':
        list_experiments()

    elif command == 'show':
        if len(sys.argv) < 3:
            print("Usage: python tools/experiments.py show <experiment_id>")
            sys.exit(1)
        show_experiment(sys.argv[2])

    elif command == 'compare':
        if len(sys.argv) < 4:
            print("Usage: python tools/experiments.py compare <exp_id1> <exp_id2>")
            sys.exit(1)
        compare_experiments(sys.argv[2], sys.argv[3])

    elif command == 'update':
        if len(sys.argv) < 4:
            print("Usage: python tools/experiments.py update <exp_id> --key value [--key2 value2 ...]")
            sys.exit(1)

        exp_id = sys.argv[2]
        updates = {}

        i = 3
        while i < len(sys.argv):
            if sys.argv[i].startswith('--'):
                key = sys.argv[i][2:].replace('-', '_')
                if i + 1 < len(sys.argv):
                    value_str = sys.argv[i + 1]
                    # Try to parse as number
                    try:
                        value = float(value_str)
                        if value.is_integer():
                            value = int(value)
                    except ValueError:
                        value = value_str
                    updates[key] = value
                    i += 2
                else:
                    print(f"Error: --{key} requires a value")
                    sys.exit(1)
            else:
                i += 1

        update_experiment(exp_id, updates)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
