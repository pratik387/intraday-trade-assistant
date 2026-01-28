#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze which structure detectors are actually finding setups in the backtest.

Checks:
1. Which detectors are enabled in config (37 expected)
2. Which detectors are actually finding setups in decisions
3. Mapping between detector names and setup_type names
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251113-115738_extracted\20251113-115738_full\20251113-115738")
CONFIG_FILE = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\config\configuration.json")

def main():
    print("=" * 80)
    print("STRUCTURE DETECTOR ACTIVITY ANALYSIS")
    print("=" * 80)

    # Load config to see which setups are enabled
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    setups_config = config.get('setups', {})
    enabled_setups = [name for name, cfg in setups_config.items() if cfg.get('enabled', False)]

    print(f"\n1. CONFIGURATION:")
    print(f"   Total setups defined: {len(setups_config)}")
    print(f"   Enabled setups: {len(enabled_setups)}")
    print(f"\nEnabled setup names:")
    for name in sorted(enabled_setups):
        print(f"   - {name}")

    # Load all decisions from backtest
    all_decisions = []

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get('type') == 'DECISION':
                            all_decisions.append(event)

    print(f"\n2. BACKTEST DECISIONS:")
    print(f"   Total decisions: {len(all_decisions)}")

    # Analyze setup_type distribution
    setup_type_counts = defaultdict(int)
    for d in all_decisions:
        setup_type = d.get('decision', {}).get('setup_type', 'unknown')
        setup_type_counts[setup_type] += 1

    print(f"\n3. SETUP TYPES IN DECISIONS:")
    print(f"   Unique setup types: {len(setup_type_counts)}")
    print(f"\n{'Setup Type':<30} {'Count':>10} {'%':>8}")
    print("-" * 80)

    for setup_type, count in sorted(setup_type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(all_decisions) * 100
        print(f"{setup_type:<30} {count:>10} {pct:>7.1f}%")

    # Analyze detector names from reasons
    detector_counts = defaultdict(int)
    for d in all_decisions:
        reasons = d.get('decision', {}).get('reasons', '')
        # Parse detector name from reasons (format: "structure:detector:failure_fade_long")
        if 'structure:detector:' in reasons:
            for part in reasons.split(';'):
                part = part.strip()
                if 'structure:detector:' in part:
                    detector_name = part.split('structure:detector:')[1].strip()
                    detector_counts[detector_name] += 1

    print(f"\n4. DETECTOR NAMES IN DECISIONS (from reasons):")
    print(f"   Detectors that found setups: {len(detector_counts)}")
    print(f"\n{'Detector Name':<30} {'Count':>10} {'%':>8}")
    print("-" * 80)

    for detector, count in sorted(detector_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(all_decisions) * 100
        print(f"{detector:<30} {count:>10} {pct:>7.1f}%")

    # Find missing detectors
    print(f"\n5. MISSING DETECTORS:")
    print(f"   Enabled but not producing decisions:")

    # Map setup names to detector keys (from main_detector.py)
    setup_to_detector = {
        "level_breakout_long": "level_breakout_long",
        "level_breakout_short": "level_breakout_short",
        "failure_fade_long": "failure_fade_long",
        "failure_fade_short": "failure_fade_short",
        "squeeze_release_long": "squeeze_release_long",
        "squeeze_release_short": "squeeze_release_short",
        "flag_continuation_long": "flag_continuation_long",
        "flag_continuation_short": "flag_continuation_short",
        "momentum_breakout_long": "momentum_breakout_long",
        "momentum_breakout_short": "momentum_breakout_short",
        "vwap_reclaim_long": "vwap_reclaim_long",
        "vwap_lose_short": "vwap_lose_short",
        "gap_fill_long": "gap_fill_long",
        "gap_fill_short": "gap_fill_short",
        "orb_breakout_long": "orb_breakout_long",
        "orb_breakout_short": "orb_breakout_short",
        "orb_breakdown_short": "orb_breakdown_short",
        "orb_breakout": "orb_breakout",
        "orb_breakdown": "orb_breakdown",
        "orb_pullback_long": "orb_pullback_long",
        "orb_pullback_short": "orb_pullback_short",
        "support_bounce_long": "support_bounce_long",
        "resistance_bounce_short": "resistance_bounce_short",
        "trend_pullback_long": "trend_pullback_long",
        "trend_pullback_short": "trend_pullback_short",
        "volume_spike_reversal_long": "volume_spike_reversal_long",
        "volume_spike_reversal_short": "volume_spike_reversal_short",
        "range_rejection_long": "range_rejection_long",
        "range_rejection_short": "range_rejection_short",
        "vwap_mean_reversion_long": "vwap_mean_reversion_long",
        "vwap_mean_reversion_short": "vwap_mean_reversion_short",
        "support_breakdown_short": "support_breakdown_short",
        "resistance_breakout_long": "resistance_breakout_long",
        "range_breakout_long": "range_breakout_long",
        "range_breakdown_short": "range_breakdown_short",
        "range_bounce_long": "range_bounce_long",
        "range_bounce_short": "range_bounce_short",
        "trend_continuation_long": "trend_continuation_long",
        "trend_continuation_short": "trend_continuation_short",
        "ict_comprehensive": "ict",
    }

    active_detectors = set(detector_counts.keys())

    missing_count = 0
    for setup_name in sorted(enabled_setups):
        detector_key = setup_to_detector.get(setup_name, setup_name)
        if detector_key not in active_detectors:
            missing_count += 1
            print(f"   - {setup_name} (detector: {detector_key})")

    print(f"\n   Total missing: {missing_count}/{len(enabled_setups)} enabled setups")

    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Configuration: {len(enabled_setups)} setups enabled")
    print(f"✓ Decisions: {len(all_decisions)} decisions made")
    print(f"✓ Setup types appearing: {len(setup_type_counts)}")
    print(f"✓ Detectors producing decisions: {len(detector_counts)}")
    print(f"✗ Missing detectors: {missing_count} ({missing_count/len(enabled_setups)*100:.1f}%)")

    print(f"\n⚠️ GAP IDENTIFIED:")
    print(f"   {len(enabled_setups)} setups enabled in config")
    print(f"   {len(detector_counts)} detectors actually producing decisions")
    print(f"   {missing_count} detectors not producing any decisions")

if __name__ == '__main__':
    main()
