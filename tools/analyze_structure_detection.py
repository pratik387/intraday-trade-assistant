"""
Analyze which structures are being detected vs configured in the backtest.
"""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

def analyze_structure_detection(sessions_root, config_path):
    """Analyze structure detection across all sessions."""

    sessions_path = Path(sessions_root)

    print("=" * 80)
    print("STRUCTURE DETECTION ANALYSIS")
    print("=" * 80)
    print()

    # 1. Get enabled structures from config
    print("1. ENABLED STRUCTURES IN CONFIGURATION")
    print("=" * 80)

    with open(config_path) as f:
        config = json.load(f)

    enabled_structures = {}
    setups = config.get('setups', {})
    for name, setup_config in setups.items():
        if setup_config.get('enabled', False):
            enabled_structures[name] = setup_config

    print(f"Total Enabled: {len(enabled_structures)}")
    print()

    # 2. Analyze detection across all sessions
    print("2. DETECTED STRUCTURES ACROSS ALL SESSIONS")
    print("=" * 80)

    all_structures_detected = Counter()
    structures_by_session = defaultdict(list)
    total_decisions = 0
    sessions_processed = 0

    for session_dir in sorted(sessions_path.iterdir()):
        if not session_dir.is_dir():
            continue

        events_file = session_dir / "events.jsonl"
        if not events_file.exists():
            continue

        sessions_processed += 1
        session_structures = set()

        with open(events_file) as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event.get('type') == 'DECISION':
                        total_decisions += 1
                        setup_type = event.get('setup_type')
                        if setup_type:
                            all_structures_detected[setup_type] += 1
                            session_structures.add(setup_type)
                except json.JSONDecodeError:
                    continue

        if session_structures:
            structures_by_session[session_dir.name] = list(session_structures)

    print(f"Sessions Processed: {sessions_processed}")
    print(f"Total Decisions: {total_decisions}")
    print(f"Unique Structures Detected: {len(all_structures_detected)}")
    print()

    if all_structures_detected:
        print("Detection Breakdown:")
        for structure, count in all_structures_detected.most_common():
            pct = (count / total_decisions * 100) if total_decisions > 0 else 0
            print(f"  {structure}: {count} ({pct:.1f}%)")
    else:
        print("  NO STRUCTURES DETECTED!")

    print()

    # 3. Compare enabled vs detected
    print("3. ENABLED BUT NOT DETECTING")
    print("=" * 80)

    enabled_names = set(enabled_structures.keys())
    detected_names = set(all_structures_detected.keys())

    never_detected = enabled_names - detected_names

    if never_detected:
        print(f"Found {len(never_detected)} structures that never detected:")
        print()

        # Group by setup type
        by_category = defaultdict(list)
        for name in sorted(never_detected):
            if 'orb' in name.lower():
                by_category['ORB'].append(name)
            elif 'squeeze' in name.lower():
                by_category['Squeeze'].append(name)
            elif 'vwap' in name.lower():
                by_category['VWAP'].append(name)
            elif 'level' in name.lower() or 'resistance' in name.lower() or 'support' in name.lower():
                by_category['Level/Support/Resistance'].append(name)
            elif 'range' in name.lower():
                by_category['Range'].append(name)
            elif 'flag' in name.lower() or 'continuation' in name.lower():
                by_category['Continuation'].append(name)
            elif 'volume' in name.lower():
                by_category['Volume'].append(name)
            elif 'momentum' in name.lower():
                by_category['Momentum'].append(name)
            else:
                by_category['Other'].append(name)

        for category in sorted(by_category.keys()):
            structures = by_category[category]
            print(f"{category} ({len(structures)} structures):")
            for s in structures:
                print(f"  - {s}")
            print()
    else:
        print("All enabled structures detected at least once!")

    print()

    # 4. Scanner analysis
    print("4. SCANNER BOTTLENECK ANALYSIS")
    print("=" * 80)

    # Check a few agent logs for scanner stats
    sample_sessions = list(sorted(sessions_path.iterdir()))[:5]

    total_shortlisted = 0
    total_symbols = 0
    bars_checked = 0

    for session_dir in sample_sessions:
        if not session_dir.is_dir():
            continue

        agent_log = session_dir / "agent.log"
        if not agent_log.exists():
            continue

        with open(agent_log) as f:
            for line in f:
                if "SCANNER_COMPLETE" in line and "shortlisted" in line:
                    bars_checked += 1
                    # Parse: "Processed X eligible of Y total symbols → Z shortlisted"
                    try:
                        parts = line.split("→")
                        if len(parts) >= 2:
                            shortlist_part = parts[1].split()[0]
                            total_shortlisted += int(shortlist_part)

                        before_arrow = parts[0]
                        if "of" in before_arrow:
                            total_part = before_arrow.split("of")[1].split()[0]
                            total_symbols = int(total_part)
                    except (ValueError, IndexError):
                        pass

    if bars_checked > 0:
        avg_shortlist = total_shortlisted / bars_checked if bars_checked > 0 else 0
        print(f"Sample Analysis (first 5 sessions, {bars_checked} bars):")
        print(f"  Avg shortlist size: {avg_shortlist:.1f} symbols/bar")
        print(f"  Total symbols available: {total_symbols}")
        print(f"  Shortlist rate: {avg_shortlist/total_symbols*100:.2f}%")
        print()

        if avg_shortlist < 50:
            print("  WARNING: Low shortlist rate! Scanner may be too strict.")
            print("  Recommendation: Review energy scanner thresholds")
        else:
            print("  OK: Scanner is passing symbols to structure detection")

    print()
    print("=" * 80)

    # 5. Recommendations
    print()
    print("5. RECOMMENDATIONS")
    print("=" * 80)

    if len(never_detected) > 25:
        print("CRITICAL: 25+ structures never detecting!")
        print()
        print("Likely causes:")
        print("  1. Time window restrictions (e.g., ORB needs 9:30 AM, system starts 10:15 AM)")
        print("  2. Scanner filtering too strict (symbols filtered before structures checked)")
        print("  3. Market conditions don't match structure requirements")
        print("  4. Structure detection thresholds too strict")
        print()
        print("Action items:")
        print("  [ ] Check time_policy settings (especially for ORB)")
        print("  [ ] Review energy_scanner thresholds")
        print("  [ ] Check individual structure min thresholds")
        print("  [ ] Consider relaxing parameters for low-volatility periods")

    if avg_shortlist < 50:
        print()
        print("Scanner bottleneck detected:")
        print(f"  Only {avg_shortlist:.1f} symbols/bar pass scanner")
        print("  Action: Review energy_scanner config for overly strict filters")

    print()
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python analyze_structure_detection.py <sessions_root> <config_file>")
        sys.exit(1)

    sessions_root = sys.argv[1]
    config_file = sys.argv[2]

    if not Path(sessions_root).exists():
        print(f"Error: Sessions directory not found: {sessions_root}")
        sys.exit(1)

    if not Path(config_file).exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)

    analyze_structure_detection(sessions_root, config_file)
