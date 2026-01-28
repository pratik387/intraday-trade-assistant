"""
Analyze ICT pattern detection after quality filter implementation.

Check:
1. Are ICT patterns being detected at all?
2. In which regimes are they being detected?
3. What's the conversion rate (detection -> decision -> trade)?
4. Are the quality filters too strict or just right?
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_ict_detection(backtest_dir: str):
    """Analyze ICT pattern detection across all sessions."""

    base_path = Path(backtest_dir)

    ict_patterns = [
        'order_block_long',
        'order_block_short',
        'fair_value_gap_long',
        'fair_value_gap_short',
        'break_of_structure_long',
        'break_of_structure_short',
    ]

    # Track detections, decisions, and trades
    detections = defaultdict(lambda: defaultdict(int))  # pattern -> regime -> count
    decisions = defaultdict(lambda: defaultdict(int))
    trades = defaultdict(lambda: defaultdict(int))
    trade_pnl = defaultdict(lambda: defaultdict(float))

    print("="*80)
    print("ICT PATTERN DETECTION ANALYSIS (Post Quality Filters)")
    print("="*80)

    session_count = 0
    for session_dir in sorted(base_path.iterdir()):
        if not session_dir.is_dir():
            continue

        session_count += 1

        # Count trades by pattern and regime
        analytics_file = session_dir / 'analytics.jsonl'
        if analytics_file.exists():
            with open(analytics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        trade = json.loads(line.strip())
                        setup = trade.get('setup_type')
                        if setup in ict_patterns:
                            regime = trade.get('regime', 'unknown')
                            trades[setup][regime] += 1
                            trade_pnl[setup][regime] += trade.get('pnl', 0)
                    except:
                        continue

        # Count decisions by pattern
        events_dec_file = session_dir / 'events_decisions.jsonl'
        if events_dec_file.exists():
            with open(events_dec_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('action') == 'accept':
                            setup = event.get('strategy_type')
                            if setup in ict_patterns:
                                # Don't have regime in decision, use 'all'
                                decisions[setup]['all'] += 1
                    except:
                        continue

        # Count detections (DECISION events in events.jsonl)
        events_file = session_dir / 'events.jsonl'
        if events_file.exists():
            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('type') == 'DECISION':
                            data = event.get('data', {})
                            setup = data.get('strategy_type')
                            if setup in ict_patterns:
                                detections[setup]['all'] += 1
                    except:
                        continue

    print(f"\nAnalyzed {session_count} sessions\n")

    # Print results
    for pattern in ict_patterns:
        total_detections = sum(detections[pattern].values())
        total_decisions = sum(decisions[pattern].values())
        total_trades = sum(trades[pattern].values())

        if total_detections == 0 and total_trades == 0:
            continue  # Skip patterns with no activity

        print(f"\n{pattern.upper()}")
        print(f"{'='*80}")

        print(f"\nFUNNEL:")
        print(f"  Detections:  {total_detections:>4} (patterns found)")
        print(f"  Decisions:   {total_decisions:>4} (passed planning)")
        print(f"  Trades:      {total_trades:>4} (triggered)")

        if total_detections > 0:
            decision_rate = total_decisions / total_detections * 100
            print(f"  Detection->Decision: {decision_rate:>5.1f}%")

        if total_decisions > 0:
            trigger_rate = total_trades / total_decisions * 100
            print(f"  Decision->Trade:     {trigger_rate:>5.1f}%")

        if total_detections > 0:
            overall_conversion = total_trades / total_detections * 100
            print(f"  Overall Conversion:  {overall_conversion:>5.1f}%")

        # Regime breakdown for trades
        if total_trades > 0:
            print(f"\nTRADES BY REGIME:")
            for regime, count in sorted(trades[pattern].items()):
                pnl = trade_pnl[pattern][regime]
                pct = count / total_trades * 100
                print(f"  {regime:<15} {count:>3} trades ({pct:>5.1f}%)  P&L: Rs {pnl:>10,.2f}")

            total_pnl = sum(trade_pnl[pattern].values())
            print(f"\n  TOTAL P&L: Rs {total_pnl:>10,.2f}")

    print(f"\n{'='*80}\n")

    # Summary
    print("SUMMARY:")
    print("-"*80)

    total_ict_detections = sum(sum(d.values()) for d in detections.values())
    total_ict_trades = sum(sum(t.values()) for t in trades.values())
    total_ict_pnl = sum(sum(p.values()) for p in trade_pnl.values())

    print(f"\nAll ICT Patterns Combined:")
    print(f"  Total Detections: {total_ict_detections}")
    print(f"  Total Trades: {total_ict_trades}")
    print(f"  Total P&L: Rs {total_ict_pnl:,.2f}")

    if total_ict_detections > 0:
        overall_conversion = total_ict_trades / total_ict_detections * 100
        print(f"  Overall Conversion: {overall_conversion:.1f}%")

    # Diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS:")
    print("-"*80)

    if total_ict_detections == 0:
        print("\nWARNING: ZERO ICT pattern detections!")
        print("  Possible causes:")
        print("  1. Quality filters TOO STRICT - no patterns pass")
        print("  2. Patterns genuinely not forming in market data")
        print("  3. Detection logic broken")
        print("\n  ACTION: Review quality filter thresholds in ict_structure.py")
        print("    - Current: volume > 2.5x, block_size > 0.5%, wick > 40%")
        print("    - Try: volume > 1.5x, block_size > 0.3%, wick > 30%")

    elif total_ict_detections > 0 and total_ict_trades == 0:
        print("\nINFO: ICT patterns detected but NO trades")
        print("  Detections are happening but failing at planning/ranking")
        print("  Check events_decisions.jsonl for rejection reasons")

    elif total_ict_detections > 0 and total_ict_trades > 0:
        conversion = total_ict_trades / total_ict_detections * 100
        if conversion < 20:
            print(f"\nWARNING: Low conversion rate ({conversion:.1f}%)")
            print("  Most detections not converting to trades")
            print("  Quality filters may still be too strict")
        elif conversion > 60:
            print(f"\nWARNING: Very high conversion rate ({conversion:.1f}%)")
            print("  Almost all detections trading - filters may be too loose")
        else:
            print(f"\nGOOD: Healthy conversion rate ({conversion:.1f}%)")
            print("  Quality filters appear well-calibrated")

    print()

if __name__ == "__main__":
    # Analyze original backtest (before fixes)
    print("ORIGINAL BACKTEST (Before Fixes):")
    print("="*80)
    backtest_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251117-021749_extracted\20251117-021749_full\20251117-021749"
    analyze_ict_detection(backtest_dir)

    print("\n\n")

    # Note: We'd need a new full backtest to analyze post-fixes
    print("NEW BACKTEST (After Fixes):")
    print("="*80)
    print("Need to run full backtest with current code to analyze...")
    print("Use: python tools/engine.py backtest --start 2023-12-01 --end 2024-01-31")
