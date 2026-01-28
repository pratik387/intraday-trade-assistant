#!/usr/bin/env python
"""
Analyze exit patterns from trade logs to validate optimal exit split strategy.

Compares BEFORE (40-40-20) vs AFTER (33-33-33) backtests to determine:
- How many trades reverse after T1 (hit BE stop)
- How many trades reach T2
- How many trades reach T3 (runners)
- Which exit split strategy is optimal for the actual trade distribution
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def parse_trade_logs(backtest_dir):
    """Parse all trade_logs.log files and categorize exit patterns."""

    results = {
        "hard_sl_before_t1": 0,      # Hit hard SL before any target
        "t1_only_then_reversed": 0,   # Hit T1, then BE stop (sl_post_t1)
        "t1_t2_then_reversed": 0,     # Hit T1+T2, then trail stop (sl_post_t2)
        "t1_t2_t3_runner": 0,         # Hit all three targets
        "t1_eod": 0,                  # Hit T1, then EOD exit
        "t1_t2_eod": 0,               # Hit T1+T2, then EOD exit
        "no_target_eod": 0,           # No targets, EOD exit
        "trades_by_symbol": defaultdict(list)
    }

    backtest_path = Path(backtest_dir)

    # Find all trade_logs.log files
    log_files = list(backtest_path.rglob("trade_logs.log"))
    print(f"Found {len(log_files)} trade log files")

    for log_file in log_files:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Parse trades by symbol
        current_trades = defaultdict(list)

        for line in lines:
            if "TRIGGER_EXEC" in line:
                # Extract symbol
                match = re.search(r'\| (NSE:\w+) \|', line)
                if match:
                    symbol = match.group(1)
                    current_trades[symbol] = []

            elif "EXIT" in line:
                # Extract exit details
                match = re.search(r'\| (NSE:\w+) \| .* \| PnL: Rs\.([\d\.\-]+) (\w+)', line)
                if match:
                    symbol = match.group(1)
                    pnl = float(match.group(2))
                    exit_reason = match.group(3)
                    current_trades[symbol].append({
                        'exit_reason': exit_reason,
                        'pnl': pnl
                    })

        # Categorize each trade's exit pattern
        for symbol, exits in current_trades.items():
            if not exits:
                continue

            exit_reasons = [e['exit_reason'] for e in exits]
            total_pnl = sum(e['pnl'] for e in exits)

            # Categorize based on exit sequence
            if 'hard_sl' in exit_reasons and 't1_partial' not in exit_reasons:
                # Hit hard SL before T1
                results["hard_sl_before_t1"] += 1
                category = "hard_sl_before_t1"

            elif 't1_partial' in exit_reasons and 't2_partial' not in exit_reasons:
                # Hit T1, check what happened next
                if 'sl_post_t1' in exit_reasons:
                    results["t1_only_then_reversed"] += 1
                    category = "t1_only_then_reversed"
                elif any('eod' in r for r in exit_reasons):
                    results["t1_eod"] += 1
                    category = "t1_eod"
                else:
                    # Unknown T1 exit pattern
                    category = "t1_unknown"

            elif 't1_partial' in exit_reasons and 't2_partial' in exit_reasons:
                # Hit T1+T2, check what happened next
                if 'sl_post_t2' in exit_reasons:
                    results["t1_t2_then_reversed"] += 1
                    category = "t1_t2_then_reversed"
                elif any('eod' in r for r in exit_reasons):
                    results["t1_t2_eod"] += 1
                    category = "t1_t2_eod"
                elif 'target_t3' in exit_reasons:
                    results["t1_t2_t3_runner"] += 1
                    category = "t1_t2_t3_runner"
                else:
                    category = "t1_t2_unknown"

            else:
                # No targets hit, just EOD
                results["no_target_eod"] += 1
                category = "no_target_eod"

            results["trades_by_symbol"][category].append({
                'symbol': symbol,
                'exits': exit_reasons,
                'pnl': total_pnl
            })

    return results

def analyze_split_impact(before_results, after_results):
    """Analyze which exit split is optimal based on trade distribution."""

    print("\n" + "="*80)
    print("EXIT PATTERN ANALYSIS: 40-40-20 vs 33-33-33")
    print("="*80)

    for label, results in [("BEFORE (40-40-20)", before_results), ("AFTER (33-33-33)", after_results)]:
        print(f"\n{label}")
        print("-" * 80)

        total_trades = sum([
            results["hard_sl_before_t1"],
            results["t1_only_then_reversed"],
            results["t1_t2_then_reversed"],
            results["t1_t2_t3_runner"],
            results["t1_eod"],
            results["t1_t2_eod"],
            results["no_target_eod"]
        ])

        print(f"Total trades: {total_trades}")
        print()

        # Category 1: Hard SL before T1 (exit split doesn't matter)
        pct = (results["hard_sl_before_t1"] / total_trades * 100) if total_trades > 0 else 0
        print(f"Hard SL before T1:           {results['hard_sl_before_t1']:3d} ({pct:5.1f}%) - Exit split IRRELEVANT")

        # Category 2: T1 then reversed (FAVORS HIGHER T1%)
        pct = (results["t1_only_then_reversed"] / total_trades * 100) if total_trades > 0 else 0
        print(f"T1 -> BE stop (reversed):     {results['t1_only_then_reversed']:3d} ({pct:5.1f}%) - FAVORS 40% or 60% @ T1")

        # Category 3: T1+T2 then reversed (FAVORS BALANCED SPLIT)
        pct = (results["t1_t2_then_reversed"] / total_trades * 100) if total_trades > 0 else 0
        print(f"T1+T2 -> trail stop:          {results['t1_t2_then_reversed']:3d} ({pct:5.1f}%) - FAVORS 40-40-20")

        # Category 4: Full runners T1+T2+T3 (FAVORS LOWER T1%, MORE TRAIL)
        pct = (results["t1_t2_t3_runner"] / total_trades * 100) if total_trades > 0 else 0
        print(f"T1+T2+T3 runners:            {results['t1_t2_t3_runner']:3d} ({pct:5.1f}%) - FAVORS 33-33-33")

        # Category 5: EOD exits
        eod_total = results["t1_eod"] + results["t1_t2_eod"] + results["no_target_eod"]
        pct = (eod_total / total_trades * 100) if total_trades > 0 else 0
        print(f"EOD exits (various):         {eod_total:3d} ({pct:5.1f}%) - NEUTRAL")
        print(f"  - T1 then EOD:             {results['t1_eod']:3d}")
        print(f"  - T1+T2 then EOD:          {results['t1_t2_eod']:3d}")
        print(f"  - No targets, EOD:         {results['no_target_eod']:3d}")

    # Comparative analysis
    print("\n" + "="*80)
    print("OPTIMAL SPLIT DETERMINATION")
    print("="*80)

    # Calculate which trades benefit from each split
    before_total = sum([
        before_results["hard_sl_before_t1"],
        before_results["t1_only_then_reversed"],
        before_results["t1_t2_then_reversed"],
        before_results["t1_t2_t3_runner"],
        before_results["t1_eod"],
        before_results["t1_t2_eod"],
        before_results["no_target_eod"]
    ])

    after_total = sum([
        after_results["hard_sl_before_t1"],
        after_results["t1_only_then_reversed"],
        after_results["t1_t2_then_reversed"],
        after_results["t1_t2_t3_runner"],
        after_results["t1_eod"],
        after_results["t1_t2_eod"],
        after_results["no_target_eod"]
    ])

    # Trades that favor HIGHER T1 capture (T1 reversals)
    before_favor_high_t1 = before_results["t1_only_then_reversed"]
    after_favor_high_t1 = after_results["t1_only_then_reversed"]

    # Trades that favor LOWER T1 capture (runners)
    before_favor_low_t1 = before_results["t1_t2_t3_runner"]
    after_favor_low_t1 = after_results["t1_t2_t3_runner"]

    # Trades that are neutral/balanced
    before_balanced = before_results["t1_t2_then_reversed"]
    after_balanced = after_results["t1_t2_then_reversed"]

    print(f"\nBEFORE (40-40-20) Trade Distribution:")
    print(f"  Favor HIGH T1 (60% or 40%):  {before_favor_high_t1:3d} ({before_favor_high_t1/before_total*100:5.1f}%)")
    print(f"  Favor BALANCED (40-40-20):   {before_balanced:3d} ({before_balanced/before_total*100:5.1f}%)")
    print(f"  Favor LOW T1 (33-33-33):     {before_favor_low_t1:3d} ({before_favor_low_t1/before_total*100:5.1f}%)")

    print(f"\nAFTER (33-33-33) Trade Distribution:")
    print(f"  Favor HIGH T1 (60% or 40%):  {after_favor_high_t1:3d} ({after_favor_high_t1/after_total*100:5.1f}%)")
    print(f"  Favor BALANCED (40-40-20):   {after_balanced:3d} ({after_balanced/after_total*100:5.1f}%)")
    print(f"  Favor LOW T1 (33-33-33):     {after_favor_low_t1:3d} ({after_favor_low_t1/after_total*100:5.1f}%)")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    for label, favor_high, balanced, favor_low, total in [
        ("BEFORE (40-40-20)", before_favor_high_t1, before_balanced, before_favor_low_t1, before_total),
        ("AFTER (33-33-33)", after_favor_high_t1, after_balanced, after_favor_low_t1, after_total)
    ]:
        print(f"\n{label}:")

        if favor_high > favor_low + balanced:
            print(f"  -> RECOMMENDATION: Use 60-20-20 or keep 40-40-20")
            print(f"  -> REASON: {favor_high} trades ({favor_high/total*100:.1f}%) reverse after T1")
            print(f"  -> Higher T1 capture maximizes profit from reversals")
        elif favor_low > favor_high + balanced:
            print(f"  -> RECOMMENDATION: Use 33-33-33")
            print(f"  -> REASON: {favor_low} trades ({favor_low/total*100:.1f}%) are runners")
            print(f"  -> Lower T1 capture maximizes profit from big moves")
        else:
            print(f"  -> RECOMMENDATION: Use 40-40-20 (balanced)")
            print(f"  -> REASON: Mixed distribution requires balanced approach")

    print("\n" + "="*80)

def main():
    before_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251110-132748_extracted\20251110-132748_full\20251110-132748"
    after_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251111-015916_extracted\20251111-015916_full\20251111-015916"

    print("Parsing BEFORE (40-40-20) backtest...")
    before_results = parse_trade_logs(before_dir)

    print("\nParsing AFTER (33-33-33) backtest...")
    after_results = parse_trade_logs(after_dir)

    # Analyze and compare
    analyze_split_impact(before_results, after_results)

    # Save detailed results
    output = {
        "before_40_40_20": before_results,
        "after_33_33_33": after_results
    }

    # Convert defaultdict to regular dict for JSON serialization
    for key in output:
        output[key]["trades_by_symbol"] = dict(output[key]["trades_by_symbol"])

    output_file = Path("exit_pattern_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

if __name__ == '__main__':
    main()
