"""
Analyze ICT pattern performance from backtest report.

Extracts and compares:
1. ICT pattern performance metrics
2. Non-ICT pattern performance
3. Overall statistics
4. Regime-specific performance (if available)
"""

import json
from pathlib import Path
from typing import Dict, Any

# ICT setup types
ICT_PATTERNS = {
    'order_block_long',
    'order_block_short',
    'fair_value_gap_long',
    'fair_value_gap_short',
    'liquidity_sweep_long',
    'liquidity_sweep_short',
    'premium_zone_short',
    'discount_zone_long',
    'break_of_structure_long',
    'break_of_structure_short',
    'change_of_character_long',
    'change_of_character_short',
}

def load_report(report_path: str) -> Dict[str, Any]:
    """Load the backtest report JSON."""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_setup_performance(setup_analysis: Dict[str, Any]) -> None:
    """Analyze and compare ICT vs non-ICT setup performance."""

    print("=" * 80)
    print("BACKTEST REPORT ANALYSIS - ICT vs Non-ICT Performance")
    print("=" * 80)
    print()

    # Separate ICT and non-ICT setups
    ict_setups = {}
    non_ict_setups = {}

    for setup_name, stats in setup_analysis.items():
        if setup_name in ICT_PATTERNS:
            ict_setups[setup_name] = stats
        else:
            non_ict_setups[setup_name] = stats

    # ICT Performance Summary
    print("ICT PATTERNS PERFORMANCE")
    print("-" * 80)

    if not ict_setups:
        print("NO ICT PATTERNS FOUND IN REPORT")
        print()
    else:
        ict_total_trades = sum(s.get('total_trades', 0) for s in ict_setups.values())
        ict_total_pnl = sum(s.get('total_pnl', 0) for s in ict_setups.values())
        ict_winning = sum(s.get('winning_trades', 0) for s in ict_setups.values())

        print(f"Total ICT Setups Active: {len(ict_setups)}")
        print(f"Total ICT Trades: {ict_total_trades}")
        print(f"Total ICT P&L: {ict_total_pnl:.2f}")
        if ict_total_trades > 0:
            ict_win_rate = (ict_winning / ict_total_trades) * 100
            print(f"ICT Win Rate: {ict_win_rate:.1f}%")
            print(f"Avg P&L per ICT Trade: {ict_total_pnl / ict_total_trades:.2f}")
        print()

        # Individual ICT pattern breakdown
        print("Individual ICT Pattern Performance:")
        print(f"{'Setup Type':<30} {'Trades':>8} {'Win%':>8} {'P&L':>10} {'Avg P&L':>10}")
        print("-" * 80)

        # Sort by total P&L descending
        sorted_ict = sorted(ict_setups.items(), key=lambda x: x[1].get('total_pnl', 0), reverse=True)

        for setup_name, stats in sorted_ict:
            trades = stats.get('total_trades', 0)
            wins = stats.get('winning_trades', 0)
            total_pnl = stats.get('total_pnl', 0)

            win_rate = (wins / trades * 100) if trades > 0 else 0
            avg_pnl = total_pnl / trades if trades > 0 else 0

            print(f"{setup_name:<30} {trades:>8} {win_rate:>7.1f}% {total_pnl:>10.2f} {avg_pnl:>10.2f}")

        print()

    # Non-ICT Performance Summary
    print("NON-ICT PATTERNS PERFORMANCE")
    print("-" * 80)

    non_ict_total_trades = sum(s.get('total_trades', 0) for s in non_ict_setups.values())
    non_ict_total_pnl = sum(s.get('total_pnl', 0) for s in non_ict_setups.values())
    non_ict_winning = sum(s.get('winning_trades', 0) for s in non_ict_setups.values())

    print(f"Total Non-ICT Setups Active: {len(non_ict_setups)}")
    print(f"Total Non-ICT Trades: {non_ict_total_trades}")
    print(f"Total Non-ICT P&L: {non_ict_total_pnl:.2f}")
    if non_ict_total_trades > 0:
        non_ict_win_rate = (non_ict_winning / non_ict_total_trades) * 100
        print(f"Non-ICT Win Rate: {non_ict_win_rate:.1f}%")
        print(f"Avg P&L per Non-ICT Trade: {non_ict_total_pnl / non_ict_total_trades:.2f}")
    print()

    # Top 5 Non-ICT performers
    print("Top 5 Non-ICT Patterns by P&L:")
    print(f"{'Setup Type':<30} {'Trades':>8} {'Win%':>8} {'P&L':>10} {'Avg P&L':>10}")
    print("-" * 80)

    sorted_non_ict = sorted(non_ict_setups.items(), key=lambda x: x[1].get('total_pnl', 0), reverse=True)

    for setup_name, stats in sorted_non_ict[:5]:
        trades = stats.get('total_trades', 0)
        wins = stats.get('winning_trades', 0)
        total_pnl = stats.get('total_pnl', 0)

        win_rate = (wins / trades * 100) if trades > 0 else 0
        avg_pnl = total_pnl / trades if trades > 0 else 0

        print(f"{setup_name:<30} {trades:>8} {win_rate:>7.1f}% {total_pnl:>10.2f} {avg_pnl:>10.2f}")

    print()

    # Comparative Summary
    print("COMPARATIVE SUMMARY")
    print("-" * 80)

    total_trades = ict_total_trades + non_ict_total_trades

    if total_trades > 0:
        ict_pct = (ict_total_trades / total_trades) * 100
        non_ict_pct = (non_ict_total_trades / total_trades) * 100

        print(f"ICT Patterns: {ict_total_trades} trades ({ict_pct:.1f}% of total)")
        print(f"Non-ICT Patterns: {non_ict_total_trades} trades ({non_ict_pct:.1f}% of total)")
        print()

        if ict_total_trades > 0 and non_ict_total_trades > 0:
            ict_avg = ict_total_pnl / ict_total_trades
            non_ict_avg = non_ict_total_pnl / non_ict_total_trades

            print(f"ICT Avg P&L per Trade: {ict_avg:.2f}")
            print(f"Non-ICT Avg P&L per Trade: {non_ict_avg:.2f}")

            if ict_avg > non_ict_avg:
                diff = ((ict_avg / non_ict_avg - 1) * 100) if non_ict_avg != 0 else 0
                print(f"ICT outperforming by: {diff:.1f}%")
            else:
                diff = ((non_ict_avg / ict_avg - 1) * 100) if ict_avg != 0 else 0
                print(f"Non-ICT outperforming by: {diff:.1f}%")

    print()

def analyze_decision_patterns(decision_analysis: Dict[str, Any]) -> None:
    """Analyze decision-level patterns."""

    print("=" * 80)
    print("DECISION ANALYSIS")
    print("=" * 80)
    print()

    # Overall decision stats
    overall = decision_analysis.get('overall', {})
    print("Overall Decision Stats:")
    print(f"Total Decisions: {overall.get('total_decisions', 0)}")
    print(f"Accepted Decisions: {overall.get('total_accepted', 0)}")
    print(f"Rejected Decisions: {overall.get('total_rejected', 0)}")

    if overall.get('total_decisions', 0) > 0:
        acceptance_rate = (overall.get('total_accepted', 0) / overall.get('total_decisions', 0)) * 100
        print(f"Acceptance Rate: {acceptance_rate:.1f}%")

    print()

    # By setup type
    by_setup = decision_analysis.get('by_setup', {})

    if by_setup:
        print("Decision Breakdown by Setup Type:")
        print(f"{'Setup Type':<30} {'Total':>8} {'Accepted':>10} {'Rejected':>10} {'Accept %':>10}")
        print("-" * 80)

        # Separate ICT and non-ICT
        ict_decisions = {}
        non_ict_decisions = {}

        for setup_name, stats in by_setup.items():
            if setup_name in ICT_PATTERNS:
                ict_decisions[setup_name] = stats
            else:
                non_ict_decisions[setup_name] = stats

        # Show ICT first
        if ict_decisions:
            print("ICT PATTERNS:")
            for setup_name, stats in sorted(ict_decisions.items()):
                total = stats.get('total', 0)
                accepted = stats.get('accepted', 0)
                rejected = stats.get('rejected', 0)
                accept_pct = (accepted / total * 100) if total > 0 else 0

                print(f"{setup_name:<30} {total:>8} {accepted:>10} {rejected:>10} {accept_pct:>9.1f}%")
            print()

        # Show top non-ICT
        if non_ict_decisions:
            print("TOP NON-ICT PATTERNS:")
            sorted_non_ict = sorted(non_ict_decisions.items(), key=lambda x: x[1].get('total', 0), reverse=True)

            for setup_name, stats in sorted_non_ict[:10]:
                total = stats.get('total', 0)
                accepted = stats.get('accepted', 0)
                rejected = stats.get('rejected', 0)
                accept_pct = (accepted / total * 100) if total > 0 else 0

                print(f"{setup_name:<30} {total:>8} {accepted:>10} {rejected:>10} {accept_pct:>9.1f}%")

    print()

def main():
    report_path = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\analysis\reports\misc\analysis_report_20_20251117_104148.json"

    print("Loading backtest report...")
    report = load_report(report_path)

    print(f"Report loaded successfully")
    print(f"Sessions analyzed: {report.get('session_count', 'N/A')}")
    print(f"Total trades: {report.get('total_trades', 'N/A')}")
    print(f"Overall P&L: {report.get('total_pnl', 'N/A')}")
    print()

    # Analyze setup performance
    if 'setup_analysis' in report:
        analyze_setup_performance(report['setup_analysis'])

    # Analyze decision patterns
    if 'decision_analysis' in report:
        analyze_decision_patterns(report['decision_analysis'])

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
