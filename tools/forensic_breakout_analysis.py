"""
Direct Forensic Analysis of Blocked Breakout Trades

Compares baseline vs filtered backtests to identify which breakout trades were blocked
and provides data-driven recommendations for filter adjustments.

Usage:
    python tools/forensic_breakout_analysis.py
"""

import json
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Paths to extracted backtests
BASELINE_DIR = Path("backtest_20251108-034930_extracted/20251108-034930_full/20251108-034930")
FILTERED_DIR = Path("backtest_20251108-124615_extracted/20251108-124615_full/20251108-124615")
OUTPUT_FILE = Path("FORENSIC_BREAKOUT_ANALYSIS.md")


def load_trades_from_analytics(backtest_dir: Path) -> list:
    """Load all EXIT trades from analytics.jsonl files"""
    trades = []

    for session_dir in sorted(backtest_dir.glob("20*")):
        analytics_file = session_dir / "analytics.jsonl"

        if not analytics_file.exists():
            continue

        with open(analytics_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event.get('stage') == 'EXIT':
                        event['session_date'] = session_dir.name
                        trades.append(event)
                except json.JSONDecodeError:
                    continue

    return trades


def create_trade_signature(trade: dict) -> str:
    """Create unique signature for trade matching"""
    return f"{trade['session_date']}_{trade['symbol']}_{trade['strategy']}_{trade.get('entry_time', 'NA')}"


def main():
    print("="*80)
    print("FORENSIC BREAKOUT ANALYSIS")
    print("="*80)

    # Load trades
    print("\nLoading baseline trades...")
    baseline_trades = load_trades_from_analytics(BASELINE_DIR)
    print(f"Loaded {len(baseline_trades)} baseline trades")

    print("\nLoading filtered trades...")
    filtered_trades = load_trades_from_analytics(FILTERED_DIR)
    print(f"Loaded {len(filtered_trades)} filtered trades")

    # Create trade signatures
    baseline_sigs = {create_trade_signature(t): t for t in baseline_trades}
    filtered_sigs = {create_trade_signature(t) for t in filtered_trades}

    # Find blocked trades (only breakouts)
    blocked_trades = []
    for sig, trade in baseline_sigs.items():
        if sig not in filtered_sigs and 'breakout' in trade.get('strategy', '').lower():
            blocked_trades.append(trade)

    print(f"\nFound {len(blocked_trades)} blocked breakout trades")

    if len(blocked_trades) == 0:
        print("\nNo blocked breakout trades found. Exiting.")
        return

    # Analyze blocked trades
    winners = [t for t in blocked_trades if t['pnl'] > 0]
    losers = [t for t in blocked_trades if t['pnl'] <= 0]
    hard_sl_losers = [t for t in losers if t.get('reason') == 'hard_sl']

    winner_pnl = sum(t['pnl'] for t in winners)
    loser_pnl = sum(t['pnl'] for t in losers)
    net_pnl = winner_pnl + loser_pnl

    print(f"\nBlocked Trade Breakdown:")
    print(f"  Winners: {len(winners)} ({len(winners)/len(blocked_trades)*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/len(blocked_trades)*100:.1f}%)")
    print(f"  Hard SL losers: {len(hard_sl_losers)} ({len(hard_sl_losers)/len(losers)*100:.1f}% of losers)")
    print(f"\nP&L Impact:")
    print(f"  Gross winners blocked: Rs.{winner_pnl:.2f}")
    print(f"  Gross losers blocked: Rs.{loser_pnl:.2f}")
    print(f"  Net P&L impact: Rs.{net_pnl:.2f}")

    # Generate report
    print(f"\nGenerating report: {OUTPUT_FILE}")

    report = []
    report.append("# FORENSIC BREAKOUT ANALYSIS REPORT")
    report.append(f"\n**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Baseline**: backtest_20251108-034930 (102 trades)")
    report.append(f"**Filtered**: backtest_20251108-124615 (84 trades)")
    report.append("\n---\n")

    # Executive Summary
    report.append("## EXECUTIVE SUMMARY\n")
    report.append(f"**Total Blocked Breakout Trades**: {len(blocked_trades)}")
    report.append(f"\n**Impact**:")
    report.append(f"- Winners blocked: {len(winners)} trades (Rs.{winner_pnl:.2f} opportunity cost)")
    report.append(f"- Losers blocked: {len(losers)} trades (Rs.{loser_pnl:.2f} saved)")
    report.append(f"- Hard SL losers blocked: {len(hard_sl_losers)} trades")
    report.append(f"- Net P&L impact: Rs.{net_pnl:.2f}")
    report.append("\n---\n")

    # Critical Finding
    report.append("## CRITICAL FINDING\n")

    if len(winners) > len(hard_sl_losers):
        report.append("### [CRITICAL] Filters Too Restrictive")
        report.append(f"\nBlocked **{len(winners)} potential winners** vs only **{len(hard_sl_losers)} hard SL losers**")
        report.append(f"\n**Opportunity Cost**: Rs.{winner_pnl:.2f} in potential profit")
        report.append(f"**Losses Saved**: Rs.{sum(t['pnl'] for t in hard_sl_losers):.2f}")
        report.append("\n**Recommendation**: RELAX FILTERS - Current thresholds are blocking high-quality breakouts")
    else:
        report.append("###  [OK] Filters Working as Intended")
        report.append(f"\nBlocked **{len(hard_sl_losers)} hard SL losers** vs **{len(winners)} winners**")
        report.append("\n**Recommendation**: Filters are correctly preventing bad trades")

    report.append("\n---\n")

    # Blocked Winners Detail
    report.append("## BLOCKED WINNERS (Opportunity Cost)\n")
    report.append("\n### Top 20 Blocked Winners\n")
    report.append("| Date | Symbol | Strategy | Exit Reason | P&L |")
    report.append("|------|--------|----------|-------------|-----|")

    for trade in sorted(winners, key=lambda x: x['pnl'], reverse=True)[:20]:
        report.append(f"| {trade['session_date']} | {trade['symbol']} | "
                     f"{trade['strategy']} | {trade.get('reason', 'N/A')} | "
                     f"Rs.{trade['pnl']:.2f} |")

    report.append(f"\n**Total Winner P&L Blocked**: Rs.{winner_pnl:.2f}")

    # Winner exit reasons
    winner_reasons = pd.Series([t.get('reason', 'unknown') for t in winners]).value_counts()
    report.append("\n### Winner Exit Reasons")
    report.append("| Exit Reason | Count | % |")
    report.append("|-------------|-------|---|")
    for reason, count in winner_reasons.items():
        pct = count / len(winners) * 100
        report.append(f"| {reason} | {count} | {pct:.1f}% |")

    report.append("\n---\n")

    # Blocked Losers Detail
    report.append("## BLOCKED LOSERS (Correctly Filtered)\n")
    report.append("\n### Top 20 Blocked Losers\n")
    report.append("| Date | Symbol | Strategy | Exit Reason | P&L |")
    report.append("|------|--------|----------|-------------|-----|")

    for trade in sorted(losers, key=lambda x: x['pnl'])[:20]:
        report.append(f"| {trade['session_date']} | {trade['symbol']} | "
                     f"{trade['strategy']} | {trade.get('reason', 'N/A')} | "
                     f"Rs.{trade['pnl']:.2f} |")

    report.append(f"\n**Total Loser P&L Blocked**: Rs.{loser_pnl:.2f}")
    report.append(f"\n**Hard SL Losers**: {len(hard_sl_losers)}/{len(losers)} ({len(hard_sl_losers)/len(losers)*100:.1f}%)")

    # Loser exit reasons
    loser_reasons = pd.Series([t.get('reason', 'unknown') for t in losers]).value_counts()
    report.append("\n### Loser Exit Reasons")
    report.append("| Exit Reason | Count | % |")
    report.append("|-------------|-------|---|")
    for reason, count in loser_reasons.items():
        pct = count / len(losers) * 100
        report.append(f"| {reason} | {count} | {pct:.1f}% |")

    report.append("\n---\n")

    # Strategy Breakdown
    report.append("## BLOCKED TRADES BY STRATEGY\n")

    by_strategy = defaultdict(list)
    for trade in blocked_trades:
        by_strategy[trade['strategy']].append(trade)

    for strategy, trades in sorted(by_strategy.items()):
        strat_winners = [t for t in trades if t['pnl'] > 0]
        strat_losers = [t for t in trades if t['pnl'] <= 0]
        strat_pnl = sum(t['pnl'] for t in trades)

        report.append(f"\n### {strategy}")
        report.append(f"- Total: {len(trades)} trades")
        report.append(f"- Winners: {len(strat_winners)} ({len(strat_winners)/len(trades)*100:.1f}%)")
        report.append(f"- Losers: {len(strat_losers)} ({len(strat_losers)/len(trades)*100:.1f}%)")
        report.append(f"- Net P&L: Rs.{strat_pnl:.2f}")

    report.append("\n---\n")

    # Recommendations
    report.append("## RECOMMENDED ACTIONS\n")

    if len(winners) > len(hard_sl_losers):
        report.append("### 1. RELAX INSTITUTIONAL FILTERS\n")
        report.append("Current filters are TOO RESTRICTIVE. Suggested adjustments:\n")
        report.append("- **Timing Filter**: Reduce restricted window from 9:15-9:45am to 9:15-9:30am")
        report.append("- **Candle Conviction**: Relax from 70%/30% to 60%/40%")
        report.append("- **Volume Accumulation**: Reduce from 3/5 bars to 2/5 bars")
        report.append("- **Level Cleanness**: Increase max touches from 3 to 5")
        report.append("\n### 2. RUN SPIKE TESTS\n")
        report.append(f"Run spike tests on top {min(10, len(winners))} blocked winners to validate they would have been profitable")
        report.append("\n### 3. ITERATIVE TESTING\n")
        report.append("- Implement one filter adjustment at a time")
        report.append("- Re-run backtest after each change")
        report.append("- Measure impact on hard SL rate and P&L")
    else:
        report.append("### 1. VALIDATE FILTER EFFECTIVENESS\n")
        report.append("Filters appear to be working correctly. Consider:")
        report.append("- Running spike tests on blocked hard SL losers to confirm they would have lost")
        report.append("- Minor relaxation to capture more winners while maintaining quality")

    report.append("\n---\n")

    # Next Steps
    report.append("## NEXT STEPS\n")
    report.append("1. Review top 10 blocked winners manually in logs")
    report.append("2. Check which specific filter rejected each winner")
    report.append("3. Run spike test simulations using 1m OHLC data")
    report.append("4. Implement recommended filter adjustments")
    report.append("5. Re-run backtest and compare results")
    report.append("\n---\n")
    report.append(f"\n*Report generated by forensic_breakout_analysis.py*")

    # Write report
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(report))

    print(f"Report saved: {OUTPUT_FILE}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Metrics:")
    print(f"  - Blocked trades: {len(blocked_trades)}")
    print(f"  - Winners blocked: {len(winners)} (Rs.{winner_pnl:.2f})")
    print(f"  - Losers blocked: {len(losers)} (Rs.{loser_pnl:.2f})")
    print(f"  - Net impact: Rs.{net_pnl:.2f}")
    print(f"\nFull report: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
