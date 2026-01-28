"""
Comprehensive backtest report analysis.

Analyzes ALL aspects of the backtest report:
1. Overall performance metrics
2. All setup type performance
3. Regime-specific analysis
4. Time-based patterns
5. Win/loss characteristics
6. Decision funnel analysis
7. Risk metrics
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

def load_report(report_path: str) -> Dict[str, Any]:
    """Load the backtest report JSON."""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)

def analyze_overall_performance(report: Dict[str, Any]):
    """Analyze top-level performance metrics."""
    print_section("OVERALL BACKTEST PERFORMANCE")

    print(f"\nBacktest Period:")
    print(f"  Sessions: {report.get('session_count', 'N/A')}")
    print(f"  Date Range: {report.get('date_range', {}).get('start', 'N/A')} to {report.get('date_range', {}).get('end', 'N/A')}")

    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {report.get('total_trades', 0)}")
    print(f"  Winning Trades: {report.get('winning_trades', 0)}")
    print(f"  Losing Trades: {report.get('losing_trades', 0)}")
    print(f"  Breakeven Trades: {report.get('breakeven_trades', 0)}")

    total_trades = report.get('total_trades', 0)
    if total_trades > 0:
        win_rate = (report.get('winning_trades', 0) / total_trades) * 100
        print(f"  Win Rate: {win_rate:.2f}%")

    print(f"\nP&L Metrics:")
    total_pnl = report.get('total_pnl', 0)
    print(f"  Total P&L: Rs {total_pnl:,.2f}")

    if total_trades > 0:
        avg_pnl = total_pnl / total_trades
        print(f"  Average P&L per Trade: Rs {avg_pnl:,.2f}")

    print(f"  Average Win: Rs {report.get('avg_win', 0):,.2f}")
    print(f"  Average Loss: Rs {report.get('avg_loss', 0):,.2f}")
    print(f"  Best Trade: Rs {report.get('best_trade', 0):,.2f}")
    print(f"  Worst Trade: Rs {report.get('worst_trade', 0):,.2f}")

    print(f"\nRisk Metrics:")
    print(f"  Profit Factor: {report.get('profit_factor', 0):.2f}")
    print(f"  Sharpe Ratio: {report.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown: Rs {report.get('max_drawdown', 0):,.2f}")
    print(f"  Recovery Factor: {report.get('recovery_factor', 0):.2f}")

def analyze_all_setups(setup_analysis: Dict[str, Any]):
    """Analyze all setup types comprehensively."""
    print_section("COMPLETE SETUP TYPE ANALYSIS")

    # Collect all setups with stats
    setup_stats = []
    for setup_name, stats in setup_analysis.items():
        trades = stats.get('total_trades', 0)
        if trades > 0:  # Only include setups with trades
            wins = stats.get('winning_trades', 0)
            losses = stats.get('losing_trades', 0)
            total_pnl = stats.get('total_pnl', 0)
            avg_pnl = total_pnl / trades if trades > 0 else 0
            win_rate = (wins / trades * 100) if trades > 0 else 0
            profit_factor = stats.get('profit_factor', 0)

            setup_stats.append({
                'name': setup_name,
                'trades': trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'avg_win': stats.get('avg_win', 0),
                'avg_loss': stats.get('avg_loss', 0),
                'best_trade': stats.get('best_trade', 0),
                'worst_trade': stats.get('worst_trade', 0),
                'profit_factor': profit_factor,
                'sharpe': stats.get('sharpe_ratio', 0)
            })

    # Sort by total P&L descending
    setup_stats.sort(key=lambda x: x['total_pnl'], reverse=True)

    print(f"\nTotal Active Setups: {len(setup_stats)}")
    print(f"\nAll Setup Types (sorted by P&L):")
    print(f"{'Rank':<5} {'Setup Type':<35} {'Trades':>7} {'W%':>7} {'Total P&L':>12} {'Avg P&L':>10} {'PF':>6}")
    print("-" * 100)

    for i, setup in enumerate(setup_stats, 1):
        print(f"{i:<5} {setup['name']:<35} {setup['trades']:>7} {setup['win_rate']:>6.1f}% "
              f"Rs{setup['total_pnl']:>10,.2f} Rs{setup['avg_pnl']:>8,.2f} {setup['profit_factor']:>6.2f}")

    # Summary stats
    total_trades_all = sum(s['trades'] for s in setup_stats)
    profitable_setups = [s for s in setup_stats if s['total_pnl'] > 0]
    losing_setups = [s for s in setup_stats if s['total_pnl'] < 0]

    print(f"\n{'Setup Performance Summary:'}")
    print(f"  Profitable Setups: {len(profitable_setups)} ({len(profitable_setups)/len(setup_stats)*100:.1f}%)")
    print(f"  Losing Setups: {len(losing_setups)} ({len(losing_setups)/len(setup_stats)*100:.1f}%)")

    if profitable_setups:
        total_profit = sum(s['total_pnl'] for s in profitable_setups)
        print(f"  Total Profit from Winners: Rs {total_profit:,.2f}")

    if losing_setups:
        total_loss = sum(s['total_pnl'] for s in losing_setups)
        print(f"  Total Loss from Losers: Rs {total_loss:,.2f}")

    # Top 10 performers
    print(f"\n{'TOP 10 PERFORMING SETUPS:'}")
    print(f"{'Rank':<5} {'Setup Type':<35} {'Trades':>7} {'W%':>7} {'Total P&L':>12} {'Avg P&L':>10}")
    print("-" * 100)

    for i, setup in enumerate(setup_stats[:10], 1):
        print(f"{i:<5} {setup['name']:<35} {setup['trades']:>7} {setup['win_rate']:>6.1f}% "
              f"Rs {setup['total_pnl']:>10,.2f} Rs {setup['avg_pnl']:>8,.2f}")

    # Bottom 10 performers
    print(f"\n{'BOTTOM 10 PERFORMING SETUPS:'}")
    print(f"{'Rank':<5} {'Setup Type':<35} {'Trades':>7} {'W%':>7} {'Total P&L':>12} {'Avg P&L':>10}")
    print("-" * 100)

    for i, setup in enumerate(setup_stats[-10:][::-1], 1):
        print(f"{i:<5} {setup['name']:<35} {setup['trades']:>7} {setup['win_rate']:>6.1f}% "
              f"Rs {setup['total_pnl']:>10,.2f} Rs {setup['avg_pnl']:>8,.2f}")

    return setup_stats

def analyze_by_direction(setup_stats: List[Dict]):
    """Analyze long vs short performance."""
    print_section("DIRECTIONAL ANALYSIS (LONG vs SHORT)")

    long_setups = [s for s in setup_stats if '_long' in s['name']]
    short_setups = [s for s in setup_stats if '_short' in s['name']]
    neutral_setups = [s for s in setup_stats if '_long' not in s['name'] and '_short' not in s['name']]

    print(f"\n{'LONG SETUPS:'}")
    if long_setups:
        total_long_trades = sum(s['trades'] for s in long_setups)
        total_long_pnl = sum(s['total_pnl'] for s in long_setups)
        long_wins = sum(s['wins'] for s in long_setups)
        long_win_rate = (long_wins / total_long_trades * 100) if total_long_trades > 0 else 0

        print(f"  Active Long Setups: {len(long_setups)}")
        print(f"  Total Long Trades: {total_long_trades}")
        print(f"  Long Win Rate: {long_win_rate:.1f}%")
        print(f"  Total Long P&L: Rs {total_long_pnl:,.2f}")
        print(f"  Avg P&L per Long Trade: Rs {total_long_pnl/total_long_trades:,.2f}")

        # Top 5 long setups
        long_setups_sorted = sorted(long_setups, key=lambda x: x['total_pnl'], reverse=True)
        print(f"\n  Top 5 Long Setups:")
        for setup in long_setups_sorted[:5]:
            print(f"    {setup['name']:<35} {setup['trades']:>3} trades  {setup['win_rate']:>5.1f}%  Rs {setup['total_pnl']:>10,.2f}")

    print(f"\n{'SHORT SETUPS:'}")
    if short_setups:
        total_short_trades = sum(s['trades'] for s in short_setups)
        total_short_pnl = sum(s['total_pnl'] for s in short_setups)
        short_wins = sum(s['wins'] for s in short_setups)
        short_win_rate = (short_wins / total_short_trades * 100) if total_short_trades > 0 else 0

        print(f"  Active Short Setups: {len(short_setups)}")
        print(f"  Total Short Trades: {total_short_trades}")
        print(f"  Short Win Rate: {short_win_rate:.1f}%")
        print(f"  Total Short P&L: Rs {total_short_pnl:,.2f}")
        print(f"  Avg P&L per Short Trade: Rs {total_short_pnl/total_short_trades:,.2f}")

        # Top 5 short setups
        short_setups_sorted = sorted(short_setups, key=lambda x: x['total_pnl'], reverse=True)
        print(f"\n  Top 5 Short Setups:")
        for setup in short_setups_sorted[:5]:
            print(f"    {setup['name']:<35} {setup['trades']:>3} trades  {setup['win_rate']:>5.1f}%  Rs {setup['total_pnl']:>10,.2f}")

    if neutral_setups:
        total_neutral_trades = sum(s['trades'] for s in neutral_setups)
        total_neutral_pnl = sum(s['total_pnl'] for s in neutral_setups)
        print(f"\n{'NEUTRAL/OTHER SETUPS:'}")
        print(f"  Active Neutral Setups: {len(neutral_setups)}")
        print(f"  Total Neutral Trades: {total_neutral_trades}")
        print(f"  Total Neutral P&L: Rs {total_neutral_pnl:,.2f}")

def analyze_regime_performance(report: Dict[str, Any]):
    """Analyze performance by regime if available."""
    if 'regime_analysis' not in report:
        return

    print_section("REGIME-BASED PERFORMANCE")

    regime_analysis = report['regime_analysis']

    for regime, stats in regime_analysis.items():
        print(f"\n{regime.upper()}:")
        print(f"  Trades: {stats.get('total_trades', 0)}")
        print(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
        print(f"  Total P&L: Rs {stats.get('total_pnl', 0):,.2f}")
        print(f"  Avg P&L: Rs {stats.get('avg_pnl', 0):,.2f}")

        if 'top_setups' in stats:
            print(f"  Top Setups:")
            for setup in stats['top_setups'][:3]:
                print(f"    - {setup}")

def analyze_hourly_patterns(report: Dict[str, Any]):
    """Analyze time-of-day patterns."""
    if 'decision_analysis' not in report or 'hourly_patterns' not in report['decision_analysis']:
        return

    print_section("TIME-OF-DAY ANALYSIS")

    hourly = report['decision_analysis'].get('hourly_patterns', {})

    if hourly:
        print(f"\n{'Hour':<6} {'Decisions':>10} {'Accepted':>10} {'Rejected':>10} {'Accept %':>10}")
        print("-" * 50)

        for hour in sorted(hourly.keys()):
            stats = hourly[hour]
            total = stats.get('total', 0)
            accepted = stats.get('accepted', 0)
            rejected = stats.get('rejected', 0)
            accept_pct = (accepted / total * 100) if total > 0 else 0

            print(f"{hour:<6} {total:>10} {accepted:>10} {rejected:>10} {accept_pct:>9.1f}%")

def analyze_win_loss_characteristics(setup_stats: List[Dict]):
    """Analyze win/loss size distributions."""
    print_section("WIN/LOSS CHARACTERISTICS")

    # Setups with best win rate (min 5 trades)
    high_freq = [s for s in setup_stats if s['trades'] >= 5]

    if high_freq:
        print(f"\n{'HIGHEST WIN RATE (min 5 trades):'}")
        sorted_by_wr = sorted(high_freq, key=lambda x: x['win_rate'], reverse=True)
        print(f"{'Setup Type':<35} {'Trades':>7} {'Win %':>8} {'Total P&L':>12}")
        print("-" * 70)
        for setup in sorted_by_wr[:10]:
            print(f"{setup['name']:<35} {setup['trades']:>7} {setup['win_rate']:>7.1f}% Rs {setup['total_pnl']:>10,.2f}")

        print(f"\n{'LOWEST WIN RATE (min 5 trades):'}")
        print(f"{'Setup Type':<35} {'Trades':>7} {'Win %':>8} {'Total P&L':>12}")
        print("-" * 70)
        for setup in sorted_by_wr[-10:][::-1]:
            print(f"{setup['name']:<35} {setup['trades']:>7} {setup['win_rate']:>7.1f}% Rs {setup['total_pnl']:>10,.2f}")

    # Setups with best avg win
    print(f"\n{'LARGEST AVERAGE WINS:'}")
    sorted_by_avg_win = sorted(setup_stats, key=lambda x: x['avg_win'], reverse=True)
    print(f"{'Setup Type':<35} {'Trades':>7} {'Avg Win':>12} {'Avg Loss':>12}")
    print("-" * 75)
    for setup in sorted_by_avg_win[:10]:
        print(f"{setup['name']:<35} {setup['trades']:>7} Rs {setup['avg_win']:>10,.2f} Rs {setup['avg_loss']:>10,.2f}")

    # Setups with worst avg loss
    print(f"\n{'LARGEST AVERAGE LOSSES:'}")
    sorted_by_avg_loss = sorted(setup_stats, key=lambda x: x['avg_loss'])
    print(f"{'Setup Type':<35} {'Trades':>7} {'Avg Win':>12} {'Avg Loss':>12}")
    print("-" * 75)
    for setup in sorted_by_avg_loss[:10]:
        print(f"{setup['name']:<35} {setup['trades']:>7} Rs {setup['avg_win']:>10,.2f} Rs {setup['avg_loss']:>10,.2f}")

def analyze_decision_funnel(report: Dict[str, Any]):
    """Analyze decision acceptance/rejection patterns."""
    if 'decision_analysis' not in report:
        return

    print_section("DECISION FUNNEL ANALYSIS")

    decision_analysis = report['decision_analysis']
    overall = decision_analysis.get('overall', {})

    total_decisions = overall.get('total_decisions', 0)
    accepted = overall.get('total_accepted', 0)
    rejected = overall.get('total_rejected', 0)

    print(f"\nOverall Decision Flow:")
    print(f"  Total Decisions: {total_decisions}")
    print(f"  Accepted: {accepted}")
    print(f"  Rejected: {rejected}")

    if total_decisions > 0:
        accept_rate = (accepted / total_decisions) * 100
        print(f"  Acceptance Rate: {accept_rate:.1f}%")

    # By acceptance status
    if 'by_acceptance' in decision_analysis:
        by_acceptance = decision_analysis['by_acceptance']
        print(f"\nDecision Outcomes:")
        for status, stats in by_acceptance.items():
            print(f"  {status}: {stats.get('count', 0)} ({stats.get('percentage', 0):.1f}%)")

    # Top rejected setups
    by_setup = decision_analysis.get('by_setup', {})
    if by_setup:
        setup_rejection_rates = []
        for setup, stats in by_setup.items():
            total = stats.get('total', 0)
            rejected = stats.get('rejected', 0)
            if total > 0:
                rejection_rate = (rejected / total) * 100
                setup_rejection_rates.append({
                    'setup': setup,
                    'total': total,
                    'accepted': stats.get('accepted', 0),
                    'rejected': rejected,
                    'rejection_rate': rejection_rate
                })

        setup_rejection_rates.sort(key=lambda x: x['total'], reverse=True)

        print(f"\nTop 15 Setups by Decision Volume:")
        print(f"{'Setup Type':<35} {'Total':>8} {'Accepted':>10} {'Rejected':>10} {'Reject %':>10}")
        print("-" * 80)
        for item in setup_rejection_rates[:15]:
            print(f"{item['setup']:<35} {item['total']:>8} {item['accepted']:>10} "
                  f"{item['rejected']:>10} {item['rejection_rate']:>9.1f}%")

def analyze_risk_reward(setup_stats: List[Dict]):
    """Analyze risk/reward profiles."""
    print_section("RISK/REWARD ANALYSIS")

    print(f"\n{'BEST PROFIT FACTORS (min 3 trades):'}")
    high_pf = [s for s in setup_stats if s['trades'] >= 3 and s['profit_factor'] > 0]
    sorted_by_pf = sorted(high_pf, key=lambda x: x['profit_factor'], reverse=True)

    print(f"{'Setup Type':<35} {'Trades':>7} {'PF':>8} {'Win %':>8} {'Total P&L':>12}")
    print("-" * 80)
    for setup in sorted_by_pf[:10]:
        print(f"{setup['name']:<35} {setup['trades']:>7} {setup['profit_factor']:>8.2f} "
              f"{setup['win_rate']:>7.1f}% Rs {setup['total_pnl']:>10,.2f}")

    print(f"\n{'BEST SHARPE RATIOS (min 3 trades):'}")
    high_sharpe = [s for s in setup_stats if s['trades'] >= 3]
    sorted_by_sharpe = sorted(high_sharpe, key=lambda x: x['sharpe'], reverse=True)

    print(f"{'Setup Type':<35} {'Trades':>7} {'Sharpe':>8} {'Win %':>8} {'Total P&L':>12}")
    print("-" * 80)
    for setup in sorted_by_sharpe[:10]:
        print(f"{setup['name']:<35} {setup['trades']:>7} {setup['sharpe']:>8.3f} "
              f"{setup['win_rate']:>7.1f}% Rs {setup['total_pnl']:>10,.2f}")

def main():
    report_path = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\analysis\reports\misc\analysis_report_20_20251117_104148.json"

    print("=" * 100)
    print("  COMPREHENSIVE BACKTEST ANALYSIS")
    print("=" * 100)
    print(f"\nReport: {Path(report_path).name}")

    report = load_report(report_path)

    # Run all analyses
    analyze_overall_performance(report)

    if 'setup_analysis' in report:
        setup_stats = analyze_all_setups(report['setup_analysis'])
        analyze_by_direction(setup_stats)
        analyze_win_loss_characteristics(setup_stats)
        analyze_risk_reward(setup_stats)

    analyze_regime_performance(report)
    analyze_hourly_patterns(report)
    analyze_decision_funnel(report)

    print("\n" + "=" * 100)
    print("  ANALYSIS COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
