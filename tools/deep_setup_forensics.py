"""
Deep forensic analysis of failing setups using actual trade-by-trade data.

Analyzes:
1. Exit reason distribution (why are trades losing?)
2. Regime-specific performance
3. MFE/MAE patterns (profit potential vs realized)
4. Time-based patterns
5. Symbol-specific issues
6. Entry quality
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, Counter

def load_backtest_trades(backtest_dir: str) -> pd.DataFrame:
    """Load all trade data from backtest directory."""
    base_path = Path(backtest_dir)

    # Find the nested directory structure
    full_dir = list(base_path.glob("*_full"))[0]
    run_dir = list(full_dir.iterdir())[0]

    print(f"Loading trades from: {run_dir}")

    all_trades = []

    # Iterate through all session directories
    for session_dir in sorted(run_dir.iterdir()):
        if not session_dir.is_dir():
            continue

        analytics_file = session_dir / 'analytics.jsonl'

        if not analytics_file.exists():
            continue

        # Read all trades from this session
        with open(analytics_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    trade = json.loads(line.strip())
                    trade['session_date'] = session_dir.name
                    all_trades.append(trade)
                except Exception as e:
                    continue

    print(f"Loaded {len(all_trades)} total trades")
    return pd.DataFrame(all_trades)

def analyze_setup_deeply(df: pd.DataFrame, setup_name: str) -> Dict[str, Any]:
    """Deep forensic analysis of a specific setup."""

    setup_df = df[df['setup_type'] == setup_name].copy()

    if len(setup_df) == 0:
        return None

    analysis = {
        'setup_name': setup_name,
        'total_trades': len(setup_df),
        'total_pnl': setup_df['pnl'].sum(),
        'avg_pnl': setup_df['pnl'].mean(),
    }

    # Win/loss breakdown
    wins = setup_df[setup_df['pnl'] > 0]
    losses = setup_df[setup_df['pnl'] < 0]

    analysis['wins'] = len(wins)
    analysis['losses'] = len(losses)
    analysis['win_rate'] = len(wins) / len(setup_df) * 100 if len(setup_df) > 0 else 0

    # Win/loss sizes
    analysis['avg_win'] = wins['pnl'].mean() if len(wins) > 0 else 0
    analysis['avg_loss'] = losses['pnl'].mean() if len(losses) > 0 else 0
    analysis['max_win'] = wins['pnl'].max() if len(wins) > 0 else 0
    analysis['max_loss'] = losses['pnl'].min() if len(losses) > 0 else 0

    # Risk/reward
    if analysis['avg_loss'] != 0:
        analysis['avg_rr'] = abs(analysis['avg_win'] / analysis['avg_loss'])
        breakeven_wr = abs(analysis['avg_loss']) / (abs(analysis['avg_loss']) + analysis['avg_win']) * 100
        analysis['breakeven_wr'] = breakeven_wr
        analysis['wr_gap'] = analysis['win_rate'] - breakeven_wr
    else:
        analysis['avg_rr'] = 0
        analysis['breakeven_wr'] = 0
        analysis['wr_gap'] = 0

    # EXIT REASON ANALYSIS
    if 'exit_reason' in setup_df.columns:
        exit_analysis = []

        for exit_reason in setup_df['exit_reason'].unique():
            exit_trades = setup_df[setup_df['exit_reason'] == exit_reason]
            exit_wins = exit_trades[exit_trades['pnl'] > 0]

            exit_analysis.append({
                'reason': exit_reason,
                'count': len(exit_trades),
                'pct': len(exit_trades) / len(setup_df) * 100,
                'total_pnl': exit_trades['pnl'].sum(),
                'avg_pnl': exit_trades['pnl'].mean(),
                'win_rate': len(exit_wins) / len(exit_trades) * 100 if len(exit_trades) > 0 else 0
            })

        analysis['exit_reasons'] = sorted(exit_analysis, key=lambda x: x['total_pnl'])

    # REGIME ANALYSIS
    if 'regime' in setup_df.columns:
        regime_analysis = []

        for regime in setup_df['regime'].unique():
            regime_trades = setup_df[setup_df['regime'] == regime]
            regime_wins = regime_trades[regime_trades['pnl'] > 0]

            regime_analysis.append({
                'regime': regime,
                'count': len(regime_trades),
                'pct': len(regime_trades) / len(setup_df) * 100,
                'total_pnl': regime_trades['pnl'].sum(),
                'avg_pnl': regime_trades['pnl'].mean(),
                'win_rate': len(regime_wins) / len(regime_trades) * 100 if len(regime_trades) > 0 else 0
            })

        analysis['regimes'] = sorted(regime_analysis, key=lambda x: x['total_pnl'])

    # MFE/MAE ANALYSIS
    if 'mfe' in setup_df.columns and 'mae' in setup_df.columns:
        analysis['avg_mfe'] = setup_df['mfe'].mean()
        analysis['avg_mae'] = setup_df['mae'].mean()

        # Analyze winning trades
        if len(wins) > 0:
            wins_copy = wins.copy()
            wins_copy['mfe_capture'] = (wins_copy['pnl'] / wins_copy['mfe'] * 100).clip(0, 100)
            analysis['avg_mfe_capture'] = wins_copy['mfe_capture'].mean()

        # Analyze losing trades - how many had profit before loss?
        if len(losses) > 0:
            losses_with_profit = losses[losses['mfe'] > 0]
            analysis['losses_with_profit_count'] = len(losses_with_profit)
            analysis['losses_with_profit_pct'] = len(losses_with_profit) / len(losses) * 100
            analysis['avg_mfe_before_loss'] = losses['mfe'].mean()

            # How much profit was available before loss?
            if len(losses_with_profit) > 0:
                analysis['avg_wasted_profit'] = losses_with_profit['mfe'].mean()

    # DURATION ANALYSIS
    if 'entry_time' in setup_df.columns and 'exit_time' in setup_df.columns:
        setup_df['duration_minutes'] = (
            pd.to_datetime(setup_df['exit_time']) - pd.to_datetime(setup_df['entry_time'])
        ).dt.total_seconds() / 60

        analysis['avg_duration'] = setup_df['duration_minutes'].mean()
        analysis['avg_duration_wins'] = wins['duration_minutes'].mean() if len(wins) > 0 else 0
        analysis['avg_duration_losses'] = losses['duration_minutes'].mean() if len(losses) > 0 else 0

    # SYMBOL ANALYSIS - Top losers
    if 'symbol' in setup_df.columns:
        symbol_pnl = setup_df.groupby('symbol')['pnl'].agg(['count', 'sum', 'mean']).reset_index()
        symbol_pnl = symbol_pnl.sort_values('sum')

        analysis['worst_symbols'] = symbol_pnl.head(10).to_dict('records')
        analysis['best_symbols'] = symbol_pnl.tail(10).to_dict('records')

    return analysis

def print_forensic_report(analysis: Dict[str, Any]):
    """Print detailed forensic report."""

    print("\n" + "=" * 100)
    print(f"FORENSIC ANALYSIS: {analysis['setup_name']}")
    print("=" * 100)

    # Overall stats
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total Trades: {analysis['total_trades']}")
    print(f"  Total P&L: Rs {analysis['total_pnl']:,.2f}")
    print(f"  Avg P&L: Rs {analysis['avg_pnl']:,.2f}")
    print(f"  Win Rate: {analysis['win_rate']:.1f}% ({analysis['wins']} wins / {analysis['losses']} losses)")

    # Win/Loss characteristics
    print(f"\nWIN/LOSS CHARACTERISTICS:")
    print(f"  Avg Win: Rs {analysis['avg_win']:,.2f}")
    print(f"  Avg Loss: Rs {analysis['avg_loss']:,.2f}")
    print(f"  Max Win: Rs {analysis['max_win']:,.2f}")
    print(f"  Max Loss: Rs {analysis['max_loss']:,.2f}")
    print(f"  Risk/Reward Ratio: {analysis['avg_rr']:.2f}")

    if analysis['breakeven_wr'] > 0:
        print(f"\n  Breakeven Win Rate: {analysis['breakeven_wr']:.1f}%")
        print(f"  Actual Win Rate: {analysis['win_rate']:.1f}%")
        print(f"  Gap: {analysis['wr_gap']:+.1f}%")

        if analysis['wr_gap'] < 0:
            print(f"\n  >>> PROBLEM: Win rate is {abs(analysis['wr_gap']):.1f}% BELOW breakeven")
            print(f"      Need to either:")
            print(f"      1. Improve win rate from {analysis['win_rate']:.1f}% to {analysis['breakeven_wr']:.1f}%")
            print(f"      2. OR improve R:R from {analysis['avg_rr']:.2f} to {1/(analysis['win_rate']/100) - 1:.2f}")

    # EXIT REASON ANALYSIS
    if 'exit_reasons' in analysis:
        print(f"\nEXIT REASON BREAKDOWN:")
        print(f"  {'Exit Reason':<25} {'Count':>7} {'%':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Win %':>7}")
        print(f"  {'-'*75}")

        for exit in analysis['exit_reasons']:
            status = "GOOD" if exit['total_pnl'] > 0 else "BAD"
            print(f"  {exit['reason']:<25} {exit['count']:>7} {exit['pct']:>5.1f}% "
                  f"Rs {exit['total_pnl']:>10,.2f} Rs {exit['avg_pnl']:>8,.2f} {exit['win_rate']:>6.1f}% [{status}]")

        # Identify problematic exit reasons
        worst_exit = analysis['exit_reasons'][0]
        if worst_exit['total_pnl'] < -500:
            print(f"\n  >>> MAJOR ISSUE: '{worst_exit['reason']}' exits losing Rs {worst_exit['total_pnl']:,.2f}")
            print(f"      ({worst_exit['count']} trades, {worst_exit['pct']:.1f}% of total)")

            if worst_exit['reason'] == 'stop_loss' or worst_exit['reason'] == 'hard_sl':
                print(f"      RECOMMENDATION: Stops may be too tight or poorly placed")
                print(f"      - Review stop loss logic")
                print(f"      - Check if stops are being run before trade develops")

    # REGIME ANALYSIS
    if 'regimes' in analysis:
        print(f"\nREGIME BREAKDOWN:")
        print(f"  {'Regime':<15} {'Count':>7} {'%':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Win %':>7}")
        print(f"  {'-'*70}")

        for regime in analysis['regimes']:
            status = "GOOD" if regime['total_pnl'] > 0 else "BAD"
            print(f"  {regime['regime']:<15} {regime['count']:>7} {regime['pct']:>5.1f}% "
                  f"Rs {regime['total_pnl']:>10,.2f} Rs {regime['avg_pnl']:>8,.2f} {regime['win_rate']:>6.1f}% [{status}]")

        # Identify problematic regimes
        worst_regime = analysis['regimes'][0]
        if worst_regime['total_pnl'] < -500:
            print(f"\n  >>> MAJOR ISSUE: Setup loses heavily in '{worst_regime['regime']}' regime")
            print(f"      Losing Rs {worst_regime['total_pnl']:,.2f} ({worst_regime['count']} trades)")
            print(f"      RECOMMENDATION: Disable this setup in {worst_regime['regime']} regime")

    # MFE/MAE ANALYSIS
    if 'avg_mfe' in analysis:
        print(f"\nPROFIT POTENTIAL ANALYSIS (MFE/MAE):")
        print(f"  Avg MFE (Max Favorable): Rs {analysis['avg_mfe']:,.2f}")
        print(f"  Avg MAE (Max Adverse): Rs {analysis['avg_mae']:,.2f}")

        if 'avg_mfe_capture' in analysis:
            print(f"  Winners' MFE Capture: {analysis['avg_mfe_capture']:.1f}%")

            if analysis['avg_mfe_capture'] < 50:
                print(f"\n  >>> ISSUE: Exiting winners too early (capturing {analysis['avg_mfe_capture']:.1f}% of potential)")
                print(f"      RECOMMENDATION: Use trailing stops to ride winners longer")

        if 'losses_with_profit_pct' in analysis and analysis['losses_with_profit_pct'] > 0:
            print(f"\n  Losses that showed profit: {analysis['losses_with_profit_count']} ({analysis['losses_with_profit_pct']:.1f}%)")
            print(f"  Avg profit before loss: Rs {analysis['avg_mfe_before_loss']:,.2f}")

            if analysis['losses_with_profit_pct'] > 50:
                print(f"\n  >>> CRITICAL ISSUE: {analysis['losses_with_profit_pct']:.1f}% of losses had profit before turning negative!")
                print(f"      Average wasted profit: Rs {analysis.get('avg_wasted_profit', 0):,.2f}")
                print(f"      RECOMMENDATIONS:")
                print(f"      1. Implement breakeven stops after reaching {analysis['avg_mfe_before_loss']:,.0f} profit")
                print(f"      2. Use trailing stops to lock in profits")
                print(f"      3. Consider partial exits at key levels")

    # DURATION ANALYSIS
    if 'avg_duration' in analysis:
        print(f"\nTRADE DURATION ANALYSIS:")
        print(f"  Average Duration: {analysis['avg_duration']:.1f} minutes")
        print(f"  Winners Duration: {analysis['avg_duration_wins']:.1f} minutes")
        print(f"  Losers Duration: {analysis['avg_duration_losses']:.1f} minutes")

        if analysis['avg_duration_losses'] > 0 and analysis['avg_duration_wins'] > 0:
            ratio = analysis['avg_duration_losses'] / analysis['avg_duration_wins']

            if ratio > 1.5:
                print(f"\n  >>> ISSUE: Holding losers {ratio:.1f}x longer than winners")
                print(f"      RECOMMENDATIONS:")
                print(f"      1. Implement time-based stops (exit if no profit after {analysis['avg_duration_wins']*1.5:.0f} minutes)")
                print(f"      2. Review stop loss placement - may be too wide")
                print(f"      3. Cut losses faster when trade not working")

    # SYMBOL ANALYSIS
    if 'worst_symbols' in analysis:
        print(f"\nWORST PERFORMING SYMBOLS:")
        print(f"  {'Symbol':<15} {'Trades':>7} {'Total P&L':>12} {'Avg P&L':>10}")
        print(f"  {'-'*50}")

        for symbol in analysis['worst_symbols'][:10]:
            print(f"  {symbol['symbol']:<15} {symbol['count']:>7} Rs {symbol['sum']:>10,.2f} Rs {symbol['mean']:>8,.2f}")

        # Check if specific symbols are causing most losses
        worst_symbol = analysis['worst_symbols'][0]
        if worst_symbol['sum'] < -500:
            print(f"\n  >>> ISSUE: {worst_symbol['symbol']} alone losing Rs {worst_symbol['sum']:,.2f}")
            print(f"      RECOMMENDATION: Add filter to exclude {worst_symbol['symbol']} for this setup")

def generate_fix_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate specific, actionable fix recommendations."""

    recs = []

    # Win rate vs breakeven
    if analysis.get('wr_gap', 0) < -5:
        recs.append({
            'category': 'WIN_RATE',
            'severity': 'CRITICAL',
            'issue': f"Win rate {abs(analysis['wr_gap']):.1f}% below breakeven",
            'fixes': [
                f"Add quality filters to improve setup selection",
                f"Tighten entry criteria (better structure confirmation)",
                f"OR improve R:R from {analysis['avg_rr']:.2f} to {1/(analysis['win_rate']/100) - 1:.2f}",
            ]
        })

    # MFE capture issues
    if analysis.get('avg_mfe_capture', 100) < 50:
        recs.append({
            'category': 'EXIT_TIMING',
            'severity': 'HIGH',
            'issue': f"Exiting winners too early ({analysis['avg_mfe_capture']:.1f}% MFE capture)",
            'fixes': [
                "Implement trailing stops (e.g., trail by 0.5R after 1R profit)",
                "Move profit targets further (currently leaving Rs on table)",
                "Use partial exits (50% at T1, trail remainder)",
            ]
        })

    # Profit-before-loss issue
    if analysis.get('losses_with_profit_pct', 0) > 50:
        recs.append({
            'category': 'PROFIT_PROTECTION',
            'severity': 'CRITICAL',
            'issue': f"{analysis['losses_with_profit_pct']:.1f}% of losses had profit first (avg Rs {analysis.get('avg_wasted_profit', 0):,.2f})",
            'fixes': [
                f"Move stop to breakeven after Rs {analysis.get('avg_mfe_before_loss', 0)/2:,.0f} profit",
                "Implement trailing stops immediately after profit",
                f"Partial exit at {analysis.get('avg_mfe_before_loss', 0):,.0f} to lock in gains",
            ]
        })

    # Duration issues
    if analysis.get('avg_duration_losses', 0) > analysis.get('avg_duration_wins', 0) * 1.5:
        ratio = analysis['avg_duration_losses'] / analysis['avg_duration_wins']
        recs.append({
            'category': 'CUT_LOSSES',
            'severity': 'HIGH',
            'issue': f"Holding losers {ratio:.1f}x longer than winners",
            'fixes': [
                f"Add time-based stop: exit if no profit after {analysis['avg_duration_wins']*1.5:.0f} minutes",
                "Review stop loss distance - may be too wide",
                "Implement 'not working' exit criteria",
            ]
        })

    # Regime-specific issues
    if 'regimes' in analysis:
        worst_regime = analysis['regimes'][0]
        if worst_regime['total_pnl'] < -500:
            recs.append({
                'category': 'REGIME_FILTER',
                'severity': 'HIGH',
                'issue': f"Losing Rs {worst_regime['total_pnl']:,.2f} in {worst_regime['regime']} regime",
                'fixes': [
                    f"Disable setup in {worst_regime['regime']} regime",
                    f"OR add regime-specific entry filters for {worst_regime['regime']}",
                    f"OR adjust parameters for {worst_regime['regime']} (wider stops, etc.)",
                ]
            })

    # Exit reason issues
    if 'exit_reasons' in analysis:
        worst_exit = analysis['exit_reasons'][0]
        if worst_exit['total_pnl'] < -500:
            recs.append({
                'category': 'EXIT_LOGIC',
                'severity': 'HIGH',
                'issue': f"'{worst_exit['reason']}' exits losing Rs {worst_exit['total_pnl']:,.2f} ({worst_exit['pct']:.1f}% of trades)",
                'fixes': [
                    f"Review {worst_exit['reason']} logic - may be too aggressive",
                    f"Check if {worst_exit['reason']} placement needs adjustment",
                    f"Analyze why {worst_exit['reason']} leads to losses",
                ]
            })

    return recs

def main():
    backtest_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251117-021749_extracted"

    print("=" * 100)
    print("DEEP FORENSIC ANALYSIS - FAILING SETUPS")
    print("=" * 100)

    # Load all trades
    df = load_backtest_trades(backtest_dir)

    if len(df) == 0:
        print("ERROR: No trades loaded")
        return

    print(f"\nAnalyzing {len(df)} total trades from backtest")

    # Get failing setups
    setup_pnl = df.groupby('setup_type')['pnl'].sum().sort_values()
    failing_setups = setup_pnl[setup_pnl < 0]

    print(f"\nFound {len(failing_setups)} failing setups")
    print(f"Total losses: Rs {failing_setups.sum():,.2f}")

    # Analyze top losers in detail
    print("\n" + "=" * 100)
    print("ANALYZING TOP 5 WORST SETUPS")
    print("=" * 100)

    all_recommendations = {}

    for setup_name in failing_setups.head(5).index:
        analysis = analyze_setup_deeply(df, setup_name)

        if analysis:
            print_forensic_report(analysis)

            # Generate recommendations
            recommendations = generate_fix_recommendations(analysis)
            all_recommendations[setup_name] = recommendations

            # Print recommendations
            if recommendations:
                print(f"\nACTIONABLE FIX RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n  {i}. [{rec['severity']}] {rec['category']}")
                    print(f"     Issue: {rec['issue']}")
                    print(f"     Fixes:")
                    for fix in rec['fixes']:
                        print(f"       - {fix}")

    # Summary of all recommendations
    print("\n" + "=" * 100)
    print("SUMMARY - PRIORITY FIXES")
    print("=" * 100)

    critical_fixes = []
    high_fixes = []

    for setup, recs in all_recommendations.items():
        for rec in recs:
            if rec['severity'] == 'CRITICAL':
                critical_fixes.append((setup, rec))
            elif rec['severity'] == 'HIGH':
                high_fixes.append((setup, rec))

    if critical_fixes:
        print(f"\nCRITICAL FIXES (Fix these first):")
        for setup, rec in critical_fixes:
            print(f"\n  {setup}: {rec['category']}")
            print(f"    {rec['issue']}")
            for fix in rec['fixes']:
                print(f"      - {fix}")

    if high_fixes:
        print(f"\nHIGH PRIORITY FIXES:")
        for setup, rec in high_fixes:
            print(f"\n  {setup}: {rec['category']}")
            print(f"    {rec['issue']}")

if __name__ == "__main__":
    main()
