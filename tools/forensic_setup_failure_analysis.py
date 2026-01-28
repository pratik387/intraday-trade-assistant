"""
Forensic analysis of failing setups to understand WHY they're losing money.

For each failing setup, analyze:
1. Regime distribution (which regimes are they losing in?)
2. Entry/exit characteristics (where are trades entering/exiting?)
3. Stop loss patterns (are stops being hit? Too tight/wide?)
4. Win/loss size distribution (are losses disproportionately large?)
5. Time patterns (time of day issues?)
6. Symbol patterns (certain stocks causing losses?)
7. Trade duration (holding too long/short?)
8. MFE/MAE analysis (is there edge being lost?)
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, Counter
from datetime import datetime

def load_all_session_data(base_dir: str) -> pd.DataFrame:
    """Load all trade data from session directories."""
    base_path = Path(base_dir)
    all_trades = []

    # Find all session directories
    session_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('2023') or d.name.startswith('2024')]

    print(f"Found {len(session_dirs)} session directories")

    for session_dir in sorted(session_dirs):
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
                except:
                    continue

    print(f"Loaded {len(all_trades)} total trades")
    return pd.DataFrame(all_trades)

def analyze_failing_setup(df: pd.DataFrame, setup_name: str) -> Dict[str, Any]:
    """Deep analysis of why a specific setup is failing."""

    setup_trades = df[df['setup_type'] == setup_name].copy()

    if len(setup_trades) == 0:
        return {'error': 'No trades found for this setup'}

    analysis = {
        'setup_name': setup_name,
        'total_trades': len(setup_trades),
        'total_pnl': setup_trades['pnl'].sum(),
        'avg_pnl': setup_trades['pnl'].mean(),
    }

    # Win/loss breakdown
    wins = setup_trades[setup_trades['pnl'] > 0]
    losses = setup_trades[setup_trades['pnl'] < 0]
    breakeven = setup_trades[setup_trades['pnl'] == 0]

    analysis['wins'] = len(wins)
    analysis['losses'] = len(losses)
    analysis['breakeven'] = len(breakeven)
    analysis['win_rate'] = len(wins) / len(setup_trades) * 100 if len(setup_trades) > 0 else 0

    # Win/loss size analysis
    if len(wins) > 0:
        analysis['avg_win'] = wins['pnl'].mean()
        analysis['max_win'] = wins['pnl'].max()
        analysis['total_wins_pnl'] = wins['pnl'].sum()
    else:
        analysis['avg_win'] = 0
        analysis['max_win'] = 0
        analysis['total_wins_pnl'] = 0

    if len(losses) > 0:
        analysis['avg_loss'] = losses['pnl'].mean()
        analysis['max_loss'] = losses['pnl'].min()
        analysis['total_losses_pnl'] = losses['pnl'].sum()
    else:
        analysis['avg_loss'] = 0
        analysis['max_loss'] = 0
        analysis['total_losses_pnl'] = 0

    # Risk/reward ratio
    if analysis['avg_loss'] != 0:
        analysis['avg_rr'] = abs(analysis['avg_win'] / analysis['avg_loss'])
    else:
        analysis['avg_rr'] = 0

    # Regime distribution
    if 'regime' in setup_trades.columns:
        regime_stats = setup_trades.groupby('regime').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2)
        analysis['regime_breakdown'] = regime_stats.to_dict()

    # Exit reason analysis
    if 'exit_reason' in setup_trades.columns:
        exit_reasons = setup_trades.groupby('exit_reason').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2)
        analysis['exit_reason_breakdown'] = exit_reasons.to_dict()

    # Direction analysis (if applicable)
    if 'direction' in setup_trades.columns:
        direction_stats = setup_trades.groupby('direction').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2)
        analysis['direction_breakdown'] = direction_stats.to_dict()

    # Time of day analysis (hour of entry)
    if 'entry_time' in setup_trades.columns:
        setup_trades['entry_hour'] = pd.to_datetime(setup_trades['entry_time']).dt.hour
        hour_stats = setup_trades.groupby('entry_hour').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2)
        analysis['hourly_breakdown'] = hour_stats.to_dict()

    # Trade duration analysis (if we have entry and exit times)
    if 'entry_time' in setup_trades.columns and 'exit_time' in setup_trades.columns:
        setup_trades['duration_minutes'] = (
            pd.to_datetime(setup_trades['exit_time']) - pd.to_datetime(setup_trades['entry_time'])
        ).dt.total_seconds() / 60

        analysis['avg_duration_minutes'] = setup_trades['duration_minutes'].mean()
        analysis['duration_winners'] = wins['duration_minutes'].mean() if len(wins) > 0 else 0
        analysis['duration_losers'] = losses['duration_minutes'].mean() if len(losses) > 0 else 0

    # MFE/MAE analysis (if available)
    if 'mfe' in setup_trades.columns and 'mae' in setup_trades.columns:
        analysis['avg_mfe'] = setup_trades['mfe'].mean()
        analysis['avg_mae'] = setup_trades['mae'].mean()

        # For winners: how much MFE did they achieve vs final P&L?
        if len(wins) > 0:
            wins_copy = wins.copy()
            wins_copy['mfe_capture'] = wins_copy['pnl'] / wins_copy['mfe'] * 100
            analysis['avg_mfe_capture_rate'] = wins_copy['mfe_capture'].mean()

        # For losers: how much MFE was there before loss?
        if len(losses) > 0:
            analysis['avg_mfe_before_loss'] = losses['mfe'].mean()
            analysis['trades_with_profit_before_loss'] = len(losses[losses['mfe'] > 0])
            analysis['pct_trades_with_profit_before_loss'] = len(losses[losses['mfe'] > 0]) / len(losses) * 100

    # Symbol analysis - which symbols are causing losses?
    if 'symbol' in setup_trades.columns:
        symbol_stats = setup_trades.groupby('symbol').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2)

        # Get worst performing symbols
        symbol_stats_sorted = symbol_stats.sort_values(('pnl', 'sum'))
        analysis['worst_symbols'] = symbol_stats_sorted.head(10).to_dict()
        analysis['best_symbols'] = symbol_stats_sorted.tail(10).to_dict()

    return analysis

def print_setup_forensics(analysis: Dict[str, Any]):
    """Print detailed forensic analysis for a setup."""

    print("\n" + "=" * 100)
    print(f"FORENSIC ANALYSIS: {analysis['setup_name']}")
    print("=" * 100)

    # Overall stats
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Trades: {analysis['total_trades']}")
    print(f"  Total P&L: Rs {analysis['total_pnl']:,.2f}")
    print(f"  Avg P&L per Trade: Rs {analysis['avg_pnl']:,.2f}")
    print(f"  Win Rate: {analysis['win_rate']:.1f}%")
    print(f"  Wins: {analysis['wins']} | Losses: {analysis['losses']} | Breakeven: {analysis['breakeven']}")

    # Win/Loss characteristics
    print(f"\nWIN/LOSS CHARACTERISTICS:")
    print(f"  Average Win: Rs {analysis['avg_win']:,.2f}")
    print(f"  Average Loss: Rs {analysis['avg_loss']:,.2f}")
    print(f"  Largest Win: Rs {analysis['max_win']:,.2f}")
    print(f"  Largest Loss: Rs {analysis['max_loss']:,.2f}")
    print(f"  Risk/Reward Ratio: {analysis['avg_rr']:.2f}")

    if analysis['avg_win'] != 0 and analysis['avg_loss'] != 0:
        # Breakeven win rate calculation
        breakeven_wr = abs(analysis['avg_loss']) / (abs(analysis['avg_loss']) + analysis['avg_win']) * 100
        print(f"  Breakeven Win Rate Needed: {breakeven_wr:.1f}%")
        print(f"  Current Win Rate: {analysis['win_rate']:.1f}%")

        if analysis['win_rate'] < breakeven_wr:
            deficit = breakeven_wr - analysis['win_rate']
            print(f"  >>> WIN RATE DEFICIT: {deficit:.1f}% (need to improve win rate OR risk/reward)")
        else:
            surplus = analysis['win_rate'] - breakeven_wr
            print(f"  >>> WIN RATE SURPLUS: {surplus:.1f}% (should be profitable, check for outliers)")

    # Regime breakdown
    if 'regime_breakdown' in analysis and analysis['regime_breakdown']:
        print(f"\nREGIME BREAKDOWN:")
        regime_data = analysis['regime_breakdown']

        if ('pnl', 'count') in regime_data:
            print(f"  {'Regime':<15} {'Trades':>8} {'Total P&L':>12} {'Avg P&L':>10}")
            print(f"  {'-'*50}")

            for regime in regime_data[('pnl', 'count')].keys():
                count = regime_data[('pnl', 'count')][regime]
                total_pnl = regime_data[('pnl', 'sum')][regime]
                avg_pnl = regime_data[('pnl', 'mean')][regime]

                status = "PROFIT" if total_pnl > 0 else "LOSS"
                print(f"  {regime:<15} {count:>8} Rs {total_pnl:>10,.2f} Rs {avg_pnl:>8,.2f} [{status}]")

    # Exit reason breakdown
    if 'exit_reason_breakdown' in analysis and analysis['exit_reason_breakdown']:
        print(f"\nEXIT REASON BREAKDOWN:")
        exit_data = analysis['exit_reason_breakdown']

        if ('pnl', 'count') in exit_data:
            print(f"  {'Exit Reason':<20} {'Trades':>8} {'Total P&L':>12} {'Avg P&L':>10}")
            print(f"  {'-'*55}")

            for reason in exit_data[('pnl', 'count')].keys():
                count = exit_data[('pnl', 'count')][reason]
                total_pnl = exit_data[('pnl', 'sum')][reason]
                avg_pnl = exit_data[('pnl', 'mean')][reason]

                print(f"  {reason:<20} {count:>8} Rs {total_pnl:>10,.2f} Rs {avg_pnl:>8,.2f}")

    # MFE/MAE analysis
    if 'avg_mfe' in analysis:
        print(f"\nEXCURSION ANALYSIS (MFE/MAE):")
        print(f"  Average MFE (Max Favorable): Rs {analysis['avg_mfe']:,.2f}")
        print(f"  Average MAE (Max Adverse): Rs {analysis['avg_mae']:,.2f}")

        if 'avg_mfe_capture_rate' in analysis:
            print(f"  Winners' MFE Capture Rate: {analysis['avg_mfe_capture_rate']:.1f}%")

            if analysis['avg_mfe_capture_rate'] < 50:
                print(f"  >>> LOW CAPTURE RATE: Exiting winners too early (capturing <50% of potential)")

        if 'avg_mfe_before_loss' in analysis and analysis['avg_mfe_before_loss'] > 0:
            print(f"  Average MFE before Loss: Rs {analysis['avg_mfe_before_loss']:,.2f}")
            print(f"  Trades that were profitable before turning to loss: {analysis['trades_with_profit_before_loss']} ({analysis['pct_trades_with_profit_before_loss']:.1f}%)")

            if analysis['pct_trades_with_profit_before_loss'] > 50:
                print(f"  >>> MAJOR ISSUE: {analysis['pct_trades_with_profit_before_loss']:.1f}% of losses had profit first!")
                print(f"  >>> RECOMMENDATION: Tighten profit-taking or use trailing stops")

    # Duration analysis
    if 'avg_duration_minutes' in analysis:
        print(f"\nTRADE DURATION ANALYSIS:")
        print(f"  Average Duration: {analysis['avg_duration_minutes']:.1f} minutes")
        print(f"  Winners Duration: {analysis['duration_winners']:.1f} minutes")
        print(f"  Losers Duration: {analysis['duration_losers']:.1f} minutes")

        if analysis['duration_losers'] > analysis['duration_winners'] * 1.5:
            print(f"  >>> HOLDING LOSERS TOO LONG: Losers held {analysis['duration_losers']/analysis['duration_winners']:.1f}x longer than winners")
            print(f"  >>> RECOMMENDATION: Cut losses faster (time-based stop or tighter stop loss)")

    # Worst symbols
    if 'worst_symbols' in analysis:
        print(f"\nWORST PERFORMING SYMBOLS (Top 10):")
        worst = analysis['worst_symbols']

        if ('pnl', 'count') in worst:
            print(f"  {'Symbol':<15} {'Trades':>8} {'Total P&L':>12} {'Avg P&L':>10}")
            print(f"  {'-'*50}")

            symbols = list(worst[('pnl', 'count')].keys())[:10]
            for symbol in symbols:
                count = worst[('pnl', 'count')][symbol]
                total_pnl = worst[('pnl', 'sum')][symbol]
                avg_pnl = worst[('pnl', 'mean')][symbol]

                print(f"  {symbol:<15} {count:>8} Rs {total_pnl:>10,.2f} Rs {avg_pnl:>8,.2f}")

def generate_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations based on analysis."""

    recommendations = []

    # Win rate vs breakeven analysis
    if analysis['avg_win'] != 0 and analysis['avg_loss'] != 0:
        breakeven_wr = abs(analysis['avg_loss']) / (abs(analysis['avg_loss']) + analysis['avg_win']) * 100

        if analysis['win_rate'] < breakeven_wr:
            deficit = breakeven_wr - analysis['win_rate']
            recommendations.append(f"Win rate is {deficit:.1f}% below breakeven - need to either:")
            recommendations.append(f"  - Improve win rate from {analysis['win_rate']:.1f}% to {breakeven_wr:.1f}%")
            recommendations.append(f"  - OR improve risk/reward from {analysis['avg_rr']:.2f} to {1/(analysis['win_rate']/100) - 1:.2f}")

    # Risk/reward issues
    if analysis['avg_rr'] < 1.5:
        recommendations.append(f"Poor risk/reward ratio ({analysis['avg_rr']:.2f}) - need R:R > 1.5 for this win rate")
        recommendations.append(f"  - Consider widening profit targets")
        recommendations.append(f"  - OR tightening stop losses")

    # MFE capture issues
    if 'avg_mfe_capture_rate' in analysis and analysis['avg_mfe_capture_rate'] < 50:
        recommendations.append(f"Low MFE capture rate ({analysis['avg_mfe_capture_rate']:.1f}%) - exiting winners too early")
        recommendations.append(f"  - Consider trailing stops to ride winners longer")
        recommendations.append(f"  - Review profit target placement")

    # Profit before loss issue
    if 'pct_trades_with_profit_before_loss' in analysis and analysis['pct_trades_with_profit_before_loss'] > 50:
        recommendations.append(f"CRITICAL: {analysis['pct_trades_with_profit_before_loss']:.1f}% of losses were profitable first!")
        recommendations.append(f"  - Implement breakeven stops after reaching MFE threshold")
        recommendations.append(f"  - Use trailing stops to lock in profits")

    # Duration issues
    if 'duration_losers' in analysis and 'duration_winners' in analysis:
        if analysis['duration_losers'] > analysis['duration_winners'] * 1.5:
            recommendations.append(f"Holding losers too long ({analysis['duration_losers']:.0f}m vs {analysis['duration_winners']:.0f}m for winners)")
            recommendations.append(f"  - Implement time-based stops (e.g., exit if no profit after 60 minutes)")
            recommendations.append(f"  - Review stop loss placement - may be too wide")

    # Regime-specific issues
    if 'regime_breakdown' in analysis and ('pnl', 'sum') in analysis['regime_breakdown']:
        regime_pnls = analysis['regime_breakdown'][('pnl', 'sum')]
        worst_regime = min(regime_pnls.items(), key=lambda x: x[1])

        if worst_regime[1] < -500:  # Significant loss in one regime
            recommendations.append(f"Setup loses heavily in {worst_regime[0]} regime (Rs {worst_regime[1]:,.2f})")
            recommendations.append(f"  - Consider disabling this setup in {worst_regime[0]} regime")
            recommendations.append(f"  - OR add regime-specific filters/parameters")

    return recommendations

def main():
    # Load report to get failing setups
    report_path = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\analysis\reports\misc\analysis_report_20_20251117_104148.json"

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    setup_analysis = report.get('setup_analysis', {})

    # Identify failing setups (P&L < 0)
    failing_setups = []
    for setup_name, stats in setup_analysis.items():
        if stats.get('total_pnl', 0) < 0:
            failing_setups.append({
                'name': setup_name,
                'pnl': stats.get('total_pnl', 0),
                'trades': stats.get('total_trades', 0)
            })

    # Sort by P&L (worst first)
    failing_setups.sort(key=lambda x: x['pnl'])

    print("=" * 100)
    print("FORENSIC ANALYSIS OF FAILING SETUPS")
    print("=" * 100)
    print(f"\nFound {len(failing_setups)} failing setups")
    print(f"Total losses from failing setups: Rs {sum(s['pnl'] for s in failing_setups):,.2f}")

    # For now, we'll use the report data since we don't have access to individual trade files
    # A more detailed analysis would require loading all analytics.jsonl files

    print(f"\nNOTE: This analysis is based on summary data from the report.")
    print(f"For detailed trade-by-trade forensics, we need access to individual session analytics.jsonl files.")
    print(f"\nTo enable deeper analysis, provide path to backtest directory with session folders.")

    # Show what detailed analysis would include
    print(f"\n" + "=" * 100)
    print("DETAILED FORENSIC ANALYSIS CAPABILITIES (when trade data available):")
    print("=" * 100)
    print("""
1. REGIME ANALYSIS
   - Which regimes cause losses for each setup
   - Regime-specific win rates and avg P&L
   - Recommendation: Disable in specific regimes or adjust parameters

2. EXIT REASON ANALYSIS
   - How are trades exiting? (stop_loss, target_1, target_2, timeout, etc.)
   - Which exit reasons are profitable vs losing
   - Recommendation: Adjust exit strategy

3. MFE/MAE ANALYSIS (Max Favorable/Adverse Excursion)
   - How much profit was available before exit?
   - Are we exiting winners too early?
   - Are losers showing profit before turning negative?
   - Recommendation: Improve exit timing, trailing stops

4. TRADE DURATION ANALYSIS
   - Are we holding losers longer than winners?
   - Time-based patterns (e.g., morning vs afternoon trades)
   - Recommendation: Time-based exits, faster loss cutting

5. SYMBOL ANALYSIS
   - Which specific stocks are causing losses?
   - Stock characteristics (sector, volatility, liquidity)
   - Recommendation: Filter out problematic symbols

6. ENTRY PRICE ANALYSIS
   - Entry quality (distance from structure levels)
   - Slippage analysis
   - Recommendation: Improve entry criteria

7. RISK/REWARD ANALYSIS
   - Actual R:R achieved vs planned
   - Breakeven win rate vs actual win rate
   - Recommendation: Adjust targets and stops
    """)

if __name__ == "__main__":
    main()
