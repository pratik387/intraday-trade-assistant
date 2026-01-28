"""
Aggregate 6-Month Analysis - Combine all analysis reports into comprehensive summary
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent

# Regime mapping
REGIME_MAP = {
    'run_953e4bdc': ('Correction_RiskOff', 'Oct 2024', 22),
    'run_b441ef53': ('Prolonged_Drawdown', 'Feb 2025', 18),
    'run_77ae5b72': ('Low_Vol_Range', 'Jul 2025', 23),
    'run_5cd6bb35': ('Strong_Uptrend', 'Dec 2023', 20),
    'run_c4c7186a': ('Shock_Down', 'Jan 2024', 23),
    'run_4161cf24': ('Event_Driven_HighVol', 'Jun 2024', 20)
}

def load_latest_report(run_prefix, report_type):
    """Load the latest report of given type for run prefix"""
    pattern = f'{report_type}_{run_prefix}_*.json'
    reports = sorted(ROOT.glob(pattern), reverse=True)

    if not reports:
        return None

    with open(reports[0]) as f:
        return json.load(f)

def main():
    run_prefixes = ['run_953e4bdc', 'run_b441ef53', 'run_77ae5b72',
                    'run_5cd6bb35', 'run_c4c7186a', 'run_4161cf24']

    print("\n" + "=" * 80)
    print("6-MONTH COMPREHENSIVE BACKTEST ANALYSIS")
    print("=" * 80 + "\n")

    # Aggregate metrics
    total_sessions = 0
    total_trades = 0
    total_pnl = 0
    total_decisions = 0
    total_wins = 0
    total_losses = 0

    regime_results = []
    all_spike_tests = []
    all_regime_validations = []
    all_rank_calibrations = []

    for prefix in run_prefixes:
        regime_name, period, expected_sessions = REGIME_MAP[prefix]

        # Load analysis report
        analysis = load_latest_report(prefix, 'analysis_report')
        spike = load_latest_report(prefix, 'spike_test')
        regime_val = load_latest_report(prefix, 'regime_validation')
        rank_cal = load_latest_report(prefix, 'rank_calibration')

        if not analysis:
            print(f"WARNING: No analysis report for {prefix}")
            continue

        perf = analysis.get('performance_summary', {})
        trades = perf.get('total_trades', 0)
        pnl = perf.get('total_pnl', 0)
        wr = perf.get('win_rate', 0)
        decisions = analysis.get('total_decisions', 0)
        sessions = perf.get('sessions', expected_sessions)

        total_sessions += sessions
        total_trades += trades
        total_pnl += pnl
        total_decisions += decisions

        wins = int(trades * wr / 100) if trades > 0 else 0
        losses = trades - wins
        total_wins += wins
        total_losses += losses

        regime_results.append({
            'regime': regime_name,
            'period': period,
            'prefix': prefix,
            'sessions': sessions,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': wr,
            'pnl': pnl,
            'decisions': decisions,
            'spike_test': spike,
            'regime_validation': regime_val,
            'rank_calibration': rank_cal
        })

    # Print summary by regime
    print("PERFORMANCE BY REGIME:")
    print("-" * 80)

    for result in regime_results:
        print(f"\n{result['regime']} ({result['period']}) - {result['sessions']} sessions")
        print(f"  Trades: {result['trades']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Total PnL: Rs.{result['pnl']:.2f}")
        print(f"  Decisions: {result['decisions']}")
        print(f"  Execution Rate: {(result['trades'] / result['decisions'] * 100) if result['decisions'] > 0 else 0:.1f}%")

        # Spike test summary
        if result['spike_test']:
            total_analyzed = len(result['spike_test'])
            sl_exits = sum(1 for t in result['spike_test'] if t.get('actual_exit_reason') == 'hard_sl')
            sl_rate = (sl_exits / total_analyzed * 100) if total_analyzed > 0 else 0
            avg_mfe = sum(t.get('mfe_pct', 0) for t in result['spike_test']) / total_analyzed if total_analyzed > 0 else 0
            avg_mae = sum(t.get('mae_pct', 0) for t in result['spike_test']) / total_analyzed if total_analyzed > 0 else 0

            print(f"  Hard SL Rate: {sl_rate:.1f}%")
            print(f"  Avg MFE: {avg_mfe:.2f}%, Avg MAE: {avg_mae:.2f}%")

        # Regime validation summary
        if result['regime_validation']:
            overall_acc = result['regime_validation'].get('overall_accuracy', 0)
            print(f"  Regime Classification Accuracy: {overall_acc:.1f}%")

        # Rank calibration summary
        if result['rank_calibration']:
            correlation = result['rank_calibration'].get('correlation')
            if correlation is not None:
                print(f"  Rank Score Correlation: {correlation:.3f}")

    # Overall summary
    print(f"\n" + "=" * 80)
    print("OVERALL 6-MONTH PERFORMANCE:")
    print("-" * 80)
    print(f"Total Sessions: {total_sessions}")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {(total_wins / total_trades * 100) if total_trades > 0 else 0:.1f}%")
    print(f"Wins / Losses: {total_wins} / {total_losses}")
    print(f"Total PnL: Rs.{total_pnl:.2f}")
    print(f"Avg PnL per Trade: Rs.{(total_pnl / total_trades) if total_trades > 0 else 0:.2f}")
    print(f"Total Decisions: {total_decisions}")
    print(f"Execution Rate: {(total_trades / total_decisions * 100) if total_decisions > 0 else 0:.1f}%")

    # Calculate profit factor
    total_gross_wins = sum(r['pnl'] for r in regime_results if r['pnl'] > 0)
    total_gross_losses = abs(sum(r['pnl'] for r in regime_results if r['pnl'] < 0))
    profit_factor = (total_gross_wins / total_gross_losses) if total_gross_losses > 0 else 999.99
    print(f"Profit Factor: {profit_factor:.2f}")

    print(f"\n" + "=" * 80)
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
