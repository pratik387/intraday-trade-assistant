"""
Summarize key findings from a comprehensive backtest report.
Optimized for Windows console (no Unicode characters).
"""
import json
import sys
from pathlib import Path

def summarize_report(report_path: str):
    """Summarize the comprehensive backtest report."""

    with open(report_path, 'r') as f:
        report = json.load(f)

    print("=" * 80)
    print("BACKTEST REPORT SUMMARY")
    print("=" * 80)
    print(f"Report: {Path(report_path).name}")
    print(f"Sessions: {report.get('sessions_analyzed', 0)}")
    print(f"Date Range: {report.get('methodology', {}).get('data_quality', {}).get('date_range', 'N/A')}")
    print()

    # 1. Overall Performance
    print("=" * 80)
    print("1. OVERALL PERFORMANCE")
    print("=" * 80)
    perf = report.get('performance_summary', {})
    print(f"Total Trades: {perf.get('total_trades', 0)}")
    print(f"Total PnL: Rs. {perf.get('total_pnl', 0):,.2f}")
    print(f"Win Rate: {perf.get('win_rate', 0):.1f}%")
    print(f"Avg PnL/Trade: Rs. {perf.get('avg_pnl_per_trade', 0):.2f}")
    print(f"Best Trade: Rs. {perf.get('best_trade', 0):,.2f}")
    print(f"Worst Trade: Rs. {perf.get('worst_trade', 0):,.2f}")
    print()

    # 2. Setup Performance
    print("=" * 80)
    print("2. SETUP PERFORMANCE (Sorted by PnL)")
    print("=" * 80)
    setup_analysis = report.get('setup_analysis', {})
    # Sort by total_pnl
    sorted_setups = sorted(setup_analysis.items(),
                          key=lambda x: x[1].get('total_pnl', 0),
                          reverse=True)

    for setup_name, stats in sorted_setups:
        print(f"\n{setup_name.upper()}:")
        print(f"  Trades: {stats.get('total_trades', 0)}")
        print(f"  PnL: Rs. {stats.get('total_pnl', 0):,.2f}")
        print(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
        print(f"  Avg PnL: Rs. {stats.get('avg_pnl', 0):.2f}")
        print(f"  Profit Factor: {stats.get('profit_factor', 0):.2f}")
    print()

    # 3. Regime Performance
    print("=" * 80)
    print("3. REGIME PERFORMANCE (Sorted by PnL)")
    print("=" * 80)
    regime_analysis = report.get('regime_analysis', {})
    sorted_regimes = sorted(regime_analysis.items(),
                           key=lambda x: x[1].get('total_pnl', 0),
                           reverse=True)

    for regime, stats in sorted_regimes:
        print(f"\n{regime.upper()}:")
        print(f"  Trades: {stats.get('total_trades', 0)}")
        print(f"  PnL: Rs. {stats.get('total_pnl', 0):,.2f}")
        print(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
        print(f"  Avg PnL: Rs. {stats.get('avg_pnl', 0):.2f}")
    print()

    # 4. Spike Test Analysis (Targets & Stops)
    print("=" * 80)
    print("4. STOP LOSS & MFE/MAE ANALYSIS")
    print("=" * 80)
    spike_test = report.get('spike_test_analysis', {})
    trades_analyzed = spike_test.get('total_trades_analyzed', 0)

    print(f"Trades Analyzed: {trades_analyzed}")
    print(f"\nHard Stop Loss Rate: {spike_test.get('hard_sl_rate', 0):.1f}%")

    mfe_stats = spike_test.get('mfe_stats', {})
    mae_stats = spike_test.get('mae_stats', {})
    print(f"\nMFE (Maximum Favorable Excursion):")
    print(f"  Average: {mfe_stats.get('avg_mfe_pct', 0):.2f}%")
    print(f"  Median: {mfe_stats.get('median_mfe_pct', 0):.2f}%")
    print(f"  Max: {mfe_stats.get('max_mfe_pct', 0):.2f}%")

    print(f"\nMAE (Maximum Adverse Excursion):")
    print(f"  Average: {mae_stats.get('avg_mae_pct', 0):.2f}%")
    print(f"  Median: {mae_stats.get('median_mae_pct', 0):.2f}%")
    print(f"  Max: {mae_stats.get('max_mae_pct', 0):.2f}%")

    sl_opt = spike_test.get('sl_optimization', {})
    if sl_opt:
        print(f"\nStop Loss Optimization Potential:")
        for stop_width, stats in sl_opt.items():
            if isinstance(stats, dict):
                print(f"  At {stop_width}: {stats.get('saved_trades', 0)} trades saved, Impact: Rs. {stats.get('pnl_impact', 0):,.2f}")
    print()

    # 5. Decision Quality
    print("=" * 80)
    print("5. DECISION QUALITY & MARKET VALIDATION")
    print("=" * 80)
    decision_analysis = report.get('decision_analysis', {})
    overall = decision_analysis.get('overall', {})
    print(f"Total Decisions: {overall.get('total_decisions', 0)}")
    print(f"Unique Symbols: {overall.get('unique_symbols', 0)}")
    print(f"Triggered: {overall.get('triggered_decisions', 0)} ({overall.get('trigger_rate', 0):.1f}%)")

    by_acceptance = decision_analysis.get('by_acceptance', {})
    if by_acceptance:
        print(f"\nBy Acceptance Status:")
        for status in ['excellent', 'good', 'poor']:
            if status in by_acceptance:
                stats = by_acceptance[status]
                total_dec = stats.get('total_decisions', 0)
                trig_count = stats.get('triggered', 0)
                trig_rate = stats.get('trigger_rate', 0)
                print(f"  {status.capitalize()}: {trig_rate:.1f}% trigger rate ({trig_count}/{total_dec} decisions)")

    market_val = report.get('market_validation', {})
    if market_val:
        summary = market_val.get('summary', {})
        print(f"\nMarket Validation:")
        print(f"  Validated Decisions: {summary.get('total_validated', 0)} ({summary.get('validation_rate', 0):.1f}%)")

        quality_vs_perf = market_val.get('quality_vs_performance', {})
        if quality_vs_perf:
            print(f"\n  Quality vs Market Performance:")
            for quality in ['excellent', 'good', 'poor']:
                if quality in quality_vs_perf:
                    stats = quality_vs_perf[quality]
                    avg_hyp_pnl = stats.get('avg_hypothetical_pnl', 0)
                    success = stats.get('success_rate', 0)
                    count = stats.get('count', 0)
                    print(f"    {quality.capitalize()}: {avg_hyp_pnl:.2f}% avg return, {success:.1f}% success ({count} decisions)")
    print()

    # 6. Rejected Trades
    print("=" * 80)
    print("6. REJECTED TRADES ANALYSIS")
    print("=" * 80)
    rejected = report.get('rejected_trades_analysis', {})
    print(f"Total Rejected: {rejected.get('total_rejected', 0)}")
    print(f"Simulated: {rejected.get('simulated_count', 0)}")

    print(f"\nTop Rejection Reasons:")
    top_reasons = rejected.get('top_rejection_reasons', [])
    for i, reason_data in enumerate(top_reasons[:10], 1):
        reason = reason_data.get('reason', 'N/A')
        count = reason_data.get('count', 0)
        pct = reason_data.get('percentage', 0)
        print(f"  {i}. {reason}: {count} ({pct:.1f}%)")

    simulated = rejected.get('simulated_outcomes', {})
    if simulated and simulated.get('count', 0) > 0:
        print(f"\nSimulated Outcomes (if trades were taken):")
        print(f"  Count: {simulated.get('count', 0)}")
        print(f"  Total PnL: Rs. {simulated.get('total_pnl', 0):,.2f}")
        print(f"  Win Rate: {simulated.get('win_rate', 0):.1f}%")
        print(f"  Avg PnL: Rs. {simulated.get('avg_pnl', 0):.2f}")
        print(f"  T1 Hit: {simulated.get('t1_hit_rate', 0):.1f}%")
        print(f"  T2 Hit: {simulated.get('t2_hit_rate', 0):.1f}%")
        print(f"  Hard SL: {simulated.get('hard_sl_rate', 0):.1f}%")
    print()

    # 7. Key Issues & Recommendations
    print("=" * 80)
    print("7. KEY ISSUES & RECOMMENDATIONS")
    print("=" * 80)

    issues_found = False

    # Check stop loss issues
    hard_sl_rate = spike_test.get('hard_sl_rate', 0)
    if hard_sl_rate > 30:
        print(f"\n[CRITICAL] High Hard Stop Loss Rate: {hard_sl_rate:.1f}%")
        print("  Action: Consider widening stops or improving entry timing")
        issues_found = True

    # Check MFE/MAE ratio
    avg_mfe = mfe_stats.get('avg_mfe_pct', 0)
    avg_mae = mae_stats.get('avg_mae_pct', 0)
    if avg_mae > 0:
        mfe_mae_ratio = avg_mfe / avg_mae
        if mfe_mae_ratio < 2:
            print(f"\n[WARNING] Low MFE/MAE Ratio: {mfe_mae_ratio:.2f}")
            print(f"  MFE: {avg_mfe:.2f}%, MAE: {avg_mae:.2f}%")
            print("  Action: Trades not running far enough in favor vs against")
            issues_found = True

    # Check quality calibration
    if 'excellent' in by_acceptance and 'poor' in by_acceptance:
        excellent_stats = by_acceptance['excellent']
        poor_stats = by_acceptance['poor']
        excellent_trig_rate = excellent_stats.get('trigger_rate', 0)
        poor_trig_rate = poor_stats.get('trigger_rate', 0)
        if poor_trig_rate > excellent_trig_rate * 1.5:
            print(f"\n[WARNING] Poor quality triggers more than excellent")
            print(f"  Poor: {poor_trig_rate:.1f}% vs Excellent: {excellent_trig_rate:.1f}%")
            print("  Action: Recalibrate quality filters")
            issues_found = True

    # Check rejected opportunities
    if simulated and simulated.get('count', 0) > 0:
        rejected_pnl = simulated.get('total_pnl', 0)
        executed_pnl = perf.get('total_pnl', 0)
        if rejected_pnl > executed_pnl * 0.3:
            print(f"\n[OPPORTUNITY] Significant PnL in rejected trades")
            print(f"  Rejected potential PnL: Rs. {rejected_pnl:,.2f}")
            print(f"  Executed PnL: Rs. {executed_pnl:,.2f}")
            print(f"  Ratio: {rejected_pnl/executed_pnl*100:.1f}%")
            print("  Action: Review top rejection reasons and consider relaxing filters")
            issues_found = True

    # Check regime performance spread
    if sorted_regimes:
        best_regime_pnl = sorted_regimes[0][1].get('total_pnl', 0)
        worst_regime_pnl = sorted_regimes[-1][1].get('total_pnl', 0)
        if worst_regime_pnl < 0:
            print(f"\n[WARNING] Negative regime performance: {sorted_regimes[-1][0]}")
            print(f"  PnL: Rs. {worst_regime_pnl:,.2f}")
            print("  Action: Consider filtering out trades in this regime")
            issues_found = True

    if not issues_found:
        print("\nNo critical issues identified. System performing well.")

    print()
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python summarize_report.py <report_file>")
        sys.exit(1)

    report_file = sys.argv[1]
    if not Path(report_file).exists():
        print(f"Error: Report file not found: {report_file}")
        sys.exit(1)

    summarize_report(report_file)
