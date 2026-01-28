"""
Analyze a comprehensive backtest report and provide key insights.
"""
import json
import sys
from pathlib import Path

def analyze_report(report_path: str):
    """Analyze the comprehensive backtest report."""

    with open(report_path, 'r') as f:
        report = json.load(f)

    print("=" * 80)
    print("COMPREHENSIVE BACKTEST REPORT ANALYSIS")
    print("=" * 80)
    print()

    # Overall Performance
    print("=" * 80)
    print("1. OVERALL PERFORMANCE")
    print("=" * 80)
    perf = report.get('performance_summary', {})
    print(f"Total Trades: {perf.get('total_trades', 0)}")
    print(f"Total PnL: Rs. {perf.get('total_pnl', 0):,.2f}")
    print(f"Win Rate: {perf.get('win_rate', 0):.1f}%")
    print(f"Average PnL/Trade: Rs. {perf.get('avg_pnl_per_trade', 0):.2f}")
    print(f"Max Drawdown: Rs. {perf.get('max_drawdown', 0):,.2f}")
    print()

    # Setup Performance
    print("=" * 80)
    print("2. SETUP PERFORMANCE")
    print("=" * 80)
    setup_perf = report.get('setup_analysis', {}).get('by_setup', {})
    for setup_name, stats in sorted(setup_perf.items(), key=lambda x: x[1].get('total_pnl', 0), reverse=True):
        print(f"\n{setup_name}:")
        print(f"  PnL: Rs. {stats.get('total_pnl', 0):,.2f}")
        print(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
        print(f"  Trades: {stats.get('trade_count', 0)}")
        print(f"  Avg PnL/Trade: Rs. {stats.get('avg_pnl', 0):.2f}")
    print()

    # Regime Performance
    print("=" * 80)
    print("3. REGIME PERFORMANCE")
    print("=" * 80)
    regime_perf = report.get('regime_analysis', {}).get('by_regime', {})
    for regime, stats in sorted(regime_perf.items(), key=lambda x: x[1].get('total_pnl', 0), reverse=True):
        print(f"\n{regime}:")
        print(f"  PnL: Rs. {stats.get('total_pnl', 0):,.2f}")
        print(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
        print(f"  Trades: {stats.get('trade_count', 0)}")
        print(f"  Avg PnL/Trade: Rs. {stats.get('avg_pnl', 0):.2f}")
    print()

    # Decision Quality Analysis
    print("=" * 80)
    print("4. DECISION QUALITY ANALYSIS")
    print("=" * 80)
    decision_quality = report.get('decision_quality_analysis', {})
    print(f"Total Decisions: {decision_quality.get('total_decisions', 0)}")
    print(f"Triggered Rate: {decision_quality.get('triggered_rate', 0):.1f}%")
    print(f"Unique Symbols: {decision_quality.get('unique_symbols', 0)}")

    print(f"\nBy Acceptance Status:")
    by_status = decision_quality.get('by_acceptance_status', {})
    for status in ['excellent', 'good', 'poor']:
        if status in by_status:
            stats = by_status[status]
            print(f"  {status}: {stats.get('trigger_rate', 0):.1f}% trigger rate ({stats.get('count', 0)} decisions)")
    print()

    # Market Validation
    print("=" * 80)
    print("5. MARKET VALIDATION")
    print("=" * 80)
    market_val = report.get('market_validation', {})
    print(f"Validated Decisions: {market_val.get('validated_count', 0)}/{market_val.get('total_decisions', 0)} ({market_val.get('validation_rate', 0):.1f}%)")

    print(f"\nQuality vs Market Performance:")
    quality_perf = market_val.get('quality_vs_market_performance', {})
    for quality in ['excellent', 'good', 'poor']:
        if quality in quality_perf:
            stats = quality_perf[quality]
            print(f"  {quality}: {stats.get('avg_return', 0):.2f}% avg return ({stats.get('success_rate', 0):.1f}% success rate)")

    missed_opp = market_val.get('missed_opportunities', {})
    print(f"\nMissed Opportunities:")
    print(f"  Non-triggered avg return: {missed_opp.get('non_triggered_avg_return', 0):.2f}%")
    print(f"  Triggered avg return: {missed_opp.get('triggered_avg_return', 0):.2f}%")
    print()

    # Spike Test Analysis (Target & SL Quality)
    print("=" * 80)
    print("6. SPIKE TEST ANALYSIS (TARGET & SL QUALITY)")
    print("=" * 80)
    spike_test = report.get('spike_test_analysis', {})

    print("Target Hit Analysis:")
    target_hits = spike_test.get('target_hit_analysis', {})
    print(f"  T1 Hit Rate: {target_hits.get('t1_hit_rate', 0):.1f}%")
    print(f"  T2 Hit Rate: {target_hits.get('t2_hit_rate', 0):.1f}%")
    print(f"  Both Targets Hit: {target_hits.get('both_hit_rate', 0):.1f}%")

    print(f"\nStop Loss Analysis:")
    sl_analysis = spike_test.get('stop_loss_analysis', {})
    print(f"  Hard SL Hit Rate: {sl_analysis.get('hard_sl_hit_rate', 0):.1f}%")
    print(f"  Hard SL After T1: {sl_analysis.get('hard_sl_after_t1_rate', 0):.1f}%")
    print(f"  Hard SL After T2: {sl_analysis.get('hard_sl_after_t2_rate', 0):.1f}%")

    print(f"\nMFE/MAE Analysis:")
    mfe_mae = spike_test.get('mfe_mae_analysis', {})
    print(f"  Avg MFE: {mfe_mae.get('avg_mfe', 0):.2f}%")
    print(f"  Avg MAE: {mfe_mae.get('avg_mae', 0):.2f}%")
    print(f"  MFE/MAE Ratio: {mfe_mae.get('mfe_mae_ratio', 0):.2f}")
    print()

    # Rejected Trades Analysis
    print("=" * 80)
    print("7. REJECTED TRADES ANALYSIS (MISSED OPPORTUNITIES)")
    print("=" * 80)
    rejected = report.get('rejected_trades_analysis', {})
    print(f"Total Rejected: {rejected.get('total_rejected', 0)}")
    print(f"Simulated: {rejected.get('simulated_count', 0)}")

    print(f"\nTop Rejection Reasons:")
    top_reasons = rejected.get('top_rejection_reasons', [])
    for reason in top_reasons[:5]:
        print(f"  {reason.get('reason', 'N/A')}: {reason.get('count', 0)} ({reason.get('percentage', 0):.1f}%)")

    simulated = rejected.get('simulated_outcomes', {})
    if simulated:
        print(f"\nSimulated Outcomes (if trades were taken):")
        print(f"  Total PnL: Rs. {simulated.get('total_pnl', 0):,.2f}")
        print(f"  Win Rate: {simulated.get('win_rate', 0):.1f}%")
        print(f"  Avg PnL: Rs. {simulated.get('avg_pnl', 0):.2f}")
        print(f"  T1 Hit Rate: {simulated.get('t1_hit_rate', 0):.1f}%")
        print(f"  T2 Hit Rate: {simulated.get('t2_hit_rate', 0):.1f}%")
    print()

    # Key Insights Section
    print("=" * 80)
    print("8. KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 80)

    # Analyze stop loss issues
    hard_sl_after_t1 = sl_analysis.get('hard_sl_after_t1_rate', 0)
    if hard_sl_after_t1 > 30:
        print(f"\nWARNING: High Hard SL After T1 Rate ({hard_sl_after_t1:.1f}%)")
        print("  → Consider widening stops after T1 or adjusting exit management")

    # Analyze target capture
    t1_hit = target_hits.get('t1_hit_rate', 0)
    t2_hit = target_hits.get('t2_hit_rate', 0)
    if t1_hit < 50:
        print(f"\nWARNING: Low T1 Hit Rate ({t1_hit:.1f}%)")
        print("  → Consider more conservative T1 targets or improving entry timing")
    if t2_hit < 30:
        print(f"\nWARNING: Low T2 Hit Rate ({t2_hit:.1f}%)")
        print("  → T2 targets may be too aggressive")

    # Analyze quality calibration
    excellent_trigger = by_status.get('excellent', {}).get('trigger_rate', 0)
    poor_trigger = by_status.get('poor', {}).get('trigger_rate', 0)
    if poor_trigger > excellent_trigger:
        print(f"\nWARNING: Poor quality decisions trigger more than excellent ({poor_trigger:.1f}% vs {excellent_trigger:.1f}%)")
        print("  → Quality filters may need recalibration")

    # Analyze rejected trades
    rejected_pnl = simulated.get('total_pnl', 0)
    if rejected_pnl > perf.get('total_pnl', 0) * 0.5:
        print(f"\nOPPORTUNITY: Rejected trades have significant potential PnL (Rs. {rejected_pnl:,.2f})")
        print("  → Review top rejection reasons and consider relaxing some filters")

    print()
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_report.py <report_file>")
        sys.exit(1)

    report_file = sys.argv[1]
    if not Path(report_file).exists():
        print(f"Error: Report file not found: {report_file}")
        sys.exit(1)

    analyze_report(report_file)
