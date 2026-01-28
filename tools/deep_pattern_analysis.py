#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Pattern Analysis - Professional Indian Trader Perspective
Analyzes report + logs to find opportunities to:
1. Reduce SL hits (increase WR)
2. Increase PnL per trade
3. Generate more trading opportunities
"""

import json
import sys
import io
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import pandas as pd

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def load_report(report_path: Path) -> Dict:
    """Load the comprehensive analysis report."""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_setup_quality_gaps(report: Dict):
    """Find setups with good WR but low volume - opportunity for more trades."""
    print("=" * 100)
    print("OPPORTUNITY #1: HIGH-WR SETUPS WITH LOW VOLUME")
    print("=" * 100)
    print()
    print("These setups are working well but underutilized - we need MORE of these trades:")
    print()

    setup_data = []
    for setup_name, stats in report['setup_analysis'].items():
        if stats['total_trades'] >= 3:  # Minimum statistical significance
            wr = stats['win_rate']
            pf = stats['profit_factor']
            count = stats['total_trades']
            avg_pnl = stats['avg_pnl']

            # High quality = WR > 60% OR PF > 2.0
            is_high_quality = wr > 60 or pf > 2.0
            is_low_volume = count < 20  # Less than 20 trades in 130 sessions

            if is_high_quality and is_low_volume:
                setup_data.append({
                    'setup': setup_name,
                    'count': count,
                    'wr': wr,
                    'pf': pf,
                    'avg_pnl': avg_pnl,
                    'total_pnl': stats['total_pnl']
                })

    # Sort by profit factor
    setup_data.sort(key=lambda x: x['pf'], reverse=True)

    for i, data in enumerate(setup_data, 1):
        print(f"{i}. {data['setup']}")
        print(f"   Trades: {data['count']} (LOW - Need MORE!)")
        print(f"   WR: {data['wr']:.1f}%")
        print(f"   PF: {data['pf']:.2f}")
        print(f"   Avg PnL: Rs.{data['avg_pnl']:.2f}")
        print(f"   Total PnL: Rs.{data['total_pnl']:.2f}")
        print(f"   → ACTION: Relax filters/improve detection to generate more {data['setup']} opportunities")
        print()

def analyze_sl_heavy_setups(report: Dict, logs_dir: Path):
    """Find setups with high SL rate - opportunity to reduce SL hits."""
    print("=" * 100)
    print("OPPORTUNITY #2: REDUCE SL HITS ON THESE SETUPS")
    print("=" * 100)
    print()
    print("These setups have potential but SL issues - need better stops or filters:")
    print()

    # Parse all analytics.jsonl to get exit reasons
    setup_exits = defaultdict(lambda: {'total': 0, 'hard_sl': 0, 'targets': 0, 'eod': 0})

    for session_dir in sorted(logs_dir.glob("2*")):
        if not session_dir.is_dir():
            continue

        analytics_file = session_dir / "analytics.jsonl"
        if not analytics_file.exists():
            continue

        with open(analytics_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    trade = json.loads(line)
                    setup = trade.get('setup_type', '')
                    exit_reason = trade.get('exit_reason', '')

                    setup_exits[setup]['total'] += 1

                    if 'hard_sl' in exit_reason:
                        setup_exits[setup]['hard_sl'] += 1
                    elif 'target' in exit_reason:
                        setup_exits[setup]['targets'] += 1
                    elif 'eod' in exit_reason or 'squareoff' in exit_reason:
                        setup_exits[setup]['eod'] += 1
                except:
                    continue

    # Analyze SL rates
    sl_issues = []
    for setup, exits in setup_exits.items():
        if exits['total'] >= 5:
            sl_rate = (exits['hard_sl'] / exits['total']) * 100

            if sl_rate > 40:  # More than 40% SL hits
                setup_stats = report['setup_analysis'].get(setup, {})
                sl_issues.append({
                    'setup': setup,
                    'total': exits['total'],
                    'sl_rate': sl_rate,
                    'sl_count': exits['hard_sl'],
                    'wr': setup_stats.get('win_rate', 0),
                    'avg_loser': setup_stats.get('avg_loser', 0)
                })

    sl_issues.sort(key=lambda x: x['sl_rate'], reverse=True)

    for i, data in enumerate(sl_issues, 1):
        print(f"{i}. {data['setup']}")
        print(f"   Total Trades: {data['total']}")
        print(f"   SL Rate: {data['sl_rate']:.1f}% ({data['sl_count']} hard_sl hits)")
        print(f"   Win Rate: {data['wr']:.1f}%")
        print(f"   Avg Loser: Rs.{data['avg_loser']:.2f}")
        print(f"   → ACTION: Widen stops OR add pre-trade filters to avoid choppy conditions")
        print()

def analyze_regime_opportunities(report: Dict):
    """Find regime-specific opportunities."""
    print("=" * 100)
    print("OPPORTUNITY #3: REGIME-SPECIFIC OPTIMIZATION")
    print("=" * 100)
    print()

    regime_data = report['regime_analysis']

    print("BEST PERFORMING REGIMES (Focus here):")
    print()

    regimes = []
    for regime_name, stats in regime_data.items():
        if stats['total_trades'] >= 10:
            regimes.append({
                'name': regime_name,
                'count': stats['total_trades'],
                'wr': stats['win_rate'],
                'pnl': stats['total_pnl'],
                'avg_pnl': stats['avg_pnl']
            })

    regimes.sort(key=lambda x: x['avg_pnl'], reverse=True)

    for i, data in enumerate(regimes, 1):
        print(f"{i}. {data['name'].upper()}")
        print(f"   Trades: {data['count']}")
        print(f"   WR: {data['wr']:.1f}%")
        print(f"   Total PnL: Rs.{data['pnl']:.2f}")
        print(f"   Avg PnL/Trade: Rs.{data['avg_pnl']:.2f}")

        if data['wr'] > 55:
            print(f"   → ACTION: This regime is STRONG - generate MORE trades here")
        elif data['wr'] < 45:
            print(f"   → ACTION: This regime is WEAK - reduce trade count or improve filters")
        print()

def analyze_timing_patterns(report: Dict):
    """Find optimal trading hours."""
    print("=" * 100)
    print("OPPORTUNITY #4: TIME-BASED FILTERING")
    print("=" * 100)
    print()

    if 'timing_analysis' in report:
        timing = report['timing_analysis']

        if 'hourly_performance' in timing:
            print("HOURLY PERFORMANCE (IST):")
            print()

            hourly_data = []
            for hour_str, stats in timing['hourly_performance'].items():
                if stats['trade_count'] >= 3:
                    hourly_data.append({
                        'hour': hour_str,
                        'count': stats['trade_count'],
                        'wr': stats['win_rate'],
                        'avg_pnl': stats['avg_pnl']
                    })

            hourly_data.sort(key=lambda x: x['avg_pnl'], reverse=True)

            for i, data in enumerate(hourly_data, 1):
                print(f"{i}. {data['hour']}")
                print(f"   Trades: {data['count']}")
                print(f"   WR: {data['wr']:.1f}%")
                print(f"   Avg PnL: Rs.{data['avg_pnl']:.2f}")

                if data['avg_pnl'] > 150:
                    print(f"   → ACTION: GOLDEN HOUR - prioritize trades in this time")
                elif data['avg_pnl'] < 0:
                    print(f"   → ACTION: AVOID trades in this time (losing hour)")
                print()

def analyze_indicator_correlations(report: Dict):
    """Find which indicators predict success."""
    print("=" * 100)
    print("OPPORTUNITY #5: INDICATOR-BASED FILTERING")
    print("=" * 100)
    print()

    if 'indicator_analysis' not in report:
        print("No indicator analysis available")
        return

    indicators = report['indicator_analysis']

    print("STRONG PREDICTIVE INDICATORS (Use these for filtering):")
    print()

    # Look for indicators with strong correlation to returns
    strong_indicators = []

    for indicator_name, data in indicators.items():
        if isinstance(data, dict) and 'correlation' in data:
            corr = data.get('correlation', 0)

            if abs(corr) > 0.15:  # Meaningful correlation
                strong_indicators.append({
                    'name': indicator_name,
                    'correlation': corr,
                    'data': data
                })

    strong_indicators.sort(key=lambda x: abs(x['correlation']), reverse=True)

    for i, ind in enumerate(strong_indicators[:10], 1):
        print(f"{i}. {ind['name']}")
        print(f"   Correlation: {ind['correlation']:.3f}")

        if ind['correlation'] > 0:
            print(f"   → ACTION: PREFER trades when {ind['name']} is HIGH")
        else:
            print(f"   → ACTION: PREFER trades when {ind['name']} is LOW")

        # Show best quartile if available
        if 'quartile_analysis' in ind['data']:
            quartiles = ind['data']['quartile_analysis']
            best_q = max(quartiles.items(), key=lambda x: x[1].get('avg_return', 0))
            print(f"   Best Quartile: {best_q[0]} (Avg Return: {best_q[1].get('avg_return', 0):.2f}%)")
        print()

def analyze_quality_calibration(report: Dict):
    """Check if rank_score system is working."""
    print("=" * 100)
    print("OPPORTUNITY #6: RANK SCORE CALIBRATION")
    print("=" * 100)
    print()

    if 'quality_calibration' not in report:
        print("No quality calibration data available")
        return

    cal = report['quality_calibration']

    print("RANK SCORE EFFECTIVENESS:")
    print()

    if 'decile_analysis' in cal:
        print("Performance by Rank Score Decile:")
        print()

        deciles = []
        for decile_name, stats in cal['decile_analysis'].items():
            if isinstance(stats, dict):
                deciles.append({
                    'decile': decile_name,
                    'count': stats.get('decision_count', 0),
                    'trigger_rate': stats.get('trigger_rate', 0),
                    'market_success': stats.get('market_success_rate', 0)
                })

        # Sort by decile number
        deciles.sort(key=lambda x: x['decile'])

        for data in deciles:
            print(f"  {data['decile']}: {data['count']} decisions")
            print(f"    Trigger Rate: {data['trigger_rate']:.1f}%")
            print(f"    Market Success: {data['market_success']:.1f}%")

        print()

        # Check if higher deciles perform better
        top_decile = next((d for d in deciles if 'D10' in d['decile']), None)
        bottom_decile = next((d for d in deciles if 'D1' in d['decile']), None)

        if top_decile and bottom_decile:
            if top_decile['market_success'] > bottom_decile['market_success'] + 10:
                print("✓ RANK SCORE IS WORKING - Top decile outperforms bottom by >10%")
                print(f"  → ACTION: Raise threshold to focus on top deciles only")
            else:
                print("✗ RANK SCORE NOT PREDICTIVE - Top and bottom deciles similar")
                print(f"  → ACTION: Recalibrate rank_score formula or disable ranker")
        print()

def analyze_missed_opportunities(report: Dict):
    """Find profitable setups we're rejecting."""
    print("=" * 100)
    print("OPPORTUNITY #7: MISSED PROFITABLE TRADES")
    print("=" * 100)
    print()

    if 'decision_analysis' not in report:
        print("No decision analysis available")
        return

    decisions = report['decision_analysis']

    if 'by_acceptance_status' in decisions:
        print("ACCEPTANCE STATUS BREAKDOWN:")
        print()

        for status, stats in decisions['by_acceptance_status'].items():
            if isinstance(stats, dict):
                print(f"{status.upper()}:")
                print(f"  Total Decisions: {stats.get('total_decisions', 0)}")
                print(f"  Triggered: {stats.get('triggered_trades', 0)}")
                print(f"  Trigger Rate: {stats.get('trigger_rate', 0):.1f}%")

                if 'market_validation' in stats:
                    mv = stats['market_validation']
                    print(f"  Market Success Rate: {mv.get('success_rate', 0):.1f}%")
                    print(f"  Avg Market Return: {mv.get('avg_return', 0):.2f}%")

                if status == 'poor' and stats.get('triggered_trades', 0) > 0:
                    print(f"  → ACTION: We're taking {stats['triggered_trades']} POOR trades - add rejection filters")
                elif status == 'excellent':
                    trigger_rate = stats.get('trigger_rate', 0)
                    if trigger_rate < 50:
                        print(f"  → ACTION: Only {trigger_rate:.1f}% of excellent trades triggered - relax sizing/timing filters")
                print()

def analyze_exit_timing(report: Dict, logs_dir: Path):
    """Find optimal exit timing patterns."""
    print("=" * 100)
    print("OPPORTUNITY #8: EXIT OPTIMIZATION")
    print("=" * 100)
    print()

    # Parse time-in-trade from analytics
    time_in_trade_data = []

    for session_dir in sorted(logs_dir.glob("2*")):
        if not session_dir.is_dir():
            continue

        analytics_file = session_dir / "analytics.jsonl"
        if not analytics_file.exists():
            continue

        with open(analytics_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    trade = json.loads(line)
                    entry_ts = trade.get('entry_ts')
                    exit_ts = trade.get('exit_ts')
                    exit_reason = trade.get('exit_reason', '')
                    pnl = trade.get('pnl', 0)

                    if entry_ts and exit_ts:
                        try:
                            entry = pd.to_datetime(entry_ts)
                            exit = pd.to_datetime(exit_ts)
                            duration_mins = (exit - entry).total_seconds() / 60

                            time_in_trade_data.append({
                                'duration_mins': duration_mins,
                                'exit_reason': exit_reason,
                                'pnl': pnl,
                                'is_winner': pnl > 0
                            })
                        except:
                            continue
                except:
                    continue

    if not time_in_trade_data:
        print("No time-in-trade data available")
        return

    # Bucket by duration
    buckets = {
        '0-30m': [],
        '30-60m': [],
        '60-120m': [],
        '120-180m': [],
        '180m+': []
    }

    for trade in time_in_trade_data:
        dur = trade['duration_mins']
        if dur <= 30:
            buckets['0-30m'].append(trade)
        elif dur <= 60:
            buckets['30-60m'].append(trade)
        elif dur <= 120:
            buckets['60-120m'].append(trade)
        elif dur <= 180:
            buckets['120-180m'].append(trade)
        else:
            buckets['180m+'].append(trade)

    print("TIME-IN-TRADE ANALYSIS:")
    print()

    for bucket_name, trades in buckets.items():
        if len(trades) >= 3:
            winners = sum(1 for t in trades if t['is_winner'])
            wr = (winners / len(trades)) * 100
            avg_pnl = sum(t['pnl'] for t in trades) / len(trades)

            print(f"{bucket_name}:")
            print(f"  Trades: {len(trades)}")
            print(f"  WR: {wr:.1f}%")
            print(f"  Avg PnL: Rs.{avg_pnl:.2f}")

            if wr > 60:
                print(f"  → ACTION: STRONG bucket - optimize for this duration")
            elif wr < 40:
                print(f"  → ACTION: WEAK bucket - consider time-based stops")
            print()

def generate_actionable_summary(report: Dict):
    """Generate prioritized action items."""
    print("=" * 100)
    print("PRIORITIZED ACTION ITEMS")
    print("=" * 100)
    print()

    actions = []

    # 1. Find high-WR setups with low volume
    for setup_name, stats in report['setup_analysis'].items():
        if stats['total_trades'] >= 3:
            wr = stats['win_rate']
            pf = stats['profit_factor']
            count = stats['total_trades']

            if (wr > 65 or pf > 2.5) and count < 20:
                actions.append({
                    'priority': 'HIGH',
                    'impact': 'More Trades + Higher WR',
                    'action': f"Increase {setup_name} detection (currently only {count} trades, {wr:.1f}% WR, PF:{pf:.2f})"
                })

    # 2. Find losing setups to block
    for setup_name, stats in report['setup_analysis'].items():
        if stats['total_trades'] >= 5:
            wr = stats['win_rate']
            pf = stats['profit_factor']

            if wr < 40 and pf < 1.0:
                actions.append({
                    'priority': 'HIGH',
                    'impact': 'Reduce Losses',
                    'action': f"Block or tighten filters for {setup_name} ({stats['total_trades']} trades, {wr:.1f}% WR, losing Rs.{abs(stats['total_pnl']):.2f})"
                })

    # 3. Regime opportunities
    regime_data = report['regime_analysis']
    best_regime = max(regime_data.items(), key=lambda x: x[1].get('avg_pnl', 0))
    worst_regime = min(regime_data.items(), key=lambda x: x[1].get('avg_pnl', 0))

    if best_regime[1]['total_trades'] >= 10:
        actions.append({
            'priority': 'MEDIUM',
            'impact': 'More Profitable Trades',
            'action': f"Prioritize {best_regime[0]} regime (Avg: Rs.{best_regime[1]['avg_pnl']:.2f}/trade)"
        })

    if worst_regime[1]['total_trades'] >= 10 and worst_regime[1]['avg_pnl'] < 0:
        actions.append({
            'priority': 'MEDIUM',
            'impact': 'Reduce Losses',
            'action': f"Reduce trades in {worst_regime[0]} regime (Avg: Rs.{worst_regime[1]['avg_pnl']:.2f}/trade)"
        })

    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    actions.sort(key=lambda x: priority_order[x['priority']])

    for i, action in enumerate(actions, 1):
        print(f"{i}. [{action['priority']}] {action['action']}")
        print(f"   Impact: {action['impact']}")
        print()

def main():
    if len(sys.argv) < 3:
        print("Usage: python deep_pattern_analysis.py <report.json> <logs_dir>")
        print("Example: python tools/deep_pattern_analysis.py analysis/reports/misc/analysis_report_20_20251119_163157.json backtest_20251119-082113_extracted/20251119-082113_full/20251119-082113")
        sys.exit(1)

    report_path = Path(sys.argv[1])
    logs_dir = Path(sys.argv[2])

    if not report_path.exists():
        print(f"Error: Report file {report_path} not found")
        sys.exit(1)

    if not logs_dir.exists():
        print(f"Error: Logs directory {logs_dir} not found")
        sys.exit(1)

    print(f"Loading report: {report_path}")
    print(f"Analyzing logs: {logs_dir}")
    print()

    report = load_report(report_path)

    print("=" * 100)
    print("DEEP PATTERN ANALYSIS - PROFESSIONAL INDIAN TRADER PERSPECTIVE")
    print("=" * 100)
    print(f"Sessions Analyzed: {report['sessions_analyzed']}")
    print(f"Total Trades: {report['total_trades']}")
    print(f"Overall WR: {report['performance_summary']['win_rate']:.1f}%")
    print(f"Total PnL: Rs.{report['performance_summary']['total_pnl']:.2f}")
    print()

    # Run all analyses
    analyze_setup_quality_gaps(report)
    analyze_sl_heavy_setups(report, logs_dir)
    analyze_regime_opportunities(report)
    analyze_timing_patterns(report)
    analyze_indicator_correlations(report)
    analyze_quality_calibration(report)
    analyze_missed_opportunities(report)
    analyze_exit_timing(report, logs_dir)
    generate_actionable_summary(report)

if __name__ == '__main__':
    main()
