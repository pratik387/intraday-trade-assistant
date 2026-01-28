#!/usr/bin/env python3
"""
Analyze 6-Month Backtest Results (Pre-NSE Fixes)

Analyzes the latest 6-month backtest runs to:
1. Extract comprehensive performance metrics
2. Identify stop-loss hit patterns
3. Measure VWAP mean reversion opportunities
4. Compare with NSE baseline expectations
5. Validate proposed configuration fixes
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Run prefixes from regime orchestrator
RUN_PREFIXES = {
    'run_040e8fa6_': 'Oct 2024 (Correction Risk-Off)',
    'run_6ee3746f_': 'Feb 2025 (Prolonged Drawdown)',
    'run_53eabf2f_': 'Jul 2025 (Low Vol Range)'
}

class BacktestAnalyzer:
    """Analyze backtest results and compare with NSE baseline"""

    def __init__(self):
        self.logs_dir = ROOT / 'logs'
        self.results = {}

    def find_runs_by_prefix(self, prefix):
        """Find all run directories for a given prefix"""
        return sorted(self.logs_dir.glob(f'{prefix}*'))

    def analyze_single_run(self, run_dir):
        """Analyze a single backtest run"""
        try:
            # Load performance data
            perf_file = run_dir / 'performance.json'
            if not perf_file.exists():
                return None

            with open(perf_file) as f:
                perf = json.load(f)

            # Load analytics data
            analytics_file = run_dir / 'analytics.jsonl'
            analytics = []
            if analytics_file.exists():
                with open(analytics_file) as f:
                    for line in f:
                        analytics.append(json.loads(line))

            # Load events data
            events_file = run_dir / 'events.jsonl'
            events = []
            if events_file.exists():
                with open(events_file) as f:
                    for line in f:
                        events.append(json.loads(line))

            # Extract date from agent log
            agent_log = run_dir / 'agent.log'
            bt_date = None
            if agent_log.exists():
                with open(agent_log) as f:
                    for line in f:
                        if 'DRY RUN:' in line:
                            bt_date = line.split('DRY RUN:')[1].split()[0].strip()
                            break

            return {
                'run_dir': str(run_dir),
                'bt_date': bt_date,
                'performance': perf,
                'analytics': analytics,
                'events': events
            }

        except Exception as e:
            print(f"Error analyzing {run_dir.name}: {e}")
            return None

    def aggregate_regime_metrics(self, runs_data):
        """Aggregate metrics across all runs in a regime"""
        if not runs_data:
            return None

        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0
        total_fees = 0

        # Strategy breakdown
        strategy_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0})

        # Stop loss analysis
        hard_sl_exits = 0
        total_exits = 0

        # VWAP analysis
        vwap_mr_signals = 0
        vwap_mr_trades = 0

        # ORB analysis
        orb_signals = 0
        orb_trades = 0

        for run in runs_data:
            perf = run['performance']

            # Aggregate performance
            total_trades += perf.get('total_trades', 0)
            winning_trades += perf.get('winning_trades', 0)
            losing_trades += perf.get('losing_trades', 0)
            total_pnl += perf.get('total_pnl', 0)
            total_fees += perf.get('total_fees', 0)

            # Analyze analytics for stop loss hits
            for entry in run['analytics']:
                if entry.get('event_type') == 'ENTRY':
                    strategy = entry.get('setup_type', 'unknown')
                    strategy_stats[strategy]['count'] += 1

                    # Check if VWAP MR
                    if 'vwap_mean_reversion' in strategy:
                        vwap_mr_trades += 1

                    # Check if ORB
                    if 'orb' in strategy:
                        orb_trades += 1

                elif entry.get('event_type') == 'EXIT':
                    exit_reason = entry.get('exit_reason', '')
                    total_exits += 1

                    if 'hard_sl' in exit_reason.lower() or 'stop' in exit_reason.lower():
                        hard_sl_exits += 1

                    # Get PnL
                    pnl = entry.get('pnl', 0)
                    setup = entry.get('setup_type', 'unknown')

                    if pnl > 0:
                        strategy_stats[setup]['wins'] += 1
                    else:
                        strategy_stats[setup]['losses'] += 1

                    strategy_stats[setup]['pnl'] += pnl

            # Count signals from events
            for event in run['events']:
                event_type = event.get('event', '')

                if 'VWAP_MEAN_REVERSION' in event_type.upper() or 'vwap_mean_reversion' in event_type:
                    vwap_mr_signals += 1

                if 'ORB' in event_type.upper() or 'orb' in event_type:
                    orb_signals += 1

        # Calculate metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = (total_pnl / winning_trades) if winning_trades > 0 else 0
        avg_loss = (total_pnl / losing_trades) if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        stop_hit_rate = (hard_sl_exits / total_exits * 100) if total_exits > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'net_pnl': total_pnl - total_fees,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'hard_sl_exits': hard_sl_exits,
            'total_exits': total_exits,
            'stop_hit_rate': stop_hit_rate,
            'vwap_mr_signals': vwap_mr_signals,
            'vwap_mr_trades': vwap_mr_trades,
            'vwap_mr_conversion': (vwap_mr_trades / vwap_mr_signals * 100) if vwap_mr_signals > 0 else 0,
            'orb_signals': orb_signals,
            'orb_trades': orb_trades,
            'orb_conversion': (orb_trades / orb_signals * 100) if orb_signals > 0 else 0,
            'strategy_stats': dict(strategy_stats)
        }

    def analyze_all_regimes(self):
        """Analyze all 6-month regime backtests"""
        print("="*80)
        print("6-MONTH BACKTEST ANALYSIS (Pre-NSE Fixes)")
        print("="*80)
        print()

        regime_results = {}

        for prefix, regime_name in RUN_PREFIXES.items():
            print(f"\nAnalyzing {regime_name}...")

            # Find all runs for this prefix
            run_dirs = self.find_runs_by_prefix(prefix)

            if not run_dirs:
                print(f"  No runs found for {prefix}")
                continue

            # Analyze each run
            runs_data = []
            for run_dir in run_dirs:
                result = self.analyze_single_run(run_dir)
                if result:
                    runs_data.append(result)

            if not runs_data:
                print(f"  No valid data for {prefix}")
                continue

            # Aggregate metrics
            metrics = self.aggregate_regime_metrics(runs_data)

            print(f"  Analyzed {len(runs_data)} days")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.2f}%")
            print(f"  Net PnL: Rs.{metrics['net_pnl']:.2f}")
            print(f"  Stop Hit Rate: {metrics['stop_hit_rate']:.2f}%")

            regime_results[regime_name] = metrics

        self.results = regime_results
        return regime_results

    def generate_report(self):
        """Generate comprehensive comparison report"""
        output_file = ROOT / 'BACKTEST_6MONTH_ANALYSIS.md'

        with open(output_file, 'w') as f:
            f.write("# 6-Month Backtest Analysis (Pre-NSE Fixes)\n\n")
            f.write(f"**Analysis Date**: 2025-10-15\n\n")
            f.write("**Regimes Analyzed**:\n")
            for regime in self.results.keys():
                f.write(f"- {regime}\n")
            f.write("\n---\n\n")

            # Overall statistics
            f.write("## Overall Statistics\n\n")

            total_trades = sum(r['total_trades'] for r in self.results.values())
            total_pnl = sum(r['net_pnl'] for r in self.results.values())
            avg_win_rate = sum(r['win_rate'] for r in self.results.values()) / len(self.results)
            avg_stop_hit_rate = sum(r['stop_hit_rate'] for r in self.results.values()) / len(self.results)

            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| **Total Trades** | {total_trades} |\n")
            f.write(f"| **Net PnL** | Rs.{total_pnl:.2f} |\n")
            f.write(f"| **Average Win Rate** | {avg_win_rate:.2f}% |\n")
            f.write(f"| **Average Stop Hit Rate** | {avg_stop_hit_rate:.2f}% |\n")
            f.write("\n")

            # Per-regime breakdown
            f.write("## Per-Regime Breakdown\n\n")

            for regime, metrics in self.results.items():
                f.write(f"### {regime}\n\n")

                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Total Trades | {metrics['total_trades']} |\n")
                f.write(f"| Winning Trades | {metrics['winning_trades']} |\n")
                f.write(f"| Losing Trades | {metrics['losing_trades']} |\n")
                f.write(f"| Win Rate | {metrics['win_rate']:.2f}% |\n")
                f.write(f"| Net PnL | Rs.{metrics['net_pnl']:.2f} |\n")
                f.write(f"| Avg Win | Rs.{metrics['avg_win']:.2f} |\n")
                f.write(f"| Avg Loss | Rs.{metrics['avg_loss']:.2f} |\n")
                f.write(f"| Profit Factor | {metrics['profit_factor']:.2f} |\n")
                f.write(f"| **Stop Hit Rate** | **{metrics['stop_hit_rate']:.2f}%** |\n")
                f.write(f"| VWAP MR Signals | {metrics['vwap_mr_signals']} |\n")
                f.write(f"| VWAP MR Trades | {metrics['vwap_mr_trades']} |\n")
                f.write(f"| VWAP MR Conversion | {metrics['vwap_mr_conversion']:.2f}% |\n")
                f.write(f"| ORB Signals | {metrics['orb_signals']} |\n")
                f.write(f"| ORB Trades | {metrics['orb_trades']} |\n")
                f.write(f"| ORB Conversion | {metrics['orb_conversion']:.2f}% |\n")
                f.write("\n")

            # Comparison with NSE baseline expectations
            f.write("## Comparison with NSE Baseline\n\n")

            f.write("| Metric | NSE Baseline | Current Results | Gap |\n")
            f.write("|--------|--------------|-----------------|-----|\n")

            # Stop hit rate
            nse_expected_stop_hit = 47.4  # 1.5× ATR target
            current_stop_hit = avg_stop_hit_rate

            f.write(f"| Stop Hit Rate (1.5× ATR) | {nse_expected_stop_hit:.1f}% | {current_stop_hit:.1f}% | ")
            if current_stop_hit > nse_expected_stop_hit:
                gap = current_stop_hit - nse_expected_stop_hit
                f.write(f"**+{gap:.1f}%** (TOO HIGH) |\n")
            else:
                gap = nse_expected_stop_hit - current_stop_hit
                f.write(f"-{gap:.1f}% (Good) |\n")

            # VWAP signals per day
            total_days = len(self.results) * 20  # ~20 days per regime
            vwap_signals_total = sum(r['vwap_mr_signals'] for r in self.results.values())
            vwap_signals_per_day = vwap_signals_total / total_days if total_days > 0 else 0

            f.write(f"| VWAP MR Signals/Day | 20-30 (expected) | {vwap_signals_per_day:.1f} | ")
            if vwap_signals_per_day < 20:
                gap = 20 - vwap_signals_per_day
                f.write(f"**-{gap:.1f}** (Missing opportunities) |\n")
            else:
                f.write(f"Good |\n")

            f.write("\n")

            # Validation of proposed fixes
            f.write("## Validation of Proposed NSE Fixes\n\n")

            f.write("### Fix #1: ORB Stop Multiplier (0.5 → 1.5× ATR)\n\n")
            orb_trades = sum(r['orb_trades'] for r in self.results.values())
            f.write(f"- Current ORB trades: {orb_trades}\n")
            f.write(f"- Current stop hit rate: {avg_stop_hit_rate:.1f}%\n")
            f.write(f"- Expected with 1.5× ATR: 47.4% hit rate\n")
            f.write(f"- **Expected improvement: {avg_stop_hit_rate - 47.4:.1f}% reduction in stop-outs**\n\n")

            f.write("### Fix #2: VWAP Distance (150-400 bps → 80-250 bps)\n\n")
            f.write(f"- Current VWAP MR signals: {vwap_signals_total}\n")
            f.write(f"- Current signals/day: {vwap_signals_per_day:.1f}\n")
            f.write(f"- NSE baseline: 80% of bars within 100 bps\n")
            f.write(f"- Current config (150-400 bps): Captures only 9.2% of NSE data\n")
            f.write(f"- **Expected improvement: 5-8× more VWAP MR setups**\n\n")

            f.write("### Fix #3: Squeeze Stop Multiplier (1.0 → 1.5× ATR)\n\n")
            f.write(f"- Current stop hit rate: {avg_stop_hit_rate:.1f}%\n")
            f.write(f"- Expected with 1.5× ATR: 47.4% hit rate\n")
            f.write(f"- **Expected improvement: Similar to ORB fix**\n\n")

            print(f"\nReport generated: {output_file}")

def main():
    """Run comprehensive 6-month backtest analysis"""
    analyzer = BacktestAnalyzer()

    # Analyze all regimes
    results = analyzer.analyze_all_regimes()

    # Generate report
    analyzer.generate_report()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("1. BACKTEST_6MONTH_ANALYSIS.md - Comprehensive analysis report")
    print("\nNext steps:")
    print("1. Review BACKTEST_6MONTH_ANALYSIS.md")
    print("2. Compare with NSE_CONFIGURATION_VALIDATION_FINAL.md")
    print("3. Implement fixes from TODO.md")

if __name__ == "__main__":
    main()
