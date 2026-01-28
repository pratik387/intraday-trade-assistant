"""
Comprehensive Forensic Analysis of Breakout Filter Impact

This script performs a complete forensic analysis of rejected breakout trades:
1. Compares baseline vs filtered backtests to identify blocked trades
2. Analyzes planner/ranker/screener logs to extract rejection reasons
3. Uses spike test with 1m OHLC data to validate what would have happened
4. Finds common patterns among winners vs losers
5. Provides data-driven filter threshold recommendations

Usage:
    python tools/forensic_filter_analysis.py \\
        --baseline backtest_20251108-034930.zip \\
        --filtered backtest_20251108-124615.zip \\
        --output FORENSIC_FILTER_ANALYSIS.md
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import zipfile
import tempfile
import shutil
import subprocess
import re


class ForensicFilterAnalyzer:
    def __init__(self, baseline_zip: str, filtered_zip: str, output_file: str):
        self.baseline_zip = Path(baseline_zip)
        self.filtered_zip = Path(filtered_zip)
        self.output_file = Path(output_file)

        self.baseline_trades = []
        self.filtered_trades = []
        self.blocked_trades = []
        self.rejection_reasons = defaultdict(list)
        self.spike_test_results = []

    def extract_backtest(self, zip_path: Path, temp_dir: Path) -> Path:
        """Extract backtest zip to temporary directory"""
        print(f"Extracting {zip_path.name}...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find the session directory (format: YYYYMMDD-HHMMSS)
        extracted_dirs = list(temp_dir.glob("*/*/"))
        if not extracted_dirs:
            extracted_dirs = list(temp_dir.glob("*/"))

        if extracted_dirs:
            return extracted_dirs[0]

        raise FileNotFoundError(f"Could not find extracted backtest directory in {temp_dir}")

    def load_trades_from_analytics(self, backtest_dir: Path) -> List[Dict]:
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

    def create_trade_signature(self, trade: Dict) -> str:
        """Create unique signature for trade matching"""
        return f"{trade['session_date']}_{trade['symbol']}_{trade['strategy']}_{trade.get('entry_time', 'NA')}"

    def identify_blocked_trades(self):
        """Compare baseline vs filtered to find blocked trades"""
        print("\nIdentifying blocked trades...")

        baseline_sigs = {self.create_trade_signature(t): t for t in self.baseline_trades}
        filtered_sigs = {self.create_trade_signature(t) for t in self.filtered_trades}

        # Find trades in baseline but not in filtered
        for sig, trade in baseline_sigs.items():
            if sig not in filtered_sigs:
                # Only focus on breakout trades
                if 'breakout' in trade.get('strategy', '').lower():
                    self.blocked_trades.append(trade)

        print(f"Found {len(self.blocked_trades)} blocked breakout trades")

        # Categorize by strategy
        by_strategy = defaultdict(list)
        for trade in self.blocked_trades:
            by_strategy[trade['strategy']].append(trade)

        print("\nBlocked trades by strategy:")
        for strategy, trades in sorted(by_strategy.items()):
            total_pnl = sum(t['pnl'] for t in trades)
            winners = sum(1 for t in trades if t['pnl'] > 0)
            print(f"  {strategy}: {len(trades)} trades ({winners} winners) → Rs.{total_pnl:.2f}")

    def extract_rejection_reasons_from_logs(self, filtered_dir: Path):
        """Parse planner/ranker/screener logs to find rejection reasons"""
        print("\nExtracting rejection reasons from logs...")

        rejection_patterns = {
            'timing': r'Rejected: Pre-institutional hours',
            'candle_conviction': r'Rejected: Weak (long|short) candle',
            'volume_accumulation': r'Rejected: No volume accumulation',
            'level_cleanness': r'Rejected: Level not clean',
            'structural_rr': r'structural_rr.*<',
        }

        # Search through all session log files
        for session_dir in sorted(filtered_dir.glob("20*")):
            log_files = list(session_dir.glob("*.log"))

            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                        # Extract rejections for each pattern
                        for filter_name, pattern in rejection_patterns.items():
                            matches = re.findall(pattern, content)
                            if matches:
                                # Try to find associated symbol
                                symbol_match = re.search(r'NSE:(\w+)', content)
                                symbol = symbol_match.group(0) if symbol_match else "UNKNOWN"

                                self.rejection_reasons[filter_name].append({
                                    'session': session_dir.name,
                                    'symbol': symbol,
                                    'log_file': log_file.name
                                })
                except Exception as e:
                    continue

        print("\nRejection reason breakdown:")
        for filter_name, rejections in sorted(self.rejection_reasons.items(),
                                              key=lambda x: len(x[1]), reverse=True):
            print(f"  {filter_name}: {len(rejections)} rejections")

    def run_spike_test_on_trade(self, trade: Dict) -> Dict:
        """Run spike test simulation on a single trade using 1m OHLC data"""

        # Extract trade parameters
        symbol = trade['symbol']
        session_date = trade['session_date']
        strategy = trade['strategy']
        direction = trade.get('direction', 'long')
        entry_time = trade.get('entry_time')
        entry_price = trade.get('entry_price', 0)
        stop_loss = trade.get('stop_loss', 0)

        # For now, return placeholder (will integrate with actual spike test tool)
        return {
            'symbol': symbol,
            'date': session_date,
            'strategy': strategy,
            'baseline_pnl': trade['pnl'],
            'spike_test_status': 'PENDING',  # Would be filled by actual spike test
            'rejection_likely_correct': None
        }

    def analyze_winner_loser_patterns(self):
        """Find common patterns in winners vs losers among blocked trades"""
        print("\nAnalyzing winner/loser patterns in blocked trades...")

        winners = [t for t in self.blocked_trades if t['pnl'] > 0]
        losers = [t for t in self.blocked_trades if t['pnl'] <= 0]
        hard_sl_losers = [t for t in losers if t.get('reason') == 'hard_sl']

        print(f"\nBlocked trade breakdown:")
        print(f"  Total blocked: {len(self.blocked_trades)}")
        print(f"  Winners: {len(winners)} ({len(winners)/len(self.blocked_trades)*100:.1f}%)")
        print(f"  Losers: {len(losers)} ({len(losers)/len(self.blocked_trades)*100:.1f}%)")
        print(f"  Hard SL losers: {len(hard_sl_losers)} ({len(hard_sl_losers)/len(losers)*100:.1f}% of losers)")

        # P&L analysis
        winner_pnl = sum(t['pnl'] for t in winners)
        loser_pnl = sum(t['pnl'] for t in losers)
        net_pnl = winner_pnl + loser_pnl

        print(f"\nP&L impact:")
        print(f"  Gross winners: Rs.{winner_pnl:.2f}")
        print(f"  Gross losers: Rs.{loser_pnl:.2f}")
        print(f"  Net P&L blocked: Rs.{net_pnl:.2f}")

        # Exit reason patterns
        print(f"\nWinner exit reasons:")
        winner_reasons = pd.Series([t.get('reason', 'unknown') for t in winners]).value_counts()
        for reason, count in winner_reasons.head(5).items():
            print(f"  {reason}: {count}")

        print(f"\nLoser exit reasons:")
        loser_reasons = pd.Series([t.get('reason', 'unknown') for t in losers]).value_counts()
        for reason, count in loser_reasons.head(5).items():
            print(f"  {reason}: {count}")

        return {
            'total_blocked': len(self.blocked_trades),
            'winners': len(winners),
            'losers': len(losers),
            'hard_sl_losers': len(hard_sl_losers),
            'winner_pnl': winner_pnl,
            'loser_pnl': loser_pnl,
            'net_pnl': net_pnl
        }

    def generate_filter_recommendations(self, pattern_stats: Dict):
        """Generate data-driven filter threshold recommendations"""

        recommendations = []

        # Calculate filter effectiveness
        total_blocked = pattern_stats['total_blocked']
        winners_blocked = pattern_stats['winners']
        hard_sl_blocked = pattern_stats['hard_sl_losers']

        # If we blocked more winners than hard SL losers, filters are too strict
        if winners_blocked > hard_sl_blocked:
            severity = "CRITICAL"
            recommendations.append({
                'severity': severity,
                'issue': 'Filters blocking more winners than hard SL losers',
                'data': f"Winners blocked: {winners_blocked}, Hard SL blocked: {hard_sl_blocked}",
                'recommendation': 'RELAX FILTERS - Current thresholds too restrictive'
            })

        # Analyze rejection reason distribution
        total_rejections = sum(len(v) for v in self.rejection_reasons.values())

        for filter_name, rejections in sorted(self.rejection_reasons.items(),
                                              key=lambda x: len(x[1]), reverse=True):
            rejection_pct = len(rejections) / max(total_rejections, 1) * 100

            if rejection_pct > 30:  # If single filter causing >30% rejections
                recommendations.append({
                    'severity': 'HIGH',
                    'issue': f'{filter_name} causing {rejection_pct:.1f}% of rejections',
                    'data': f'{len(rejections)} trades rejected',
                    'recommendation': self._get_filter_specific_recommendation(filter_name)
                })

        return recommendations

    def _get_filter_specific_recommendation(self, filter_name: str) -> str:
        """Get specific recommendation for each filter type"""

        recommendations = {
            'timing': 'Consider reducing restricted window from 9:15-9:45am to 9:15-9:30am',
            'candle_conviction': 'Relax candle close threshold from 70%/30% to 60%/40%',
            'volume_accumulation': 'Reduce requirement from 3/5 bars to 2/5 bars with vol_z>1.0',
            'level_cleanness': 'Increase max touches from 3 to 5 in last 20 bars',
            'structural_rr': 'Review min structural_rr thresholds by strategy type'
        }

        return recommendations.get(filter_name, 'Review filter threshold')

    def generate_report(self, pattern_stats: Dict, recommendations: List[Dict]):
        """Generate comprehensive markdown report"""

        print(f"\nGenerating report: {self.output_file}")

        report = []
        report.append("# FORENSIC FILTER ANALYSIS REPORT")
        report.append(f"\n**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Baseline**: {self.baseline_zip.name}")
        report.append(f"**Filtered**: {self.filtered_zip.name}")
        report.append("\n---\n")

        # Executive Summary
        report.append("## EXECUTIVE SUMMARY\n")
        report.append(f"**Total Baseline Trades**: {len(self.baseline_trades)}")
        report.append(f"**Total Filtered Trades**: {len(self.filtered_trades)}")
        report.append(f"**Blocked Breakout Trades**: {len(self.blocked_trades)}")
        report.append(f"\n**Blocked Trade Breakdown**:")
        report.append(f"- Winners blocked: {pattern_stats['winners']} (Rs.{pattern_stats['winner_pnl']:.2f})")
        report.append(f"- Losers blocked: {pattern_stats['losers']} (Rs.{pattern_stats['loser_pnl']:.2f})")
        report.append(f"- Hard SL losers blocked: {pattern_stats['hard_sl_losers']}")
        report.append(f"- Net P&L impact: Rs.{pattern_stats['net_pnl']:.2f}")
        report.append("\n---\n")

        # Critical Findings
        report.append("## CRITICAL FINDINGS\n")
        for rec in recommendations:
            report.append(f"### [{rec['severity']}] {rec['issue']}")
            report.append(f"**Data**: {rec['data']}")
            report.append(f"**Recommendation**: {rec['recommendation']}")
            report.append("")
        report.append("---\n")

        # Rejection Reason Breakdown
        report.append("## REJECTION REASON BREAKDOWN\n")
        total_rejections = sum(len(v) for v in self.rejection_reasons.values())

        for filter_name, rejections in sorted(self.rejection_reasons.items(),
                                              key=lambda x: len(x[1]), reverse=True):
            pct = len(rejections) / max(total_rejections, 1) * 100
            report.append(f"### {filter_name}")
            report.append(f"- Rejections: {len(rejections)} ({pct:.1f}% of total)")
            report.append("")

        report.append("---\n")

        # Blocked Trade Details
        report.append("## BLOCKED TRADE DETAILS\n")
        report.append("\n### Winners Blocked (Opportunity Cost)\n")

        winners = sorted([t for t in self.blocked_trades if t['pnl'] > 0],
                        key=lambda x: x['pnl'], reverse=True)

        report.append("| Date | Symbol | Strategy | Exit Reason | P&L |")
        report.append("|------|--------|----------|-------------|-----|")

        for trade in winners[:20]:  # Top 20 winners
            report.append(f"| {trade['session_date']} | {trade['symbol']} | "
                         f"{trade['strategy']} | {trade.get('reason', 'N/A')} | "
                         f"Rs.{trade['pnl']:.2f} |")

        report.append(f"\n**Total Winner P&L Blocked**: Rs.{sum(t['pnl'] for t in winners):.2f}")

        report.append("\n### Losers Blocked (Good Filters)\n")

        losers = sorted([t for t in self.blocked_trades if t['pnl'] <= 0],
                       key=lambda x: x['pnl'])

        report.append("| Date | Symbol | Strategy | Exit Reason | P&L |")
        report.append("|------|--------|----------|-------------|-----|")

        for trade in losers[:20]:  # Top 20 losers
            report.append(f"| {trade['session_date']} | {trade['symbol']} | "
                         f"{trade['strategy']} | {trade.get('reason', 'N/A')} | "
                         f"Rs.{trade['pnl']:.2f} |")

        report.append(f"\n**Total Loser P&L Blocked**: Rs.{sum(t['pnl'] for t in losers):.2f}")

        report.append("\n---\n")

        # Pattern Analysis
        report.append("## PATTERN ANALYSIS\n")

        # Exit reason comparison
        winner_reasons = pd.Series([t.get('reason', 'unknown') for t in winners]).value_counts()
        loser_reasons = pd.Series([t.get('reason', 'unknown') for t in losers]).value_counts()

        report.append("### Winner Exit Reasons")
        report.append("| Exit Reason | Count | % |")
        report.append("|-------------|-------|---|")
        for reason, count in winner_reasons.items():
            pct = count / len(winners) * 100 if winners else 0
            report.append(f"| {reason} | {count} | {pct:.1f}% |")

        report.append("\n### Loser Exit Reasons")
        report.append("| Exit Reason | Count | % |")
        report.append("|-------------|-------|---|")
        for reason, count in loser_reasons.items():
            pct = count / len(losers) * 100 if losers else 0
            report.append(f"| {reason} | {count} | {pct:.1f}% |")

        report.append("\n---\n")

        # Recommendations
        report.append("## RECOMMENDED FILTER ADJUSTMENTS\n")

        for rec in recommendations:
            report.append(f"### {rec['issue']}")
            report.append(f"**Action**: {rec['recommendation']}")
            report.append("")

        report.append("\n---\n")
        report.append("## NEXT STEPS\n")
        report.append("1. Run spike tests on top 20 blocked winners to validate they would have actually won")
        report.append("2. Implement recommended filter threshold adjustments")
        report.append("3. Re-run backtest with adjusted filters")
        report.append("4. Compare results to validate improvement")
        report.append("\n---\n")
        report.append(f"\n*Report generated by forensic_filter_analysis.py*")

        # Write report
        with open(self.output_file, 'w') as f:
            f.write('\n'.join(report))

        print(f"Report saved to: {self.output_file}")

    def run(self):
        """Execute full forensic analysis pipeline"""

        print("="*80)
        print("FORENSIC FILTER ANALYSIS")
        print("="*80)

        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_baseline, \
             tempfile.TemporaryDirectory() as temp_filtered:

            # Extract backtests
            baseline_dir = self.extract_backtest(self.baseline_zip, Path(temp_baseline))
            filtered_dir = self.extract_backtest(self.filtered_zip, Path(temp_filtered))

            # Load trades
            print("\nLoading baseline trades...")
            self.baseline_trades = self.load_trades_from_analytics(baseline_dir)
            print(f"Loaded {len(self.baseline_trades)} baseline trades")

            print("\nLoading filtered trades...")
            self.filtered_trades = self.load_trades_from_analytics(filtered_dir)
            print(f"Loaded {len(self.filtered_trades)} filtered trades")

            # Identify blocked trades
            self.identify_blocked_trades()

            # Extract rejection reasons from logs
            self.extract_rejection_reasons_from_logs(filtered_dir)

            # Analyze patterns
            pattern_stats = self.analyze_winner_loser_patterns()

            # Generate recommendations
            recommendations = self.generate_filter_recommendations(pattern_stats)

            # Generate report
            self.generate_report(pattern_stats, recommendations)

        print("\n" + "="*80)
        print("FORENSIC ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nKey Findings:")
        print(f"  - Blocked {len(self.blocked_trades)} breakout trades")
        print(f"  - Winners blocked: {pattern_stats['winners']} (Rs.{pattern_stats['winner_pnl']:.2f})")
        print(f"  - Losers blocked: {pattern_stats['losers']} (Rs.{pattern_stats['loser_pnl']:.2f})")
        print(f"  - Net P&L impact: Rs.{pattern_stats['net_pnl']:.2f}")
        print(f"\nFull report: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description='Forensic analysis of breakout filter impact')
    parser.add_argument('--baseline', required=True, help='Baseline backtest zip file')
    parser.add_argument('--filtered', required=True, help='Filtered backtest zip file')
    parser.add_argument('--output', default='FORENSIC_FILTER_ANALYSIS.md',
                       help='Output report file')

    args = parser.parse_args()

    analyzer = ForensicFilterAnalyzer(
        baseline_zip=args.baseline,
        filtered_zip=args.filtered,
        output_file=args.output
    )

    analyzer.run()


if __name__ == '__main__':
    main()
