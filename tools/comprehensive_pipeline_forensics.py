"""
Comprehensive Pipeline-Wide Filter Analysis

Traces blocked winner trades through the ENTIRE pipeline:
- Screener stage (universe, volatility, liquidity filters)
- Ranker stage (ranking logic, structural_rr thresholds)
- Planner stage (ALL quality checks)
- Decision gate stage (position sizing, risk limits)

Analyzes ALL filters, not just newly added institutional filters.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Set

class PipelineForensics:
    def __init__(self, baseline_dir: str, filtered_dir: str):
        self.baseline_dir = Path(baseline_dir)
        self.filtered_dir = Path(filtered_dir)

        # Load trades from both runs
        self.baseline_trades = self._load_trades(self.baseline_dir)
        self.filtered_trades = self._load_trades(self.filtered_dir)

        # Identify blocked trades
        self.blocked_trades = self._find_blocked_trades()
        self.blocked_winners = [t for t in self.blocked_trades if t['pnl'] > 0]
        self.blocked_losers = [t for t in self.blocked_trades if t['pnl'] <= 0]

        print(f"Loaded {len(self.baseline_trades)} baseline trades")
        print(f"Loaded {len(self.filtered_trades)} filtered trades")
        print(f"Found {len(self.blocked_trades)} blocked trades")
        print(f"  Winners: {len(self.blocked_winners)}")
        print(f"  Losers: {len(self.blocked_losers)}")

    def _load_trades(self, backtest_dir: Path) -> List[Dict]:
        """Load all exit trades from analytics.jsonl files"""
        trades = []

        for session_dir in sorted(backtest_dir.glob('20*')):
            analytics_file = session_dir / 'analytics.jsonl'
            if not analytics_file.exists():
                continue

            with open(analytics_file) as f:
                for line in f:
                    event = json.loads(line)
                    if event.get('stage') == 'EXIT':
                        trade = {
                            'session': session_dir.name,
                            'symbol': event.get('symbol'),
                            'strategy': event.get('strategy'),
                            'direction': event.get('direction'),
                            'entry_time': event.get('entry_time'),
                            'exit_time': event.get('exit_time'),
                            'reason': event.get('reason'),
                            'pnl': event.get('pnl'),
                            'entry_price': event.get('entry_price'),
                            'exit_price': event.get('exit_price'),
                        }
                        # Create unique ID
                        trade['trade_id'] = f"{trade['session']}_{trade['symbol']}_{trade['strategy']}_{trade['entry_time']}"
                        trades.append(trade)

        return trades

    def _find_blocked_trades(self) -> List[Dict]:
        """Find trades that exist in baseline but not in filtered"""
        baseline_ids = {t['trade_id']: t for t in self.baseline_trades}
        filtered_ids = {t['trade_id'] for t in self.filtered_trades}

        blocked = []
        for trade_id, trade in baseline_ids.items():
            if trade_id not in filtered_ids:
                blocked.append(trade)

        return blocked

    def parse_agent_log(self, log_file: Path, symbol: str) -> Dict[str, List[str]]:
        """
        Parse agent.log for all filter rejections for a specific symbol.

        Returns dict of stage -> list of rejection messages
        """
        rejections = defaultdict(list)

        if not log_file.exists():
            return rejections

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Skip if symbol not in line
                if symbol not in line:
                    continue

                # Check for structure rejections (planner stage)
                if 'LEVEL_BREAKOUT:' in line and symbol in line:
                    if 'Rejected:' in line:
                        # Extract rejection reason
                        match = re.search(r'Rejected: (.+?)(?:\n|$)', line)
                        if match:
                            rejections['planner_structure'].append(match.group(1).strip())

                # Check for planner rejections (structural_rr, quality checks)
                if 'PLANNER:' in line or 'planner_internal' in line:
                    if 'reject' in line.lower() or 'skip' in line.lower():
                        if symbol in line:
                            rejections['planner_quality'].append(line.strip())

                # Check for ranker rejections
                if 'RANKER:' in line or 'ranker.py' in line:
                    if 'reject' in line.lower() or 'filter' in line.lower():
                        if symbol in line:
                            rejections['ranker'].append(line.strip())

                # Check for screener rejections
                if 'SCREENER:' in line or 'screener.py' in line:
                    if 'reject' in line.lower() or 'skip' in line.lower():
                        if symbol in line:
                            rejections['screener'].append(line.strip())

                # Check for decision gate rejections
                if 'DECISION_GATE:' in line or 'decision_gate' in line:
                    if 'reject' in line.lower() or 'block' in line.lower():
                        if symbol in line:
                            rejections['decision_gate'].append(line.strip())

                # Check for specific filter patterns
                if symbol in line:
                    # Timing filter
                    if 'Pre-institutional hours' in line or '9:15-9:45' in line:
                        rejections['filter_timing'].append(line.strip())

                    # Conviction filter
                    if 'Weak long candle' in line or 'Weak short candle' in line:
                        rejections['filter_conviction'].append(line.strip())

                    # Volume accumulation filter
                    if 'No volume accumulation' in line or 'vol_z>1.0' in line:
                        rejections['filter_volume_accumulation'].append(line.strip())

                    # Level cleanness filter
                    if 'Level not clean' in line or 'touches in last' in line:
                        rejections['filter_level_cleanness'].append(line.strip())

                    # structural_rr filter
                    if 'structural_rr' in line and ('below' in line.lower() or '<' in line):
                        rejections['filter_structural_rr'].append(line.strip())

        return dict(rejections)

    def analyze_trade(self, trade: Dict) -> Dict:
        """
        Analyze a single blocked trade to find why it was rejected.

        Returns dict with rejection reasons at each stage.
        """
        session = trade['session']
        symbol = trade['symbol']

        # Find log file for this session in filtered run
        log_file = self.filtered_dir / session / 'agent.log'

        # Parse log for rejections
        rejections = self.parse_agent_log(log_file, symbol)

        analysis = {
            'trade': trade,
            'rejections': rejections,
            'rejection_summary': self._summarize_rejections(rejections)
        }

        return analysis

    def _summarize_rejections(self, rejections: Dict[str, List[str]]) -> str:
        """Create a concise summary of all rejections"""
        if not rejections:
            return "No explicit rejection found in logs"

        summary_parts = []

        # Priority order: structure filters first, then quality filters
        filter_order = [
            'filter_timing',
            'filter_conviction',
            'filter_volume_accumulation',
            'filter_level_cleanness',
            'filter_structural_rr',
            'planner_structure',
            'planner_quality',
            'ranker',
            'screener',
            'decision_gate'
        ]

        for stage in filter_order:
            if stage in rejections and rejections[stage]:
                # Take first rejection for this stage
                first_rejection = rejections[stage][0]

                # Extract key info
                if 'Rejected:' in first_rejection:
                    reason = first_rejection.split('Rejected:')[1].split('(')[0].strip()
                elif 'filter_' in stage:
                    reason = f"{stage.replace('filter_', '').replace('_', ' ').title()}"
                else:
                    reason = stage.title()

                summary_parts.append(reason)

        return " | ".join(summary_parts) if summary_parts else "Unknown"

    def analyze_all_blocked_winners(self) -> pd.DataFrame:
        """Analyze all blocked winner trades"""
        print(f"\nAnalyzing {len(self.blocked_winners)} blocked winner trades...")

        analyses = []
        for i, trade in enumerate(self.blocked_winners, 1):
            if i % 5 == 0:
                print(f"  Processed {i}/{len(self.blocked_winners)} trades...")

            analysis = self.analyze_trade(trade)

            analyses.append({
                'session': trade['session'],
                'symbol': trade['symbol'],
                'strategy': trade['strategy'],
                'pnl': trade['pnl'],
                'exit_reason': trade['reason'],
                'rejection_summary': analysis['rejection_summary'],
                'rejection_count': sum(len(v) for v in analysis['rejections'].values()),
                'rejections_detail': json.dumps(analysis['rejections'])
            })

        df = pd.DataFrame(analyses)
        return df

    def generate_filter_impact_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a matrix showing:
        - Which filters blocked which trades
        - P&L impact per filter
        - Frequency of each filter
        """
        # Extract filter types from rejection summaries
        filter_counts = defaultdict(lambda: {'count': 0, 'pnl': 0.0, 'trades': []})

        for _, row in df.iterrows():
            summary = row['rejection_summary']
            pnl = row['pnl']
            symbol = row['symbol']

            # Split by pipe and get individual filters
            filters = [f.strip() for f in summary.split('|')]

            for filter_name in filters:
                if filter_name and filter_name != "Unknown":
                    filter_counts[filter_name]['count'] += 1
                    filter_counts[filter_name]['pnl'] += pnl
                    filter_counts[filter_name]['trades'].append(symbol)

        # Convert to DataFrame
        impact_data = []
        for filter_name, data in filter_counts.items():
            impact_data.append({
                'Filter': filter_name,
                'Blocked_Winners': data['count'],
                'Opportunity_Cost_Rs': data['pnl'],
                'Avg_Winner_Rs': data['pnl'] / data['count'] if data['count'] > 0 else 0,
                'Sample_Symbols': ', '.join(data['trades'][:5])
            })

        impact_df = pd.DataFrame(impact_data)
        impact_df = impact_df.sort_values('Opportunity_Cost_Rs', ascending=False)

        return impact_df

    def run_full_analysis(self, output_file: str = None):
        """Run complete pipeline-wide analysis"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PIPELINE-WIDE FILTER ANALYSIS")
        print("="*80)

        # Analyze all blocked winners
        winner_df = self.analyze_all_blocked_winners()

        # Generate filter impact matrix
        filter_impact = self.generate_filter_impact_matrix(winner_df)

        # Print results
        print("\n" + "="*80)
        print("FILTER IMPACT MATRIX (Sorted by Opportunity Cost)")
        print("="*80)
        print(filter_impact.to_string(index=False))

        print("\n" + "="*80)
        print("DETAILED BLOCKED WINNER TRADES")
        print("="*80)
        print(winner_df[['session', 'symbol', 'strategy', 'pnl', 'rejection_summary']].to_string(index=False))

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)

            with open(output_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("COMPREHENSIVE PIPELINE-WIDE FILTER ANALYSIS\n")
                f.write("="*80 + "\n\n")

                f.write("SUMMARY\n")
                f.write("-"*80 + "\n")
                f.write(f"Total blocked trades: {len(self.blocked_trades)}\n")
                f.write(f"Blocked winners: {len(self.blocked_winners)} (Rs.{sum(t['pnl'] for t in self.blocked_winners):.2f})\n")
                f.write(f"Blocked losers: {len(self.blocked_losers)} (Rs.{sum(t['pnl'] for t in self.blocked_losers):.2f})\n")
                f.write(f"Net impact: Rs.{sum(t['pnl'] for t in self.blocked_trades):.2f}\n\n")

                f.write("FILTER IMPACT MATRIX\n")
                f.write("-"*80 + "\n")
                f.write(filter_impact.to_string(index=False) + "\n\n")

                f.write("DETAILED BLOCKED WINNER TRADES\n")
                f.write("-"*80 + "\n")
                f.write(winner_df.to_string(index=False) + "\n\n")

                f.write("COMPLETE REJECTION DETAILS\n")
                f.write("-"*80 + "\n")
                for _, row in winner_df.iterrows():
                    f.write(f"\n{row['symbol']} ({row['session']}) - {row['strategy']} - Rs.{row['pnl']:.2f}\n")
                    f.write(f"Rejection Summary: {row['rejection_summary']}\n")

                    # Parse and print detailed rejections
                    rejections = json.loads(row['rejections_detail'])
                    if rejections:
                        f.write("Detailed Rejections:\n")
                        for stage, messages in rejections.items():
                            f.write(f"  {stage}:\n")
                            for msg in messages[:3]:  # Limit to first 3 messages
                                f.write(f"    - {msg[:150]}...\n" if len(msg) > 150 else f"    - {msg}\n")
                    f.write("\n")

            print(f"\nFull analysis saved to: {output_path}")

        return winner_df, filter_impact


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python comprehensive_pipeline_forensics.py <baseline_dir> <filtered_dir> [output_file]")
        print("\nExample:")
        print("  python comprehensive_pipeline_forensics.py \\")
        print("    backtest_20251108-034930_extracted/20251108-034930_full/20251108-034930 \\")
        print("    backtest_20251108-124615_extracted/20251108-124615_full/20251108-124615 \\")
        print("    COMPREHENSIVE_FILTER_ANALYSIS.txt")
        sys.exit(1)

    baseline_dir = sys.argv[1]
    filtered_dir = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "COMPREHENSIVE_FILTER_ANALYSIS.txt"

    # Run analysis
    forensics = PipelineForensics(baseline_dir, filtered_dir)
    winner_df, filter_impact = forensics.run_full_analysis(output_file)
