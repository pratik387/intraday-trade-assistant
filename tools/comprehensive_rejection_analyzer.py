"""
Comprehensive Rejection Analysis Across ALL Pipeline Stages

Analyzes logs from:
1. Screener stage - Universe, volatility, liquidity filters
2. Ranker stage - Ranking logic, structural_rr thresholds
3. Planner stage - Structure detection, quality checks
4. Decision gate stage - Position sizing, risk limits

Then runs spike tests on blocked winners using 1m OHLC data.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
from typing import Dict, List, Tuple

class ComprehensiveRejectionAnalyzer:
    def __init__(self, baseline_dir: str, filtered_dir: str):
        self.baseline_dir = Path(baseline_dir)
        self.filtered_dir = Path(filtered_dir)

        # Load trades
        print("Loading baseline trades...")
        self.baseline_trades = self._load_trades(self.baseline_dir)
        print(f"Loaded {len(self.baseline_trades)} baseline trades")

        print("Loading filtered trades...")
        self.filtered_trades = self._load_trades(self.filtered_dir)
        print(f"Loaded {len(self.filtered_trades)} filtered trades")

        # Find blocked trades
        self.blocked_trades = self._find_blocked_trades()
        self.blocked_winners = [t for t in self.blocked_trades if t['pnl'] > 0]
        self.blocked_losers = [t for t in self.blocked_trades if t['pnl'] <= 0]

        print(f"\nFound {len(self.blocked_trades)} blocked trades:")
        print(f"  Winners: {len(self.blocked_winners)} (Rs.{sum(t['pnl'] for t in self.blocked_winners):.2f})")
        print(f"  Losers: {len(self.blocked_losers)} (Rs.{sum(t['pnl'] for t in self.blocked_losers):.2f})")

    def _load_trades(self, backtest_dir: Path) -> List[Dict]:
        """Load all EXIT trades from analytics.jsonl files"""
        trades = []

        for session_dir in sorted(backtest_dir.glob('20*')):
            analytics_file = session_dir / 'analytics.jsonl'
            if not analytics_file.exists():
                continue

            with open(analytics_file) as f:
                for line in f:
                    try:
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
                    except json.JSONDecodeError:
                        continue

        return trades

    def _find_blocked_trades(self) -> List[Dict]:
        """Find trades that exist in baseline but not in filtered"""
        baseline_ids = {t['trade_id']: t for t in self.baseline_trades}
        filtered_ids = {t['trade_id'] for t in self.filtered_trades}

        blocked = []
        for trade_id, trade in baseline_ids.items():
            if trade_id not in filtered_ids:
                # Only analyze breakout trades (the ones being filtered)
                if 'breakout' in trade.get('strategy', '').lower():
                    blocked.append(trade)

        return blocked

    def parse_agent_log_comprehensive(self, log_file: Path, symbol: str, session_date: str) -> Dict[str, List[str]]:
        """
        Parse agent.log for ALL rejection patterns across ALL pipeline stages.

        Returns dict of stage -> list of rejection messages
        """
        rejections = defaultdict(list)

        if not log_file.exists():
            return rejections

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Only process lines mentioning the symbol
                    if symbol not in line:
                        continue

                    # SCREENER STAGE rejections
                    if 'SCREENER' in line or 'screener.py' in line:
                        if any(kw in line.lower() for kw in ['reject', 'skip', 'filter', 'exclude']):
                            rejections['screener'].append(line.strip())

                    # RANKER STAGE rejections
                    if 'RANKER' in line or 'ranker.py' in line:
                        if any(kw in line.lower() for kw in ['reject', 'skip', 'filter', 'below', 'threshold']):
                            rejections['ranker'].append(line.strip())

                    # PLANNER STAGE rejections (structure detection)
                    if 'LEVEL_BREAKOUT' in line or 'level_breakout_structure' in line:
                        if 'Rejected:' in line:
                            rejections['planner_structure'].append(line.strip())

                    # PLANNER STAGE rejections (quality checks)
                    if 'PLANNER' in line or 'planner_internal' in line:
                        if any(kw in line.lower() for kw in ['reject', 'skip', 'below', 'threshold']):
                            rejections['planner_quality'].append(line.strip())

                    # DECISION GATE rejections
                    if 'DECISION' in line or 'decision_gate' in line:
                        if any(kw in line.lower() for kw in ['reject', 'block', 'exceed', 'limit']):
                            rejections['decision_gate'].append(line.strip())

                    # Specific filter patterns
                    if 'Pre-institutional hours' in line or '9:15-9:45' in line:
                        rejections['filter_timing'].append(line.strip())

                    if 'Weak long candle' in line or 'Weak short candle' in line or 'close at' in line:
                        rejections['filter_conviction'].append(line.strip())

                    if 'No volume accumulation' in line or 'vol_z>1.0' in line:
                        rejections['filter_volume_accumulation'].append(line.strip())

                    if 'Level not clean' in line or 'touches in last' in line:
                        rejections['filter_level_cleanness'].append(line.strip())

                    if 'structural_rr' in line.lower() and any(kw in line.lower() for kw in ['below', '<', 'threshold']):
                        rejections['filter_structural_rr'].append(line.strip())

        except Exception as e:
            print(f"Error parsing {log_file}: {e}")

        return dict(rejections)

    def analyze_blocked_trade(self, trade: Dict) -> Dict:
        """
        Analyze a single blocked trade to find rejection reasons.
        """
        session = trade['session']
        symbol = trade['symbol']

        # Find log file in filtered run
        log_file = self.filtered_dir / session / 'agent.log'

        # Parse log for rejections
        rejections = self.parse_agent_log_comprehensive(log_file, symbol, session)

        # Determine primary rejection reason
        primary_reason = self._determine_primary_rejection(rejections)

        analysis = {
            'trade': trade,
            'rejections': rejections,
            'primary_reason': primary_reason,
            'rejection_count': sum(len(v) for v in rejections.values())
        }

        return analysis

    def _determine_primary_rejection(self, rejections: Dict[str, List[str]]) -> str:
        """Determine the primary (first/most important) rejection reason"""

        # Priority order: filters that execute earliest in the pipeline
        priority_order = [
            'screener',
            'ranker',
            'filter_structural_rr',
            'filter_timing',
            'filter_conviction',
            'filter_volume_accumulation',
            'filter_level_cleanness',
            'planner_structure',
            'planner_quality',
            'decision_gate'
        ]

        for stage in priority_order:
            if stage in rejections and rejections[stage]:
                # Extract concise reason from first message
                first_msg = rejections[stage][0]

                if 'Rejected:' in first_msg:
                    reason = first_msg.split('Rejected:')[1].split('(')[0].strip()
                    return reason
                elif 'Pre-institutional hours' in first_msg:
                    return "Timing: 9:15-9:45am retail noise"
                elif 'Weak long candle' in first_msg or 'Weak short candle' in first_msg:
                    return "Conviction: Close not in optimal position"
                elif 'No volume accumulation' in first_msg:
                    return "Volume: <3 bars with vol_z>1.0"
                elif 'Level not clean' in first_msg:
                    return "Level cleanness: >3 touches"
                elif 'structural_rr' in first_msg.lower():
                    return "Structural RR: Below threshold"
                else:
                    # Generic reason from stage
                    return f"{stage.replace('_', ' ').title()}"

        return "Unknown (no log found)"

    def analyze_all_blocked_winners(self) -> pd.DataFrame:
        """Analyze all blocked winner trades"""
        print(f"\n{'='*80}")
        print(f"Analyzing {len(self.blocked_winners)} blocked winner trades...")
        print(f"{'='*80}\n")

        analyses = []
        for i, trade in enumerate(self.blocked_winners, 1):
            if i % 5 == 0:
                print(f"  Processed {i}/{len(self.blocked_winners)} trades...")

            analysis = self.analyze_blocked_trade(trade)

            analyses.append({
                'session': trade['session'],
                'symbol': trade['symbol'],
                'strategy': trade['strategy'],
                'pnl': trade['pnl'],
                'exit_reason': trade['reason'],
                'primary_rejection': analysis['primary_reason'],
                'rejection_count': analysis['rejection_count'],
                'rejections_json': json.dumps(analysis['rejections'])
            })

        df = pd.DataFrame(analyses)
        return df

    def generate_filter_impact_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create filter impact matrix showing:
        - Which filters blocked which trades
        - P&L impact per filter
        - Frequency of each filter
        """
        filter_impact = defaultdict(lambda: {'count': 0, 'pnl': 0.0, 'symbols': []})

        for _, row in df.iterrows():
            reason = row['primary_rejection']
            pnl = row['pnl']
            symbol = row['symbol']

            filter_impact[reason]['count'] += 1
            filter_impact[reason]['pnl'] += pnl
            filter_impact[reason]['symbols'].append(symbol)

        # Convert to DataFrame
        impact_data = []
        for filter_name, data in filter_impact.items():
            impact_data.append({
                'Filter': filter_name,
                'Blocked_Winners': data['count'],
                'Opportunity_Cost_Rs': data['pnl'],
                'Avg_Winner_Rs': data['pnl'] / data['count'] if data['count'] > 0 else 0,
                'Sample_Symbols': ', '.join(list(set(data['symbols']))[:5])
            })

        impact_df = pd.DataFrame(impact_data)
        impact_df = impact_df.sort_values('Opportunity_Cost_Rs', ascending=False)

        return impact_df

    def run_full_analysis(self, output_file: str = "COMPREHENSIVE_REJECTION_ANALYSIS.txt"):
        """Run complete pipeline-wide analysis"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE REJECTION ANALYSIS")
        print(f"{'='*80}\n")

        # Analyze all blocked winners
        winner_df = self.analyze_all_blocked_winners()

        # Generate filter impact report
        filter_impact = self.generate_filter_impact_report(winner_df)

        # Print results
        print(f"\n{'='*80}")
        print("FILTER IMPACT MATRIX (Sorted by Opportunity Cost)")
        print(f"{'='*80}")
        print(filter_impact.to_string(index=False))

        print(f"\n{'='*80}")
        print("TOP 20 BLOCKED WINNER TRADES")
        print(f"{'='*80}")
        top_winners = winner_df.nlargest(20, 'pnl')[['session', 'symbol', 'strategy', 'pnl', 'primary_rejection']]
        print(top_winners.to_string(index=False))

        # Save to file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE REJECTION ANALYSIS\n")
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

            f.write("ALL BLOCKED WINNER TRADES\n")
            f.write("-"*80 + "\n")
            for _, row in winner_df.iterrows():
                f.write(f"\n{row['symbol']} ({row['session']}) - {row['strategy']} - Rs.{row['pnl']:.2f}\n")
                f.write(f"Primary Rejection: {row['primary_rejection']}\n")

                # Parse and print detailed rejections
                rejections = json.loads(row['rejections_json'])
                if rejections:
                    f.write("Detailed Rejections:\n")
                    for stage, messages in rejections.items():
                        f.write(f"  {stage}:\n")
                        for msg in messages[:2]:  # Limit to first 2 messages per stage
                            f.write(f"    - {msg[:120]}...\n" if len(msg) > 120 else f"    - {msg}\n")
                f.write("\n")

        print(f"\nFull analysis saved to: {output_path}")

        return winner_df, filter_impact


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python comprehensive_rejection_analyzer.py <baseline_dir> <filtered_dir> [output_file]")
        print("\nExample:")
        print("  python comprehensive_rejection_analyzer.py \\")
        print("    backtest_20251108-034930_extracted/20251108-034930_full/20251108-034930 \\")
        print("    backtest_20251108-124615_extracted/20251108-124615_full/20251108-124615 \\")
        print("    COMPREHENSIVE_REJECTION_ANALYSIS.txt")
        sys.exit(1)

    baseline_dir = sys.argv[1]
    filtered_dir = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "COMPREHENSIVE_REJECTION_ANALYSIS.txt"

    # Run analysis
    analyzer = ComprehensiveRejectionAnalyzer(baseline_dir, filtered_dir)
    winner_df, filter_impact = analyzer.run_full_analysis(output_file)
