"""
Analyze Rejected Breakout Structures from Filtered Run

This script:
1. Parses screening.jsonl to find ALL rejected breakout structures
2. Extracts their parameters (symbol, timestamp, setup_type, rejection reason)
3. Runs spike tests using 1m OHLC data to simulate performance
4. Finds patterns between winners and losers
5. Generates data-driven filter threshold recommendations

NO baseline comparison - purely independent analysis of filtered run.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

class RejectedBreakoutAnalyzer:
    def __init__(self, filtered_run_dir: str):
        self.filtered_run_dir = Path(filtered_run_dir)
        self.ohlcv_archive = Path("ohlcv_archive")

        # Load all rejected breakouts from screening.jsonl
        self.rejected_breakouts = self._load_rejected_breakouts()

        print(f"Loaded {len(self.rejected_breakouts)} rejected breakout structures")

    def _load_rejected_breakouts(self) -> List[Dict]:
        """Load ALL rejected breakout structures from screening.jsonl files"""
        rejected = []

        for session_dir in sorted(self.filtered_run_dir.glob('20*')):
            screening_file = session_dir / 'screening.jsonl'
            if not screening_file.exists():
                continue

            with open(screening_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line)

                        # Looking for rejected breakout structures
                        if (event.get('action') == 'reject' and
                            event.get('setup_type') and
                            'breakout' in event.get('setup_type', '').lower()):

                            rejected.append({
                                'session': session_dir.name,
                                'timestamp': event.get('timestamp'),
                                'symbol': event.get('symbol'),
                                'setup_type': event.get('setup_type'),
                                'reason': event.get('reason'),
                                'all_reasons': event.get('all_reasons', []),
                                'current_price': event.get('current_price'),
                                'regime': event.get('regime'),
                            })
                    except json.JSONDecodeError:
                        continue

        return rejected

    def categorize_by_rejection_reason(self) -> pd.DataFrame:
        """Categorize rejected breakouts by their rejection reason"""

        reason_counts = defaultdict(int)
        for trade in self.rejected_breakouts:
            reason = trade['reason']
            reason_counts[reason] += 1

        df = pd.DataFrame([
            {'Rejection_Reason': reason, 'Count': count}
            for reason, count in reason_counts.items()
        ])

        df = df.sort_values('Count', ascending=False)
        return df

    def load_1m_ohlcv_data(self, symbol: str, session_date: str) -> pd.DataFrame:
        """Load 1-minute OHLCV data for a specific symbol and date"""

        # Remove NSE: prefix
        symbol_clean = symbol.replace('NSE:', '')

        # Parse date
        date_obj = pd.to_datetime(session_date)
        year = date_obj.year
        month = f"{date_obj.month:02d}"

        # Path to 1m data: ohlcv_archive/{symbol}/1m/{year}/{month}/{symbol}_{date}.parquet
        data_file = (self.ohlcv_archive / symbol_clean / '1m' /
                     str(year) / month / f"{symbol_clean}_{session_date}.parquet")

        if not data_file.exists():
            return None

        try:
            df = pd.read_parquet(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            return df
        except Exception as e:
            print(f"Error loading {data_file}: {e}")
            return None

    def simulate_trade(self, rejected_trade: Dict) -> Dict:
        """
        Simulate what would have happened if we took this trade.

        Simple logic:
        - Entry: Current price at rejection time
        - Stop loss: -2% (typical hard SL)
        - Exit: EOD squareoff at 15:15

        Returns simulation results with P&L.
        """
        symbol = rejected_trade['symbol']
        session = rejected_trade['session']
        entry_time_str = rejected_trade['timestamp']
        entry_price = rejected_trade['current_price']

        if entry_price is None:
            return {'simulated': False, 'reason': 'No entry price'}

        # Load 1m data
        df_1m = self.load_1m_ohlcv_data(symbol, session)

        if df_1m is None or len(df_1m) == 0:
            return {'simulated': False, 'reason': 'No 1m data available'}

        # Parse entry time
        try:
            entry_time = pd.to_datetime(entry_time_str)
        except:
            return {'simulated': False, 'reason': 'Invalid timestamp'}

        # Find bars after entry
        future_bars = df_1m[df_1m.index > entry_time]

        if len(future_bars) == 0:
            return {'simulated': False, 'reason': 'No future bars'}

        # Determine direction from setup_type
        is_long = 'long' in rejected_trade['setup_type'].lower()

        # Calculate stop loss (2% from entry)
        if is_long:
            sl_price = entry_price * 0.98  # 2% below
        else:
            sl_price = entry_price * 1.02  # 2% above

        # Simulate trade bar by bar
        hit_sl = False
        sl_bar = None

        for idx, bar in future_bars.iterrows():
            # Check if SL hit
            if is_long:
                if bar['low'] <= sl_price:
                    hit_sl = True
                    sl_bar = idx
                    break
            else:
                if bar['high'] >= sl_price:
                    hit_sl = True
                    sl_bar = idx
                    break

            # Check if EOD squareoff time (15:15)
            if idx.hour == 15 and idx.minute >= 15:
                break

        # Calculate P&L
        if hit_sl:
            exit_price = sl_price
            exit_time = sl_bar
            exit_reason = 'hard_sl'
        else:
            # Exit at last available bar (EOD)
            exit_price = future_bars.iloc[-1]['close']
            exit_time = future_bars.index[-1]
            exit_reason = 'eod_squareoff'

        # Calculate P&L (assuming 1 share for simplicity)
        if is_long:
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        return {
            'simulated': True,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_time': str(exit_time),
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': (pnl / entry_price) * 100,
            'is_winner': pnl > 0
        }

    def run_spike_tests_on_sample(self, sample_size: int = 50) -> pd.DataFrame:
        """
        Run spike tests on a sample of rejected breakouts.

        Returns DataFrame with simulation results.
        """

        print(f"\nRunning spike tests on {sample_size} rejected breakouts...")

        # Sample rejected breakouts (prioritize different rejection reasons)
        sample = self.rejected_breakouts[:sample_size]

        results = []
        for i, trade in enumerate(sample, 1):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(sample)} spike tests...")

            sim_result = self.simulate_trade(trade)

            result = {
                'session': trade['session'],
                'symbol': trade['symbol'],
                'setup_type': trade['setup_type'],
                'rejection_reason': trade['reason'],
                'simulated': sim_result.get('simulated', False),
                'pnl': sim_result.get('pnl', 0),
                'pnl_pct': sim_result.get('pnl_pct', 0),
                'exit_reason': sim_result.get('exit_reason', 'N/A'),
                'is_winner': sim_result.get('is_winner', False),
            }

            results.append(result)

        df = pd.DataFrame(results)

        # Filter only successfully simulated trades
        df_sim = df[df['simulated'] == True].copy()

        print(f"\nSuccessfully simulated: {len(df_sim)}/{len(df)} trades")

        return df_sim

    def analyze_patterns(self, spike_df: pd.DataFrame) -> Dict:
        """
        Analyze patterns between winners and losers.

        Returns insights about which rejection reasons blocked good trades.
        """

        if len(spike_df) == 0:
            return {'error': 'No simulated trades'}

        winners = spike_df[spike_df['is_winner'] == True]
        losers = spike_df[spike_df['is_winner'] == False]

        print(f"\n{'='*80}")
        print("SPIKE TEST RESULTS")
        print(f"{'='*80}")
        print(f"Total simulated: {len(spike_df)} trades")
        print(f"Winners: {len(winners)} ({len(winners)/len(spike_df)*100:.1f}%)")
        print(f"Losers: {len(losers)} ({len(losers)/len(spike_df)*100:.1f}%)")
        print(f"Avg winner P&L: Rs.{winners['pnl'].mean():.2f}") if len(winners) > 0 else print("No winners")
        print(f"Avg loser P&L: Rs.{losers['pnl'].mean():.2f}") if len(losers) > 0 else print("No losers")
        print(f"Net P&L: Rs.{spike_df['pnl'].sum():.2f}")

        # Analyze by rejection reason
        print(f"\n{'='*80}")
        print("REJECTION REASON PERFORMANCE")
        print(f"{'='*80}")

        reason_performance = {}
        for reason in spike_df['rejection_reason'].unique():
            reason_trades = spike_df[spike_df['rejection_reason'] == reason]
            reason_winners = reason_trades[reason_trades['is_winner'] == True]

            reason_performance[reason] = {
                'total': len(reason_trades),
                'winners': len(reason_winners),
                'win_rate': len(reason_winners) / len(reason_trades) * 100 if len(reason_trades) > 0 else 0,
                'net_pnl': reason_trades['pnl'].sum(),
                'avg_pnl': reason_trades['pnl'].mean(),
            }

            print(f"\n{reason}:")
            print(f"  Total: {reason_performance[reason]['total']} trades")
            print(f"  Win rate: {reason_performance[reason]['win_rate']:.1f}%")
            print(f"  Net P&L: Rs.{reason_performance[reason]['net_pnl']:.2f}")
            print(f"  Avg P&L: Rs.{reason_performance[reason]['avg_pnl']:.2f}")

        return {
            'total_trades': len(spike_df),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(spike_df) * 100,
            'net_pnl': spike_df['pnl'].sum(),
            'reason_performance': reason_performance,
        }

    def generate_recommendations(self, patterns: Dict, spike_df: pd.DataFrame):
        """Generate data-driven filter recommendations"""

        print(f"\n{'='*80}")
        print("FILTER RECOMMENDATIONS")
        print(f"{'='*80}")

        reason_perf = patterns.get('reason_performance', {})

        # Find rejection reasons that blocked profitable setups
        for reason, perf in sorted(reason_perf.items(), key=lambda x: x[1]['net_pnl'], reverse=True):
            if perf['win_rate'] >= 50 and perf['net_pnl'] > 0:
                print(f"\n[ACTION REQUIRED] {reason}")
                print(f"  This filter blocked {perf['total']} trades with {perf['win_rate']:.1f}% win rate")
                print(f"  Net P&L if allowed: Rs.{perf['net_pnl']:.2f}")
                print(f"  RECOMMENDATION: RELAX or REMOVE this filter")
            elif perf['win_rate'] < 40 and perf['net_pnl'] < 0:
                print(f"\n[WORKING CORRECTLY] {reason}")
                print(f"  This filter blocked {perf['total']} trades with {perf['win_rate']:.1f}% win rate")
                print(f"  Net P&L avoided: Rs.{perf['net_pnl']:.2f}")
                print(f"  RECOMMENDATION: KEEP this filter")

    def run_full_analysis(self, sample_size: int = 100, output_file: str = "REJECTED_BREAKOUT_ANALYSIS.txt"):
        """Run complete independent analysis"""

        print(f"\n{'='*80}")
        print("REJECTED BREAKOUT ANALYSIS - INDEPENDENT")
        print(f"{'='*80}\n")

        # 1. Categorize by rejection reason
        reason_df = self.categorize_by_rejection_reason()

        print("\nREJECTION REASONS:")
        print(reason_df.to_string(index=False))

        # 2. Run spike tests
        spike_df = self.run_spike_tests_on_sample(sample_size)

        if len(spike_df) == 0:
            print("\nNo spike tests could be run (missing 1m data)")
            return

        # 3. Analyze patterns
        patterns = self.analyze_patterns(spike_df)

        # 4. Generate recommendations
        self.generate_recommendations(patterns, spike_df)

        # 5. Save detailed results
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REJECTED BREAKOUT ANALYSIS - INDEPENDENT\n")
            f.write("="*80 + "\n\n")

            f.write("REJECTION REASONS\n")
            f.write("-"*80 + "\n")
            f.write(reason_df.to_string(index=False) + "\n\n")

            f.write("SPIKE TEST RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(spike_df.to_string(index=False) + "\n\n")

            f.write("PATTERN ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(json.dumps(patterns, indent=2) + "\n\n")

        print(f"\nFull analysis saved to: {output_path}")

        return spike_df, patterns


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_rejected_breakouts.py <filtered_run_dir> [sample_size] [output_file]")
        print("\nExample:")
        print("  python analyze_rejected_breakouts.py \\")
        print("    backtest_20251108-124615_extracted/20251108-124615_full/20251108-124615 \\")
        print("    100 \\")
        print("    REJECTED_BREAKOUT_ANALYSIS.txt")
        sys.exit(1)

    filtered_run_dir = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    output_file = sys.argv[3] if len(sys.argv) > 3 else "REJECTED_BREAKOUT_ANALYSIS.txt"

    analyzer = RejectedBreakoutAnalyzer(filtered_run_dir)
    spike_df, patterns = analyzer.run_full_analysis(sample_size, output_file)
