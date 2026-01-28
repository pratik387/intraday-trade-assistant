"""
Analyze Institutional Filter Impact on Breakout Trades

This script:
1. Extracts ALL accepted breakouts from baseline screening.jsonl
2. For each accepted breakout, checks which institutional filter would block it
3. Runs spike tests using 1m OHLC data to validate profitability
4. Generates filter-specific recommendations based on actual P&L data

NO baseline comparison - purely independent analysis of filtered run.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

class InstitutionalFilterAnalyzer:
    def __init__(self, baseline_dir: str):
        self.baseline_dir = Path(baseline_dir)
        self.ohlcv_archive = Path("cache/ohlcv_archive")

        # Load accepted breakouts from screening.jsonl
        self.accepted_breakouts = self._load_accepted_breakouts()

        print(f"Loaded {len(self.accepted_breakouts)} accepted breakout structures")

    def _load_accepted_breakouts(self) -> List[Dict]:
        """Load ALL accepted breakout structures from screening.jsonl files"""
        accepted = []

        for session_dir in sorted(self.baseline_dir.glob('20*')):
            screening_file = session_dir / 'screening.jsonl'
            if not screening_file.exists():
                continue

            with open(screening_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line)

                        # Looking for accepted breakout structures
                        if (event.get('action') == 'accept' and
                            event.get('setup_type') and
                            'breakout' in event.get('setup_type', '').lower()):

                            accepted.append({
                                'session': session_dir.name,
                                'timestamp': event.get('timestamp'),
                                'symbol': event.get('symbol'),
                                'setup_type': event.get('setup_type'),
                                'regime': event.get('regime'),
                                'current_price': event.get('current_price'),
                                'vwap': event.get('vwap'),
                                'all_reasons': event.get('all_reasons', []),
                            })
                    except json.JSONDecodeError:
                        continue

        return accepted

    def check_timing_filter(self, timestamp_str: str) -> Tuple[bool, str]:
        """
        Filter 1: TIMING - Reject 9:15-9:45am (retail noise period)

        Returns (passes, rejection_reason)
        """
        try:
            timestamp = pd.to_datetime(timestamp_str)
            time_minutes = timestamp.hour * 60 + timestamp.minute

            if 555 <= time_minutes < 585:  # 9:15am - 9:45am
                return False, "Timing: 9:15-9:45am retail noise"

            return True, ""
        except:
            return True, ""

    def check_conviction_filter(self, df: pd.DataFrame, is_long: bool) -> Tuple[bool, str]:
        """
        Filter 2: CONVICTION - Close in top 70% (longs) or bottom 30% (shorts)

        Returns (passes, rejection_reason)
        """
        try:
            current_bar = df.iloc[-1]
            bar_range = float(current_bar['high']) - float(current_bar['low'])

            if bar_range < 1e-9:
                return False, "Conviction: Doji candle"

            close_position = (float(current_bar['close']) - float(current_bar['low'])) / bar_range

            if is_long:
                if close_position < 0.7:
                    return False, f"Conviction: Weak long (close at {close_position:.1%})"
            else:
                if close_position > 0.3:
                    return False, f"Conviction: Weak short (close at {close_position:.1%})"

            return True, ""
        except:
            return True, ""

    def check_volume_accumulation_filter(self, df: pd.DataFrame, lookback: int = 5) -> Tuple[bool, str]:
        """
        Filter 3: VOLUME ACCUMULATION - Need 3+ bars with vol_z > 1.0

        Returns (passes, rejection_reason)
        """
        try:
            if 'vol_z' not in df.columns or len(df) < lookback:
                return True, ""

            prior_bars = df['vol_z'].iloc[-(lookback+1):-1]
            institutional_bars = (prior_bars > 1.0).sum()
            min_required = 3

            if institutional_bars < min_required:
                return False, f"Volume: {institutional_bars}/{lookback} bars with vol_z>1.0"

            return True, ""
        except:
            return True, ""

    def check_level_cleanness_filter(self, df: pd.DataFrame, level_value: float,
                                     is_long: bool, lookback: int = 20) -> Tuple[bool, str]:
        """
        Filter 4: LEVEL CLEANNESS - Max 3 touches in last 20 bars

        Returns (passes, rejection_reason)
        """
        try:
            if len(df) < lookback:
                return True, ""

            recent_bars = df.tail(lookback)
            tolerance = level_value * 0.005

            if is_long:
                touches = ((recent_bars['high'] >= level_value - tolerance) &
                          (recent_bars['high'] <= level_value + tolerance)).sum()
            else:
                touches = ((recent_bars['low'] >= level_value - tolerance) &
                          (recent_bars['low'] <= level_value + tolerance)).sum()

            max_allowed_touches = 3

            if touches > max_allowed_touches:
                return False, f"Level cleanness: {touches} touches (max {max_allowed_touches})"

            return True, ""
        except:
            return True, ""

    def load_5m_data_for_breakout(self, symbol: str, session_date: str, timestamp_str: str) -> pd.DataFrame:
        """Load 5m OHLCV data up to the breakout timestamp"""

        # Convert NSE:SYMBOL to SYMBOL.NS (Yahoo Finance format)
        symbol_clean = symbol.replace('NSE:', '') + '.NS'

        # Path format: cache/ohlcv_archive/SYMBOL.NS/SYMBOL.NS_5minutes.feather
        data_file = self.ohlcv_archive / symbol_clean / f"{symbol_clean}_5minutes.feather"

        if not data_file.exists():
            return None

        try:
            df = pd.read_feather(data_file)

            # Feather files use short column names: 'ts', 'o', 'h', 'l', 'c', 'v'
            # Rename to standard names for compatibility
            df = df.rename(columns={'ts': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})

            # Filter to only the session date
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'].dt.date == pd.to_datetime(session_date).date()]

            if len(df) == 0:
                return None

            # Filter to only bars up to and including the breakout time
            breakout_time = pd.to_datetime(timestamp_str)
            # Convert both to timezone-naive for comparison (feather files are timezone-naive)
            if breakout_time.tz is not None:
                breakout_time = breakout_time.tz_localize(None)

            df = df[df['timestamp'] <= breakout_time]

            if len(df) == 0:
                return None

            df = df.set_index('timestamp')
            return df
        except Exception as e:
            # print(f"Error loading {data_file}: {e}")
            return None

    def analyze_filters_for_breakout(self, breakout: Dict) -> Dict:
        """
        Check which institutional filters would block this breakout.

        Returns dict with filter results.
        """
        symbol = breakout['symbol']
        session = breakout['session']
        timestamp = breakout['timestamp']
        setup_type = breakout['setup_type']
        is_long = 'long' in setup_type.lower()

        # Filter 1: Timing
        timing_pass, timing_reason = self.check_timing_filter(timestamp)

        # Load 5m data for remaining filters
        df_5m = self.load_5m_data_for_breakout(symbol, session, timestamp)

        if df_5m is None or len(df_5m) == 0:
            return {
                'symbol': symbol,
                'session': session,
                'timestamp': timestamp,
                'setup_type': setup_type,
                'blocked_by': 'NO_DATA',
                'filter_results': {
                    'timing': (timing_pass, timing_reason),
                    'conviction': (False, "No 5m data"),
                    'volume_accumulation': (False, "No 5m data"),
                    'level_cleanness': (False, "No 5m data"),
                }
            }

        # Filter 2: Conviction
        conviction_pass, conviction_reason = self.check_conviction_filter(df_5m, is_long)

        # Filter 3: Volume accumulation
        volume_pass, volume_reason = self.check_volume_accumulation_filter(df_5m)

        # Filter 4: Level cleanness
        # Need to infer level value from all_reasons
        level_value = breakout.get('current_price')  # Approximation
        cleanness_pass, cleanness_reason = self.check_level_cleanness_filter(
            df_5m, level_value, is_long
        )

        # Determine primary blocker (first filter that fails)
        blocked_by = None
        if not timing_pass:
            blocked_by = 'timing'
        elif not conviction_pass:
            blocked_by = 'conviction'
        elif not volume_pass:
            blocked_by = 'volume_accumulation'
        elif not cleanness_pass:
            blocked_by = 'level_cleanness'
        else:
            blocked_by = None  # Would pass all filters

        return {
            'symbol': symbol,
            'session': session,
            'timestamp': timestamp,
            'setup_type': setup_type,
            'blocked_by': blocked_by,
            'filter_results': {
                'timing': (timing_pass, timing_reason),
                'conviction': (conviction_pass, conviction_reason),
                'volume_accumulation': (volume_pass, volume_reason),
                'level_cleanness': (cleanness_pass, cleanness_reason),
            }
        }

    def load_1m_ohlcv_data(self, symbol: str, session_date: str) -> pd.DataFrame:
        """Load 1-minute OHLCV data for spike testing"""

        # Convert NSE:SYMBOL to SYMBOL.NS (Yahoo Finance format)
        symbol_clean = symbol.replace('NSE:', '') + '.NS'

        # Path format: cache/ohlcv_archive/SYMBOL.NS/SYMBOL.NS_1minutes.feather
        data_file = self.ohlcv_archive / symbol_clean / f"{symbol_clean}_1minutes.feather"

        if not data_file.exists():
            return None

        try:
            df = pd.read_feather(data_file)

            # 1m files already have full column names: 'date', 'open', 'high', 'low', 'close', 'volume'
            # (no renaming needed, unlike 5m files)

            # Filter to only the session date
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.date == pd.to_datetime(session_date).date()]

            if len(df) == 0:
                return None

            df = df.set_index('date')
            return df
        except Exception as e:
            # print(f"Error loading {data_file}: {e}")
            return None

    def simulate_trade(self, breakout: Dict) -> Dict:
        """
        Simulate what would have happened if we took this trade.

        Simple logic:
        - Entry: Current price at breakout time
        - Stop loss: -2% (typical hard SL)
        - Exit: EOD squareoff at 15:15

        Returns simulation results with P&L.
        """
        symbol = breakout['symbol']
        session = breakout['session']
        entry_time_str = breakout['timestamp']
        entry_price = breakout['current_price']

        if entry_price is None:
            return {'simulated': False, 'reason': 'No entry price'}

        # Load 1m data
        df_1m = self.load_1m_ohlcv_data(symbol, session)

        if df_1m is None or len(df_1m) == 0:
            return {'simulated': False, 'reason': 'No 1m data available'}

        # Parse entry time and localize to IST to match 1m data
        try:
            entry_time = pd.to_datetime(entry_time_str)
            # Localize to IST if not already timezone-aware
            if entry_time.tz is None:
                entry_time = entry_time.tz_localize('Asia/Kolkata')
        except:
            return {'simulated': False, 'reason': 'Invalid timestamp'}

        # Find bars after entry
        future_bars = df_1m[df_1m.index > entry_time]

        if len(future_bars) == 0:
            return {'simulated': False, 'reason': 'No future bars'}

        # Determine direction
        is_long = 'long' in breakout['setup_type'].lower()

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

    def run_full_analysis(self, output_file: str = "INSTITUTIONAL_FILTER_IMPACT_ANALYSIS.txt"):
        """Run complete analysis"""

        print(f"\n{'='*80}")
        print("INSTITUTIONAL FILTER IMPACT ANALYSIS")
        print(f"{'='*80}\n")

        # Analyze all accepted breakouts
        print(f"Analyzing {len(self.accepted_breakouts)} accepted breakouts...")

        analyses = []
        simulations = []

        for i, breakout in enumerate(self.accepted_breakouts, 1):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(self.accepted_breakouts)} breakouts...")

            # Check filters
            filter_analysis = self.analyze_filters_for_breakout(breakout)

            # Run spike test
            sim_result = self.simulate_trade(breakout)

            analyses.append(filter_analysis)

            if sim_result.get('simulated'):
                simulations.append({
                    'symbol': breakout['symbol'],
                    'session': breakout['session'],
                    'setup_type': breakout['setup_type'],
                    'blocked_by': filter_analysis['blocked_by'],
                    'pnl': sim_result['pnl'],
                    'pnl_pct': sim_result['pnl_pct'],
                    'exit_reason': sim_result['exit_reason'],
                    'is_winner': sim_result['is_winner'],
                })

        df_analyses = pd.DataFrame(analyses)
        df_simulations = pd.DataFrame(simulations)

        print(f"\nSuccessfully simulated: {len(df_simulations)}/{len(self.accepted_breakouts)} breakouts")

        # Generate filter impact report
        self._generate_filter_impact_report(df_simulations, output_file)

        return df_analyses, df_simulations

    def _generate_filter_impact_report(self, df: pd.DataFrame, output_file: str):
        """Generate comprehensive filter impact report"""

        if len(df) == 0:
            print("\nNo simulations available - cannot generate report")
            return

        print(f"\n{'='*80}")
        print("FILTER IMPACT MATRIX")
        print(f"{'='*80}")

        # Group by blocking filter
        filter_impact = defaultdict(lambda: {'count': 0, 'winners': 0, 'losers': 0,
                                              'total_pnl': 0.0, 'symbols': []})

        for _, row in df.iterrows():
            blocked_by = row['blocked_by'] if row['blocked_by'] else 'WOULD_PASS'
            pnl = row['pnl']
            symbol = row['symbol']
            is_winner = row['is_winner']

            filter_impact[blocked_by]['count'] += 1
            filter_impact[blocked_by]['total_pnl'] += pnl
            filter_impact[blocked_by]['symbols'].append(symbol)

            if is_winner:
                filter_impact[blocked_by]['winners'] += 1
            else:
                filter_impact[blocked_by]['losers'] += 1

        # Print results sorted by opportunity cost
        impact_data = []
        for filter_name, data in sorted(filter_impact.items(),
                                       key=lambda x: x[1]['total_pnl'], reverse=True):
            win_rate = (data['winners'] / data['count'] * 100) if data['count'] > 0 else 0
            avg_pnl = data['total_pnl'] / data['count'] if data['count'] > 0 else 0

            impact_data.append({
                'Filter': filter_name,
                'Blocked_Trades': data['count'],
                'Winners': data['winners'],
                'Losers': data['losers'],
                'Win_Rate_%': round(win_rate, 1),
                'Total_PnL_Rs': round(data['total_pnl'], 2),
                'Avg_PnL_Rs': round(avg_pnl, 2),
            })

        impact_df = pd.DataFrame(impact_data)
        print(impact_df.to_string(index=False))

        # Generate recommendations
        print(f"\n{'='*80}")
        print("FILTER RECOMMENDATIONS")
        print(f"{'='*80}")

        for _, row in impact_df.iterrows():
            filter_name = row['Filter']

            if filter_name == 'WOULD_PASS':
                continue

            if filter_name == 'NO_DATA':
                print(f"\n[SKIP] {filter_name}: Cannot analyze (missing data)")
                continue

            if row['Win_Rate_%'] >= 50 and row['Total_PnL_Rs'] > 0:
                print(f"\n[RELAX/REMOVE] {filter_name}")
                print(f"  This filter blocked {row['Blocked_Trades']} trades with {row['Win_Rate_%']:.1f}% win rate")
                print(f"  Opportunity cost: Rs.{row['Total_PnL_Rs']:.2f}")
                print(f"  RECOMMENDATION: RELAX or REMOVE this filter")
            elif row['Win_Rate_%'] < 40 and row['Total_PnL_Rs'] < 0:
                print(f"\n[KEEP] {filter_name}")
                print(f"  This filter blocked {row['Blocked_Trades']} trades with {row['Win_Rate_%']:.1f}% win rate")
                print(f"  Losses avoided: Rs.{abs(row['Total_PnL_Rs']):.2f}")
                print(f"  RECOMMENDATION: KEEP this filter")
            else:
                print(f"\n[NEUTRAL] {filter_name}")
                print(f"  Win rate: {row['Win_Rate_%']:.1f}%, P&L: Rs.{row['Total_PnL_Rs']:.2f}")
                print(f"  RECOMMENDATION: Requires deeper analysis")

        # Save to file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("INSTITUTIONAL FILTER IMPACT ANALYSIS\n")
            f.write("="*80 + "\n\n")

            f.write("FILTER IMPACT MATRIX\n")
            f.write("-"*80 + "\n")
            f.write(impact_df.to_string(index=False) + "\n\n")

            f.write("DETAILED SIMULATION RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(df.to_string(index=False) + "\n\n")

        print(f"\nFull analysis saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_institutional_filter_impact.py <baseline_dir> [output_file]")
        print("\nExample:")
        print("  python analyze_institutional_filter_impact.py \\")
        print("    backtest_20251108-034930_extracted/20251108-034930_full/20251108-034930 \\")
        print("    INSTITUTIONAL_FILTER_IMPACT_ANALYSIS.txt")
        sys.exit(1)

    baseline_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "INSTITUTIONAL_FILTER_IMPACT_ANALYSIS.txt"

    analyzer = InstitutionalFilterAnalyzer(baseline_dir)
    df_analyses, df_simulations = analyzer.run_full_analysis(output_file)
