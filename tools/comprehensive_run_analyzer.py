#!/usr/bin/env python3
"""
Comprehensive Run Analysis Script

Usage: python comprehensive_run_analyzer.py <run_prefix>
Example: python comprehensive_run_analyzer.py run_146c0d45_

This script:
1. Finds all log folders matching the prefix
2. Combines all trades.csv files
3. Performs deep analysis on setup performance, timing, regime effectiveness
4. Generates actionable recommendations for system improvements
5. Outputs detailed reports that can be fed back for optimization

The analysis is designed to provide data-driven insights for:
- Setup filtering recommendations
- Regime-specific adjustments
- Risk management improvements
- Entry/exit timing optimization
- Performance enhancement opportunities
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveRunAnalyzer:
    def __init__(self, run_prefix: str, logs_dir: str = "logs", ohlcv_cache_dir: str = "cache/ohlcv_archive", baseline_run_prefix: str = None):
        self.run_prefix = run_prefix
        self.logs_dir = logs_dir
        self.ohlcv_cache_dir = ohlcv_cache_dir
        self.baseline_run_prefix = baseline_run_prefix  # NEW: For baseline comparison
        self.sessions = []
        self.combined_trades = pd.DataFrame()
        self.combined_decisions = pd.DataFrame()
        self.ohlcv_cache = {}  # Cache loaded OHLCV data
        self.performance_summary = {}
        self.actionable_insights = {}

    def find_sessions(self):
        """Find all log folders matching the run prefix"""
        pattern = os.path.join(self.logs_dir, f"{self.run_prefix}*")
        self.sessions = sorted(glob.glob(pattern))
        print(f"Found {len(self.sessions)} sessions for prefix '{self.run_prefix}'")
        return len(self.sessions)

    def load_and_combine_trades(self):
        """Load and combine all trade data from matching sessions including events.jsonl"""
        all_trades = []
        all_executed_trades = []
        all_decisions = []
        session_summary = {}

        for session_dir in self.sessions:
            session_name = os.path.basename(session_dir)
            trades_file = os.path.join(session_dir, "trade_report.csv")
            analytics_file = os.path.join(session_dir, "analytics.jsonl")
            events_file = os.path.join(session_dir, "events.jsonl")

            # Load planned trades from CSV
            if os.path.exists(trades_file):
                try:
                    df = pd.read_csv(trades_file)
                    if not df.empty:
                        df['session'] = session_name
                        df['data_source'] = 'planned'
                        all_trades.append(df)
                except Exception as e:
                    print(f"Error loading {trades_file}: {e}")

            # Load executed trades from analytics.jsonl
            if os.path.exists(analytics_file):
                try:
                    executed_trades = []
                    with open(analytics_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    trade_data = json.loads(line)
                                    if trade_data.get('stage') == 'EXIT' and 'pnl' in trade_data:
                                        # Convert to DataFrame-compatible format
                                        executed_trade = {
                                            'session': session_name,
                                            'symbol': trade_data.get('symbol', ''),
                                            'trade_id': trade_data.get('trade_id', ''),
                                            'setup_type': trade_data.get('setup_type', ''),
                                            'regime': trade_data.get('regime', ''),
                                            'bias': trade_data.get('bias', ''),
                                            'strategy': trade_data.get('strategy', ''),
                                            'entry_reference': trade_data.get('entry_reference', 0),
                                            'entry_price': trade_data.get('actual_entry_price', trade_data.get('entry_price', 0)),  # Use actual_entry_price first
                                            'qty': trade_data.get('qty', 0),
                                            'exit_price': trade_data.get('exit_price', 0),
                                            'realized_pnl': trade_data.get('pnl', 0),
                                            'exit_reason': trade_data.get('reason', ''),
                                            'exit_ts': trade_data.get('timestamp', ''),
                                            'elapsed_from_decision': trade_data.get('elapsed_from_decision', 0),
                                            'data_source': 'executed'
                                        }
                                        executed_trades.append(executed_trade)
                                except json.JSONDecodeError:
                                    continue

                    if executed_trades:
                        executed_df = pd.DataFrame(executed_trades)
                        all_executed_trades.append(executed_df)

                except Exception as e:
                    print(f"Error loading {analytics_file}: {e}")

            # Load decision events from events.jsonl
            if os.path.exists(events_file):
                try:
                    decisions = []
                    with open(events_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    event_data = json.loads(line)
                                    if event_data.get('type') == 'DECISION':
                                        # Extract key decision data
                                        decision = {
                                            'session': session_name,
                                            'trade_id': event_data.get('trade_id', ''),
                                            'symbol': event_data.get('symbol', ''),
                                            'ts': event_data.get('ts', ''),
                                            'decision_setup_type': event_data.get('decision', {}).get('setup_type', ''),
                                            'decision_regime': event_data.get('decision', {}).get('regime', ''),
                                            'acceptance_status': event_data.get('plan', {}).get('quality', {}).get('acceptance_status', ''),
                                            'entry_reference': event_data.get('plan', {}).get('entry', {}).get('reference', 0),
                                            'stop_hard': event_data.get('plan', {}).get('stop', {}).get('hard', 0),
                                            'target_t1': event_data.get('plan', {}).get('targets', [{}])[0].get('level', 0) if event_data.get('plan', {}).get('targets') else 0,
                                            'target_t2': event_data.get('plan', {}).get('targets', [{}])[1].get('level', 0) if len(event_data.get('plan', {}).get('targets', [])) > 1 else 0,
                                            'rr_t1': event_data.get('plan', {}).get('targets', [{}])[0].get('rr', 0) if event_data.get('plan', {}).get('targets') else 0,
                                            'rr_t2': event_data.get('plan', {}).get('targets', [{}])[1].get('rr', 0) if len(event_data.get('plan', {}).get('targets', [])) > 1 else 0,
                                            'rank_score': event_data.get('features', {}).get('rank_score', 0),
                                            'vwap': event_data.get('plan', {}).get('indicators', {}).get('vwap', 0),
                                            'ema20': event_data.get('plan', {}).get('indicators', {}).get('ema20', 0),
                                            'atr': event_data.get('plan', {}).get('indicators', {}).get('atr', 0),
                                            'vol_ratio': event_data.get('plan', {}).get('indicators', {}).get('vol_ratio', 0),
                                            'bar5_close': event_data.get('bar5', {}).get('close', 0),
                                            'bar5_volume': event_data.get('bar5', {}).get('volume', 0),
                                            'minute_of_day': event_data.get('timectx', {}).get('minute_of_day', 0),
                                            'data_source': 'decision'
                                        }
                                        decisions.append(decision)
                                except json.JSONDecodeError:
                                    continue

                    if decisions:
                        decisions_df = pd.DataFrame(decisions)
                        all_decisions.append(decisions_df)

                except Exception as e:
                    print(f"Error loading {events_file}: {e}")

            # Calculate session summary
            session_pnl = 0
            session_trades = 0

            if all_executed_trades:
                for df in all_executed_trades:
                    session_data = df[df['session'] == session_name]
                    session_pnl += session_data['realized_pnl'].sum()
                    session_trades += len(session_data)

            session_summary[session_name] = {
                'planned_trades': len(df) if 'df' in locals() and not df.empty else 0,
                'executed_trades': session_trades,
                'pnl': session_pnl
            }

        # Combine executed trades (priority) and planned trades
        if all_executed_trades:
            self.combined_trades = pd.concat(all_executed_trades, ignore_index=True)
            print(f"Combined {len(self.combined_trades)} EXECUTED trades from {len(all_executed_trades)} sessions")
        elif all_trades:
            self.combined_trades = pd.concat(all_trades, ignore_index=True)
            print(f"Combined {len(self.combined_trades)} PLANNED trades from {len(all_trades)} sessions (no executed trades found)")
        else:
            print("No trade data found")
            self.combined_trades = pd.DataFrame()

        # Combine decision events
        if all_decisions:
            self.combined_decisions = pd.concat(all_decisions, ignore_index=True)
            print(f"Combined {len(self.combined_decisions)} DECISION events from {len(all_decisions)} sessions")
        else:
            print("No decision data found")
            self.combined_decisions = pd.DataFrame()

        return session_summary

    def load_ohlcv_data(self, symbol: str, date_str: str = None):
        """Load 1-minute OHLCV data for a symbol from feather files"""
        # Convert NSE:SYMBOL format to SYMBOL.NS format for file lookup
        if symbol.startswith('NSE:'):
            symbol_file = symbol.replace('NSE:', '') + '.NS'
        else:
            symbol_file = symbol

        # Use cache to avoid reloading
        cache_key = f"{symbol_file}_{date_str}"
        if cache_key in self.ohlcv_cache:
            return self.ohlcv_cache[cache_key]

        # Look for minute data files
        symbol_dir = os.path.join(self.ohlcv_cache_dir, symbol_file)
        if not os.path.exists(symbol_dir):
            print(f"Warning: No OHLCV data directory found for {symbol_file}")
            return None

        # Find minute data file - look for 1minute files
        minute_files = glob.glob(os.path.join(symbol_dir, "*1minute*.feather"))

        if not minute_files:
            print(f"Warning: No minute data files found for {symbol_file}")
            return None

        # Try to load the first available minute file
        try:
            minute_file = minute_files[0]  # Use first available file
            df = pd.read_feather(minute_file)

            # Ensure datetime index - handle different column names
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
                df.set_index('datetime', inplace=True)
                df.drop('date', axis=1, inplace=True)
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif df.index.name != 'datetime':
                df.index = pd.to_datetime(df.index)

            # Cache the loaded data
            self.ohlcv_cache[cache_key] = df

            return df

        except Exception as e:
            print(f"Error loading OHLCV data for {symbol_file}: {e}")
            return None

    def get_market_data_at_time(self, symbol: str, timestamp: str, minutes_after: int = 60):
        """Get market data for a symbol at specific timestamp and following period"""
        ohlcv_df = self.load_ohlcv_data(symbol)

        if ohlcv_df is None or ohlcv_df.empty:
            return None

        try:
            # Parse the timestamp and handle timezone
            decision_time = pd.to_datetime(timestamp)

            # If OHLCV data has timezone but decision_time doesn't, align them
            if ohlcv_df.index.tz is not None and decision_time.tz is None:
                # Assume decision time is in IST (UTC+05:30)
                decision_time = decision_time.tz_localize('Asia/Kolkata')
            elif ohlcv_df.index.tz is None and decision_time.tz is not None:
                # Make OHLCV data timezone aware
                ohlcv_df.index = ohlcv_df.index.tz_localize('Asia/Kolkata')

            end_time = decision_time + timedelta(minutes=minutes_after)

            # Filter data for the time window
            time_window = ohlcv_df[
                (ohlcv_df.index >= decision_time) &
                (ohlcv_df.index <= end_time)
            ].copy()

            if time_window.empty:
                return None

            # Calculate key metrics
            start_price = time_window.iloc[0]['close'] if not time_window.empty else None
            high_price = time_window['high'].max()
            low_price = time_window['low'].min()
            end_price = time_window.iloc[-1]['close'] if not time_window.empty else None
            total_volume = time_window['volume'].sum()

            return {
                'start_price': start_price,
                'high_price': high_price,
                'low_price': low_price,
                'end_price': end_price,
                'total_volume': total_volume,
                'price_change': end_price - start_price if start_price and end_price else 0,
                'price_change_pct': ((end_price - start_price) / start_price * 100) if start_price else 0,
                'bars_count': len(time_window)
            }

        except Exception as e:
            print(f"Error getting market data for {symbol} at {timestamp}: {e}")
            return None

    def analyze_setup_performance(self):
        """Detailed analysis of setup performance"""
        if self.combined_trades.empty:
            return {}

        setup_analysis = {}

        # Group by setup type
        if 'setup_type' in self.combined_trades.columns:
            setup_groups = self.combined_trades.groupby('setup_type')

            for setup, group in setup_groups:
                if 'realized_pnl' in group.columns:
                    pnls = group['realized_pnl'].dropna()

                    setup_analysis[setup] = {
                        'total_trades': len(group),
                        'winning_trades': len(pnls[pnls > 0]),
                        'losing_trades': len(pnls[pnls < 0]),
                        'breakeven_trades': len(pnls[pnls == 0]),
                        'win_rate': len(pnls[pnls > 0]) / len(pnls) * 100 if len(pnls) > 0 else 0,
                        'total_pnl': pnls.sum(),
                        'avg_pnl': pnls.mean(),
                        'median_pnl': pnls.median(),
                        'best_trade': pnls.max(),
                        'worst_trade': pnls.min(),
                        'profit_factor': pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum()) if pnls[pnls < 0].sum() != 0 else 999.99,  # Use 999.99 instead of infinity for JSON compatibility
                        'avg_winner': pnls[pnls > 0].mean() if len(pnls[pnls > 0]) > 0 else 0,
                        'avg_loser': pnls[pnls < 0].mean() if len(pnls[pnls < 0]) > 0 else 0,
                    }

                    # Risk-reward analysis
                    if 'label_hit_t1' in group.columns and 'label_hit_t2' in group.columns:
                        setup_analysis[setup]['t1_hit_rate'] = group['label_hit_t1'].sum() / len(group) * 100
                        setup_analysis[setup]['t2_hit_rate'] = group['label_hit_t2'].sum() / len(group) * 100

                    # Exit reason analysis
                    if 'last_exit_reason' in group.columns:
                        exit_reasons = group['last_exit_reason'].value_counts().to_dict()
                        setup_analysis[setup]['exit_reasons'] = exit_reasons

                        # Hard SL rate (critical metric)
                        hard_sl_rate = group['last_exit_reason'].str.contains('hard_sl', na=False).sum() / len(group) * 100
                        setup_analysis[setup]['hard_sl_rate'] = hard_sl_rate

        return setup_analysis

    def analyze_regime_performance(self):
        """Analyze performance by market regime"""
        if self.combined_trades.empty or 'regime' not in self.combined_trades.columns:
            return {}

        regime_analysis = {}
        regime_groups = self.combined_trades.groupby('regime')

        for regime, group in regime_groups:
            if 'realized_pnl' in group.columns:
                pnls = group['realized_pnl'].dropna()

                regime_analysis[regime] = {
                    'total_trades': len(group),
                    'win_rate': len(pnls[pnls > 0]) / len(pnls) * 100 if len(pnls) > 0 else 0,
                    'total_pnl': pnls.sum(),
                    'avg_pnl': pnls.mean(),
                    'best_regime_setup': None,
                    'worst_regime_setup': None
                }

                # Find best/worst setups in this regime
                if 'setup_type' in group.columns:
                    setup_pnl = group.groupby('setup_type')['realized_pnl'].sum().sort_values(ascending=False)
                    if not setup_pnl.empty:
                        regime_analysis[regime]['best_regime_setup'] = setup_pnl.index[0]
                        regime_analysis[regime]['worst_regime_setup'] = setup_pnl.index[-1]

        return regime_analysis

    def analyze_timing_performance(self):
        """Analyze performance by time of day, day of week"""
        if self.combined_trades.empty:
            return {}

        timing_analysis = {}

        # Convert timestamp columns - use available column names
        time_cols = ['exit_ts', 'timestamp']
        timestamp_col = None

        for col in time_cols:
            if col in self.combined_trades.columns:
                self.combined_trades[col] = pd.to_datetime(self.combined_trades[col], errors='coerce')
                timestamp_col = col
                break

        # Hour of day analysis
        if timestamp_col:
            self.combined_trades['trade_hour'] = self.combined_trades[timestamp_col].dt.hour
            hour_analysis = {}

            for hour in range(9, 16):  # Market hours
                hour_trades = self.combined_trades[self.combined_trades['trade_hour'] == hour]
                if not hour_trades.empty and 'realized_pnl' in hour_trades.columns:
                    pnls = hour_trades['realized_pnl'].dropna()
                    hour_analysis[hour] = {
                        'trades': len(hour_trades),
                        'win_rate': len(pnls[pnls > 0]) / len(pnls) * 100 if len(pnls) > 0 else 0,
                        'avg_pnl': pnls.mean()
                    }

            timing_analysis['hourly'] = hour_analysis

        # Day of week analysis
        if timestamp_col:
            self.combined_trades['trade_dow'] = self.combined_trades[timestamp_col].dt.dayofweek
            dow_analysis = {}
            dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

            for dow in range(5):  # Weekdays only
                dow_trades = self.combined_trades[self.combined_trades['trade_dow'] == dow]
                if not dow_trades.empty and 'realized_pnl' in dow_trades.columns:
                    pnls = dow_trades['realized_pnl'].dropna()
                    dow_analysis[dow_names[dow]] = {
                        'trades': len(dow_trades),
                        'win_rate': len(pnls[pnls > 0]) / len(pnls) * 100 if len(pnls) > 0 else 0,
                        'avg_pnl': pnls.mean()
                    }

            timing_analysis['daily'] = dow_analysis

        return timing_analysis

    def analyze_risk_management(self):
        """Analyze risk management effectiveness"""
        if self.combined_trades.empty:
            return {}

        risk_analysis = {}

        # Calculate basic risk metrics from available data
        if 'realized_pnl' in self.combined_trades.columns:
            pnls = self.combined_trades['realized_pnl'].dropna()

            if not pnls.empty:
                # Basic PnL distribution analysis
                risk_analysis['pnl_distribution'] = {
                    'big_winners_1000+': len(pnls[pnls > 1000]),
                    'winners_0_1000': len(pnls[(pnls > 0) & (pnls <= 1000)]),
                    'small_losers_0_500': len(pnls[(pnls < 0) & (pnls >= -500)]),
                    'big_losers_500+': len(pnls[pnls < -500])
                }

                # Trade size analysis
                if 'qty' in self.combined_trades.columns:
                    qty_data = self.combined_trades['qty'].dropna()
                    if not qty_data.empty:
                        risk_analysis['position_sizing'] = {
                            'avg_quantity': qty_data.mean(),
                            'median_quantity': qty_data.median(),
                            'max_quantity': qty_data.max(),
                            'min_quantity': qty_data.min(),
                            'qty_consistency': qty_data.std() / qty_data.mean() if qty_data.mean() != 0 else 0
                        }

                # Exit reason analysis
                if 'exit_reason' in self.combined_trades.columns:
                    # Normalize exit reasons to avoid case duplicates
                    normalized_reasons = self.combined_trades['exit_reason'].str.replace('R', 'r', regex=False)
                    exit_reasons = normalized_reasons.value_counts().to_dict()
                    risk_analysis['exit_patterns'] = exit_reasons

        # Position sizing analysis
        if 'qty' in self.combined_trades.columns and 'plan_notional' in self.combined_trades.columns:
            sizing_data = self.combined_trades.dropna(subset=['qty', 'plan_notional'])
            if not sizing_data.empty:
                risk_analysis['avg_position_size'] = sizing_data['plan_notional'].mean()
                risk_analysis['position_size_consistency'] = sizing_data['plan_notional'].std() / sizing_data['plan_notional'].mean()

        return risk_analysis

    def analyze_decision_quality(self):
        """Analyze decision quality and trigger rates"""
        if self.combined_decisions.empty:
            return {}

        decision_analysis = {}

        # Overall decision statistics
        total_decisions = len(self.combined_decisions)
        unique_symbols = self.combined_decisions['symbol'].nunique()

        # Trigger rate analysis (decisions that resulted in executed trades)
        if not self.combined_trades.empty:
            executed_trade_ids = set(self.combined_trades['trade_id'].tolist())
            triggered_decisions = self.combined_decisions[self.combined_decisions['trade_id'].isin(executed_trade_ids)]
            trigger_rate = len(triggered_decisions) / total_decisions * 100 if total_decisions > 0 else 0
        else:
            triggered_decisions = pd.DataFrame()
            trigger_rate = 0

        decision_analysis['overall'] = {
            'total_decisions': total_decisions,
            'unique_symbols': unique_symbols,
            'triggered_decisions': len(triggered_decisions),
            'trigger_rate': trigger_rate
        }

        # Analysis by setup type
        if 'decision_setup_type' in self.combined_decisions.columns:
            setup_decisions = {}
            for setup, group in self.combined_decisions.groupby('decision_setup_type'):
                setup_triggered = group[group['trade_id'].isin(executed_trade_ids)] if not self.combined_trades.empty else pd.DataFrame()

                setup_decisions[setup] = {
                    'total_decisions': len(group),
                    'triggered': len(setup_triggered),
                    'trigger_rate': len(setup_triggered) / len(group) * 100 if len(group) > 0 else 0,
                    'avg_rank_score': group['rank_score'].mean(),
                    'acceptance_breakdown': group['acceptance_status'].value_counts().to_dict()
                }

            decision_analysis['by_setup'] = setup_decisions

        # Analysis by acceptance status
        if 'acceptance_status' in self.combined_decisions.columns:
            acceptance_analysis = {}
            for status, group in self.combined_decisions.groupby('acceptance_status'):
                status_triggered = group[group['trade_id'].isin(executed_trade_ids)] if not self.combined_trades.empty else pd.DataFrame()

                acceptance_analysis[status] = {
                    'total_decisions': len(group),
                    'triggered': len(status_triggered),
                    'trigger_rate': len(status_triggered) / len(group) * 100 if len(group) > 0 else 0,
                    'avg_rank_score': group['rank_score'].mean()
                }

            decision_analysis['by_acceptance'] = acceptance_analysis

        # Timing analysis for decisions using timestamp
        hourly_decisions = {}
        if 'ts' in self.combined_decisions.columns:
            # Create decision hour from timestamp
            self.combined_decisions['ts_parsed'] = pd.to_datetime(self.combined_decisions['ts'], errors='coerce')
            self.combined_decisions['decision_hour'] = self.combined_decisions['ts_parsed'].dt.hour

            for hour in range(9, 16):  # Market hours 9 AM to 3 PM
                hour_decisions = self.combined_decisions[self.combined_decisions['decision_hour'] == hour]
                if not hour_decisions.empty:
                    hour_triggered = hour_decisions[hour_decisions['trade_id'].isin(executed_trade_ids)] if not self.combined_trades.empty else pd.DataFrame()

                    hourly_decisions[str(hour)] = {
                        'decisions': len(hour_decisions),
                        'triggered': len(hour_triggered),
                        'trigger_rate': len(hour_triggered) / len(hour_decisions) * 100 if len(hour_decisions) > 0 else 0
                    }
        elif 'minute_of_day' in self.combined_decisions.columns:
            # Fallback to minute_of_day if available
            self.combined_decisions['decision_hour'] = self.combined_decisions['minute_of_day'] // 60 + 9

            for hour in range(9, 16):  # Market hours
                hour_decisions = self.combined_decisions[self.combined_decisions['decision_hour'] == hour]
                if not hour_decisions.empty:
                    hour_triggered = hour_decisions[hour_decisions['trade_id'].isin(executed_trade_ids)] if not self.combined_trades.empty else pd.DataFrame()

                    hourly_decisions[str(hour)] = {
                        'decisions': len(hour_decisions),
                        'triggered': len(hour_triggered),
                        'trigger_rate': len(hour_triggered) / len(hour_decisions) * 100 if len(hour_decisions) > 0 else 0
                    }

        decision_analysis['hourly_patterns'] = hourly_decisions

        return decision_analysis

    def analyze_market_validation(self):
        """Validate decision quality against actual market movement"""
        if self.combined_decisions.empty:
            return {}

        print("Loading 1-minute market data for validation...")
        validation_results = {}
        successful_validations = 0
        total_decisions = len(self.combined_decisions)

        # Initialize validation tracking
        triggered_ids = set(self.combined_trades['trade_id'].tolist()) if not self.combined_trades.empty else set()

        validation_data = []

        for idx, decision in self.combined_decisions.iterrows():
            symbol = decision['symbol']
            timestamp = decision['ts']
            trade_id = decision['trade_id']
            was_triggered = trade_id in triggered_ids

            # Get market data for 60 minutes after decision
            market_data = self.get_market_data_at_time(symbol, timestamp, minutes_after=60)

            if market_data is None:
                if idx < 5:  # Debug first few failures
                    print(f"  No market data for {symbol} at {timestamp}")
                continue

            # Extract planned levels
            entry_ref = decision.get('entry_reference', 0)
            stop_hard = decision.get('stop_hard', 0)
            target_t1 = decision.get('target_t1', 0)
            target_t2 = decision.get('target_t2', 0)

            # Skip if essential data missing
            if not all([entry_ref, stop_hard, target_t1]):
                continue

            # Determine trade direction (long/short based on targets vs entry)
            is_long = target_t1 > entry_ref if target_t1 > 0 else True

            # Check what would have happened
            high_price = market_data['high_price']
            low_price = market_data['low_price']

            # For long trades
            if is_long:
                stop_hit = low_price <= stop_hard
                t1_hit = high_price >= target_t1
                t2_hit = high_price >= target_t2 if target_t2 > 0 else False
            else:
                # For short trades
                stop_hit = high_price >= stop_hard
                t1_hit = low_price <= target_t1
                t2_hit = low_price <= target_t2 if target_t2 > 0 else False

            # Calculate hypothetical outcome
            if stop_hit and (not t1_hit or (target_t1 - entry_ref) * (stop_hard - entry_ref) > 0):
                # Stop hit first
                hypothetical_outcome = 'stop_loss'
                hypothetical_pnl_pct = (stop_hard - entry_ref) / entry_ref * 100 if is_long else (entry_ref - stop_hard) / entry_ref * 100
            elif t2_hit:
                hypothetical_outcome = 'target_t2'
                hypothetical_pnl_pct = (target_t2 - entry_ref) / entry_ref * 100 if is_long else (entry_ref - target_t2) / entry_ref * 100
            elif t1_hit:
                hypothetical_outcome = 'target_t1'
                hypothetical_pnl_pct = (target_t1 - entry_ref) / entry_ref * 100 if is_long else (entry_ref - target_t1) / entry_ref * 100
            else:
                hypothetical_outcome = 'no_result'
                end_price = market_data['end_price']
                hypothetical_pnl_pct = (end_price - entry_ref) / entry_ref * 100 if is_long else (entry_ref - end_price) / entry_ref * 100

            validation_entry = {
                'trade_id': trade_id,
                'symbol': symbol,
                'timestamp': timestamp,
                'was_triggered': was_triggered,
                'acceptance_status': decision.get('acceptance_status', ''),
                'rank_score': decision.get('rank_score', 0),
                'setup_type': decision.get('decision_setup_type', ''),
                'is_long': is_long,
                'entry_reference': entry_ref,
                'stop_hard': stop_hard,
                'target_t1': target_t1,
                'target_t2': target_t2,
                'market_high': high_price,
                'market_low': low_price,
                'market_change_pct': market_data['price_change_pct'],
                'hypothetical_outcome': hypothetical_outcome,
                'hypothetical_pnl_pct': hypothetical_pnl_pct,
                'stop_hit': stop_hit,
                't1_hit': t1_hit,
                't2_hit': t2_hit
            }
            validation_data.append(validation_entry)
            successful_validations += 1

        if not validation_data:
            print("No market validation data available")
            return {}

        validation_df = pd.DataFrame(validation_data)
        print(f"Successfully validated {successful_validations}/{total_decisions} decisions against market data")

        # Analyze validation results
        validation_results = {
            'summary': {
                'total_validated': successful_validations,
                'validation_rate': successful_validations / total_decisions * 100
            }
        }

        # Compare triggered vs non-triggered performance
        if 'was_triggered' in validation_df.columns:
            triggered_decisions = validation_df[validation_df['was_triggered'] == True]
            non_triggered_decisions = validation_df[validation_df['was_triggered'] == False]

            validation_results['triggered_vs_non_triggered'] = {
                'triggered': {
                    'count': len(triggered_decisions),
                    'avg_hypothetical_pnl': triggered_decisions['hypothetical_pnl_pct'].mean(),
                    'success_rate': len(triggered_decisions[triggered_decisions['hypothetical_pnl_pct'] > 0]) / len(triggered_decisions) * 100 if len(triggered_decisions) > 0 else 0
                },
                'non_triggered': {
                    'count': len(non_triggered_decisions),
                    'avg_hypothetical_pnl': non_triggered_decisions['hypothetical_pnl_pct'].mean(),
                    'success_rate': len(non_triggered_decisions[non_triggered_decisions['hypothetical_pnl_pct'] > 0]) / len(non_triggered_decisions) * 100 if len(non_triggered_decisions) > 0 else 0
                }
            }

        # Quality vs actual performance
        if 'acceptance_status' in validation_df.columns:
            quality_performance = {}
            for status in validation_df['acceptance_status'].unique():
                status_data = validation_df[validation_df['acceptance_status'] == status]
                quality_performance[status] = {
                    'count': len(status_data),
                    'avg_hypothetical_pnl': status_data['hypothetical_pnl_pct'].mean(),
                    'success_rate': len(status_data[status_data['hypothetical_pnl_pct'] > 0]) / len(status_data) * 100 if len(status_data) > 0 else 0,
                    'trigger_rate': len(status_data[status_data['was_triggered'] == True]) / len(status_data) * 100 if len(status_data) > 0 else 0
                }

            validation_results['quality_vs_performance'] = quality_performance

        # Store detailed validation data
        self.validation_data = validation_df

        return validation_results

    def analyze_indicator_effectiveness(self):
        """Analyze how well technical indicators predicted market outcomes"""
        if self.combined_decisions.empty or not hasattr(self, 'validation_data'):
            return {}

        print("Analyzing indicator effectiveness...")

        # Merge decision indicators with market validation results
        decisions_with_indicators = []

        for idx, decision in self.combined_decisions.iterrows():
            # Extract indicators from flattened columns
            indicator_cols = ['vwap', 'ema20', 'ema50', 'rsi14', 'adx14', 'macd_hist', 'vol_ratio', 'atr']
            indicators = {}
            for col in indicator_cols:
                if col in decision and pd.notna(decision[col]) and decision[col] != 0:
                    indicators[col] = decision[col]

            if not indicators:
                continue

            # Find corresponding validation result
            trade_id = decision['trade_id']
            validation_row = self.validation_data[self.validation_data['trade_id'] == trade_id]

            if validation_row.empty:
                continue

            validation_result = validation_row.iloc[0]

            # Combine data (indicators already flattened in combined_decisions)
            combined_row = {
                'trade_id': trade_id,
                'symbol': decision['symbol'],
                'setup_type': decision.get('setup_type', ''),
                'acceptance_status': decision.get('acceptance_status', ''),
                'was_triggered': validation_result['was_triggered'],
                'hypothetical_pnl_pct': validation_result['hypothetical_pnl_pct'],
                'market_success': validation_result['hypothetical_pnl_pct'] > 0,
                'price': decision.get('entry_ref', 0),  # Add decision price
                **indicators  # Add all indicator values
            }
            decisions_with_indicators.append(combined_row)

        if not decisions_with_indicators:
            return {}

        indicator_df = pd.DataFrame(decisions_with_indicators)

        # Analyze each indicator's predictive power
        indicator_analysis = {}

        # Key indicators to analyze
        key_indicators = ['vwap', 'ema20', 'ema50', 'rsi14', 'adx14', 'macd_hist', 'vol_ratio', 'atr']

        for indicator in key_indicators:
            if indicator not in indicator_df.columns:
                continue

            # Remove NaN values
            valid_data = indicator_df.dropna(subset=[indicator, 'hypothetical_pnl_pct'])
            if len(valid_data) < 10:  # Need minimum data points
                continue

            # Analyze indicator vs market performance
            indicator_analysis[indicator] = self._analyze_single_indicator(valid_data, indicator)

        # Price-to-indicator relationship analysis
        price_relationships = {}
        if all(col in indicator_df.columns for col in ['price', 'vwap', 'ema20', 'ema50']):
            price_relationships = self._analyze_price_relationships(indicator_df)

        return {
            'indicator_effectiveness': indicator_analysis,
            'price_relationships': price_relationships,
            'sample_size': len(indicator_df),
            'indicators_available': list(key_indicators)
        }

    def _analyze_single_indicator(self, data, indicator_name):
        """Analyze a single indicator's effectiveness"""
        # Split into quartiles
        quartiles = data[indicator_name].quantile([0.25, 0.5, 0.75]).tolist()

        performance_by_quartile = {}

        # Q1 (bottom 25%)
        q1_data = data[data[indicator_name] <= quartiles[0]]
        # Q2 (25-50%)
        q2_data = data[(data[indicator_name] > quartiles[0]) & (data[indicator_name] <= quartiles[1])]
        # Q3 (50-75%)
        q3_data = data[(data[indicator_name] > quartiles[1]) & (data[indicator_name] <= quartiles[2])]
        # Q4 (top 25%)
        q4_data = data[data[indicator_name] > quartiles[2]]

        for i, quartile_data in enumerate([q1_data, q2_data, q3_data, q4_data], 1):
            if len(quartile_data) > 0:
                performance_by_quartile[f'Q{i}'] = {
                    'count': len(quartile_data),
                    'avg_return': quartile_data['hypothetical_pnl_pct'].mean(),
                    'success_rate': len(quartile_data[quartile_data['market_success']]) / len(quartile_data) * 100,
                    'trigger_rate': len(quartile_data[quartile_data['was_triggered']]) / len(quartile_data) * 100,
                    'range': [quartile_data[indicator_name].min(), quartile_data[indicator_name].max()]
                }

        # Overall correlation
        correlation = data[indicator_name].corr(data['hypothetical_pnl_pct'])

        return {
            'performance_by_quartile': performance_by_quartile,
            'correlation_with_returns': correlation,
            'range': [data[indicator_name].min(), data[indicator_name].max()],
            'median': data[indicator_name].median()
        }

    def _analyze_price_relationships(self, data):
        """Analyze price relationships with key levels"""
        relationships = {}

        # Price vs VWAP
        if all(col in data.columns for col in ['price', 'vwap']):
            above_vwap = data[data['price'] > data['vwap']]
            below_vwap = data[data['price'] <= data['vwap']]

            relationships['price_vs_vwap'] = {
                'above_vwap': {
                    'count': len(above_vwap),
                    'avg_return': above_vwap['hypothetical_pnl_pct'].mean() if len(above_vwap) > 0 else 0,
                    'success_rate': len(above_vwap[above_vwap['market_success']]) / len(above_vwap) * 100 if len(above_vwap) > 0 else 0
                },
                'below_vwap': {
                    'count': len(below_vwap),
                    'avg_return': below_vwap['hypothetical_pnl_pct'].mean() if len(below_vwap) > 0 else 0,
                    'success_rate': len(below_vwap[below_vwap['market_success']]) / len(below_vwap) * 100 if len(below_vwap) > 0 else 0
                }
            }

        # EMA alignment
        if all(col in data.columns for col in ['price', 'ema20', 'ema50']):
            # Price above both EMAs
            bullish_alignment = data[(data['price'] > data['ema20']) & (data['ema20'] > data['ema50'])]
            # Price below both EMAs
            bearish_alignment = data[(data['price'] < data['ema20']) & (data['ema20'] < data['ema50'])]
            # Mixed signals
            mixed_alignment = data[~data.index.isin(bullish_alignment.index) & ~data.index.isin(bearish_alignment.index)]

            relationships['ema_alignment'] = {
                'bullish': {
                    'count': len(bullish_alignment),
                    'avg_return': bullish_alignment['hypothetical_pnl_pct'].mean() if len(bullish_alignment) > 0 else 0,
                    'success_rate': len(bullish_alignment[bullish_alignment['market_success']]) / len(bullish_alignment) * 100 if len(bullish_alignment) > 0 else 0
                },
                'bearish': {
                    'count': len(bearish_alignment),
                    'avg_return': bearish_alignment['hypothetical_pnl_pct'].mean() if len(bearish_alignment) > 0 else 0,
                    'success_rate': len(bearish_alignment[bearish_alignment['market_success']]) / len(bearish_alignment) * 100 if len(bearish_alignment) > 0 else 0
                },
                'mixed': {
                    'count': len(mixed_alignment),
                    'avg_return': mixed_alignment['hypothetical_pnl_pct'].mean() if len(mixed_alignment) > 0 else 0,
                    'success_rate': len(mixed_alignment[mixed_alignment['market_success']]) / len(mixed_alignment) * 100 if len(mixed_alignment) > 0 else 0
                }
            }

        return relationships

    def analyze_quality_calibration(self):
        """Analyze if rank_score actually predicts outcomes - CRITICAL analysis"""
        if not hasattr(self, 'validation_data') or self.validation_data.empty:
            return {}

        print("Analyzing quality calibration (rank_score vs outcomes)...")

        calibration_results = {}

        # Get data with rank scores and outcomes
        calibration_data = []
        for idx, decision in self.combined_decisions.iterrows():
            trade_id = decision['trade_id']
            rank_score = decision.get('rank_score', 0)

            # Find validation result for this decision
            validation_row = self.validation_data[self.validation_data['trade_id'] == trade_id]
            if validation_row.empty:
                continue

            validation_result = validation_row.iloc[0]

            # Find executed trade for this decision (if any)
            executed_trade = None
            if not self.combined_trades.empty:
                executed_trade_row = self.combined_trades[self.combined_trades['trade_id'] == trade_id]
                if not executed_trade_row.empty:
                    executed_trade = executed_trade_row.iloc[0]

            calibration_row = {
                'trade_id': trade_id,
                'rank_score': rank_score,
                'was_triggered': validation_result['was_triggered'],
                'hypothetical_pnl_pct': validation_result['hypothetical_pnl_pct'],
                'market_success': validation_result['hypothetical_pnl_pct'] > 0,
                'acceptance_status': decision.get('acceptance_status', ''),
                'setup_type': decision.get('setup_type', ''),
                'actual_pnl': executed_trade['realized_pnl'] if executed_trade is not None else None,
                'actual_win': executed_trade['realized_pnl'] > 0 if executed_trade is not None else None
            }
            calibration_data.append(calibration_row)

        if not calibration_data:
            return {'error': 'No calibration data available'}

        calib_df = pd.DataFrame(calibration_data)

        # 1. Rank Score Decile Analysis
        calib_df['rank_decile'] = pd.qcut(calib_df['rank_score'], q=10, labels=False, duplicates='drop') + 1

        decile_analysis = {}
        for decile in sorted(calib_df['rank_decile'].unique()):
            decile_data = calib_df[calib_df['rank_decile'] == decile]

            # Market performance metrics
            market_win_rate = len(decile_data[decile_data['market_success']]) / len(decile_data) * 100
            avg_market_return = decile_data['hypothetical_pnl_pct'].mean()

            # Actual trading metrics (for triggered trades only)
            triggered_data = decile_data[decile_data['was_triggered'] == True]
            actual_trades = triggered_data.dropna(subset=['actual_pnl'])

            actual_win_rate = len(actual_trades[actual_trades['actual_win'] == True]) / len(actual_trades) * 100 if len(actual_trades) > 0 else 0
            avg_actual_return = actual_trades['actual_pnl'].mean() if len(actual_trades) > 0 else 0

            decile_analysis[f'decile_{decile}'] = {
                'n_total': len(decile_data),
                'n_triggered': len(triggered_data),
                'trigger_rate': len(triggered_data) / len(decile_data) * 100,
                'rank_score_range': [decile_data['rank_score'].min(), decile_data['rank_score'].max()],
                'market_win_rate': market_win_rate,
                'avg_market_return_pct': avg_market_return,
                'actual_win_rate': actual_win_rate,
                'avg_actual_pnl': avg_actual_return,
                'n_actual_trades': len(actual_trades)
            }

        calibration_results['rank_deciles'] = decile_analysis

        # 2. Acceptance Status vs Outcomes (Confusion Matrix)
        acceptance_confusion = {}
        for status in calib_df['acceptance_status'].unique():
            if status == '':
                continue
            status_data = calib_df[calib_df['acceptance_status'] == status]

            # Market outcomes
            market_wins = len(status_data[status_data['market_success']])
            market_losses = len(status_data[~status_data['market_success']])

            # Actual trade outcomes
            status_triggered = status_data[status_data['was_triggered'] == True]
            actual_trades = status_triggered.dropna(subset=['actual_win'])
            actual_wins = len(actual_trades[actual_trades['actual_win'] == True])
            actual_losses = len(actual_trades[actual_trades['actual_win'] == False])

            acceptance_confusion[status] = {
                'total_decisions': len(status_data),
                'market_wins': market_wins,
                'market_losses': market_losses,
                'market_win_rate': market_wins / len(status_data) * 100,
                'actual_wins': actual_wins,
                'actual_losses': actual_losses,
                'actual_win_rate': actual_wins / len(actual_trades) * 100 if len(actual_trades) > 0 else 0,
                'n_actual_trades': len(actual_trades)
            }

        calibration_results['acceptance_confusion'] = acceptance_confusion

        # 3. Rank Score Correlation Analysis
        correlations = {
            'rank_vs_market_return': calib_df['rank_score'].corr(calib_df['hypothetical_pnl_pct']),
            'rank_vs_trigger_rate': None,  # Calculate per rank bucket
        }

        # Calculate trigger rate correlation by grouping ranks
        rank_bins = pd.qcut(calib_df['rank_score'], q=5, duplicates='drop')
        rank_trigger_corr_data = []
        for bin_label, group in calib_df.groupby(rank_bins):
            avg_rank = group['rank_score'].mean()
            trigger_rate = len(group[group['was_triggered']]) / len(group)
            rank_trigger_corr_data.append({'avg_rank': avg_rank, 'trigger_rate': trigger_rate})

        if rank_trigger_corr_data:
            rank_trigger_df = pd.DataFrame(rank_trigger_corr_data)
            correlations['rank_vs_trigger_rate'] = rank_trigger_df['avg_rank'].corr(rank_trigger_df['trigger_rate'])

        calibration_results['correlations'] = correlations

        # 4. Lift Analysis - Cumulative performance if trading only top K%
        calib_df_sorted = calib_df.sort_values('rank_score', ascending=False)

        lift_analysis = {}
        for top_pct in [10, 20, 30, 50, 75, 100]:
            n_top = int(len(calib_df_sorted) * top_pct / 100)
            top_decisions = calib_df_sorted.head(n_top)

            # Market performance of top K%
            market_pnl = top_decisions['hypothetical_pnl_pct'].sum()
            market_win_rate = len(top_decisions[top_decisions['market_success']]) / len(top_decisions) * 100

            # Actual trading performance of top K%
            top_triggered = top_decisions[top_decisions['was_triggered'] == True]
            top_actual = top_triggered.dropna(subset=['actual_pnl'])
            actual_pnl = top_actual['actual_pnl'].sum() if len(top_actual) > 0 else 0
            actual_win_rate = len(top_actual[top_actual['actual_win']]) / len(top_actual) * 100 if len(top_actual) > 0 else 0

            lift_analysis[f'top_{top_pct}pct'] = {
                'n_decisions': len(top_decisions),
                'market_total_return_pct': market_pnl,
                'market_win_rate': market_win_rate,
                'n_actual_trades': len(top_actual),
                'actual_total_pnl': actual_pnl,
                'actual_win_rate': actual_win_rate
            }

        calibration_results['lift_analysis'] = lift_analysis

        # 5. Critical Finding Summary
        top_decile = decile_analysis.get('decile_10', {})
        bottom_decile = decile_analysis.get('decile_1', {})

        calibration_results['summary'] = {
            'is_ranking_working': correlations['rank_vs_market_return'] > 0.1,
            'top_vs_bottom_decile': {
                'top_market_wr': top_decile.get('market_win_rate', 0),
                'bottom_market_wr': bottom_decile.get('market_win_rate', 0),
                'performance_gap': top_decile.get('market_win_rate', 0) - bottom_decile.get('market_win_rate', 0)
            },
            'rank_correlation': correlations['rank_vs_market_return'],
            'total_decisions_analyzed': len(calib_df)
        }

        return calibration_results

    def analyze_sequence_and_risk(self):
        """Analyze trade sequences, streaks, drawdowns, and stop-trading stress tests"""
        if self.combined_trades.empty:
            return {}

        print("Analyzing sequence & risk patterns...")

        sequence_results = {}

        # Sort trades by session and timestamp to get proper sequence
        if 'session' in self.combined_trades.columns and 'exit_ts' in self.combined_trades.columns:
            trades_sorted = self.combined_trades.sort_values(['session', 'exit_ts'])
        else:
            trades_sorted = self.combined_trades.copy()

        # 1. Calculate equity curves per session
        equity_curves = {}
        session_summaries = {}

        if 'session' in trades_sorted.columns:
            for session_name, session_trades in trades_sorted.groupby('session'):
                session_trades_sorted = session_trades.sort_values('exit_ts') if 'exit_ts' in session_trades.columns else session_trades

                # Calculate cumulative PnL
                pnls = session_trades_sorted['realized_pnl'].fillna(0)
                cum_pnl = pnls.cumsum()

                # Build equity curve
                equity_curve = []
                for idx, (_, trade) in enumerate(session_trades_sorted.iterrows()):
                    equity_curve.append({
                        'trade_idx': idx + 1,
                        'trade_id': trade['trade_id'],
                        'pnl': trade['realized_pnl'],
                        'cum_pnl': cum_pnl.iloc[idx],
                        'timestamp': trade.get('exit_ts', '')
                    })

                equity_curves[session_name] = equity_curve

                # Calculate session-level metrics
                session_pnl = pnls.sum()
                max_drawdown = 0
                peak = 0
                for cum_val in cum_pnl:
                    if cum_val > peak:
                        peak = cum_val
                    drawdown = peak - cum_val
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

                session_summaries[session_name] = {
                    'total_trades': len(session_trades_sorted),
                    'total_pnl': session_pnl,
                    'max_intraday_drawdown': max_drawdown,
                    'final_cum_pnl': cum_pnl.iloc[-1] if len(cum_pnl) > 0 else 0
                }

        sequence_results['equity_curves'] = equity_curves
        sequence_results['session_summaries'] = session_summaries

        # 2. Streak Analysis (wins/losses in a row)
        wins_losses = (trades_sorted['realized_pnl'] > 0).astype(int)  # 1 for win, 0 for loss

        # Calculate streaks
        def calculate_streaks(series):
            streaks = []
            current_streak = 1
            current_value = series.iloc[0] if len(series) > 0 else 0

            for i in range(1, len(series)):
                if series.iloc[i] == current_value:
                    current_streak += 1
                else:
                    streaks.append((current_value, current_streak))
                    current_value = series.iloc[i]
                    current_streak = 1

            if len(series) > 0:
                streaks.append((current_value, current_streak))

            return streaks

        all_streaks = calculate_streaks(wins_losses)
        win_streaks = [length for is_win, length in all_streaks if is_win == 1]
        loss_streaks = [length for is_win, length in all_streaks if is_win == 0]

        streak_analysis = {
            'overall': {
                'max_consec_wins': max(win_streaks) if win_streaks else 0,
                'max_consec_losses': max(loss_streaks) if loss_streaks else 0,
                'avg_win_streak': sum(win_streaks) / len(win_streaks) if win_streaks else 0,
                'avg_loss_streak': sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0,
                'total_win_streaks': len(win_streaks),
                'total_loss_streaks': len(loss_streaks)
            }
        }

        # Per-session streak analysis
        session_streaks = {}
        if 'session' in trades_sorted.columns:
            for session_name, session_trades in trades_sorted.groupby('session'):
                session_wins_losses = (session_trades['realized_pnl'] > 0).astype(int)
                session_streaks_data = calculate_streaks(session_wins_losses)
                session_win_streaks = [length for is_win, length in session_streaks_data if is_win == 1]
                session_loss_streaks = [length for is_win, length in session_streaks_data if is_win == 0]

                session_streaks[session_name] = {
                    'max_consec_wins': max(session_win_streaks) if session_win_streaks else 0,
                    'max_consec_losses': max(session_loss_streaks) if session_loss_streaks else 0
                }

        streak_analysis['by_session'] = session_streaks
        sequence_results['streak_analysis'] = streak_analysis

        # 3. Stop Trading Stress Tests
        stop_trading_tests = {}

        # Test different stop rules
        stop_rules = [
            {'name': '-1R', 'type': 'cumulative_r', 'threshold': -1.0},
            {'name': '-2R', 'type': 'cumulative_r', 'threshold': -2.0},
            {'name': '3_losses', 'type': 'consecutive_losses', 'threshold': 3}
        ]

        for rule in stop_rules:
            rule_results = {'sessions_tested': 0, 'sessions_stopped': 0, 'total_delta_pnl': 0, 'delta_wr_points': 0}

            if 'session' in trades_sorted.columns:
                for session_name, session_trades in trades_sorted.groupby('session'):
                    session_trades_sorted = session_trades.sort_values('exit_ts') if 'exit_ts' in session_trades.columns else session_trades
                    rule_results['sessions_tested'] += 1

                    # Simulate stop rule
                    stop_triggered = False
                    stop_index = len(session_trades_sorted)

                    if rule['type'] == 'cumulative_r':
                        # Estimate R as average risk per trade (rough approximation)
                        avg_risk = 500  # Rs 500 average risk assumption
                        cum_r = 0
                        for idx, (_, trade) in enumerate(session_trades_sorted.iterrows()):
                            cum_r += trade['realized_pnl'] / avg_risk
                            if cum_r <= rule['threshold']:
                                stop_triggered = True
                                stop_index = idx + 1
                                break

                    elif rule['type'] == 'consecutive_losses':
                        consecutive_losses = 0
                        for idx, (_, trade) in enumerate(session_trades_sorted.iterrows()):
                            if trade['realized_pnl'] <= 0:
                                consecutive_losses += 1
                                if consecutive_losses >= rule['threshold']:
                                    stop_triggered = True
                                    stop_index = idx + 1
                                    break
                            else:
                                consecutive_losses = 0

                    if stop_triggered:
                        rule_results['sessions_stopped'] += 1

                        # Calculate impact
                        kept_trades = session_trades_sorted.iloc[:stop_index]
                        removed_trades = session_trades_sorted.iloc[stop_index:]

                        delta_pnl = -removed_trades['realized_pnl'].sum()  # Negative because we avoid these

                        original_wr = len(session_trades_sorted[session_trades_sorted['realized_pnl'] > 0]) / len(session_trades_sorted) * 100
                        new_wr = len(kept_trades[kept_trades['realized_pnl'] > 0]) / len(kept_trades) * 100 if len(kept_trades) > 0 else 0
                        delta_wr = new_wr - original_wr

                        rule_results['total_delta_pnl'] += delta_pnl
                        rule_results['delta_wr_points'] += delta_wr

            # Calculate averages
            if rule_results['sessions_tested'] > 0:
                rule_results['stop_rate'] = rule_results['sessions_stopped'] / rule_results['sessions_tested'] * 100
                rule_results['avg_delta_pnl_per_session'] = rule_results['total_delta_pnl'] / rule_results['sessions_tested']
                rule_results['avg_delta_wr_points'] = rule_results['delta_wr_points'] / rule_results['sessions_tested']

            stop_trading_tests[rule['name']] = rule_results

        sequence_results['stop_trading_stress_tests'] = stop_trading_tests

        # 4. Overall Risk Metrics
        if not trades_sorted['realized_pnl'].empty:
            all_pnls = trades_sorted['realized_pnl']

            # Calculate overall maximum drawdown
            cum_pnl_overall = all_pnls.cumsum()
            overall_max_dd = 0
            overall_peak = 0
            for cum_val in cum_pnl_overall:
                if cum_val > overall_peak:
                    overall_peak = cum_val
                drawdown = overall_peak - cum_val
                if drawdown > overall_max_dd:
                    overall_max_dd = drawdown

            risk_metrics = {
                'overall_max_drawdown': overall_max_dd,
                'total_sessions': len(session_summaries) if session_summaries else 1,
                'profitable_sessions': len([s for s in session_summaries.values() if s['total_pnl'] > 0]) if session_summaries else 0,
                'avg_session_pnl': sum([s['total_pnl'] for s in session_summaries.values()]) / len(session_summaries) if session_summaries else all_pnls.sum(),
                'worst_session_pnl': min([s['total_pnl'] for s in session_summaries.values()]) if session_summaries else all_pnls.min(),
                'best_session_pnl': max([s['total_pnl'] for s in session_summaries.values()]) if session_summaries else all_pnls.max()
            }

            sequence_results['risk_metrics'] = risk_metrics

        # 5. Summary and Recommendations
        sequence_results['summary'] = {
            'max_consecutive_losses_hit': streak_analysis['overall']['max_consec_losses'],
            'risk_level': 'HIGH' if streak_analysis['overall']['max_consec_losses'] >= 5 else 'MEDIUM' if streak_analysis['overall']['max_consec_losses'] >= 3 else 'LOW',
            'stop_rule_recommendation': None,
            'total_trades_analyzed': len(trades_sorted)
        }

        # Recommend best stop rule
        best_rule = None
        best_score = float('-inf')
        for rule_name, results in stop_trading_tests.items():
            if results.get('sessions_tested', 0) > 0:
                # Score based on PnL improvement and reasonable stop rate
                score = results.get('avg_delta_pnl_per_session', 0) - (results.get('stop_rate', 0) * 5)  # Penalize high stop rates
                if score > best_score:
                    best_score = score
                    best_rule = rule_name

        sequence_results['summary']['stop_rule_recommendation'] = best_rule

        return sequence_results

    def analyze_time_in_trade(self):
        """Analyze time-in-trade patterns to optimize exits and reduce hard_sl rate"""
        if self.combined_trades.empty:
            return {}

        print("Analyzing time-in-trade patterns...")

        time_results = {}

        # Calculate time in trade for each trade
        # Use decision timestamp as entry time and exit timestamp from analytics
        trades_with_time = []
        for idx, trade in self.combined_trades.iterrows():
            trade_id = trade['trade_id']

            # Find the decision timestamp for this trade_id
            decision_row = self.combined_decisions[self.combined_decisions['trade_id'] == trade_id]
            if decision_row.empty:
                continue

            entry_ts = decision_row.iloc[0].get('ts', '')
            exit_ts = trade.get('exit_ts', trade.get('timestamp', ''))

            if entry_ts and exit_ts:
                try:
                    entry_time = pd.to_datetime(entry_ts)
                    exit_time = pd.to_datetime(exit_ts)
                    time_in_trade_minutes = (exit_time - entry_time).total_seconds() / 60

                    trade_data = {
                        'trade_id': trade['trade_id'],
                        'time_in_trade_minutes': time_in_trade_minutes,
                        'realized_pnl': trade.get('realized_pnl', 0),
                        'exit_reason': trade.get('last_exit_reason', '').lower(),
                        'setup_type': trade.get('setup_type', ''),
                        'is_winner': trade.get('realized_pnl', 0) > 0,
                        'session': trade.get('session', ''),
                        'entry_time': entry_time,
                        'exit_time': exit_time
                    }
                    trades_with_time.append(trade_data)
                except Exception as e:
                    continue

        if not trades_with_time:
            return {'error': 'No time data available for analysis'}

        time_df = pd.DataFrame(trades_with_time)

        # 1. Overall Time-in-Trade Statistics
        overall_stats = {
            'total_trades_with_time': len(time_df),
            'avg_time_minutes': time_df['time_in_trade_minutes'].mean(),
            'median_time_minutes': time_df['time_in_trade_minutes'].median(),
            'min_time_minutes': time_df['time_in_trade_minutes'].min(),
            'max_time_minutes': time_df['time_in_trade_minutes'].max(),
            'std_time_minutes': time_df['time_in_trade_minutes'].std()
        }

        time_results['overall_stats'] = overall_stats

        # 2. Time Bucketing Analysis
        # Define time buckets (quartiles + some key breakpoints)
        time_buckets = [
            {'name': '0-5min', 'min': 0, 'max': 5},
            {'name': '5-15min', 'min': 5, 'max': 15},
            {'name': '15-30min', 'min': 15, 'max': 30},
            {'name': '30-60min', 'min': 30, 'max': 60},
            {'name': '60-120min', 'min': 60, 'max': 120},
            {'name': '120min+', 'min': 120, 'max': float('inf')}
        ]

        bucket_analysis = {}
        for bucket in time_buckets:
            bucket_trades = time_df[
                (time_df['time_in_trade_minutes'] >= bucket['min']) &
                (time_df['time_in_trade_minutes'] < bucket['max'])
            ]

            if len(bucket_trades) > 0:
                winners = bucket_trades[bucket_trades['is_winner']]
                losers = bucket_trades[~bucket_trades['is_winner']]

                bucket_analysis[bucket['name']] = {
                    'count': len(bucket_trades),
                    'win_rate': len(winners) / len(bucket_trades) * 100,
                    'avg_pnl': bucket_trades['realized_pnl'].mean(),
                    'median_pnl': bucket_trades['realized_pnl'].median(),
                    'avg_winner': winners['realized_pnl'].mean() if len(winners) > 0 else 0,
                    'avg_loser': losers['realized_pnl'].mean() if len(losers) > 0 else 0,
                    'hard_sl_rate': len(bucket_trades[bucket_trades['exit_reason'].str.contains('hard_sl', na=False)]) / len(bucket_trades) * 100,
                    'target_hit_rate': len(bucket_trades[bucket_trades['exit_reason'].str.contains('target', na=False)]) / len(bucket_trades) * 100
                }

        time_results['time_buckets'] = bucket_analysis

        # 3. Exit Reason vs Time Analysis
        exit_reason_time = {}
        main_exit_reasons = ['hard_sl', 'target', 'or_kill', 'eod']

        for reason in main_exit_reasons:
            reason_trades = time_df[time_df['exit_reason'].str.contains(reason, na=False)]
            if len(reason_trades) > 0:
                exit_reason_time[reason] = {
                    'count': len(reason_trades),
                    'avg_time_minutes': reason_trades['time_in_trade_minutes'].mean(),
                    'median_time_minutes': reason_trades['time_in_trade_minutes'].median(),
                    'avg_pnl': reason_trades['realized_pnl'].mean(),
                    'win_rate': len(reason_trades[reason_trades['is_winner']]) / len(reason_trades) * 100
                }

        time_results['exit_reason_vs_time'] = exit_reason_time

        # 4. Setup Type vs Time Analysis
        setup_time_analysis = {}
        for setup_type in time_df['setup_type'].unique():
            if setup_type and setup_type != '':
                setup_trades = time_df[time_df['setup_type'] == setup_type]
                if len(setup_trades) > 0:
                    setup_time_analysis[setup_type] = {
                        'count': len(setup_trades),
                        'avg_time_minutes': setup_trades['time_in_trade_minutes'].mean(),
                        'median_time_minutes': setup_trades['time_in_trade_minutes'].median(),
                        'avg_pnl': setup_trades['realized_pnl'].mean(),
                        'win_rate': len(setup_trades[setup_trades['is_winner']]) / len(setup_trades) * 100,
                        'hard_sl_rate': len(setup_trades[setup_trades['exit_reason'].str.contains('hard_sl', na=False)]) / len(setup_trades) * 100
                    }

        time_results['setup_vs_time'] = setup_time_analysis

        # 5. Time-based Exit Optimization Analysis
        # Analyze what happens if we force exit at certain time thresholds
        optimization_tests = {}
        time_thresholds = [15, 30, 45, 60, 90, 120]  # minutes

        for threshold in time_thresholds:
            # Simulate forced exit at threshold for trades that went longer
            long_trades = time_df[time_df['time_in_trade_minutes'] > threshold]

            if len(long_trades) > 0:
                # Estimate PnL at threshold (simplified - assumes linear progression)
                # This is rough but gives directional insight
                estimated_pnl = []
                for _, trade in long_trades.iterrows():
                    # Simple heuristic: if trade ended positive, assume it was positive at threshold
                    # If negative, assume it was negative but smaller loss
                    actual_pnl = trade['realized_pnl']
                    if actual_pnl > 0:
                        # Winner - assume partial profit at threshold
                        est_pnl = actual_pnl * 0.6  # Rough estimate
                    else:
                        # Loser - assume smaller loss at threshold
                        time_ratio = threshold / trade['time_in_trade_minutes']
                        est_pnl = actual_pnl * time_ratio
                    estimated_pnl.append(est_pnl)

                avg_estimated_pnl = sum(estimated_pnl) / len(estimated_pnl)
                avg_actual_pnl = long_trades['realized_pnl'].mean()

                optimization_tests[f'{threshold}min_exit'] = {
                    'trades_affected': len(long_trades),
                    'current_avg_pnl': avg_actual_pnl,
                    'estimated_forced_exit_pnl': avg_estimated_pnl,
                    'potential_improvement': avg_estimated_pnl - avg_actual_pnl,
                    'hard_sl_rate_affected': len(long_trades[long_trades['exit_reason'].str.contains('hard_sl', na=False)]) / len(long_trades) * 100
                }

        time_results['exit_optimization_tests'] = optimization_tests

        # 6. Intraday Time Pattern Analysis
        if 'entry_time' in time_df.columns:
            time_df['entry_hour'] = time_df['entry_time'].dt.hour
            time_df['entry_minute_of_day'] = time_df['entry_time'].dt.hour * 60 + time_df['entry_time'].dt.minute

            hourly_time_patterns = {}
            for hour in range(9, 16):  # Market hours
                hour_trades = time_df[time_df['entry_hour'] == hour]
                if len(hour_trades) > 0:
                    hourly_time_patterns[str(hour)] = {
                        'count': len(hour_trades),
                        'avg_time_in_trade': hour_trades['time_in_trade_minutes'].mean(),
                        'win_rate': len(hour_trades[hour_trades['is_winner']]) / len(hour_trades) * 100,
                        'hard_sl_rate': len(hour_trades[hour_trades['exit_reason'].str.contains('hard_sl', na=False)]) / len(hour_trades) * 100,
                        'avg_pnl': hour_trades['realized_pnl'].mean()
                    }

            time_results['hourly_entry_patterns'] = hourly_time_patterns

        # 7. Critical Insights and Recommendations
        # Find optimal time threshold based on analysis
        best_threshold = None
        best_improvement = float('-inf')

        for threshold_name, results in optimization_tests.items():
            improvement = results.get('potential_improvement', 0)
            if improvement > best_improvement and results['trades_affected'] > 10:  # Need minimum sample
                best_improvement = improvement
                best_threshold = threshold_name

        # Analyze hard_sl patterns
        hard_sl_trades = time_df[time_df['exit_reason'].str.contains('hard_sl', na=False)]
        hard_sl_avg_time = hard_sl_trades['time_in_trade_minutes'].mean() if len(hard_sl_trades) > 0 else 0

        time_results['insights'] = {
            'hard_sl_avg_time_minutes': hard_sl_avg_time,
            'hard_sl_percentage': len(hard_sl_trades) / len(time_df) * 100,
            'recommended_time_exit': best_threshold,
            'potential_improvement_pnl': best_improvement,
            'longest_losing_trade_minutes': time_df[~time_df['is_winner']]['time_in_trade_minutes'].max() if len(time_df[~time_df['is_winner']]) > 0 else 0,
            'shortest_winning_trade_minutes': time_df[time_df['is_winner']]['time_in_trade_minutes'].min() if len(time_df[time_df['is_winner']]) > 0 else 0
        }

        return time_results

    def generate_actionable_insights(self, setup_analysis, regime_analysis, timing_analysis, risk_analysis):
        """Generate actionable insights and recommendations"""
        insights = {
            'setup_recommendations': {},
            'regime_adjustments': {},
            'timing_optimizations': {},
            'risk_management_improvements': {},
            'priority_actions': []
        }

        # Setup recommendations
        if setup_analysis:
            setup_performance = [(setup, data['total_pnl'], data['win_rate'], data.get('hard_sl_rate', 0))
                               for setup, data in setup_analysis.items()]
            setup_performance.sort(key=lambda x: x[1])  # Sort by total PnL

            # Identify problematic setups
            for setup, total_pnl, win_rate, hard_sl_rate in setup_performance:
                if total_pnl < -500 or win_rate < 25 or hard_sl_rate > 60:
                    insights['setup_recommendations'][setup] = {
                        'action': 'FILTER_OUT' if total_pnl < -1000 else 'REDUCE_EXPOSURE',
                        'reasons': [],
                        'suggested_changes': []
                    }

                    if total_pnl < -500:
                        insights['setup_recommendations'][setup]['reasons'].append(f'Poor PnL: Rs.{total_pnl:.0f}')
                    if win_rate < 25:
                        insights['setup_recommendations'][setup]['reasons'].append(f'Low win rate: {win_rate:.1f}%')
                    if hard_sl_rate > 60:
                        insights['setup_recommendations'][setup]['reasons'].append(f'High hard SL rate: {hard_sl_rate:.1f}%')
                        insights['setup_recommendations'][setup]['suggested_changes'].append('Widen stop loss')
                        insights['setup_recommendations'][setup]['suggested_changes'].append('Improve entry timing')

                elif total_pnl > 200 and win_rate > 40:
                    insights['setup_recommendations'][setup] = {
                        'action': 'BOOST_EXPOSURE',
                        'reasons': [f'Good PnL: Rs.{total_pnl:.0f}', f'Good win rate: {win_rate:.1f}%'],
                        'suggested_changes': ['Increase position sizing', 'Boost ranking multiplier']
                    }

        # Regime adjustments
        if regime_analysis:
            for regime, data in regime_analysis.items():
                if data['total_pnl'] < -200:
                    insights['regime_adjustments'][regime] = {
                        'action': 'REDUCE_EXPOSURE',
                        'multiplier_change': -0.2,
                        'reason': f'Poor performance: Rs.{data["total_pnl"]:.0f}'
                    }
                elif data['total_pnl'] > 300:
                    insights['regime_adjustments'][regime] = {
                        'action': 'INCREASE_EXPOSURE',
                        'multiplier_change': +0.3,
                        'reason': f'Strong performance: Rs.{data["total_pnl"]:.0f}'
                    }

        # Timing optimizations
        if timing_analysis.get('hourly'):
            best_hours = []
            worst_hours = []
            for hour, data in timing_analysis['hourly'].items():
                if data['avg_pnl'] > 50:
                    best_hours.append(hour)
                elif data['avg_pnl'] < -30:
                    worst_hours.append(hour)

            if best_hours:
                insights['timing_optimizations']['boost_hours'] = {
                    'hours': best_hours,
                    'action': 'Increase exposure during these hours'
                }
            if worst_hours:
                insights['timing_optimizations']['avoid_hours'] = {
                    'hours': worst_hours,
                    'action': 'Reduce exposure or filter trades during these hours'
                }

        # Risk management improvements
        if risk_analysis:
            if risk_analysis.get('avg_r_multiple', 0) < 0:
                insights['risk_management_improvements']['r_multiple'] = {
                    'issue': f'Negative average R-multiple: {risk_analysis["avg_r_multiple"]:.2f}',
                    'suggestions': ['Improve entry timing', 'Adjust stop loss levels', 'Review exit strategy']
                }

            r_dist = risk_analysis.get('r_distribution', {})
            if r_dist.get('r_less_than_minus_1', 0) > r_dist.get('r_greater_than_2', 0):
                insights['risk_management_improvements']['r_distribution'] = {
                    'issue': 'More big losses than big wins',
                    'suggestions': ['Tighten stop losses', 'Improve setup quality', 'Consider scaling out at T1']
                }

        # Priority actions
        priority_actions = []

        # Find worst setup
        if setup_analysis:
            worst_setup = min(setup_analysis.items(), key=lambda x: x[1]['total_pnl'])
            if worst_setup[1]['total_pnl'] < -1000:
                priority_actions.append({
                    'priority': 'HIGH',
                    'action': f'Immediately blacklist {worst_setup[0]}',
                    'impact': f'Could save Rs.{abs(worst_setup[1]["total_pnl"]):.0f}'
                })

        # Find regime issues
        if regime_analysis:
            worst_regime = min(regime_analysis.items(), key=lambda x: x[1]['total_pnl'])
            if worst_regime[1]['total_pnl'] < -500:
                priority_actions.append({
                    'priority': 'MEDIUM',
                    'action': f'Reduce exposure in {worst_regime[0]} regime',
                    'impact': f'Could improve Rs.{abs(worst_regime[1]["total_pnl"]):.0f}'
                })

        insights['priority_actions'] = priority_actions
        return insights

    def generate_config_changes(self, insights):
        """Generate specific configuration changes based on insights"""
        config_changes = {
            'blacklist_setups': [],
            'ranking_multiplier_changes': {},
            'regime_multiplier_changes': {},
            'time_based_filters': {},
            'risk_management_updates': {}
        }

        # Setup blacklisting
        for setup, rec in insights['setup_recommendations'].items():
            if rec['action'] == 'FILTER_OUT':
                config_changes['blacklist_setups'].append(setup)
            elif rec['action'] == 'BOOST_EXPOSURE':
                config_changes['ranking_multiplier_changes'][setup] = 1.3
            elif rec['action'] == 'REDUCE_EXPOSURE':
                config_changes['ranking_multiplier_changes'][setup] = 0.8

        # Regime adjustments
        for regime, rec in insights['regime_adjustments'].items():
            config_changes['regime_multiplier_changes'][regime] = rec['multiplier_change']

        # Time filters
        if 'avoid_hours' in insights['timing_optimizations']:
            config_changes['time_based_filters']['blocked_hours'] = insights['timing_optimizations']['avoid_hours']['hours']

        return config_changes

    def save_analysis_report(self, output_file=None):
        """Save comprehensive analysis report"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Save directly to analysis/reports/misc/ folder
            output_file = f"analysis/reports/misc/analysis_report_{self.run_prefix}_{timestamp}.json"

        # Generate report with ANALYSIS ONLY (no recommendations)
        report = {
            'run_prefix': self.run_prefix,
            'analysis_timestamp': datetime.now().isoformat(),
            'sessions_analyzed': len(self.sessions),
            'total_trades': len(self.combined_trades),
            'methodology': {
                'data_sources': {
                    'executed_trades': 'analytics.jsonl - Contains actual executed trades with realized PnL, exit reasons, and timestamps',
                    'decisions': 'events.jsonl - Contains all trading decisions (triggered and non-triggered) with quality assessments, technical indicators, and acceptance status',
                    'market_data': 'cache/ohlcv_archive/[SYMBOL].NS/*.feather - 1-minute OHLCV data for market validation'
                },
                'analysis_methods': {
                    'setup_performance': 'Calculated from executed trades grouped by setup_type field. Win rate = winning_trades/total_trades, profit factor = gross_wins/gross_losses',
                    'regime_analysis': 'Executed trades grouped by regime field (trend_up, trend_down, chop, squeeze). Shows which market conditions favor which setups',
                    'timing_analysis': 'Based on exit timestamps from executed trades. Hourly patterns from 10:00-15:30, daily patterns Monday-Friday',
                    'risk_analysis': 'PnL distribution buckets, position sizing consistency (qty field), exit pattern frequency from exit_reason field',
                    'decision_analysis': 'From events.jsonl decision events. Compares total decisions vs triggered trades, grouped by acceptance_status (excellent/good/poor)',
                    'market_validation': 'Loads 1-minute OHLCV data for each decision timestamp, calculates hypothetical outcomes 60 minutes post-decision. Compares system quality ratings vs actual market performance',
                    'indicator_analysis': 'Analyzes technical indicators (VWAP, EMAs, RSI, ADX, MACD, volume ratio, ATR) from decision events against actual market outcomes. Tests predictive power via quartile analysis and correlations',
                    'quality_calibration': 'CRITICAL ANALYSIS: Tests if rank_score actually predicts outcomes. Includes decile analysis, confusion matrix, lift charts, and correlation analysis to validate ranking system effectiveness',
                    'sequence_analysis': 'Analyzes trade sequences, win/loss streaks, drawdowns, and stop-trading stress tests. Provides equity curves per session and recommends optimal daily loss limits',
                    'time_in_trade_analysis': 'Analyzes time-in-trade patterns to optimize exits and reduce hard_sl rate. Includes time buckets, exit optimization tests, and intraday timing patterns'
                },
                'key_metrics_definitions': {
                    'win_rate': 'Percentage of trades with positive PnL',
                    'profit_factor': 'Total gross profits divided by total gross losses (999.99 indicates no losing trades)',
                    'trigger_rate': 'Percentage of decisions that resulted in actual trades',
                    'market_success_rate': 'Percentage of decisions where price moved favorably within 60 minutes',
                    'avg_return': 'Average percentage price change 60 minutes after decision',
                    'missed_opportunities': 'Performance comparison between triggered vs non-triggered decisions',
                    'indicator_quartiles': 'Technical indicators divided into performance quartiles (Q1-Q4) to identify optimal ranges',
                    'correlation_coefficient': 'Pearson correlation between indicator values and market returns (-1 to +1)',
                    'price_vs_vwap': 'Performance comparison when price is above vs below VWAP at decision time',
                    'ema_alignment': 'Performance based on EMA trend alignment (bullish/bearish/mixed signals)'
                },
                'timezone_handling': 'Decision timestamps assumed IST (Asia/Kolkata), OHLCV data timezone-aligned for accurate correlation',
                'data_quality': {
                    'decisions_validated': f'{len(getattr(self, "validation_data", pd.DataFrame()))}/{len(self.combined_decisions)} decisions had corresponding market data',
                    'sessions_coverage': f'{len(self.sessions)} trading sessions analyzed',
                    'date_range': 'July 2025 (based on available OHLCV archive data)'
                }
            },
            'total_decisions': len(getattr(self, 'combined_decisions', pd.DataFrame())),
            'performance_summary': self.performance_summary,
            'setup_analysis': getattr(self, 'setup_analysis', {}),
            'regime_analysis': getattr(self, 'regime_analysis', {}),
            'timing_analysis': getattr(self, 'timing_analysis', {}),
            'risk_analysis': getattr(self, 'risk_analysis', {}),
            'decision_analysis': getattr(self, 'decision_analysis', {}),
            'indicator_analysis': getattr(self, 'indicator_analysis', {}),
            'quality_calibration': getattr(self, 'quality_calibration', {}),
            'sequence_analysis': getattr(self, 'sequence_analysis', {}),
            'time_in_trade_analysis': getattr(self, 'time_analysis', {}),
            'market_validation': getattr(self, 'market_validation', {}),
            # NEW: Enhanced sections per ANALYSIS_WORKFLOW.md
            'spike_test_analysis': getattr(self, 'spike_test_analysis', {}),
            'rejected_trades_analysis': getattr(self, 'rejected_trades_analysis', {}),
            'regime_validation': getattr(self, 'regime_rank_validation', {}).get('regime_validation', {}),
            'rank_calibration': getattr(self, 'regime_rank_validation', {}).get('rank_calibration', {}),
            'baseline_comparison': getattr(self, 'baseline_comparison', {})
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Analysis report saved to: {output_file}")
        return output_file

    def analyze_spike_tests(self):
        """
        Spike Test Analysis - MFE/MAE/Target Hits/SL Quality
        Per ANALYSIS_WORKFLOW.md Step 2.1

        Analyzes executed trades to determine if SL/target placement was optimal
        """
        print("  - Spike test analysis (MFE/MAE/targets/SL quality)...")

        if self.combined_trades.empty:
            return {}

        spike_analysis = {
            'hard_sl_rate': 0,
            'mfe_stats': {},
            'mae_stats': {},
            'target_hit_rates': {},
            'sl_optimization': {},
            'total_trades_analyzed': 0
        }

        # Calculate hard SL rate
        if 'exit_reason' in self.combined_trades.columns:
            total = len(self.combined_trades)
            hard_sl_count = len(self.combined_trades[self.combined_trades['exit_reason'].str.contains('hard_sl', na=False)])
            spike_analysis['hard_sl_rate'] = (hard_sl_count / total * 100) if total > 0 else 0

        # Build a map of trade_id -> entry_timestamp from TRIGGER events
        trade_entry_times = {}
        for session in self.sessions:
            events_file = os.path.join(session, 'events.jsonl')
            if os.path.exists(events_file):
                try:
                    with open(events_file, 'r') as f:
                        for line in f:
                            try:
                                event = json.loads(line.strip())
                                if event.get('type') == 'TRIGGER':
                                    trade_id = event.get('trade_id')
                                    entry_ts = event.get('ts')
                                    if trade_id and entry_ts:
                                        trade_entry_times[trade_id] = pd.to_datetime(entry_ts)
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    continue

        # For each executed trade, load 1m data and calculate MFE/MAE
        mfe_list = []
        mae_list = []
        t1_hits = 0
        t2_hits = 0
        sl_too_tight_count = 0

        # Limit to avoid long processing
        max_trades = min(len(self.combined_trades), 50)

        for idx, trade in self.combined_trades.head(max_trades).iterrows():
            try:
                symbol = trade['symbol'] if 'symbol' in trade else None
                trade_id = trade['trade_id'] if 'trade_id' in trade else None
                exit_ts = trade['exit_ts'] if 'exit_ts' in trade else None

                if not exit_ts or not symbol or not trade_id:
                    continue

                # Get entry timestamp from TRIGGER events
                entry_ts = trade_entry_times.get(trade_id)
                if not entry_ts:
                    continue

                # Extract date and load full day's 1m data
                date_str = str(exit_ts)[:10]  # YYYY-MM-DD
                ohlcv = self.load_ohlcv_data(symbol, date_str)

                if ohlcv is None or ohlcv.empty:
                    continue

                # Extract trade parameters (use columns from combined_trades)
                # Direction: detect from setup_type (breakout_long/short, failure_fade_long/short)
                setup_type = trade['setup_type'] if 'setup_type' in trade else ''
                direction = 'long' if 'long' in setup_type else 'short'

                entry_price = trade['entry_price'] if 'entry_price' in trade else None

                if not entry_price or entry_price == 0:
                    continue

                # Filter OHLCV data to the trade time window (entry to exit)
                exit_ts_dt = pd.to_datetime(exit_ts)

                # Ensure timezone alignment
                if ohlcv.index.tz is not None and entry_ts.tz is None:
                    entry_ts = entry_ts.tz_localize('Asia/Kolkata')
                    exit_ts_dt = exit_ts_dt.tz_localize('Asia/Kolkata')
                elif ohlcv.index.tz is None and entry_ts.tz is not None:
                    ohlcv.index = ohlcv.index.tz_localize('Asia/Kolkata')

                trade_window = ohlcv[(ohlcv.index >= entry_ts) & (ohlcv.index <= exit_ts_dt)]

                if trade_window.empty:
                    continue

                spike_analysis['total_trades_analyzed'] += 1

                # Calculate MFE and MAE from trade window only
                if direction == 'long':
                    mfe_pct = ((trade_window['high'].max() - entry_price) / entry_price * 100)
                    mae_pct = ((trade_window['low'].min() - entry_price) / entry_price * 100)
                else:  # short
                    mfe_pct = ((entry_price - trade_window['low'].min()) / entry_price * 100)
                    mae_pct = ((entry_price - trade_window['high'].max()) / entry_price * 100)

                mfe_list.append(mfe_pct)
                mae_list.append(mae_pct)

            except Exception as e:
                # Skip trades that fail
                continue

        # Aggregate statistics
        if mfe_list:
            spike_analysis['mfe_stats'] = {
                'avg_mfe_pct': round(np.mean(mfe_list), 2),
                'median_mfe_pct': round(np.median(mfe_list), 2),
                'max_mfe_pct': round(max(mfe_list), 2),
                'trades_analyzed': len(mfe_list)
            }

        if mae_list:
            spike_analysis['mae_stats'] = {
                'avg_mae_pct': round(np.mean(mae_list), 2),
                'median_mae_pct': round(np.median(mae_list), 2),
                'worst_mae_pct': round(min(mae_list), 2),
                'trades_analyzed': len(mae_list)
            }

        return spike_analysis

    def _simulate_hypothetical_trade(self, decision_event, holding_period_minutes=120):
        """
        Simulate what would have happened if we took this rejected trade.

        Returns dict with:
        - would_be_profitable: bool
        - hypothetical_pnl_pct: float (% return)
        - max_favorable: float (best price reached)
        - max_adverse: float (worst price reached)
        """
        try:
            symbol = decision_event.get('symbol', '')
            plan = decision_event.get('plan', {})
            direction = plan.get('bias', 'long')
            trigger_price = plan.get('trigger_price')
            hard_sl = plan.get('stop', {}).get('hard')

            # Get targets
            targets = plan.get('targets', [])
            t1_level = targets[0]['level'] if len(targets) > 0 else None
            t2_level = targets[1]['level'] if len(targets) > 1 else None

            if not trigger_price or not hard_sl:
                return None

            # Get timestamp
            ts = decision_event.get('ts', decision_event.get('timestamp'))
            if not ts:
                return None

            entry_dt = pd.to_datetime(ts)
            date_str = entry_dt.strftime('%Y-%m-%d')

            # Load 1m data
            ohlcv_1m = self.get_market_data_at_time(symbol, ts, minutes_after=holding_period_minutes)

            if ohlcv_1m is None or ohlcv_1m.empty:
                return None

            # Simulate trade from trigger price
            entry_price = trigger_price

            # Filter bars after entry
            bars_after_entry = ohlcv_1m[ohlcv_1m['date'] > entry_dt].copy()

            if bars_after_entry.empty:
                return None

            # Calculate MFE and MAE
            if direction == 'long':
                max_favorable = bars_after_entry['high'].max()
                max_adverse = bars_after_entry['low'].min()
                mfe_pct = (max_favorable - entry_price) / entry_price * 100
                mae_pct = (max_adverse - entry_price) / entry_price * 100

                # Check if SL would have been hit
                sl_hit = max_adverse <= hard_sl

                # Check if targets would have been hit
                t1_hit = t1_level and max_favorable >= t1_level
                t2_hit = t2_level and max_favorable >= t2_level

            else:  # short
                max_favorable = bars_after_entry['low'].min()
                max_adverse = bars_after_entry['high'].max()
                mfe_pct = (entry_price - max_favorable) / entry_price * 100
                mae_pct = (entry_price - max_adverse) / entry_price * 100

                # Check if SL would have been hit
                sl_hit = max_adverse >= hard_sl

                # Check if targets would have been hit
                t1_hit = t1_level and max_favorable <= t1_level
                t2_hit = t2_level and max_favorable <= t2_level

            # Estimate PnL based on exit priority
            if sl_hit:
                # Would have hit SL
                hypothetical_pnl_pct = mae_pct  # Negative
                exit_reason = 'hard_sl'
            elif t2_hit:
                # Would have hit T2
                hypothetical_pnl_pct = mfe_pct * 0.8  # Assume 80% of MFE captured
                exit_reason = 'target_t2'
            elif t1_hit:
                # Would have hit T1
                hypothetical_pnl_pct = mfe_pct * 0.5  # Assume 50% of MFE captured
                exit_reason = 'target_t1'
            else:
                # Would have held until EOD/time exit
                # Use last bar price
                last_price = bars_after_entry.iloc[-1]['close']
                if direction == 'long':
                    hypothetical_pnl_pct = (last_price - entry_price) / entry_price * 100
                else:
                    hypothetical_pnl_pct = (entry_price - last_price) / entry_price * 100
                exit_reason = 'time_exit'

            return {
                'would_be_profitable': hypothetical_pnl_pct > 0,
                'hypothetical_pnl_pct': hypothetical_pnl_pct,
                'max_favorable_pct': mfe_pct,
                'max_adverse_pct': mae_pct,
                'sl_would_hit': sl_hit,
                't1_would_hit': t1_hit,
                't2_would_hit': t2_hit,
                'simulated_exit_reason': exit_reason
            }

        except Exception as e:
            # Silently return None if simulation fails
            return None

    def analyze_rejected_trades(self):
        """
        Rejected Trades Analysis - Missed Opportunities/Gate Effectiveness
        Per ANALYSIS_WORKFLOW.md Step 2.2

        Now includes "What if we took it?" analysis using 1m data
        """
        print("  - Rejected trades analysis (missed opportunities)...")

        rejected_analysis = {
            'total_rejected': 0,
            'rejection_reasons': {},
            'gate_effectiveness': {},
            'missed_opportunities': {
                'total_would_be_winners': 0,
                'total_would_be_losers': 0,
                'total_missed_profit_pct': 0,
                'total_avoided_loss_pct': 0
            }
        }

        # Track per-gate statistics
        gate_stats = defaultdict(lambda: {
            'total_rejected': 0,
            'simulated_count': 0,
            'would_be_winners': 0,
            'would_be_losers': 0,
            'total_hypothetical_pnl_pct': 0,
            'accuracy': 0  # % of rejections that were correct (would have lost)
        })

        # Parse events_decisions.jsonl for REJECT decisions
        rejected_events = []
        for session in self.sessions:
            events_file = os.path.join(session, 'events_decisions.jsonl')
            if not os.path.exists(events_file):
                continue

            try:
                with open(events_file, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            if event.get('action') == 'reject':
                                rejected_events.append(event)
                                rejected_analysis['total_rejected'] += 1

                                # Track rejection reason
                                reason = event.get('rejection_reason', event.get('reason', 'unknown'))
                                rejected_analysis['rejection_reasons'][reason] = rejected_analysis['rejection_reasons'].get(reason, 0) + 1
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"    Warning: Could not parse {events_file}: {e}")

        # Simulate hypothetical outcomes for rejected trades (sample if too many)
        max_simulations = min(len(rejected_events), 100)  # Limit to 100 to avoid long processing

        if rejected_events:
            print(f"    Simulating outcomes for {max_simulations}/{len(rejected_events)} rejected trades...")

            for event in rejected_events[:max_simulations]:
                outcome = self._simulate_hypothetical_trade(event)

                if outcome is None:
                    continue  # Skip if simulation failed

                reason = event.get('rejection_reason', 'unknown')
                gate_stats[reason]['simulated_count'] += 1

                if outcome['would_be_profitable']:
                    gate_stats[reason]['would_be_winners'] += 1
                    gate_stats[reason]['total_hypothetical_pnl_pct'] += outcome['hypothetical_pnl_pct']
                    rejected_analysis['missed_opportunities']['total_would_be_winners'] += 1
                    rejected_analysis['missed_opportunities']['total_missed_profit_pct'] += outcome['hypothetical_pnl_pct']
                else:
                    gate_stats[reason]['would_be_losers'] += 1
                    gate_stats[reason]['total_hypothetical_pnl_pct'] += outcome['hypothetical_pnl_pct']
                    rejected_analysis['missed_opportunities']['total_would_be_losers'] += 1
                    rejected_analysis['missed_opportunities']['total_avoided_loss_pct'] += abs(outcome['hypothetical_pnl_pct'])

        # Calculate gate effectiveness
        for reason, stats in gate_stats.items():
            if stats['simulated_count'] > 0:
                # Accuracy = % of rejections that were correct (would have lost money)
                stats['accuracy'] = (stats['would_be_losers'] / stats['simulated_count']) * 100
                stats['avg_hypothetical_pnl_pct'] = stats['total_hypothetical_pnl_pct'] / stats['simulated_count']

            rejected_analysis['gate_effectiveness'][reason] = {
                'total_rejected': rejected_analysis['rejection_reasons'].get(reason, 0),
                'simulated': stats['simulated_count'],
                'would_be_winners': stats['would_be_winners'],
                'would_be_losers': stats['would_be_losers'],
                'accuracy_pct': round(stats['accuracy'], 1),
                'avg_hypothetical_pnl_pct': round(stats['avg_hypothetical_pnl_pct'], 2),
                'verdict': 'GOOD_FILTER' if stats['accuracy'] > 60 else 'OVER_FILTERING' if stats['accuracy'] < 40 else 'NEUTRAL'
            }

        return rejected_analysis

    def analyze_baseline_comparison(self, baseline_run_prefix=None):
        """
        Baseline Comparison - Same Month Performance Delta
        Per ANALYSIS_WORKFLOW.md Step 4
        """
        if not baseline_run_prefix:
            return {}

        print(f"  - Baseline comparison vs {baseline_run_prefix}...")

        # Find baseline analysis report
        baseline_report_pattern = f"analysis/reports/misc/analysis_report_{baseline_run_prefix}_*.json"
        baseline_files = glob.glob(baseline_report_pattern)

        if not baseline_files:
            print(f"    Warning: No baseline report found for {baseline_run_prefix}")
            return {'error': 'baseline_not_found'}

        try:
            with open(baseline_files[0], 'r') as f:
                baseline_data = json.load(f)
        except Exception as e:
            print(f"    Warning: Could not load baseline: {e}")
            return {'error': str(e)}

        # Compare metrics
        baseline_perf = baseline_data.get('performance_summary', {})
        current_perf = self.performance_summary

        comparison = {
            'baseline_run': baseline_run_prefix,
            'performance_delta': {
                'win_rate': f"{current_perf.get('win_rate', 0) - baseline_perf.get('win_rate', 0):+.1f}%",
                'total_pnl': f"{current_perf.get('total_pnl', 0) - baseline_perf.get('total_pnl', 0):+.0f}",
                'profit_factor': 'TBD',  # Need to calculate profit factor
                'trade_count': f"{current_perf.get('total_trades', 0) - baseline_perf.get('total_trades', 0):+d}"
            },
            'baseline_metrics': baseline_perf,
            'current_metrics': current_perf
        }

        return comparison

    def analyze_regime_and_rank_validation(self):
        """
        Regime Validation & Rank Calibration
        Per ANALYSIS_WORKFLOW.md Step 4.5 & 4.6
        """
        print("  - Regime classification & rank score validation...")

        validation = {
            'regime_validation': {},
            'rank_calibration': {}
        }

        # Regime validation - check if regime classifications make sense
        if 'regime' in self.combined_trades.columns and 'entry_adx' in self.combined_trades.columns:
            regimes = {}
            for regime in self.combined_trades['regime'].unique():
                regime_trades = self.combined_trades[self.combined_trades['regime'] == regime]
                avg_adx = regime_trades['entry_adx'].mean() if 'entry_adx' in regime_trades.columns else None

                regimes[regime] = {
                    'trade_count': len(regime_trades),
                    'avg_adx': avg_adx,
                    'accuracy': 'TBD'  # Needs ADX threshold validation
                }

            validation['regime_validation'] = regimes

        # Rank calibration - check if rank_score predicts outcomes
        if 'rank_score' in self.combined_trades.columns and 'realized_pnl' in self.combined_trades.columns:
            # Calculate correlation
            correlation = self.combined_trades[['rank_score', 'realized_pnl']].corr().iloc[0, 1]

            # Group by quality rating if available
            if 'acceptance_status' in self.combined_trades.columns:
                by_quality = {}
                for quality in self.combined_trades['acceptance_status'].unique():
                    quality_trades = self.combined_trades[self.combined_trades['acceptance_status'] == quality]
                    pnls = quality_trades['realized_pnl'].dropna()
                    by_quality[quality] = {
                        'trade_count': len(quality_trades),
                        'win_rate': len(pnls[pnls > 0]) / len(pnls) * 100 if len(pnls) > 0 else 0,
                        'avg_pnl': pnls.mean() if len(pnls) > 0 else 0
                    }

                validation['rank_calibration'] = {
                    'correlation': correlation,
                    'by_quality': by_quality,
                    'predictive_power': 'STRONG' if correlation > 0.5 else 'MODERATE' if correlation > 0.3 else 'WEAK'
                }

        return validation

    def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 80)
        print(f"COMPREHENSIVE RUN ANALYSIS: {self.run_prefix}")
        print("=" * 80)

        # Step 1: Find sessions
        if self.find_sessions() == 0:
            print("No sessions found. Exiting.")
            return None

        # Step 2: Load and combine trade data
        session_summary = self.load_and_combine_trades()

        if self.combined_trades.empty:
            print("No trade data found. Exiting.")
            return None

        # Step 3: Analyze different aspects
        print("\nRunning detailed analysis...")
        self.setup_analysis = self.analyze_setup_performance()
        self.regime_analysis = self.analyze_regime_performance()
        self.timing_analysis = self.analyze_timing_performance()
        self.risk_analysis = self.analyze_risk_management()
        self.decision_analysis = self.analyze_decision_quality()
        self.market_validation = self.analyze_market_validation()
        self.indicator_analysis = self.analyze_indicator_effectiveness()
        self.quality_calibration = self.analyze_quality_calibration()
        self.sequence_analysis = self.analyze_sequence_and_risk()
        self.time_analysis = self.analyze_time_in_trade()

        # NEW: Enhanced analysis per ANALYSIS_WORKFLOW.md
        self.spike_test_analysis = self.analyze_spike_tests()
        self.rejected_trades_analysis = self.analyze_rejected_trades()
        self.regime_rank_validation = self.analyze_regime_and_rank_validation()

        # Step 4: Skip recommendations generation (analysis only)

        # Step 6: Create performance summary
        if 'realized_pnl' in self.combined_trades.columns:
            pnls = self.combined_trades['realized_pnl'].dropna()
            self.performance_summary = {
                'total_trades': len(self.combined_trades),
                'total_pnl': pnls.sum(),
                'win_rate': len(pnls[pnls > 0]) / len(pnls) * 100 if len(pnls) > 0 else 0,
                'avg_pnl_per_trade': pnls.mean(),
                'best_trade': pnls.max(),
                'worst_trade': pnls.min(),
                'sessions': len(self.sessions)
            }

        # Step 6.5: Baseline comparison (if provided)
        if self.baseline_run_prefix:
            self.baseline_comparison = self.analyze_baseline_comparison(self.baseline_run_prefix)

        # Step 7: Print summary
        self.print_summary()

        # Step 8: Save report
        report_file = self.save_analysis_report()

        return report_file

    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)

        if self.performance_summary:
            print(f"Total Trades: {self.performance_summary['total_trades']}")
            print(f"Total PnL: Rs.{self.performance_summary['total_pnl']:.2f}")
            print(f"Win Rate: {self.performance_summary['win_rate']:.1f}%")
            print(f"Average PnL/Trade: Rs.{self.performance_summary['avg_pnl_per_trade']:.2f}")

        print(f"\nSETUP PERFORMANCE:")
        if self.setup_analysis:
            setup_sorted = sorted(self.setup_analysis.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
            for setup, data in setup_sorted[:5]:  # Top 5
                print(f"  {setup}: Rs.{data['total_pnl']:.0f} ({data['win_rate']:.1f}% WR, {data['total_trades']} trades)")

        # Display regime performance
        print(f"\nREGIME PERFORMANCE:")
        if self.regime_analysis:
            regime_sorted = sorted(self.regime_analysis.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
            for regime, data in regime_sorted:
                print(f"  {regime}: Rs.{data['total_pnl']:.0f} ({data['win_rate']:.1f}% WR, {data['total_trades']} trades)")

        # Display decision quality summary
        if hasattr(self, 'decision_analysis') and self.decision_analysis:
            print(f"\nDECISION QUALITY:")
            overall = self.decision_analysis.get('overall', {})
            print(f"  Total Decisions: {overall.get('total_decisions', 0)}")
            print(f"  Triggered Rate: {overall.get('trigger_rate', 0):.1f}%")
            print(f"  Unique Symbols: {overall.get('unique_symbols', 0)}")

            if 'by_acceptance' in self.decision_analysis:
                print(f"\n  By Acceptance Status:")
                for status, data in self.decision_analysis['by_acceptance'].items():
                    print(f"    {status}: {data['trigger_rate']:.1f}% trigger rate ({data['total_decisions']} decisions)")

        # Display market validation summary
        if hasattr(self, 'market_validation') and self.market_validation:
            print(f"\nMARKET VALIDATION:")
            summary = self.market_validation.get('summary', {})
            print(f"  Validated Decisions: {summary.get('total_validated', 0)}/{summary.get('total_validated', 0)} ({summary.get('validation_rate', 0):.1f}%)")

            # Quality vs performance comparison
            if 'quality_vs_performance' in self.market_validation:
                print(f"\n  Quality vs Market Performance:")
                for status, data in self.market_validation['quality_vs_performance'].items():
                    print(f"    {status}: {data['avg_hypothetical_pnl']:.2f}% avg return ({data['success_rate']:.1f}% success rate)")

            # Triggered vs non-triggered comparison
            if 'triggered_vs_non_triggered' in self.market_validation:
                comp = self.market_validation['triggered_vs_non_triggered']
                print(f"\n  Missed Opportunities:")
                print(f"    Non-triggered avg return: {comp['non_triggered']['avg_hypothetical_pnl']:.2f}%")
                print(f"    Triggered avg return: {comp['triggered']['avg_hypothetical_pnl']:.2f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive Run Analysis per ANALYSIS_WORKFLOW.md')
    parser.add_argument('run_prefix', help='Run prefix to analyze (e.g., run_146c0d45_)')
    parser.add_argument('--baseline', '-b', help='Baseline run prefix for comparison (e.g., run_f2cc7fba_)', default=None)
    args = parser.parse_args()

    run_prefix = args.run_prefix
    baseline = args.baseline

    print(f"Analyzing: {run_prefix}")
    if baseline:
        print(f"Baseline: {baseline}")

    analyzer = ComprehensiveRunAnalyzer(run_prefix, baseline_run_prefix=baseline)

    try:
        report_file = analyzer.run_comprehensive_analysis()
        if report_file:
            print(f"\nAnalysis complete! Report saved to: {report_file}")
            print("You can now feed this report back to Claude for optimization recommendations.")
        else:
            print("Analysis failed - no data found.")
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()