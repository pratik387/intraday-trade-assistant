#!/usr/bin/env python3
"""
Trade Price Action Analysis Script

Analyzes each trade using 1-minute OHLCV data to understand:
1. SL hit trades: Did price bounce back to targets? How far down before bounce?
2. Pre-SL behavior: Did price approach T1 before SL hit? Distance from T1?
3. Post-T1 behavior: After T1 hit, how close to T2 before sl_post_t1?
4. Post-sl_post_t1: Did price bounce back to T2? How far down before bounce?
5. Post-T2: How far did price go after T2? Any pullback before continuation?

Usage:
    python tools/analyze_trade_price_action.py <source>

    source can be:
    - backtest_20251127-124353.zip (zip file)
    - backtest_20251127-124353_extracted/ (extracted folder)
    - run_01b0b751_ (run prefix for logs/)
"""

import json
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent


def load_1m_data(symbol: str, date_str: str) -> Optional[pd.DataFrame]:
    """Load 1-minute OHLCV data for symbol on given date."""
    symbol_clean = symbol.replace('NSE:', '')
    cache_base = ROOT / 'cache' / 'ohlcv_archive'

    for suffix in ['.NS', '']:
        symbol_dir = cache_base / f'{symbol_clean}{suffix}'
        if symbol_dir.exists():
            feather_file = symbol_dir / f'{symbol_clean}{suffix}_1minutes.feather'
            feather_files = [feather_file] if feather_file.exists() else list(symbol_dir.glob('*1minutes*.feather'))
            if feather_files:
                df = pd.read_feather(feather_files[0])

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                elif 'ts' in df.columns:
                    df['date'] = pd.to_datetime(df['ts'])

                # Remove timezone
                if df['date'].dt.tz is not None:
                    df['date'] = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df['date']])

                # Filter to trading day
                trade_date = pd.Timestamp(date_str)
                start = trade_date.replace(hour=9, minute=15)
                end = trade_date.replace(hour=15, minute=30)

                df_day = df[(df['date'] >= start) & (df['date'] <= end)]
                return df_day.sort_values('date').reset_index(drop=True)

    return None


def collect_trades_from_backtest(base_path: Path) -> Dict[str, Dict]:
    """Collect trades from backtest extracted folder structure."""
    trades = {}
    plans = {}

    # Find the actual data directory (handles nested structure)
    # Structure: backtest_xxx_extracted/YYYYMMDD-HHMMSS/YYYY-MM-DD/
    data_dirs = []

    # Check if base_path contains date folders directly
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if it's a date folder (YYYY-MM-DD)
            if len(item.name) == 10 and item.name[4] == '-' and item.name[7] == '-':
                data_dirs.append(item)
            else:
                # Check nested structure
                for subitem in item.iterdir():
                    if subitem.is_dir() and len(subitem.name) == 10 and subitem.name[4] == '-':
                        data_dirs.append(subitem)

    if not data_dirs:
        print(f"Warning: No date folders found in {base_path}")
        return {}

    for date_dir in sorted(data_dirs):
        analytics_file = date_dir / 'analytics.jsonl'
        events_file = date_dir / 'events.jsonl'

        # Load analytics for exit info
        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            event = json.loads(line)
                            trade_id = event.get('trade_id')
                            if trade_id and event.get('stage') == 'EXIT':
                                if trade_id not in trades:
                                    trades[trade_id] = {
                                        'symbol': event.get('symbol'),
                                        'trade_id': trade_id,
                                        'setup_type': event.get('setup_type'),
                                        'strategy': event.get('strategy'),
                                        'bias': event.get('bias'),
                                        'regime': event.get('regime'),
                                        'date': date_dir.name,
                                        'exits': []
                                    }
                                trades[trade_id]['exits'].append({
                                    'timestamp': event.get('timestamp'),
                                    'exit_price': event.get('exit_price'),
                                    'reason': event.get('reason'),
                                    'pnl': event.get('pnl'),
                                    'qty': event.get('qty'),
                                    'exit_sequence': event.get('exit_sequence'),
                                    'total_exits': event.get('total_exits'),
                                    'is_final_exit': event.get('is_final_exit'),
                                    'total_trade_pnl': event.get('total_trade_pnl')
                                })
                        except json.JSONDecodeError:
                            continue

        # Load events for plan details
        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            event = json.loads(line)
                            trade_id = event.get('trade_id')
                            if trade_id and event.get('type') == 'DECISION':
                                plan = event.get('plan', {})
                                stop = plan.get('stop', {})
                                targets = plan.get('targets', [])
                                entry = plan.get('entry', {})

                                plans[trade_id] = {
                                    'hard_sl': stop.get('hard'),
                                    'sl_post_t1': stop.get('trail_to'),  # Trailing SL after T1
                                    't1': targets[0]['level'] if len(targets) > 0 else None,
                                    't2': targets[1]['level'] if len(targets) > 1 else None,
                                    't3': targets[2]['level'] if len(targets) > 2 else None,
                                    'entry_zone': entry.get('zone'),
                                    'entry_ref_price': plan.get('entry_ref_price'),
                                    'risk_per_share': stop.get('risk_per_share'),
                                    'ts': event.get('ts'),
                                    'atr': plan.get('indicators', {}).get('atr')
                                }

                            # Also capture EXECUTED events for actual entry time/price
                            if trade_id and event.get('type') == 'EXECUTED':
                                if trade_id in plans:
                                    plans[trade_id]['actual_entry_time'] = event.get('ts')
                                    plans[trade_id]['actual_entry_price'] = event.get('entry_price')
                                else:
                                    plans[trade_id] = {
                                        'actual_entry_time': event.get('ts'),
                                        'actual_entry_price': event.get('entry_price')
                                    }

                            # Also check TRIGGER events (used in backtests instead of EXECUTED)
                            if trade_id and event.get('type') == 'TRIGGER':
                                trigger = event.get('trigger', {})
                                actual_price = trigger.get('actual_price')
                                if actual_price is not None:
                                    if trade_id in plans:
                                        plans[trade_id]['actual_entry_time'] = event.get('ts')
                                        plans[trade_id]['actual_entry_price'] = actual_price
                                    else:
                                        plans[trade_id] = {
                                            'actual_entry_time': event.get('ts'),
                                            'actual_entry_price': actual_price
                                        }
                        except json.JSONDecodeError:
                            continue

    # Merge plans into trades
    for trade_id, plan in plans.items():
        if trade_id in trades:
            trades[trade_id].update(plan)

    return trades


def collect_trades_from_logs(run_prefix: str) -> Dict[str, Dict]:
    """Collect trades from run logs matching prefix."""
    trades = {}
    plans = {}

    logs_dir = ROOT / 'logs'

    for run_dir in logs_dir.glob(f'{run_prefix}*'):
        analytics_file = run_dir / 'analytics.jsonl'
        events_file = run_dir / 'events.jsonl'

        # Extract date from run directory name
        # Format: run_xxx_YYYYMMDD_HHMMSS
        parts = run_dir.name.split('_')
        if len(parts) >= 3:
            date_str = parts[-2]  # YYYYMMDD
            if len(date_str) == 8:
                date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                date_formatted = "unknown"
        else:
            date_formatted = "unknown"

        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            event = json.loads(line)
                            trade_id = event.get('trade_id')
                            if trade_id and event.get('stage') == 'EXIT':
                                if trade_id not in trades:
                                    trades[trade_id] = {
                                        'symbol': event.get('symbol'),
                                        'trade_id': trade_id,
                                        'setup_type': event.get('setup_type'),
                                        'strategy': event.get('strategy'),
                                        'bias': event.get('bias'),
                                        'regime': event.get('regime'),
                                        'date': date_formatted,
                                        'exits': []
                                    }
                                trades[trade_id]['exits'].append({
                                    'timestamp': event.get('timestamp'),
                                    'exit_price': event.get('exit_price'),
                                    'reason': event.get('reason'),
                                    'pnl': event.get('pnl'),
                                    'qty': event.get('qty'),
                                    'exit_sequence': event.get('exit_sequence'),
                                    'is_final_exit': event.get('is_final_exit'),
                                    'total_trade_pnl': event.get('total_trade_pnl')
                                })
                        except json.JSONDecodeError:
                            continue

        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            event = json.loads(line)
                            trade_id = event.get('trade_id')
                            if trade_id and event.get('type') == 'DECISION':
                                plan = event.get('plan', {})
                                stop = plan.get('stop', {})
                                targets = plan.get('targets', [])
                                entry = plan.get('entry', {})

                                plans[trade_id] = {
                                    'hard_sl': stop.get('hard'),
                                    'sl_post_t1': stop.get('trail_to'),
                                    't1': targets[0]['level'] if len(targets) > 0 else None,
                                    't2': targets[1]['level'] if len(targets) > 1 else None,
                                    't3': targets[2]['level'] if len(targets) > 2 else None,
                                    'entry_zone': entry.get('zone'),
                                    'entry_ref_price': plan.get('entry_ref_price'),
                                    'risk_per_share': stop.get('risk_per_share'),
                                    'ts': event.get('ts'),
                                    'atr': plan.get('indicators', {}).get('atr')
                                }

                            if trade_id and event.get('type') == 'EXECUTED':
                                if trade_id in plans:
                                    plans[trade_id]['actual_entry_time'] = event.get('ts')
                                    plans[trade_id]['actual_entry_price'] = event.get('entry_price')
                                else:
                                    plans[trade_id] = {
                                        'actual_entry_time': event.get('ts'),
                                        'actual_entry_price': event.get('entry_price')
                                    }

                            # Also check TRIGGER events (used in backtests instead of EXECUTED)
                            if trade_id and event.get('type') == 'TRIGGER':
                                trigger = event.get('trigger', {})
                                actual_price = trigger.get('actual_price')
                                if actual_price is not None:
                                    if trade_id in plans:
                                        plans[trade_id]['actual_entry_time'] = event.get('ts')
                                        plans[trade_id]['actual_entry_price'] = actual_price
                                    else:
                                        plans[trade_id] = {
                                            'actual_entry_time': event.get('ts'),
                                            'actual_entry_price': actual_price
                                        }
                        except json.JSONDecodeError:
                            continue

    for trade_id, plan in plans.items():
        if trade_id in trades:
            trades[trade_id].update(plan)

    return trades


def analyze_trade(trade: Dict) -> Optional[Dict]:
    """
    Analyze a single trade using 1-minute data.

    Returns detailed analysis including:
    - For SL hits: bounce back analysis
    - For all trades: MFE/MAE from entry
    - Pre-T1, post-T1, post-T2 behavior
    """
    symbol = trade.get('symbol')
    if not symbol:
        return {'error': 'no_symbol'}

    exits = trade.get('exits', [])
    if not exits:
        return {'error': 'no_exits'}

    # Get date from trade
    trade_date = trade.get('date')
    if not trade_date or trade_date == 'unknown':
        # Try to extract from exit timestamp
        first_exit_ts = exits[0].get('timestamp')
        if first_exit_ts:
            trade_date = first_exit_ts.split(' ')[0]
        else:
            return {'error': 'no_date'}

    # Load 1m data
    df_1m = load_1m_data(symbol, trade_date)
    if df_1m is None or df_1m.empty:
        return {'error': 'no_1m_data', 'symbol': symbol, 'date': trade_date}

    # Get trade parameters
    entry_price = trade.get('actual_entry_price') or trade.get('entry_ref_price')
    bias = trade.get('bias', 'long')
    hard_sl = trade.get('hard_sl')
    sl_post_t1 = trade.get('sl_post_t1')  # Trailing SL after T1
    t1 = trade.get('t1')
    t2 = trade.get('t2')
    t3 = trade.get('t3')

    if not entry_price or not hard_sl:
        return {'error': 'missing_params', 'symbol': symbol}

    # Determine entry time
    entry_time_str = trade.get('actual_entry_time') or trade.get('ts')
    if entry_time_str:
        try:
            entry_ts = pd.Timestamp(entry_time_str)
        except:
            entry_ts = pd.Timestamp(trade_date) + timedelta(hours=9, minutes=30)
    else:
        entry_ts = pd.Timestamp(trade_date) + timedelta(hours=9, minutes=30)

    # Get bars AFTER entry
    df_after = df_1m[df_1m['date'] >= entry_ts].copy()
    if df_after.empty:
        return {'error': 'no_bars_after_entry', 'symbol': symbol}

    # Determine final exit reason
    final_exit = next((e for e in exits if e.get('is_final_exit')), exits[-1] if exits else None)
    exit_reason = final_exit.get('reason') if final_exit else 'unknown'
    actual_pnl = final_exit.get('total_trade_pnl') if final_exit else None

    # Initialize result
    result = {
        'trade_id': trade.get('trade_id'),
        'symbol': symbol,
        'date': trade_date,
        'bias': bias,
        'setup_type': trade.get('setup_type'),
        'entry_price': entry_price,
        'entry_time': str(entry_ts),
        'hard_sl': hard_sl,
        'sl_post_t1': sl_post_t1,
        't1': t1,
        't2': t2,
        't3': t3,
        'exit_reason': exit_reason,
        'actual_pnl': actual_pnl,
        'risk_per_share': abs(entry_price - hard_sl),
    }

    # Direction-aware calculations
    is_long = bias == 'long'

    # Calculate MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion)
    if is_long:
        mfe_price = df_after['high'].max()
        mae_price = df_after['low'].min()
        result['mfe'] = mfe_price - entry_price
        result['mae'] = entry_price - mae_price
    else:
        mfe_price = df_after['low'].min()
        mae_price = df_after['high'].max()
        result['mfe'] = entry_price - mfe_price
        result['mae'] = mae_price - entry_price

    result['mfe_pct'] = (result['mfe'] / entry_price) * 100
    result['mae_pct'] = (result['mae'] / entry_price) * 100

    risk = result['risk_per_share']
    if risk > 0:
        result['mfe_in_r'] = result['mfe'] / risk
        result['mae_in_r'] = result['mae'] / risk

    # ===== ANALYSIS BY EXIT TYPE =====

    # ----- 1. SL HIT ANALYSIS -----
    if exit_reason == 'hard_sl':
        result['analysis_type'] = 'sl_hit'

        # Find when SL was hit
        sl_hit_idx = None
        for idx, row in df_after.iterrows():
            if is_long and row['low'] <= hard_sl:
                sl_hit_idx = idx
                break
            elif not is_long and row['high'] >= hard_sl:
                sl_hit_idx = idx
                break

        if sl_hit_idx is not None:
            df_after_sl = df_1m[df_1m.index > sl_hit_idx].copy()

            if not df_after_sl.empty:
                # Did price bounce back to T1 after SL hit?
                if t1:
                    if is_long:
                        bounce_high = df_after_sl['high'].max()
                        result['post_sl_bounce_to_t1'] = bounce_high >= t1
                        result['post_sl_max_favorable'] = bounce_high - hard_sl
                        result['post_sl_closest_to_t1'] = t1 - bounce_high if bounce_high < t1 else 0
                    else:
                        bounce_low = df_after_sl['low'].min()
                        result['post_sl_bounce_to_t1'] = bounce_low <= t1
                        result['post_sl_max_favorable'] = hard_sl - bounce_low
                        result['post_sl_closest_to_t1'] = bounce_low - t1 if bounce_low > t1 else 0

                # How far down did it go BEFORE bouncing back (if it bounced)?
                if is_long:
                    # Track cumulative low until we see new high
                    worst_after_sl = hard_sl
                    best_after_worst = hard_sl
                    for _, row in df_after_sl.iterrows():
                        worst_after_sl = min(worst_after_sl, row['low'])
                        best_after_worst = max(best_after_worst, row['high'])
                    result['post_sl_worst_drawdown'] = hard_sl - worst_after_sl
                    result['post_sl_recovery_from_worst'] = best_after_worst - worst_after_sl
                else:
                    worst_after_sl = hard_sl
                    best_after_worst = hard_sl
                    for _, row in df_after_sl.iterrows():
                        worst_after_sl = max(worst_after_sl, row['high'])
                        best_after_worst = min(best_after_worst, row['low'])
                    result['post_sl_worst_drawdown'] = worst_after_sl - hard_sl
                    result['post_sl_recovery_from_worst'] = worst_after_sl - best_after_worst

        # Did price get close to T1 BEFORE hitting SL?
        if t1:
            df_before_sl = df_after[df_after.index <= sl_hit_idx] if sl_hit_idx else df_after
            if is_long:
                closest_to_t1 = df_before_sl['high'].max()
                result['pre_sl_closest_to_t1'] = closest_to_t1
                result['pre_sl_distance_from_t1'] = t1 - closest_to_t1
                result['pre_sl_t1_reached_pct'] = ((closest_to_t1 - entry_price) / (t1 - entry_price)) * 100 if t1 != entry_price else 0
            else:
                closest_to_t1 = df_before_sl['low'].min()
                result['pre_sl_closest_to_t1'] = closest_to_t1
                result['pre_sl_distance_from_t1'] = closest_to_t1 - t1
                result['pre_sl_t1_reached_pct'] = ((entry_price - closest_to_t1) / (entry_price - t1)) * 100 if t1 != entry_price else 0

    # ----- 2. T1 PARTIAL EXIT (then sl_post_t1) -----
    elif exit_reason == 'sl_post_t1':
        result['analysis_type'] = 'sl_post_t1'

        # Find T1 exit event
        t1_exit = next((e for e in exits if e.get('reason') in ('t1_partial', 'target_t1_partial')), None)
        t1_exit_ts = pd.Timestamp(t1_exit['timestamp']) if t1_exit else None

        if t1_exit_ts and t2:
            # After T1 was hit, how close did it get to T2?
            df_after_t1 = df_1m[df_1m['date'] > t1_exit_ts].copy()

            if not df_after_t1.empty:
                if is_long:
                    closest_to_t2 = df_after_t1['high'].max()
                    result['post_t1_closest_to_t2'] = closest_to_t2
                    result['post_t1_distance_from_t2'] = t2 - closest_to_t2
                    result['post_t1_t2_reached_pct'] = ((closest_to_t2 - t1) / (t2 - t1)) * 100 if t2 != t1 else 0
                else:
                    closest_to_t2 = df_after_t1['low'].min()
                    result['post_t1_closest_to_t2'] = closest_to_t2
                    result['post_t1_distance_from_t2'] = closest_to_t2 - t2
                    result['post_t1_t2_reached_pct'] = ((t1 - closest_to_t2) / (t1 - t2)) * 100 if t2 != t1 else 0

        # After sl_post_t1 hit, did price bounce back to T2?
        final_exit_ts_str = final_exit.get('timestamp') if final_exit else None
        if final_exit_ts_str and t2:
            final_exit_ts = pd.Timestamp(final_exit_ts_str)
            df_after_final = df_1m[df_1m['date'] > final_exit_ts].copy()

            if not df_after_final.empty:
                if is_long:
                    bounce_high = df_after_final['high'].max()
                    result['post_sl_post_t1_bounce_to_t2'] = bounce_high >= t2
                    result['post_sl_post_t1_max_price'] = bounce_high

                    # How far down before bouncing back?
                    sl_post_t1_level = sl_post_t1 if sl_post_t1 else t1  # Trailing SL is usually at entry or T1
                    worst_after = df_after_final['low'].min()
                    result['post_sl_post_t1_worst_drawdown'] = sl_post_t1_level - worst_after if worst_after < sl_post_t1_level else 0
                    result['post_sl_post_t1_recovery'] = bounce_high - worst_after
                else:
                    bounce_low = df_after_final['low'].min()
                    result['post_sl_post_t1_bounce_to_t2'] = bounce_low <= t2
                    result['post_sl_post_t1_max_price'] = bounce_low

                    sl_post_t1_level = sl_post_t1 if sl_post_t1 else t1
                    worst_after = df_after_final['high'].max()
                    result['post_sl_post_t1_worst_drawdown'] = worst_after - sl_post_t1_level if worst_after > sl_post_t1_level else 0
                    result['post_sl_post_t1_recovery'] = worst_after - bounce_low

    # ----- 3. T2 HIT ANALYSIS -----
    elif exit_reason in ('t2_partial', 't2_full', 'target_t2_partial', 'target_t2_full'):
        result['analysis_type'] = 't2_hit'

        # Find T2 exit
        t2_exit = next((e for e in exits if 't2' in e.get('reason', '')), None)
        t2_exit_ts = pd.Timestamp(t2_exit['timestamp']) if t2_exit else None

        if t2_exit_ts:
            df_after_t2 = df_1m[df_1m['date'] > t2_exit_ts].copy()

            if not df_after_t2.empty:
                if is_long:
                    # How far did price go after T2?
                    max_after_t2 = df_after_t2['high'].max()
                    min_after_t2 = df_after_t2['low'].min()
                    result['post_t2_max_extension'] = max_after_t2 - t2
                    result['post_t2_max_pullback'] = t2 - min_after_t2

                    # Track worst pullback before continuation
                    worst_pullback = t2
                    best_after_pullback = t2
                    for _, row in df_after_t2.iterrows():
                        worst_pullback = min(worst_pullback, row['low'])
                        if row['high'] > best_after_pullback:
                            best_after_pullback = row['high']
                    result['post_t2_worst_before_continuation'] = t2 - worst_pullback
                    result['post_t2_continuation_after_pullback'] = best_after_pullback - worst_pullback

                    # Did it reach T3?
                    if t3:
                        result['post_t2_reached_t3'] = max_after_t2 >= t3
                        result['post_t2_distance_from_t3'] = t3 - max_after_t2 if max_after_t2 < t3 else 0
                else:
                    max_after_t2 = df_after_t2['high'].max()
                    min_after_t2 = df_after_t2['low'].min()
                    result['post_t2_max_extension'] = t2 - min_after_t2
                    result['post_t2_max_pullback'] = max_after_t2 - t2

                    worst_pullback = t2
                    best_after_pullback = t2
                    for _, row in df_after_t2.iterrows():
                        worst_pullback = max(worst_pullback, row['high'])
                        if row['low'] < best_after_pullback:
                            best_after_pullback = row['low']
                    result['post_t2_worst_before_continuation'] = worst_pullback - t2
                    result['post_t2_continuation_after_pullback'] = worst_pullback - best_after_pullback

                    if t3:
                        result['post_t2_reached_t3'] = min_after_t2 <= t3
                        result['post_t2_distance_from_t3'] = min_after_t2 - t3 if min_after_t2 > t3 else 0

    # ----- 4. T1 ONLY (exited at T1 without going further) -----
    elif exit_reason in ('t1_partial', 't1_full', 'target_t1_partial', 'target_t1_full'):
        result['analysis_type'] = 't1_only'

        t1_exit = next((e for e in exits if 't1' in e.get('reason', '')), None)
        t1_exit_ts = pd.Timestamp(t1_exit['timestamp']) if t1_exit else None

        if t1_exit_ts and t2:
            df_after_t1 = df_1m[df_1m['date'] > t1_exit_ts].copy()

            if not df_after_t1.empty:
                if is_long:
                    max_after_t1 = df_after_t1['high'].max()
                    result['post_t1_would_hit_t2'] = max_after_t1 >= t2
                    result['post_t1_max_extension'] = max_after_t1 - t1
                    result['post_t1_distance_from_t2'] = t2 - max_after_t1 if max_after_t1 < t2 else 0
                else:
                    min_after_t1 = df_after_t1['low'].min()
                    result['post_t1_would_hit_t2'] = min_after_t1 <= t2
                    result['post_t1_max_extension'] = t1 - min_after_t1
                    result['post_t1_distance_from_t2'] = min_after_t1 - t2 if min_after_t1 > t2 else 0

    else:
        result['analysis_type'] = 'other'

    return result


def generate_summary(results: List[Dict]) -> Dict:
    """Generate summary statistics from all trade analyses."""
    summary = {
        'total_trades': len(results),
        'by_exit_reason': defaultdict(int),
        'by_analysis_type': defaultdict(list),
    }

    for r in results:
        if 'error' in r:
            summary['by_exit_reason']['error'] += 1
            continue

        exit_reason = r.get('exit_reason', 'unknown')
        summary['by_exit_reason'][exit_reason] += 1

        analysis_type = r.get('analysis_type', 'unknown')
        summary['by_analysis_type'][analysis_type].append(r)

    # SL Hit Summary - filter out NaN values
    sl_hits = summary['by_analysis_type'].get('sl_hit', [])
    if sl_hits:
        pre_sl_t1_pcts = [r.get('pre_sl_t1_reached_pct') for r in sl_hits
                         if r.get('pre_sl_t1_reached_pct') is not None and not np.isnan(r.get('pre_sl_t1_reached_pct', np.nan))]
        post_sl_recoveries = [r.get('post_sl_recovery_from_worst') for r in sl_hits
                              if r.get('post_sl_recovery_from_worst') is not None and not np.isnan(r.get('post_sl_recovery_from_worst', np.nan))]
        summary['sl_hit_summary'] = {
            'count': len(sl_hits),
            'avg_pre_sl_t1_reached_pct': np.mean(pre_sl_t1_pcts) if pre_sl_t1_pcts else 0.0,
            'trades_bounced_to_t1_after_sl': sum(1 for r in sl_hits if r.get('post_sl_bounce_to_t1')),
            'avg_post_sl_recovery': np.mean(post_sl_recoveries) if post_sl_recoveries else 0.0,
        }

    # sl_post_t1 Summary - filter out NaN values
    sl_post_t1_hits = summary['by_analysis_type'].get('sl_post_t1', [])
    if sl_post_t1_hits:
        post_t1_t2_pcts = [r.get('post_t1_t2_reached_pct') for r in sl_post_t1_hits
                          if r.get('post_t1_t2_reached_pct') is not None and not np.isnan(r.get('post_t1_t2_reached_pct', np.nan))]
        post_exit_recoveries = [r.get('post_sl_post_t1_recovery') for r in sl_post_t1_hits
                                if r.get('post_sl_post_t1_recovery') is not None and not np.isnan(r.get('post_sl_post_t1_recovery', np.nan))]
        summary['sl_post_t1_summary'] = {
            'count': len(sl_post_t1_hits),
            'avg_post_t1_t2_reached_pct': np.mean(post_t1_t2_pcts) if post_t1_t2_pcts else 0.0,
            'trades_bounced_to_t2_after_exit': sum(1 for r in sl_post_t1_hits if r.get('post_sl_post_t1_bounce_to_t2')),
            'avg_post_exit_recovery': np.mean(post_exit_recoveries) if post_exit_recoveries else 0.0,
        }

    # T2 Hit Summary - filter out NaN values
    t2_hits = summary['by_analysis_type'].get('t2_hit', [])
    if t2_hits:
        post_t2_extensions = [r.get('post_t2_max_extension') for r in t2_hits
                             if r.get('post_t2_max_extension') is not None and not np.isnan(r.get('post_t2_max_extension', np.nan))]
        post_t2_pullbacks = [r.get('post_t2_max_pullback') for r in t2_hits
                            if r.get('post_t2_max_pullback') is not None and not np.isnan(r.get('post_t2_max_pullback', np.nan))]
        summary['t2_hit_summary'] = {
            'count': len(t2_hits),
            'avg_post_t2_extension': np.mean(post_t2_extensions) if post_t2_extensions else 0.0,
            'avg_post_t2_pullback': np.mean(post_t2_pullbacks) if post_t2_pullbacks else 0.0,
            'trades_reached_t3': sum(1 for r in t2_hits if r.get('post_t2_reached_t3')),
        }

    # Overall MFE/MAE - use nanmean to handle NaN values from division errors
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        mfe_pcts = [r.get('mfe_pct') for r in valid_results if r.get('mfe_pct') is not None and not np.isnan(r.get('mfe_pct', np.nan))]
        mae_pcts = [r.get('mae_pct') for r in valid_results if r.get('mae_pct') is not None and not np.isnan(r.get('mae_pct', np.nan))]
        mfe_in_rs = [r.get('mfe_in_r') for r in valid_results if r.get('mfe_in_r') is not None and not np.isnan(r.get('mfe_in_r', np.nan))]
        mae_in_rs = [r.get('mae_in_r') for r in valid_results if r.get('mae_in_r') is not None and not np.isnan(r.get('mae_in_r', np.nan))]
        summary['overall'] = {
            'avg_mfe_pct': np.mean(mfe_pcts) if mfe_pcts else 0.0,
            'avg_mae_pct': np.mean(mae_pcts) if mae_pcts else 0.0,
            'avg_mfe_in_r': np.mean(mfe_in_rs) if mfe_in_rs else 0.0,
            'avg_mae_in_r': np.mean(mae_in_rs) if mae_in_rs else 0.0,
        }

    # Convert defaultdicts to regular dicts for JSON
    summary['by_exit_reason'] = dict(summary['by_exit_reason'])
    summary['by_analysis_type'] = {k: len(v) for k, v in summary['by_analysis_type'].items()}

    return summary


def generate_actionable_insights(summary: Dict, results: List[Dict]) -> Dict:
    """Generate actionable insights and recommendations based on price action analysis.

    Args:
        summary: The summary statistics from generate_summary()
        results: List of individual trade analysis results

    Returns:
        Dict with insights, recommendations, and priority actions
    """
    insights = {
        'key_findings': [],
        'recommendations': [],
        'priority_actions': [],
        'metrics': {},
        'setup_breakdown': {},
        'bounce_analysis': {}
    }

    total_trades = summary.get('total_trades', 0)
    if total_trades == 0:
        return insights

    # Extract key metrics
    sl_hit_count = summary.get('by_analysis_type', {}).get('sl_hit', 0)
    sl_post_t1_count = summary.get('by_analysis_type', {}).get('sl_post_t1', 0)
    t2_hit_count = summary.get('by_analysis_type', {}).get('t2_hit', 0)
    other_count = summary.get('by_analysis_type', {}).get('other', 0)

    sl_hit_pct = (sl_hit_count / total_trades) * 100 if total_trades > 0 else 0
    t2_hit_pct = (t2_hit_count / total_trades) * 100 if total_trades > 0 else 0

    # SL Hit Analysis
    sl_summary = summary.get('sl_hit_summary', {})
    avg_progress_to_t1 = sl_summary.get('avg_pre_sl_t1_reached_pct', 0)
    bounced_to_t1 = sl_summary.get('trades_bounced_to_t1_after_sl', 0)
    bounce_rate = (bounced_to_t1 / sl_hit_count * 100) if sl_hit_count > 0 else 0

    # SL Post T1 Analysis
    sl_post_t1_summary = summary.get('sl_post_t1_summary', {})
    avg_progress_to_t2 = sl_post_t1_summary.get('avg_post_t1_t2_reached_pct', 0)
    bounced_to_t2 = sl_post_t1_summary.get('trades_bounced_to_t2_after_exit', 0)
    t2_bounce_rate = (bounced_to_t2 / sl_post_t1_count * 100) if sl_post_t1_count > 0 else 0

    # MFE/MAE Analysis
    overall = summary.get('overall', {})
    avg_mfe_r = overall.get('avg_mfe_in_r', 0)
    avg_mae_r = overall.get('avg_mae_in_r', 0)

    # Store metrics
    insights['metrics'] = {
        'sl_hit_rate': round(sl_hit_pct, 1),
        't2_hit_rate': round(t2_hit_pct, 1),
        'avg_progress_to_t1_before_sl': round(avg_progress_to_t1, 1),
        'bounce_to_t1_rate': round(bounce_rate, 1),
        'avg_progress_to_t2_before_trailing_sl': round(avg_progress_to_t2, 1),
        'bounce_to_t2_rate': round(t2_bounce_rate, 1),
        'avg_mfe_r': round(avg_mfe_r, 2),
        'avg_mae_r': round(avg_mae_r, 2),
        'mfe_mae_ratio': round(avg_mfe_r / avg_mae_r, 2) if avg_mae_r > 0 else 0
    }

    # === KEY FINDINGS ===

    # 1. SL Hit Rate Assessment
    if sl_hit_pct > 50:
        insights['key_findings'].append({
            'severity': 'high',
            'category': 'stop_loss',
            'finding': f"High SL hit rate: {sl_hit_pct:.1f}% of trades stopped out",
            'detail': f"{sl_hit_count} of {total_trades} trades hit hard stop loss"
        })
    elif sl_hit_pct > 40:
        insights['key_findings'].append({
            'severity': 'medium',
            'category': 'stop_loss',
            'finding': f"Moderate SL hit rate: {sl_hit_pct:.1f}% of trades stopped out",
            'detail': f"{sl_hit_count} of {total_trades} trades hit hard stop loss"
        })

    # 2. Early Stop-outs
    if avg_progress_to_t1 < 35 and sl_hit_count > 10:
        insights['key_findings'].append({
            'severity': 'high',
            'category': 'entry_timing',
            'finding': f"Early stop-outs: Trades only reach {avg_progress_to_t1:.1f}% toward T1 before SL",
            'detail': "Trades are being stopped before they have a chance to work"
        })

    # 3. Bounce Analysis - Stops too tight?
    if bounce_rate > 15 and sl_hit_count > 10:
        insights['key_findings'].append({
            'severity': 'medium',
            'category': 'stop_placement',
            'finding': f"{bounce_rate:.1f}% of stopped trades would have hit T1",
            'detail': f"{bounced_to_t1} trades bounced to T1 after being stopped out"
        })

    # 4. T2 Hit Rate
    if t2_hit_pct < 15:
        insights['key_findings'].append({
            'severity': 'medium',
            'category': 'target_achievement',
            'finding': f"Low T2 hit rate: Only {t2_hit_pct:.1f}% of trades reach T2",
            'detail': f"{t2_hit_count} of {total_trades} trades achieved full target"
        })

    # 5. Trailing Stop Analysis
    if avg_progress_to_t2 > 100 and sl_post_t1_count > 5:
        insights['key_findings'].append({
            'severity': 'medium',
            'category': 'trailing_stop',
            'finding': f"Trailing stops exit at {avg_progress_to_t2:.1f}% toward T2",
            'detail': f"Trades exceed T2 distance ({avg_progress_to_t2:.0f}%) before trailing stop hits"
        })

    if t2_bounce_rate > 25 and sl_post_t1_count > 5:
        insights['key_findings'].append({
            'severity': 'medium',
            'category': 'trailing_stop',
            'finding': f"{t2_bounce_rate:.1f}% of trailing-stopped trades would have hit T2",
            'detail': f"{bounced_to_t2} trades reached T2 after trailing stop exit"
        })

    # 6. MFE vs MAE
    if avg_mae_r > avg_mfe_r and avg_mae_r > 0:
        insights['key_findings'].append({
            'severity': 'high',
            'category': 'risk_reward',
            'finding': f"MAE exceeds MFE: Avg drawdown ({avg_mae_r:.2f}R) > Avg gain ({avg_mfe_r:.2f}R)",
            'detail': "Trades experience larger adverse moves than favorable moves on average"
        })

    # === RECOMMENDATIONS ===

    # Stop Loss Recommendations
    if avg_progress_to_t1 < 35 and sl_hit_pct > 40:
        insights['recommendations'].append({
            'category': 'stop_loss',
            'action': 'Consider widening initial stop loss',
            'rationale': f"Trades reach only {avg_progress_to_t1:.1f}% to T1 before SL. Wider stops may give trades room to work.",
            'config_hint': 'Increase sl_atr_mult in pipeline config (currently getting stopped too early)'
        })

    if bounce_rate > 20:
        insights['recommendations'].append({
            'category': 'stop_loss',
            'action': 'Review stop placement methodology',
            'rationale': f"{bounce_rate:.1f}% of stopped trades later hit T1. Consider swing-based stops instead of fixed ATR.",
            'config_hint': 'Use swing_sl_buffer_atr in structure configs for swing-based stop placement'
        })

    # Entry Timing Recommendations
    if avg_progress_to_t1 < 30 and sl_hit_pct > 40:
        insights['recommendations'].append({
            'category': 'entry_timing',
            'action': 'Improve entry confirmation',
            'rationale': "Trades are failing early, suggesting premature entries before confirmation.",
            'config_hint': 'Consider adding momentum/volume confirmation gates before entry'
        })

    # Trailing Stop Recommendations
    if avg_progress_to_t2 > 120 and t2_bounce_rate > 20:
        insights['recommendations'].append({
            'category': 'trailing_stop',
            'action': 'Loosen trailing stop after T1',
            'rationale': f"Trailing stops exit at {avg_progress_to_t2:.0f}% to T2 but {t2_bounce_rate:.1f}% reach T2 after.",
            'config_hint': 'Increase trailing stop distance or use time-based trailing'
        })

    # Target Recommendations
    if t2_hit_pct < 12 and sl_hit_pct < 50:
        insights['recommendations'].append({
            'category': 'targets',
            'action': 'Review T2 target placement',
            'rationale': f"Only {t2_hit_pct:.1f}% hit T2. Targets may be too ambitious.",
            'config_hint': 'Consider reducing target_mult_t2 or using ATR-based targets'
        })

    # === PRIORITY ACTIONS ===

    # Determine top priority based on severity
    high_severity = [f for f in insights['key_findings'] if f['severity'] == 'high']

    if any(f['category'] == 'stop_loss' for f in high_severity):
        insights['priority_actions'].append({
            'priority': 1,
            'action': 'Address high SL hit rate',
            'description': 'Focus on stop loss placement - either widen stops or improve entry timing',
            'expected_impact': 'Reduce unnecessary stop-outs, improve win rate'
        })

    if any(f['category'] == 'entry_timing' for f in high_severity):
        insights['priority_actions'].append({
            'priority': 2,
            'action': 'Improve entry timing/confirmation',
            'description': 'Add confirmation signals before entry to avoid premature entries',
            'expected_impact': 'Higher quality entries, better progress toward targets'
        })

    if any(f['category'] == 'risk_reward' for f in high_severity):
        insights['priority_actions'].append({
            'priority': 3,
            'action': 'Improve risk/reward profile',
            'description': 'MAE > MFE indicates systematic issue - review both entry and exit logic',
            'expected_impact': 'Better overall profitability'
        })

    # Add trailing stop action if relevant
    if any(f['category'] == 'trailing_stop' for f in insights['key_findings']):
        insights['priority_actions'].append({
            'priority': 4,
            'action': 'Optimize trailing stop mechanism',
            'description': 'Trailing stops may be too aggressive post-T1',
            'expected_impact': 'Capture more of the move after T1'
        })

    # === SETUP TYPE BREAKDOWN ===
    # Analyze SL hit rate by setup type
    setup_stats = defaultdict(lambda: {'total': 0, 'sl_hit': 0, 'sl_hit_pct': 0, 'avg_progress_to_t1': []})

    for r in results:
        if 'error' in r:
            continue
        setup_type = r.get('setup_type', 'unknown')
        setup_stats[setup_type]['total'] += 1

        if r.get('analysis_type') == 'sl_hit':
            setup_stats[setup_type]['sl_hit'] += 1
            progress = r.get('pre_sl_t1_reached_pct')
            if progress is not None and not np.isnan(progress):
                setup_stats[setup_type]['avg_progress_to_t1'].append(progress)

    # Calculate percentages and averages
    for setup_type, stats in setup_stats.items():
        if stats['total'] > 0:
            stats['sl_hit_pct'] = round((stats['sl_hit'] / stats['total']) * 100, 1)
        if stats['avg_progress_to_t1']:
            stats['avg_progress_to_t1'] = round(np.mean(stats['avg_progress_to_t1']), 1)
        else:
            stats['avg_progress_to_t1'] = 0

    # Sort by SL hit rate (worst first)
    sorted_setups = sorted(
        [(k, v) for k, v in setup_stats.items()],
        key=lambda x: x[1]['sl_hit_pct'],
        reverse=True
    )

    insights['setup_breakdown'] = {
        'by_sl_rate': [
            {
                'setup_type': setup,
                'total_trades': stats['total'],
                'sl_hit_count': stats['sl_hit'],
                'sl_hit_rate': stats['sl_hit_pct'],
                'avg_progress_to_t1': stats['avg_progress_to_t1']
            }
            for setup, stats in sorted_setups if stats['total'] >= 3  # Only include setups with 3+ trades
        ],
        'worst_performers': [
            setup for setup, stats in sorted_setups[:3]
            if stats['sl_hit_pct'] > 50 and stats['total'] >= 5
        ],
        'best_performers': [
            setup for setup, stats in sorted_setups[-3:]
            if stats['sl_hit_pct'] < 30 and stats['total'] >= 5
        ]
    }

    # === BOUNCE ANALYSIS ===
    # Analyze trades that bounced to T1 after SL hit
    bounced_trades = [r for r in results if r.get('post_sl_bounce_to_t1') == True]

    if bounced_trades:
        # Analyze characteristics of bounced trades
        bounce_setups = defaultdict(int)
        bounce_progress = []
        bounce_times = []

        for r in bounced_trades:
            bounce_setups[r.get('setup_type', 'unknown')] += 1
            progress = r.get('pre_sl_t1_reached_pct')
            if progress is not None and not np.isnan(progress):
                bounce_progress.append(progress)

        insights['bounce_analysis'] = {
            'total_bounced': len(bounced_trades),
            'bounce_rate': round((len(bounced_trades) / sl_hit_count * 100), 1) if sl_hit_count > 0 else 0,
            'avg_progress_before_sl': round(np.mean(bounce_progress), 1) if bounce_progress else 0,
            'by_setup': dict(bounce_setups),
            'most_common_setup': max(bounce_setups.items(), key=lambda x: x[1])[0] if bounce_setups else None,
            'interpretation': 'These trades would have been winners with wider stops',
            'sample_trades': [
                {
                    'trade_id': r.get('trade_id'),
                    'setup_type': r.get('setup_type'),
                    'progress_to_t1_before_sl': r.get('pre_sl_t1_reached_pct'),
                    'symbol': r.get('symbol'),
                    'date': r.get('date')
                }
                for r in bounced_trades[:5]  # Top 5 examples
            ]
        }
    else:
        insights['bounce_analysis'] = {
            'total_bounced': 0,
            'bounce_rate': 0,
            'interpretation': 'No trades bounced to T1 after SL hit'
        }

    return insights


def run_analysis(source: str) -> Optional[Dict]:
    """
    Run price action analysis on the given source.
    Can be called programmatically by other scripts.

    Args:
        source: Path to zip, extracted folder, or run prefix

    Returns:
        Dict with 'summary' and 'output_file' keys, or None on error
    """
    source_path = Path(source)
    trades = {}
    temp_dir = None

    try:
        # Determine source type
        if source_path.suffix == '.zip':
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(source_path, 'r') as zf:
                zf.extractall(temp_dir)
            extracted_items = list(Path(temp_dir).iterdir())
            if extracted_items:
                base_path = extracted_items[0] if extracted_items[0].is_dir() else Path(temp_dir)
            else:
                base_path = Path(temp_dir)
            trades = collect_trades_from_backtest(base_path)

        elif source_path.is_dir():
            trades = collect_trades_from_backtest(source_path)

        else:
            # Assume it's a run prefix
            trades = collect_trades_from_logs(source)

        if not trades:
            return None

        # Analyze each trade
        results = []
        for trade_id, trade in trades.items():
            result = analyze_trade(trade)
            if result:
                results.append(result)

        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            return None

        # Generate summary
        summary = generate_summary(results)

        # Generate actionable insights
        insights = generate_actionable_insights(summary, results)

        # Save results
        output = {
            'summary': summary,
            'actionable_insights': insights,
            'trades': results
        }

        output_dir = ROOT / 'analysis' / 'reports' / 'spike_test'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'price_action_analysis_{timestamp}.json'

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        return {
            'summary': summary,
            'actionable_insights': insights,
            'output_file': str(output_file),
            'trades_analyzed': len(valid_results),
            'total_trades': len(trades)
        }

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    source_path = Path(source)

    print("=" * 80)
    print("TRADE PRICE ACTION ANALYSIS")
    print("=" * 80)

    trades = {}
    temp_dir = None

    try:
        # Determine source type
        if source_path.suffix == '.zip':
            print(f"\nExtracting zip file: {source}")
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(source_path, 'r') as zf:
                zf.extractall(temp_dir)
            # Find the extracted folder
            extracted_items = list(Path(temp_dir).iterdir())
            if extracted_items:
                base_path = extracted_items[0] if extracted_items[0].is_dir() else Path(temp_dir)
            else:
                base_path = Path(temp_dir)
            trades = collect_trades_from_backtest(base_path)

        elif source_path.is_dir():
            print(f"\nLoading from extracted folder: {source}")
            trades = collect_trades_from_backtest(source_path)

        else:
            # Assume it's a run prefix
            print(f"\nLoading from logs with prefix: {source}")
            trades = collect_trades_from_logs(source)

        print(f"Found {len(trades)} trades")

        if not trades:
            print("No trades found!")
            return

        # Analyze each trade
        print("\nAnalyzing trades with 1-minute data...")
        results = []
        errors = defaultdict(int)

        for trade_id, trade in trades.items():
            result = analyze_trade(trade)
            if result:
                if 'error' in result:
                    errors[result['error']] += 1
                results.append(result)

        if errors:
            print(f"Analysis errors: {dict(errors)}")

        valid_results = [r for r in results if 'error' not in r]
        print(f"Successfully analyzed {len(valid_results)} trades")

        if not valid_results:
            print("No trades could be analyzed!")
            return

        # Generate summary
        summary = generate_summary(results)

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        print(f"\nTotal trades analyzed: {summary['total_trades']}")
        print(f"\nBy exit reason: {summary['by_exit_reason']}")
        print(f"By analysis type: {summary['by_analysis_type']}")

        if 'sl_hit_summary' in summary:
            print("\n--- SL HIT TRADES ---")
            s = summary['sl_hit_summary']
            print(f"  Count: {s['count']}")
            print(f"  Avg % of way to T1 before SL: {s['avg_pre_sl_t1_reached_pct']:.1f}%")
            print(f"  Trades that bounced back to T1 after SL: {s['trades_bounced_to_t1_after_sl']}")
            print(f"  Avg recovery from worst after SL: {s['avg_post_sl_recovery']:.2f}")

        if 'sl_post_t1_summary' in summary:
            print("\n--- SL POST T1 TRADES ---")
            s = summary['sl_post_t1_summary']
            print(f"  Count: {s['count']}")
            print(f"  Avg % of way to T2 before trailing SL hit: {s['avg_post_t1_t2_reached_pct']:.1f}%")
            print(f"  Trades that bounced to T2 after exit: {s['trades_bounced_to_t2_after_exit']}")
            print(f"  Avg recovery after exit: {s['avg_post_exit_recovery']:.2f}")

        if 't2_hit_summary' in summary:
            print("\n--- T2 HIT TRADES ---")
            s = summary['t2_hit_summary']
            print(f"  Count: {s['count']}")
            print(f"  Avg extension beyond T2: {s['avg_post_t2_extension']:.2f}")
            print(f"  Avg pullback after T2: {s['avg_post_t2_pullback']:.2f}")
            print(f"  Trades that reached T3: {s['trades_reached_t3']}")

        if 'overall' in summary:
            print("\n--- OVERALL MFE/MAE ---")
            s = summary['overall']
            print(f"  Avg MFE: {s['avg_mfe_pct']:.2f}% ({s['avg_mfe_in_r']:.2f}R)")
            print(f"  Avg MAE: {s['avg_mae_pct']:.2f}% ({s['avg_mae_in_r']:.2f}R)")

        # Generate actionable insights
        insights = generate_actionable_insights(summary, results)

        # Print actionable insights
        print("\n" + "=" * 80)
        print("ACTIONABLE INSIGHTS")
        print("=" * 80)

        if insights['key_findings']:
            print("\n--- KEY FINDINGS ---")
            for i, finding in enumerate(insights['key_findings'], 1):
                severity_icon = "!!" if finding['severity'] == 'high' else "!"
                print(f"  {i}. [{severity_icon}] {finding['finding']}")
                print(f"      -> {finding['detail']}")

        if insights['recommendations']:
            print("\n--- RECOMMENDATIONS ---")
            for i, rec in enumerate(insights['recommendations'], 1):
                print(f"  {i}. [{rec['category'].upper()}] {rec['action']}")
                print(f"      Rationale: {rec['rationale']}")
                print(f"      Config: {rec['config_hint']}")

        if insights['priority_actions']:
            print("\n--- PRIORITY ACTIONS ---")
            for action in sorted(insights['priority_actions'], key=lambda x: x['priority']):
                print(f"  P{action['priority']}: {action['action']}")
                print(f"       {action['description']}")
                print(f"       Expected: {action['expected_impact']}")

        if insights['metrics']:
            print("\n--- METRICS SUMMARY ---")
            m = insights['metrics']
            print(f"  SL Hit Rate: {m['sl_hit_rate']}%")
            print(f"  T2 Hit Rate: {m['t2_hit_rate']}%")
            print(f"  Avg Progress to T1 before SL: {m['avg_progress_to_t1_before_sl']}%")
            print(f"  Bounce to T1 Rate: {m['bounce_to_t1_rate']}%")
            print(f"  MFE/MAE Ratio: {m['mfe_mae_ratio']}")

        # Setup breakdown
        if insights.get('setup_breakdown', {}).get('by_sl_rate'):
            print("\n--- SETUP TYPE BREAKDOWN (by SL Rate) ---")
            for s in insights['setup_breakdown']['by_sl_rate'][:10]:  # Top 10
                print(f"  {s['setup_type']}: {s['sl_hit_rate']}% SL ({s['sl_hit_count']}/{s['total_trades']} trades), avg {s['avg_progress_to_t1']}% to T1")

            if insights['setup_breakdown'].get('worst_performers'):
                print(f"\n  WORST PERFORMERS: {', '.join(insights['setup_breakdown']['worst_performers'])}")
            if insights['setup_breakdown'].get('best_performers'):
                print(f"  BEST PERFORMERS: {', '.join(insights['setup_breakdown']['best_performers'])}")

        # Bounce analysis
        if insights.get('bounce_analysis', {}).get('total_bounced', 0) > 0:
            ba = insights['bounce_analysis']
            print("\n--- BOUNCE TO T1 ANALYSIS ---")
            print(f"  Trades that bounced to T1 after SL: {ba['total_bounced']} ({ba['bounce_rate']}% of SL trades)")
            print(f"  Avg progress before SL: {ba['avg_progress_before_sl']}%")
            print(f"  Most common setup: {ba['most_common_setup']}")
            print(f"  Interpretation: {ba['interpretation']}")
            if ba.get('sample_trades'):
                print("\n  Sample bounced trades:")
                for t in ba['sample_trades']:
                    print(f"    - {t['symbol']} ({t['setup_type']}): {t['progress_to_t1_before_sl']:.1f}% to T1 before SL")

        # Save results
        output = {
            'summary': summary,
            'actionable_insights': insights,
            'trades': results
        }

        # Create output directory if it doesn't exist
        output_dir = ROOT / 'analysis' / 'reports' / 'spike_test'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'price_action_analysis_{timestamp}.json'

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n\nResults saved to: {output_file}")

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
