"""
Comprehensive Trade Analysis - Data-Driven Optimization

Uses actual logged data:
- analytics.jsonl: Trade executions (ENTRY → EXIT)
- events.jsonl: Full decision context (plans, targets, stops)
- 1m OHLCV cache: Actual price movements

Analysis performed:
1. SL Optimization: Were stops too tight? Did price recover?
2. Target Optimization: Were T1/T2 achievable? Near misses?
3. Entry Timing: Filled vs not filled, slippage analysis
4. Exit Quality: Did we exit too early? Leave money on table?
5. Rejected Decisions: What profitable trades did we miss?
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_1m_data(symbol, date_str, hours_after=6):
    """Load 1m OHLCV data for symbol"""
    symbol_clean = symbol.replace('NSE:', '')
    cache_base = ROOT / 'cache' / 'ohlcv_archive'

    for suffix in ['.NS', '']:
        symbol_dir = cache_base / f'{symbol_clean}{suffix}'
        if symbol_dir.exists():
            feather_files = list(symbol_dir.glob('*_1minutes*.feather'))
            if feather_files:
                df = pd.read_feather(feather_files[0])

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                elif 'ts' in df.columns:
                    df['date'] = pd.to_datetime(df['ts']).dt.tz_localize(None)
                else:
                    continue

                trade_date = pd.to_datetime(date_str).tz_localize(None)
                end_date = trade_date + timedelta(hours=hours_after)

                df_filtered = df[(df['date'] >= trade_date) & (df['date'] <= end_date)]
                return df_filtered.sort_values('date').reset_index(drop=True)

    return None


def analyze_single_trade(trade_exit, trade_decision, ohlcv_1m):
    """
    Deep analysis of a single trade using:
    - trade_exit: analytics.jsonl EXIT event
    - trade_decision: events.jsonl DECISION event with full plan
    - ohlcv_1m: 1-minute price data
    """

    if ohlcv_1m is None or ohlcv_1m.empty:
        return None

    result = {
        'symbol': trade_exit['symbol'],
        'trade_id': trade_exit['trade_id'],
        'setup_type': trade_exit['setup_type'],
        'regime': trade_exit['regime'],
        'direction': trade_exit['bias'],
        'actual_pnl': trade_exit['pnl'],
        'exit_reason': trade_exit['reason'],
        'entry_price': trade_exit['actual_entry_price'],
        'exit_price': trade_exit['exit_price'],
    }

    # Get plan details from decision event
    plan = trade_decision.get('plan', {})
    hard_sl = plan.get('stop', {}).get('hard')
    targets = plan.get('targets', [])
    t1_level = targets[0]['level'] if len(targets) > 0 else None
    t2_level = targets[1]['level'] if len(targets) > 1 else None
    entry_zone = plan.get('entry', {}).get('zone', [])

    result['hard_sl'] = hard_sl
    result['t1_level'] = t1_level
    result['t2_level'] = t2_level
    result['entry_zone'] = entry_zone

    # Get entry/exit timestamps
    # Find ENTRY timestamp from analytics or use decision_ts
    decision_ts = pd.to_datetime(trade_decision['ts'])
    exit_ts = pd.to_datetime(trade_exit['timestamp'])

    # Filter 1m data to trade window
    trade_bars = ohlcv_1m[
        (ohlcv_1m['date'] >= decision_ts) &
        (ohlcv_1m['date'] <= exit_ts)
    ].copy()

    if trade_bars.empty:
        return result

    result['bars_in_trade'] = len(trade_bars)

    entry_price = trade_exit['actual_entry_price']
    direction = trade_exit['bias']

    # Calculate MFE/MAE
    if direction == 'long':
        mfe = (trade_bars['high'].max() - entry_price) / entry_price * 100
        mae = (trade_bars['low'].min() - entry_price) / entry_price * 100

        # Check what happened
        if hard_sl:
            sl_hit_bars = trade_bars[trade_bars['low'] <= hard_sl]
            result['sl_was_touched'] = not sl_hit_bars.empty

        if t1_level:
            t1_hit_bars = trade_bars[trade_bars['high'] >= t1_level]
            result['t1_was_hit'] = not t1_hit_bars.empty
            if result['t1_was_hit']:
                t1_idx = trade_bars[trade_bars['high'] >= t1_level].index[0]
                result['t1_hit_bar'] = trade_bars.index.get_loc(t1_idx) + 1

        if t2_level:
            t2_hit_bars = trade_bars[trade_bars['high'] >= t2_level]
            result['t2_was_hit'] = not t2_hit_bars.empty
            if result['t2_was_hit']:
                t2_idx = trade_bars[trade_bars['high'] >= t2_level].index[0]
                result['t2_hit_bar'] = trade_bars.index.get_loc(t2_idx) + 1

    else:  # short
        mfe = (entry_price - trade_bars['low'].min()) / entry_price * 100
        mae = (trade_bars['high'].max() - entry_price) / entry_price * 100

        if hard_sl:
            sl_hit_bars = trade_bars[trade_bars['high'] >= hard_sl]
            result['sl_was_touched'] = not sl_hit_bars.empty

        if t1_level:
            t1_hit_bars = trade_bars[trade_bars['low'] <= t1_level]
            result['t1_was_hit'] = not t1_hit_bars.empty
            if result['t1_was_hit']:
                t1_idx = trade_bars[trade_bars['low'] <= t1_level].index[0]
                result['t1_hit_bar'] = trade_bars.index.get_loc(t1_idx) + 1

        if t2_level:
            t2_hit_bars = trade_bars[trade_bars['low'] <= t2_level]
            result['t2_was_hit'] = not t2_hit_bars.empty
            if result['t2_was_hit']:
                t2_idx = trade_bars[trade_bars['low'] <= t2_level].index[0]
                result['t2_hit_bar'] = trade_bars.index.get_loc(t2_idx) + 1

    result['mfe_pct'] = mfe
    result['mae_pct'] = mae

    # SL OPTIMIZATION ANALYSIS
    if trade_exit['reason'] == 'hard_sl':
        # Trade hit SL - analyze if it was too tight
        result['was_sl_exit'] = True

        # Check if price recovered after SL
        if direction == 'long' and hard_sl:
            # Did price go back above entry after SL?
            if not sl_hit_bars.empty:
                sl_hit_idx = sl_hit_bars.index[0]
                after_sl = trade_bars[trade_bars.index > sl_hit_idx]
                if not after_sl.empty:
                    max_recovery = after_sl['high'].max()
                    if max_recovery > entry_price:
                        result['sl_too_tight'] = True
                        result['recovery_pct'] = (max_recovery - entry_price) / entry_price * 100
                        result['missed_pnl'] = (max_recovery - entry_price) / (entry_price - hard_sl) * 500  # Rs per R

        elif direction == 'short' and hard_sl:
            if not sl_hit_bars.empty:
                sl_hit_idx = sl_hit_bars.index[0]
                after_sl = trade_bars[trade_bars.index > sl_hit_idx]
                if not after_sl.empty:
                    min_recovery = after_sl['low'].min()
                    if min_recovery < entry_price:
                        result['sl_too_tight'] = True
                        result['recovery_pct'] = (entry_price - min_recovery) / entry_price * 100
                        result['missed_pnl'] = (entry_price - min_recovery) / (hard_sl - entry_price) * 500

    # TARGET ANALYSIS
    if trade_exit['reason'] not in ['target_t1', 'target_t2']:
        # Didn't hit target - analyze near misses
        if t1_level and not result.get('t1_was_hit'):
            if direction == 'long':
                closest = trade_bars['high'].max()
                distance_pct = (t1_level - closest) / entry_price * 100
            else:
                closest = trade_bars['low'].min()
                distance_pct = (closest - t1_level) / entry_price * 100

            result['t1_miss_distance_pct'] = distance_pct
            if abs(distance_pct) < 0.3:
                result['t1_near_miss'] = True

        if t2_level and not result.get('t2_was_hit'):
            if direction == 'long':
                closest = trade_bars['high'].max()
                distance_pct = (t2_level - closest) / entry_price * 100
            else:
                closest = trade_bars['low'].min()
                distance_pct = (closest - t2_level) / entry_price * 100

            result['t2_miss_distance_pct'] = distance_pct
            if abs(distance_pct) < 0.5:
                result['t2_near_miss'] = True

    # EXIT QUALITY
    # Did we leave money on table?
    exit_price = trade_exit['exit_price']
    if direction == 'long':
        max_after_exit = trade_bars[trade_bars.index > len(trade_bars)//2]['high'].max() if len(trade_bars) > 10 else exit_price
        if max_after_exit > exit_price:
            result['left_on_table_pct'] = (max_after_exit - exit_price) / entry_price * 100
    else:
        min_after_exit = trade_bars[trade_bars.index > len(trade_bars)//2]['low'].min() if len(trade_bars) > 10 else exit_price
        if min_after_exit < exit_price:
            result['left_on_table_pct'] = (exit_price - min_after_exit) / entry_price * 100

    return result


def analyze_session(session_dir):
    """Analyze all trades in a session"""

    analytics_file = session_dir / 'analytics.jsonl'
    events_file = session_dir / 'events.jsonl'

    if not analytics_file.exists() or not events_file.exists():
        return []

    # Load all EXIT events from analytics
    exits = []
    with open(analytics_file) as f:
        for line in f:
            event = json.loads(line)
            if event.get('stage') == 'EXIT':
                exits.append(event)

    # Load all DECISION events
    decisions = {}
    with open(events_file) as f:
        for line in f:
            event = json.loads(line)
            if event.get('type') == 'DECISION':
                trade_id = event.get('trade_id')
                decisions[trade_id] = event

    # Analyze each trade
    results = []
    for exit_event in exits:
        trade_id = exit_event['trade_id']

        if trade_id not in decisions:
            continue

        decision_event = decisions[trade_id]
        symbol = exit_event['symbol']
        date_str = exit_event['timestamp'][:10]

        ohlcv_1m = load_1m_data(symbol, date_str)
        analysis = analyze_single_trade(exit_event, decision_event, ohlcv_1m)

        if analysis:
            results.append(analysis)

    return results


def print_summary(all_results, run_prefix):
    """Print comprehensive summary"""

    print(f'\n{"="*100}')
    print(f'COMPREHENSIVE TRADE ANALYSIS - {run_prefix}')
    print(f'{"="*100}\n')

    print(f'Total trades analyzed: {len(all_results)}')

    # 1. SL ANALYSIS
    print(f'\n{"="*100}')
    print('1. STOP LOSS OPTIMIZATION ANALYSIS')
    print(f'{"="*100}')

    sl_exits = [r for r in all_results if r.get('was_sl_exit')]
    sl_too_tight = [r for r in sl_exits if r.get('sl_too_tight')]

    print(f'\nHard SL exits: {len(sl_exits)}/{len(all_results)} ({len(sl_exits)/len(all_results)*100:.1f}%)')
    print(f'SL too tight (price recovered): {len(sl_too_tight)}/{len(sl_exits)} ({len(sl_too_tight)/len(sl_exits)*100:.1f}% if sl_exits else 0)')

    if sl_too_tight:
        total_missed = sum(r.get('missed_pnl', 0) for r in sl_too_tight)
        print(f'\nEstimated PnL lost to tight stops: Rs.{total_missed:.2f}')
        print(f'\nTop 5 SL optimization opportunities:')
        for r in sorted(sl_too_tight, key=lambda x: x.get('missed_pnl', 0), reverse=True)[:5]:
            print(f'  {r["symbol"]}: SL hit, then recovered +{r.get("recovery_pct", 0):.2f}%, missed Rs.{r.get("missed_pnl", 0):.0f}')

    # MFE/MAE analysis
    avg_mfe = np.mean([r['mfe_pct'] for r in all_results])
    avg_mae = np.mean([r['mae_pct'] for r in all_results])

    print(f'\nExcursion Analysis:')
    print(f'  Average MFE (max favorable): {avg_mfe:.2f}%')
    print(f'  Average MAE (max adverse): {avg_mae:.2f}%')
    print(f'  MFE/MAE ratio: {abs(avg_mfe/avg_mae):.2f}')

    # 2. TARGET ANALYSIS
    print(f'\n{"="*100}')
    print('2. TARGET OPTIMIZATION ANALYSIS')
    print(f'{"="*100}')

    t1_hit = [r for r in all_results if r.get('t1_was_hit')]
    t2_hit = [r for r in all_results if r.get('t2_was_hit')]
    t1_near_miss = [r for r in all_results if r.get('t1_near_miss')]
    t2_near_miss = [r for r in all_results if r.get('t2_near_miss')]

    print(f'\nT1 hit: {len(t1_hit)}/{len(all_results)} ({len(t1_hit)/len(all_results)*100:.1f}%)')
    print(f'T2 hit: {len(t2_hit)}/{len(all_results)} ({len(t2_hit)/len(all_results)*100:.1f}%)')
    print(f'T1 near-miss (<0.3%): {len(t1_near_miss)}')
    print(f'T2 near-miss (<0.5%): {len(t2_near_miss)}')

    if t1_near_miss:
        print(f'\nT1 Near Misses (targets too aggressive):')
        for r in t1_near_miss[:5]:
            print(f'  {r["symbol"]}: Missed T1 by {abs(r.get("t1_miss_distance_pct", 0)):.2f}%')

    if t2_near_miss:
        print(f'\nT2 Near Misses:')
        for r in t2_near_miss[:5]:
            print(f'  {r["symbol"]}: Missed T2 by {abs(r.get("t2_miss_distance_pct", 0)):.2f}%')

    # 3. EXIT QUALITY
    print(f'\n{"="*100}')
    print('3. EXIT QUALITY ANALYSIS')
    print(f'{"="*100}')

    left_on_table = [r for r in all_results if r.get('left_on_table_pct', 0) > 0.5]

    if left_on_table:
        total_left = sum(r.get('left_on_table_pct', 0) for r in left_on_table)
        print(f'\nTrades that left money on table: {len(left_on_table)}/{len(all_results)}')
        print(f'Total % left: {total_left:.2f}%')
        print(f'\nTop 5 early exits:')
        for r in sorted(left_on_table, key=lambda x: x.get('left_on_table_pct', 0), reverse=True)[:5]:
            print(f'  {r["symbol"]} ({r["exit_reason"]}): Left {r.get("left_on_table_pct", 0):.2f}% more available')

    # 4. SETUP BREAKDOWN
    print(f'\n{"="*100}')
    print('4. SETUP-WISE ANALYSIS')
    print(f'{"="*100}')

    by_setup = defaultdict(lambda: {'count': 0, 'sl_tight': 0, 't1_near': 0, 't2_near': 0, 'total_pnl': 0})

    for r in all_results:
        setup = r['setup_type']
        by_setup[setup]['count'] += 1
        by_setup[setup]['total_pnl'] += r['actual_pnl']
        if r.get('sl_too_tight'):
            by_setup[setup]['sl_tight'] += 1
        if r.get('t1_near_miss'):
            by_setup[setup]['t1_near'] += 1
        if r.get('t2_near_miss'):
            by_setup[setup]['t2_near'] += 1

    print(f'\n{"Setup":<25} {"Trades":<8} {"SL Tight":<10} {"T1 Near":<10} {"T2 Near":<10} {"Avg PnL":<10}')
    print('-' * 100)
    for setup, stats in sorted(by_setup.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
        avg_pnl = stats['total_pnl'] / stats['count']
        print(f'{setup:<25} {stats["count"]:<8} {stats["sl_tight"]:<10} {stats["t1_near"]:<10} {stats["t2_near"]:<10} Rs.{avg_pnl:<8.0f}')

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive trade analysis')
    parser.add_argument('run_prefix', help='Run prefix (e.g., run_953e4bdc)')
    parser.add_argument('--save', action='store_true', help='Save detailed results to JSON')
    args = parser.parse_args()

    logs_dir = ROOT / 'logs'
    sessions = sorted(logs_dir.glob(f'{args.run_prefix}_*'))

    print(f'\nAnalyzing {len(sessions)} sessions for {args.run_prefix}...')

    all_results = []
    for session_dir in sessions:
        results = analyze_session(session_dir)
        all_results.extend(results)

    if not all_results:
        print('No trades found to analyze')
        return

    # Print summary
    all_results = print_summary(all_results, args.run_prefix)

    # Save if requested
    if args.save:
        output_file = ROOT / f'trade_analysis_{args.run_prefix}.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nDetailed results saved to: {output_file}')


if __name__ == '__main__':
    main()
