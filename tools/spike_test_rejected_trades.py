#!/usr/bin/env python
"""
Spike Test: Rejected Trades Analysis

Question: Did we miss good trades by rejecting them?
Method: Simulate what would have happened if we took the rejected/skipped trades
Data: Use 1m OHLC data to simulate trade execution with same SL/target logic
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
logs_dir = ROOT / 'logs'
cache_dir = ROOT / 'cache'

print('=' * 80)
print('SPIKE TEST: REJECTED TRADES ANALYSIS')
print('=' * 80)
print()

# Step 1: Get all decisions (both triggered and skipped)
all_decisions = []
triggered_trade_ids = set()

for session_dir in sorted(logs_dir.glob('bt_*_20251103_100842')):
    events_file = session_dir / 'events.jsonl'
    planning_file = session_dir / 'planning.jsonl'

    if not events_file.exists():
        continue

    date = session_dir.name.split('_')[1]

    # First, get timestamps from planning.jsonl
    planning_timestamps = {}
    if planning_file.exists():
        with open(planning_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    symbol = data.get('symbol')
                    timestamp = data.get('timestamp')
                    if symbol and timestamp:
                        planning_timestamps[symbol] = timestamp
                except:
                    pass

    # Then get decisions
    with open(events_file, 'r') as f:
        for line in f:
            try:
                event = json.loads(line)
                event_type = event.get('type')

                if event_type == 'DECISION':
                    trade_id = event.get('trade_id', '')
                    symbol = event.get('symbol', '')
                    features = event.get('features', {})
                    decision = event.get('decision', {})

                    if isinstance(decision, dict):
                        # Use planning timestamp if available
                        timestamp = planning_timestamps.get(symbol, '')

                        if timestamp:  # Only add if we have a timestamp
                            all_decisions.append({
                                'trade_id': trade_id,
                                'symbol': symbol,
                                'date': date,
                                'timestamp': timestamp,
                                'setup_type': decision.get('setup_type', ''),
                                'regime': decision.get('regime', ''),
                                'rank_score': features.get('rank_score', 0),
                            })

                elif event_type == 'TRIGGER':
                    # Mark this as triggered
                    trade_id = event.get('trade_id', '')
                    if trade_id:
                        triggered_trade_ids.add(trade_id)

            except:
                pass

print(f'Total decisions: {len(all_decisions)}')
print(f'Triggered trades: {len(triggered_trade_ids)}')
print()

# Step 2: Identify skipped trades (had decision but no trigger)
skipped_trades = [d for d in all_decisions if d['trade_id'] and d['trade_id'] not in triggered_trade_ids]

print(f'Skipped trades: {len(skipped_trades)}')
print()

if len(skipped_trades) == 0:
    print('No skipped trades found. Exiting.')
    exit(0)

# Step 3: Sample and simulate skipped trades
print('Simulating skipped trades (sample of 100)...')
print()

simulated_results = []
data_unavailable = 0

for trade in skipped_trades:  # Analyze ALL skipped trades
    symbol = trade['symbol'].replace('NSE:', '')
    date = trade['date']
    entry_ts = trade['timestamp']

    # Find 1m feather file
    feather_file = cache_dir / 'ohlcv_archive' / f'{symbol}.NS' / f'{symbol}.NS_1minutes.feather'

    if not feather_file.exists():
        data_unavailable += 1
        continue

    try:
        # Load 1m data
        df_1m = pd.read_feather(feather_file)
        if 'date' in df_1m.columns:
            df_1m = df_1m.rename(columns={'date': 'timestamp'})
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])

        # Filter to date
        date_obj = pd.to_datetime(date).date()
        date_mask = df_1m['timestamp'].dt.date == date_obj
        df_day = df_1m[date_mask].copy()

        if len(df_day) == 0:
            data_unavailable += 1
            continue

        # Find entry bar
        entry_time = pd.to_datetime(entry_ts)
        if df_day['timestamp'].dt.tz is not None:
            entry_time = entry_time.tz_localize(df_day['timestamp'].dt.tz)

        # Get entry bar or next bar after entry_time
        entry_bar_idx = df_day[df_day['timestamp'] >= entry_time].index
        if len(entry_bar_idx) == 0:
            data_unavailable += 1
            continue

        entry_bar = df_day.loc[entry_bar_idx[0]]
        entry_price = entry_bar['close']

        # Calculate theoretical SL and targets
        # Assume 1.5% ATR (typical for Indian stocks)
        atr_approx = entry_price * 0.015

        # For long trades
        if 'long' in trade['setup_type']:
            hard_sl = entry_price - (atr_approx * 1.5)
            t1_target = entry_price + (atr_approx * 1.5)  # 1R
            t2_target = entry_price + (atr_approx * 3.0)  # 2R

            # Simulate execution
            after_entry = df_day[df_day['timestamp'] > entry_time]
            hit_sl = False
            hit_t1 = False
            hit_t2 = False
            exit_price = entry_price
            exit_reason = 'eod'

            for idx, bar in after_entry.iterrows():
                # Check SL first
                if bar['low'] <= hard_sl:
                    hit_sl = True
                    exit_price = hard_sl
                    exit_reason = 'hard_sl'
                    break

                # Check T2
                if bar['high'] >= t2_target:
                    hit_t2 = True
                    exit_price = t2_target
                    exit_reason = 'target_t2'
                    break

                # Check T1
                if bar['high'] >= t1_target:
                    hit_t1 = True
                    exit_price = t1_target
                    exit_reason = 't1_partial'
                    break

            # If nothing hit, exit at close
            if not (hit_sl or hit_t1 or hit_t2) and len(after_entry) > 0:
                exit_price = after_entry.iloc[-1]['close']
                exit_reason = 'eod'

        else:  # Short trades
            hard_sl = entry_price + (atr_approx * 1.5)
            t1_target = entry_price - (atr_approx * 1.5)
            t2_target = entry_price - (atr_approx * 3.0)

            after_entry = df_day[df_day['timestamp'] > entry_time]
            hit_sl = False
            hit_t1 = False
            hit_t2 = False
            exit_price = entry_price
            exit_reason = 'eod'

            for idx, bar in after_entry.iterrows():
                if bar['high'] >= hard_sl:
                    hit_sl = True
                    exit_price = hard_sl
                    exit_reason = 'hard_sl'
                    break

                if bar['low'] <= t2_target:
                    hit_t2 = True
                    exit_price = t2_target
                    exit_reason = 'target_t2'
                    break

                if bar['low'] <= t1_target:
                    hit_t1 = True
                    exit_price = t1_target
                    exit_reason = 't1_partial'
                    break

            if not (hit_sl or hit_t1 or hit_t2) and len(after_entry) > 0:
                exit_price = after_entry.iloc[-1]['close']
                exit_reason = 'eod'

        # Calculate P&L
        position_size = 10000 / entry_price
        pnl = position_size * (exit_price - entry_price) if 'long' in trade['setup_type'] else position_size * (entry_price - exit_price)
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if 'long' in trade['setup_type'] else ((entry_price - exit_price) / entry_price * 100)

        simulated_results.append({
            'symbol': trade['symbol'],
            'date': date,
            'setup_type': trade['setup_type'],
            'regime': trade['regime'],
            'rank_score': trade['rank_score'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hit_sl': hit_sl,
            'hit_t1': hit_t1,
            'hit_t2': hit_t2,
        })

    except Exception as e:
        data_unavailable += 1
        continue

print(f'Simulated: {len(simulated_results)} trades')
print(f'Data unavailable: {data_unavailable}')
print()

if len(simulated_results) == 0:
    print('No simulation results. Exiting.')
    exit(0)

# Step 4: Analyze results
print('=' * 80)
print('SIMULATION RESULTS: SKIPPED TRADES')
print('=' * 80)
print()

winners = [t for t in simulated_results if t['pnl'] > 0]
losers = [t for t in simulated_results if t['pnl'] <= 0]
sl_hits = [t for t in simulated_results if t['hit_sl']]

total_pnl = sum(t['pnl'] for t in simulated_results)
avg_pnl = total_pnl / len(simulated_results)
win_rate = len(winners) / len(simulated_results) * 100

print(f'Win Rate: {len(winners)}/{len(simulated_results)} ({win_rate:.1f}%)')
print(f'Total P&L: Rs.{total_pnl:,.0f}')
print(f'Avg P&L per trade: Rs.{avg_pnl:.0f}')
print(f'SL hit rate: {len(sl_hits)}/{len(simulated_results)} ({len(sl_hits)/len(simulated_results)*100:.1f}%)')
print()

# Compare to actual triggered trades
print('=' * 80)
print('COMPARISON: SKIPPED vs TRIGGERED TRADES')
print('=' * 80)
print()

print('SKIPPED (simulated):')
print(f'  Win rate: {win_rate:.1f}%')
print(f'  Avg P&L: Rs.{avg_pnl:.0f}')
print(f'  SL hit rate: {len(sl_hits)/len(simulated_results)*100:.1f}%')
print()

print('TRIGGERED (actual from backtest):')
print(f'  Win rate: 54.3%')
print(f'  Avg P&L: Rs.231')
print(f'  SL hit rate: 38.3% (31/81)')
print()

if win_rate > 54.3:
    print('FINDING: Skipped trades have BETTER win rate!')
    print('  -> We may be filtering out good trades')
    print('  -> Consider LOOSENING filters or INCREASING position capacity')
elif win_rate < 50:
    print('FINDING: Skipped trades have WORSE win rate!')
    print('  -> Filters are working correctly')
    print('  -> We are correctly rejecting low-quality setups')
else:
    print('FINDING: Skipped trades have SIMILAR win rate')
    print('  -> Capital/position limits are the main constraint')
    print('  -> Rank scoring is working as intended')
print()

# Best missed opportunities
print('=' * 80)
print('TOP 10 MISSED OPPORTUNITIES (Best Skipped Winners)')
print('=' * 80)
print()

best_missed = sorted(winners, key=lambda x: x['pnl'], reverse=True)[:10]

print(f'{"Symbol":20s} {"Date":12s} {"Setup":20s} {"Rank":6s} {"PnL":10s} {"Exit"}')
print('-' * 90)
for t in best_missed:
    print(f'{t["symbol"]:20s} {t["date"]:12s} {t["setup_type"]:20s} {t["rank_score"]:6.2f} Rs.{t["pnl"]:7,.0f} {t["exit_reason"]}')
print()

# Worst skips (correctly rejected)
print('=' * 80)
print('TOP 10 GOOD REJECTIONS (Worst Skipped Losers)')
print('=' * 80)
print()

worst_skipped = sorted(losers, key=lambda x: x['pnl'])[:10]

print(f'{"Symbol":20s} {"Date":12s} {"Setup":20s} {"Rank":6s} {"PnL":10s} {"Exit"}')
print('-' * 90)
for t in worst_skipped:
    print(f'{t["symbol"]:20s} {t["date"]:12s} {t["setup_type"]:20s} {t["rank_score"]:6.2f} Rs.{t["pnl"]:7,.0f} {t["exit_reason"]}')
print()

print('=' * 80)
print('END OF ANALYSIS')
print('=' * 80)
