"""
FHM Long Trade-by-Trade Comparison: Current vs Proposed Parameters

Shows which specific trades would be rescued or harmed by parameter changes.
"""

import json
import os
import pandas as pd
from collections import defaultdict

BACKTEST_DIR = 'backtest_20251210-123728_extracted'
OHLCV_DIR = 'cache/ohlcv_archive'


def load_fhm_trades():
    all_trades = []
    for date_folder in sorted(os.listdir(BACKTEST_DIR)):
        events_file = f'{BACKTEST_DIR}/{date_folder}/events.jsonl'
        if not os.path.exists(events_file):
            continue
        decisions = {}
        triggers = {}
        exits = {}
        with open(events_file) as f:
            for line in f:
                event = json.loads(line)
                trade_id = event.get('trade_id')
                evt_type = event.get('type')
                if evt_type == 'DECISION':
                    decisions[trade_id] = event
                elif evt_type == 'TRIGGER':
                    triggers[trade_id] = event
                elif evt_type == 'EXIT':
                    exits[trade_id] = event
        for trade_id, decision_event in decisions.items():
            decision = decision_event.get('decision', {})
            plan = decision_event.get('plan', {})
            setup_type = decision.get('setup_type', '')
            if 'first_hour_momentum_long' in setup_type.lower():
                if trade_id in triggers:
                    trigger_event = triggers[trade_id]
                    exit_event = exits.get(trade_id, {})
                    exit_data = exit_event.get('exit', {}) if isinstance(exit_event, dict) else {}
                    targets = plan.get('targets', [])
                    t1_level = None
                    for t in targets:
                        if t.get('name') == 'T1':
                            t1_level = t.get('level')
                    all_trades.append({
                        'date': date_folder,
                        'trade_id': trade_id,
                        'symbol': decision_event.get('symbol'),
                        'entry_ts': trigger_event.get('ts'),
                        'entry_price': trigger_event.get('trigger', {}).get('actual_price'),
                        'sl': plan.get('stop', {}).get('price'),
                        't1': t1_level,
                        'qty': plan.get('sizing', {}).get('qty'),
                        'exit_reason': exit_data.get('reason', ''),
                        'actual_pnl': exit_data.get('pnl', 0)
                    })
    return all_trades


def load_1m_data(symbol):
    if ':' in symbol:
        ticker = symbol.split(':')[1] + '.NS'
    else:
        ticker = symbol + '.NS'
    ohlcv_path = f'{OHLCV_DIR}/{ticker}/{ticker}_1minutes.feather'
    if not os.path.exists(ohlcv_path):
        return None
    df = pd.read_feather(ohlcv_path)
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    df = df.set_index('datetime').sort_index()
    return df


def simulate_both(trade, ohlcv_df):
    entry_price = trade['entry_price']
    entry_ts = trade['entry_ts']
    qty = trade['qty'] or 1
    if not entry_price or not entry_ts:
        return None
    entry_dt = pd.to_datetime(entry_ts)
    if entry_dt.tzinfo is not None:
        entry_dt = entry_dt.tz_localize(None)

    # Current params
    sl_current = entry_price * (1 - 0.008)  # 0.8%
    t1_current = entry_price + (entry_price - sl_current) * 1.0  # 1.0R

    # New params
    sl_new = entry_price * (1 - 0.012)  # 1.2%
    t1_new = entry_price + (entry_price - sl_new) * 1.5  # 1.5R

    try:
        bars = ohlcv_df.loc[entry_dt:]
    except:
        return None
    if len(bars) == 0:
        return None

    result_current = {'exit_reason': None, 'pnl': 0}
    result_new = {'exit_reason': None, 'pnl': 0}

    for bar_dt, bar in bars.iterrows():
        if bar_dt.date() != entry_dt.date():
            if result_current['exit_reason'] is None:
                result_current = {'exit_reason': 'eod', 'pnl': (bar['close'] - entry_price) * qty}
            if result_new['exit_reason'] is None:
                result_new = {'exit_reason': 'eod', 'pnl': (bar['close'] - entry_price) * qty}
            break

        # Current params
        if result_current['exit_reason'] is None:
            if bar['low'] <= sl_current:
                result_current = {'exit_reason': 'sl', 'pnl': (sl_current - entry_price) * qty}
            elif bar['high'] >= t1_current:
                result_current = {'exit_reason': 't1', 'pnl': (t1_current - entry_price) * qty}

        # New params
        if result_new['exit_reason'] is None:
            if bar['low'] <= sl_new:
                result_new = {'exit_reason': 'sl', 'pnl': (sl_new - entry_price) * qty}
            elif bar['high'] >= t1_new:
                result_new = {'exit_reason': 't1', 'pnl': (t1_new - entry_price) * qty}

    return result_current, result_new


def main():
    # Main analysis
    trades = load_fhm_trades()
    print('=' * 70)
    print('TRADE-BY-TRADE COMPARISON: Current vs Proposed Parameters')
    print('Current: 0.8% SL, 1.0R T1')
    print('Proposed: 1.2% SL, 1.5R T1')
    print('=' * 70)
    print()

    ohlcv_cache = {}
    saved_trades = []  # SL in current but T1 in new
    worse_trades = []  # T1 in current but SL in new
    all_comparisons = []

    for trade in trades:
        symbol = trade['symbol']
        if symbol not in ohlcv_cache:
            ohlcv_cache[symbol] = load_1m_data(symbol)
        ohlcv_df = ohlcv_cache[symbol]
        if ohlcv_df is None:
            continue

        results = simulate_both(trade, ohlcv_df)
        if results is None:
            continue

        current, new = results
        all_comparisons.append({
            'symbol': symbol,
            'date': trade['date'],
            'current_exit': current['exit_reason'],
            'current_pnl': current['pnl'],
            'new_exit': new['exit_reason'],
            'new_pnl': new['pnl']
        })

        # Track rescued trades
        if current['exit_reason'] == 'sl' and new['exit_reason'] == 't1':
            saved_trades.append({
                'symbol': symbol,
                'date': trade['date'],
                'current_pnl': current['pnl'],
                'new_pnl': new['pnl'],
                'improvement': new['pnl'] - current['pnl']
            })

        # Track trades that got worse
        if current['exit_reason'] == 't1' and new['exit_reason'] == 'sl':
            worse_trades.append({
                'symbol': symbol,
                'date': trade['date'],
                'current_pnl': current['pnl'],
                'new_pnl': new['pnl'],
                'deterioration': current['pnl'] - new['pnl']
            })

    print(f'RESCUED TRADES (SL->T1 with wider params): {len(saved_trades)}')
    print('-' * 70)
    total_rescued_pnl = 0
    for t in saved_trades:
        print(f"  {t['symbol']:20} {t['date']} | Was: Rs {t['current_pnl']:>8,.0f} | Now: Rs {t['new_pnl']:>8,.0f} | Gain: Rs {t['improvement']:>8,.0f}")
        total_rescued_pnl += t['improvement']
    print(f'  TOTAL RESCUED P&L: Rs {total_rescued_pnl:,.0f}')
    print()

    print(f'DETERIORATED TRADES (T1->SL with wider SL): {len(worse_trades)}')
    print('-' * 70)
    total_deterioration = 0
    for t in worse_trades:
        print(f"  {t['symbol']:20} {t['date']} | Was: Rs {t['current_pnl']:>8,.0f} | Now: Rs {t['new_pnl']:>8,.0f} | Loss: Rs {t['deterioration']:>8,.0f}")
        total_deterioration += t['deterioration']
    if worse_trades:
        print(f'  TOTAL DETERIORATION: Rs {total_deterioration:,.0f}')
    else:
        print('  None! Wider SL does not cause any T1 trades to become SL trades.')
    print()

    # Detailed outcome transition matrix
    print('=' * 70)
    print('OUTCOME TRANSITION MATRIX')
    print('=' * 70)
    transitions = defaultdict(int)
    for c in all_comparisons:
        key = f"{c['current_exit']} -> {c['new_exit']}"
        transitions[key] += 1

    for transition, count in sorted(transitions.items()):
        print(f'  {transition:20} : {count} trades')
    print()

    print('=' * 70)
    print('NET IMPACT')
    print('=' * 70)
    net = total_rescued_pnl - total_deterioration
    print(f'  Rescued P&L:     Rs {total_rescued_pnl:>10,.0f}')
    print(f'  Deterioration:   Rs {total_deterioration:>10,.0f}')
    print(f'  NET IMPROVEMENT: Rs {net:>10,.0f}')
    print()

    # Calculate total P&L under both scenarios
    current_total = sum(c['current_pnl'] for c in all_comparisons)
    new_total = sum(c['new_pnl'] for c in all_comparisons)
    print('=' * 70)
    print('TOTAL P&L COMPARISON')
    print('=' * 70)
    print(f'  Current params (0.8% SL, 1.0R T1): Rs {current_total:>10,.0f}')
    print(f'  Proposed params (1.2% SL, 1.5R T1): Rs {new_total:>10,.0f}')
    print(f'  Improvement: Rs {new_total - current_total:>10,.0f} ({((new_total - current_total) / abs(current_total) * 100 if current_total != 0 else 0):+.1f}%)')


if __name__ == '__main__':
    main()
