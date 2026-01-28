"""
Premium Zone Short: Deep Parameter Analysis with 1m Data

Comprehensive analysis of premium_zone_short setup:
1. MFE/MAE calculation using 1m bars
2. Parameter simulation (SL%, T1 R:R)
3. Winner vs Loser pattern comparison
4. Optimal parameter recommendations
"""

import json
import os
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

BACKTEST_DIR = 'backtest_20251211-032449_extracted'
OHLCV_DIR = 'cache/ohlcv_archive'


def load_premium_zone_short_trades():
    """Load all premium_zone_short triggered trades from backtest."""
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

            if 'premium_zone_short' in setup_type.lower():
                if trade_id in triggers:
                    trigger_event = triggers[trade_id]
                    exit_event = exits.get(trade_id, {})
                    exit_data = exit_event.get('exit', {}) if isinstance(exit_event, dict) else {}

                    features = decision_event.get('features', {})
                    stop_info = plan.get('stop', {})
                    targets = plan.get('targets', [])

                    t1_level = None
                    t1_rr = None
                    for t in targets:
                        if t.get('name') == 'T1':
                            t1_level = t.get('level')
                            t1_rr = t.get('rr')

                    entry_price = trigger_event.get('trigger', {}).get('actual_price')
                    sl_price = stop_info.get('price')
                    sl_pct = abs(sl_price - entry_price) / entry_price * 100 if entry_price and sl_price else None

                    all_trades.append({
                        'date': date_folder,
                        'trade_id': trade_id,
                        'symbol': decision_event.get('symbol'),
                        'entry_ts': trigger_event.get('ts'),
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'sl_pct': sl_pct,
                        't1_level': t1_level,
                        't1_rr': t1_rr,
                        'qty': plan.get('sizing', {}).get('qty'),
                        'exit_reason': exit_data.get('reason', ''),
                        'actual_pnl': exit_data.get('pnl', 0),
                        # Features for pattern analysis
                        'regime': features.get('regime'),
                        'adx': features.get('adx'),
                        'rsi': features.get('rsi'),
                        'atr': features.get('atr'),
                        'rvol': features.get('rvol'),
                        'vwap_dist': features.get('vwap_dist_pct'),
                    })

    return all_trades


def load_1m_data(symbol):
    """Load 1-minute OHLCV data for a symbol."""
    if ':' in symbol:
        ticker = symbol.split(':')[1] + '.NS'
    else:
        ticker = symbol + '.NS'

    # Try different path patterns
    paths_to_try = [
        f'{OHLCV_DIR}/{ticker}/{ticker}_1minutes.feather',
        f'{OHLCV_DIR}/{ticker}/{ticker}_1m.feather',
        f'cache/ohlcv/{ticker}_1m.feather',
    ]

    for ohlcv_path in paths_to_try:
        if os.path.exists(ohlcv_path):
            df = pd.read_feather(ohlcv_path)
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
            elif 'datetime' not in df.columns:
                continue
            if df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_localize(None)
            df = df.set_index('datetime').sort_index()
            return df

    return None


def calculate_mfe_mae(trade, ohlcv_df):
    """Calculate MFE (max favorable) and MAE (max adverse) for a SHORT trade."""
    entry_price = trade['entry_price']
    entry_ts = trade['entry_ts']

    if not entry_price or not entry_ts:
        return None

    entry_dt = pd.to_datetime(entry_ts)
    if entry_dt.tzinfo is not None:
        entry_dt = entry_dt.tz_localize(None)

    try:
        # Get bars from entry until end of day
        bars = ohlcv_df.loc[entry_dt:]
        bars = bars[bars.index.date == entry_dt.date()]
    except:
        return None

    if len(bars) == 0:
        return None

    # For SHORT: MFE = entry - lowest low (profit), MAE = highest high - entry (loss)
    mfe_price = bars['low'].min()
    mae_price = bars['high'].max()

    mfe_pct = (entry_price - mfe_price) / entry_price * 100  # Positive = good for short
    mae_pct = (mae_price - entry_price) / entry_price * 100  # Positive = adverse move

    return {
        'mfe_pct': mfe_pct,
        'mae_pct': mae_pct,
        'mfe_price': mfe_price,
        'mae_price': mae_price,
    }


def simulate_trade(trade, ohlcv_df, sl_pct, t1_rr):
    """Simulate a SHORT trade with given SL% and T1 R:R."""
    entry_price = trade['entry_price']
    entry_ts = trade['entry_ts']
    qty = trade['qty'] or 1

    if not entry_price or not entry_ts:
        return None

    entry_dt = pd.to_datetime(entry_ts)
    if entry_dt.tzinfo is not None:
        entry_dt = entry_dt.tz_localize(None)

    # Calculate SL and T1 for SHORT
    sl_price = entry_price * (1 + sl_pct / 100)  # SL above entry for short
    risk = sl_price - entry_price
    t1_price = entry_price - (risk * t1_rr)  # T1 below entry for short

    try:
        bars = ohlcv_df.loc[entry_dt:]
    except:
        return None

    if len(bars) == 0:
        return None

    for bar_dt, bar in bars.iterrows():
        if bar_dt.date() != entry_dt.date():
            # EOD exit
            eod_pnl = (entry_price - bar['close']) * qty
            return {'exit_reason': 'eod', 'pnl': eod_pnl, 'exit_price': bar['close']}

        # Check SL first (worst case for short = price goes up)
        if bar['high'] >= sl_price:
            sl_pnl = (entry_price - sl_price) * qty  # Negative for short
            return {'exit_reason': 'sl', 'pnl': sl_pnl, 'exit_price': sl_price}

        # Check T1 (profit for short = price goes down)
        if bar['low'] <= t1_price:
            t1_pnl = (entry_price - t1_price) * qty  # Positive for short
            return {'exit_reason': 't1', 'pnl': t1_pnl, 'exit_price': t1_price}

    # Shouldn't reach here, but EOD as fallback
    last_bar = bars.iloc[-1]
    return {'exit_reason': 'eod', 'pnl': (entry_price - last_bar['close']) * qty, 'exit_price': last_bar['close']}


def avg(lst, key):
    """Calculate average of a key in list of dicts."""
    vals = [x[key] for x in lst if x.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0


def main():
    print('=' * 80)
    print('PREMIUM ZONE SHORT: DEEP PARAMETER ANALYSIS WITH 1M DATA')
    print('=' * 80)
    print()

    # Load trades
    trades = load_premium_zone_short_trades()
    print(f'Loaded {len(trades)} triggered premium_zone_short trades')
    print()

    # Load OHLCV data
    ohlcv_cache = {}
    trades_with_data = []

    for trade in trades:
        symbol = trade['symbol']
        if symbol not in ohlcv_cache:
            ohlcv_cache[symbol] = load_1m_data(symbol)

        if ohlcv_cache[symbol] is not None:
            trades_with_data.append(trade)

    print(f'Trades with 1m data available: {len(trades_with_data)}')
    print()

    # ========================================================================
    # SECTION 1: MFE/MAE ANALYSIS
    # ========================================================================
    print('=' * 80)
    print('SECTION 1: MFE/MAE ANALYSIS')
    print('=' * 80)

    mfe_mae_results = []
    for trade in trades_with_data:
        ohlcv_df = ohlcv_cache[trade['symbol']]
        result = calculate_mfe_mae(trade, ohlcv_df)
        if result:
            result['trade'] = trade
            mfe_mae_results.append(result)

    print(f'Analyzed {len(mfe_mae_results)} trades')
    print()

    # Split by outcome
    winners = [r for r in mfe_mae_results if r['trade']['actual_pnl'] > 0]
    losers = [r for r in mfe_mae_results if r['trade']['actual_pnl'] <= 0]
    sl_trades = [r for r in mfe_mae_results if 'sl' in r['trade']['exit_reason'].lower()]
    target_trades = [r for r in mfe_mae_results if 'target' in r['trade']['exit_reason'].lower()]

    print('MFE/MAE by Outcome:')
    print('-' * 80)
    print(f"{'Category':<20} {'Count':>6} {'Avg MFE%':>10} {'Avg MAE%':>10} {'MFE/MAE':>10}")
    print('-' * 80)

    for name, group in [('All Trades', mfe_mae_results), ('Winners', winners), ('Losers', losers),
                        ('SL Exits', sl_trades), ('Target Exits', target_trades)]:
        if group:
            avg_mfe = sum(r['mfe_pct'] for r in group) / len(group)
            avg_mae = sum(r['mae_pct'] for r in group) / len(group)
            ratio = avg_mfe / avg_mae if avg_mae > 0 else 0
            print(f'{name:<20} {len(group):>6} {avg_mfe:>10.2f}% {avg_mae:>10.2f}% {ratio:>10.2f}')

    print()

    # MFE distribution for SL trades (could we have saved them?)
    print('SL TRADES - Could we have saved them with wider SL?')
    print('-' * 80)
    if sl_trades:
        for r in sl_trades[:10]:  # Show first 10
            t = r['trade']
            sl_pct_str = f"{t['sl_pct']:.2f}%" if t['sl_pct'] else 'N/A'
            print(f"  {t['symbol']:20} | MFE: {r['mfe_pct']:>5.2f}% | MAE: {r['mae_pct']:>5.2f}% | Current SL: {sl_pct_str}")

        # How many SL trades had MFE > current SL (meaning they hit T1 level at some point)?
        could_be_saved = sum(1 for r in sl_trades if r['mfe_pct'] >= (r['trade']['t1_rr'] or 1.5) * (r['trade']['sl_pct'] or 0.5))
        print(f'\n  Trades that reached T1 level before SL: {could_be_saved}/{len(sl_trades)} ({could_be_saved/len(sl_trades)*100:.1f}%)')
    print()

    # ========================================================================
    # SECTION 2: PARAMETER SIMULATION
    # ========================================================================
    print('=' * 80)
    print('SECTION 2: PARAMETER SIMULATION')
    print('=' * 80)

    # Test different SL% and T1 R:R combinations
    sl_options = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    t1_options = [1.0, 1.2, 1.5, 2.0, 2.5]

    results_matrix = []

    for sl_pct in sl_options:
        for t1_rr in t1_options:
            sim_results = []
            for trade in trades_with_data:
                ohlcv_df = ohlcv_cache[trade['symbol']]
                result = simulate_trade(trade, ohlcv_df, sl_pct, t1_rr)
                if result:
                    sim_results.append(result)

            if sim_results:
                total_pnl = sum(r['pnl'] for r in sim_results)
                wins = sum(1 for r in sim_results if r['pnl'] > 0)
                wr = wins / len(sim_results) * 100
                t1_hits = sum(1 for r in sim_results if r['exit_reason'] == 't1')
                sl_hits = sum(1 for r in sim_results if r['exit_reason'] == 'sl')
                eod_exits = sum(1 for r in sim_results if r['exit_reason'] == 'eod')

                results_matrix.append({
                    'sl_pct': sl_pct,
                    't1_rr': t1_rr,
                    'total_pnl': total_pnl,
                    'win_rate': wr,
                    'trades': len(sim_results),
                    't1_hits': t1_hits,
                    'sl_hits': sl_hits,
                    'eod_exits': eod_exits,
                    'avg_pnl': total_pnl / len(sim_results)
                })

    # Sort by total PnL
    results_matrix.sort(key=lambda x: x['total_pnl'], reverse=True)

    print(f"{'SL%':>6} {'T1 R:R':>8} {'Trades':>7} {'WR%':>7} {'T1':>5} {'SL':>5} {'EOD':>5} {'Total PnL':>12} {'Avg PnL':>10}")
    print('-' * 80)

    for r in results_matrix:
        print(f"{r['sl_pct']:>6.1f}% {r['t1_rr']:>8.1f} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['t1_hits']:>5} {r['sl_hits']:>5} {r['eod_exits']:>5} Rs {r['total_pnl']:>9,.0f} Rs {r['avg_pnl']:>8,.0f}")

    print()
    print('TOP 5 PARAMETER COMBINATIONS:')
    print('-' * 80)
    for i, r in enumerate(results_matrix[:5]):
        print(f"  {i+1}. SL: {r['sl_pct']:.1f}%, T1: {r['t1_rr']:.1f}R -> Rs {r['total_pnl']:,.0f} ({r['win_rate']:.1f}% WR)")

    # ========================================================================
    # SECTION 3: WINNER VS LOSER PATTERNS
    # ========================================================================
    print()
    print('=' * 80)
    print('SECTION 3: WINNER VS LOSER PATTERNS')
    print('=' * 80)

    actual_winners = [t for t in trades_with_data if t['actual_pnl'] > 0 and t['exit_reason'] != 'eod']
    actual_losers = [t for t in trades_with_data if t['actual_pnl'] <= 0 and t['exit_reason'] != 'eod']

    print(f'\nWinners: {len(actual_winners)} | Losers: {len(actual_losers)}')
    print()

    features_to_compare = ['adx', 'rsi', 'rvol', 'vwap_dist', 'sl_pct']

    print(f"{'Feature':<15} {'Winners Avg':>12} {'Losers Avg':>12} {'Diff':>10} {'Signal':>15}")
    print('-' * 70)

    for feat in features_to_compare:
        w_avg = avg(actual_winners, feat)
        l_avg = avg(actual_losers, feat)
        diff = w_avg - l_avg

        if abs(diff) > 0.1 * max(abs(w_avg), abs(l_avg), 1):
            signal = 'HIGHER=BETTER' if diff > 0 else 'LOWER=BETTER'
        else:
            signal = 'NO DIFF'

        print(f'{feat:<15} {w_avg:>12.2f} {l_avg:>12.2f} {diff:>+10.2f} {signal:>15}')

    # Regime breakdown
    print()
    print('REGIME BREAKDOWN:')
    print('-' * 70)
    regimes = defaultdict(lambda: {'win': 0, 'lose': 0, 'pnl': 0})
    for t in trades_with_data:
        r = t['regime'] or 'unknown'
        if t['actual_pnl'] > 0:
            regimes[r]['win'] += 1
        else:
            regimes[r]['lose'] += 1
        regimes[r]['pnl'] += t['actual_pnl']

    for regime, data in sorted(regimes.items(), key=lambda x: x[1]['pnl'], reverse=True):
        total = data['win'] + data['lose']
        wr = data['win'] / total * 100 if total > 0 else 0
        print(f"  {regime:<15}: {total:3} trades | {wr:5.1f}% WR | Rs {data['pnl']:>8,.0f}")

    # ADX buckets
    print()
    print('ADX BUCKETS:')
    print('-' * 70)
    adx_buckets = {'<25': [], '25-35': [], '35-45': [], '>45': []}
    for t in trades_with_data:
        a = t['adx'] or 0
        if a < 25:
            adx_buckets['<25'].append(t)
        elif a < 35:
            adx_buckets['25-35'].append(t)
        elif a < 45:
            adx_buckets['35-45'].append(t)
        else:
            adx_buckets['>45'].append(t)

    for bucket, bucket_trades in adx_buckets.items():
        if bucket_trades:
            wins = sum(1 for t in bucket_trades if t['actual_pnl'] > 0)
            total_pnl = sum(t['actual_pnl'] for t in bucket_trades)
            wr = wins / len(bucket_trades) * 100
            print(f"  {bucket:<10}: {len(bucket_trades):3} trades | {wr:5.1f}% WR | Rs {total_pnl:>8,.0f}")

    # ========================================================================
    # SECTION 4: RECOMMENDATIONS
    # ========================================================================
    print()
    print('=' * 80)
    print('SECTION 4: RECOMMENDATIONS')
    print('=' * 80)

    # Get current baseline
    current_results = [r for r in results_matrix if r['sl_pct'] == 0.6 and r['t1_rr'] == 1.5]
    best_result = results_matrix[0]

    print()
    if current_results:
        current = current_results[0]
        print(f"CURRENT PARAMS (SL: 0.6%, T1: 1.5R):")
        print(f"  Total PnL: Rs {current['total_pnl']:,.0f}")
        print(f"  Win Rate: {current['win_rate']:.1f}%")
        print(f"  T1/SL/EOD: {current['t1_hits']}/{current['sl_hits']}/{current['eod_exits']}")
    print()
    print(f"RECOMMENDED PARAMS (SL: {best_result['sl_pct']:.1f}%, T1: {best_result['t1_rr']:.1f}R):")
    print(f"  Total PnL: Rs {best_result['total_pnl']:,.0f}")
    print(f"  Win Rate: {best_result['win_rate']:.1f}%")
    print(f"  T1/SL/EOD: {best_result['t1_hits']}/{best_result['sl_hits']}/{best_result['eod_exits']}")

    if current_results:
        improvement = best_result['total_pnl'] - current_results[0]['total_pnl']
        print(f"\n  IMPROVEMENT: Rs {improvement:+,.0f}")

    print()
    print('KEY FINDINGS:')
    print('-' * 80)

    # Analyze what makes the best params work
    if best_result['sl_pct'] > 0.6:
        print(f"  • Wider SL ({best_result['sl_pct']:.1f}% vs 0.6%) reduces premature stops")
    if best_result['sl_pct'] < 0.6:
        print(f"  • Tighter SL ({best_result['sl_pct']:.1f}% vs 0.6%) cuts losses faster")
    if best_result['t1_rr'] != 1.5:
        print(f"  • T1 at {best_result['t1_rr']:.1f}R {'captures more quick wins' if best_result['t1_rr'] < 1.5 else 'lets winners run'}")

    # Check regime filter opportunity
    best_regime = max(regimes.items(), key=lambda x: x[1]['pnl'])
    worst_regime = min(regimes.items(), key=lambda x: x[1]['pnl'])
    print(f"  • Best regime: {best_regime[0]} (Rs {best_regime[1]['pnl']:,.0f})")
    print(f"  • Worst regime: {worst_regime[0]} (Rs {worst_regime[1]['pnl']:,.0f})")
    if worst_regime[1]['pnl'] < 0:
        print(f"  • Consider BLOCKING {worst_regime[0]} regime")


if __name__ == '__main__':
    main()
