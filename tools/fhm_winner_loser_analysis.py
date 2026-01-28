"""
FHM Long: Winner vs Loser Pattern Analysis

Identifies distinguishing characteristics between winning and losing trades
to find potential filters.
"""

import json
import os
from collections import defaultdict

BACKTEST_DIR = 'backtest_20251210-123728_extracted'


def load_fhm_trades():
    all_trades = []
    for date_folder in sorted(os.listdir(BACKTEST_DIR)):
        events_file = BACKTEST_DIR + '/' + date_folder + '/events.jsonl'
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

                    features = decision_event.get('features', {})
                    indicators = plan.get('indicators', {})
                    fhm_context = plan.get('fhm_context', {})
                    bar5 = decision_event.get('bar5', {})

                    all_trades.append({
                        'date': date_folder,
                        'symbol': decision_event.get('symbol'),
                        'entry_price': trigger_event.get('trigger', {}).get('actual_price'),
                        'exit_reason': exit_data.get('reason', ''),
                        'pnl': exit_data.get('pnl', 0),
                        # Features
                        'regime': features.get('regime'),
                        'adx': features.get('adx'),
                        'rsi': features.get('rsi'),
                        'atr': features.get('atr'),
                        'volume': bar5.get('volume'),
                        'rvol': features.get('rvol') or fhm_context.get('rvol'),
                        'vwap_dist': features.get('vwap_dist_pct'),
                        'squeeze_pct': features.get('squeeze_pct'),
                        # FHM specific
                        'price_move_pct': fhm_context.get('price_move_pct'),
                        'bar_count': fhm_context.get('bar_count'),
                        'continuation_bars': fhm_context.get('continuation_bars'),
                        # Quality
                        'rank_score': plan.get('ranking', {}).get('total_score'),
                        'quality_grade': plan.get('quality', {}).get('grade'),
                    })

    return all_trades


def avg(lst, key):
    vals = [x[key] for x in lst if x.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0


def main():
    all_trades = load_fhm_trades()

    # Split into winners and losers
    winners = [t for t in all_trades if t['pnl'] > 0 and t['exit_reason'] != 'eod']
    losers = [t for t in all_trades if t['pnl'] <= 0 and t['exit_reason'] != 'eod']
    eod_trades = [t for t in all_trades if t['exit_reason'] == 'eod']

    print('=' * 80)
    print('FHM LONG: WINNER vs LOSER PATTERN ANALYSIS')
    print('=' * 80)
    print(f'Total triggered: {len(all_trades)} | Winners: {len(winners)} | Losers: {len(losers)} | EOD: {len(eod_trades)}')
    print()

    # Compare features
    features_to_compare = ['adx', 'rsi', 'volume', 'rvol', 'vwap_dist', 'squeeze_pct', 'price_move_pct', 'rank_score']

    print('FEATURE COMPARISON (Winners vs Losers)')
    print('-' * 80)
    print(f"{'Feature':<20} {'Winners Avg':>12} {'Losers Avg':>12} {'Diff':>10} {'Signal':>15}")
    print('-' * 80)

    for feat in features_to_compare:
        w_avg = avg(winners, feat)
        l_avg = avg(losers, feat)
        diff = w_avg - l_avg

        # Determine signal direction
        if abs(diff) > 0.1 * max(abs(w_avg), abs(l_avg), 1):
            if diff > 0:
                signal = 'HIGHER=BETTER'
            else:
                signal = 'LOWER=BETTER'
        else:
            signal = 'NO DIFF'

        print(f'{feat:<20} {w_avg:>12.2f} {l_avg:>12.2f} {diff:>+10.2f} {signal:>15}')

    print()
    print('REGIME BREAKDOWN')
    print('-' * 80)
    regimes = defaultdict(lambda: {'win': 0, 'lose': 0, 'pnl': 0})
    for t in all_trades:
        r = t['regime'] or 'unknown'
        if t['pnl'] > 0:
            regimes[r]['win'] += 1
        else:
            regimes[r]['lose'] += 1
        regimes[r]['pnl'] += t['pnl']

    for regime, data in sorted(regimes.items()):
        total = data['win'] + data['lose']
        wr = data['win'] / total * 100 if total > 0 else 0
        print(f"{regime:<15}: {total:3} trades | {wr:5.1f}% WR | Rs {data['pnl']:>8,.0f}")

    print()
    print('VOLUME BUCKETS')
    print('-' * 80)
    vol_buckets = {'<50k': [], '50k-100k': [], '100k-500k': [], '500k-1M': [], '>1M': []}
    for t in all_trades:
        v = t['volume'] or 0
        if v < 50000:
            vol_buckets['<50k'].append(t)
        elif v < 100000:
            vol_buckets['50k-100k'].append(t)
        elif v < 500000:
            vol_buckets['100k-500k'].append(t)
        elif v < 1000000:
            vol_buckets['500k-1M'].append(t)
        else:
            vol_buckets['>1M'].append(t)

    for bucket, trades in vol_buckets.items():
        if trades:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            total_pnl = sum(t['pnl'] for t in trades)
            wr = wins / len(trades) * 100
            print(f"{bucket:<12}: {len(trades):3} trades | {wr:5.1f}% WR | Rs {total_pnl:>8,.0f} | Avg Rs {total_pnl/len(trades):>6,.0f}")

    print()
    print('ADX BUCKETS')
    print('-' * 80)
    adx_buckets = {'<20': [], '20-30': [], '30-40': [], '>40': []}
    for t in all_trades:
        a = t['adx'] or 0
        if a < 20:
            adx_buckets['<20'].append(t)
        elif a < 30:
            adx_buckets['20-30'].append(t)
        elif a < 40:
            adx_buckets['30-40'].append(t)
        else:
            adx_buckets['>40'].append(t)

    for bucket, trades in adx_buckets.items():
        if trades:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            total_pnl = sum(t['pnl'] for t in trades)
            wr = wins / len(trades) * 100
            print(f"{bucket:<12}: {len(trades):3} trades | {wr:5.1f}% WR | Rs {total_pnl:>8,.0f} | Avg Rs {total_pnl/len(trades):>6,.0f}")

    print()
    print('RVOL BUCKETS')
    print('-' * 80)
    rvol_buckets = {'<2x': [], '2x-3x': [], '3x-5x': [], '>5x': []}
    for t in all_trades:
        r = t['rvol'] or 0
        if r < 2:
            rvol_buckets['<2x'].append(t)
        elif r < 3:
            rvol_buckets['2x-3x'].append(t)
        elif r < 5:
            rvol_buckets['3x-5x'].append(t)
        else:
            rvol_buckets['>5x'].append(t)

    for bucket, trades in rvol_buckets.items():
        if trades:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            total_pnl = sum(t['pnl'] for t in trades)
            wr = wins / len(trades) * 100
            print(f"{bucket:<12}: {len(trades):3} trades | {wr:5.1f}% WR | Rs {total_pnl:>8,.0f} | Avg Rs {total_pnl/len(trades):>6,.0f}")

    # Test filter combinations
    print()
    print('=' * 80)
    print('FILTER COMBINATION TESTS')
    print('=' * 80)

    filter_combos = [
        ('volume >= 500k', lambda t: (t['volume'] or 0) >= 500000),
        ('volume >= 1M', lambda t: (t['volume'] or 0) >= 1000000),
        ('adx >= 25', lambda t: (t['adx'] or 0) >= 25),
        ('adx >= 30', lambda t: (t['adx'] or 0) >= 30),
        ('rvol >= 3x', lambda t: (t['rvol'] or 0) >= 3),
        ('rvol >= 4x', lambda t: (t['rvol'] or 0) >= 4),
        ('regime = trend_up', lambda t: t['regime'] == 'trend_up'),
        ('vol >= 500k + adx >= 25', lambda t: (t['volume'] or 0) >= 500000 and (t['adx'] or 0) >= 25),
        ('vol >= 500k + rvol >= 3x', lambda t: (t['volume'] or 0) >= 500000 and (t['rvol'] or 0) >= 3),
        ('vol >= 1M + adx >= 25', lambda t: (t['volume'] or 0) >= 1000000 and (t['adx'] or 0) >= 25),
        ('trend_up + vol >= 500k', lambda t: t['regime'] == 'trend_up' and (t['volume'] or 0) >= 500000),
    ]

    baseline_pnl = sum(t['pnl'] for t in all_trades)
    baseline_wr = sum(1 for t in all_trades if t['pnl'] > 0) / len(all_trades) * 100 if all_trades else 0

    print(f"{'Filter':<30} {'Trades':>7} {'WR%':>7} {'Total PnL':>12} {'Avg PnL':>10} {'vs Baseline':>12}")
    print('-' * 90)
    print(f"{'BASELINE (no filter)':<30} {len(all_trades):>7} {baseline_wr:>6.1f}% Rs {baseline_pnl:>10,.0f} Rs {baseline_pnl/len(all_trades) if all_trades else 0:>8,.0f} {'':>12}")

    for name, filter_fn in filter_combos:
        filtered = [t for t in all_trades if filter_fn(t)]
        if filtered:
            total_pnl = sum(t['pnl'] for t in filtered)
            wins = sum(1 for t in filtered if t['pnl'] > 0)
            wr = wins / len(filtered) * 100
            avg_pnl = total_pnl / len(filtered)
            improvement = total_pnl - baseline_pnl * len(filtered) / len(all_trades) if all_trades else 0
            print(f"{name:<30} {len(filtered):>7} {wr:>6.1f}% Rs {total_pnl:>10,.0f} Rs {avg_pnl:>8,.0f} Rs {improvement:>+10,.0f}")


if __name__ == '__main__':
    main()
