import json
import os
from collections import defaultdict

backtest_dir = 'backtest_20251208-122302_extracted'

trades = []
decisions = {}

for session_dir in os.listdir(backtest_dir):
    session_path = os.path.join(backtest_dir, session_dir)
    if not os.path.isdir(session_path):
        continue

    analytics_file = os.path.join(session_path, 'analytics.jsonl')
    if os.path.exists(analytics_file):
        with open(analytics_file) as f:
            for line in f:
                if line.strip():
                    try:
                        trade = json.loads(line)
                        if trade.get('setup_type') == 'first_hour_momentum_short' and trade.get('is_final_exit'):
                            trades.append(trade)
                    except:
                        pass

    events_file = os.path.join(session_path, 'events.jsonl')
    if os.path.exists(events_file):
        with open(events_file) as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        if event.get('type') == 'DECISION':
                            setup = event.get('decision', {}).get('setup_type', '')
                            if setup == 'first_hour_momentum_short':
                                tid = event.get('trade_id')
                                decisions[tid] = event
                    except:
                        pass

def get_features(trade, decision):
    plan = decision.get('plan', {})
    bar5 = decision.get('bar5', {})
    indicators = plan.get('indicators', {})
    ranking = plan.get('ranking', {})
    quality = plan.get('quality', {})
    components = ranking.get('components', {})

    bar5_close = bar5.get('close', 0)
    bar5_vwap = bar5.get('vwap', 0)
    vwap_dist_pct = ((bar5_close - bar5_vwap) / bar5_vwap * 100) if bar5_vwap else 0

    return {
        'pnl': trade.get('total_trade_pnl', 0),
        'regime': trade.get('regime'),
        'vwap_dist_pct': vwap_dist_pct,
        'below_vwap': vwap_dist_pct < 0,
        'bar5_volume': bar5.get('volume', 0),
        'bar5_adx': bar5.get('adx', 0),
        'bar5_bb_width': bar5.get('bb_width_proxy', 0),
        'adx': indicators.get('adx', 0),
        'rsi': indicators.get('rsi', 0),
        'rank_score': ranking.get('score', 0),
        'rank_volume': components.get('volume', 0),
        'structural_rr': quality.get('structural_rr', 0),
    }

all_data = []
for trade in trades:
    tid = trade.get('trade_id')
    if tid in decisions:
        features = get_features(trade, decisions[tid])
        all_data.append(features)

print(f'Total FHM Short trades with features: {len(all_data)}')

results = []

def test_filter(name, condition):
    filtered = [f for f in all_data if condition(f)]
    if not filtered or len(filtered) < 5:
        return None
    pnl = sum(f['pnl'] for f in filtered)
    avg_pnl = pnl / len(filtered)
    wins = sum(1 for f in filtered if f['pnl'] > 50)
    losses = sum(1 for f in filtered if f['pnl'] < -50)
    wr = wins / len(filtered) * 100 if filtered else 0
    result = {'name': name, 'trades': len(filtered), 'pnl': pnl, 'avg': avg_pnl, 'wins': wins, 'losses': losses, 'wr': wr}
    results.append(result)
    return result

# Baseline
test_filter('No filter (baseline)', lambda f: True)

# Regime filters
test_filter('chop only', lambda f: f['regime'] == 'chop')
test_filter('trend_down only', lambda f: f['regime'] == 'trend_down')
test_filter('squeeze only', lambda f: f['regime'] == 'squeeze')
test_filter('chop OR trend_down', lambda f: f['regime'] in ['chop', 'trend_down'])
test_filter('NOT squeeze', lambda f: f['regime'] != 'squeeze')

# Volume filters
test_filter('vol>=100k', lambda f: f['bar5_volume'] >= 100000)
test_filter('vol>=150k', lambda f: f['bar5_volume'] >= 150000)
test_filter('vol>=200k', lambda f: f['bar5_volume'] >= 200000)
test_filter('vol>=300k', lambda f: f['bar5_volume'] >= 300000)
test_filter('vol>=500k', lambda f: f['bar5_volume'] >= 500000)

# ADX filters
test_filter('adx>=25', lambda f: f['bar5_adx'] >= 25)
test_filter('adx>=30', lambda f: f['bar5_adx'] >= 30)
test_filter('adx>=35', lambda f: f['bar5_adx'] >= 35)
test_filter('adx>=40', lambda f: f['bar5_adx'] >= 40)

# VWAP filters (for shorts, below VWAP is good)
test_filter('below_vwap', lambda f: f['below_vwap'])
test_filter('vwap_dist<-0.1%', lambda f: f['vwap_dist_pct'] < -0.1)
test_filter('vwap_dist<-0.2%', lambda f: f['vwap_dist_pct'] < -0.2)
test_filter('vwap_dist<-0.3%', lambda f: f['vwap_dist_pct'] < -0.3)

# Rank score filters
test_filter('rank>=2.0', lambda f: f['rank_score'] >= 2.0)
test_filter('rank>=2.5', lambda f: f['rank_score'] >= 2.5)
test_filter('rank>=3.0', lambda f: f['rank_score'] >= 3.0)

# Structural RR filters
test_filter('srr>=0.8', lambda f: f['structural_rr'] >= 0.8)
test_filter('srr>=1.0', lambda f: f['structural_rr'] >= 1.0)
test_filter('srr>=1.2', lambda f: f['structural_rr'] >= 1.2)

# RSI filters (for shorts, high RSI is good - overbought)
test_filter('rsi>=50', lambda f: f['rsi'] >= 50)
test_filter('rsi>=55', lambda f: f['rsi'] >= 55)
test_filter('rsi>=60', lambda f: f['rsi'] >= 60)
test_filter('rsi<=45', lambda f: f['rsi'] <= 45)

# Combination filters
test_filter('chop + vol>=100k', lambda f: f['regime'] == 'chop' and f['bar5_volume'] >= 100000)
test_filter('chop + vol>=150k', lambda f: f['regime'] == 'chop' and f['bar5_volume'] >= 150000)
test_filter('chop + vol>=200k', lambda f: f['regime'] == 'chop' and f['bar5_volume'] >= 200000)
test_filter('chop + adx>=25', lambda f: f['regime'] == 'chop' and f['bar5_adx'] >= 25)
test_filter('chop + adx>=30', lambda f: f['regime'] == 'chop' and f['bar5_adx'] >= 30)
test_filter('chop + adx>=35', lambda f: f['regime'] == 'chop' and f['bar5_adx'] >= 35)
test_filter('chop + below_vwap', lambda f: f['regime'] == 'chop' and f['below_vwap'])
test_filter('chop + rank>=2.5', lambda f: f['regime'] == 'chop' and f['rank_score'] >= 2.5)

test_filter('!squeeze + vol>=100k', lambda f: f['regime'] != 'squeeze' and f['bar5_volume'] >= 100000)
test_filter('!squeeze + vol>=150k', lambda f: f['regime'] != 'squeeze' and f['bar5_volume'] >= 150000)
test_filter('!squeeze + vol>=200k', lambda f: f['regime'] != 'squeeze' and f['bar5_volume'] >= 200000)
test_filter('!squeeze + adx>=25', lambda f: f['regime'] != 'squeeze' and f['bar5_adx'] >= 25)
test_filter('!squeeze + adx>=30', lambda f: f['regime'] != 'squeeze' and f['bar5_adx'] >= 30)
test_filter('!squeeze + adx>=35', lambda f: f['regime'] != 'squeeze' and f['bar5_adx'] >= 35)

# 3-way combinations
test_filter('chop + vol>=100k + adx>=25', lambda f: f['regime'] == 'chop' and f['bar5_volume'] >= 100000 and f['bar5_adx'] >= 25)
test_filter('chop + vol>=150k + adx>=25', lambda f: f['regime'] == 'chop' and f['bar5_volume'] >= 150000 and f['bar5_adx'] >= 25)
test_filter('chop + vol>=100k + adx>=30', lambda f: f['regime'] == 'chop' and f['bar5_volume'] >= 100000 and f['bar5_adx'] >= 30)
test_filter('!squeeze + vol>=100k + adx>=25', lambda f: f['regime'] != 'squeeze' and f['bar5_volume'] >= 100000 and f['bar5_adx'] >= 25)
test_filter('!squeeze + vol>=100k + adx>=30', lambda f: f['regime'] != 'squeeze' and f['bar5_volume'] >= 100000 and f['bar5_adx'] >= 30)
test_filter('!squeeze + vol>=150k + adx>=30', lambda f: f['regime'] != 'squeeze' and f['bar5_volume'] >= 150000 and f['bar5_adx'] >= 30)

# More specific combinations
test_filter('chop + vol>=100k + below_vwap', lambda f: f['regime'] == 'chop' and f['bar5_volume'] >= 100000 and f['below_vwap'])
test_filter('!squeeze + vol>=100k + below_vwap', lambda f: f['regime'] != 'squeeze' and f['bar5_volume'] >= 100000 and f['below_vwap'])
test_filter('trend_down + vol>=100k', lambda f: f['regime'] == 'trend_down' and f['bar5_volume'] >= 100000)
test_filter('trend_down + adx>=30', lambda f: f['regime'] == 'trend_down' and f['bar5_adx'] >= 30)

# Sort by avg and show top filters
print('\n' + '='*100)
print('=== TOP FILTERS BY AVG PnL (min 10 trades) ===')
print('='*100)
print(f"{'Filter':<45} {'Trades':<8} {'Total PnL':<12} {'Avg':<8} {'W/L':<10} {'WR%':<6}")
print('-' * 100)

high_avg = [r for r in results if r and r['trades'] >= 10]
high_avg.sort(key=lambda x: x['avg'], reverse=True)

for r in high_avg[:25]:
    print(f"{r['name']:<45} {r['trades']:<8} {r['pnl']:>+8,.0f} Rs  {r['avg']:>+6.0f}  {r['wins']}/{r['losses']:<6} {r['wr']:.0f}%")

# Also show filters with >= 100 Rs/trade
print('\n' + '='*100)
print('=== FILTERS WITH >= 100 Rs/trade (min 5 trades) ===')
print('='*100)

high_100 = [r for r in results if r and r['avg'] >= 100 and r['trades'] >= 5]
high_100.sort(key=lambda x: x['avg'], reverse=True)

for r in high_100:
    print(f"{r['name']:<45} {r['trades']:<8} {r['pnl']:>+8,.0f} Rs  {r['avg']:>+6.0f}  {r['wins']}/{r['losses']:<6} {r['wr']:.0f}%")
