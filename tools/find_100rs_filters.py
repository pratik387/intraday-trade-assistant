import json
import os

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
                        if trade.get('setup_type') == 'first_hour_momentum_long' and trade.get('is_final_exit'):
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
                            if setup == 'first_hour_momentum_long':
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
        'above_vwap': vwap_dist_pct > 0,
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

results = []

def test_filter(name, condition):
    filtered = [f for f in all_data if condition(f)]
    if not filtered or len(filtered) < 5:  # Need at least 5 trades
        return None
    pnl = sum(f['pnl'] for f in filtered)
    avg_pnl = pnl / len(filtered)
    wins = sum(1 for f in filtered if f['pnl'] > 50)
    losses = sum(1 for f in filtered if f['pnl'] < -50)
    wr = wins / len(filtered) * 100 if filtered else 0
    result = {'name': name, 'trades': len(filtered), 'pnl': pnl, 'avg': avg_pnl, 'wins': wins, 'losses': losses, 'wr': wr}
    results.append(result)
    return result

print('Analyzing all filter combinations...')

# Test all squeeze combinations
test_filter('squeeze only', lambda f: f['regime'] == 'squeeze')
test_filter('squeeze + adx>=30', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 30)
test_filter('squeeze + adx>=32', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 32)
test_filter('squeeze + adx>=35', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 35)
test_filter('squeeze + adx>=38', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 38)
test_filter('squeeze + adx>=40', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 40)
test_filter('squeeze + adx>=45', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 45)
test_filter('squeeze + vol>=100k', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 100000)
test_filter('squeeze + vol>=150k', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 150000)
test_filter('squeeze + vol>=200k', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 200000)
test_filter('squeeze + vol>=250k', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 250000)
test_filter('squeeze + vol>=300k', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 300000)
test_filter('squeeze + vol>=400k', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 400000)
test_filter('squeeze + vol>=500k', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 500000)
test_filter('squeeze + vol>=600k', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 600000)
test_filter('squeeze + vol>=700k', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 700000)
test_filter('squeeze + vol>=800k', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 800000)
test_filter('squeeze + vol>=1M', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 1000000)

# ADX + Volume combos
test_filter('squeeze + adx>=30 + vol>=100k', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 30 and f['bar5_volume'] >= 100000)
test_filter('squeeze + adx>=30 + vol>=150k', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 30 and f['bar5_volume'] >= 150000)
test_filter('squeeze + adx>=30 + vol>=200k', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 30 and f['bar5_volume'] >= 200000)
test_filter('squeeze + adx>=30 + vol>=250k', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 30 and f['bar5_volume'] >= 250000)
test_filter('squeeze + adx>=30 + vol>=300k', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 30 and f['bar5_volume'] >= 300000)
test_filter('squeeze + adx>=35 + vol>=100k', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 35 and f['bar5_volume'] >= 100000)
test_filter('squeeze + adx>=35 + vol>=150k', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 35 and f['bar5_volume'] >= 150000)
test_filter('squeeze + adx>=35 + vol>=200k', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 35 and f['bar5_volume'] >= 200000)
test_filter('squeeze + adx>=35 + vol>=250k', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 35 and f['bar5_volume'] >= 250000)
test_filter('squeeze + adx>=35 + vol>=300k', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 35 and f['bar5_volume'] >= 300000)

# Rank score combos
test_filter('squeeze + rank>=1.5', lambda f: f['regime'] == 'squeeze' and f['rank_score'] >= 1.5)
test_filter('squeeze + rank>=2.0', lambda f: f['regime'] == 'squeeze' and f['rank_score'] >= 2.0)
test_filter('squeeze + rank>=2.5', lambda f: f['regime'] == 'squeeze' and f['rank_score'] >= 2.5)
test_filter('squeeze + rank>=3.0', lambda f: f['regime'] == 'squeeze' and f['rank_score'] >= 3.0)
test_filter('squeeze + rank>=3.5', lambda f: f['regime'] == 'squeeze' and f['rank_score'] >= 3.5)

# RR combos
test_filter('squeeze + rr>=0.7', lambda f: f['regime'] == 'squeeze' and f['structural_rr'] >= 0.7)
test_filter('squeeze + rr>=0.8', lambda f: f['regime'] == 'squeeze' and f['structural_rr'] >= 0.8)
test_filter('squeeze + rr>=0.9', lambda f: f['regime'] == 'squeeze' and f['structural_rr'] >= 0.9)
test_filter('squeeze + rr>=1.0', lambda f: f['regime'] == 'squeeze' and f['structural_rr'] >= 1.0)
test_filter('squeeze + rr>=1.2', lambda f: f['regime'] == 'squeeze' and f['structural_rr'] >= 1.2)

# Above VWAP combos
test_filter('squeeze + above_vwap', lambda f: f['regime'] == 'squeeze' and f['above_vwap'])
test_filter('squeeze + above_vwap + vol>=200k', lambda f: f['regime'] == 'squeeze' and f['above_vwap'] and f['bar5_volume'] >= 200000)
test_filter('squeeze + above_vwap + vol>=300k', lambda f: f['regime'] == 'squeeze' and f['above_vwap'] and f['bar5_volume'] >= 300000)
test_filter('squeeze + above_vwap + vol>=400k', lambda f: f['regime'] == 'squeeze' and f['above_vwap'] and f['bar5_volume'] >= 400000)
test_filter('squeeze + above_vwap + vol>=500k', lambda f: f['regime'] == 'squeeze' and f['above_vwap'] and f['bar5_volume'] >= 500000)
test_filter('squeeze + above_vwap + adx>=30', lambda f: f['regime'] == 'squeeze' and f['above_vwap'] and f['bar5_adx'] >= 30)
test_filter('squeeze + above_vwap + adx>=35', lambda f: f['regime'] == 'squeeze' and f['above_vwap'] and f['bar5_adx'] >= 35)
test_filter('squeeze + above_vwap + adx>=40', lambda f: f['regime'] == 'squeeze' and f['above_vwap'] and f['bar5_adx'] >= 40)

# 3-way combos
test_filter('squeeze + adx>=35 + rank>=2.0', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 35 and f['rank_score'] >= 2.0)
test_filter('squeeze + adx>=35 + rank>=2.5', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 35 and f['rank_score'] >= 2.5)
test_filter('squeeze + adx>=35 + rr>=0.7', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 35 and f['structural_rr'] >= 0.7)
test_filter('squeeze + adx>=35 + rr>=0.8', lambda f: f['regime'] == 'squeeze' and f['bar5_adx'] >= 35 and f['structural_rr'] >= 0.8)
test_filter('squeeze + vol>=200k + rank>=2.0', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 200000 and f['rank_score'] >= 2.0)
test_filter('squeeze + vol>=200k + rank>=2.5', lambda f: f['regime'] == 'squeeze' and f['bar5_volume'] >= 200000 and f['rank_score'] >= 2.5)

# VWAP distance combos (above VWAP by margin)
test_filter('squeeze + vwap_dist>0.05%', lambda f: f['regime'] == 'squeeze' and f['vwap_dist_pct'] > 0.05)
test_filter('squeeze + vwap_dist>0.1%', lambda f: f['regime'] == 'squeeze' and f['vwap_dist_pct'] > 0.1)
test_filter('squeeze + vwap_dist>0.15%', lambda f: f['regime'] == 'squeeze' and f['vwap_dist_pct'] > 0.15)
test_filter('squeeze + vwap_dist>0.2%', lambda f: f['regime'] == 'squeeze' and f['vwap_dist_pct'] > 0.2)
test_filter('squeeze + vwap_dist>0.3%', lambda f: f['regime'] == 'squeeze' and f['vwap_dist_pct'] > 0.3)
test_filter('squeeze + vwap_dist>0.1% + adx>=30', lambda f: f['regime'] == 'squeeze' and f['vwap_dist_pct'] > 0.1 and f['bar5_adx'] >= 30)
test_filter('squeeze + vwap_dist>0.2% + adx>=30', lambda f: f['regime'] == 'squeeze' and f['vwap_dist_pct'] > 0.2 and f['bar5_adx'] >= 30)

# Filter results for avg >= 100
print('\n' + '='*100)
print('=== FILTERS WITH >= 100 Rs/trade (min 5 trades) ===')
print('='*100)
print(f'{"Filter":<55} {"Trades":<6} {"Total PnL":<12} {"Avg":<8} {"W/L":<10} {"WR%"}')
print('-' * 100)

high_avg = [r for r in results if r and r['avg'] >= 100]
high_avg.sort(key=lambda x: x['avg'], reverse=True)

for r in high_avg:
    print(f"{r['name']:<55} {r['trades']:<6} {r['pnl']:>+8,.0f} Rs  {r['avg']:>+6.0f}  {r['wins']}/{r['losses']:<6} {r['wr']:.0f}%")

# Also show positive but below 100
print('\n' + '='*100)
print('=== PROFITABLE FILTERS (50-100 Rs/trade, min 5 trades) ===')
print('='*100)
mid_avg = [r for r in results if r and 50 <= r['avg'] < 100]
mid_avg.sort(key=lambda x: x['avg'], reverse=True)

for r in mid_avg:
    print(f"{r['name']:<55} {r['trades']:<6} {r['pnl']:>+8,.0f} Rs  {r['avg']:>+6.0f}  {r['wins']}/{r['losses']:<6} {r['wr']:.0f}%")

# Show all positive
print('\n' + '='*100)
print('=== ALL PROFITABLE FILTERS (>0 Rs/trade, min 5 trades) ===')
print('='*100)
positive = [r for r in results if r and r['avg'] > 0]
positive.sort(key=lambda x: x['avg'], reverse=True)

for r in positive:
    print(f"{r['name']:<55} {r['trades']:<6} {r['pnl']:>+8,.0f} Rs  {r['avg']:>+6.0f}  {r['wins']}/{r['losses']:<6} {r['wr']:.0f}%")
