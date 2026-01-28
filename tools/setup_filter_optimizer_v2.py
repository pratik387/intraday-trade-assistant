"""
Setup-by-Setup Filter Optimizer v2
Phase 1: Test individual filters to find impactful ones
Phase 2: Combine only the filters that show positive impact

This is much faster than testing all combinations.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# Backtest folders
BACKTEST_FOLDERS = [
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-141442_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-185203_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-194823_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251223-111540_extracted",
]

# Zerodha charges
BROKERAGE_PER_ORDER = 20
STT_RATE = 0.00025
EXCHANGE_TXN_RATE = 0.0000345
GST_RATE = 0.18
SEBI_RATE = 0.000001
STAMP_DUTY_RATE = 0.00003


def calculate_charges(entry_price, exit_price, qty, direction):
    buy_value = entry_price * qty if direction == "LONG" else exit_price * qty
    sell_value = exit_price * qty if direction == "LONG" else entry_price * qty
    turnover = buy_value + sell_value

    brokerage = min(BROKERAGE_PER_ORDER * 2, turnover * 0.0003)
    stt = sell_value * STT_RATE
    exchange_txn = turnover * EXCHANGE_TXN_RATE
    gst = (brokerage + exchange_txn) * GST_RATE
    sebi = turnover * SEBI_RATE
    stamp_duty = buy_value * STAMP_DUTY_RATE

    return brokerage + stt + exchange_txn + gst + sebi + stamp_duty


def load_all_data():
    """Load and merge data from all backtest folders"""
    trades = []
    seen_ids = set()

    for folder in BACKTEST_FOLDERS:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Folder not found: {folder}")
            continue

        for date_dir in sorted(folder_path.iterdir()):
            if not date_dir.is_dir():
                continue

            events_file = date_dir / "events.jsonl"
            analytics_file = date_dir / "analytics.jsonl"

            if not events_file.exists() or not analytics_file.exists():
                continue

            # Load events - build decision map
            decisions = {}
            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get('type') == 'DECISION':
                            trade_id = event.get('trade_id')
                            if trade_id:
                                plan = event.get('plan', {})
                                decision = event.get('decision', {})
                                indicators = plan.get('indicators', {})
                                sizing = plan.get('sizing', {})
                                ranking = plan.get('ranking', {})
                                bar5 = event.get('bar5', {})

                                decisions[trade_id] = {
                                    'setup': decision.get('setup_type', plan.get('strategy', 'unknown')),
                                    'direction': plan.get('bias', 'long').upper(),
                                    'symbol': event.get('symbol', ''),
                                    'regime': plan.get('regime', decision.get('regime', 'unknown')),
                                    'cap_segment': sizing.get('cap_segment', 'unknown'),
                                    'adx': indicators.get('adx', bar5.get('adx', 0)),
                                    'rsi': indicators.get('rsi', bar5.get('rsi', 50)),
                                    'volume': bar5.get('volume', 0),
                                    'rank_score': ranking.get('score', 0),
                                    'ranking_components': ranking.get('components', {}),
                                    'structural_rr': plan.get('quality', {}).get('structural_rr', 0),
                                    'entry_price': plan.get('entry_ref_price', 0),
                                    'qty': sizing.get('qty', 0),
                                    'decision_ts': plan.get('decision_ts', event.get('ts', '')),
                                    'date': str(date_dir.name)
                                }
                    except:
                        continue

            # Load analytics - get trade outcomes
            with open(analytics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        analytics = json.loads(line)
                        trade_id = analytics.get('trade_id')

                        if not trade_id or trade_id in seen_ids:
                            continue

                        if not analytics.get('is_final_exit', False):
                            continue

                        if trade_id not in decisions:
                            continue

                        seen_ids.add(trade_id)
                        d = decisions[trade_id]

                        entry_price = analytics.get('actual_entry_price', d.get('entry_price', 0))
                        exit_price = analytics.get('exit_price', 0)
                        qty = analytics.get('qty', d.get('qty', 0))
                        direction = d.get('direction', 'LONG')

                        charges = 0
                        if entry_price > 0 and exit_price > 0 and qty > 0:
                            charges = calculate_charges(entry_price, exit_price, qty, direction)

                        gross_pnl = analytics.get('total_trade_pnl', 0)
                        net_pnl = gross_pnl - charges

                        entry_hour = 10
                        try:
                            ts = d.get('decision_ts', '')
                            if ts and len(ts) >= 13:
                                entry_hour = int(ts[11:13])
                        except:
                            pass

                        components = d.get('ranking_components', {})

                        trade = {
                            'trade_id': trade_id,
                            'date': d['date'],
                            'symbol': d['symbol'],
                            'setup': d['setup'],
                            'direction': direction,
                            'gross_pnl': gross_pnl,
                            'charges': charges,
                            'net_pnl': net_pnl,
                            'exit_reason': analytics.get('reason', ''),
                            'cap_segment': d.get('cap_segment', 'unknown'),
                            'daily_regime': d.get('regime', 'unknown'),
                            'adx': d.get('adx', 0),
                            'rsi': d.get('rsi', 50),
                            'volume': d.get('volume', 0),
                            'rank_score': d.get('rank_score', 0),
                            'comp_volume': components.get('volume', 0),
                            'comp_rsi': components.get('rsi', 0),
                            'comp_adx': components.get('adx', 0),
                            'comp_vwap': components.get('vwap', 0),
                            'comp_distance': components.get('distance', 0),
                            'comp_squeeze': components.get('squeeze', 0),
                            'comp_acceptance': components.get('acceptance', 0),
                            'structural_rr': d.get('structural_rr', 0),
                            'entry_hour': entry_hour,
                        }

                        trades.append(trade)

                    except:
                        continue

    return trades


def calc_stats(trades_list):
    """Calculate statistics for a list of trades"""
    if not trades_list:
        return {'count': 0, 'winners': 0, 'wr': 0, 'total_pnl': 0, 'avg_pnl': 0}

    winners = len([t for t in trades_list if t['net_pnl'] > 0])
    total = sum(t['net_pnl'] for t in trades_list)
    avg = total / len(trades_list) if trades_list else 0

    return {
        'count': len(trades_list),
        'winners': winners,
        'wr': winners / len(trades_list) * 100 if trades_list else 0,
        'total_pnl': total,
        'avg_pnl': avg
    }


def test_single_filter(setup_trades, filter_name, filter_fn):
    """Test a single filter and return impact"""
    passed = [t for t in setup_trades if filter_fn(t)]
    failed = [t for t in setup_trades if not filter_fn(t)]

    if not passed or len(passed) < 5:
        return None

    passed_stats = calc_stats(passed)
    failed_stats = calc_stats(failed)
    baseline_stats = calc_stats(setup_trades)

    removed_winners = len([t for t in failed if t['net_pnl'] > 0])

    return {
        'filter': filter_name,
        'kept': passed_stats['count'],
        'removed': failed_stats['count'],
        'removed_winners': removed_winners,
        'kept_wr': passed_stats['wr'],
        'kept_avg_pnl': passed_stats['avg_pnl'],
        'kept_total_pnl': passed_stats['total_pnl'],
        'baseline_avg_pnl': baseline_stats['avg_pnl'],
        'avg_pnl_improvement': passed_stats['avg_pnl'] - baseline_stats['avg_pnl'],
        'total_pnl_improvement': passed_stats['total_pnl'] - baseline_stats['total_pnl']
    }


def analyze_setup_filters(trades, setup_name):
    """Analyze individual filters for a setup"""
    setup_trades = [t for t in trades if t['setup'] == setup_name]

    if len(setup_trades) < 10:
        return None

    baseline = calc_stats(setup_trades)

    # Define individual filters to test
    filters = {}

    # Cap segment filters
    for cap in ['large_cap', 'mid_cap', 'small_cap', 'micro_cap']:
        filters[f'cap={cap}'] = lambda t, c=cap: t['cap_segment'] == c
        filters[f'cap!={cap}'] = lambda t, c=cap: t['cap_segment'] != c

    # Regime filters
    for regime in ['trend_up', 'trend_down', 'chop', 'squeeze']:
        filters[f'regime={regime}'] = lambda t, r=regime: t['daily_regime'] == r
        filters[f'regime!={regime}'] = lambda t, r=regime: t['daily_regime'] != r

    # ADX filters
    for adx_val in [15, 20, 25, 30, 35]:
        filters[f'adx>={adx_val}'] = lambda t, v=adx_val: t['adx'] >= v
        filters[f'adx<={adx_val}'] = lambda t, v=adx_val: t['adx'] <= v

    # RSI filters
    for rsi_val in [30, 35, 40, 45, 50, 55, 60, 65, 70]:
        filters[f'rsi>={rsi_val}'] = lambda t, v=rsi_val: t['rsi'] >= v
        filters[f'rsi<={rsi_val}'] = lambda t, v=rsi_val: t['rsi'] <= v

    # Volume filters
    for vol in [50000, 100000, 150000, 200000]:
        filters[f'vol>={vol/1000:.0f}k'] = lambda t, v=vol: t['volume'] >= v

    # Rank score filters
    for score in [1.0, 1.5, 2.0, 2.5]:
        filters[f'rank>={score}'] = lambda t, v=score: t['rank_score'] >= v

    # Entry hour filters
    for hour in [9, 10, 11, 12, 13, 14]:
        filters[f'hour<={hour}'] = lambda t, h=hour: t['entry_hour'] <= h

    # Structural RR filters
    for rr in [1.0, 1.5, 2.0, 2.5]:
        filters[f'srr>={rr}'] = lambda t, v=rr: t['structural_rr'] >= v

    # Test each filter
    results = []
    for filter_name, filter_fn in filters.items():
        result = test_single_filter(setup_trades, filter_name, filter_fn)
        if result:
            results.append(result)

    # Sort by avg PnL improvement
    results.sort(key=lambda x: x['avg_pnl_improvement'], reverse=True)

    return {
        'setup': setup_name,
        'baseline': baseline,
        'filter_results': results[:30],  # Top 30 filters
    }


def main():
    print("="*100)
    print("SETUP-BY-SETUP FILTER OPTIMIZER v2")
    print("="*100)

    print("\nLoading data from all backtest folders...")
    trades = load_all_data()
    print(f"Loaded {len(trades)} trades")

    if len(trades) == 0:
        print("ERROR: No trades loaded. Check folder paths and data format.")
        return

    # Get unique setups and sort by total PnL
    setup_counts = defaultdict(list)
    for t in trades:
        setup_counts[t['setup']].append(t)

    setup_stats = []
    for setup, setup_trades in setup_counts.items():
        stats = calc_stats(setup_trades)
        setup_stats.append({
            'setup': setup,
            'count': stats['count'],
            'total_pnl': stats['total_pnl'],
            'avg_pnl': stats['avg_pnl'],
            'wr': stats['wr']
        })

    setup_stats.sort(key=lambda x: x['total_pnl'], reverse=True)

    print("\n" + "="*100)
    print("SETUP OVERVIEW (sorted by Total Net PnL)")
    print("="*100)
    print(f"{'Setup':<40} {'Count':>8} {'Total PnL':>14} {'Avg PnL':>10} {'WR%':>7} {'Status':<10}")
    print("-"*100)

    for s in setup_stats:
        status = "PROFITABLE" if s['avg_pnl'] > 0 else "LOSING"
        print(f"{s['setup']:<40} {s['count']:>8} {s['total_pnl']:>14,.0f} {s['avg_pnl']:>10,.1f} {s['wr']:>6.1f}% {status:<10}")

    print("\n" + "="*100)
    print("INDIVIDUAL FILTER ANALYSIS BY SETUP")
    print("="*100)

    all_recommendations = []

    for s in setup_stats:
        if s['count'] < 10:
            continue

        print(f"\n{'='*100}")
        print(f"SETUP: {s['setup']}")
        print(f"Baseline: {s['count']} trades, Rs {s['avg_pnl']:.1f} avg PnL, {s['wr']:.1f}% WR, Total: Rs {s['total_pnl']:,.0f}")
        print(f"{'='*100}")

        result = analyze_setup_filters(trades, s['setup'])

        if result and result['filter_results']:
            # Show top 10 filters that IMPROVE avg PnL
            positive_filters = [f for f in result['filter_results'] if f['avg_pnl_improvement'] > 10]

            if positive_filters:
                print(f"\nTop Filters that IMPROVE Avg PnL:")
                print(f"{'Filter':<25} {'Kept':>7} {'Rmvd':>6} {'RmvdW':>6} {'WR%':>7} {'AvgPnL':>10} {'Improve':>10} {'TotalPnL':>12}")
                print("-"*100)

                for f in positive_filters[:10]:
                    print(f"{f['filter']:<25} {f['kept']:>7} {f['removed']:>6} {f['removed_winners']:>6} {f['kept_wr']:>6.1f}% {f['kept_avg_pnl']:>10,.0f} {f['avg_pnl_improvement']:>+10,.0f} {f['kept_total_pnl']:>12,.0f}")

                # Find best single filter
                best = positive_filters[0]
                target_met = "YES" if best['kept_avg_pnl'] >= 100 else "NO"
                print(f"\nBest single filter: {best['filter']}")
                print(f"  Result: {best['kept']} trades, Rs {best['kept_avg_pnl']:.0f} avg PnL, {best['kept_wr']:.1f}% WR")
                print(f"  Target (Avg >= Rs 100): {target_met}")

                if best['kept_avg_pnl'] >= 100:
                    all_recommendations.append({
                        'setup': s['setup'],
                        'filter': best['filter'],
                        'trades': best['kept'],
                        'avg_pnl': best['kept_avg_pnl'],
                        'total_pnl': best['kept_total_pnl'],
                        'wr': best['kept_wr'],
                        'removed_winners': best['removed_winners']
                    })
            else:
                print(f"\nNo single filter significantly improves this setup.")

    # Summary recommendations
    print("\n" + "="*100)
    print("SUMMARY: SETUPS WITH FILTERS ACHIEVING Rs 100+ AVG PNL")
    print("="*100)

    if all_recommendations:
        total_trades = sum(r['trades'] for r in all_recommendations)
        total_pnl = sum(r['total_pnl'] for r in all_recommendations)

        print(f"\n{'Setup':<40} {'Filter':<25} {'Trades':>8} {'AvgPnL':>10} {'TotalPnL':>12} {'WR%':>7} {'RmvdW':>6}")
        print("-"*110)

        for r in all_recommendations:
            print(f"{r['setup']:<40} {r['filter']:<25} {r['trades']:>8} {r['avg_pnl']:>10,.0f} {r['total_pnl']:>12,.0f} {r['wr']:>6.1f}% {r['removed_winners']:>6}")

        print("-"*110)
        print(f"{'TOTAL':<40} {'':<25} {total_trades:>8} {total_pnl/total_trades:>10,.0f} {total_pnl:>12,.0f}")

        # Days calculation
        unique_dates = set()
        for t in trades:
            unique_dates.add(t['date'])
        trading_days = len(unique_dates)
        trades_per_day = total_trades / trading_days if trading_days > 0 else 0

        print(f"\nTrading Days: {trading_days}")
        print(f"Trades/Day with filters: {trades_per_day:.1f}")
        print(f"Target: 4-5 trades/day")
    else:
        print("\nNo setups achieved Rs 100+ avg PnL with single filters.")
        print("May need to combine multiple filters or adjust targets.")


if __name__ == "__main__":
    main()
