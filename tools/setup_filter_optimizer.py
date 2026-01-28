"""
Setup-by-Setup Filter Optimizer
Analyzes each setup type and finds optimal filter combinations using existing filters.

Uses filters available in the codebase:
- Pipeline-level: cap_segment, regime, volume, adx, rsi
- Setup-level: As defined in config files
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from itertools import product

# Backtest folders
BACKTEST_FOLDERS = [
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-141442_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-185203_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-194823_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251223-111540_extracted",
]

# Zerodha charges for net calculation
BROKERAGE_PER_ORDER = 20  # Rs 20 per executed order
STT_RATE = 0.00025  # 0.025% on sell side
EXCHANGE_TXN_RATE = 0.0000345  # NSE: 0.00345%
GST_RATE = 0.18  # 18% on brokerage + exchange
SEBI_RATE = 0.000001  # Rs 1 per crore
STAMP_DUTY_RATE = 0.00003  # 0.003% on buy side

def calculate_charges(entry_price, exit_price, qty, direction):
    """Calculate trading charges for a trade"""
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
                                    'volume_ratio': 1,  # Not directly available
                                    'rank_score': ranking.get('score', 0),
                                    'ranking_components': ranking.get('components', {}),
                                    'structural_rr': plan.get('quality', {}).get('structural_rr', 0),
                                    'entry_price': plan.get('entry_ref_price', sizing.get('notional', 0) / max(sizing.get('qty', 1), 1)),
                                    'qty': sizing.get('qty', 0),
                                    'decision_ts': plan.get('decision_ts', event.get('ts', '')),
                                    'date': str(date_dir.name)
                                }
                    except json.JSONDecodeError:
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
                        decision_data = decisions[trade_id]

                        # Get entry details
                        entry_price = analytics.get('actual_entry_price', decision_data.get('entry_price', 0))
                        exit_price = analytics.get('exit_price', 0)
                        qty = analytics.get('qty', decision_data.get('qty', 0))
                        direction = decision_data.get('direction', 'LONG')

                        # Calculate charges (estimate based on typical trade)
                        charges = 0
                        if entry_price > 0 and exit_price > 0 and qty > 0:
                            charges = calculate_charges(entry_price, exit_price, qty, direction)

                        gross_pnl = analytics.get('total_trade_pnl', 0)
                        net_pnl = gross_pnl - charges

                        # Extract entry hour from timestamp
                        entry_hour = 10
                        try:
                            ts = decision_data.get('decision_ts', '')
                            if ts and len(ts) >= 13:
                                entry_hour = int(ts[11:13])
                        except:
                            pass

                        components = decision_data.get('ranking_components', {})

                        trade = {
                            'trade_id': trade_id,
                            'date': decision_data['date'],
                            'symbol': decision_data['symbol'],
                            'setup': decision_data['setup'],
                            'direction': direction,
                            'gross_pnl': gross_pnl,
                            'charges': charges,
                            'net_pnl': net_pnl,
                            'exit_reason': analytics.get('reason', ''),

                            # Filter parameters
                            'cap_segment': decision_data.get('cap_segment', 'unknown'),
                            'daily_regime': decision_data.get('regime', 'unknown'),
                            'adx': decision_data.get('adx', 0),
                            'rsi': decision_data.get('rsi', 50),
                            'volume': decision_data.get('volume', 0),
                            'volume_ratio': decision_data.get('volume_ratio', 1),

                            # Ranking components
                            'rank_score': decision_data.get('rank_score', 0),
                            'comp_volume': components.get('volume', 0),
                            'comp_rsi': components.get('rsi', 0),
                            'comp_adx': components.get('adx', 0),
                            'comp_vwap': components.get('vwap', 0),
                            'comp_distance': components.get('distance', 0),
                            'comp_squeeze': components.get('squeeze', 0),
                            'comp_acceptance': components.get('acceptance', 0),

                            # Plan parameters
                            'structural_rr': decision_data.get('structural_rr', 0),
                            'entry_hour': entry_hour,
                            'entry_price': entry_price,
                            'qty': qty,
                        }

                        trades.append(trade)

                    except json.JSONDecodeError:
                        continue

    return trades


def analyze_setup(trades, setup_name):
    """Analyze a single setup and find optimal filter combinations"""
    setup_trades = [t for t in trades if t['setup'] == setup_name]

    if len(setup_trades) < 10:
        return None

    baseline = {
        'trades': len(setup_trades),
        'winners': len([t for t in setup_trades if t['net_pnl'] > 0]),
        'gross_pnl': sum(t['gross_pnl'] for t in setup_trades),
        'net_pnl': sum(t['net_pnl'] for t in setup_trades),
        'avg_pnl': sum(t['net_pnl'] for t in setup_trades) / len(setup_trades) if setup_trades else 0,
        'win_rate': len([t for t in setup_trades if t['net_pnl'] > 0]) / len(setup_trades) * 100 if setup_trades else 0
    }

    # Define filter options based on config files
    cap_options = [None, ['large_cap'], ['mid_cap'], ['small_cap'], ['micro_cap'],
                   ['large_cap', 'mid_cap'], ['small_cap', 'micro_cap']]

    regime_options = [None,
                      ['trend_up'], ['trend_down'], ['chop'], ['squeeze'],
                      ['trend_up', 'trend_down'],  # trending
                      ['chop', 'squeeze']]  # ranging

    regime_block_options = [None, ['trend_up'], ['trend_down'], ['chop']]

    adx_min_options = [None, 15, 20, 25, 30]
    adx_max_options = [None, 25, 30, 35]

    rsi_min_options = [None, 30, 35, 40, 45]
    rsi_max_options = [None, 55, 60, 65, 70]

    volume_min_options = [None, 50000, 100000, 150000, 200000]

    results = []

    # Test filter combinations
    for cap_filter in cap_options:
        for regime_block in regime_block_options:
            for adx_min in adx_min_options:
                for adx_max in adx_max_options:
                    for rsi_min in rsi_min_options:
                        for rsi_max in rsi_max_options:
                            for vol_min in volume_min_options:
                                # Apply filters
                                filtered = setup_trades.copy()

                                if cap_filter:
                                    filtered = [t for t in filtered if t['cap_segment'] in cap_filter]

                                if regime_block:
                                    filtered = [t for t in filtered if t['daily_regime'] not in regime_block]

                                if adx_min is not None:
                                    filtered = [t for t in filtered if t['adx'] >= adx_min]

                                if adx_max is not None:
                                    filtered = [t for t in filtered if t['adx'] <= adx_max]

                                if rsi_min is not None:
                                    filtered = [t for t in filtered if t['rsi'] >= rsi_min]

                                if rsi_max is not None:
                                    filtered = [t for t in filtered if t['rsi'] <= rsi_max]

                                if vol_min is not None:
                                    filtered = [t for t in filtered if t['volume'] >= vol_min]

                                if len(filtered) < 5:
                                    continue

                                # Calculate metrics
                                total_pnl = sum(t['net_pnl'] for t in filtered)
                                avg_pnl = total_pnl / len(filtered) if filtered else 0
                                winners = len([t for t in filtered if t['net_pnl'] > 0])
                                win_rate = winners / len(filtered) * 100 if filtered else 0

                                # Trades removed
                                removed = len(setup_trades) - len(filtered)
                                removed_winning = len([t for t in setup_trades if t['net_pnl'] > 0 and t not in filtered])

                                filter_desc = []
                                if cap_filter:
                                    filter_desc.append(f"cap={cap_filter}")
                                if regime_block:
                                    filter_desc.append(f"block_regime={regime_block}")
                                if adx_min is not None:
                                    filter_desc.append(f"adx>={adx_min}")
                                if adx_max is not None:
                                    filter_desc.append(f"adx<={adx_max}")
                                if rsi_min is not None:
                                    filter_desc.append(f"rsi>={rsi_min}")
                                if rsi_max is not None:
                                    filter_desc.append(f"rsi<={rsi_max}")
                                if vol_min is not None:
                                    filter_desc.append(f"vol>={vol_min}")

                                if not filter_desc:
                                    continue

                                results.append({
                                    'filters': ' + '.join(filter_desc),
                                    'trades': len(filtered),
                                    'winners': winners,
                                    'win_rate': win_rate,
                                    'total_pnl': total_pnl,
                                    'avg_pnl': avg_pnl,
                                    'pnl_improvement': total_pnl - baseline['net_pnl'],
                                    'avg_improvement': avg_pnl - baseline['avg_pnl'],
                                    'removed_total': removed,
                                    'removed_winning': removed_winning,
                                    'cap_filter': cap_filter,
                                    'regime_block': regime_block,
                                    'adx_min': adx_min,
                                    'adx_max': adx_max,
                                    'rsi_min': rsi_min,
                                    'rsi_max': rsi_max,
                                    'vol_min': vol_min
                                })

    # Sort by avg_pnl (target: maximize avg PnL)
    results.sort(key=lambda x: x['avg_pnl'], reverse=True)

    return {
        'setup': setup_name,
        'baseline': baseline,
        'top_combinations': results[:20] if results else [],
        'total_combinations_tested': len(results)
    }


def main():
    print("="*80)
    print("SETUP-BY-SETUP FILTER OPTIMIZER")
    print("="*80)

    print("\nLoading data from all backtest folders...")
    trades = load_all_data()
    print(f"Loaded {len(trades)} trades")

    # Get unique setups
    setup_counts = defaultdict(list)
    for t in trades:
        setup_counts[t['setup']].append(t)

    # Sort by total PnL to start with most profitable
    setup_stats = []
    for setup, setup_trades in setup_counts.items():
        total_pnl = sum(t['net_pnl'] for t in setup_trades)
        avg_pnl = total_pnl / len(setup_trades) if setup_trades else 0
        setup_stats.append({
            'setup': setup,
            'count': len(setup_trades),
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'trades': setup_trades
        })

    setup_stats.sort(key=lambda x: x['total_pnl'], reverse=True)

    print("\n" + "="*80)
    print("SETUP OVERVIEW (sorted by Total Net PnL)")
    print("="*80)
    print(f"{'Setup':<35} {'Count':>8} {'Total PnL':>12} {'Avg PnL':>10} {'Status':<10}")
    print("-"*80)

    for s in setup_stats:
        status = "PROFITABLE" if s['avg_pnl'] > 0 else "LOSING"
        print(f"{s['setup']:<35} {s['count']:>8} {s['total_pnl']:>12,.0f} {s['avg_pnl']:>10,.1f} {status:<10}")

    # Analyze each setup
    print("\n" + "="*80)
    print("DETAILED SETUP-BY-SETUP ANALYSIS")
    print("="*80)

    all_results = []

    for s in setup_stats:
        if s['count'] < 10:
            print(f"\n{s['setup']}: Skipping (only {s['count']} trades)")
            continue

        print(f"\n{'='*80}")
        print(f"ANALYZING: {s['setup']}")
        print(f"{'='*80}")

        result = analyze_setup(trades, s['setup'])
        if result:
            all_results.append(result)

            baseline = result['baseline']
            print(f"\nBASELINE:")
            print(f"  Trades: {baseline['trades']}, Win Rate: {baseline['win_rate']:.1f}%")
            print(f"  Total Net PnL: Rs {baseline['net_pnl']:,.0f}")
            print(f"  Avg Net PnL: Rs {baseline['avg_pnl']:,.1f}")

            if result['top_combinations']:
                print(f"\nTOP 5 FILTER COMBINATIONS (by Avg PnL):")
                print(f"  {'Filters':<55} {'Trades':>7} {'WR%':>6} {'Avg PnL':>10} {'Total PnL':>12} {'Rmvd Win':>8}")
                print("  " + "-"*100)

                for i, combo in enumerate(result['top_combinations'][:5]):
                    filters_str = combo['filters'][:55]
                    print(f"  {filters_str:<55} {combo['trades']:>7} {combo['win_rate']:>5.1f}% {combo['avg_pnl']:>10,.0f} {combo['total_pnl']:>12,.0f} {combo['removed_winning']:>8}")

            # Check if targets achievable
            best = result['top_combinations'][0] if result['top_combinations'] else None
            if best:
                target_met = "YES" if best['avg_pnl'] >= 100 else "NO"
                print(f"\n  Target (Avg PnL >= Rs 100): {target_met} (Best: Rs {best['avg_pnl']:.0f})")

    # Summary recommendations
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    # Calculate total potential
    total_baseline_pnl = sum(r['baseline']['net_pnl'] for r in all_results)

    print(f"\nTotal Baseline Net PnL: Rs {total_baseline_pnl:,.0f}")

    print("\nRECOMMENDED FILTERS BY SETUP:")
    print("-"*80)

    for result in all_results:
        setup = result['setup']
        baseline = result['baseline']

        if result['top_combinations']:
            best = result['top_combinations'][0]
            improvement = best['total_pnl'] - baseline['net_pnl']

            if best['avg_pnl'] > baseline['avg_pnl'] + 20:  # Only if meaningful improvement
                print(f"\n{setup}:")
                print(f"  Baseline: {baseline['trades']} trades, Rs {baseline['avg_pnl']:.0f} avg")
                print(f"  Recommended: {best['filters']}")
                print(f"  Result: {best['trades']} trades, Rs {best['avg_pnl']:.0f} avg")
                print(f"  Improvement: Rs {improvement:,.0f} total ({best['removed_winning']} winners removed)")


if __name__ == "__main__":
    main()
