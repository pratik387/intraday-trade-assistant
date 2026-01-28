"""
Setup-by-Setup Filter Optimizer v3
Combines the best performing individual filters to find optimal combinations.
Targets: 4-5 trades/day, profitable setups Rs 300-400 avg, losing setups Rs 100+ avg
"""

import json
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
    trades = []
    seen_ids = set()

    for folder in BACKTEST_FOLDERS:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue

        for date_dir in sorted(folder_path.iterdir()):
            if not date_dir.is_dir():
                continue

            events_file = date_dir / "events.jsonl"
            analytics_file = date_dir / "analytics.jsonl"

            if not events_file.exists() or not analytics_file.exists():
                continue

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
                            'structural_rr': d.get('structural_rr', 0),
                            'entry_hour': entry_hour,
                        }
                        trades.append(trade)
                    except:
                        continue

    return trades


def calc_stats(trades_list):
    if not trades_list:
        return {'count': 0, 'winners': 0, 'wr': 0, 'total_pnl': 0, 'avg_pnl': 0}
    winners = len([t for t in trades_list if t['net_pnl'] > 0])
    total = sum(t['net_pnl'] for t in trades_list)
    avg = total / len(trades_list)
    return {'count': len(trades_list), 'winners': winners, 'wr': winners / len(trades_list) * 100, 'total_pnl': total, 'avg_pnl': avg}


def apply_filters(trades, cap_filter=None, cap_block=None, regime_filter=None, regime_block=None,
                  adx_min=None, adx_max=None, rsi_min=None, rsi_max=None,
                  vol_min=None, rank_min=None, srr_min=None, hour_max=None):
    """Apply multiple filters to trades"""
    result = trades.copy()

    if cap_filter:
        result = [t for t in result if t['cap_segment'] in cap_filter]
    if cap_block:
        result = [t for t in result if t['cap_segment'] not in cap_block]
    if regime_filter:
        result = [t for t in result if t['daily_regime'] in regime_filter]
    if regime_block:
        result = [t for t in result if t['daily_regime'] not in regime_block]
    if adx_min is not None:
        result = [t for t in result if t['adx'] >= adx_min]
    if adx_max is not None:
        result = [t for t in result if t['adx'] <= adx_max]
    if rsi_min is not None:
        result = [t for t in result if t['rsi'] >= rsi_min]
    if rsi_max is not None:
        result = [t for t in result if t['rsi'] <= rsi_max]
    if vol_min is not None:
        result = [t for t in result if t['volume'] >= vol_min]
    if rank_min is not None:
        result = [t for t in result if t['rank_score'] >= rank_min]
    if srr_min is not None:
        result = [t for t in result if t['structural_rr'] >= srr_min]
    if hour_max is not None:
        result = [t for t in result if t['entry_hour'] <= hour_max]

    return result


def analyze_setup_combinations(setup_trades, baseline_stats, target_avg_pnl):
    """Test filter combinations for a setup"""

    # Define filter options based on what worked in single filter analysis
    cap_options = [None, ['micro_cap'], ['small_cap'], ['mid_cap'], ['large_cap'],
                   ['micro_cap', 'small_cap'], ['mid_cap', 'large_cap']]
    cap_block_options = [None, ['large_cap'], ['micro_cap'], ['mid_cap']]
    regime_options = [None, ['squeeze'], ['trend_up'], ['trend_down'], ['chop']]
    regime_block_options = [None, ['trend_up'], ['trend_down'], ['chop']]
    adx_min_options = [None, 15, 20, 25, 30]
    adx_max_options = [None, 20, 25, 30, 35]
    rsi_min_options = [None, 30, 40, 50, 55, 60, 65, 70]
    rsi_max_options = [None, 40, 50, 55, 60, 65]
    vol_min_options = [None, 100000, 150000, 200000]
    rank_min_options = [None, 1.0, 1.5, 2.0]
    srr_min_options = [None, 1.5, 2.0, 2.5]

    best_results = []

    # Test focused combinations (not all combinations - too many)
    # Strategy: Test each filter category with a few combinations

    # 1. Cap-based filters
    for cap in cap_options:
        for cap_block in cap_block_options:
            if cap and cap_block:
                continue  # Don't use both
            filtered = apply_filters(setup_trades, cap_filter=cap, cap_block=cap_block)
            if len(filtered) >= 5:
                stats = calc_stats(filtered)
                if stats['avg_pnl'] > baseline_stats['avg_pnl'] + 20:
                    removed_winners = len([t for t in setup_trades if t['net_pnl'] > 0 and t not in filtered])
                    best_results.append({
                        'filters': f"cap={'|'.join(cap) if cap else 'any'}, cap_block={'|'.join(cap_block) if cap_block else 'none'}",
                        'params': {'cap_filter': cap, 'cap_block': cap_block},
                        'stats': stats,
                        'removed_winners': removed_winners
                    })

    # 2. Regime-based filters
    for regime in regime_options:
        for regime_block in regime_block_options:
            if regime and regime_block:
                continue
            filtered = apply_filters(setup_trades, regime_filter=regime, regime_block=regime_block)
            if len(filtered) >= 5:
                stats = calc_stats(filtered)
                if stats['avg_pnl'] > baseline_stats['avg_pnl'] + 20:
                    removed_winners = len([t for t in setup_trades if t['net_pnl'] > 0 and t not in filtered])
                    best_results.append({
                        'filters': f"regime={'|'.join(regime) if regime else 'any'}, block={'|'.join(regime_block) if regime_block else 'none'}",
                        'params': {'regime_filter': regime, 'regime_block': regime_block},
                        'stats': stats,
                        'removed_winners': removed_winners
                    })

    # 3. ADX + RSI combinations
    for adx_min in adx_min_options[:3]:  # Limit to reduce combinations
        for adx_max in adx_max_options[:3]:
            for rsi_min in rsi_min_options[:4]:
                for rsi_max in rsi_max_options[:3]:
                    filtered = apply_filters(setup_trades, adx_min=adx_min, adx_max=adx_max,
                                           rsi_min=rsi_min, rsi_max=rsi_max)
                    if len(filtered) >= 5:
                        stats = calc_stats(filtered)
                        if stats['avg_pnl'] > baseline_stats['avg_pnl'] + 30:
                            removed_winners = len([t for t in setup_trades if t['net_pnl'] > 0 and t not in filtered])
                            best_results.append({
                                'filters': f"adx>={adx_min or '-'} adx<={adx_max or '-'} rsi>={rsi_min or '-'} rsi<={rsi_max or '-'}",
                                'params': {'adx_min': adx_min, 'adx_max': adx_max, 'rsi_min': rsi_min, 'rsi_max': rsi_max},
                                'stats': stats,
                                'removed_winners': removed_winners
                            })

    # 4. Cap + Regime combinations
    for cap in [['micro_cap'], ['small_cap'], ['micro_cap', 'small_cap']]:
        for regime_block in [['trend_up'], ['chop'], None]:
            filtered = apply_filters(setup_trades, cap_filter=cap, regime_block=regime_block)
            if len(filtered) >= 5:
                stats = calc_stats(filtered)
                if stats['avg_pnl'] > baseline_stats['avg_pnl'] + 20:
                    removed_winners = len([t for t in setup_trades if t['net_pnl'] > 0 and t not in filtered])
                    best_results.append({
                        'filters': f"cap={'|'.join(cap)}, block_regime={'|'.join(regime_block) if regime_block else 'none'}",
                        'params': {'cap_filter': cap, 'regime_block': regime_block},
                        'stats': stats,
                        'removed_winners': removed_winners
                    })

    # 5. Rank + Cap combinations
    for rank_min in rank_min_options:
        for cap in cap_options[:4]:
            filtered = apply_filters(setup_trades, cap_filter=cap, rank_min=rank_min)
            if len(filtered) >= 5:
                stats = calc_stats(filtered)
                if stats['avg_pnl'] > baseline_stats['avg_pnl'] + 30:
                    removed_winners = len([t for t in setup_trades if t['net_pnl'] > 0 and t not in filtered])
                    best_results.append({
                        'filters': f"rank>={rank_min or '-'}, cap={'|'.join(cap) if cap else 'any'}",
                        'params': {'rank_min': rank_min, 'cap_filter': cap},
                        'stats': stats,
                        'removed_winners': removed_winners
                    })

    # Sort by avg_pnl
    best_results.sort(key=lambda x: x['stats']['avg_pnl'], reverse=True)

    # Filter to those meeting target
    meeting_target = [r for r in best_results if r['stats']['avg_pnl'] >= target_avg_pnl]

    return best_results[:15], meeting_target


def main():
    print("="*120)
    print("SETUP-BY-SETUP FILTER OPTIMIZER v3 - COMBINED FILTERS")
    print("="*120)

    print("\nLoading data...")
    trades = load_all_data()
    print(f"Loaded {len(trades)} trades")

    if len(trades) == 0:
        print("ERROR: No trades loaded")
        return

    # Get unique dates for trades/day calculation
    unique_dates = set(t['date'] for t in trades)
    trading_days = len(unique_dates)

    # Group by setup
    setup_counts = defaultdict(list)
    for t in trades:
        setup_counts[t['setup']].append(t)

    setup_stats = []
    for setup, setup_trades in setup_counts.items():
        stats = calc_stats(setup_trades)
        setup_stats.append({
            'setup': setup,
            'trades': setup_trades,
            'count': stats['count'],
            'total_pnl': stats['total_pnl'],
            'avg_pnl': stats['avg_pnl'],
            'wr': stats['wr']
        })

    setup_stats.sort(key=lambda x: x['total_pnl'], reverse=True)

    print("\n" + "="*120)
    print("SETUP OVERVIEW")
    print("="*120)
    print(f"{'Setup':<40} {'Count':>8} {'Total PnL':>14} {'Avg PnL':>10} {'WR%':>7} {'Status':<12}")
    print("-"*95)

    for s in setup_stats:
        status = "PROFITABLE" if s['avg_pnl'] > 0 else "LOSING"
        print(f"{s['setup']:<40} {s['count']:>8} {s['total_pnl']:>14,.0f} {s['avg_pnl']:>10,.1f} {s['wr']:>6.1f}% {status:<12}")

    # Analyze each setup
    print("\n" + "="*120)
    print("DETAILED FILTER RECOMMENDATIONS BY SETUP")
    print("="*120)

    final_recommendations = []

    for s in setup_stats:
        if s['count'] < 10:
            continue

        setup_name = s['setup']
        setup_trades = s['trades']
        baseline = calc_stats(setup_trades)

        # Determine target
        if baseline['avg_pnl'] > 0:
            target = 300  # Profitable setups: aim for Rs 300+
        else:
            target = 100  # Losing setups: aim for Rs 100+

        print(f"\n{'='*120}")
        print(f"SETUP: {setup_name}")
        print(f"Baseline: {baseline['count']} trades, Rs {baseline['avg_pnl']:.1f} avg, {baseline['wr']:.1f}% WR")
        print(f"Target: Rs {target}+ avg PnL")
        print(f"{'='*120}")

        best_combos, meeting_target = analyze_setup_combinations(setup_trades, baseline, target)

        if meeting_target:
            print(f"\n{len(meeting_target)} combinations meet the target Rs {target}+ avg PnL:")
            print(f"{'Filters':<60} {'Trades':>7} {'AvgPnL':>10} {'TotalPnL':>12} {'WR%':>7} {'RmvdWin':>8}")
            print("-"*110)

            for r in meeting_target[:5]:
                print(f"{r['filters']:<60} {r['stats']['count']:>7} {r['stats']['avg_pnl']:>10,.0f} {r['stats']['total_pnl']:>12,.0f} {r['stats']['wr']:>6.1f}% {r['removed_winners']:>8}")

            # Add best to recommendations
            best = meeting_target[0]
            final_recommendations.append({
                'setup': setup_name,
                'filters': best['filters'],
                'params': best['params'],
                'trades': best['stats']['count'],
                'avg_pnl': best['stats']['avg_pnl'],
                'total_pnl': best['stats']['total_pnl'],
                'wr': best['stats']['wr'],
                'removed_winners': best['removed_winners'],
                'baseline_trades': baseline['count'],
                'baseline_pnl': baseline['total_pnl']
            })

        elif best_combos:
            print(f"\nNo combination meets Rs {target}+ target. Best available:")
            best = best_combos[0]
            print(f"  {best['filters']}: {best['stats']['count']} trades, Rs {best['stats']['avg_pnl']:.0f} avg")

            if best['stats']['avg_pnl'] > 50:
                final_recommendations.append({
                    'setup': setup_name,
                    'filters': best['filters'],
                    'params': best['params'],
                    'trades': best['stats']['count'],
                    'avg_pnl': best['stats']['avg_pnl'],
                    'total_pnl': best['stats']['total_pnl'],
                    'wr': best['stats']['wr'],
                    'removed_winners': best['removed_winners'],
                    'baseline_trades': baseline['count'],
                    'baseline_pnl': baseline['total_pnl']
                })
        else:
            print(f"\nNo filter combinations improve this setup. Consider BLOCKING entirely.")
            # Check if blocking is best
            if baseline['avg_pnl'] < -50:
                print(f"  RECOMMENDATION: BLOCK this setup (saves Rs {abs(baseline['total_pnl']):,.0f})")

    # Final Summary
    print("\n" + "="*120)
    print("FINAL RECOMMENDATIONS SUMMARY")
    print("="*120)

    if final_recommendations:
        total_trades = sum(r['trades'] for r in final_recommendations)
        total_pnl = sum(r['total_pnl'] for r in final_recommendations)
        total_baseline_pnl = sum(r['baseline_pnl'] for r in final_recommendations)

        print(f"\n{'Setup':<35} {'Filters':<45} {'Trades':>7} {'AvgPnL':>10} {'TotalPnL':>12}")
        print("-"*115)

        for r in final_recommendations:
            short_filters = r['filters'][:45]
            print(f"{r['setup']:<35} {short_filters:<45} {r['trades']:>7} {r['avg_pnl']:>10,.0f} {r['total_pnl']:>12,.0f}")

        print("-"*115)
        weighted_avg = total_pnl / total_trades if total_trades > 0 else 0
        print(f"{'TOTAL WITH FILTERS':<35} {'':<45} {total_trades:>7} {weighted_avg:>10,.0f} {total_pnl:>12,.0f}")

        trades_per_day = total_trades / trading_days if trading_days > 0 else 0
        improvement = total_pnl - total_baseline_pnl

        print(f"\n{'Trading Days':<35}: {trading_days}")
        print(f"{'Trades per Day with filters':<35}: {trades_per_day:.1f}")
        print(f"{'Baseline Total PnL (these setups)':<35}: Rs {total_baseline_pnl:,.0f}")
        print(f"{'Filtered Total PnL':<35}: Rs {total_pnl:,.0f}")
        print(f"{'PnL Improvement':<35}: Rs {improvement:,.0f}")

        # Setups to block
        blocked_setups = [s for s in setup_stats if s['avg_pnl'] < -100 and s['setup'] not in [r['setup'] for r in final_recommendations]]
        if blocked_setups:
            print(f"\n{'SETUPS TO BLOCK (no profitable filter found):'}")
            block_savings = 0
            for bs in blocked_setups:
                print(f"  - {bs['setup']}: {bs['count']} trades, Rs {bs['total_pnl']:,.0f} loss")
                block_savings += abs(bs['total_pnl'])
            print(f"\n  Total savings from blocking: Rs {block_savings:,.0f}")


if __name__ == "__main__":
    main()
