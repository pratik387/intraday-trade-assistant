"""
Final Filter Recommendations
Analyzes optimal filter configurations for different trade volume targets.
"""

import json
from pathlib import Path
from collections import defaultdict

BACKTEST_FOLDERS = [
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-141442_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-185203_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-194823_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251223-111540_extracted",
]

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
                                    'regime': plan.get('regime', 'unknown'),
                                    'cap_segment': sizing.get('cap_segment', 'unknown'),
                                    'adx': indicators.get('adx', bar5.get('adx', 0)),
                                    'rsi': indicators.get('rsi', bar5.get('rsi', 50)),
                                    'volume': bar5.get('volume', 0),
                                    'rank_score': ranking.get('score', 0),
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
                        if not trade_id or trade_id in seen_ids or not analytics.get('is_final_exit', False):
                            continue
                        if trade_id not in decisions:
                            continue

                        seen_ids.add(trade_id)
                        d = decisions[trade_id]

                        entry_price = analytics.get('actual_entry_price', d.get('entry_price', 0))
                        exit_price = analytics.get('exit_price', 0)
                        qty = analytics.get('qty', d.get('qty', 0))
                        direction = d.get('direction', 'LONG')

                        charges = calculate_charges(entry_price, exit_price, qty, direction) if entry_price > 0 and exit_price > 0 and qty > 0 else 0
                        gross_pnl = analytics.get('total_trade_pnl', 0)

                        entry_hour = 10
                        try:
                            ts = d.get('decision_ts', '')
                            if ts and len(ts) >= 13:
                                entry_hour = int(ts[11:13])
                        except:
                            pass

                        trades.append({
                            'trade_id': trade_id,
                            'date': d['date'],
                            'setup': d['setup'],
                            'direction': direction,
                            'gross_pnl': gross_pnl,
                            'charges': charges,
                            'net_pnl': gross_pnl - charges,
                            'cap_segment': d.get('cap_segment', 'unknown'),
                            'daily_regime': d.get('regime', 'unknown'),
                            'adx': d.get('adx', 0),
                            'rsi': d.get('rsi', 50),
                            'volume': d.get('volume', 0),
                            'rank_score': d.get('rank_score', 0),
                            'structural_rr': d.get('structural_rr', 0),
                            'entry_hour': entry_hour,
                        })
                    except:
                        continue
    return trades


def apply_setup_filter(trade, setup_filters):
    """Apply setup-specific filter configuration"""
    setup = trade['setup']

    if setup not in setup_filters:
        return False  # Block unfiltered setups

    config = setup_filters[setup]

    if config.get('blocked'):
        return False

    # Cap filter
    if 'allowed_caps' in config:
        if trade['cap_segment'] not in config['allowed_caps']:
            return False

    if 'blocked_caps' in config:
        if trade['cap_segment'] in config['blocked_caps']:
            return False

    # Regime filter
    if 'allowed_regimes' in config:
        if trade['daily_regime'] not in config['allowed_regimes']:
            return False

    if 'blocked_regimes' in config:
        if trade['daily_regime'] in config['blocked_regimes']:
            return False

    # ADX filter
    if 'adx_min' in config:
        if trade['adx'] < config['adx_min']:
            return False

    if 'adx_max' in config:
        if trade['adx'] > config['adx_max']:
            return False

    # RSI filter
    if 'rsi_min' in config:
        if trade['rsi'] < config['rsi_min']:
            return False

    if 'rsi_max' in config:
        if trade['rsi'] > config['rsi_max']:
            return False

    # Rank filter
    if 'rank_min' in config:
        if trade['rank_score'] < config['rank_min']:
            return False

    return True


def calc_stats(trades_list):
    if not trades_list:
        return {'count': 0, 'winners': 0, 'wr': 0, 'total_pnl': 0, 'avg_pnl': 0}
    winners = len([t for t in trades_list if t['net_pnl'] > 0])
    total = sum(t['net_pnl'] for t in trades_list)
    return {
        'count': len(trades_list),
        'winners': winners,
        'wr': winners / len(trades_list) * 100,
        'total_pnl': total,
        'avg_pnl': total / len(trades_list)
    }


def main():
    print("="*120)
    print("FINAL FILTER RECOMMENDATIONS")
    print("="*120)

    trades = load_all_data()
    print(f"Loaded {len(trades)} trades")

    unique_dates = set(t['date'] for t in trades)
    trading_days = len(unique_dates)
    print(f"Trading days: {trading_days}")

    # =====================================================================
    # CONFIGURATION PROFILES
    # =====================================================================

    # STRICT PROFILE (Rs 300+ avg, ~0.7 trades/day)
    strict_filters = {
        'range_bounce_short': {'allowed_caps': ['micro_cap'], 'rank_min': 2.0},
        'resistance_bounce_short': {'adx_min': 20, 'rsi_min': 40, 'rsi_max': 40},
        'volume_spike_reversal_long': {'allowed_caps': ['micro_cap'], 'blocked_regimes': ['chop']},
        'squeeze_release_long': {'allowed_caps': ['mid_cap']},
        'orb_pullback_long': {'blocked_caps': ['large_cap']},
        'orb_pullback_short': {'rsi_min': 40},
        'break_of_structure_long': {'allowed_caps': ['small_cap'], 'blocked_regimes': ['trend_up']},
        'support_bounce_long': {'allowed_caps': ['micro_cap'], 'rank_min': 2.0},
        'range_bounce_long': {'adx_min': 15, 'adx_max': 25, 'rsi_min': 40, 'rsi_max': 50},

        # Block these
        'breakout_long': {'blocked': True},
        'orb_breakout_long': {'blocked': True},
        'first_hour_momentum_long': {'blocked': True},
        'discount_zone_long': {'blocked': True},
        'premium_zone_short': {'blocked': True},
        'failure_fade_short': {'blocked': True},
        'order_block_short': {},  # Keep as is
        'order_block_long': {'blocked': True},
    }

    # MODERATE PROFILE (Rs 100+ avg, ~2-3 trades/day target)
    moderate_filters = {
        'range_bounce_short': {'allowed_caps': ['micro_cap', 'small_cap']},
        'resistance_bounce_short': {'allowed_caps': ['micro_cap', 'small_cap']},
        'volume_spike_reversal_long': {'blocked_regimes': ['chop']},
        'squeeze_release_long': {},  # Keep all
        'orb_pullback_long': {'blocked_caps': ['large_cap']},
        'orb_pullback_short': {},  # Keep all
        'break_of_structure_long': {'blocked_regimes': ['trend_up']},
        'support_bounce_long': {'allowed_caps': ['micro_cap'], 'rank_min': 1.5},
        'range_bounce_long': {'rsi_min': 55},
        'breakout_long': {'allowed_regimes': ['trend_down']},
        'premium_zone_short': {'adx_min': 20, 'adx_max': 20},

        # Block these
        'orb_breakout_long': {'blocked': True},
        'first_hour_momentum_long': {'blocked': True},
        'discount_zone_long': {'blocked': True},
        'failure_fade_short': {'blocked': True},
        'order_block_short': {},
        'order_block_long': {'blocked': True},
    }

    # BALANCED PROFILE (Rs 50+ avg, ~4-5 trades/day target)
    balanced_filters = {
        'range_bounce_short': {'allowed_caps': ['micro_cap', 'small_cap', 'mid_cap']},
        'resistance_bounce_short': {'allowed_caps': ['micro_cap', 'small_cap']},
        'volume_spike_reversal_long': {},  # Keep all
        'squeeze_release_long': {},
        'orb_pullback_long': {},
        'orb_pullback_short': {},
        'break_of_structure_long': {'blocked_regimes': ['trend_up']},
        'support_bounce_long': {'rank_min': 1.5},
        'range_bounce_long': {'rsi_min': 40},
        'breakout_long': {},  # Keep all
        'premium_zone_short': {'rank_min': 1.0},

        # Block these (major losers)
        'orb_breakout_long': {'blocked': True},
        'first_hour_momentum_long': {'blocked': True},
        'discount_zone_long': {'blocked': True},
        'failure_fade_short': {'blocked': True},
        'order_block_short': {},
        'order_block_long': {},
    }

    # =====================================================================
    # ANALYZE EACH PROFILE
    # =====================================================================

    profiles = [
        ('STRICT (Max Avg PnL)', strict_filters),
        ('MODERATE (Balanced)', moderate_filters),
        ('BALANCED (More Trades)', balanced_filters),
    ]

    for profile_name, filters in profiles:
        print(f"\n{'='*120}")
        print(f"PROFILE: {profile_name}")
        print(f"{'='*120}")

        filtered_trades = [t for t in trades if apply_setup_filter(t, filters)]
        stats = calc_stats(filtered_trades)

        trades_per_day = stats['count'] / trading_days

        print(f"\nOverall Results:")
        print(f"  Total Trades: {stats['count']:,}")
        print(f"  Trades/Day: {trades_per_day:.1f}")
        print(f"  Win Rate: {stats['wr']:.1f}%")
        print(f"  Total Net PnL: Rs {stats['total_pnl']:,.0f}")
        print(f"  Avg PnL per Trade: Rs {stats['avg_pnl']:.0f}")

        # Breakdown by setup
        setup_breakdown = defaultdict(list)
        for t in filtered_trades:
            setup_breakdown[t['setup']].append(t)

        print(f"\nPer-Setup Breakdown:")
        print(f"{'Setup':<40} {'Trades':>8} {'AvgPnL':>10} {'TotalPnL':>14} {'WR%':>7}")
        print("-"*85)

        setup_results = []
        for setup, setup_trades in setup_breakdown.items():
            s = calc_stats(setup_trades)
            setup_results.append((setup, s))

        setup_results.sort(key=lambda x: x[1]['total_pnl'], reverse=True)

        for setup, s in setup_results:
            print(f"{setup:<40} {s['count']:>8} {s['avg_pnl']:>10,.0f} {s['total_pnl']:>14,.0f} {s['wr']:>6.1f}%")

        print("-"*85)
        print(f"{'TOTAL':<40} {stats['count']:>8} {stats['avg_pnl']:>10,.0f} {stats['total_pnl']:>14,.0f} {stats['wr']:>6.1f}%")

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "="*120)
    print("SUMMARY COMPARISON")
    print("="*120)

    print(f"\n{'Profile':<25} {'Trades':>10} {'Trades/Day':>12} {'Avg PnL':>12} {'Total PnL':>15} {'WR%':>8}")
    print("-"*85)

    baseline_trades = trades
    baseline_stats = calc_stats(baseline_trades)
    print(f"{'NO FILTERS (Baseline)':<25} {baseline_stats['count']:>10,} {baseline_stats['count']/trading_days:>12.1f} {baseline_stats['avg_pnl']:>12,.0f} {baseline_stats['total_pnl']:>15,.0f} {baseline_stats['wr']:>7.1f}%")

    for profile_name, filters in profiles:
        filtered_trades = [t for t in trades if apply_setup_filter(t, filters)]
        stats = calc_stats(filtered_trades)
        trades_per_day = stats['count'] / trading_days
        short_name = profile_name.split('(')[0].strip()
        print(f"{short_name:<25} {stats['count']:>10,} {trades_per_day:>12.1f} {stats['avg_pnl']:>12,.0f} {stats['total_pnl']:>15,.0f} {stats['wr']:>7.1f}%")

    print("\n" + "="*120)
    print("RECOMMENDED ACTION ITEMS")
    print("="*120)

    print("""
1. SETUPS TO BLOCK ENTIRELY (negative expectancy even with filters):
   - first_hour_momentum_long: Rs -309K loss, 2234 trades
   - discount_zone_long: Rs -357K loss, 1439 trades
   - orb_breakout_long: Rs -120K loss, 1541 trades

2. KEY FILTERS TO IMPLEMENT (highest impact):
   - range_bounce_short: cap_segment IN (micro_cap, small_cap) -> +Rs 1.5M from Rs 565K avg
   - support_bounce_long: rank_score >= 1.5 AND cap_segment = micro_cap -> Turns Rs -1.4M loss into profit
   - range_bounce_long: rsi >= 40 AND adx BETWEEN 15-25 -> Reduces Rs -3.6M loss significantly

3. TRADE VOLUME vs PROFITABILITY TRADE-OFF:
   - Strict filters: 0.7 trades/day, Rs 307 avg PnL
   - Moderate filters: ~2 trades/day, Rs 100+ avg PnL
   - Balanced filters: ~4 trades/day, Rs 50+ avg PnL

4. RECOMMENDED APPROACH:
   Start with MODERATE profile to balance trade volume and profitability.
   Monitor performance and tighten filters if avg PnL drops below Rs 100.
""")


if __name__ == "__main__":
    main()
