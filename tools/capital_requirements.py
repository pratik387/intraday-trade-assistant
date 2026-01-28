"""
Capital Requirements Analysis
Calculates max capital needed based on concurrent positions
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BACKTEST_FOLDERS = [
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-141442_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-185203_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-194823_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251223-111540_extracted",
]


def load_trade_timings():
    """Load trade entry/exit times and calculate capital requirements"""
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

            # Load decisions with entry times
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
                                sizing = plan.get('sizing', {})

                                decisions[trade_id] = {
                                    'setup': decision.get('setup_type', plan.get('strategy', 'unknown')),
                                    'cap_segment': sizing.get('cap_segment', 'unknown'),
                                    'regime': plan.get('regime', 'unknown'),
                                    'entry_price': plan.get('entry_ref_price', 0),
                                    'qty': sizing.get('qty', 0),
                                    'notional': sizing.get('notional', 0),
                                    'decision_ts': plan.get('decision_ts', event.get('ts', '')),
                                    'date': str(date_dir.name),
                                    'adx': plan.get('indicators', {}).get('adx', 0),
                                    'rsi': plan.get('indicators', {}).get('rsi', 50),
                                    'rank_score': plan.get('ranking', {}).get('score', 0),
                                }
                    except:
                        continue

            # Load analytics for exit times
            analytics_by_trade = defaultdict(list)
            with open(analytics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        analytics = json.loads(line)
                        trade_id = analytics.get('trade_id')
                        if trade_id:
                            analytics_by_trade[trade_id].append(analytics)
                    except:
                        continue

            # Build complete trade records
            for trade_id, decision in decisions.items():
                if trade_id in seen_ids:
                    continue

                # Find entry and final exit
                trade_analytics = analytics_by_trade.get(trade_id, [])
                if not trade_analytics:
                    continue

                # Find actual entry (TRIGGER event) and final exit
                entry_time = decision['decision_ts']
                exit_time = None
                actual_entry_price = decision['entry_price']
                total_pnl = 0

                for a in trade_analytics:
                    if a.get('is_final_exit'):
                        exit_time = a.get('timestamp')
                        total_pnl = a.get('total_trade_pnl', 0)
                        if a.get('actual_entry_price'):
                            actual_entry_price = a.get('actual_entry_price')

                if not exit_time:
                    continue

                seen_ids.add(trade_id)

                trades.append({
                    'trade_id': trade_id,
                    'date': decision['date'],
                    'setup': decision['setup'],
                    'cap_segment': decision['cap_segment'],
                    'regime': decision['regime'],
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': actual_entry_price,
                    'qty': decision['qty'],
                    'notional': decision['notional'] or (actual_entry_price * decision['qty']),
                    'pnl': total_pnl,
                    'adx': decision['adx'],
                    'rsi': decision['rsi'],
                    'rank_score': decision['rank_score'],
                })

    return trades


def apply_filter(trade, profile):
    """Apply filter profile to a trade"""
    setup = trade['setup']

    if profile == 'STRICT':
        filters = {
            'range_bounce_short': {'allowed_caps': ['micro_cap'], 'rank_min': 2.0},
            'resistance_bounce_short': {'adx_min': 20, 'rsi_min': 40, 'rsi_max': 40},
            'volume_spike_reversal_long': {'allowed_caps': ['micro_cap'], 'blocked_regimes': ['chop']},
            'squeeze_release_long': {'allowed_caps': ['mid_cap']},
            'orb_pullback_long': {'blocked_caps': ['large_cap']},
            'orb_pullback_short': {'rsi_min': 40},
            'break_of_structure_long': {'allowed_caps': ['small_cap'], 'blocked_regimes': ['trend_up']},
            'support_bounce_long': {'allowed_caps': ['micro_cap'], 'rank_min': 2.0},
            'range_bounce_long': {'adx_min': 15, 'adx_max': 25, 'rsi_min': 40, 'rsi_max': 50},
            'order_block_short': {},
            # Block everything else
        }
    elif profile == 'MODERATE':
        filters = {
            'range_bounce_short': {'allowed_caps': ['micro_cap', 'small_cap']},
            'resistance_bounce_short': {'allowed_caps': ['micro_cap', 'small_cap']},
            'volume_spike_reversal_long': {'blocked_regimes': ['chop']},
            'squeeze_release_long': {},
            'orb_pullback_long': {'blocked_caps': ['large_cap']},
            'orb_pullback_short': {},
            'break_of_structure_long': {'blocked_regimes': ['trend_up']},
            'support_bounce_long': {'allowed_caps': ['micro_cap'], 'rank_min': 1.5},
            'range_bounce_long': {'rsi_min': 55},
            'breakout_long': {'allowed_regimes': ['trend_down']},
            'premium_zone_short': {'adx_min': 20, 'adx_max': 20},
            'order_block_short': {},
            # Blocked
            'orb_breakout_long': {'blocked': True},
            'first_hour_momentum_long': {'blocked': True},
            'discount_zone_long': {'blocked': True},
            'failure_fade_short': {'blocked': True},
            'order_block_long': {'blocked': True},
        }
    elif profile == 'BALANCED':
        filters = {
            'range_bounce_short': {'allowed_caps': ['micro_cap', 'small_cap', 'mid_cap']},
            'resistance_bounce_short': {'allowed_caps': ['micro_cap', 'small_cap']},
            'volume_spike_reversal_long': {},
            'squeeze_release_long': {},
            'orb_pullback_long': {},
            'orb_pullback_short': {},
            'break_of_structure_long': {'blocked_regimes': ['trend_up']},
            'support_bounce_long': {'rank_min': 1.5},
            'range_bounce_long': {'rsi_min': 40},
            'breakout_long': {},
            'premium_zone_short': {'rank_min': 1.0},
            'order_block_short': {},
            'order_block_long': {},
            # Blocked
            'orb_breakout_long': {'blocked': True},
            'first_hour_momentum_long': {'blocked': True},
            'discount_zone_long': {'blocked': True},
            'failure_fade_short': {'blocked': True},
        }
    else:
        return True  # No filter

    if setup not in filters:
        return False

    config = filters[setup]

    if config.get('blocked'):
        return False

    if 'allowed_caps' in config:
        if trade['cap_segment'] not in config['allowed_caps']:
            return False

    if 'blocked_caps' in config:
        if trade['cap_segment'] in config['blocked_caps']:
            return False

    if 'allowed_regimes' in config:
        if trade['regime'] not in config['allowed_regimes']:
            return False

    if 'blocked_regimes' in config:
        if trade['regime'] in config['blocked_regimes']:
            return False

    if 'adx_min' in config:
        if trade['adx'] < config['adx_min']:
            return False

    if 'adx_max' in config:
        if trade['adx'] > config['adx_max']:
            return False

    if 'rsi_min' in config:
        if trade['rsi'] < config['rsi_min']:
            return False

    if 'rank_min' in config:
        if trade['rank_score'] < config['rank_min']:
            return False

    return True


def calculate_concurrent_capital(trades):
    """Calculate max concurrent capital needed per day"""
    # Group by date
    by_date = defaultdict(list)
    for t in trades:
        by_date[t['date']].append(t)

    daily_stats = []

    for date, day_trades in sorted(by_date.items()):
        # Build timeline of capital changes
        events = []
        for t in day_trades:
            entry_ts = t['entry_time']
            exit_ts = t['exit_time']
            notional = t['notional']

            events.append((entry_ts, 'entry', notional, t))
            events.append((exit_ts, 'exit', notional, t))

        # Sort by timestamp
        events.sort(key=lambda x: x[0])

        # Calculate concurrent capital
        current_capital = 0
        max_capital = 0
        max_positions = 0
        current_positions = 0

        for ts, event_type, notional, trade in events:
            if event_type == 'entry':
                current_capital += notional
                current_positions += 1
            else:
                current_capital -= notional
                current_positions -= 1

            if current_capital > max_capital:
                max_capital = current_capital
                max_positions = current_positions

        total_pnl = sum(t['pnl'] for t in day_trades)

        daily_stats.append({
            'date': date,
            'trades': len(day_trades),
            'max_concurrent_positions': max_positions,
            'max_capital_needed': max_capital,
            'total_pnl': total_pnl,
        })

    return daily_stats


def main():
    print("="*120)
    print("CAPITAL REQUIREMENTS ANALYSIS")
    print("="*120)

    print("\nLoading trade data with entry/exit times...")
    all_trades = load_trade_timings()
    print(f"Loaded {len(all_trades)} trades")

    profiles = ['NO_FILTER', 'STRICT', 'MODERATE', 'BALANCED']

    for profile in profiles:
        print(f"\n{'='*120}")
        print(f"PROFILE: {profile}")
        print(f"{'='*120}")

        if profile == 'NO_FILTER':
            filtered_trades = all_trades
        else:
            filtered_trades = [t for t in all_trades if apply_filter(t, profile)]

        if not filtered_trades:
            print("No trades after filtering")
            continue

        daily_stats = calculate_concurrent_capital(filtered_trades)

        # Calculate summary stats
        max_capital_ever = max(d['max_capital_needed'] for d in daily_stats)
        avg_max_capital = sum(d['max_capital_needed'] for d in daily_stats) / len(daily_stats)
        max_positions_ever = max(d['max_concurrent_positions'] for d in daily_stats)
        avg_max_positions = sum(d['max_concurrent_positions'] for d in daily_stats) / len(daily_stats)

        # P95 capital (95th percentile - more realistic)
        sorted_capitals = sorted(d['max_capital_needed'] for d in daily_stats)
        p95_idx = int(len(sorted_capitals) * 0.95)
        p95_capital = sorted_capitals[p95_idx] if sorted_capitals else 0

        total_pnl = sum(d['total_pnl'] for d in daily_stats)
        total_trades = sum(d['trades'] for d in daily_stats)

        print(f"\nSummary:")
        print(f"  Total Trades: {total_trades:,}")
        print(f"  Trading Days: {len(daily_stats)}")
        print(f"  Total P&L (gross): Rs {total_pnl:,.0f}")

        print(f"\nCapital Requirements:")
        print(f"  Max Capital Ever Needed: Rs {max_capital_ever:,.0f}")
        print(f"  95th Percentile Capital: Rs {p95_capital:,.0f}")
        print(f"  Average Daily Max Capital: Rs {avg_max_capital:,.0f}")

        print(f"\nConcurrent Positions:")
        print(f"  Max Positions Ever: {max_positions_ever}")
        print(f"  Average Max Positions/Day: {avg_max_positions:.1f}")

        # Margin requirement (Zerodha intraday = 5x leverage, so need 20% margin)
        margin_rate = 0.20  # 20% margin for intraday
        print(f"\nMargin Requirement (20% for intraday):")
        print(f"  Max Margin Needed: Rs {max_capital_ever * margin_rate:,.0f}")
        print(f"  95th Percentile Margin: Rs {p95_capital * margin_rate:,.0f}")
        print(f"  Average Daily Margin: Rs {avg_max_capital * margin_rate:,.0f}")

        # ROI calculation
        annual_pnl = total_pnl / 3  # 3 years of data
        roi_on_max = (annual_pnl / max_capital_ever) * 100 if max_capital_ever > 0 else 0
        roi_on_p95 = (annual_pnl / p95_capital) * 100 if p95_capital > 0 else 0
        roi_on_margin = (annual_pnl / (p95_capital * margin_rate)) * 100 if p95_capital > 0 else 0

        print(f"\nAnnual ROI (before tax):")
        print(f"  ROI on Max Capital: {roi_on_max:.1f}%")
        print(f"  ROI on P95 Capital: {roi_on_p95:.1f}%")
        print(f"  ROI on P95 Margin (with leverage): {roi_on_margin:.1f}%")

        # Show top 10 highest capital days
        print(f"\nTop 10 Highest Capital Days:")
        print(f"  {'Date':<12} {'Trades':>8} {'Max Pos':>10} {'Max Capital':>15} {'P&L':>12}")
        print("  " + "-"*65)

        top_days = sorted(daily_stats, key=lambda x: x['max_capital_needed'], reverse=True)[:10]
        for d in top_days:
            print(f"  {d['date']:<12} {d['trades']:>8} {d['max_concurrent_positions']:>10} Rs {d['max_capital_needed']:>12,.0f} Rs {d['total_pnl']:>10,.0f}")

    # Final comparison
    print("\n" + "="*120)
    print("CAPITAL COMPARISON SUMMARY")
    print("="*120)

    print(f"\n{'Profile':<15} {'Trades':<10} {'Max Capital':<18} {'P95 Capital':<18} {'Margin (20%)':<18} {'Annual ROI':<12}")
    print("-"*95)

    for profile in profiles:
        if profile == 'NO_FILTER':
            filtered = all_trades
        else:
            filtered = [t for t in all_trades if apply_filter(t, profile)]

        if not filtered:
            continue

        stats = calculate_concurrent_capital(filtered)
        max_cap = max(d['max_capital_needed'] for d in stats)
        sorted_caps = sorted(d['max_capital_needed'] for d in stats)
        p95_cap = sorted_caps[int(len(sorted_caps) * 0.95)] if sorted_caps else 0
        total_pnl = sum(d['total_pnl'] for d in stats)
        annual_pnl = total_pnl / 3
        roi = (annual_pnl / (p95_cap * 0.20)) * 100 if p95_cap > 0 else 0

        print(f"{profile:<15} {len(filtered):<10,} Rs {max_cap:<15,.0f} Rs {p95_cap:<15,.0f} Rs {p95_cap*0.20:<15,.0f} {roi:.1f}%")


if __name__ == "__main__":
    main()
