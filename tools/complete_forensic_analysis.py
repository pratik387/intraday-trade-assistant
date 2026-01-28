"""
COMPLETE FORENSIC ANALYSIS - Using all available data fields
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BACKTEST_DIR = Path("backtest_20251205-102858_extracted/20251205-102858_full/20251205-102858")

def load_all_data():
    """Load comprehensive data from all sessions"""
    all_decisions = []
    all_triggers = []
    all_exits = []

    sessions = [d for d in BACKTEST_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()]

    for session_dir in sessions:
        events_file = session_dir / "events.jsonl"
        analytics_file = session_dir / "analytics.jsonl"

        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('type') == 'DECISION':
                            all_decisions.append(event)
                        elif event.get('type') == 'TRIGGER':
                            all_triggers.append(event)
                    except:
                        pass

        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('stage') == 'EXIT' and event.get('is_final_exit', False):
                            all_exits.append(event)
                    except:
                        pass

    return all_decisions, all_triggers, all_exits

def build_comprehensive_trades():
    """Build complete trade records with all data"""
    decisions, triggers, exits = load_all_data()

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    trigger_lookup = {t.get('trade_id'): t for t in triggers}

    trades = []
    for e in exits:
        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})
        t = trigger_lookup.get(trade_id, {})

        if not d:
            continue

        plan = d.get('plan', {})
        stop = plan.get('stop', {})
        quality = plan.get('quality', {})
        indicators = plan.get('indicators', {})
        sizing = plan.get('sizing', {})
        ranking = plan.get('ranking', {})
        entry = plan.get('entry', {})

        # Calculate SL distance
        entry_ref = entry.get('reference', 0)
        hard_sl = stop.get('hard', 0)
        sl_distance_pct = abs(entry_ref - hard_sl) / entry_ref * 100 if entry_ref else 0

        # Calculate SL in ATR
        atr = indicators.get('atr', 0)
        sl_in_atr = abs(entry_ref - hard_sl) / atr if atr else 0

        # Calculate time in trade
        trigger_ts = t.get('ts', '') if t else ''
        exit_ts = e.get('timestamp', '')
        duration_mins = 0

        if trigger_ts and exit_ts:
            try:
                trigger_time = datetime.strptime(trigger_ts, '%Y-%m-%d %H:%M:%S')
                exit_time = datetime.strptime(exit_ts, '%Y-%m-%d %H:%M:%S')
                duration_mins = (exit_time - trigger_time).total_seconds() / 60
            except:
                try:
                    trigger_time = datetime.fromisoformat(trigger_ts.replace('Z', '+00:00'))
                    exit_time = datetime.fromisoformat(exit_ts.replace('Z', '+00:00'))
                    duration_mins = (exit_time - trigger_time).total_seconds() / 60
                except:
                    pass

        # Actual entry price
        actual_entry = t.get('trigger', {}).get('actual_price', 0) if t else 0
        entry_slippage_pct = abs(actual_entry - entry_ref) / entry_ref * 100 if entry_ref and actual_entry else 0

        # Calculate how much SL was eaten by slippage
        if plan.get('bias') == 'long':
            effective_sl_distance = actual_entry - hard_sl if actual_entry else 0
        else:
            effective_sl_distance = hard_sl - actual_entry if actual_entry else 0

        trades.append({
            'trade_id': trade_id,
            'symbol': e.get('symbol', ''),
            'setup': e.get('setup_type', ''),
            'bias': plan.get('bias', ''),
            'regime': e.get('regime', ''),
            'pnl': e.get('total_trade_pnl', e.get('pnl', 0)),
            'exit_reason': e.get('reason', ''),
            'duration_mins': duration_mins,

            # SL Data
            'entry_ref': entry_ref,
            'hard_sl': hard_sl,
            'sl_distance_pct': sl_distance_pct,
            'sl_in_atr': sl_in_atr,
            'risk_per_share': stop.get('risk_per_share', 0),

            # Entry Data
            'actual_entry': actual_entry,
            'entry_slippage_pct': entry_slippage_pct,
            'effective_sl_distance': effective_sl_distance,
            'slippage_bps': e.get('slippage_bps', 0),

            # Quality Data
            'structural_rr': quality.get('structural_rr', 0),
            'quality_status': quality.get('status', ''),
            't1_feasible': quality.get('t1_feasible', False),
            't2_feasible': quality.get('t2_feasible', False),

            # Indicators
            'atr': atr,
            'adx': indicators.get('adx', 0),
            'rsi': indicators.get('rsi', 0),

            # Ranking
            'rank_score': ranking.get('score', 0),

            # Sizing
            'qty': sizing.get('qty', 0),
            'notional': sizing.get('notional', 0),
            'risk_rupees': sizing.get('risk_rupees', 0),

            # Timing
            'entry_hour': trigger_ts[11:13] if len(trigger_ts) >= 13 else '',
            'date': trigger_ts[:10] if len(trigger_ts) >= 10 else '',

            # Targets
            'targets': plan.get('targets', []),
        })

    return trades

def main():
    trades = build_comprehensive_trades()

    print("="*100)
    print("COMPLETE FORENSIC ANALYSIS")
    print("="*100)

    total_pnl = sum(t['pnl'] for t in trades)
    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]

    print(f"\nTotal trades: {len(trades)}")
    print(f"Total PnL: Rs {total_pnl:.0f}")
    print(f"Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")

    # =========================================================================
    # STOP LOSS ANALYSIS
    # =========================================================================
    print("\n" + "="*100)
    print("STOP LOSS ANALYSIS")
    print("="*100)

    hard_sl_trades = [t for t in trades if t['exit_reason'] == 'hard_sl']
    print(f"\nHard SL trades: {len(hard_sl_trades)}")
    print(f"Hard SL total loss: Rs {sum(t['pnl'] for t in hard_sl_trades):.0f}")

    # SL distance in ATR
    print("\n--- HARD SL BY SL DISTANCE (ATR) ---")
    atr_buckets = [
        ("< 0.5 ATR (very tight)", lambda x: x < 0.5),
        ("0.5-1.0 ATR", lambda x: 0.5 <= x < 1.0),
        ("1.0-1.5 ATR", lambda x: 1.0 <= x < 1.5),
        ("1.5-2.0 ATR", lambda x: 1.5 <= x < 2.0),
        ("> 2.0 ATR", lambda x: x >= 2.0),
    ]

    for bucket_name, bucket_fn in atr_buckets:
        bucket = [t for t in hard_sl_trades if t['sl_in_atr'] and bucket_fn(t['sl_in_atr'])]
        if bucket:
            print(f"  {bucket_name:<25} {len(bucket):>3} trades, Rs {sum(t['pnl'] for t in bucket):>8.0f}")

    # SL distance in %
    print("\n--- HARD SL BY SL DISTANCE (%) ---")
    pct_buckets = [
        ("< 0.5%", lambda x: x < 0.5),
        ("0.5-1.0%", lambda x: 0.5 <= x < 1.0),
        ("1.0-1.5%", lambda x: 1.0 <= x < 1.5),
        ("1.5-2.0%", lambda x: 1.5 <= x < 2.0),
        ("> 2.0%", lambda x: x >= 2.0),
    ]

    for bucket_name, bucket_fn in pct_buckets:
        bucket = [t for t in hard_sl_trades if t['sl_distance_pct'] and bucket_fn(t['sl_distance_pct'])]
        if bucket:
            print(f"  {bucket_name:<25} {len(bucket):>3} trades, Rs {sum(t['pnl'] for t in bucket):>8.0f}")

    # Hard SL by setup
    print("\n--- HARD SL BY SETUP ---")
    by_setup = defaultdict(lambda: {'count': 0, 'pnl': 0, 'sl_atr': [], 'sl_pct': []})
    for t in hard_sl_trades:
        by_setup[t['setup']]['count'] += 1
        by_setup[t['setup']]['pnl'] += t['pnl']
        if t['sl_in_atr']:
            by_setup[t['setup']]['sl_atr'].append(t['sl_in_atr'])
        if t['sl_distance_pct']:
            by_setup[t['setup']]['sl_pct'].append(t['sl_distance_pct'])

    print(f"{'Setup':<30} {'Count':>6} {'PnL':>10} {'Avg SL ATR':>12} {'Avg SL %':>10}")
    print("-"*75)
    for setup, data in sorted(by_setup.items(), key=lambda x: x[1]['pnl']):
        avg_atr = sum(data['sl_atr']) / len(data['sl_atr']) if data['sl_atr'] else 0
        avg_pct = sum(data['sl_pct']) / len(data['sl_pct']) if data['sl_pct'] else 0
        print(f"{setup:<30} {data['count']:>6} {data['pnl']:>10.0f} {avg_atr:>12.2f} {avg_pct:>9.2f}%")

    # =========================================================================
    # TIME IN TRADE ANALYSIS
    # =========================================================================
    print("\n" + "="*100)
    print("TIME IN TRADE ANALYSIS")
    print("="*100)

    trades_with_duration = [t for t in trades if t['duration_mins'] > 0]
    print(f"\nTrades with duration data: {len(trades_with_duration)}")

    # Duration buckets
    print("\n--- PERFORMANCE BY DURATION ---")
    dur_buckets = [
        ("0-30 mins", lambda x: x < 30),
        ("30-60 mins", lambda x: 30 <= x < 60),
        ("60-120 mins", lambda x: 60 <= x < 120),
        ("120-180 mins", lambda x: 120 <= x < 180),
        ("180-240 mins", lambda x: 180 <= x < 240),
        ("240+ mins", lambda x: x >= 240),
    ]

    print(f"{'Duration':<15} {'Count':>6} {'Wins':>6} {'WR':>8} {'PnL':>12} {'Avg PnL':>10}")
    print("-"*65)
    for bucket_name, bucket_fn in dur_buckets:
        bucket = [t for t in trades_with_duration if bucket_fn(t['duration_mins'])]
        if bucket:
            wins = len([t for t in bucket if t['pnl'] > 0])
            wr = wins / len(bucket) * 100
            total_pnl = sum(t['pnl'] for t in bucket)
            avg_pnl = total_pnl / len(bucket)
            print(f"{bucket_name:<15} {len(bucket):>6} {wins:>6} {wr:>7.1f}% {total_pnl:>12.0f} {avg_pnl:>10.0f}")

    # Hard SL duration analysis
    print("\n--- HARD SL BY DURATION (How fast are stops hit?) ---")
    hard_sl_with_dur = [t for t in hard_sl_trades if t['duration_mins'] > 0]

    sl_dur_buckets = [
        ("< 15 mins (v.quick)", lambda x: x < 15),
        ("15-30 mins", lambda x: 15 <= x < 30),
        ("30-60 mins", lambda x: 30 <= x < 60),
        ("60+ mins", lambda x: x >= 60),
    ]

    for bucket_name, bucket_fn in sl_dur_buckets:
        bucket = [t for t in hard_sl_with_dur if bucket_fn(t['duration_mins'])]
        if bucket:
            avg_sl_atr = sum(t['sl_in_atr'] for t in bucket if t['sl_in_atr']) / len(bucket)
            print(f"  {bucket_name:<20} {len(bucket):>3} trades, Rs {sum(t['pnl'] for t in bucket):>8.0f}, avg SL={avg_sl_atr:.2f} ATR")

    # =========================================================================
    # ENTRY SLIPPAGE ANALYSIS
    # =========================================================================
    print("\n" + "="*100)
    print("ENTRY SLIPPAGE IMPACT")
    print("="*100)

    trades_with_slippage = [t for t in trades if t['slippage_bps'] > 0]

    # How much of the SL buffer is eaten by slippage?
    print("\n--- SLIPPAGE AS % OF SL BUFFER ---")
    for t in hard_sl_trades[:10]:  # Sample
        if t['risk_per_share'] and t['slippage_bps'] and t['entry_ref']:
            slippage_rs = t['entry_ref'] * t['slippage_bps'] / 10000
            sl_eaten_pct = slippage_rs / t['risk_per_share'] * 100 if t['risk_per_share'] else 0
            if sl_eaten_pct > 20:  # Significant slippage
                print(f"  {t['symbol'][:25]:<25} {t['setup']:<25} Slip={t['slippage_bps']:.0f}bps = {sl_eaten_pct:.0f}% of SL buffer")

    # High slippage setups
    print("\n--- SLIPPAGE BY SETUP ---")
    by_setup = defaultdict(lambda: {'count': 0, 'avg_slippage': []})
    for t in trades:
        by_setup[t['setup']]['count'] += 1
        by_setup[t['setup']]['avg_slippage'].append(t['slippage_bps'])

    print(f"{'Setup':<30} {'Count':>6} {'Avg Slippage':>15}")
    print("-"*55)
    for setup, data in sorted(by_setup.items(), key=lambda x: -sum(x[1]['avg_slippage'])/len(x[1]['avg_slippage']) if x[1]['avg_slippage'] else 0):
        avg_slip = sum(data['avg_slippage']) / len(data['avg_slippage']) if data['avg_slippage'] else 0
        print(f"{setup:<30} {data['count']:>6} {avg_slip:>14.1f} bps")

    # =========================================================================
    # QUALITY STATUS ANALYSIS
    # =========================================================================
    print("\n" + "="*100)
    print("QUALITY STATUS ANALYSIS")
    print("="*100)

    by_quality = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        by_quality[t['quality_status']]['count'] += 1
        by_quality[t['quality_status']]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            by_quality[t['quality_status']]['wins'] += 1

    print(f"\n{'Quality':<15} {'Count':>6} {'Wins':>6} {'WR':>8} {'PnL':>12}")
    print("-"*50)
    for quality, data in sorted(by_quality.items(), key=lambda x: x[1]['pnl']):
        wr = data['wins'] / data['count'] * 100 if data['count'] else 0
        print(f"{quality:<15} {data['count']:>6} {data['wins']:>6} {wr:>7.1f}% {data['pnl']:>12.0f}")

    # =========================================================================
    # STRUCTURAL RR ANALYSIS
    # =========================================================================
    print("\n" + "="*100)
    print("STRUCTURAL RR ANALYSIS")
    print("="*100)

    rr_buckets = [
        ("< 1.0", lambda x: x < 1.0),
        ("1.0-1.5", lambda x: 1.0 <= x < 1.5),
        ("1.5-2.0", lambda x: 1.5 <= x < 2.0),
        ("2.0-2.5", lambda x: 2.0 <= x < 2.5),
        ("2.5-3.0", lambda x: 2.5 <= x < 3.0),
        ("3.0+", lambda x: x >= 3.0),
    ]

    print(f"\n{'RR Bucket':<15} {'Count':>6} {'Wins':>6} {'WR':>8} {'PnL':>12} {'Avg PnL':>10}")
    print("-"*60)
    for bucket_name, bucket_fn in rr_buckets:
        bucket = [t for t in trades if t['structural_rr'] and bucket_fn(t['structural_rr'])]
        if bucket:
            wins = len([t for t in bucket if t['pnl'] > 0])
            wr = wins / len(bucket) * 100
            total_pnl = sum(t['pnl'] for t in bucket)
            avg_pnl = total_pnl / len(bucket)
            print(f"{bucket_name:<15} {len(bucket):>6} {wins:>6} {wr:>7.1f}% {total_pnl:>12.0f} {avg_pnl:>10.0f}")

    # =========================================================================
    # ENTRY HOUR ANALYSIS
    # =========================================================================
    print("\n" + "="*100)
    print("ENTRY HOUR ANALYSIS")
    print("="*100)

    by_hour = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0, 'hard_sl': 0})
    for t in trades:
        hour = t['entry_hour']
        by_hour[hour]['count'] += 1
        by_hour[hour]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            by_hour[hour]['wins'] += 1
        if t['exit_reason'] == 'hard_sl':
            by_hour[hour]['hard_sl'] += 1

    print(f"\n{'Hour':>4} {'Count':>6} {'Wins':>6} {'WR':>8} {'Hard SL':>8} {'PnL':>12}")
    print("-"*55)
    for hour in sorted(by_hour.keys()):
        data = by_hour[hour]
        wr = data['wins'] / data['count'] * 100 if data['count'] else 0
        print(f"{hour:>4} {data['count']:>6} {data['wins']:>6} {wr:>7.1f}% {data['hard_sl']:>8} {data['pnl']:>12.0f}")

    # =========================================================================
    # SETUP + REGIME COMBINATIONS
    # =========================================================================
    print("\n" + "="*100)
    print("SETUP + REGIME COMBINATIONS")
    print("="*100)

    combos = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0, 'hard_sl': 0})
    for t in trades:
        key = (t['setup'], t['regime'])
        combos[key]['count'] += 1
        combos[key]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            combos[key]['wins'] += 1
        if t['exit_reason'] == 'hard_sl':
            combos[key]['hard_sl'] += 1

    # Worst combinations
    print("\n--- WORST COMBINATIONS (min 3 trades, WR < 50%) ---")
    print(f"{'Setup':<25} {'Regime':<12} {'Count':>6} {'WR':>8} {'SL%':>8} {'PnL':>12}")
    print("-"*80)

    worst = []
    for key, data in combos.items():
        if data['count'] >= 3:
            wr = data['wins'] / data['count'] * 100
            if wr < 50:
                worst.append((key, data, wr))

    worst.sort(key=lambda x: x[1]['pnl'])
    for key, data, wr in worst[:20]:
        setup, regime = key
        sl_pct = data['hard_sl'] / data['count'] * 100
        print(f"{setup:<25} {regime:<12} {data['count']:>6} {wr:>7.1f}% {sl_pct:>7.1f}% {data['pnl']:>12.0f}")

    # Best combinations
    print("\n--- BEST COMBINATIONS (min 3 trades, WR > 60%) ---")
    print(f"{'Setup':<25} {'Regime':<12} {'Count':>6} {'WR':>8} {'SL%':>8} {'PnL':>12}")
    print("-"*80)

    best = []
    for key, data in combos.items():
        if data['count'] >= 3:
            wr = data['wins'] / data['count'] * 100
            if wr > 60:
                best.append((key, data, wr))

    best.sort(key=lambda x: -x[1]['pnl'])
    for key, data, wr in best[:20]:
        setup, regime = key
        sl_pct = data['hard_sl'] / data['count'] * 100
        print(f"{setup:<25} {regime:<12} {data['count']:>6} {wr:>7.1f}% {sl_pct:>7.1f}% {data['pnl']:>12.0f}")

    # =========================================================================
    # ADX FILTER ANALYSIS
    # =========================================================================
    print("\n" + "="*100)
    print("ADX FILTER ANALYSIS")
    print("="*100)

    # ADX for shorts
    short_trades = [t for t in trades if t['bias'] == 'short']
    print(f"\n--- SHORT TRADES BY ADX ---")
    adx_buckets = [
        ("< 15 (weak)", lambda x: x < 15),
        ("15-20", lambda x: 15 <= x < 20),
        ("20-25", lambda x: 20 <= x < 25),
        ("25-30", lambda x: 25 <= x < 30),
        ("30-40", lambda x: 30 <= x < 40),
        ("40+ (strong)", lambda x: x >= 40),
    ]

    print(f"{'ADX':<15} {'Count':>6} {'Wins':>6} {'WR':>8} {'PnL':>12}")
    print("-"*50)
    for bucket_name, bucket_fn in adx_buckets:
        bucket = [t for t in short_trades if t['adx'] and bucket_fn(t['adx'])]
        if bucket:
            wins = len([t for t in bucket if t['pnl'] > 0])
            wr = wins / len(bucket) * 100
            total_pnl = sum(t['pnl'] for t in bucket)
            print(f"{bucket_name:<15} {len(bucket):>6} {wins:>6} {wr:>7.1f}% {total_pnl:>12.0f}")

    # ADX for longs
    long_trades = [t for t in trades if t['bias'] == 'long']
    print(f"\n--- LONG TRADES BY ADX ---")
    print(f"{'ADX':<15} {'Count':>6} {'Wins':>6} {'WR':>8} {'PnL':>12}")
    print("-"*50)
    for bucket_name, bucket_fn in adx_buckets:
        bucket = [t for t in long_trades if t['adx'] and bucket_fn(t['adx'])]
        if bucket:
            wins = len([t for t in bucket if t['pnl'] > 0])
            wr = wins / len(bucket) * 100
            total_pnl = sum(t['pnl'] for t in bucket)
            print(f"{bucket_name:<15} {len(bucket):>6} {wins:>6} {wr:>7.1f}% {total_pnl:>12.0f}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*100)
    print("ACTIONABLE FILTERS TO IMPLEMENT")
    print("="*100)

    # Calculate impact of each filter
    filters = [
        ("Block vwap_lose_short entirely", lambda t: t['setup'] == 'vwap_lose_short'),
        ("Block resistance_bounce_short at hour 13", lambda t: t['setup'] == 'resistance_bounce_short' and t['entry_hour'] == '13'),
        ("Block resistance_bounce_short in trend_up", lambda t: t['setup'] == 'resistance_bounce_short' and t['regime'] == 'trend_up'),
        ("Block discount_zone_long entirely", lambda t: t['setup'] == 'discount_zone_long'),
        ("Block short trades with ADX < 18", lambda t: t['bias'] == 'short' and t['adx'] and t['adx'] < 18),
        ("Block trades with SL < 1.0 ATR", lambda t: t['sl_in_atr'] and t['sl_in_atr'] < 1.0),
        ("Block entries after hour 14", lambda t: t['entry_hour'] and t['entry_hour'] >= '14'),
        ("Block quality=poor trades", lambda t: t['quality_status'] == 'poor'),
    ]

    print(f"\n{'Filter':<50} {'Blocked':>8} {'PnL Blocked':>12} {'Improvement':>12}")
    print("-"*90)

    for name, filter_fn in filters:
        blocked = [t for t in trades if filter_fn(t)]
        blocked_pnl = sum(t['pnl'] for t in blocked)
        improvement = -blocked_pnl
        marker = " ***" if improvement > 500 else ""
        print(f"{name:<50} {len(blocked):>8} {blocked_pnl:>12.0f} {improvement:>+12.0f}{marker}")

    # Combined filter simulation
    print("\n" + "="*100)
    print("COMBINED FILTER SIMULATION")
    print("="*100)

    # Apply only beneficial filters
    remaining = trades.copy()
    applied_filters = []

    beneficial_filters = [
        ("Block vwap_lose_short", lambda t: t['setup'] == 'vwap_lose_short'),
        ("Block resistance_bounce_short hour 13", lambda t: t['setup'] == 'resistance_bounce_short' and t['entry_hour'] == '13'),
        ("Block resistance_bounce_short trend_up", lambda t: t['setup'] == 'resistance_bounce_short' and t['regime'] == 'trend_up'),
        ("Block discount_zone_long", lambda t: t['setup'] == 'discount_zone_long'),
        ("Block shorts with ADX < 18", lambda t: t['bias'] == 'short' and t['adx'] and t['adx'] < 18),
    ]

    print(f"\nStarting: {len(remaining)} trades, Rs {sum(t['pnl'] for t in remaining):.0f}")

    for name, filter_fn in beneficial_filters:
        blocked = [t for t in remaining if filter_fn(t)]
        blocked_pnl = sum(t['pnl'] for t in blocked)

        if blocked_pnl < 0:  # Only apply if it helps
            remaining = [t for t in remaining if not filter_fn(t)]
            new_pnl = sum(t['pnl'] for t in remaining)
            applied_filters.append(name)
            print(f"  + {name}: blocked {len(blocked)} trades (Rs {blocked_pnl:.0f}), now {len(remaining)} trades (Rs {new_pnl:.0f})")

    final_pnl = sum(t['pnl'] for t in remaining)
    original_pnl = sum(t['pnl'] for t in trades)

    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"  Original: {len(trades)} trades, Rs {original_pnl:.0f}")
    print(f"  After filters: {len(remaining)} trades, Rs {final_pnl:.0f}")
    print(f"  Improvement: Rs {final_pnl - original_pnl:.0f}")
    print(f"  Gap to 50,000: Rs {50000 - final_pnl:.0f}")

if __name__ == "__main__":
    main()
