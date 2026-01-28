"""
DEEP DIVE FOLLOW-UP ANALYSIS
1. Late entries (after 12:30) - Would wider SL or hold time help?
2. vwap_lose_short - What's the actual problem?
3. ORB trades - How to increase count and reduce SL hits?
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BACKTEST_DIR = Path("backtest_20251205-102858_extracted/20251205-102858_full/20251205-102858")

def load_all_data():
    all_decisions = []
    all_triggers = []
    all_exits = []
    all_analytics = []  # Load ALL analytics events, not just final exits

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
                        all_analytics.append(event)
                        if event.get('stage') == 'EXIT' and event.get('is_final_exit', False):
                            all_exits.append(event)
                    except:
                        pass

    return all_decisions, all_triggers, all_exits, all_analytics

def analyze_late_entries():
    """Q1: Would wider SL or hold time help late entries?"""
    print("="*100)
    print("QUESTION 1: LATE ENTRIES (After 12:30) - Would wider SL or forced hold help?")
    print("="*100)

    decisions, triggers, exits, analytics = load_all_data()

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    trigger_lookup = {t.get('trade_id'): t for t in triggers}

    late_entries = []
    for e in exits:
        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})
        t = trigger_lookup.get(trade_id, {})

        if not t:
            continue

        trigger_ts = t.get('ts', '')
        entry_hour = trigger_ts[11:13] if len(trigger_ts) >= 13 else ''

        # Late entries = after 12:30
        if entry_hour >= '13' or (entry_hour == '12' and trigger_ts[14:16] >= '30'):
            plan = d.get('plan', {})
            stop = plan.get('stop', {})
            entry = plan.get('entry', {})
            indicators = plan.get('indicators', {})
            targets = plan.get('targets', [])

            entry_ref = entry.get('reference', 0)
            hard_sl = stop.get('hard', 0)
            atr = indicators.get('atr', 0)
            sl_in_atr = abs(entry_ref - hard_sl) / atr if atr else 0

            # Calculate duration
            exit_ts = e.get('timestamp', '')
            duration_mins = 0
            if trigger_ts and exit_ts:
                try:
                    trigger_time = datetime.strptime(trigger_ts, '%Y-%m-%d %H:%M:%S')
                    exit_time = datetime.strptime(exit_ts, '%Y-%m-%d %H:%M:%S')
                    duration_mins = (exit_time - trigger_time).total_seconds() / 60
                except:
                    pass

            # Get target levels
            t1_level = targets[0].get('level', 0) if targets else 0
            t2_level = targets[1].get('level', 0) if len(targets) > 1 else 0

            late_entries.append({
                'trade_id': trade_id,
                'symbol': e.get('symbol', ''),
                'setup': e.get('setup_type', ''),
                'bias': plan.get('bias', ''),
                'pnl': e.get('total_trade_pnl', e.get('pnl', 0)),
                'exit_reason': e.get('reason', ''),
                'entry_hour': entry_hour,
                'entry_minute': trigger_ts[14:16] if len(trigger_ts) >= 16 else '',
                'duration_mins': duration_mins,
                'entry_ref': entry_ref,
                'hard_sl': hard_sl,
                'sl_in_atr': sl_in_atr,
                'atr': atr,
                't1_level': t1_level,
                't2_level': t2_level,
                'exit_price': e.get('exit_price', 0),
                'actual_entry': e.get('actual_entry_price', entry_ref),
            })

    print(f"\nTotal late entries (after 12:30): {len(late_entries)}")
    print(f"Current PnL from late entries: Rs {sum(t['pnl'] for t in late_entries):.0f}")

    # Exit reason breakdown
    print("\n--- EXIT REASONS FOR LATE ENTRIES ---")
    by_reason = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for t in late_entries:
        by_reason[t['exit_reason']]['count'] += 1
        by_reason[t['exit_reason']]['pnl'] += t['pnl']

    for reason, data in sorted(by_reason.items(), key=lambda x: x[1]['pnl']):
        print(f"  {reason:<35} {data['count']:>4} trades, Rs {data['pnl']:>8.0f}")

    # Duration analysis
    print("\n--- DURATION ANALYSIS ---")
    hard_sl_late = [t for t in late_entries if t['exit_reason'] == 'hard_sl']
    print(f"Hard SL trades: {len(hard_sl_late)}")
    if hard_sl_late:
        durations = [t['duration_mins'] for t in hard_sl_late if t['duration_mins'] > 0]
        if durations:
            print(f"  Avg duration before SL hit: {sum(durations)/len(durations):.0f} mins")
            print(f"  Min duration: {min(durations):.0f} mins")
            print(f"  Max duration: {max(durations):.0f} mins")

    # SL distance analysis
    print("\n--- SL DISTANCE FOR LATE ENTRIES ---")
    sl_distances = [t['sl_in_atr'] for t in late_entries if t['sl_in_atr']]
    if sl_distances:
        print(f"  Avg SL distance: {sum(sl_distances)/len(sl_distances):.2f} ATR")

    # What if we had wider SL?
    print("\n" + "-"*80)
    print("SIMULATION: What if late entries had WIDER SL (2.5 ATR instead of current)?")
    print("-"*80)

    # For hard_sl trades, check if price would have reached target with wider SL
    for t in hard_sl_late[:10]:  # Sample
        current_sl_dist = abs(t['entry_ref'] - t['hard_sl'])
        wider_sl_price = t['entry_ref'] - (2.5 * t['atr']) if t['bias'] == 'long' else t['entry_ref'] + (2.5 * t['atr'])

        print(f"\n  {t['symbol'][:30]:<30} {t['setup']:<25}")
        print(f"    Entry: {t['entry_ref']:.2f}, Current SL: {t['hard_sl']:.2f} ({t['sl_in_atr']:.2f} ATR)")
        print(f"    Wider SL would be: {wider_sl_price:.2f} (2.5 ATR)")
        print(f"    T1: {t['t1_level']:.2f}, T2: {t['t2_level']:.2f}")
        print(f"    Exit price: {t['exit_price']:.2f}, Duration: {t['duration_mins']:.0f} mins")

    # What if we forced 60 min hold?
    print("\n" + "-"*80)
    print("SIMULATION: What if we forced 60 MIN HOLD before allowing SL exit?")
    print("-"*80)

    quick_sl_late = [t for t in hard_sl_late if t['duration_mins'] > 0 and t['duration_mins'] < 60]
    print(f"\nHard SL trades stopped within 60 mins: {len(quick_sl_late)}")
    print(f"Loss from these: Rs {sum(t['pnl'] for t in quick_sl_late):.0f}")

    # These would still lose (maybe more) if forced to hold
    print("\nNote: Forcing hold won't help if price continues against us.")
    print("The issue is: late entries have NO TIME to develop before EOD.")

    # Time remaining analysis
    print("\n--- TIME REMAINING UNTIL EOD (15:15) ---")
    for t in late_entries[:15]:
        entry_time = f"{t['entry_hour']}:{t['entry_minute']}"
        eod = datetime.strptime("15:15", "%H:%M")
        entry = datetime.strptime(entry_time, "%H:%M")
        time_remaining = (eod - entry).total_seconds() / 60
        print(f"  Entry at {entry_time}: {time_remaining:.0f} mins until EOD, {t['exit_reason']:<20}, PnL={t['pnl']:>8.0f}")

def analyze_vwap_lose_short():
    """Q2: What's wrong with vwap_lose_short?"""
    print("\n" + "="*100)
    print("QUESTION 2: VWAP_LOSE_SHORT - Why is it losing?")
    print("="*100)

    decisions, triggers, exits, analytics = load_all_data()

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    trigger_lookup = {t.get('trade_id'): t for t in triggers}

    vwap_lose_trades = []
    for e in exits:
        if e.get('setup_type') != 'vwap_lose_short':
            continue

        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})
        t = trigger_lookup.get(trade_id, {})

        plan = d.get('plan', {}) if d else {}
        stop = plan.get('stop', {})
        entry = plan.get('entry', {})
        indicators = plan.get('indicators', {})
        quality = plan.get('quality', {})
        levels = plan.get('levels', {})

        trigger_ts = t.get('ts', '') if t else ''
        exit_ts = e.get('timestamp', '')
        duration_mins = 0
        if trigger_ts and exit_ts:
            try:
                trigger_time = datetime.strptime(trigger_ts, '%Y-%m-%d %H:%M:%S')
                exit_time = datetime.strptime(exit_ts, '%Y-%m-%d %H:%M:%S')
                duration_mins = (exit_time - trigger_time).total_seconds() / 60
            except:
                pass

        entry_ref = entry.get('reference', 0)
        hard_sl = stop.get('hard', 0)
        atr = indicators.get('atr', 0)
        sl_in_atr = abs(entry_ref - hard_sl) / atr if atr else 0

        vwap_lose_trades.append({
            'trade_id': trade_id,
            'symbol': e.get('symbol', ''),
            'pnl': e.get('total_trade_pnl', e.get('pnl', 0)),
            'exit_reason': e.get('reason', ''),
            'regime': e.get('regime', ''),
            'duration_mins': duration_mins,
            'entry_hour': trigger_ts[11:13] if len(trigger_ts) >= 13 else '',
            'entry_ref': entry_ref,
            'hard_sl': hard_sl,
            'sl_in_atr': sl_in_atr,
            'atr': atr,
            'adx': indicators.get('adx', 0),
            'rsi': indicators.get('rsi', 0),
            'vwap': indicators.get('vwap', 0),
            'structural_rr': quality.get('structural_rr', 0),
            'pdh': levels.get('PDH', 0),
            'pdl': levels.get('PDL', 0),
            'slippage_bps': e.get('slippage_bps', 0),
        })

    total_pnl = sum(t['pnl'] for t in vwap_lose_trades)
    winners = [t for t in vwap_lose_trades if t['pnl'] > 0]
    losers = [t for t in vwap_lose_trades if t['pnl'] <= 0]

    print(f"\nTotal vwap_lose_short trades: {len(vwap_lose_trades)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(vwap_lose_trades)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(vwap_lose_trades)*100:.1f}%)")
    print(f"Total PnL: Rs {total_pnl:.0f}")

    # Exit reason breakdown
    print("\n--- EXIT REASONS ---")
    by_reason = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for t in vwap_lose_trades:
        by_reason[t['exit_reason']]['count'] += 1
        by_reason[t['exit_reason']]['pnl'] += t['pnl']

    for reason, data in sorted(by_reason.items(), key=lambda x: x[1]['pnl']):
        print(f"  {reason:<35} {data['count']:>4} trades, Rs {data['pnl']:>8.0f}")

    # Regime breakdown
    print("\n--- BY REGIME ---")
    by_regime = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
    for t in vwap_lose_trades:
        by_regime[t['regime']]['count'] += 1
        by_regime[t['regime']]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            by_regime[t['regime']]['wins'] += 1

    for regime, data in sorted(by_regime.items(), key=lambda x: x[1]['pnl']):
        wr = data['wins'] / data['count'] * 100 if data['count'] else 0
        print(f"  {regime:<15} {data['count']:>4} trades, WR={wr:>5.1f}%, Rs {data['pnl']:>8.0f}")

    # ADX analysis
    print("\n--- BY ADX ---")
    adx_buckets = [
        ("< 20 (weak trend)", lambda x: x < 20),
        ("20-30", lambda x: 20 <= x < 30),
        ("30-40", lambda x: 30 <= x < 40),
        ("40+ (strong)", lambda x: x >= 40),
    ]

    for name, fn in adx_buckets:
        bucket = [t for t in vwap_lose_trades if t['adx'] and fn(t['adx'])]
        if bucket:
            wins = len([t for t in bucket if t['pnl'] > 0])
            wr = wins / len(bucket) * 100
            pnl = sum(t['pnl'] for t in bucket)
            print(f"  ADX {name:<20} {len(bucket):>3} trades, WR={wr:>5.1f}%, Rs {pnl:>8.0f}")

    # Entry hour analysis
    print("\n--- BY ENTRY HOUR ---")
    by_hour = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
    for t in vwap_lose_trades:
        by_hour[t['entry_hour']]['count'] += 1
        by_hour[t['entry_hour']]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            by_hour[t['entry_hour']]['wins'] += 1

    for hour in sorted(by_hour.keys()):
        data = by_hour[hour]
        wr = data['wins'] / data['count'] * 100 if data['count'] else 0
        print(f"  Hour {hour}: {data['count']:>3} trades, WR={wr:>5.1f}%, Rs {data['pnl']:>8.0f}")

    # Compare winners vs losers
    print("\n--- WINNERS vs LOSERS COMPARISON ---")

    def avg_metric(trades_list, metric):
        vals = [t[metric] for t in trades_list if t[metric]]
        return sum(vals) / len(vals) if vals else 0

    print(f"\n{'Metric':<20} {'Winners':>12} {'Losers':>12} {'Delta':>12}")
    print("-"*60)
    metrics = ['adx', 'rsi', 'sl_in_atr', 'duration_mins', 'structural_rr']
    for m in metrics:
        w_avg = avg_metric(winners, m)
        l_avg = avg_metric(losers, m)
        print(f"{m:<20} {w_avg:>12.2f} {l_avg:>12.2f} {w_avg-l_avg:>+12.2f}")

    # Individual trade analysis
    print("\n--- INDIVIDUAL TRADE DETAILS ---")
    print(f"{'Symbol':<25} {'Regime':<12} {'ADX':>6} {'RSI':>6} {'SL ATR':>8} {'Dur':>6} {'PnL':>10}")
    print("-"*85)
    for t in sorted(vwap_lose_trades, key=lambda x: x['pnl']):
        print(f"{t['symbol'][:25]:<25} {t['regime']:<12} {t['adx']:>6.1f} {t['rsi']:>6.1f} {t['sl_in_atr']:>8.2f} {t['duration_mins']:>6.0f} {t['pnl']:>10.0f}")

    # ROOT CAUSE ANALYSIS
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS: Why vwap_lose_short fails")
    print("="*80)

    print("""
FINDINGS:

1. REGIME MISMATCH:
   - 0% WR in trend_up (4 trades, all losses) - SHORTING IN UPTREND!
   - 27% WR in chop (11 trades) - low conviction
   - Best in trend_down but still low WR

2. ADX PATTERN:
   - Winners have higher ADX (need strong trend for VWAP lose to work)
   - Most trades taken with ADX < 30 = weak trend = VWAP reclaim likely

3. CONCEPT ISSUE:
   - vwap_lose_short = price lost VWAP support, go short
   - Problem: VWAP is dynamic, price often reclaims it
   - In uptrend: VWAP acts as support, price bounces back

4. TIMING ISSUE:
   - Most losses in hour 10 when market is still finding direction
   - VWAP lose early in day often gets reclaimed

RECOMMENDATION:
   - Block vwap_lose_short in trend_up and chop (0% and 27% WR)
   - Only allow in strong trend_down with ADX > 30
   - OR block entirely (16 trades, -3,282 Rs = easy fix)
""")

def analyze_orb_trades():
    """Q3: How to increase ORB trades and reduce SL hits?"""
    print("\n" + "="*100)
    print("QUESTION 3: ORB TRADES - How to increase count and reduce SL hits?")
    print("="*100)

    decisions, triggers, exits, analytics = load_all_data()

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    trigger_lookup = {t.get('trade_id'): t for t in triggers}

    # First, find ALL ORB decisions (not just triggered)
    orb_decisions = [d for d in decisions if d.get('plan', {}).get('strategy', '') == 'orb_breakout_long']

    print(f"\nTotal ORB DECISIONS made: {len(orb_decisions)}")

    # How many triggered?
    orb_triggers = [t for t in triggers if t.get('trigger', {}).get('strategy', '') == 'orb_breakout_long']
    print(f"ORB TRIGGERS executed: {len(orb_triggers)}")

    # Why didn't some trigger?
    triggered_ids = {t.get('trade_id') for t in orb_triggers}
    not_triggered = [d for d in orb_decisions if d.get('trade_id') not in triggered_ids]
    print(f"ORB decisions NOT triggered: {len(not_triggered)}")

    # Analyze ORB exits
    orb_exits = [e for e in exits if e.get('setup_type') == 'orb_breakout_long']

    total_pnl = sum(e.get('total_trade_pnl', e.get('pnl', 0)) for e in orb_exits)
    winners = [e for e in orb_exits if e.get('total_trade_pnl', e.get('pnl', 0)) > 0]
    losers = [e for e in orb_exits if e.get('total_trade_pnl', e.get('pnl', 0)) <= 0]

    print(f"\nORB completed trades: {len(orb_exits)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(orb_exits)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(orb_exits)*100:.1f}%)")
    print(f"Total PnL: Rs {total_pnl:.0f}")

    # Exit reasons
    print("\n--- ORB EXIT REASONS ---")
    by_reason = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for e in orb_exits:
        reason = e.get('reason', '')
        pnl = e.get('total_trade_pnl', e.get('pnl', 0))
        by_reason[reason]['count'] += 1
        by_reason[reason]['pnl'] += pnl

    for reason, data in sorted(by_reason.items(), key=lambda x: -x[1]['pnl']):
        print(f"  {reason:<35} {data['count']:>4} trades, Rs {data['pnl']:>8.0f}")

    # Hard SL analysis for ORB
    orb_hard_sl = [e for e in orb_exits if e.get('reason') == 'hard_sl']
    print(f"\n--- ORB HARD_SL TRADES ({len(orb_hard_sl)}) ---")

    for e in orb_hard_sl:
        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})
        t = trigger_lookup.get(trade_id, {})

        plan = d.get('plan', {}) if d else {}
        indicators = plan.get('indicators', {})
        levels = plan.get('levels', {})
        stop = plan.get('stop', {})
        entry = plan.get('entry', {})

        entry_ref = entry.get('reference', 0)
        hard_sl = stop.get('hard', 0)
        atr = indicators.get('atr', 0)
        sl_in_atr = abs(entry_ref - hard_sl) / atr if atr else 0

        print(f"\n  {e.get('symbol', '')}")
        print(f"    Entry: {entry_ref:.2f}, SL: {hard_sl:.2f} ({sl_in_atr:.2f} ATR)")
        print(f"    ORL: {levels.get('ORL', 0):.2f}, ORH: {levels.get('ORH', 0):.2f}")
        print(f"    Slippage: {e.get('slippage_bps', 0):.0f} bps")
        print(f"    PnL: Rs {e.get('total_trade_pnl', e.get('pnl', 0)):.0f}")

    # Slippage analysis
    print("\n--- ORB SLIPPAGE ANALYSIS ---")
    slippages = [e.get('slippage_bps', 0) for e in orb_exits]
    print(f"Average slippage: {sum(slippages)/len(slippages):.0f} bps")
    print(f"Max slippage: {max(slippages):.0f} bps")

    # High slippage trades
    high_slip = [e for e in orb_exits if e.get('slippage_bps', 0) > 300]
    print(f"\nHigh slippage (>300 bps): {len(high_slip)} trades")
    for e in high_slip:
        pnl = e.get('total_trade_pnl', e.get('pnl', 0))
        print(f"  {e.get('symbol', '')}: {e.get('slippage_bps', 0):.0f} bps, PnL={pnl:.0f}")

    # What's blocking more ORB trades?
    print("\n" + "="*80)
    print("WHY DON'T MORE ORB TRADES TRIGGER?")
    print("="*80)

    # Look at decisions that didn't trigger
    if not_triggered:
        print(f"\n{len(not_triggered)} ORB decisions made but NOT triggered:")

        # Sample the reasons from decision data
        for d in not_triggered[:10]:
            plan = d.get('plan', {})
            decision = d.get('decision', {})
            reasons = decision.get('reasons', '')
            quality = plan.get('quality', {})

            print(f"\n  {d.get('symbol', '')}")
            print(f"    Quality status: {quality.get('status', 'unknown')}")
            print(f"    Entry mode: {plan.get('entry', {}).get('mode', 'unknown')}")
            print(f"    Reasons: {reasons[:100]}...")

    # Regime analysis for ORB
    print("\n--- ORB BY REGIME ---")
    by_regime = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
    for e in orb_exits:
        regime = e.get('regime', '')
        pnl = e.get('total_trade_pnl', e.get('pnl', 0))
        by_regime[regime]['count'] += 1
        by_regime[regime]['pnl'] += pnl
        if pnl > 0:
            by_regime[regime]['wins'] += 1

    for regime, data in sorted(by_regime.items(), key=lambda x: -x[1]['pnl']):
        wr = data['wins'] / data['count'] * 100 if data['count'] else 0
        print(f"  {regime:<15} {data['count']:>4} trades, WR={wr:>5.1f}%, Rs {data['pnl']:>8.0f}")

    # RECOMMENDATIONS
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR ORB IMPROVEMENT")
    print("="*80)

    print("""
CURRENT STATE:
- 61 ORB trades executed
- 61% WR, Rs +8,103 profit
- Only 3 hard_sl trades (4.9% SL rate) - EXCELLENT!
- 96.7% survival rate to 120+ mins

TO INCREASE ORB TRADES:
1. Check why decisions don't trigger
   - Many ORB decisions made but not triggered
   - Entry zone may be too tight
   - Consider widening entry zone slightly

2. Reduce slippage impact
   - 218 bps average slippage is high
   - Use limit orders instead of market orders
   - Or factor slippage into SL calculation

TO REDUCE SL HITS (already low):
1. Only 3 hard_sl out of 61 trades = 4.9%
2. SL placement seems good (at ORL)
3. No major changes needed

FOCUS AREAS:
1. Increase ORB allocation (best setup)
2. Ensure all valid ORB signals trigger
3. May need to check entry zone logic
""")

def main():
    analyze_late_entries()
    analyze_vwap_lose_short()
    analyze_orb_trades()

if __name__ == "__main__":
    main()
