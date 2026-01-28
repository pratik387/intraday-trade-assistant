#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Analysis: Would stop widening save hard_sl trades?

Uses events.jsonl for plan data and 1m cache for price movement analysis.
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

BACKTEST_DIR = Path("backtest_20251106-165927_extracted/20251106-165927_full/20251106-165927")
CACHE_DIR = Path("cache/ohlcv_archive")

def load_1m_data(symbol, date_str):
    """Load 1m bar data from cache for a specific date."""
    # NSE:SYMBOL format -> SYMBOL.NS
    cache_symbol = symbol.replace("NSE:", "") + ".NS"

    # Cache structure: cache/ohlcv_archive/SYMBOL.NS/SYMBOL.NS_1minutes.feather
    cache_file = CACHE_DIR / cache_symbol / f"{cache_symbol}_1minutes.feather"

    if not cache_file.exists():
        return None

    try:
        df = pd.read_feather(cache_file)

        # Column is 'date' with timezone-aware timestamps
        if 'date' not in df.columns:
            return None

        df['timestamp'] = pd.to_datetime(df['date'])

        # Filter to specific date
        target_date = pd.to_datetime(date_str).date()
        df['date_only'] = df['timestamp'].dt.date
        df = df[df['date_only'] == target_date].copy()

        if len(df) > 0:
            return df.sort_values('timestamp')

    except Exception as e:
        return None

    return None

def load_trade_from_events(session_dir, symbol):
    """Load trade plan and timestamps from events.jsonl"""
    events_file = session_dir / 'events.jsonl'

    if not events_file.exists():
        return None

    plan_data = None
    entry_time = None
    exit_time = None

    with open(events_file, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    event = json.loads(line)

                    if event.get('symbol') != symbol:
                        continue

                    event_type = event.get('type')

                    # Get plan from DECISION event
                    if event_type == 'DECISION':
                        plan_data = event.get('plan', {})

                    # Get entry time from TRIGGER event
                    elif event_type == 'TRIGGER':
                        entry_time = event.get('ts')

                    # Get exit time from EXIT event with hard_sl reason
                    elif event_type == 'EXIT':
                        exit_event = event.get('exit', {})
                        if exit_event.get('reason') == 'hard_sl':
                            exit_time = event.get('ts')

                except:
                    continue

    if not plan_data or not entry_time:
        return None

    return {
        'plan': plan_data,
        'entry_time': entry_time,
        'exit_time': exit_time
    }

def simulate_widened_stop(df, entry_price, entry_time_str, initial_sl, widened_sl, side, t1_target, t2_target):
    """
    Simulate trade with widened stop logic.
    """
    # Parse entry time
    try:
        entry_dt = pd.to_datetime(entry_time_str)
        # Make timezone-aware to match df timestamps (IST = UTC+5:30)
        if entry_dt.tz is None and df['timestamp'].dt.tz is not None:
            entry_dt = entry_dt.tz_localize('Asia/Kolkata')
    except:
        return None

    # Filter bars after entry
    df_after_entry = df[df['timestamp'] >= entry_dt].copy()

    if len(df_after_entry) == 0:
        return None

    result = {
        'hit_initial_sl': False,
        'hit_widened_sl': False,
        'hit_t1': False,
        'hit_t2': False,
        'time_to_initial_sl': None,
        'time_to_widened_sl': None,
        'time_to_t1': None,
        'time_to_t2': None,
    }

    for idx, bar in df_after_entry.iterrows():
        low = bar['low']
        high = bar['high']
        ts = bar['timestamp']

        if side == 'BUY' or side == 'long':
            # Check initial SL hit
            if not result['hit_initial_sl'] and low <= initial_sl:
                result['hit_initial_sl'] = True
                result['time_to_initial_sl'] = ts

            # Check widened SL hit
            if not result['hit_widened_sl'] and low <= widened_sl:
                result['hit_widened_sl'] = True
                result['time_to_widened_sl'] = ts

            # Check T1 hit
            if not result['hit_t1'] and high >= t1_target:
                result['hit_t1'] = True
                result['time_to_t1'] = ts

            # Check T2 hit
            if not result['hit_t2'] and high >= t2_target:
                result['hit_t2'] = True
                result['time_to_t2'] = ts

        else:  # SELL / short
            # Check initial SL hit
            if not result['hit_initial_sl'] and high >= initial_sl:
                result['hit_initial_sl'] = True
                result['time_to_initial_sl'] = ts

            # Check widened SL hit
            if not result['hit_widened_sl'] and high >= widened_sl:
                result['hit_widened_sl'] = True
                result['time_to_widened_sl'] = ts

            # Check T1 hit
            if not result['hit_t1'] and low <= t1_target:
                result['hit_t1'] = True
                result['time_to_t1'] = ts

            # Check T2 hit
            if not result['hit_t2'] and low <= t2_target:
                result['hit_t2'] = True
                result['time_to_t2'] = ts

    return result

def main():
    print("="*120)
    print("PHASE 1 ANALYSIS: STOP WIDENING IMPACT ON HARD_SL TRADES")
    print("="*120)
    print()

    # Collect all hard_sl trades from analytics.jsonl
    hard_sl_trades = []

    session_dirs = sorted(BACKTEST_DIR.glob('20*'))

    for session_dir in session_dirs:
        analytics_file = session_dir / 'analytics.jsonl'

        if not analytics_file.exists():
            continue

        with open(analytics_file, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get('stage') == 'EXIT' and data.get('reason') == 'hard_sl':
                            hard_sl_trades.append({
                                'date': session_dir.name,
                                'symbol': data.get('symbol'),
                                'session_dir': session_dir,
                            })
                    except:
                        pass

    print(f"Found {len(hard_sl_trades)} hard_sl trades to analyze")
    print()

    # Analyze each trade
    results = []
    no_data_count = 0
    no_events_count = 0

    for i, trade in enumerate(hard_sl_trades, 1):
        date = trade['date']
        symbol = trade['symbol']
        session_dir = trade['session_dir']

        print(f"[{i}/{len(hard_sl_trades)}] Analyzing {symbol} on {date}...", end=" ")

        # Load 1m data
        df = load_1m_data(symbol, date)

        if df is None or len(df) == 0:
            print("NO CACHE DATA")
            no_data_count += 1
            continue

        # Load trade details from events
        trade_data = load_trade_from_events(session_dir, symbol)

        if not trade_data:
            print("NO EVENTS DATA")
            no_events_count += 1
            continue

        plan = trade_data['plan']
        entry_time = trade_data['entry_time']

        # Extract parameters from plan
        entry_price = float(plan.get('price', plan.get('entry', {}).get('reference', 0)))
        side = plan.get('bias', 'long')

        initial_sl = float(plan.get('stop', {}).get('hard', 0))

        targets = plan.get('targets', [])
        t1_target = float(targets[0].get('level', 0)) if len(targets) > 0 else 0
        t2_target = float(targets[1].get('level', 0)) if len(targets) > 1 else 0

        if entry_price == 0 or initial_sl == 0 or t1_target == 0:
            print("INCOMPLETE PLAN DATA")
            continue

        # Calculate widened SL (50% more protection)
        # This simulates Phase 1 widening: 1.5× ATR → 2.25× ATR
        sl_distance = abs(entry_price - initial_sl)
        widened_sl_distance = sl_distance * 1.5

        if side in ['BUY', 'long']:
            widened_sl = entry_price - widened_sl_distance
        else:
            widened_sl = entry_price + widened_sl_distance

        # Simulate both scenarios
        sim_result = simulate_widened_stop(
            df, entry_price, entry_time, initial_sl, widened_sl, side, t1_target, t2_target
        )

        if sim_result is None:
            print("SIMULATION FAILED")
            continue

        # Determine outcome
        initial_hit = sim_result['hit_initial_sl']
        widened_hit = sim_result['hit_widened_sl']
        t1_hit = sim_result['hit_t1']
        t2_hit = sim_result['hit_t2']

        # Key question: If we widened, would we have survived to hit targets?
        saved = False
        outcome = ""

        if initial_hit and not widened_hit:
            # Widening saved us from SL
            if t1_hit:
                saved = True
                if t2_hit:
                    outcome = "SAVED → T2 HIT"
                else:
                    outcome = "SAVED → T1 HIT"
            else:
                outcome = "SAVED but NO TARGETS"
        elif initial_hit and widened_hit:
            # Both hit - check order
            if t1_hit:
                # Check timing
                if sim_result['time_to_t1'] < sim_result['time_to_widened_sl']:
                    saved = True
                    outcome = "SAVED → T1 BEFORE WIDENED SL"
                else:
                    outcome = "WIDENED SL HIT (T1 too late)"
            else:
                outcome = "WIDENED SL ALSO HIT"
        else:
            outcome = "NO INITIAL SL HIT??"

        print(outcome)

        results.append({
            'date': date,
            'symbol': symbol,
            'entry': entry_price,
            'side': side,
            'initial_sl': initial_sl,
            'widened_sl': widened_sl,
            'sl_distance': sl_distance,
            'initial_hit': initial_hit,
            'widened_hit': widened_hit,
            't1_hit': t1_hit,
            't2_hit': t2_hit,
            'saved': saved,
            'outcome': outcome,
            'time_to_initial_sl': str(sim_result['time_to_initial_sl']) if sim_result['time_to_initial_sl'] else None,
            'time_to_widened_sl': str(sim_result['time_to_widened_sl']) if sim_result['time_to_widened_sl'] else None,
            'time_to_t1': str(sim_result['time_to_t1']) if sim_result['time_to_t1'] else None,
            'time_to_t2': str(sim_result['time_to_t2']) if sim_result['time_to_t2'] else None,
        })

    print()
    print("="*120)
    print("RESULTS SUMMARY")
    print("="*120)
    print()

    print(f"Total hard_sl trades: {len(hard_sl_trades)}")
    print(f"Successfully analyzed: {len(results)}")
    print(f"No cache data: {no_data_count}")
    print(f"No events data: {no_events_count}")
    print()

    if len(results) == 0:
        print("No results to analyze.")
        return

    # Calculate statistics
    saved_trades = [r for r in results if r['saved']]
    saved_to_t1 = [r for r in saved_trades if r['t1_hit'] and not r['t2_hit']]
    saved_to_t2 = [r for r in saved_trades if r['t2_hit']]
    not_saved = [r for r in results if not r['saved']]

    print(f"WIDENING WOULD SAVE: {len(saved_trades)} trades ({len(saved_trades)/len(results)*100:.1f}%)")
    print(f"  → T1 only: {len(saved_to_t1)} trades")
    print(f"  → T2: {len(saved_to_t2)} trades")
    print()
    print(f"STILL STOPPED OUT: {len(not_saved)} trades ({len(not_saved)/len(results)*100:.1f}%)")
    print()

    # Show examples
    if len(saved_trades) > 0:
        print("="*120)
        print("SAVED TRADES (would have hit targets with widened stop)")
        print("="*120)
        print()
        print(f"{'Date':<12} {'Symbol':<20} {'Side':<6} {'Entry':>10} {'Initial SL':>12} {'Widened SL':>12} {'Outcome':<30}")
        print("-"*120)

        for r in saved_trades[:15]:
            print(f"{r['date']:<12} {r['symbol']:<20} {r['side']:<6} {r['entry']:>10.2f} {r['initial_sl']:>12.2f} {r['widened_sl']:>12.2f} {r['outcome']:<30}")

        print()

    if len(not_saved) > 0 and len(not_saved) <= 20:
        print("="*120)
        print("TRADES STILL STOPPED OUT (widening wouldn't help)")
        print("="*120)
        print()
        print(f"{'Date':<12} {'Symbol':<20} {'Outcome':<50}")
        print("-"*120)

        for r in not_saved:
            print(f"{r['date']:<12} {r['symbol']:<20} {r['outcome']:<50}")

        print()

    # Final verdict
    print("="*120)
    print("VERDICT")
    print("="*120)
    print()

    save_rate = len(saved_trades) / len(results) * 100 if len(results) > 0 else 0

    if save_rate >= 50:
        verdict = "STOP WIDENING IS CRITICAL"
        action = "IMPLEMENT IMMEDIATELY - Would save 50%+ of hard_sl trades"
    elif save_rate >= 30:
        verdict = "STOP WIDENING HELPS SIGNIFICANTLY"
        action = "IMPLEMENT - Would save 30-50% of hard_sl trades"
    elif save_rate >= 15:
        verdict = "STOP WIDENING HAS MODERATE IMPACT"
        action = "Consider implementing - Would save 15-30% of trades"
    else:
        verdict = "STOP WIDENING HAS LIMITED IMPACT"
        action = "Look at entry timing and setup quality instead"

    print(f"{verdict}")
    print(f"→ {save_rate:.1f}% of hard_sl trades would be saved with stop widening")
    print()
    print(f"RECOMMENDATION: {action}")
    print()

    # Calculate potential P&L impact
    # Assume: Hard SL average loss = Rs.493.75 (from earlier analysis)
    # Assume: T1 average win = Rs.569.74 × 0.4 (40% partial) = Rs.227.90
    # Assume: T2 average win = Rs.569.74 × 0.8 (80% combined) = Rs.455.79

    hard_sl_avg_loss = 493.75
    saved_count = len(saved_trades)

    # Estimate P&L recovery
    avoided_losses = saved_count * hard_sl_avg_loss

    # Assume saved trades would have average 1R profit (conservative)
    # 1R ≈ Rs.200 average
    expected_gains = saved_count * 200

    net_improvement = avoided_losses + expected_gains

    print(f"P&L IMPACT ESTIMATE:")
    print(f"  Avoided losses: Rs.{avoided_losses:.2f} ({saved_count} trades × Rs.{hard_sl_avg_loss:.2f})")
    print(f"  Expected gains: Rs.{expected_gains:.2f} ({saved_count} trades × Rs.200 avg)")
    print(f"  NET IMPROVEMENT: Rs.{net_improvement:.2f}")
    print()

    # Save detailed results
    output_file = Path("phase1_stop_widening_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_analyzed': len(results),
                'saved_trades': len(saved_trades),
                'save_rate_pct': save_rate,
                'saved_to_t1': len(saved_to_t1),
                'saved_to_t2': len(saved_to_t2),
                'estimated_pnl_improvement': net_improvement,
            },
            'trades': results
        }, f, indent=2, default=str)

    print(f"Detailed results saved to: {output_file}")
    print()
    print("="*120)

if __name__ == "__main__":
    main()
