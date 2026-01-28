#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Analysis: Entry Timing Analysis for hard_sl trades

For each hard_sl trade, analyze the 1m bars to determine:
1. Did we enter at the worst possible price (chase)?
2. Was there a better entry opportunity within 5-15 minutes?
3. Would waiting for a pullback/retest have worked?
4. What was the optimal entry price that day?
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

BACKTEST_DIR = Path("backtest_20251106-165927_extracted/20251106-165927_full/20251106-165927")
CACHE_DIR = Path("cache/ohlcv_archive")

def load_1m_data(symbol, date_str):
    """Load 1m bar data from cache for a specific date."""
    cache_symbol = symbol.replace("NSE:", "") + ".NS"
    cache_file = CACHE_DIR / cache_symbol / f"{cache_symbol}_1minutes.feather"

    if not cache_file.exists():
        return None

    try:
        df = pd.read_feather(cache_file)
        if 'date' not in df.columns:
            return None

        df['timestamp'] = pd.to_datetime(df['date'])
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

                    if event_type == 'DECISION':
                        plan_data = event.get('plan', {})
                    elif event_type == 'TRIGGER':
                        entry_time = event.get('ts')
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

def analyze_entry_timing(df, actual_entry_price, actual_entry_time, side, initial_sl, t1_target, t2_target):
    """
    Analyze if entry timing was optimal.

    Returns dict with:
    - entry_quality: "CHASE" | "GOOD" | "EXCELLENT"
    - better_entry_available: bool
    - better_entry_price: float or None
    - better_entry_time: timestamp or None
    - price_vs_range: percentile (0-100) where we entered
    """

    try:
        entry_dt = pd.to_datetime(actual_entry_time)
        if entry_dt.tz is None and df['timestamp'].dt.tz is not None:
            entry_dt = entry_dt.tz_localize('Asia/Kolkata')
    except:
        return None

    # Get bars before and after entry
    bars_before = df[df['timestamp'] < entry_dt].copy()
    bars_after = df[df['timestamp'] >= entry_dt].copy()

    if len(bars_before) < 5 or len(bars_after) < 5:
        return None

    # Analysis window: 15 minutes before entry
    lookback_window = entry_dt - timedelta(minutes=15)
    recent_bars = df[(df['timestamp'] >= lookback_window) & (df['timestamp'] < entry_dt)].copy()

    if len(recent_bars) == 0:
        return None

    result = {
        'actual_entry_price': actual_entry_price,
        'actual_entry_time': str(entry_dt),
        'entry_quality': 'UNKNOWN',
        'better_entry_available': False,
        'better_entry_price': None,
        'better_entry_time': None,
        'price_vs_range_pct': 0,
        'chase_analysis': '',
        'optimal_entry_analysis': '',
    }

    # Determine if we chased
    if side in ['BUY', 'long']:
        recent_high = recent_bars['high'].max()
        recent_low = recent_bars['low'].min()
        recent_range = recent_high - recent_low

        if recent_range > 0:
            # Where in the range did we enter? (0 = low, 100 = high)
            price_position = ((actual_entry_price - recent_low) / recent_range) * 100
            result['price_vs_range_pct'] = price_position

            # LONG: Entering near high (>80%) = CHASE
            if price_position >= 80:
                result['entry_quality'] = 'CHASE'
                result['chase_analysis'] = f"Entered at {price_position:.0f}% of 15m range (near high)"
            elif price_position <= 40:
                result['entry_quality'] = 'EXCELLENT'
                result['chase_analysis'] = f"Entered at {price_position:.0f}% of 15m range (near low)"
            else:
                result['entry_quality'] = 'GOOD'
                result['chase_analysis'] = f"Entered at {price_position:.0f}% of 15m range (mid)"

        # Check if better entry was available (pullback)
        # Look for lowest price within 5 minutes before entry
        last_5min = df[(df['timestamp'] >= entry_dt - timedelta(minutes=5)) &
                       (df['timestamp'] < entry_dt)].copy()

        if len(last_5min) > 0:
            best_price = last_5min['low'].min()
            best_time_idx = last_5min['low'].idxmin()
            best_time = last_5min.loc[best_time_idx, 'timestamp']

            # If best price was >0.3% better, we chased
            if (actual_entry_price - best_price) / actual_entry_price >= 0.003:
                result['better_entry_available'] = True
                result['better_entry_price'] = best_price
                result['better_entry_time'] = str(best_time)

    else:  # SHORT
        recent_high = recent_bars['high'].max()
        recent_low = recent_bars['low'].min()
        recent_range = recent_high - recent_low

        if recent_range > 0:
            # Where in the range did we enter? (0 = low, 100 = high)
            price_position = ((actual_entry_price - recent_low) / recent_range) * 100
            result['price_vs_range_pct'] = price_position

            # SHORT: Entering near low (<20%) = CHASE
            if price_position <= 20:
                result['entry_quality'] = 'CHASE'
                result['chase_analysis'] = f"Entered at {price_position:.0f}% of 15m range (near low - chasing short)"
            elif price_position >= 60:
                result['entry_quality'] = 'EXCELLENT'
                result['chase_analysis'] = f"Entered at {price_position:.0f}% of 15m range (near high)"
            else:
                result['entry_quality'] = 'GOOD'
                result['chase_analysis'] = f"Entered at {price_position:.0f}% of 15m range (mid)"

        # Check if better entry was available (bounce to better short price)
        last_5min = df[(df['timestamp'] >= entry_dt - timedelta(minutes=5)) &
                       (df['timestamp'] < entry_dt)].copy()

        if len(last_5min) > 0:
            best_price = last_5min['high'].max()
            best_time_idx = last_5min['high'].idxmax()
            best_time = last_5min.loc[best_time_idx, 'timestamp']

            # If best price was >0.3% better, we chased
            if (best_price - actual_entry_price) / actual_entry_price >= 0.003:
                result['better_entry_available'] = True
                result['better_entry_price'] = best_price
                result['better_entry_time'] = str(best_time)

    # Find optimal entry that would have hit targets
    optimal = find_optimal_entry(df, entry_dt, side, initial_sl, t1_target, t2_target)
    if optimal:
        result['optimal_entry_analysis'] = optimal

    return result

def find_optimal_entry(df, decision_time, side, initial_sl, t1_target, t2_target):
    """
    Find the optimal entry price and time after decision that would have:
    1. Not hit initial SL
    2. Hit at least T1
    """

    try:
        decision_dt = pd.to_datetime(decision_time)
        if decision_dt.tz is None and df['timestamp'].dt.tz is not None:
            decision_dt = decision_dt.tz_localize('Asia/Kolkata')
    except:
        return None

    # Look at bars from decision time to 45 minutes later (typical entry window)
    end_time = decision_dt + timedelta(minutes=45)
    window = df[(df['timestamp'] >= decision_dt) & (df['timestamp'] <= end_time)].copy()

    if len(window) == 0:
        return None

    # For each potential entry bar, simulate the trade
    best_entries = []

    for idx, bar in window.iterrows():
        entry_time = bar['timestamp']

        # Try entering at different points in the bar
        if side in ['BUY', 'long']:
            # Try entering at low (best price)
            entry_price = bar['low']

            # Simulate from this entry
            future_bars = df[df['timestamp'] > entry_time].copy()

            if len(future_bars) == 0:
                continue

            hit_sl = False
            hit_t1 = False
            hit_t2 = False

            for _, future_bar in future_bars.iterrows():
                # Check SL
                if future_bar['low'] <= initial_sl:
                    hit_sl = True
                    break

                # Check T1
                if future_bar['high'] >= t1_target:
                    hit_t1 = True

                # Check T2
                if future_bar['high'] >= t2_target:
                    hit_t2 = True
                    break

                # Stop checking after 2 hours
                if (future_bar['timestamp'] - entry_time).total_seconds() > 7200:
                    break

            # Valid entry: didn't hit SL and hit at least T1
            if not hit_sl and hit_t1:
                best_entries.append({
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'hit_t2': hit_t2,
                    'minutes_after_decision': (entry_time - decision_dt).total_seconds() / 60
                })

        else:  # SHORT
            entry_price = bar['high']

            future_bars = df[df['timestamp'] > entry_time].copy()

            if len(future_bars) == 0:
                continue

            hit_sl = False
            hit_t1 = False
            hit_t2 = False

            for _, future_bar in future_bars.iterrows():
                # Check SL
                if future_bar['high'] >= initial_sl:
                    hit_sl = True
                    break

                # Check T1
                if future_bar['low'] <= t1_target:
                    hit_t1 = True

                # Check T2
                if future_bar['low'] <= t2_target:
                    hit_t2 = True
                    break

                if (future_bar['timestamp'] - entry_time).total_seconds() > 7200:
                    break

            if not hit_sl and hit_t1:
                best_entries.append({
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'hit_t2': hit_t2,
                    'minutes_after_decision': (entry_time - decision_dt).total_seconds() / 60
                })

    if len(best_entries) == 0:
        return "No optimal entry found - trade was fundamentally bad"

    # Return the earliest good entry
    best = min(best_entries, key=lambda x: x['minutes_after_decision'])

    return f"Optimal entry @ {best['entry_price']:.2f} at {best['entry_time'].strftime('%H:%M')} " \
           f"({best['minutes_after_decision']:.0f}m after decision) → " \
           f"{'T2' if best['hit_t2'] else 'T1'} hit"

def main():
    print("="*120)
    print("PHASE 2 ANALYSIS: ENTRY TIMING FOR HARD_SL TRADES")
    print("="*120)
    print()

    # Collect all hard_sl trades
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

    for i, trade in enumerate(hard_sl_trades, 1):
        date = trade['date']
        symbol = trade['symbol']
        session_dir = trade['session_dir']

        print(f"[{i}/{len(hard_sl_trades)}] Analyzing {symbol} on {date}...", end=" ")

        # Load 1m data
        df = load_1m_data(symbol, date)

        if df is None or len(df) == 0:
            print("NO CACHE DATA")
            continue

        # Load trade details
        trade_data = load_trade_from_events(session_dir, symbol)

        if not trade_data:
            print("NO EVENTS DATA")
            continue

        plan = trade_data['plan']
        entry_time = trade_data['entry_time']

        # Extract parameters
        entry_price = float(plan.get('price', plan.get('entry', {}).get('reference', 0)))
        side = plan.get('bias', 'long')
        initial_sl = float(plan.get('stop', {}).get('hard', 0))

        targets = plan.get('targets', [])
        t1_target = float(targets[0].get('level', 0)) if len(targets) > 0 else 0
        t2_target = float(targets[1].get('level', 0)) if len(targets) > 1 else 0

        if entry_price == 0 or initial_sl == 0 or t1_target == 0:
            print("INCOMPLETE PLAN DATA")
            continue

        # Analyze entry timing
        timing_analysis = analyze_entry_timing(
            df, entry_price, entry_time, side, initial_sl, t1_target, t2_target
        )

        if timing_analysis is None:
            print("ANALYSIS FAILED")
            continue

        print(f"{timing_analysis['entry_quality']} - {timing_analysis['chase_analysis']}")

        results.append({
            'date': date,
            'symbol': symbol,
            'side': side,
            **timing_analysis
        })

    print()
    print("="*120)
    print("RESULTS SUMMARY")
    print("="*120)
    print()

    if len(results) == 0:
        print("No results to analyze.")
        return

    # Categorize by entry quality
    chase_trades = [r for r in results if r['entry_quality'] == 'CHASE']
    good_trades = [r for r in results if r['entry_quality'] == 'GOOD']
    excellent_trades = [r for r in results if r['entry_quality'] == 'EXCELLENT']

    print(f"Total analyzed: {len(results)}")
    print(f"CHASE entries (poor timing): {len(chase_trades)} ({len(chase_trades)/len(results)*100:.1f}%)")
    print(f"GOOD entries: {len(good_trades)} ({len(good_trades)/len(results)*100:.1f}%)")
    print(f"EXCELLENT entries: {len(excellent_trades)} ({len(excellent_trades)/len(results)*100:.1f}%)")
    print()

    # Better entry available
    better_available = [r for r in results if r['better_entry_available']]
    print(f"Better entry was available within 5min: {len(better_available)} ({len(better_available)/len(results)*100:.1f}%)")
    print()

    # Show CHASE examples
    if len(chase_trades) > 0:
        print("="*120)
        print("CHASE TRADES (entered at worst price in range)")
        print("="*120)
        print()
        print(f"{'Date':<12} {'Symbol':<20} {'Side':<6} {'Position in Range':<20} {'Better Entry?':<15}")
        print("-"*120)

        for r in chase_trades[:15]:
            better = "YES" if r['better_entry_available'] else "NO"
            print(f"{r['date']:<12} {r['symbol']:<20} {r['side']:<6} {r['price_vs_range_pct']:>6.0f}% {better:<15}")

        print()

    # Show trades with optimal entries
    trades_with_optimal = [r for r in results if r.get('optimal_entry_analysis') and
                           'No optimal entry' not in r['optimal_entry_analysis']]

    if len(trades_with_optimal) > 0:
        print("="*120)
        print("TRADES WITH OPTIMAL ENTRY OPPORTUNITY")
        print("="*120)
        print()
        print(f"{'Date':<12} {'Symbol':<20} {'Optimal Entry Analysis':<80}")
        print("-"*120)

        for r in trades_with_optimal[:15]:
            print(f"{r['date']:<12} {r['symbol']:<20} {r['optimal_entry_analysis']:<80}")

        print()

    # Final verdict
    print("="*120)
    print("VERDICT")
    print("="*120)
    print()

    chase_rate = len(chase_trades) / len(results) * 100
    better_rate = len(better_available) / len(results) * 100
    optimal_rate = len(trades_with_optimal) / len(results) * 100

    if chase_rate >= 50:
        verdict = "ENTRY TIMING IS CRITICAL PROBLEM"
        action = "IMPLEMENT PATIENCE - waiting 5-15min for pullback would save 50%+ trades"
    elif chase_rate >= 30:
        verdict = "ENTRY TIMING IS SIGNIFICANT PROBLEM"
        action = "IMPROVE ENTRY LOGIC - too much chasing happening"
    elif optimal_rate >= 40:
        verdict = "TRADES ARE VIABLE BUT ENTRY TIMING OFF"
        action = "Better entry discipline could save 40%+ trades"
    else:
        verdict = "ENTRY TIMING NOT THE MAIN ISSUE"
        action = "Look at setup quality and trade selection instead"

    print(f"{verdict}")
    print(f"→ {chase_rate:.1f}% of trades CHASED price (entered at worst point)")
    print(f"→ {better_rate:.1f}% had better entry within 5 minutes")
    print(f"→ {optimal_rate:.1f}% had winning entry opportunity available")
    print()
    print(f"RECOMMENDATION: {action}")
    print()

    # P&L impact estimate
    # If we fixed entry timing on chase trades + trades with better entries
    fixable_count = len(set([r['symbol'] + r['date'] for r in chase_trades] +
                            [r['symbol'] + r['date'] for r in better_available]))

    hard_sl_avg_loss = 493.75
    avoided_losses = fixable_count * hard_sl_avg_loss
    expected_gains = fixable_count * 200  # Conservative 1R average

    net_improvement = avoided_losses + expected_gains

    print(f"P&L IMPACT ESTIMATE (if entry timing fixed):")
    print(f"  Fixable trades: {fixable_count}")
    print(f"  Avoided losses: Rs.{avoided_losses:.2f}")
    print(f"  Expected gains: Rs.{expected_gains:.2f}")
    print(f"  NET IMPROVEMENT: Rs.{net_improvement:.2f}")
    print()

    # Save results
    output_file = Path("phase2_entry_timing_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_analyzed': len(results),
                'chase_trades': len(chase_trades),
                'chase_rate_pct': chase_rate,
                'better_entry_available': len(better_available),
                'trades_with_optimal': len(trades_with_optimal),
                'estimated_pnl_improvement': net_improvement,
            },
            'trades': results
        }, f, indent=2, default=str)

    print(f"Detailed results saved to: {output_file}")
    print()
    print("="*120)

if __name__ == "__main__":
    main()
