#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Analysis: Would stop widening save hard_sl trades?

For each hard_sl trade:
1. Load 1m bar data from cache
2. Simulate stop widening (1.5× ATR → 2.25× ATR)
3. Check if widened stop would have been hit
4. If survived, check if targets would have been reached
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, time

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
        print(f"ERROR loading {cache_file}: {e}")
        return None

    return None

def load_trade_details(session_dir, symbol):
    """Load entry details from analytics.jsonl and trade_logs.log"""

    # Get entry details from analytics
    analytics_file = session_dir / 'analytics.jsonl'
    entry_data = None
    exit_data = None

    if analytics_file.exists():
        with open(analytics_file, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('symbol') == symbol:
                        if data.get('stage') == 'ENTRY':
                            entry_data = data
                        elif data.get('stage') == 'EXIT' and data.get('reason') == 'hard_sl':
                            exit_data = data

    # Get timestamps from trade logs
    log_file = session_dir / 'trade_logs.log'
    entry_time = None
    exit_time = None

    if log_file.exists():
        with open(log_file, encoding='utf-8') as f:
            for line in f:
                if symbol in line:
                    # Parse timestamp from log line
                    # Format: 2025-11-06 17:10:28,631 — INFO — ...
                    parts = line.split(' — ')
                    if len(parts) >= 3:
                        timestamp_str = parts[0]

                        if 'TRIGGER_EXEC' in line or 'BUY' in line or 'SELL' in line:
                            if 'EXIT' not in line:
                                entry_time = timestamp_str
                        elif 'EXIT' in line and 'hard_sl' in line:
                            exit_time = timestamp_str

    return entry_data, exit_data, entry_time, exit_time

def simulate_widened_stop(df, entry_price, entry_time_str, initial_sl, widened_sl, side, t1_target, t2_target):
    """
    Simulate trade with widened stop logic.

    Returns:
    - hit_initial_sl: bool
    - hit_widened_sl: bool
    - hit_t1: bool
    - hit_t2: bool
    - time_to_initial_sl: timestamp or None
    - time_to_widened_sl: timestamp or None
    - time_to_t1: timestamp or None
    - time_to_t2: timestamp or None
    """

    # Parse entry time
    try:
        entry_dt = pd.to_datetime(entry_time_str)
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

        if side == 'BUY':
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

        else:  # SELL
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
                    data = json.loads(line)
                    if data.get('stage') == 'EXIT' and data.get('reason') == 'hard_sl':
                        hard_sl_trades.append({
                            'date': session_dir.name,
                            'symbol': data.get('symbol'),
                            'session_dir': session_dir,
                        })

    print(f"Found {len(hard_sl_trades)} hard_sl trades to analyze")
    print()

    # Analyze each trade
    results = []
    no_data_count = 0

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

        # Load trade details
        entry_data, exit_data, entry_time, exit_time = load_trade_details(session_dir, symbol)

        if not entry_data or not exit_data or not entry_time:
            print("INCOMPLETE TRADE DATA")
            continue

        # Extract trade parameters
        entry_price = float(entry_data.get('actual_entry_price', entry_data.get('entry_price', 0)))
        side = entry_data.get('side', 'BUY')

        # Get stop and targets from plan
        plan = entry_data.get('plan', {})
        initial_sl = float(plan.get('hard_sl', 0))
        t1_target = float(plan.get('target1', 0))
        t2_target = float(plan.get('target2', 0))

        # Calculate widened SL (2.25× ATR instead of 1.5× ATR)
        # Widening = 50% more protection (2.25/1.5 = 1.5)
        sl_distance = abs(entry_price - initial_sl)
        widened_sl_distance = sl_distance * 1.5  # 50% wider

        if side == 'BUY':
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
                    outcome = "SAVED → T1 HIT BEFORE WIDENED SL"
                else:
                    outcome = "WIDENED SL HIT (but later)"
            else:
                outcome = "WIDENED SL ALSO HIT"
        else:
            outcome = "NO INITIAL SL??"

        print(outcome)

        results.append({
            'date': date,
            'symbol': symbol,
            'entry': entry_price,
            'side': side,
            'initial_sl': initial_sl,
            'widened_sl': widened_sl,
            'initial_hit': initial_hit,
            'widened_hit': widened_hit,
            't1_hit': t1_hit,
            't2_hit': t2_hit,
            'saved': saved,
            'outcome': outcome,
            'time_to_initial_sl': sim_result['time_to_initial_sl'],
            'time_to_widened_sl': sim_result['time_to_widened_sl'],
            'time_to_t1': sim_result['time_to_t1'],
            'time_to_t2': sim_result['time_to_t2'],
        })

    print()
    print("="*120)
    print("RESULTS SUMMARY")
    print("="*120)
    print()

    print(f"Total hard_sl trades: {len(hard_sl_trades)}")
    print(f"Successfully analyzed: {len(results)}")
    print(f"No cache data: {no_data_count}")
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
        print("TOP SAVED TRADES (would have hit targets with widened stop)")
        print("="*120)
        print()
        print(f"{'Date':<12} {'Symbol':<20} {'Side':<6} {'Entry':>10} {'Initial SL':>12} {'Widened SL':>12} {'Outcome':<30}")
        print("-"*120)

        for r in saved_trades[:10]:
            print(f"{r['date']:<12} {r['symbol']:<20} {r['side']:<6} {r['entry']:>10.2f} {r['initial_sl']:>12.2f} {r['widened_sl']:>12.2f} {r['outcome']:<30}")

        print()

    if len(not_saved) > 0:
        print("="*120)
        print("TRADES STILL STOPPED OUT (widening wouldn't help)")
        print("="*120)
        print()
        print(f"{'Date':<12} {'Symbol':<20} {'Outcome':<50}")
        print("-"*120)

        for r in not_saved[:10]:
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
            },
            'trades': results
        }, f, indent=2, default=str)

    print(f"Detailed results saved to: {output_file}")
    print()
    print("="*120)

if __name__ == "__main__":
    main()
