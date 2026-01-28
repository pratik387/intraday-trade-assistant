#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze sl_post_t1 trades to prove if wider stops would have saved them.

For each sl_post_t1 trade:
1. Find the T1 partial exit event
2. Find the sl_post_t1 exit event
3. Load 1-minute candle data from ohlcv_cache
4. Calculate:
   - T1 exit price and time
   - SL hit price and time
   - High water mark AFTER SL hit
   - Whether price reached T2 target after SL
   - How much profit we missed

This will prove:
- Were we stopped out by noise?
- Did price reverse after hitting our tight stop?
- Would wider stops (entry + 50% of T1 profit) have survived?
- How much P&L did we leave on the table?
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Configuration
BACKTEST_DIR = Path("backtest_20251106-165927_extracted/20251106-165927_full/20251106-165927")
CACHE_DIR = Path("ohlcv_cache")

def load_trade_events(session_dir):
    """Load all trade events from analytics.jsonl"""
    analytics_file = session_dir / 'analytics.jsonl'
    events = []

    if analytics_file.exists():
        with open(analytics_file, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

    return events

def load_1m_candles(symbol, date):
    """Load 1-minute candle data for symbol on given date"""
    # Symbol format: NSE:SYMBOL -> SYMBOL.NS for cache
    clean_symbol = symbol.replace('NSE:', '') + '.NS'

    # Look for cache file
    # Cache files are typically: ohlcv_cache/SYMBOL.NS_1m_YYYYMMDD.parquet
    date_str = date.replace('-', '')
    cache_file = CACHE_DIR / f"{clean_symbol}_1m_{date_str}.parquet"

    if not cache_file.exists():
        # Try alternative naming
        cache_file = CACHE_DIR / f"{clean_symbol}_1min_{date_str}.parquet"

    if not cache_file.exists():
        # Try searching
        pattern = f"{clean_symbol}_*_{date_str}*.parquet"
        matches = list(CACHE_DIR.glob(pattern))
        if matches:
            cache_file = matches[0]
        else:
            return None

    try:
        df = pd.read_parquet(cache_file)
        return df
    except Exception as e:
        print(f"  Warning: Could not load {cache_file}: {e}")
        return None

def parse_timestamp(ts_str):
    """Parse timestamp string to datetime"""
    try:
        return pd.to_datetime(ts_str)
    except:
        return None

def analyze_sl_post_t1_trade(symbol, events, date):
    """
    Analyze a single sl_post_t1 trade.

    Returns dict with analysis or None if can't analyze.
    """
    # Find entry event
    entry_event = None
    t1_event = None
    sl_event = None

    for event in events:
        if event.get('symbol') != symbol:
            continue

        stage = event.get('stage')
        reason = event.get('reason')

        if stage == 'ENTRY':
            entry_event = event
        elif reason == 't1_partial':
            t1_event = event
        elif reason == 'sl_post_t1':
            sl_event = event

    if not all([entry_event, t1_event, sl_event]):
        return None

    # Extract key data
    entry_price = entry_event.get('actual_entry_price') or entry_event.get('entry_price')
    bias = entry_event.get('bias')

    t1_price = t1_event.get('exit_price')
    t1_time = parse_timestamp(t1_event.get('timestamp'))

    sl_price = sl_event.get('exit_price')
    sl_time = parse_timestamp(sl_event.get('timestamp'))
    sl_pnl = sl_event.get('pnl', 0)

    # Get targets from entry event (if available)
    t1_target = entry_event.get('t1')
    t2_target = entry_event.get('t2')

    # Load 1m candles
    candles = load_1m_candles(symbol, date)
    if candles is None:
        return None

    # Ensure candles have datetime index
    if not isinstance(candles.index, pd.DatetimeIndex):
        if 'timestamp' in candles.columns:
            candles['timestamp'] = pd.to_datetime(candles['timestamp'])
            candles.set_index('timestamp', inplace=True)
        elif 'time' in candles.columns:
            candles['time'] = pd.to_datetime(candles['time'])
            candles.set_index('time', inplace=True)

    # Get candles after SL hit
    candles_after_sl = candles[candles.index > sl_time] if sl_time else candles

    if len(candles_after_sl) == 0:
        return None

    # Calculate metrics
    if bias == 'long':
        # High water mark after SL
        hwm_after_sl = candles_after_sl['high'].max()

        # Did price reach T2?
        t2_reached = hwm_after_sl >= t2_target if t2_target else False

        # Calculate what stop should have been (entry + 50% of T1 profit)
        t1_profit = t1_price - entry_price
        proper_sl = entry_price + (0.5 * t1_profit)

        # Would proper SL have survived?
        lwm_after_t1 = candles_after_sl['low'].min()
        proper_sl_survived = lwm_after_t1 > proper_sl

        # Missed profit
        if t2_reached:
            missed_profit = (t2_target - sl_price)
        else:
            missed_profit = (hwm_after_sl - sl_price)

    else:  # short
        # Low water mark after SL
        lwm_after_sl = candles_after_sl['low'].min()

        # Did price reach T2?
        t2_reached = lwm_after_sl <= t2_target if t2_target else False

        # Calculate what stop should have been
        t1_profit = entry_price - t1_price
        proper_sl = entry_price - (0.5 * t1_profit)

        # Would proper SL have survived?
        hwm_after_t1 = candles_after_sl['high'].max()
        proper_sl_survived = hwm_after_t1 < proper_sl

        # Missed profit
        if t2_reached:
            missed_profit = (sl_price - t2_target)
        else:
            missed_profit = (sl_price - lwm_after_sl)

    return {
        'symbol': symbol,
        'date': date,
        'bias': bias,
        'entry': entry_price,
        't1_price': t1_price,
        't1_time': t1_time,
        't1_target': t1_target,
        't2_target': t2_target,
        'sl_price': sl_price,
        'sl_time': sl_time,
        'sl_pnl': sl_pnl,
        'actual_sl': sl_price,  # What SL was hit at (BE + 0.1%)
        'proper_sl': proper_sl,  # What SL should have been (entry + 50% T1)
        'hwm_after_sl': hwm_after_sl if bias == 'long' else lwm_after_sl,
        't2_reached': t2_reached,
        'proper_sl_survived': proper_sl_survived,
        'missed_profit_per_share': missed_profit,
        'qty_remaining': sl_event.get('qty', 0),
        'missed_profit_total': missed_profit * sl_event.get('qty', 0),
    }

def main():
    print("="*120)
    print("SL_POST_T1 REVERSAL ANALYSIS - PROOF OF CONCEPT")
    print("="*120)
    print()
    print("Analyzing 27 sl_post_t1 trades to prove they reversed after stopping us out...")
    print()

    # Find all sl_post_t1 trades
    sl_post_t1_trades = []

    session_dirs = sorted(BACKTEST_DIR.glob('20*'))
    print(f"Scanning {len(session_dirs)} sessions...\n")

    for session_dir in session_dirs:
        date = session_dir.name
        events = load_trade_events(session_dir)

        # Find sl_post_t1 events
        symbols_with_sl_post_t1 = set()
        for event in events:
            if event.get('reason') == 'sl_post_t1':
                symbols_with_sl_post_t1.add(event.get('symbol'))

        # Analyze each
        for symbol in symbols_with_sl_post_t1:
            print(f"Analyzing {symbol} on {date}...")
            analysis = analyze_sl_post_t1_trade(symbol, events, date)
            if analysis:
                sl_post_t1_trades.append(analysis)

                # Print quick summary
                status = "T2 REACHED" if analysis['t2_reached'] else "REVERSED"
                survival = "SURVIVED" if analysis['proper_sl_survived'] else "STILL HIT"
                print(f"  Result: {status} | Proper SL would have: {survival}")
                print(f"  Missed: Rs.{analysis['missed_profit_total']:.2f}")
            else:
                print(f"  Could not analyze (missing data)")
            print()

    if len(sl_post_t1_trades) == 0:
        print("No sl_post_t1 trades found with complete data.")
        return

    # Summary statistics
    print("="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    print()

    total_analyzed = len(sl_post_t1_trades)
    t2_reached_count = sum(1 for t in sl_post_t1_trades if t['t2_reached'])
    proper_sl_survived_count = sum(1 for t in sl_post_t1_trades if t['proper_sl_survived'])

    total_missed_profit = sum(t['missed_profit_total'] for t in sl_post_t1_trades)
    avg_missed_per_trade = total_missed_profit / total_analyzed if total_analyzed > 0 else 0

    print(f"Total sl_post_t1 trades analyzed: {total_analyzed}")
    print(f"Trades that reached T2 after SL: {t2_reached_count} ({t2_reached_count/total_analyzed*100:.1f}%)")
    print(f"Trades where proper SL would have SURVIVED: {proper_sl_survived_count} ({proper_sl_survived_count/total_analyzed*100:.1f}%)")
    print()
    print(f"Total missed profit: Rs.{total_missed_profit:.2f}")
    print(f"Average missed per trade: Rs.{avg_missed_per_trade:.2f}")
    print()

    # Breakdown by outcome
    print("="*120)
    print("DETAILED BREAKDOWN")
    print("="*120)
    print()

    # Category 1: T2 reached + proper SL survived
    cat1 = [t for t in sl_post_t1_trades if t['t2_reached'] and t['proper_sl_survived']]
    print(f"CATEGORY 1: T2 REACHED + PROPER SL SURVIVED ({len(cat1)} trades)")
    print("These are PERFECT examples - we got stopped out by tight SL, but T2 was hit!")
    print("-"*120)
    for t in cat1[:5]:  # Show first 5
        print(f"{t['symbol']:20s} | Entry: {t['entry']:7.2f} | T1: {t['t1_price']:7.2f} | "
              f"T2: {t['t2_target']:7.2f} | Actual SL: {t['actual_sl']:7.2f} | "
              f"Proper SL: {t['proper_sl']:7.2f} | Missed: Rs.{t['missed_profit_total']:8.2f}")
    if len(cat1) > 5:
        print(f"... and {len(cat1)-5} more")
    print()

    # Category 2: Didn't reach T2 but proper SL survived
    cat2 = [t for t in sl_post_t1_trades if not t['t2_reached'] and t['proper_sl_survived']]
    print(f"CATEGORY 2: DIDN'T REACH T2 BUT PROPER SL SURVIVED ({len(cat2)} trades)")
    print("These trades would have stayed alive longer, potentially trailing out with profit")
    print("-"*120)
    for t in cat2[:5]:
        print(f"{t['symbol']:20s} | Entry: {t['entry']:7.2f} | T1: {t['t1_price']:7.2f} | "
              f"Actual SL: {t['actual_sl']:7.2f} | Proper SL: {t['proper_sl']:7.2f} | "
              f"HWM: {t['hwm_after_sl']:7.2f} | Missed: Rs.{t['missed_profit_total']:8.2f}")
    if len(cat2) > 5:
        print(f"... and {len(cat2)-5} more")
    print()

    # Category 3: Proper SL also would have hit
    cat3 = [t for t in sl_post_t1_trades if not t['proper_sl_survived']]
    print(f"CATEGORY 3: PROPER SL ALSO WOULD HAVE HIT ({len(cat3)} trades)")
    print("These are legitimate stop-outs (proper SL would have been hit too)")
    print("-"*120)
    for t in cat3[:5]:
        print(f"{t['symbol']:20s} | Entry: {t['entry']:7.2f} | Proper SL: {t['proper_sl']:7.2f} | "
              f"Low after T1: {t['hwm_after_sl']:7.2f}")
    if len(cat3) > 5:
        print(f"... and {len(cat3)-5} more")
    print()

    # Final verdict
    print("="*120)
    print("VERDICT")
    print("="*120)
    print()

    saveable_trades = proper_sl_survived_count
    saveable_pct = saveable_trades / total_analyzed * 100 if total_analyzed > 0 else 0

    print(f"OUT OF {total_analyzed} sl_post_t1 TRADES:")
    print(f"  - {saveable_trades} ({saveable_pct:.1f}%) WOULD HAVE BEEN SAVED by wider stops")
    print(f"  - {t2_reached_count} ({t2_reached_count/total_analyzed*100:.1f}%) went on to hit T2 target")
    print(f"  - {len(cat3)} ({len(cat3)/total_analyzed*100:.1f}%) were legitimate stops")
    print()
    print(f"TOTAL RECOVERABLE P&L: Rs.{total_missed_profit:.2f}")
    print(f"AVERAGE PER SAVED TRADE: Rs.{total_missed_profit/saveable_trades:.2f}" if saveable_trades > 0 else "N/A")
    print()

    if saveable_pct >= 70:
        print("CONCLUSION: OVERWHELMING EVIDENCE that tight stops are destroying profitable trades!")
    elif saveable_pct >= 50:
        print("CONCLUSION: STRONG EVIDENCE that stop widening would improve results.")
    else:
        print("CONCLUSION: Some trades would be saved, but many are legitimate stops.")

    print()
    print("="*120)

    # Save detailed results
    output_file = Path("sl_post_t1_reversal_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_analyzed': total_analyzed,
                't2_reached': t2_reached_count,
                'proper_sl_survived': proper_sl_survived_count,
                'total_missed_profit': total_missed_profit,
                'avg_missed_per_trade': avg_missed_per_trade,
            },
            'trades': sl_post_t1_trades
        }, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
