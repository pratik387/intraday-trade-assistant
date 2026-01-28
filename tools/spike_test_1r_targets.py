#!/usr/bin/env python3
"""
Spike Test: Would lowering T1 from 1.5R to 1.0R improve target hit rate?

Analyzes trade_logs.log files to check:
1. How many trades hit 1.0R but missed 1.5R target
2. Maximum favorable excursion (MFE) for each trade
3. What % of trades would hit T1 at 1.0R vs current 1.5R

Uses events.jsonl to get entry details and 1-minute candle data to check
if price moved 1.0R in favor before exit.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd

def parse_analytics_for_entries(backtest_dir: Path) -> Dict[str, Dict]:
    """Parse analytics.jsonl to get entry prices and stop losses for each trade."""

    entries = {}  # session_date_symbol -> {entry_price, stop_loss, direction, session_date}

    for session_dir in sorted(backtest_dir.glob("2*")):
        if not session_dir.is_dir():
            continue

        analytics_file = session_dir / "analytics.jsonl"
        if not analytics_file.exists():
            continue

        session_date = session_dir.name

        with open(analytics_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line)

                    # Look for EXIT events (which have entry info in plan)
                    if event.get('stage') == 'EXIT':
                        symbol = event.get('symbol', '').replace('NSE:', '')
                        if not symbol:
                            continue

                        # Get plan details from exit event
                        plan = event.get('plan', {})
                        entry_price = plan.get('entry')
                        stop_loss = plan.get('sl')
                        direction = plan.get('direction')

                        if entry_price and stop_loss:
                            key = f"{session_date}_{symbol}"
                            # Only store if not already present (to avoid duplicates from partial exits)
                            if key not in entries:
                                entries[key] = {
                                    'entry_price': float(entry_price),
                                    'stop_loss': float(stop_loss),
                                    'direction': direction,
                                    'session_date': session_date,
                                    'symbol': symbol
                                }

                except Exception as e:
                    continue

    return entries


def parse_trade_exits(backtest_dir: Path) -> Dict[str, List]:
    """Parse trade_logs.log to get actual exit information."""

    exits = defaultdict(list)  # session_date_symbol -> [exit_events]

    for session_dir in sorted(backtest_dir.glob("2*/trade_logs.log")):
        session_date = session_dir.parent.name

        with open(session_dir, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse EXIT
                if "EXIT" in line:
                    match = re.search(r'NSE:(\w+)\s+\|\s+Qty:\s+\d+\s+\|\s+Entry:\s+Rs\.([\d.]+)\s+\|\s+Exit:\s+Rs\.([\d.]+)\s+\|\s+PnL:\s+Rs\.([-\d.]+)\s+(\w+)', line)
                    if match:
                        symbol = match.group(1)
                        entry_px = float(match.group(2))
                        exit_px = float(match.group(3))
                        pnl = float(match.group(4))
                        exit_reason = match.group(5)

                        exits[f"{session_date}_{symbol}"].append({
                            'exit_px': exit_px,
                            'entry_px': entry_px,
                            'pnl': pnl,
                            'exit_reason': exit_reason
                        })

    return exits


def load_1min_data(cache_dir: Path, symbol: str, session_date: str) -> pd.DataFrame:
    """Load 1-minute candle data for a symbol from the ohlcv_archive cache."""

    # Symbol in cache has .NS suffix
    symbol_ns = f"{symbol}.NS"
    bars_file = cache_dir / "ohlcv_archive" / symbol_ns / f"{symbol_ns}_1minutes.feather"

    if not bars_file.exists():
        return None

    try:
        df = pd.read_feather(bars_file)

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'Datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Datetime'])
            df = df.drop('Datetime', axis=1)

        # Filter to session date
        session_dt = pd.to_datetime(session_date)
        df = df[df['timestamp'].dt.date == session_dt.date()]

        if df.empty:
            return None

        df = df.set_index('timestamp')
        return df
    except Exception as e:
        return None


def calculate_mfe(entry_info: Dict, cache_dir: Path) -> Tuple[float, float, float]:
    """
    Calculate Maximum Favorable Excursion (MFE) for a trade.

    Returns:
        - MFE in price points
        - MFE in R (risk multiples)
        - Whether trade hit 1.0R
    """

    symbol = entry_info['symbol']
    entry_price = entry_info['entry_price']
    stop_loss = entry_info['stop_loss']
    direction = entry_info['direction']
    session_date = entry_info['session_date']

    # Calculate 1R
    risk = abs(entry_price - stop_loss)

    # Load 1-minute data from cache
    df = load_1min_data(cache_dir, symbol, session_date)

    if df is None or df.empty:
        return 0.0, 0.0, False

    # Filter to trading hours (9:15 to 15:30)
    try:
        df = df.between_time('09:15', '15:30')
    except:
        # If index is not datetime, skip
        return 0.0, 0.0, False

    # Normalize column names (could be 'high'/'low' or 'High'/'Low')
    df.columns = [c.lower() for c in df.columns]

    if 'high' not in df.columns or 'low' not in df.columns:
        return 0.0, 0.0, False

    if direction == 'long':
        # For long, MFE is max high minus entry
        max_high = df['high'].max()
        mfe_price = max_high - entry_price
        mfe_r = mfe_price / risk if risk > 0 else 0.0
        hit_1r = max_high >= (entry_price + 1.0 * risk)
    else:
        # For short, MFE is entry minus min low
        min_low = df['low'].min()
        mfe_price = entry_price - min_low
        mfe_r = mfe_price / risk if risk > 0 else 0.0
        hit_1r = min_low <= (entry_price - 1.0 * risk)

    return mfe_price, mfe_r, hit_1r


def main():
    backtest_dir = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251106-041140_extracted\20251106-041140_full\20251106-041140")
    cache_dir = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\cache")

    print("="*100)
    print("SPIKE TEST: Would T1 @ 1.0R improve target hit rate?")
    print("="*100)

    print("\n[1/3] Parsing entry information from analytics.jsonl...")
    entries = parse_analytics_for_entries(backtest_dir)
    print(f"Found {len(entries)} trade entries")

    print("\n[2/3] Parsing exit information from trade_logs.log...")
    exits = parse_trade_exits(backtest_dir)
    print(f"Found exits for {len(exits)} trades")

    print("\n[3/3] Analyzing 1-minute bars for Maximum Favorable Excursion (MFE)...")

    results = {
        'hit_1_5r': 0,  # Hit current 1.5R T1
        'hit_1_0r_only': 0,  # Hit 1.0R but NOT 1.5R
        'hit_neither': 0,  # Didn't hit either
        'no_data': 0,  # No 1m data available
        'trades_analyzed': 0
    }

    trades_detail = []

    for trade_key, entry_info in entries.items():
        results['trades_analyzed'] += 1

        session_date = entry_info['session_date']
        symbol = entry_info['symbol']

        # Calculate MFE using cache
        mfe_price, mfe_r, hit_1r = calculate_mfe(entry_info, cache_dir)

        # Check exit reasons for this trade
        exit_events = exits.get(trade_key, [])

        # Determine if trade hit T1 based on exit reasons
        hit_t1_actual = any(e['exit_reason'] in ['t1_partial', 't2_partial'] for e in exit_events)

        if mfe_r == 0.0:
            results['no_data'] += 1
            status = 'NO_DATA'
        elif mfe_r >= 1.5:
            results['hit_1_5r'] += 1
            status = 'HIT_1.5R'
        elif mfe_r >= 1.0:
            results['hit_1_0r_only'] += 1
            status = 'HIT_1.0R_ONLY'
        else:
            results['hit_neither'] += 1
            status = 'MISS_BOTH'

        trades_detail.append({
            'symbol': symbol,
            'date': session_date,
            'mfe_r': mfe_r,
            'status': status,
            'hit_t1_actual': hit_t1_actual,
            'direction': entry_info['direction']
        })

        if results['trades_analyzed'] % 10 == 0:
            print(f"  Analyzed {results['trades_analyzed']}/{len(entries)} trades...", end='\r')

    print(f"\n  Analysis complete!{' '*50}")

    # Calculate percentages
    total_with_data = results['trades_analyzed'] - results['no_data']

    if total_with_data == 0:
        print("\n[ERROR] No 1-minute data available for analysis")
        return

    current_hit_rate = (results['hit_1_5r'] / total_with_data) * 100
    new_hit_rate = ((results['hit_1_5r'] + results['hit_1_0r_only']) / total_with_data) * 100
    improvement = new_hit_rate - current_hit_rate

    # Print results
    print("\n" + "="*100)
    print("RESULTS: Target Hit Rate Analysis")
    print("="*100)

    print(f"\nTotal trades analyzed: {results['trades_analyzed']}")
    print(f"Trades with 1m data: {total_with_data}")
    print(f"Trades without 1m data: {results['no_data']}")

    print(f"\n{'Outcome':<30} | {'Count':>10} | {'% of Total':>12} | {'Description':<40}")
    print("-" * 100)
    print(f"{'Hit 1.5R (current T1)':<30} | {results['hit_1_5r']:>10} | {(results['hit_1_5r']/total_with_data*100):>11.1f}% | Current target hit rate")
    print(f"{'Hit 1.0R only (NEW)':<30} | {results['hit_1_0r_only']:>10} | {(results['hit_1_0r_only']/total_with_data*100):>11.1f}% | Would hit with lower target")
    print(f"{'Missed both targets':<30} | {results['hit_neither']:>10} | {(results['hit_neither']/total_with_data*100):>11.1f}% | Wouldn't help")

    print("\n" + "="*100)
    print("IMPACT ANALYSIS")
    print("="*100)

    print(f"\n{'Metric':<40} | {'Current (1.5R T1)':>20} | {'Proposed (1.0R T1)':>20} | {'Change':>15}")
    print("-" * 100)
    print(f"{'Target Hit Rate':<40} | {current_hit_rate:>19.1f}% | {new_hit_rate:>19.1f}% | {improvement:>+14.1f}%")
    print(f"{'Trades Hitting Target':<40} | {results['hit_1_5r']:>20} | {results['hit_1_5r'] + results['hit_1_0r_only']:>20} | {results['hit_1_0r_only']:>+15}")
    print(f"{'Trades Missing Target':<40} | {total_with_data - results['hit_1_5r']:>20} | {results['hit_neither']:>20} | {-(results['hit_1_0r_only']):>15}")

    # Estimate P&L impact (assuming same Rs.256 avg per partial exit)
    avg_partial_pnl = 256  # From TARGET_HIT_ANALYSIS.md
    additional_partial_exits = results['hit_1_0r_only'] * 2  # T1 + potentially T2
    estimated_additional_pnl = additional_partial_exits * avg_partial_pnl

    print(f"\n{'Estimated P&L Impact':<40} | {'':>20} | {f'+Rs.{estimated_additional_pnl:,.0f}':>20} | {'':>15}")

    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    if improvement > 5:
        print(f"\n[+] IMPLEMENT: Lowering T1 from 1.5R to 1.0R would improve hit rate by {improvement:.1f}%")
        print(f"    - {results['hit_1_0r_only']} additional trades would hit T1")
        print(f"    - Estimated additional profit: Rs.{estimated_additional_pnl:,.0f}")
        print(f"    - New target hit rate: {new_hit_rate:.1f}% (vs current {current_hit_rate:.1f}%)")
    elif improvement > 0:
        print(f"\n[~] MARGINAL: Lowering T1 would only improve hit rate by {improvement:.1f}%")
        print(f"    - Consider testing, but impact may be limited")
    else:
        print(f"\n[-] NO BENEFIT: Lowering T1 would not improve hit rate significantly")

    print("\n" + "="*100)

    # Show some example trades
    print("\nEXAMPLE TRADES (Hit 1.0R but missed 1.5R):")
    print("="*100)

    examples = [t for t in trades_detail if t['status'] == 'HIT_1.0R_ONLY'][:10]

    if examples:
        print(f"\n{'Date':<15} | {'Symbol':<15} | {'Direction':<10} | {'MFE (R)':<10} | {'Hit T1?':<10}")
        print("-" * 100)
        for ex in examples:
            print(f"{ex['date']:<15} | {ex['symbol']:<15} | {ex['direction']:<10} | {ex['mfe_r']:<10.2f} | {ex['hit_t1_actual']}")
    else:
        print("No examples found")

    print("\n" + "="*100)


if __name__ == "__main__":
    main()
