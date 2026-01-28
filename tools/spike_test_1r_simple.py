#!/usr/bin/env python3
"""
Simplified Spike Test: Would lowering T1 from 1.5R to 1.0R improve target hit rate?

Uses trade_logs.log for entries/exits and cache for 1m data.
Calculates stop loss from entry using configuration sl_atr_mult = 2.0.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Configuration parameters
SL_ATR_MULT = 2.25  # From Phase 1 changes
T1_RR = 1.5  # Current T1 target
T2_RR = 3.0  # Current T2 target

def parse_trade_entries(backtest_dir):
    """Parse TRIGGER_EXEC lines from trade_logs.log to get entries."""

    entries = {}  # session_date_symbol -> {entry_price, strategy, direction}

    for log_file in sorted(backtest_dir.glob("2*/trade_logs.log")):
        session_date = log_file.parent.name

        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if "TRIGGER_EXEC" in line:
                    match = re.search(r'NSE:(\w+)\s+\|\s+(BUY|SELL)\s+\d+\s+@\s+([\d.]+)\s+\|\s+strategy=(\w+)', line)
                    if match:
                        symbol = match.group(1)
                        direction = "long" if match.group(2) == "BUY" else "short"
                        entry_price = float(match.group(3))
                        strategy = match.group(4)

                        key = f"{session_date}_{symbol}"
                        entries[key] = {
                            'symbol': symbol,
                            'session_date': session_date,
                            'entry_price': entry_price,
                            'direction': direction,
                            'strategy': strategy
                        }

    return entries


def get_atr_from_cache(cache_dir, symbol, session_date):
    """Estimate ATR from 1-minute data."""

    symbol_ns = f"{symbol}.NS"
    bars_file = cache_dir / "ohlcv_archive" / symbol_ns / f"{symbol_ns}_1minutes.feather"

    if not bars_file.exists():
        return None

    try:
        df = pd.read_feather(bars_file)

        # Convert date column to timestamp
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'Datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Datetime'])
        else:
            return None

        # Filter to session date and days before for ATR calc
        session_dt = pd.to_datetime(session_date)
        start_date = session_dt - pd.Timedelta(days=5)
        df = df[(df['timestamp'].dt.date >= start_date.date()) & (df['timestamp'].dt.date <= session_dt.date())]

        if df.empty:
            return None

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]

        # Calculate True Range
        df['tr'] = df[['high', 'low', 'close']].apply(
            lambda row: max(row['high'] - row['low'],
                          abs(row['high'] - row['close']),
                          abs(row['low'] - row['close'])),
            axis=1
        )

        # Get ATR (average of recent TRs)
        atr = df['tr'].tail(200).mean()  # ~1 day of 1m bars

        return atr

    except Exception as e:
        return None


def calculate_mfe_from_cache(cache_dir, symbol, session_date, entry_price, direction, atr):
    """Calculate Maximum Favorable Excursion from 1m cache data."""

    symbol_ns = f"{symbol}.NS"
    bars_file = cache_dir / "ohlcv_archive" / symbol_ns / f"{symbol_ns}_1minutes.feather"

    if not bars_file.exists():
        return None, None

    try:
        df = pd.read_feather(bars_file)

        # Convert date column to timestamp
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'Datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Datetime'])
        else:
            return None, None

        # Filter to session date
        session_dt = pd.to_datetime(session_date)
        df = df[df['timestamp'].dt.date == session_dt.date()]

        if df.empty:
            return None, None

        df = df.set_index('timestamp')

        # Filter to trading hours
        try:
            df = df.between_time('09:15', '15:30')
        except:
            return None, None

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]

        if 'high' not in df.columns or 'low' not in df.columns:
            return None, None

        # Calculate 1R (risk) using ATR
        risk = SL_ATR_MULT * atr

        if direction == 'long':
            # For long, MFE is max high minus entry
            max_high = df['high'].max()
            mfe_r = (max_high - entry_price) / risk if risk > 0 else 0.0
        else:
            # For short, MFE is entry minus min low
            min_low = df['low'].min()
            mfe_r = (entry_price - min_low) / risk if risk > 0 else 0.0

        return mfe_r, risk

    except Exception as e:
        return None, None


def main():
    backtest_dir = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251106-041140_extracted\20251106-041140_full\20251106-041140")
    cache_dir = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\cache")

    print("="*100)
    print("SPIKE TEST: Would T1 @ 1.0R improve target hit rate vs current 1.5R?")
    print("="*100)
    print(f"\nConfiguration: SL @ {SL_ATR_MULT}× ATR, T1 @ {T1_RR}R, T2 @ {T2_RR}R")

    print("\n[1/2] Parsing trade entries...")
    entries = parse_trade_entries(backtest_dir)
    print(f"Found {len(entries)} trade entries")

    if len(entries) == 0:
        print("[ERROR] No entries found")
        return

    print("\n[2/2] Analyzing Maximum Favorable Excursion (MFE) from 1m cache...")

    results = {
        'hit_1_5r': 0,  # Hit current 1.5R T1
        'hit_1_0r_only': 0,  # Hit 1.0R but NOT 1.5R
        'hit_neither': 0,  # Didn't hit either
        'no_data': 0  # No 1m data available
    }

    trades_detail = []

    for i, (trade_key, entry_info) in enumerate(entries.items(), 1):
        symbol = entry_info['symbol']
        session_date = entry_info['session_date']
        entry_price = entry_info['entry_price']
        direction = entry_info['direction']

        # Get ATR
        atr = get_atr_from_cache(cache_dir, symbol, session_date)

        if atr is None:
            results['no_data'] += 1
            continue

        # Calculate MFE
        mfe_r, risk = calculate_mfe_from_cache(cache_dir, symbol, session_date, entry_price, direction, atr)

        if mfe_r is None:
            results['no_data'] += 1
            continue

        # Categorize
        if mfe_r >= T1_RR:  # Hit 1.5R
            results['hit_1_5r'] += 1
            status = f'HIT_{T1_RR}R'
        elif mfe_r >= 1.0:  # Hit 1.0R but not 1.5R
            results['hit_1_0r_only'] += 1
            status = 'HIT_1.0R_ONLY'
        else:  # Didn't hit either
            results['hit_neither'] += 1
            status = 'MISS_BOTH'

        trades_detail.append({
            'symbol': symbol,
            'date': session_date,
            'mfe_r': mfe_r,
            'status': status,
            'direction': direction
        })

        if i % 5 == 0:
            print(f"  Analyzed {i}/{len(entries)} trades...", end='\r')

    print(f"\n  Analysis complete!{' '*50}")

    # Calculate results
    total_with_data = len(entries) - results['no_data']

    if total_with_data == 0:
        print("\n[ERROR] No 1-minute data available for analysis")
        return

    current_hit_rate = (results['hit_1_5r'] / total_with_data) * 100
    new_hit_rate = ((results['hit_1_5r'] + results['hit_1_0r_only']) / total_with_data) * 100
    improvement = new_hit_rate - current_hit_rate

    # Print results
    print("\n" + "="*100)
    print("RESULTS")
    print("="*100)

    print(f"\nTotal trades: {len(entries)}")
    print(f"Trades with 1m data: {total_with_data}")
    print(f"Trades without data: {results['no_data']}")

    print(f"\n{'Outcome':<30} | {'Count':>10} | {'% of Total':>12}")
    print("-" * 60)
    print(f"{'Hit 1.5R (current T1)':<30} | {results['hit_1_5r']:>10} | {current_hit_rate:>11.1f}%")
    print(f"{'Hit 1.0R only (NEW)':<30} | {results['hit_1_0r_only']:>10} | {(results['hit_1_0r_only']/total_with_data*100):>11.1f}%")
    print(f"{'Missed both':<30} | {results['hit_neither']:>10} | {(results['hit_neither']/total_with_data*100):>11.1f}%")

    print("\n" + "="*100)
    print("IMPACT")
    print("="*100)

    print(f"\n{'Metric':<40} | {'Current (1.5R)':>20} | {'Proposed (1.0R)':>20} | {'Change':>15}")
    print("-" * 100)
    print(f"{'Target Hit Rate':<40} | {current_hit_rate:>19.1f}% | {new_hit_rate:>19.1f}% | {improvement:>+14.1f}%")
    print(f"{'Trades Hitting Target':<40} | {results['hit_1_5r']:>20} | {results['hit_1_5r'] + results['hit_1_0r_only']:>20} | {results['hit_1_0r_only']:>+15}")

    # Estimate P&L impact
    avg_partial_pnl = 256  # From TARGET_HIT_ANALYSIS.md
    additional_partial_exits = results['hit_1_0r_only'] * 2  # T1 + potentially T2
    estimated_pnl = additional_partial_exits * avg_partial_pnl

    print(f"{'Estimated Additional P&L':<40} | {'-':>20} | {f'+Rs.{estimated_pnl:,.0f}':>20} | {'-':>15}")

    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    if improvement >= 10:
        print(f"\n[+] STRONGLY RECOMMEND: Lower T1 from 1.5R to 1.0R")
        print(f"    - Improvement: +{improvement:.1f}% hit rate ({results['hit_1_0r_only']} more trades)")
        print(f"    - Estimated profit: +Rs.{estimated_pnl:,.0f}")
    elif improvement >= 5:
        print(f"\n[+] RECOMMEND: Lower T1 from 1.5R to 1.0R")
        print(f"    - Moderate improvement: +{improvement:.1f}% hit rate")
    else:
        print(f"\n[~] LIMITED BENEFIT: Only +{improvement:.1f}% improvement")
        print(f"    - Consider other optimizations first")

    # Show examples
    print("\n" + "="*100)
    print("EXAMPLES (Hit 1.0R but missed 1.5R):")
    print("="*100)

    examples = [t for t in trades_detail if t['status'] == 'HIT_1.0R_ONLY'][:10]

    if examples:
        print(f"\n{'Date':<15} | {'Symbol':<15} | {'Direction':<10} | {'MFE (R)':<10}")
        print("-" * 60)
        for ex in examples:
            print(f"{ex['date']:<15} | {ex['symbol']:<15} | {ex['direction']:<10} | {ex['mfe_r']:.2f}")
    else:
        print("\nNo examples found")

    print("\n" + "="*100)


if __name__ == "__main__":
    main()
