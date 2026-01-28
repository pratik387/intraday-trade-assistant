#!/usr/bin/env python3
"""
Pre-aggregate all 1m Feather files to 5m
Run this ONCE before backtesting for massive speedup
"""
import sys
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "cache" / "ohlcv_archive"

def aggregate_symbol(symbol_dir: Path):
    """Aggregate 1m → 5m for a single symbol"""
    symbol = symbol_dir.name
    feather_1m = symbol_dir / f"{symbol}_1minutes.feather"
    feather_5m = symbol_dir / f"{symbol}_5minutes.feather"

    # Skip if 5m already exists and is newer than 1m
    if feather_5m.exists() and feather_5m.stat().st_mtime > feather_1m.stat().st_mtime:
        return f"SKIP {symbol} (5m exists)"

    try:
        # Load 1m data
        df1m = pd.read_feather(feather_1m)

        # Rename columns if needed
        col_map = {
            'date': 'ts',
            'open': 'o',
            'high': 'h',
            'low': 'l',
            'close': 'c',
            'volume': 'v'
        }
        df1m = df1m.rename(columns=col_map)

        # Ensure timestamp
        df1m['ts'] = pd.to_datetime(df1m['ts'])

        # CRITICAL: Match CORRECTED BarBuilder logic (line 274 of bar_builder.py)
        # BarBuilder NOW uses: window = df1[(df1.index >= start_ts) & (df1.index <= end_ts)]
        # Standard OHLC convention: 09:20 bar contains 09:15-09:20 INCLUSIVE (all 5 minutes)
        # Previous bug used > (exclusive start) which missed first minute

        df1m_indexed = df1m.set_index('ts').sort_index()

        # Create 5m groups with INCLUSIVE start (standard OHLC convention)
        # CRITICAL: BarBuilder only closes 5m bars when minute % 5 == 0
        results = []
        first_close = df1m_indexed.index.min().ceil('5min')
        # If first_close.minute % 5 != 0, round up to next 5-minute mark
        while first_close.minute % 5 != 0:
            first_close += pd.Timedelta(minutes=1)

        for end_ts in pd.date_range(start=first_close, end=df1m_indexed.index.max(), freq='5min'):
            start_ts = end_ts - pd.Timedelta(minutes=5)
            # CRITICAL FIX: Use START-STAMPED convention (Zerodha/broker standard)
            # 09:15 bar = [09:15, 09:20) data (EXCLUSIVE end, labeled at START)
            # 09:20 bar = [09:20, 09:25) data
            window = df1m_indexed[(df1m_indexed.index >= start_ts) & (df1m_indexed.index < end_ts)]
            if len(window) > 0:
                results.append({
                    'ts': start_ts,  # ← START-LABELED (matches Zerodha convention)
                    'o': window.iloc[0]['o'],
                    'h': window['h'].max(),
                    'l': window['l'].min(),
                    'c': window.iloc[-1]['c'],
                    'v': window['v'].sum()
                })

        df5m = pd.DataFrame(results) if results else pd.DataFrame(columns=['ts', 'o', 'h', 'l', 'c', 'v'])

        # Drop NaN rows
        df5m = df5m.dropna(subset=['o', 'h', 'l', 'c'])

        # Save
        df5m.to_feather(feather_5m)
        return f"OK {symbol} ({len(df5m)} bars)"

    except Exception as e:
        return f"ERR {symbol}: {str(e)[:60]}"

def main():
    symbol_dirs = [d for d in CACHE_ROOT.iterdir() if d.is_dir() and (d / f"{d.name}_1minutes.feather").exists()]

    print(f"Pre-aggregating {len(symbol_dirs)} symbols (1m -> 5m)")
    print("This will take ~10-20 minutes but only needs to run ONCE")
    print("="*60)

    completed = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(aggregate_symbol, d): d for d in symbol_dirs}

        for future in as_completed(futures):
            result = future.result()
            print(result)
            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed}/{len(symbol_dirs)}")

    print("="*60)
    print(f"DONE! Pre-aggregation complete! Backtest will now be 10-100x faster")

if __name__ == "__main__":
    main()
