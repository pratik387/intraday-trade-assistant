#!/usr/bin/env python
"""
Generate big mover lists for ALL dates in the backtest.

Uses find_big_movers.py logic to scan all dates and generate
individual JSON files for each date.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict

def analyze_single_stock_day(symbol_dir, target_date):
    """
    Analyze one stock for one day.

    Returns metrics if data exists for target_date, else None.
    """

    ohlcv_1m_file = symbol_dir / f"{symbol_dir.name}_1minutes.feather"

    if not ohlcv_1m_file.exists():
        return None

    try:
        df = pd.read_feather(ohlcv_1m_file)

        # Filter to target date
        df['day'] = pd.to_datetime(df['date']).dt.date
        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()

        day_data = df[df['day'] == target_date_obj].copy()

        if len(day_data) < 10:  # Need meaningful data
            return None

        # Calculate intraday metrics
        day_open = day_data.iloc[0]['open']
        day_high = day_data['high'].max()
        day_low = day_data['low'].min()
        day_close = day_data.iloc[-1]['close']

        # Range and directional move
        intraday_range_pct = ((day_high - day_low) / day_open) * 100
        directional_move_pct = ((day_close - day_open) / day_open) * 100

        # Movement consistency
        day_data['price_change'] = day_data['close'].diff()
        day_data['direction'] = np.sign(day_data['price_change'])
        direction_changes = (day_data['direction'].diff() != 0).sum()
        consistency_score = 100 - (direction_changes / len(day_data) * 100)

        # Time of biggest move
        day_data['abs_change'] = day_data['price_change'].abs()
        max_move_idx = day_data['abs_change'].idxmax()
        time_of_move = pd.to_datetime(day_data.loc[max_move_idx, 'date']).hour

        if time_of_move < 10:
            time_category = 'ORB'
        elif time_of_move < 13:
            time_category = 'Morning'
        elif time_of_move < 15:
            time_category = 'Afternoon'
        else:
            time_category = 'Close'

        # Volume (if available)
        total_volume = day_data['volume'].sum() if 'volume' in day_data.columns else 0

        return {
            'symbol': symbol_dir.name,
            'intraday_range_pct': intraday_range_pct,
            'directional_move_pct': directional_move_pct,
            'abs_move': abs(directional_move_pct),
            'consistency_score': consistency_score,
            'time_of_move': time_category,
            'time_of_move_hour': time_of_move,
            'total_volume': total_volume
        }

    except Exception as e:
        return None

def find_top_movers(ohlcv_archive_dir, target_date, top_n=100):
    """
    Scan all stocks and find top N movers for target date.
    """

    archive_path = Path(ohlcv_archive_dir)
    all_symbol_dirs = [d for d in archive_path.iterdir() if d.is_dir()]

    movers = []

    for symbol_dir in all_symbol_dirs:
        result = analyze_single_stock_day(symbol_dir, target_date)
        if result:
            movers.append(result)

    # Sort by absolute move
    movers.sort(key=lambda x: x['abs_move'], reverse=True)

    return movers[:top_n]

def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

def main():
    """Generate big movers for all dates in backtest."""

    ohlcv_archive = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\cache\ohlcv_archive")
    backtest_dir = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251112-023811_extracted\20251112-023811_full\20251112-023811")

    # Get all date folders
    date_folders = sorted([
        d.name for d in backtest_dir.iterdir()
        if d.is_dir() and d.name.startswith('202')
    ])

    print(f"Found {len(date_folders)} dates in backtest")
    print(f"Generating big mover lists for each date...")
    print()

    success_count = 0
    skip_count = 0

    for i, date_str in enumerate(date_folders, 1):
        # Check if file already exists
        output_file = Path(f"big_movers_{date_str}.json")
        if output_file.exists():
            skip_count += 1
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(date_folders)} dates processed (skipped {skip_count} existing files)...")
            continue

        # Find top movers for this date
        top_movers = find_top_movers(ohlcv_archive, date_str, top_n=100)

        if not top_movers:
            continue

        # Save to JSON
        output = {
            'date': date_str,
            'top_movers': convert_to_native(top_movers)
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        success_count += 1

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(date_folders)} dates processed ({success_count} new files generated)...")

    print()
    print("="*80)
    print("BIG MOVER GENERATION COMPLETE")
    print("="*80)
    print(f"Total dates processed: {len(date_folders)}")
    print(f"New files generated: {success_count}")
    print(f"Existing files skipped: {skip_count}")
    print(f"Total files available: {success_count + skip_count}")
    print()
    print("Next step: Run analyze_big_movers_all_dates.py to analyze rejection patterns")
    print("="*80)

if __name__ == '__main__':
    main()
