#!/usr/bin/env python3
"""
Spike Test: resistance_bounce_short R:R Analysis

Tests whether the issue is:
1. Stops too tight (hitting SL before targets)
2. Targets too tight (missing larger moves)

Uses 1m bar data to check:
- How far did price move favorably (MFE)?
- Did we get stopped out too early?
- Would wider targets have been hit?
- Would tighter stops have worked?
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import pandas as pd

def parse_resistance_bounce_trades(backtest_dir: Path) -> List[Dict]:
    """Parse events.jsonl to get resistance_bounce_short trades."""

    trades = {}  # trade_id -> trade_data

    for session_dir in sorted(backtest_dir.glob("2*")):
        if not session_dir.is_dir():
            continue

        events_file = session_dir / "events.jsonl"
        if not events_file.exists():
            continue

        session_date = session_dir.name

        with open(events_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line)

                    event_type = event.get('type')
                    trade_id = event.get('trade_id')

                    # DECISION event has setup_type
                    if event_type == 'DECISION':
                        setup = event.get('decision', {}).get('setup_type')
                        if setup == 'resistance_bounce_short':
                            symbol = event.get('symbol', '').replace('NSE:', '')

                            # Get plan details
                            plan = event.get('plan', {})
                            stop_info = plan.get('stop', {})
                            targets_list = plan.get('targets', [])

                            trades[trade_id] = {
                                'session_date': session_date,
                                'symbol': symbol,
                                'trade_id': trade_id,
                                'entry_price': None,
                                'hard_sl': stop_info.get('hard'),
                                't1_target': targets_list[0].get('level') if len(targets_list) > 0 else None,
                                't2_target': targets_list[1].get('level') if len(targets_list) > 1 else None,
                                'exit_price': None,
                                'pnl': 0,
                                'exit_reason': 'unknown'
                            }

                    # TRIGGER event has actual entry price
                    elif event_type == 'TRIGGER' and trade_id in trades:
                        trigger_info = event.get('trigger', {})
                        entry_px = trigger_info.get('actual_price')
                        if entry_px:
                            trades[trade_id]['entry_price'] = float(entry_px)

                    # EXIT event has exit price and PnL
                    elif event_type == 'EXIT' and trade_id in trades:
                        exit_info = event.get('exit', {})
                        trades[trade_id]['exit_price'] = exit_info.get('price')
                        trades[trade_id]['pnl'] = exit_info.get('pnl', 0)
                        trades[trade_id]['exit_reason'] = exit_info.get('reason', 'unknown')

                except Exception as e:
                    continue

    # Return only trades that have entry prices
    return [t for t in trades.values() if t['entry_price'] is not None and t['hard_sl'] is not None]


def load_1m_data(session_date: str, symbol: str) -> pd.DataFrame:
    """Load 1-minute bar data for the session."""

    # Try common data paths
    data_paths = [
        Path(f"cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_1minutes.feather"),
        Path(f"data/nse_1m/{session_date}/{symbol}.feather"),
        Path(f"data/nse_1m_cache/{session_date}/{symbol}.feather"),
        Path(f"data/feather/1m/{session_date}/{symbol}.feather"),
    ]

    for path in data_paths:
        if path.exists():
            try:
                df = pd.read_feather(path)

                # Handle both 'date' and 'timestamp' column names
                time_col = 'date' if 'date' in df.columns else 'timestamp'
                df[time_col] = pd.to_datetime(df[time_col])

                # Filter to specific session date
                df = df[df[time_col].dt.date == pd.to_datetime(session_date).date()]

                if len(df) > 0:
                    # Rename to 'timestamp' for consistency
                    df = df.rename(columns={time_col: 'timestamp'})
                    return df
            except:
                continue

    return None


def calculate_mfe_mae(df_1m: pd.DataFrame, entry_price: float, direction: str,
                       entry_time: pd.Timestamp) -> Dict:
    """
    Calculate Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE).

    For short trades:
    - MFE = max downward move (entry - lowest low)
    - MAE = max upward move (highest high - entry)
    """

    # Get bars after entry
    after_entry = df_1m[df_1m['timestamp'] >= entry_time].copy()

    if len(after_entry) == 0:
        return {'mfe': 0, 'mae': 0, 'mfe_pct': 0, 'mae_pct': 0}

    if direction == 'short':
        # MFE is maximum downward move
        lowest_low = after_entry['low'].min()
        mfe = entry_price - lowest_low
        mfe_pct = (mfe / entry_price) * 100

        # MAE is maximum upward move
        highest_high = after_entry['high'].max()
        mae = highest_high - entry_price
        mae_pct = (mae / entry_price) * 100
    else:
        # For long (if needed)
        highest_high = after_entry['high'].max()
        mfe = highest_high - entry_price
        mfe_pct = (mfe / entry_price) * 100

        lowest_low = after_entry['low'].min()
        mae = entry_price - lowest_low
        mae_pct = (mae / entry_price) * 100

    return {
        'mfe': mfe,
        'mae': mae,
        'mfe_pct': mfe_pct,
        'mae_pct': mae_pct
    }


def analyze_trade(trade: Dict, df_1m: pd.DataFrame) -> Dict:
    """Analyze a single trade to see if stops/targets were optimal."""

    entry_price = trade['entry_price']
    hard_sl = trade['hard_sl']
    t1_target = trade['t1_target']
    t2_target = trade['t2_target']

    # Calculate risk/reward
    risk = abs(entry_price - hard_sl)
    reward_t1 = abs(entry_price - t1_target) if t1_target else 0
    reward_t2 = abs(entry_price - t2_target) if t2_target else 0

    rr_t1 = reward_t1 / risk if risk > 0 else 0
    rr_t2 = reward_t2 / risk if risk > 0 else 0

    # Assume entry at session start (09:15)
    session_start = df_1m['timestamp'].iloc[0].replace(hour=9, minute=15, second=0)

    # Calculate MFE/MAE
    mfe_mae = calculate_mfe_mae(df_1m, entry_price, 'short', session_start)

    mfe = mfe_mae['mfe']
    mae = mfe_mae['mae']

    # Check what would have happened with different configurations
    analysis = {
        'hit_sl': mae >= risk,
        'hit_t1': mfe >= reward_t1 if t1_target else False,
        'hit_t2': mfe >= reward_t2 if t2_target else False,
        'mfe': mfe,
        'mae': mae,
        'mfe_r': mfe / risk if risk > 0 else 0,
        'mae_r': mae / risk if risk > 0 else 0,
        'current_rr_t1': rr_t1,
        'current_rr_t2': rr_t2,
        'risk': risk,
        'reward_t1': reward_t1,
        'reward_t2': reward_t2
    }

    # Test alternative configurations
    # Option 1: Wider targets (T1: 2.0R, T2: 3.5R)
    alt_t1_wider = entry_price - (risk * 2.0)
    alt_t2_wider = entry_price - (risk * 3.5)
    analysis['would_hit_t1_wider'] = mfe >= abs(entry_price - alt_t1_wider)
    analysis['would_hit_t2_wider'] = mfe >= abs(entry_price - alt_t2_wider)

    # Option 2: Tighter stops (0.3 ATR instead of 0.5 ATR)
    # Estimate: 40% tighter stop
    alt_sl_tighter = entry_price + (risk * 0.6)
    tighter_risk = abs(entry_price - alt_sl_tighter)
    analysis['would_hit_sl_tighter'] = mae >= tighter_risk

    return analysis


def main():
    if len(sys.argv) < 2:
        print("Usage: python spike_test_resistance_bounce.py <backtest_dir>")
        print("Example: python tools/spike_test_resistance_bounce.py backtest_20251117-021749_extracted/20251117-021749_full/20251117-021749")
        sys.exit(1)

    backtest_dir = Path(sys.argv[1])

    if not backtest_dir.exists():
        print(f"Error: Directory {backtest_dir} does not exist")
        sys.exit(1)

    print("=" * 100)
    print("RESISTANCE_BOUNCE_SHORT R:R SPIKE TEST")
    print("=" * 100)
    print()

    # Parse trades
    trades = parse_resistance_bounce_trades(backtest_dir)

    print(f"Found {len(trades)} resistance_bounce_short trades")
    print()

    if len(trades) == 0:
        print("No trades found. Exiting.")
        sys.exit(0)

    # Analyze each trade
    results = []

    for trade in trades:
        # Load 1m data
        df_1m = load_1m_data(trade['session_date'], trade['symbol'])

        if df_1m is None:
            print(f"Warning: No 1m data for {trade['session_date']} {trade['symbol']}, skipping")
            continue

        analysis = analyze_trade(trade, df_1m)
        results.append({**trade, **analysis})

    if len(results) == 0:
        print("No trades could be analyzed (missing 1m data). Exiting.")
        sys.exit(0)

    # Summary statistics
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print()

    total = len(results)
    hit_sl_count = sum(1 for r in results if r['hit_sl'])
    hit_t1_count = sum(1 for r in results if r['hit_t1'])
    hit_t2_count = sum(1 for r in results if r['hit_t2'])

    print(f"Total Trades Analyzed: {total}")
    print()
    print("Current Configuration:")
    print(f"  Avg R:R (T1): {sum(r['current_rr_t1'] for r in results) / total:.2f}")
    print(f"  Avg R:R (T2): {sum(r['current_rr_t2'] for r in results) / total:.2f}")
    print(f"  SL Hit Rate: {hit_sl_count}/{total} ({hit_sl_count/total*100:.1f}%)")
    print(f"  T1 Hit Rate: {hit_t1_count}/{total} ({hit_t1_count/total*100:.1f}%)")
    print(f"  T2 Hit Rate: {hit_t2_count}/{total} ({hit_t2_count/total*100:.1f}%)")
    print()

    avg_mfe_r = sum(r['mfe_r'] for r in results) / total
    avg_mae_r = sum(r['mae_r'] for r in results) / total

    print(f"Average MFE: {avg_mfe_r:.2f}R (max favorable move)")
    print(f"Average MAE: {avg_mae_r:.2f}R (max adverse move)")
    print()

    # Option 1: Wider Targets
    print("=" * 100)
    print("OPTION 1: WIDER TARGETS (T1: 2.0R, T2: 3.5R)")
    print("=" * 100)
    print()

    would_hit_t1_wider = sum(1 for r in results if r['would_hit_t1_wider'])
    would_hit_t2_wider = sum(1 for r in results if r['would_hit_t2_wider'])

    print(f"  T1 Hit Rate (2.0R): {would_hit_t1_wider}/{total} ({would_hit_t1_wider/total*100:.1f}%)")
    print(f"  T2 Hit Rate (3.5R): {would_hit_t2_wider}/{total} ({would_hit_t2_wider/total*100:.1f}%)")
    print(f"  Change: T1 {would_hit_t1_wider - hit_t1_count:+d} trades, T2 {would_hit_t2_wider - hit_t2_count:+d} trades")
    print()

    # Option 2: Tighter Stops
    print("=" * 100)
    print("OPTION 2: TIGHTER STOPS (0.3 ATR vs current 0.5 ATR)")
    print("=" * 100)
    print()

    would_hit_sl_tighter = sum(1 for r in results if r['would_hit_sl_tighter'])

    print(f"  SL Hit Rate (tighter): {would_hit_sl_tighter}/{total} ({would_hit_sl_tighter/total*100:.1f}%)")
    print(f"  Change: {would_hit_sl_tighter - hit_sl_count:+d} additional SL hits")
    print()

    # Recommendation
    print("=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print()

    if avg_mfe_r > 2.0:
        print("MFE shows price moves well beyond current targets.")
        print("RECOMMENDED: Widen targets (Option 1)")
        print(f"  - Average MFE of {avg_mfe_r:.2f}R suggests targets at 2.0R/3.5R are achievable")

    if would_hit_sl_tighter > hit_sl_count * 1.3:
        print("\nTighter stops would get hit 30%+ more often.")
        print("NOT RECOMMENDED: Tighten stops (Option 2)")
        print(f"  - Would increase SL hits from {hit_sl_count} to {would_hit_sl_tighter}")

    if avg_mfe_r < 1.5 and hit_sl_count > total * 0.5:
        print("\nMFE is low and SL hit rate is high (>50%).")
        print("ISSUE: Stops may be too tight OR setup quality is poor")
        print("  - Consider improving entry timing or setup filters")

    print()
    print("=" * 100)

    # Detailed trade breakdown
    print()
    print("DETAILED TRADE ANALYSIS (First 10 trades):")
    print("=" * 100)

    for i, r in enumerate(results[:10]):
        print(f"\nTrade {i+1}: {r['session_date']} {r['symbol']}")
        print(f"  Entry: {r['entry_price']:.2f} | SL: {r['hard_sl']:.2f} | Risk: {r['risk']:.2f}")
        print(f"  T1: {r['t1_target']:.2f} ({r['current_rr_t1']:.2f}R) | T2: {r['t2_target']:.2f} ({r['current_rr_t2']:.2f}R)")
        print(f"  MFE: {r['mfe']:.2f} ({r['mfe_r']:.2f}R) | MAE: {r['mae']:.2f} ({r['mae_r']:.2f}R)")
        print(f"  Outcome: {r['exit_reason']} | PnL: Rs {r['pnl']:.0f}")
        print(f"  Hit SL: {r['hit_sl']} | Hit T1: {r['hit_t1']} | Hit T2: {r['hit_t2']}")


if __name__ == '__main__':
    main()
