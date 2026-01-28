#!/usr/bin/env python3
"""
Spike Test: failure_fade_long with PROPER TIMING

Checks WHICH hit first: SL or T1, using bar-by-bar analysis from entry time.
Adapted from resistance_bounce spike test.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

def parse_failure_fade_trades(backtest_dir: Path) -> List[Dict]:
    """Parse events.jsonl to get failure_fade_long trades with entry timestamps."""

    trades = {}

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

                    if event_type == 'DECISION':
                        setup = event.get('decision', {}).get('setup_type')
                        if setup == 'failure_fade_long':
                            symbol = event.get('symbol', '').replace('NSE:', '')
                            plan = event.get('plan', {})
                            stop_info = plan.get('stop', {})
                            targets_list = plan.get('targets', [])

                            trades[trade_id] = {
                                'session_date': session_date,
                                'symbol': symbol,
                                'trade_id': trade_id,
                                'entry_price': None,
                                'entry_time': None,
                                'hard_sl': stop_info.get('hard'),
                                't1_target': targets_list[0].get('level') if len(targets_list) > 0 else None,
                                't2_target': targets_list[1].get('level') if len(targets_list) > 1 else None,
                                'exit_price': None,
                                'exit_time': None,
                                'pnl': 0,
                                'exit_reason': 'unknown'
                            }

                    elif event_type == 'TRIGGER' and trade_id in trades:
                        trigger_info = event.get('trigger', {})
                        entry_px = trigger_info.get('actual_price')
                        entry_ts = event.get('ts')
                        if entry_px:
                            trades[trade_id]['entry_price'] = float(entry_px)
                            trades[trade_id]['entry_time'] = entry_ts

                    elif event_type == 'EXIT' and trade_id in trades:
                        exit_info = event.get('exit', {})
                        trades[trade_id]['exit_price'] = exit_info.get('price')
                        trades[trade_id]['exit_time'] = event.get('ts')
                        trades[trade_id]['pnl'] = exit_info.get('pnl', 0)
                        trades[trade_id]['exit_reason'] = exit_info.get('reason', 'unknown')

                except Exception as e:
                    continue

    return [t for t in trades.values() if t['entry_price'] is not None and t['hard_sl'] is not None]


def load_1m_data(session_date: str, symbol: str) -> pd.DataFrame:
    """Load 1-minute bar data for the session."""

    data_paths = [
        Path(f"cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_1minutes.feather"),
        Path(f"data/nse_1m/{session_date}/{symbol}.feather"),
    ]

    for path in data_paths:
        if path.exists():
            try:
                df = pd.read_feather(path)
                time_col = 'date' if 'date' in df.columns else 'timestamp'
                df[time_col] = pd.to_datetime(df[time_col])
                df = df[df[time_col].dt.date == pd.to_datetime(session_date).date()]

                if len(df) > 0:
                    df = df.rename(columns={time_col: 'timestamp'})
                    return df
            except:
                continue

    return None


def analyze_trade_timing(trade: Dict, df_1m: pd.DataFrame) -> Dict:
    """Analyze WHEN SL and T1 were hit (if at all) using bar-by-bar timing."""

    entry_price = trade['entry_price']
    entry_time_str = trade['entry_time']
    hard_sl = trade['hard_sl']
    t1_target = trade['t1_target']

    # Filter data after entry
    df_after_entry = df_1m[df_1m['timestamp'] >= entry_time_str].copy()

    if len(df_after_entry) == 0:
        return {
            'sl_hit_time': None,
            't1_hit_time': None,
            'hit_order': 'no_data',
            'analysis': 'No data after entry'
        }

    # Calculate risk/reward for R-multiples
    risk = abs(entry_price - hard_sl)
    reward_t1 = abs(entry_price - t1_target)

    # Track first hit
    sl_hit_time = None
    t1_hit_time = None

    for _, bar in df_after_entry.iterrows():
        # For LONG trades (failure_fade_long):
        # SL hit if low <= SL (going down)
        # T1 hit if high >= T1 (going up)

        if sl_hit_time is None and bar['low'] <= hard_sl:
            sl_hit_time = bar['timestamp']

        if t1_hit_time is None and bar['high'] >= t1_target:
            t1_hit_time = bar['timestamp']

        # Stop once both are determined
        if sl_hit_time and t1_hit_time:
            break

    # Determine which hit first
    if sl_hit_time and t1_hit_time:
        if sl_hit_time < t1_hit_time:
            hit_order = 'sl_first'
            analysis = f"SL hit first @ {sl_hit_time.strftime('%H:%M')}, T1 hit later @ {t1_hit_time.strftime('%H:%M')}"
        else:
            hit_order = 't1_first'
            analysis = f"T1 hit first @ {t1_hit_time.strftime('%H:%M')}, SL hit later @ {sl_hit_time.strftime('%H:%M')}"
    elif sl_hit_time:
        hit_order = 'sl_only'
        analysis = f"Only SL hit @ {sl_hit_time.strftime('%H:%M')}, T1 never reached"
    elif t1_hit_time:
        hit_order = 't1_only'
        analysis = f"Only T1 hit @ {t1_hit_time.strftime('%H:%M')}, SL never hit"
    else:
        hit_order = 'neither'
        analysis = "Neither SL nor T1 hit"

    # Calculate MFE/MAE for LONG trades
    mfe = df_after_entry['high'].max() - entry_price  # Max favorable (upward for long)
    mae = entry_price - df_after_entry['low'].min()   # Max adverse (downward for long)

    return {
        'sl_hit_time': sl_hit_time,
        't1_hit_time': t1_hit_time,
        'hit_order': hit_order,
        'analysis': analysis,
        'risk': risk,
        'reward_t1': reward_t1,
        'mfe': mfe,
        'mae': mae,
        'mfe_r': mfe / risk if risk > 0 else 0,
        'mae_r': mae / risk if risk > 0 else 0,
        'mfe_pct': (mfe / entry_price * 100) if entry_price > 0 else 0,
        'mae_pct': (mae / entry_price * 100) if entry_price > 0 else 0
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python spike_test_failure_fade_timing.py <backtest_dir>")
        print("Example: python tools/spike_test_failure_fade_timing.py backtest_20251119-082113_extracted/20251119-082113_full/20251119-082113")
        sys.exit(1)

    backtest_dir = Path(sys.argv[1])

    if not backtest_dir.exists():
        print(f"Error: Directory {backtest_dir} does not exist")
        sys.exit(1)

    print("=" * 100)
    print("FAILURE_FADE_LONG TIMING ANALYSIS")
    print("=" * 100)
    print()

    # Parse trades
    trades = parse_failure_fade_trades(backtest_dir)
    print(f"Found {len(trades)} failure_fade_long trades")
    print()

    if len(trades) == 0:
        print("No trades found. Exiting.")
        sys.exit(0)

    # Analyze each trade
    results = []

    for trade in trades:
        df_1m = load_1m_data(trade['session_date'], trade['symbol'])

        if df_1m is None:
            print(f"Warning: No 1m data for {trade['session_date']} {trade['symbol']}, skipping")
            continue

        timing = analyze_trade_timing(trade, df_1m)
        results.append({**trade, **timing})

    if len(results) == 0:
        print("No trades could be analyzed (missing 1m data). Exiting.")
        sys.exit(0)

    # Summary statistics
    print("=" * 100)
    print("TIMING SUMMARY")
    print("=" * 100)
    print()

    sl_first = [r for r in results if r['hit_order'] == 'sl_first']
    t1_first = [r for r in results if r['hit_order'] == 't1_first']
    sl_only = [r for r in results if r['hit_order'] == 'sl_only']
    t1_only = [r for r in results if r['hit_order'] == 't1_only']
    neither = [r for r in results if r['hit_order'] == 'neither']

    total = len(results)

    print(f"Total Trades: {total}")
    print()
    print(f"SL hit FIRST (then T1): {len(sl_first)} ({len(sl_first)/total*100:.1f}%)")
    print(f"T1 hit FIRST (then SL): {len(t1_first)} ({len(t1_first)/total*100:.1f}%)")
    print(f"SL hit ONLY (T1 never): {len(sl_only)} ({len(sl_only)/total*100:.1f}%)")
    print(f"T1 hit ONLY (SL never): {len(t1_only)} ({len(t1_only)/total*100:.1f}%)")
    print(f"Neither hit: {len(neither)} ({len(neither)/total*100:.1f}%)")
    print()

    # Key insight
    print("=" * 100)
    print("KEY INSIGHT")
    print("=" * 100)
    print()

    sl_before_t1 = len(sl_first) + len(sl_only)
    t1_before_sl = len(t1_first) + len(t1_only)

    print(f"Trades stopped out before/without T1: {sl_before_t1}/{total} ({sl_before_t1/total*100:.1f}%)")
    print(f"Trades that hit T1 before/without SL: {t1_before_sl}/{total} ({t1_before_sl/total*100:.1f}%)")
    print()

    if sl_before_t1 > total * 0.6:
        print("*** STOPS TOO TIGHT - Majority of trades hit SL before reaching T1 ***")
        print("Recommendation: WIDEN STOPS")
    elif t1_before_sl > total * 0.6:
        print("*** STOPS APPROPRIATE - Majority of trades reach T1 ***")
        print("Issue may be elsewhere (exit management, targets too tight, etc.)")
    else:
        print("Mixed results - further investigation needed")

    print()

    # MFE/MAE Analysis
    print("=" * 100)
    print("MFE/MAE ANALYSIS")
    print("=" * 100)
    print()

    # Trades that had MFE but still hit SL
    sl_trades = sl_first + sl_only
    sl_with_mfe = [r for r in sl_trades if r['mfe_pct'] > 0.3]

    if len(sl_with_mfe) > 0:
        print(f"Trades stopped out AFTER moving favorably (MFE > 0.3%): {len(sl_with_mfe)}/{len(sl_trades)} ({len(sl_with_mfe)/len(sl_trades)*100:.1f}%)")
        print()
        print("These trades moved in the right direction before reversing:")
        for r in sl_with_mfe[:10]:
            print(f"  {r['symbol']:12s} | MFE: +{r['mfe_pct']:5.2f}% ({r['mfe_r']:.2f}R) | MAE: -{r['mae_pct']:5.2f}% ({r['mae_r']:.2f}R)")
        print()
        print("INSIGHT: These trades had potential but needed more room or trailing stop.")
        print()

    # Winners that needed drawdown room
    winners = [r for r in results if r['pnl'] > 0]
    if len(winners) > 0:
        avg_winner_mae_pct = sum(r['mae_pct'] for r in winners) / len(winners)
        max_winner_mae_pct = max(r['mae_pct'] for r in winners)

        print(f"Winners analysis:")
        print(f"  Average MAE (drawdown): {avg_winner_mae_pct:.2f}%")
        print(f"  Max MAE (max drawdown): {max_winner_mae_pct:.2f}%")
        print()

        # Compare to losers
        losers = [r for r in sl_trades if r.get('mae_pct') is not None]
        if len(losers) > 0:
            avg_loser_mae_pct = sum(r['mae_pct'] for r in losers) / len(losers)

            print(f"Losers (SL hits) analysis:")
            print(f"  Average MAE before SL: {avg_loser_mae_pct:.2f}%")
            print()

            if max_winner_mae_pct > avg_loser_mae_pct:
                print(f"CRITICAL: Winners needed up to {max_winner_mae_pct:.2f}% drawdown to succeed")
                print(f"Current SL being hit at ~{avg_loser_mae_pct:.2f}% is TOO TIGHT")
                print(f"ACTION: Widen stops to at least {max_winner_mae_pct * 1.1:.2f}% (10% buffer)")
                print()

    print()
    print("=" * 100)
    print("DETAILED TRADE ANALYSIS (First 20)")
    print("=" * 100)
    print()

    for i, r in enumerate(results[:20], 1):
        print(f"Trade {i}: {r['session_date']} {r['symbol']}")
        print(f"  Entry: {r['entry_price']:.2f} @ {r['entry_time']}")
        print(f"  SL: {r['hard_sl']:.2f} | T1: {r['t1_target']:.2f}")
        print(f"  Risk: {r['risk']:.2f} ({r['risk']/r['entry_price']*100:.2f}%)")
        print(f"  MFE: {r['mfe']:.2f} (+{r['mfe_pct']:.2f}%, {r['mfe_r']:.2f}R) | MAE: {r['mae']:.2f} (-{r['mae_pct']:.2f}%, {r['mae_r']:.2f}R)")
        print(f"  Outcome: {r['exit_reason']} | PnL: Rs {r['pnl']:.0f}")
        print(f"  {r['analysis']}")
        print()

    # Summary recommendations
    print("=" * 100)
    print("ACTIONABLE RECOMMENDATIONS")
    print("=" * 100)
    print()

    sl_rate = sl_before_t1 / total * 100 if total > 0 else 0

    print(f"1. STOP LOSS WIDTH:")
    print(f"   Current: {sl_rate:.1f}% of trades hit SL before T1")
    if sl_rate > 60:
        print(f"   STATUS: TOO TIGHT")
        if len(winners) > 0:
            print(f"   ACTION: Widen stops to {max_winner_mae_pct * 1.1:.2f}% (based on winner MAE analysis)")
    print()

    if len(sl_with_mfe) > 0:
        print(f"2. TRAILING STOP:")
        print(f"   {len(sl_with_mfe)} trades moved favorably (+0.3%) before SL")
        print(f"   ACTION: Implement trailing stop after MFE > 0.3% or T1 partial")
        print()

    print(f"3. ENTRY QUALITY:")
    print(f"   ACTION: Wait for consolidation/confirmation before entry")
    print(f"   Current issue: Too many premature entries getting stopped immediately")
    print()


if __name__ == '__main__':
    main()
