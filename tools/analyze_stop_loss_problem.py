#!/usr/bin/env python
"""
Analyze Stop Loss Problem - Why are we hitting 42.4% hard SLs?

This script analyzes all hard_sl exits using 1m spike data to determine:
1. Would wider stops (1.5R vs 1R) have saved the trade?
2. Did price reverse after SL hit (false stop-out)?
3. Which setups/regimes have the worst SL problem?
4. Should we delay BE move until after T1?
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
CACHE_DIR = ROOT / "cache"

def load_1m_data(symbol: str, date: str) -> pd.DataFrame:
    """Load 1m data for a symbol on a specific date"""
    if symbol.startswith('NSE:'):
        symbol_clean = symbol.replace('NSE:', '')
    else:
        symbol_clean = symbol

    # Try archive location first
    archive_dir = CACHE_DIR / "ohlcv_archive" / f"{symbol_clean}.NS"
    feather_1m = None

    if archive_dir.exists():
        archive_feather = archive_dir / f"{symbol_clean}.NS_1minutes.feather"
        if archive_feather.exists():
            feather_1m = archive_feather

    # Fallback to old location
    if not feather_1m:
        symbol_cache = CACHE_DIR / symbol_clean
        if symbol_cache.exists():
            old_feather = symbol_cache / "1m.feather"
            if old_feather.exists():
                feather_1m = old_feather

    if not feather_1m:
        return pd.DataFrame()

    try:
        df = pd.read_feather(feather_1m)

        # Handle different column names
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        elif df.index.name == 'timestamp':
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif df.index.name == 'date':
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            return pd.DataFrame()

        df['date_only'] = df['timestamp'].dt.date
        target_date = pd.to_datetime(date).date()
        df = df[df['date_only'] == target_date].copy()

        return df.sort_values('timestamp')
    except Exception as e:
        return pd.DataFrame()

def load_trade_plan(session_dir, trade_id):
    """Load trade plan from events.jsonl"""
    events_file = session_dir / "events.jsonl"
    if not events_file.exists():
        return None

    try:
        with open(events_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        if event.get('trade_id') == trade_id and event.get('type') == 'DECISION':
                            return event.get('plan', {})
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass

    return None

def analyze_stop_loss(trade, session_dir, debug=False):
    """Analyze if wider stops would have saved the trade"""
    trade_id = trade.get('trade_id', '')
    symbol = trade.get('symbol', '')
    exit_time_str = trade.get('timestamp', '')
    exit_price = trade.get('exit_price', 0)

    # Load trade plan
    plan = load_trade_plan(session_dir, trade_id)
    if not plan:
        if debug:
            print(f"  [DEBUG] No plan found in events.jsonl")
        return None

    entry_price = plan.get('entry', {}).get('reference', 0)
    stop_loss = plan.get('stop', {}).get('hard', 0)
    side = plan.get('bias', 'long')

    # Get T1 target for later analysis (targets is a list)
    targets = plan.get('targets', [])
    t1_target = targets[0].get('level', 0) if len(targets) > 0 else 0

    if not all([entry_price, stop_loss, exit_price]):
        if debug:
            print(f"  [DEBUG] Missing prices: entry={entry_price}, stop={stop_loss}, exit={exit_price}")
        return None

    # Extract date
    try:
        exit_dt = pd.to_datetime(exit_time_str)
        date_str = exit_dt.strftime('%Y-%m-%d')
    except:
        if debug:
            print(f"  [DEBUG] Failed to parse exit time: {exit_time_str}")
        return None

    # Load 1m data
    df_1m = load_1m_data(symbol, date_str)
    if df_1m.empty:
        if debug:
            print(f"  [DEBUG] No 1m data for {symbol} on {date_str}")
        return None

    # Handle timezone
    if df_1m['timestamp'].dt.tz is not None:
        exit_dt = exit_dt.tz_localize('Asia/Kolkata')

    # Get entry time from plan (simpler and more reliable)
    entry_time_str = plan.get('decision_ts', '') or plan.get('price', '')
    if not entry_time_str:
        if debug:
            print(f"  [DEBUG] No entry timestamp in plan")
        return None

    try:
        entry_dt = pd.to_datetime(entry_time_str)
        if df_1m['timestamp'].dt.tz is not None:
            entry_dt = entry_dt.tz_localize('Asia/Kolkata')
    except:
        if debug:
            print(f"  [DEBUG] Failed to parse entry time: {entry_time_str}")
        return None

    # Get bars between entry and exit
    trade_bars = df_1m[(df_1m['timestamp'] >= entry_dt) & (df_1m['timestamp'] <= exit_dt)].copy()

    if trade_bars.empty:
        if debug:
            print(f"  [DEBUG] No bars between entry ({entry_dt}) and exit ({exit_dt})")
            print(f"  [DEBUG] Data range: {df_1m['timestamp'].min()} to {df_1m['timestamp'].max()}")
        return None

    # Calculate risk
    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return None

    # Test different stop widths
    results = {
        'current_stop_R': 1.0,  # Current stop
        'wider_1.5R_saved': False,
        'wider_2.0R_saved': False,
        'max_adverse_move_R': 0.0,
        'reversed_after_sl': False,
        'hit_t1_before_sl': False,
        'minutes_to_sl': 0,
    }

    if side == 'long':
        # Max adverse excursion
        max_ae = (trade_bars['low'].min() - entry_price) / risk
        results['max_adverse_move_R'] = float(max_ae)

        # Would wider stops have saved it?
        if max_ae > -1.5:  # Didn't go below 1.5R
            results['wider_1.5R_saved'] = True
        if max_ae > -2.0:  # Didn't go below 2R
            results['wider_2.0R_saved'] = True

        # Did it hit T1 before SL?
        if t1_target > 0:
            hit_t1 = (trade_bars['high'] >= t1_target).any()
            results['hit_t1_before_sl'] = hit_t1

        # Did price reverse after SL hit?
        after_sl = df_1m[df_1m['timestamp'] > exit_dt].copy()
        if not after_sl.empty:
            # Check if price went back above entry within next hour
            after_sl_1h = after_sl[after_sl['timestamp'] <= exit_dt + pd.Timedelta(hours=1)]
            if not after_sl_1h.empty:
                reversed = (after_sl_1h['high'] >= entry_price).any()
                results['reversed_after_sl'] = reversed

    else:  # short
        # Max adverse excursion
        max_ae = (entry_price - trade_bars['high'].max()) / risk
        results['max_adverse_move_R'] = float(max_ae)

        # Would wider stops have saved it?
        if max_ae > -1.5:
            results['wider_1.5R_saved'] = True
        if max_ae > -2.0:
            results['wider_2.0R_saved'] = True

        # Did it hit T1 before SL?
        if t1_target > 0:
            hit_t1 = (trade_bars['low'] <= t1_target).any()
            results['hit_t1_before_sl'] = hit_t1

        # Did price reverse after SL?
        after_sl = df_1m[df_1m['timestamp'] > exit_dt].copy()
        if not after_sl.empty:
            after_sl_1h = after_sl[after_sl['timestamp'] <= exit_dt + pd.Timedelta(hours=1)]
            if not after_sl_1h.empty:
                reversed = (after_sl_1h['low'] <= entry_price).any()
                results['reversed_after_sl'] = reversed

    # Time to SL
    results['minutes_to_sl'] = (exit_dt - entry_dt).total_seconds() / 60

    return results

def main():
    print("="*80)
    print("STOP LOSS PROBLEM ANALYSIS")
    print("="*80)
    print()

    # Find all hard_sl exits
    all_sessions = sorted([d for d in LOGS_DIR.glob("bt_*_20251105-035540") if d.is_dir()])

    print(f"Scanning {len(all_sessions)} sessions for hard_sl exits...")
    print()

    hard_sl_trades = []

    for session_dir in all_sessions:
        analytics_file = session_dir / "analytics.jsonl"
        if not analytics_file.exists():
            continue

        try:
            with open(analytics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            trade = json.loads(line)
                            if trade.get('stage') == 'EXIT' and trade.get('reason') == 'hard_sl':
                                trade['session_dir'] = session_dir
                                hard_sl_trades.append(trade)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            continue

    print(f"Found {len(hard_sl_trades)} hard_sl exits")
    print()

    if not hard_sl_trades:
        print("No hard_sl exits found!")
        return

    print(f"Analyzing stop loss behavior using 1m data...")
    print()

    results = []
    debug_first = True
    for i, trade in enumerate(hard_sl_trades):
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(hard_sl_trades)} ({(i+1)/len(hard_sl_trades)*100:.0f}%)")

        session_dir = trade.pop('session_dir')

        # Debug first trade
        is_debug = debug_first
        sl_analysis = analyze_stop_loss(trade, session_dir, debug=is_debug)

        # Debug first failure
        if sl_analysis is None and debug_first:
            print(f"\n[DEBUG] First failure:")
            print(f"  Symbol: {trade.get('symbol')}")
            print(f"  Trade ID: {trade.get('trade_id')}")
            print(f"  Session: {session_dir.name}")
            print(f"  Timestamp: {trade.get('timestamp')}")
            debug_first = False

        if sl_analysis:
            results.append({
                'symbol': trade.get('symbol', ''),
                'setup_type': trade.get('setup_type', ''),
                'regime': trade.get('regime', ''),
                'bias': trade.get('bias', 'long'),
                'pnl': trade.get('pnl', 0),
                'exit_time': trade.get('timestamp', ''),
                **sl_analysis
            })

    print()
    print(f"Successfully analyzed {len(results)}/{len(hard_sl_trades)} hard_sl exits ({len(results)/len(hard_sl_trades)*100:.1f}%)")
    print()

    if not results:
        print("No valid results!")
        return

    df = pd.DataFrame(results)

    # Analysis
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print()

    print(f"Total hard_sl exits analyzed: {len(df)}")
    print(f"Average max adverse move: {df['max_adverse_move_R'].mean():.2f}R")
    print(f"Median max adverse move: {df['max_adverse_move_R'].median():.2f}R")
    print()

    saved_15r = (df['wider_1.5R_saved']).sum()
    saved_20r = (df['wider_2.0R_saved']).sum()
    reversed = (df['reversed_after_sl']).sum()
    hit_t1 = (df['hit_t1_before_sl']).sum()

    print("STOP WIDTH ANALYSIS:")
    print(f"  Trades saved by 1.5R stop: {saved_15r:3d} ({saved_15r/len(df)*100:5.1f}%)")
    print(f"  Trades saved by 2.0R stop: {saved_20r:3d} ({saved_20r/len(df)*100:5.1f}%)")
    print()

    print("FALSE STOP-OUTS:")
    print(f"  Reversed after SL hit: {reversed:3d} ({reversed/len(df)*100:5.1f}%)")
    print(f"  Hit T1 before SL: {hit_t1:3d} ({hit_t1/len(df)*100:5.1f}%)")
    print()

    print(f"Average time to SL: {df['minutes_to_sl'].mean():.1f} minutes")
    print()

    # By setup
    print("="*80)
    print("BY SETUP TYPE")
    print("="*80)
    print()

    for setup in sorted(df['setup_type'].unique()):
        setup_df = df[df['setup_type'] == setup]
        saved_15 = (setup_df['wider_1.5R_saved']).sum()
        pct_15 = saved_15 / len(setup_df) * 100
        avg_mae = setup_df['max_adverse_move_R'].mean()

        print(f"{setup:30s} {len(setup_df):2d} SLs, Avg MAE: {avg_mae:5.2f}R, 1.5R saves: {saved_15:2d} ({pct_15:5.1f}%)")

    print()

    # By regime
    print("="*80)
    print("BY REGIME")
    print("="*80)
    print()

    for regime in sorted(df['regime'].unique()):
        regime_df = df[df['regime'] == regime]
        saved_15 = (regime_df['wider_1.5R_saved']).sum()
        pct_15 = saved_15 / len(regime_df) * 100
        avg_mae = regime_df['max_adverse_move_R'].mean()

        print(f"{regime:15s} {len(regime_df):2d} SLs, Avg MAE: {avg_mae:5.2f}R, 1.5R saves: {saved_15:2d} ({pct_15:5.1f}%)")

    print()

    # Recommendation
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()

    pct_saved_15 = saved_15r / len(df) * 100
    pct_saved_20 = saved_20r / len(df) * 100
    pct_reversed = reversed / len(df) * 100

    print(f"Current stop: 1.0R")
    print(f"Trades saved by widening to 1.5R: {pct_saved_15:.1f}%")
    print(f"Trades saved by widening to 2.0R: {pct_saved_20:.1f}%")
    print(f"False stop-outs (reversed after SL): {pct_reversed:.1f}%")
    print()

    if pct_saved_15 > 40:
        print("STRONG CASE: Increase stops to 1.5R")
        print()
        print("Rationale:")
        print(f"  - {pct_saved_15:.0f}% of SL hits would be saved with 1.5R stops")
        print(f"  - {pct_reversed:.0f}% reversed after SL (false stop-outs)")
        print(f"  - Current stops too tight for NSE volatility")
        print()
        print("Additional recommendation:")
        print("  - Move SL to BE only AFTER T1 hit (not immediately)")
        print(f"  - {hit_t1} trades hit T1 before SL - could have been winners")
    elif pct_saved_15 > 25:
        print("MODERATE CASE: Test 1.5R stops on specific setups")
        print()
        print("Identify worst offenders above and widen stops for those setups only")
    else:
        print("CURRENT STOPS ADEQUATE")
        print()
        print("Focus on other improvements (entry timing, setup filters)")

    print()

if __name__ == "__main__":
    main()
