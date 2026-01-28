#!/usr/bin/env python
"""
Analyze T2 target headroom - Simplified version using events.jsonl

This loads DECISION events (which have stop/entry/target info) and EXIT events
(from analytics.jsonl) to measure post-T2 price movement.
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

    # Try archive location first (cache/ohlcv_archive/SYMBOL.NS/SYMBOL.NS_1minutes.feather)
    archive_dir = CACHE_DIR / "ohlcv_archive" / f"{symbol_clean}.NS"
    feather_1m = None

    # DEBUG
    # print(f"      Looking for: {archive_dir}")

    if archive_dir.exists():
        archive_feather = archive_dir / f"{symbol_clean}.NS_1minutes.feather"
        # print(f"      Checking: {archive_feather}")
        if archive_feather.exists():
            feather_1m = archive_feather
            # print(f"      FOUND archive file")

    # Fallback to old location (cache/SYMBOL/1m.feather)
    if not feather_1m:
        symbol_cache = CACHE_DIR / symbol_clean
        if symbol_cache.exists():
            old_feather = symbol_cache / "1m.feather"
            if old_feather.exists():
                feather_1m = old_feather
                # print(f"      FOUND old location file")

    # No data found
    if not feather_1m:
        # DEBUG: Show what we looked for
        print(f"      Tried: {archive_dir} (exists={archive_dir.exists()})")
        if archive_dir.exists():
            print(f"      Tried: {archive_dir / f'{symbol_clean}.NS_1minutes.feather'} (exists={False})")
        return pd.DataFrame()

    try:
        df = pd.read_feather(feather_1m)

        # Handle different column names (timestamp vs date)
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

def analyze_headroom(trade, session_dir):
    """Analyze headroom for a single T2 exit"""
    trade_id = trade.get('trade_id', '')
    symbol = trade.get('symbol', '')
    exit_time_str = trade.get('timestamp', '')  # e.g., "2023-12-04 12:50:00"
    exit_price = trade.get('exit_price', 0)

    # Load trade plan to get entry price and stop loss
    plan = load_trade_plan(session_dir, trade_id)
    if not plan:
        print(f"    FAIL {symbol}: No trade plan found for {trade_id}")
        return None

    entry_price = plan.get('entry', {}).get('reference', 0)
    stop_loss = plan.get('stop', {}).get('hard', 0)
    side = plan.get('bias', 'long')

    if not all([entry_price, stop_loss, exit_price]):
        print(f"    FAIL {symbol}: Missing prices (entry={entry_price}, stop={stop_loss}, exit={exit_price})")
        return None

    # Extract date from exit timestamp
    try:
        exit_dt = pd.to_datetime(exit_time_str)
        date_str = exit_dt.strftime('%Y-%m-%d')
    except:
        print(f"    FAIL {symbol}: Failed to parse exit time {exit_time_str}")
        return None

    # Load 1m data
    df_1m = load_1m_data(symbol, date_str)
    if df_1m.empty:
        print(f"    FAIL {symbol}: No 1m data for {date_str}")
        return None

    print(f"    OK {symbol}: Loaded {len(df_1m)} bars for {date_str}")

    # Handle timezone mismatch - make exit_dt tz-aware if df has tz
    if df_1m['timestamp'].dt.tz is not None:
        # Make exit_dt timezone-aware (localize to IST)
        exit_dt = exit_dt.tz_localize('Asia/Kolkata')

    # Get all bars after exit
    after_exit = df_1m[df_1m['timestamp'] > exit_dt].copy()
    after_exit = after_exit[after_exit['timestamp'].dt.time <= pd.Timestamp('15:30').time()]

    if after_exit.empty:
        return None

    # Calculate R-multiple
    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return None

    if side == 'long':
        after_exit['r_multiple'] = (after_exit['high'] - exit_price) / risk
        max_fe = after_exit['r_multiple'].max()
        max_ae = (after_exit['low'].min() - exit_price) / risk

        max_fe_idx = after_exit['r_multiple'].idxmax()
        peak_bar = after_exit.loc[max_fe_idx]
    else:
        after_exit['r_multiple'] = (exit_price - after_exit['low']) / risk
        max_fe = after_exit['r_multiple'].max()
        max_ae = (exit_price - after_exit['high'].max()) / risk

        max_fe_idx = after_exit['r_multiple'].idxmax()
        peak_bar = after_exit.loc[max_fe_idx]

    bars_to_max_fe = len(after_exit[after_exit['timestamp'] <= peak_bar['timestamp']])
    minutes_to_peak = (peak_bar['timestamp'] - exit_dt).total_seconds() / 60

    return {
        'max_favorable_excursion_R': float(max_fe),
        'max_adverse_excursion_R': float(max_ae),
        'bars_to_max_fe': bars_to_max_fe,
        'peak_time': peak_bar['timestamp'].strftime('%H:%M'),
        'minutes_to_peak': minutes_to_peak
    }

def main():
    print("="*80)
    print("T2 TARGET HEADROOM ANALYSIS (SPIKE TEST)")
    print("="*80)
    print()

    # Find recent sessions
    all_sessions = sorted([d for d in LOGS_DIR.glob("bt_*") if d.is_dir()])[-130:]

    print(f"Scanning {len(all_sessions)} recent sessions for T2 exits...")
    print()

    t2_trades = []

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
                            if trade.get('stage') == 'EXIT' and trade.get('reason') == 'target_t2':
                                trade['session_dir'] = session_dir
                                t2_trades.append(trade)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            continue

    print(f"Found {len(t2_trades)} T2 exits")
    print()

    if not t2_trades:
        print("No T2 exits found!")
        return

    print(f"Analyzing headroom using 1m spike data...")
    print()

    results = []
    for i, trade in enumerate(t2_trades):
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(t2_trades)} ({(i+1)/len(t2_trades)*100:.0f}%)")

        session_dir = trade.pop('session_dir')
        headroom = analyze_headroom(trade, session_dir)

        if headroom:
            results.append({
                'symbol': trade.get('symbol', ''),
                'setup_type': trade.get('setup_type', ''),
                'regime': trade.get('regime', ''),
                'bias': trade.get('bias', 'long'),
                'pnl': trade.get('pnl', 0),
                'exit_time': trade.get('timestamp', ''),
                **headroom
            })

    print()
    print(f"Successfully analyzed {len(results)}/{len(t2_trades)} T2 exits ({len(results)/len(t2_trades)*100:.1f}%)")
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

    print(f"Average headroom: {df['max_favorable_excursion_R'].mean():.2f}R")
    print(f"Median headroom:  {df['max_favorable_excursion_R'].median():.2f}R")
    print(f"75th percentile:  {df['max_favorable_excursion_R'].quantile(0.75):.2f}R")
    print(f"90th percentile:  {df['max_favorable_excursion_R'].quantile(0.90):.2f}R")
    print()

    print("Trades with significant continuation:")
    for threshold in [0.5, 1.0, 1.5, 2.0]:
        count = (df['max_favorable_excursion_R'] >= threshold).sum()
        pct = count / len(df) * 100
        print(f"  >={threshold}R: {count:3d} ({pct:5.1f}%)")

    print()
    print(f"Avg time to peak: {df['minutes_to_peak'].mean():.1f} minutes")
    print()

    # By setup
    print("="*80)
    print("HEADROOM BY SETUP")
    print("="*80)
    print()

    for setup in sorted(df['setup_type'].unique()):
        setup_df = df[df['setup_type'] == setup]
        avg = setup_df['max_favorable_excursion_R'].mean()
        pct_1r = (setup_df['max_favorable_excursion_R'] >= 1.0).sum() / len(setup_df) * 100
        print(f"{setup:30s} {len(setup_df):3d} trades, avg: {avg:5.2f}R, >=1R: {pct_1r:5.1f}%")

    print()

    # Recommendation
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()

    avg = df['max_favorable_excursion_R'].mean()
    pct_1r = (df['max_favorable_excursion_R'] >= 1.0).sum() / len(df) * 100

    print(f"Current T2: 2.0R")
    print(f"Avg continuation: {avg:.2f}R")
    print(f"Trades with >=1R more: {pct_1r:.1f}%")
    print()

    if avg > 1.0 and pct_1r > 50:
        print("STRONG CASE: Increase T2 to 3.0R")
    elif avg > 0.5 and pct_1r > 30:
        print("MODERATE CASE: Test T2 at 2.5R")
    else:
        print("KEEP T2 at 2.0R")

    print()

if __name__ == "__main__":
    main()
