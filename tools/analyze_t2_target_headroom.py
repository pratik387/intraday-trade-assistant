#!/usr/bin/env python
"""
Analyze T2 target headroom - Are we exiting too early?

This script checks all T2 exits and measures how much further price moved
after we exited. Uses 1m data for precise spike testing.

Questions to answer:
1. What % of T2 exits had significant continuation (>0.5R, >1R, >2R)?
2. What's the average max favorable excursion after T2 exit?
3. Which setups/regimes have the most headroom?
4. Should we increase T2 targets (from 2R to 2.5R or 3R)?
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
CACHE_DIR = ROOT / "cache"

def load_1m_data(symbol: str, date: str) -> pd.DataFrame:
    """Load 1m data for a symbol on a specific date"""
    # symbol format: NSE:SYMBOL -> SYMBOL.NS
    if symbol.startswith('NSE:'):
        symbol_clean = symbol.replace('NSE:', '')
    else:
        symbol_clean = symbol

    # Try to find the symbol's cache file
    symbol_cache = CACHE_DIR / symbol_clean
    if not symbol_cache.exists():
        return pd.DataFrame()

    feather_1m = symbol_cache / "1m.feather"
    if not feather_1m.exists():
        return pd.DataFrame()

    try:
        df = pd.read_feather(feather_1m)

        # Normalize timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif df.index.name == 'timestamp':
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter by date
        df['date'] = df['timestamp'].dt.date
        target_date = pd.to_datetime(date).date()
        df = df[df['date'] == target_date].copy()

        return df.sort_values('timestamp')
    except Exception as e:
        print(f"Error loading 1m data for {symbol} on {date}: {e}")
        return pd.DataFrame()

def analyze_t2_exit(trade: dict) -> dict:
    """
    Analyze a T2 exit to see how much headroom was left.

    Returns dict with:
    - max_favorable_excursion_R: How much further price went in our favor
    - max_adverse_excursion_R: How much it went against us after exit
    - bars_to_max_fe: How many bars until max favorable
    - peak_time: When the peak occurred
    """
    symbol = trade.get('symbol', '')
    session = trade.get('session', '')
    exit_time_str = trade.get('exit_time', '')
    exit_price = trade.get('exit_price', 0)
    entry_price = trade.get('entry_price', 0)
    side = trade.get('side', 'long')
    stop_loss = trade.get('stop_loss', 0)

    # Extract date from session name (bt_YYYYMMDD_* or bt_YYYY-MM-DD_*)
    if 'bt_' in session:
        date_part = session.split('_')[1]
        if '-' in date_part:
            date_str = date_part  # Already YYYY-MM-DD
        else:
            # YYYYMMDD -> YYYY-MM-DD
            date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
    else:
        return None

    # Load 1m data
    df_1m = load_1m_data(symbol, date_str)
    if df_1m.empty:
        return None

    # Find exit time
    try:
        exit_time = pd.to_datetime(f"{date_str} {exit_time_str}")
    except:
        return None

    # Get all bars after exit (until 15:30)
    after_exit = df_1m[df_1m['timestamp'] > exit_time].copy()
    after_exit = after_exit[after_exit['timestamp'].dt.time <= pd.Timestamp('15:30').time()]

    if after_exit.empty:
        return None

    # Calculate R-multiple for each bar
    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return None

    if side == 'long':
        # For long, favorable = higher prices
        after_exit['r_multiple'] = (after_exit['high'] - exit_price) / risk
        max_fe = after_exit['r_multiple'].max()
        max_ae = (after_exit['low'].min() - exit_price) / risk  # Adverse = lower prices

        max_fe_idx = after_exit['r_multiple'].idxmax()
        peak_bar = after_exit.loc[max_fe_idx]
    else:
        # For short, favorable = lower prices
        after_exit['r_multiple'] = (exit_price - after_exit['low']) / risk
        max_fe = after_exit['r_multiple'].max()
        max_ae = (exit_price - after_exit['high'].max()) / risk  # Adverse = higher prices

        max_fe_idx = after_exit['r_multiple'].idxmax()
        peak_bar = after_exit.loc[max_fe_idx]

    # Calculate bars to max FE
    bars_to_max_fe = len(after_exit[after_exit['timestamp'] <= peak_bar['timestamp']])

    return {
        'max_favorable_excursion_R': float(max_fe),
        'max_adverse_excursion_R': float(max_ae),
        'bars_to_max_fe': bars_to_max_fe,
        'peak_time': peak_bar['timestamp'].strftime('%H:%M'),
        'minutes_to_peak': (peak_bar['timestamp'] - exit_time).total_seconds() / 60
    }

def main():
    print("="*80)
    print("ANALYZING T2 TARGET HEADROOM")
    print("="*80)
    print()
    print("Loading all T2 exits from recent runs...")
    print()

    # Find all sessions (last 130 days for speed)
    all_sessions = sorted([d for d in LOGS_DIR.glob("bt_*") if d.is_dir()])[-130:]

    t2_exits = []
    total_exits = 0

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
                            if trade.get('stage') == 'EXIT':
                                total_exits += 1
                                exit_reason = trade.get('reason', '')

                                # Check if it's a T2 exit
                                if exit_reason == 'target_t2':
                                    trade['session'] = session_dir.name
                                    t2_exits.append(trade)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            continue

    print(f"Found {len(t2_exits)} T2 exits out of {total_exits} total exits ({len(t2_exits)/total_exits*100:.1f}%)")
    print()

    if not t2_exits:
        print("No T2 exits found!")
        return

    print(f"Analyzing headroom for {len(t2_exits)} T2 exits...")
    print(f"This may take a few minutes (loading 1m data for each trade)...")
    print()

    # Analyze each T2 exit
    results = []
    analyzed_count = 0

    for i, trade in enumerate(t2_exits):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(t2_exits)} ({i/len(t2_exits)*100:.0f}%)")

        headroom = analyze_t2_exit(trade)
        if headroom:
            results.append({
                'symbol': trade.get('symbol', ''),
                'setup_type': trade.get('setup_type', ''),
                'regime': trade.get('regime', ''),
                'side': trade.get('side', ''),
                'pnl': trade.get('pnl', 0),
                'exit_time': trade.get('exit_time', ''),
                'session': trade.get('session', ''),
                **headroom
            })
            analyzed_count += 1

    print()
    print(f"Successfully analyzed {analyzed_count}/{len(t2_exits)} T2 exits ({analyzed_count/len(t2_exits)*100:.1f}%)")
    print()

    if not results:
        print("No valid analysis results!")
        return

    df = pd.DataFrame(results)

    # ============================================================================
    # ANALYSIS 1: Overall Headroom Statistics
    # ============================================================================
    print("="*80)
    print("OVERALL HEADROOM STATISTICS")
    print("="*80)
    print()

    print(f"Total T2 exits analyzed: {len(df)}")
    print(f"Average max favorable excursion: {df['max_favorable_excursion_R'].mean():.2f}R")
    print(f"Median max favorable excursion: {df['max_favorable_excursion_R'].median():.2f}R")
    print(f"75th percentile: {df['max_favorable_excursion_R'].quantile(0.75):.2f}R")
    print(f"90th percentile: {df['max_favorable_excursion_R'].quantile(0.90):.2f}R")
    print()

    print("Distribution of additional movement after T2 exit:")
    for threshold in [0.5, 1.0, 1.5, 2.0, 3.0]:
        count = (df['max_favorable_excursion_R'] >= threshold).sum()
        pct = count / len(df) * 100
        print(f"  >={threshold}R additional: {count:3d} trades ({pct:5.1f}%)")

    print()
    print(f"Average time to peak: {df['minutes_to_peak'].mean():.1f} minutes")
    print(f"Median time to peak: {df['minutes_to_peak'].median():.1f} minutes")
    print()

    # ============================================================================
    # ANALYSIS 2: Headroom by Setup Type
    # ============================================================================
    print("="*80)
    print("HEADROOM BY SETUP TYPE")
    print("="*80)
    print()

    for setup in sorted(df['setup_type'].unique()):
        setup_df = df[df['setup_type'] == setup]
        avg_headroom = setup_df['max_favorable_excursion_R'].mean()
        median_headroom = setup_df['max_favorable_excursion_R'].median()
        count_1r_plus = (setup_df['max_favorable_excursion_R'] >= 1.0).sum()
        pct_1r_plus = count_1r_plus / len(setup_df) * 100

        print(f"{setup:30s}")
        print(f"  Count: {len(setup_df):3d} trades")
        print(f"  Avg headroom: {avg_headroom:5.2f}R")
        print(f"  Median headroom: {median_headroom:5.2f}R")
        print(f"  >=1R additional: {count_1r_plus:3d} ({pct_1r_plus:5.1f}%)")
        print()

    # ============================================================================
    # ANALYSIS 3: Headroom by Regime
    # ============================================================================
    print("="*80)
    print("HEADROOM BY REGIME")
    print("="*80)
    print()

    for regime in sorted(df['regime'].unique()):
        regime_df = df[df['regime'] == regime]
        avg_headroom = regime_df['max_favorable_excursion_R'].mean()
        median_headroom = regime_df['max_favorable_excursion_R'].median()
        count_1r_plus = (regime_df['max_favorable_excursion_R'] >= 1.0).sum()
        pct_1r_plus = count_1r_plus / len(regime_df) * 100

        print(f"{regime:15s}")
        print(f"  Count: {len(regime_df):3d} trades")
        print(f"  Avg headroom: {avg_headroom:5.2f}R")
        print(f"  Median headroom: {median_headroom:5.2f}R")
        print(f"  >=1R additional: {count_1r_plus:3d} ({pct_1r_plus:5.1f}%)")
        print()

    # ============================================================================
    # ANALYSIS 4: Top Headroom Examples
    # ============================================================================
    print("="*80)
    print("TOP 10 TRADES WITH MOST HEADROOM")
    print("="*80)
    print()

    top_headroom = df.nlargest(10, 'max_favorable_excursion_R')

    for idx, row in top_headroom.iterrows():
        print(f"{row['symbol']:20s} {row['setup_type']:20s} {row['regime']:10s}")
        print(f"  Exit time: {row['exit_time']}, Peak time: {row['peak_time']} ({row['minutes_to_peak']:.0f} min later)")
        print(f"  Additional move: {row['max_favorable_excursion_R']:.2f}R (P&L: Rs.{row['pnl']:.0f})")
        print()

    # ============================================================================
    # ANALYSIS 5: Recommendation
    # ============================================================================
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()

    avg_headroom = df['max_favorable_excursion_R'].mean()
    median_headroom = df['max_favorable_excursion_R'].median()
    pct_1r_plus = (df['max_favorable_excursion_R'] >= 1.0).sum() / len(df) * 100
    pct_2r_plus = (df['max_favorable_excursion_R'] >= 2.0).sum() / len(df) * 100

    print(f"Current T2 target: 2.0R (2x risk)")
    print()
    print(f"Key findings:")
    print(f"  - Average additional move after T2 exit: {avg_headroom:.2f}R")
    print(f"  - {pct_1r_plus:.1f}% of T2 exits had >=1R additional movement")
    print(f"  - {pct_2r_plus:.1f}% of T2 exits had >=2R additional movement")
    print()

    if avg_headroom > 1.0 and pct_1r_plus > 50:
        print("✅ STRONG CASE FOR INCREASING T2 TARGET")
        print()
        print("Recommendation: Increase T2 from 2.0R to 3.0R")
        print()
        print("Rationale:")
        print(f"  - More than 50% of trades ({pct_1r_plus:.0f}%) continue >=1R beyond current T2")
        print(f"  - Average headroom of {avg_headroom:.2f}R suggests significant money left on table")
        print(f"  - This could increase T2 exit P&L by ~{avg_headroom * 100:.0f}% on average")
        print()
        print("Alternative: Use trailing stop after T2 instead of fixed T2 exit")

    elif avg_headroom > 0.5 and pct_1r_plus > 30:
        print("⚠️  MODERATE CASE FOR INCREASING T2 TARGET")
        print()
        print("Recommendation: Test T2 increase from 2.0R to 2.5R")
        print()
        print("Rationale:")
        print(f"  - Significant portion ({pct_1r_plus:.0f}%) continue >=1R beyond current T2")
        print(f"  - Average headroom of {avg_headroom:.2f}R shows some money left on table")
        print(f"  - Conservative increase to 2.5R could capture {avg_headroom * 50:.0f}% of headroom")
        print()
        print("Risk: Some trades may reverse before hitting 2.5R")

    else:
        print("✅ CURRENT T2 TARGET (2.0R) IS OPTIMAL")
        print()
        print("Recommendation: Keep T2 at 2.0R")
        print()
        print("Rationale:")
        print(f"  - Only {pct_1r_plus:.0f}% of trades continue >=1R beyond current T2")
        print(f"  - Average headroom of {avg_headroom:.2f}R is modest")
        print(f"  - Risk of reversal outweighs potential additional gains")
        print()
        print("Consider: Trailing stop after T2 to capture occasional runners")

    print()
    print("="*80)

if __name__ == "__main__":
    main()
