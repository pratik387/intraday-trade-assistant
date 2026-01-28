"""
FHM Long 1-Minute Bar-by-Bar Simulation

Tests different SL and target parameters using actual 1m price data
to validate whether widening SL/targets would improve performance.
"""

import json
import os
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

BACKTEST_DIR = 'backtest_20251210-123728_extracted'
OHLCV_DIR = 'cache/ohlcv_archive'


def load_fhm_trades():
    """Load all FHM long triggered trades from backtest events."""
    all_trades = []

    for date_folder in sorted(os.listdir(BACKTEST_DIR)):
        events_file = f'{BACKTEST_DIR}/{date_folder}/events.jsonl'
        if not os.path.exists(events_file):
            continue

        decisions = {}
        triggers = {}
        exits = {}

        with open(events_file) as f:
            for line in f:
                event = json.loads(line)
                trade_id = event.get('trade_id')
                evt_type = event.get('type')

                if evt_type == 'DECISION':
                    decisions[trade_id] = event
                elif evt_type == 'TRIGGER':
                    triggers[trade_id] = event
                elif evt_type == 'EXIT':
                    exits[trade_id] = event

        # Find FHM long trades
        for trade_id, decision_event in decisions.items():
            decision = decision_event.get('decision', {})
            plan = decision_event.get('plan', {})
            setup_type = decision.get('setup_type', '')

            if 'first_hour_momentum_long' in setup_type.lower():
                if trade_id in triggers:
                    trigger_event = triggers[trade_id]
                    exit_event = exits.get(trade_id, {})
                    exit_data = exit_event.get('exit', {}) if isinstance(exit_event, dict) else {}

                    # Parse targets list
                    targets = plan.get('targets', [])
                    t1_level = None
                    t2_level = None
                    for t in targets:
                        if t.get('name') == 'T1':
                            t1_level = t.get('level')
                        elif t.get('name') == 'T2':
                            t2_level = t.get('level')

                    all_trades.append({
                        'date': date_folder,
                        'trade_id': trade_id,
                        'symbol': decision_event.get('symbol'),
                        'entry_ts': trigger_event.get('ts'),
                        'entry_price': trigger_event.get('trigger', {}).get('actual_price'),
                        'sl': plan.get('stop', {}).get('price'),
                        't1': t1_level,
                        't2': t2_level,
                        'qty': plan.get('sizing', {}).get('qty'),
                        'setup_type': setup_type,
                        'exit_reason': exit_data.get('reason', ''),
                        'actual_pnl': exit_data.get('pnl', 0)
                    })

    return all_trades


def load_1m_data(symbol):
    """Load 1m OHLCV data for a symbol."""
    # Convert NSE:SYMBOL to SYMBOL.NS
    if ':' in symbol:
        ticker = symbol.split(':')[1] + '.NS'
    else:
        ticker = symbol + '.NS'

    ohlcv_path = f'{OHLCV_DIR}/{ticker}/{ticker}_1minutes.feather'
    if not os.path.exists(ohlcv_path):
        return None

    df = pd.read_feather(ohlcv_path)

    # Normalize datetime column
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        return None

    # Remove timezone if present
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)

    df = df.set_index('datetime').sort_index()
    return df


def simulate_trade(trade, ohlcv_df, sl_pct, t1_rr, verbose=False):
    """
    Simulate a single trade bar-by-bar with given SL% and T1 R:R.

    Returns:
        dict: result with outcome, exit_reason, pnl, etc.
    """
    entry_price = trade['entry_price']
    entry_ts = trade['entry_ts']
    qty = trade['qty'] or 1

    if not entry_price or not entry_ts:
        return None

    # Parse entry timestamp
    try:
        entry_dt = pd.to_datetime(entry_ts)
        if entry_dt.tzinfo is not None:
            entry_dt = entry_dt.tz_localize(None)
    except:
        return None

    # Calculate SL price based on SL%
    sl_price = entry_price * (1 - sl_pct / 100)
    sl_distance = entry_price - sl_price

    # Calculate T1 based on R:R
    t1_price = entry_price + (sl_distance * t1_rr)

    # Filter OHLCV to bars after entry
    try:
        bars_after_entry = ohlcv_df.loc[entry_dt:]
    except:
        return None

    if len(bars_after_entry) == 0:
        return None

    # Walk bar-by-bar
    mfe = 0  # Max Favorable Excursion
    mae = 0  # Max Adverse Excursion
    exit_reason = None
    exit_price = None
    bars_held = 0

    for bar_dt, bar in bars_after_entry.iterrows():
        bars_held += 1

        # Check if within same trading day
        if bar_dt.date() != entry_dt.date():
            # EOD exit
            exit_reason = 'eod'
            exit_price = bar['close']
            break

        # Check SL hit (using low)
        if bar['low'] <= sl_price:
            exit_reason = 'sl'
            exit_price = sl_price
            break

        # Check T1 hit (using high)
        if bar['high'] >= t1_price:
            exit_reason = 't1'
            exit_price = t1_price
            break

        # Track MFE/MAE
        bar_mfe = (bar['high'] - entry_price) / entry_price * 100
        bar_mae = (entry_price - bar['low']) / entry_price * 100
        mfe = max(mfe, bar_mfe)
        mae = max(mae, bar_mae)

        # Safety: don't process more than 400 bars (full day)
        if bars_held > 400:
            exit_reason = 'timeout'
            exit_price = bar['close']
            break

    if exit_reason is None:
        return None

    pnl = (exit_price - entry_price) * qty
    pnl_pct = (exit_price - entry_price) / entry_price * 100

    return {
        'exit_reason': exit_reason,
        'exit_price': exit_price,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'bars_held': bars_held,
        'mfe': mfe,
        'mae': mae,
        't1_price': t1_price,
        'sl_price': sl_price
    }


def run_simulation(trades, sl_pct, t1_rr, name):
    """Run simulation across all trades with given parameters."""
    results = []
    skipped = 0

    # Cache OHLCV data
    ohlcv_cache = {}

    for trade in trades:
        symbol = trade['symbol']

        # Load OHLCV if not cached
        if symbol not in ohlcv_cache:
            ohlcv_cache[symbol] = load_1m_data(symbol)

        ohlcv_df = ohlcv_cache[symbol]
        if ohlcv_df is None:
            skipped += 1
            continue

        result = simulate_trade(trade, ohlcv_df, sl_pct, t1_rr)
        if result:
            result['trade_id'] = trade['trade_id']
            result['symbol'] = symbol
            result['actual_exit'] = trade['exit_reason']
            result['actual_pnl'] = trade['actual_pnl']
            results.append(result)
        else:
            skipped += 1

    return results, skipped


def analyze_results(results, name):
    """Analyze simulation results."""
    if not results:
        return None

    n_trades = len(results)

    # Outcome breakdown
    outcomes = defaultdict(int)
    for r in results:
        outcomes[r['exit_reason']] += 1

    # P&L stats
    total_pnl = sum(r['pnl'] for r in results)
    winners = [r for r in results if r['pnl'] > 0]
    losers = [r for r in results if r['pnl'] <= 0]

    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0
    avg_win = sum(r['pnl'] for r in winners) / len(winners) if winners else 0
    avg_loss = sum(r['pnl'] for r in losers) / len(losers) if losers else 0
    avg_pnl = total_pnl / n_trades if n_trades > 0 else 0

    # MFE/MAE stats
    avg_mfe = sum(r['mfe'] for r in results) / n_trades
    avg_mae = sum(r['mae'] for r in results) / n_trades

    return {
        'name': name,
        'n_trades': n_trades,
        'outcomes': dict(outcomes),
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_mfe': avg_mfe,
        'avg_mae': avg_mae
    }


def main():
    print("=" * 70)
    print("FHM LONG: 1-MINUTE BAR-BY-BAR SIMULATION")
    print("=" * 70)
    print()

    # Load trades
    print("Loading FHM long trades from backtest...")
    trades = load_fhm_trades()
    print(f"Found {len(trades)} triggered FHM long trades")
    print()

    # Test scenarios
    scenarios = [
        # (SL%, T1 R:R, name)
        (0.8, 1.0, "Current (0.8% SL, 1.0R T1)"),
        (1.0, 1.0, "Wider SL (1.0% SL, 1.0R T1)"),
        (1.2, 1.0, "Wide SL (1.2% SL, 1.0R T1)"),
        (0.8, 1.2, "Higher T1 (0.8% SL, 1.2R T1)"),
        (1.0, 1.2, "Balanced (1.0% SL, 1.2R T1)"),
        (1.0, 1.5, "Aggressive T1 (1.0% SL, 1.5R T1)"),
        (1.2, 1.5, "Wide + Aggressive (1.2% SL, 1.5R T1)"),
    ]

    all_results = []

    for sl_pct, t1_rr, name in scenarios:
        print(f"Testing: {name}...")
        results, skipped = run_simulation(trades, sl_pct, t1_rr, name)

        if results:
            analysis = analyze_results(results, name)
            all_results.append(analysis)
            print(f"  Trades: {analysis['n_trades']} (skipped: {skipped})")
        else:
            print(f"  No valid results (skipped: {skipped})")

    # Print comparison table
    print()
    print("=" * 70)
    print("SIMULATION RESULTS COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Scenario':<35} {'Trades':>7} {'WR%':>7} {'Total P&L':>12} {'Avg P&L':>10} {'T1':>6} {'SL':>6} {'EOD':>6}")
    print("-" * 100)

    for r in all_results:
        outcomes = r['outcomes']
        print(f"{r['name']:<35} {r['n_trades']:>7} {r['win_rate']:>6.1f}% {r['total_pnl']:>11,.0f} {r['avg_pnl']:>10,.0f} "
              f"{outcomes.get('t1', 0):>6} {outcomes.get('sl', 0):>6} {outcomes.get('eod', 0):>6}")

    print()
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    if len(all_results) >= 2:
        baseline = all_results[0]
        best = max(all_results, key=lambda x: x['total_pnl'])

        if best['total_pnl'] > baseline['total_pnl']:
            improvement = best['total_pnl'] - baseline['total_pnl']
            print(f"Best scenario: {best['name']}")
            print(f"  - Improves P&L by Rs {improvement:,.0f}")
            print(f"  - Win rate: {best['win_rate']:.1f}% vs baseline {baseline['win_rate']:.1f}%")
            print(f"  - SL hits: {best['outcomes'].get('sl', 0)} vs baseline {baseline['outcomes'].get('sl', 0)}")
            print(f"  - T1 hits: {best['outcomes'].get('t1', 0)} vs baseline {baseline['outcomes'].get('t1', 0)}")
        else:
            print("Current parameters appear optimal or close to optimal")

    # Show MFE/MAE insights from first scenario
    if all_results:
        r = all_results[0]
        print()
        print(f"Price movement stats (from baseline):")
        print(f"  - Avg MFE (max favorable): {r['avg_mfe']:.2f}%")
        print(f"  - Avg MAE (max adverse): {r['avg_mae']:.2f}%")
        print(f"  - If avg MAE > SL%, many trades hit SL before recovering")


if __name__ == '__main__':
    main()
