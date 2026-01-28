#!/usr/bin/env python3
"""
Trigger Timeout Spike Test

Analyzes:
1. Enqueued decisions that never triggered (timeout analysis)
2. Time gap between enqueue (DECISION) and trigger (TRIGGER)
3. For SL hit trades: pattern analysis of enqueue -> trigger timing
4. Uses 1m OHLCV cache to spike test: would longer timeout increase trade count?

Usage:
    python tools/spike_test_trigger_timeout.py <backtest_extracted_dir> <cache_dir>

Example:
    python tools/spike_test_trigger_timeout.py \\
        backtest_20251109-125133_extracted/20251109-125133_full/20251109-125133 \\
        cache
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd

# Configuration
CURRENT_TRIGGER_TIMEOUT_MINUTES = 45  # From configuration
ALTERNATIVE_TIMEOUTS = [30, 45, 60, 90, 120]  # Test different timeouts


def parse_events(sessions_root):
    """Parse events.jsonl for DECISION and TRIGGER events."""

    decisions = []  # All DECISION events
    triggers = {}   # trade_id -> TRIGGER event
    exits = {}      # trade_id -> EXIT event (from analytics.jsonl)

    sessions_path = Path(sessions_root)

    for session_dir in sorted(sessions_path.iterdir()):
        if not session_dir.is_dir():
            continue

        session_date = session_dir.name
        events_file = session_dir / "events.jsonl"
        analytics_file = session_dir / "analytics.jsonl"

        # Parse events.jsonl
        if events_file.exists():
            with open(events_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        event_type = event.get('type')

                        if event_type == 'DECISION':
                            decisions.append({
                                'trade_id': event.get('trade_id'),
                                'symbol': event.get('symbol'),
                                'ts': event.get('ts'),
                                'session_date': session_date,
                                'setup_type': event.get('decision', {}).get('setup_type'),
                                'regime': event.get('decision', {}).get('regime'),
                                'rank_score': event.get('features', {}).get('rank_score'),
                                'acceptance_status': event.get('plan', {}).get('quality', {}).get('acceptance_status'),
                                'entry_price': event.get('plan', {}).get('price'),
                                'stop_price': event.get('plan', {}).get('stop', {}).get('hard'),
                                'direction': event.get('plan', {}).get('bias')
                            })

                        elif event_type == 'TRIGGER':
                            trade_id = event.get('trade_id')
                            triggers[trade_id] = {
                                'ts': event.get('ts'),
                                'trigger_price': event.get('trigger', {}).get('actual_price'),
                                'session_date': session_date
                            }

                    except json.JSONDecodeError:
                        continue

        # Parse analytics.jsonl for exits
        if analytics_file.exists():
            with open(analytics_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get('stage') == 'EXIT':
                            trade_id = event.get('trade_id')
                            exits[trade_id] = {
                                'exit_price': event.get('exit_price'),
                                'exit_reason': event.get('reason'),
                                'pnl': event.get('pnl'),
                                'entry_price': event.get('actual_entry_price'),
                                'strategy': event.get('strategy')
                            }
                    except json.JSONDecodeError:
                        continue

    return decisions, triggers, exits


def analyze_enqueue_to_trigger_timing(decisions, triggers):
    """Analyze time gap between enqueue (DECISION) and trigger (TRIGGER)."""

    triggered_trades = []
    never_triggered = []

    for decision in decisions:
        trade_id = decision['trade_id']
        enqueue_ts = datetime.fromisoformat(decision['ts'])

        if trade_id in triggers:
            trigger_ts = datetime.fromisoformat(triggers[trade_id]['ts'])
            time_gap_minutes = (trigger_ts - enqueue_ts).total_seconds() / 60

            triggered_trades.append({
                **decision,
                'trigger_ts': triggers[trade_id]['ts'],
                'time_gap_minutes': time_gap_minutes,
                'trigger_price': triggers[trade_id]['trigger_price']
            })
        else:
            never_triggered.append(decision)

    return triggered_trades, never_triggered


def analyze_sl_hit_timing(triggered_trades, exits):
    """For SL hit trades, analyze enqueue -> trigger timing pattern."""

    sl_hit_trades = []

    for trade in triggered_trades:
        trade_id = trade['trade_id']
        if trade_id in exits:
            exit_info = exits[trade_id]
            exit_reason = exit_info['exit_reason']

            # Check if hard SL hit
            if 'hard_sl' in exit_reason or 'stop_loss' in exit_reason:
                sl_hit_trades.append({
                    **trade,
                    'exit_price': exit_info['exit_price'],
                    'exit_reason': exit_reason,
                    'pnl': exit_info['pnl']
                })

    return sl_hit_trades


def get_1m_data(cache_dir, symbol, session_date):
    """Load 1-minute OHLCV data from cache."""

    # Remove NSE: prefix
    symbol_clean = symbol.replace('NSE:', '')
    symbol_ns = f"{symbol_clean}.NS"

    bars_file = Path(cache_dir) / "ohlcv_archive" / symbol_ns / f"{symbol_ns}_1minutes.feather"

    if not bars_file.exists():
        return None

    try:
        df = pd.read_feather(bars_file)

        # Normalize column names
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'Datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Datetime'])
        else:
            return None

        # Normalize OHLCV columns
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'high', 'low', 'close', 'volume']:
                col_map[col] = col_lower

        df = df.rename(columns=col_map)

        # Filter to session date
        session_dt = pd.to_datetime(session_date).date()
        df = df[df['timestamp'].dt.date == session_dt]

        if df.empty:
            return None

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        return None


def spike_test_timeout_extension(never_triggered, cache_dir, alternative_timeouts):
    """Spike test: Would extending timeout allow more trades?"""

    results = {timeout: {'additional_trades': 0, 'trades': []} for timeout in alternative_timeouts}

    for decision in never_triggered:
        symbol = decision['symbol']
        session_date = decision['session_date']
        enqueue_ts = datetime.fromisoformat(decision['ts'])
        entry_price = decision.get('entry_price')
        direction = decision.get('direction')

        if not entry_price or not direction:
            continue

        # Load 1m data
        df = get_1m_data(cache_dir, symbol, session_date)
        if df is None:
            continue

        # Check if price would have triggered within different timeouts
        for timeout_minutes in alternative_timeouts:
            timeout_ts = enqueue_ts + timedelta(minutes=timeout_minutes)

            # Filter bars after enqueue and before timeout
            bars_in_window = df[
                (df['timestamp'] > enqueue_ts) &
                (df['timestamp'] <= timeout_ts)
            ]

            if bars_in_window.empty:
                continue

            # Check if trigger would occur
            triggered = False
            trigger_time = None

            for idx, bar in bars_in_window.iterrows():
                if direction == 'long':
                    # Long: trigger if price >= entry_price
                    if bar['high'] >= entry_price:
                        triggered = True
                        trigger_time = bar['timestamp']
                        break
                else:
                    # Short: trigger if price <= entry_price
                    if bar['low'] <= entry_price:
                        triggered = True
                        trigger_time = bar['timestamp']
                        break

            if triggered:
                time_gap_minutes = (trigger_time - enqueue_ts).total_seconds() / 60

                # Only count if it DIDN'T trigger in current timeout (45min)
                if time_gap_minutes > CURRENT_TRIGGER_TIMEOUT_MINUTES:
                    results[timeout_minutes]['additional_trades'] += 1
                    results[timeout_minutes]['trades'].append({
                        'symbol': symbol,
                        'session_date': session_date,
                        'setup_type': decision['setup_type'],
                        'enqueue_ts': decision['ts'],
                        'trigger_ts': str(trigger_time),
                        'time_gap_minutes': time_gap_minutes,
                        'entry_price': entry_price
                    })

    return results


def main(sessions_root, cache_dir):
    print("=" * 80)
    print("TRIGGER TIMEOUT SPIKE TEST")
    print("=" * 80)
    print()

    # Step 1: Parse events
    print("Step 1: Parsing events.jsonl and analytics.jsonl...")
    decisions, triggers, exits = parse_events(sessions_root)

    print(f"  Total DECISION events: {len(decisions)}")
    print(f"  Total TRIGGER events: {len(triggers)}")
    print(f"  Total EXIT events: {len(exits)}")
    print()

    # Step 2: Analyze timing
    print("Step 2: Analyzing enqueue -> trigger timing...")
    triggered_trades, never_triggered = analyze_enqueue_to_trigger_timing(decisions, triggers)

    print(f"  Decisions that triggered: {len(triggered_trades)} ({len(triggered_trades)/len(decisions)*100:.1f}%)")
    print(f"  Decisions that NEVER triggered: {len(never_triggered)} ({len(never_triggered)/len(decisions)*100:.1f}%)")
    print()

    # Step 3: Timing distribution for triggered trades
    print("Step 3: Enqueue -> Trigger timing distribution")
    print("=" * 80)

    time_gaps = [t['time_gap_minutes'] for t in triggered_trades]
    if time_gaps:
        time_gaps_series = pd.Series(time_gaps)
        print(f"  Avg time gap: {time_gaps_series.mean():.1f} minutes")
        print(f"  Median time gap: {time_gaps_series.median():.1f} minutes")
        print(f"  Min time gap: {time_gaps_series.min():.1f} minutes")
        print(f"  Max time gap: {time_gaps_series.max():.1f} minutes")
        print()

        print("  Percentile Distribution:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"    {p}th percentile: {time_gaps_series.quantile(p/100):.1f} minutes")
        print()

        # Breakdown by time ranges
        print("  Time Gap Breakdown:")
        ranges = [
            (0, 5, "0-5 minutes (immediate)"),
            (5, 15, "5-15 minutes (quick)"),
            (15, 30, "15-30 minutes (moderate)"),
            (30, 45, "30-45 minutes (slow)"),
            (45, 60, "45-60 minutes (timeout extension needed)"),
            (60, 120, "60-120 minutes (way beyond current timeout)")
        ]

        for min_gap, max_gap, label in ranges:
            count = sum(1 for t in time_gaps if min_gap <= t < max_gap)
            pct = count / len(time_gaps) * 100 if time_gaps else 0
            print(f"    {label}: {count} ({pct:.1f}%)")

    print()

    # Step 4: SL hit timing analysis
    print("Step 4: Stop Loss Hit Timing Analysis")
    print("=" * 80)

    sl_hit_trades = analyze_sl_hit_timing(triggered_trades, exits)

    print(f"  Total SL hit trades: {len(sl_hit_trades)}")

    if sl_hit_trades:
        sl_time_gaps = [t['time_gap_minutes'] for t in sl_hit_trades]
        sl_time_gaps_series = pd.Series(sl_time_gaps)

        print(f"  Avg enqueue -> trigger gap (SL trades): {sl_time_gaps_series.mean():.1f} minutes")
        print(f"  Median: {sl_time_gaps_series.median():.1f} minutes")
        print()

        print("  Pattern Analysis:")
        quick_sl = sum(1 for t in sl_time_gaps if t < 15)
        slow_sl = sum(1 for t in sl_time_gaps if t >= 30)

        print(f"    Quick trigger (<15min) -> SL hit: {quick_sl} ({quick_sl/len(sl_time_gaps)*100:.1f}%)")
        print(f"    Slow trigger (>=30min) -> SL hit: {slow_sl} ({slow_sl/len(sl_time_gaps)*100:.1f}%)")

        print()
        print("  Hypothesis: If slow triggers have high SL rate, they're fading setups")

    print()

    # Step 5: Spike test - Would extending timeout help?
    print("Step 5: Spike Test - Timeout Extension Analysis")
    print("=" * 80)
    print(f"  Current timeout: {CURRENT_TRIGGER_TIMEOUT_MINUTES} minutes")
    print(f"  Testing alternative timeouts: {ALTERNATIVE_TIMEOUTS}")
    print()
    print("  Analyzing never-triggered decisions with 1m cache data...")

    spike_results = spike_test_timeout_extension(never_triggered, cache_dir, ALTERNATIVE_TIMEOUTS)

    print()
    print("  Results:")
    print()

    for timeout in ALTERNATIVE_TIMEOUTS:
        additional = spike_results[timeout]['additional_trades']
        total_triggered = len(triggered_trades) + additional
        new_trigger_rate = total_triggered / len(decisions) * 100 if decisions else 0

        improvement = additional / len(triggered_trades) * 100 if triggered_trades else 0

        print(f"    Timeout: {timeout} minutes")
        print(f"      Additional trades: {additional}")
        print(f"      New total triggered: {total_triggered} ({new_trigger_rate:.1f}% of all decisions)")
        print(f"      Improvement: +{improvement:.1f}%")

        if timeout == CURRENT_TRIGGER_TIMEOUT_MINUTES:
            print(f"      ^^^ CURRENT SETTING ^^^")

        print()

    # Step 6: Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Check 75th percentile
    if time_gaps:
        p75 = time_gaps_series.quantile(0.75)
        p90 = time_gaps_series.quantile(0.90)
        p95 = time_gaps_series.quantile(0.95)

        print(f"  75% of triggers occur within: {p75:.1f} minutes")
        print(f"  90% of triggers occur within: {p90:.1f} minutes")
        print(f"  95% of triggers occur within: {p95:.1f} minutes")
        print()

        if p90 < CURRENT_TRIGGER_TIMEOUT_MINUTES:
            print(f"  OK: CURRENT TIMEOUT ({CURRENT_TRIGGER_TIMEOUT_MINUTES}min) is ADEQUATE")
            print(f"     Covers 90%+ of triggers")
        else:
            recommended_timeout = int(p90 + 10)  # Add 10min buffer
            print(f"  WARNING: CONSIDER EXTENDING TIMEOUT to {recommended_timeout} minutes")
            print(f"     Would capture 90%+ of triggers")

        print()

    # Check spike test results
    timeout_60 = spike_results[60]['additional_trades']
    timeout_90 = spike_results[90]['additional_trades']

    if timeout_60 > len(triggered_trades) * 0.1:  # >10% improvement
        print(f"  OPPORTUNITY: Extending to 60min would add {timeout_60} trades (+{timeout_60/len(triggered_trades)*100:.1f}%)")

    if timeout_90 > timeout_60 * 1.5:  # 50% more trades from 90min
        print(f"  OPPORTUNITY: Extending to 90min would add {timeout_90} trades (+{timeout_90/len(triggered_trades)*100:.1f}%)")

    print()

    # SL hit pattern
    if sl_hit_trades and len(sl_hit_trades) > 10:
        quick_sl_rate = quick_sl / len(sl_hit_trades) * 100
        slow_sl_rate = slow_sl / len(sl_hit_trades) * 100

        if slow_sl_rate > quick_sl_rate * 1.5:
            print(f"  WARNING: Slow triggers (30+ min) have {slow_sl_rate:.1f}% SL rate")
            print(f"     vs quick triggers (<15min) at {quick_sl_rate:.1f}%")
            print(f"     -> Consider SHORTENING timeout to filter out fading setups")
        elif quick_sl_rate > slow_sl_rate * 1.5:
            print(f"  GOOD: Quick triggers have higher SL rate ({quick_sl_rate:.1f}%)")
            print(f"     Slow triggers are better quality")

    print()
    print("=" * 80)

    # Save detailed results
    output_file = Path(sessions_root).parent / "trigger_timeout_analysis.json"
    output_data = {
        'summary': {
            'total_decisions': len(decisions),
            'triggered': len(triggered_trades),
            'never_triggered': len(never_triggered),
            'trigger_rate': len(triggered_trades) / len(decisions) * 100 if decisions else 0,
            'current_timeout_minutes': CURRENT_TRIGGER_TIMEOUT_MINUTES
        },
        'timing_distribution': {
            'mean': float(time_gaps_series.mean()) if time_gaps else 0,
            'median': float(time_gaps_series.median()) if time_gaps else 0,
            'p75': float(time_gaps_series.quantile(0.75)) if time_gaps else 0,
            'p90': float(time_gaps_series.quantile(0.90)) if time_gaps else 0,
            'p95': float(time_gaps_series.quantile(0.95)) if time_gaps else 0
        },
        'sl_hit_analysis': {
            'total_sl_hits': len(sl_hit_trades),
            'quick_triggers_sl': quick_sl if sl_hit_trades else 0,
            'slow_triggers_sl': slow_sl if sl_hit_trades else 0
        },
        'spike_test_results': {
            str(timeout): {
                'additional_trades': results['additional_trades'],
                'sample_trades': results['trades'][:5]  # First 5 examples
            }
            for timeout, results in spike_results.items()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Detailed results saved to: {output_file}")
    print()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python spike_test_trigger_timeout.py <backtest_extracted_dir> <cache_dir>")
        print()
        print("Example:")
        print("  python tools/spike_test_trigger_timeout.py \\")
        print("    backtest_20251109-125133_extracted/20251109-125133_full/20251109-125133 \\")
        print("    cache")
        sys.exit(1)

    sessions_root = sys.argv[1]
    cache_dir = sys.argv[2]

    if not Path(sessions_root).exists():
        print(f"Error: Sessions directory not found: {sessions_root}")
        sys.exit(1)

    if not Path(cache_dir).exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        sys.exit(1)

    main(sessions_root, cache_dir)
