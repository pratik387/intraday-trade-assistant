#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Missed Opportunities Analysis (Phase 2B - Spike Test for Trade Count Expansion)

Analyzes symbols that were scanned but NOT traded to find missed profit opportunities.

Questions:
1. Which symbols had strong setups but were rejected? (Why were they filtered out?)
2. If we had taken those trades, would they have been profitable?
3. What filter is blocking the most profitable opportunities?
4. Can we safely relax filters to capture more winners?

Approach:
1. Load all DECISION events (both accepted and rejected)
2. Identify rejected trades (plan generated but no TRIGGER)
3. Simulate what would have happened if we took those trades
4. Calculate potential P&L and win rate
5. Identify which filter caused the rejection
6. Recommend filter adjustments

Data Sources:
- events.jsonl: DECISION events for rejected trades
- analytics.jsonl: Actual trades taken (for comparison)
- cache: 1m bar data for simulation
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

BACKTEST_DIR = Path("backtest_20251106-165927_extracted/20251106-165927_full/20251106-165927")
CACHE_DIR = Path("cache/ohlcv_archive")

def load_1m_data(symbol, date_str):
    """Load 1m bar data from cache for a specific date."""
    cache_symbol = symbol.replace("NSE:", "") + ".NS"
    cache_file = CACHE_DIR / cache_symbol / f"{cache_symbol}_1minutes.feather"

    if not cache_file.exists():
        return None

    try:
        df = pd.read_feather(cache_file)
        if 'date' not in df.columns:
            return None

        df['timestamp'] = pd.to_datetime(df['date'])
        target_date = pd.to_datetime(date_str).date()
        df['date_only'] = df['timestamp'].dt.date
        df = df[df['date_only'] == target_date].copy()

        if len(df) > 0:
            return df.sort_values('timestamp')
    except:
        return None

    return None

def simulate_hypothetical_trade(df, decision_time, plan, symbol):
    """
    Simulate what would have happened if we took this rejected trade.

    Returns:
    - would_hit_t1: bool
    - would_hit_t2: bool
    - would_hit_sl: bool
    - pnl_at_eod: float (if held to EOD)
    - best_pnl: float (best case)
    - time_to_t1: timestamp or None
    - time_to_sl: timestamp or None
    """
    try:
        decision_dt = pd.to_datetime(decision_time)
        if decision_dt.tz is None and df['timestamp'].dt.tz is not None:
            decision_dt = decision_dt.tz_localize('Asia/Kolkata')
    except:
        return None

    # Get trade parameters from plan
    entry_price = float(plan.get('price', plan.get('entry', {}).get('reference', 0)))
    if entry_price == 0:
        return None

    side = plan.get('bias', 'long')

    # Get stop and targets
    stop_dict = plan.get('stop', {})
    initial_sl = float(stop_dict.get('hard', 0))

    targets = plan.get('targets', [])
    t1_target = float(targets[0].get('level', 0)) if len(targets) > 0 else 0
    t2_target = float(targets[1].get('level', 0)) if len(targets) > 1 else 0

    if initial_sl == 0 or t1_target == 0:
        return None

    # Filter bars after decision
    df_after = df[df['timestamp'] >= decision_dt].copy()

    if len(df_after) == 0:
        return None

    result = {
        'would_hit_t1': False,
        'would_hit_t2': False,
        'would_hit_sl': False,
        'time_to_t1': None,
        'time_to_sl': None,
        'time_to_t2': None,
        'best_pnl': 0.0,
        'pnl_at_eod': 0.0,
    }

    best_price = entry_price

    for idx, bar in df_after.iterrows():
        low = bar['low']
        high = bar['high']
        close = bar['close']
        ts = bar['timestamp']

        if side in ['BUY', 'long']:
            # Check SL hit
            if not result['would_hit_sl'] and low <= initial_sl:
                result['would_hit_sl'] = True
                result['time_to_sl'] = ts

            # Check T1 hit
            if not result['would_hit_t1'] and high >= t1_target:
                result['would_hit_t1'] = True
                result['time_to_t1'] = ts

            # Check T2 hit
            if not result['would_hit_t2'] and t2_target > 0 and high >= t2_target:
                result['would_hit_t2'] = True
                result['time_to_t2'] = ts

            # Track best price
            if high > best_price:
                best_price = high

            # EOD P&L (last bar)
            if idx == df_after.index[-1]:
                result['pnl_at_eod'] = close - entry_price

        else:  # SELL / short
            # Check SL hit
            if not result['would_hit_sl'] and high >= initial_sl:
                result['would_hit_sl'] = True
                result['time_to_sl'] = ts

            # Check T1 hit
            if not result['would_hit_t1'] and low <= t1_target:
                result['would_hit_t1'] = True
                result['time_to_t1'] = ts

            # Check T2 hit
            if not result['would_hit_t2'] and t2_target > 0 and low <= t2_target:
                result['would_hit_t2'] = True
                result['time_to_t2'] = ts

            # Track best price
            if low < best_price:
                best_price = low

            # EOD P&L (last bar)
            if idx == df_after.index[-1]:
                result['pnl_at_eod'] = entry_price - close

    # Calculate best case P&L
    if side in ['BUY', 'long']:
        result['best_pnl'] = best_price - entry_price
    else:
        result['best_pnl'] = entry_price - best_price

    # Determine outcome
    if result['would_hit_sl']:
        # Would have hit SL
        if result['would_hit_t1'] and result['time_to_t1'] < result['time_to_sl']:
            result['outcome'] = 'T1_BEFORE_SL'
            result['estimated_pnl'] = (t1_target - entry_price) if side in ['BUY', 'long'] else (entry_price - t1_target)
        else:
            result['outcome'] = 'SL_HIT'
            result['estimated_pnl'] = (initial_sl - entry_price) if side in ['BUY', 'long'] else (entry_price - initial_sl)
    elif result['would_hit_t2']:
        result['outcome'] = 'T2_HIT'
        result['estimated_pnl'] = result['pnl_at_eod']  # Conservative: book at T2 then hold to EOD
    elif result['would_hit_t1']:
        result['outcome'] = 'T1_HIT'
        result['estimated_pnl'] = result['pnl_at_eod']  # Conservative: book at T1 then hold to EOD
    else:
        result['outcome'] = 'NO_TARGETS'
        result['estimated_pnl'] = result['pnl_at_eod']

    return result

def extract_rejection_reason(plan):
    """
    Try to infer why the trade was rejected based on plan contents.

    Common rejection reasons:
    - ADX too low
    - RSI not extreme enough (for fades)
    - Rank score below threshold
    - Structural R:R too low
    - Regime mismatch
    """
    indicators = plan.get('indicators', {})
    quality = plan.get('quality', {})

    adx = indicators.get('adx14', 0) or 0
    rsi = indicators.get('rsi14', 50) or 50
    structural_rr = quality.get('structural_rr', 0)
    regime = plan.get('regime', 'unknown')
    strategy = plan.get('strategy', 'unknown')

    reasons = []

    # ADX check
    if adx < 15:
        reasons.append(f"ADX_LOW({adx:.0f})")

    # RSI check (for fades)
    if 'fade' in strategy:
        if 'long' in strategy and rsi > 35:
            reasons.append(f"RSI_NOT_OVERSOLD({rsi:.0f})")
        elif 'short' in strategy and rsi < 65:
            reasons.append(f"RSI_NOT_OVERBOUGHT({rsi:.0f})")

    # Structural R:R check
    if structural_rr < 1.2:
        reasons.append(f"POOR_RR({structural_rr:.2f})")

    # Rank score (if available)
    rank = plan.get('rank_score', 0)
    if rank < 0.62:
        reasons.append(f"LOW_RANK({rank:.2f})")

    if not reasons:
        reasons.append("UNKNOWN")

    return " | ".join(reasons)

def main():
    print("="*120)
    print("MISSED OPPORTUNITIES ANALYSIS - Phase 2B (Spike Test for Trade Count Expansion)")
    print("="*120)
    print()

    # Collect all decisions (both accepted and rejected)
    all_decisions = []
    taken_trades = set()

    session_dirs = sorted(BACKTEST_DIR.glob('20*'))

    print("Loading decisions and trades...")

    for session_dir in session_dirs:
        date = session_dir.name
        events_file = session_dir / 'events.jsonl'
        analytics_file = session_dir / 'analytics.jsonl'

        # Load taken trades
        if analytics_file.exists():
            with open(analytics_file, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get('stage') == 'ENTRY':
                                taken_trades.add((date, data.get('symbol')))
                        except:
                            pass

        # Load all decisions
        if events_file.exists():
            with open(events_file, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            event = json.loads(line)
                            if event.get('type') == 'DECISION':
                                symbol = event.get('symbol')
                                plan = event.get('plan', {})
                                decision_time = event.get('ts')

                                all_decisions.append({
                                    'date': date,
                                    'symbol': symbol,
                                    'decision_time': decision_time,
                                    'plan': plan,
                                    'was_taken': (date, symbol) in taken_trades,
                                })
                        except:
                            pass

    # Filter to rejected decisions only
    rejected_decisions = [d for d in all_decisions if not d['was_taken']]

    print(f"Total decisions: {len(all_decisions)}")
    print(f"Trades taken: {len(taken_trades)}")
    print(f"Trades rejected: {len(rejected_decisions)}")
    print()

    if len(rejected_decisions) == 0:
        print("No rejected trades found. All decisions were executed.")
        return

    # Simulate rejected trades
    print("Simulating rejected trades (this will take a few minutes)...")
    print()

    results = []
    no_data_count = 0

    for i, decision in enumerate(rejected_decisions, 1):
        date = decision['date']
        symbol = decision['symbol']
        decision_time = decision['decision_time']
        plan = decision['plan']

        if i % 10 == 0:
            print(f"[{i}/{len(rejected_decisions)}] Processing {symbol} on {date}...")

        # Load 1m data
        df = load_1m_data(symbol, date)

        if df is None or len(df) == 0:
            no_data_count += 1
            continue

        # Simulate trade
        sim_result = simulate_hypothetical_trade(df, decision_time, plan, symbol)

        if sim_result is None:
            continue

        # Extract rejection reason
        rejection_reason = extract_rejection_reason(plan)

        strategy = plan.get('strategy', 'unknown')
        regime = plan.get('regime', 'unknown')

        results.append({
            'date': date,
            'symbol': symbol,
            'strategy': strategy,
            'regime': regime,
            'rejection_reason': rejection_reason,
            'would_hit_t1': sim_result['would_hit_t1'],
            'would_hit_t2': sim_result['would_hit_t2'],
            'would_hit_sl': sim_result['would_hit_sl'],
            'outcome': sim_result['outcome'],
            'estimated_pnl': sim_result['estimated_pnl'],
            'best_pnl': sim_result['best_pnl'],
            'pnl_at_eod': sim_result['pnl_at_eod'],
        })

    print()
    print("="*120)
    print("SIMULATION RESULTS")
    print("="*120)
    print()

    print(f"Total rejected decisions: {len(rejected_decisions)}")
    print(f"Successfully simulated: {len(results)}")
    print(f"No cache data: {no_data_count}")
    print()

    if len(results) == 0:
        print("No results to analyze.")
        return

    # Calculate statistics
    winners = [r for r in results if r['estimated_pnl'] > 0]
    losers = [r for r in results if r['estimated_pnl'] <= 0]

    total_pnl = sum(r['estimated_pnl'] for r in results)
    win_rate = len(winners) / len(results) * 100 if len(results) > 0 else 0

    print(f"HYPOTHETICAL PERFORMANCE IF ALL REJECTED TRADES WERE TAKEN:")
    print(f"  Total trades: {len(results)}")
    print(f"  Winners: {len(winners)} ({win_rate:.1f}%)")
    print(f"  Losers: {len(losers)} ({100-win_rate:.1f}%)")
    print(f"  Total P&L: Rs.{total_pnl:.2f}")
    print(f"  Avg P&L per trade: Rs.{total_pnl/len(results):.2f}")
    print()

    # Outcome breakdown
    print("OUTCOME BREAKDOWN:")
    outcomes = defaultdict(int)
    outcome_pnl = defaultdict(float)
    for r in results:
        outcomes[r['outcome']] += 1
        outcome_pnl[r['outcome']] += r['estimated_pnl']

    for outcome in sorted(outcomes.keys()):
        count = outcomes[outcome]
        pnl = outcome_pnl[outcome]
        pct = count / len(results) * 100
        print(f"  {outcome:<20} {count:>4} ({pct:>5.1f}%) → Rs.{pnl:>10.2f}")
    print()

    # Rejection reason analysis
    print("="*120)
    print("REJECTION REASON ANALYSIS")
    print("="*120)
    print()

    rejection_stats = defaultdict(lambda: {'count': 0, 'winners': 0, 'pnl': 0.0})

    for r in results:
        reason = r['rejection_reason']
        rejection_stats[reason]['count'] += 1
        if r['estimated_pnl'] > 0:
            rejection_stats[reason]['winners'] += 1
        rejection_stats[reason]['pnl'] += r['estimated_pnl']

    print(f"{'Rejection Reason':<40} {'Count':>8} {'Win Rate':>10} {'Total P&L':>12} {'Avg P&L':>10}")
    print("-"*120)

    for reason in sorted(rejection_stats.keys(), key=lambda x: rejection_stats[x]['pnl'], reverse=True):
        stats = rejection_stats[reason]
        count = stats['count']
        win_rate = stats['winners'] / count * 100 if count > 0 else 0
        pnl = stats['pnl']
        avg_pnl = pnl / count if count > 0 else 0

        print(f"{reason:<40} {count:>8} {win_rate:>9.1f}% Rs.{pnl:>10.2f} Rs.{avg_pnl:>8.2f}")

    print()

    # Strategy breakdown
    print("="*120)
    print("STRATEGY-SPECIFIC MISSED OPPORTUNITIES")
    print("="*120)
    print()

    strategy_stats = defaultdict(lambda: {'count': 0, 'winners': 0, 'pnl': 0.0})

    for r in results:
        strategy = r['strategy']
        strategy_stats[strategy]['count'] += 1
        if r['estimated_pnl'] > 0:
            strategy_stats[strategy]['winners'] += 1
        strategy_stats[strategy]['pnl'] += r['estimated_pnl']

    print(f"{'Strategy':<30} {'Count':>8} {'Win Rate':>10} {'Total P&L':>12} {'Avg P&L':>10}")
    print("-"*120)

    for strategy in sorted(strategy_stats.keys(), key=lambda x: strategy_stats[x]['pnl'], reverse=True):
        stats = strategy_stats[strategy]
        count = stats['count']
        win_rate = stats['winners'] / count * 100 if count > 0 else 0
        pnl = stats['pnl']
        avg_pnl = pnl / count if count > 0 else 0

        print(f"{strategy:<30} {count:>8} {win_rate:>9.1f}% Rs.{pnl:>10.2f} Rs.{avg_pnl:>8.2f}")

    print()

    # Show top missed opportunities
    print("="*120)
    print("TOP 20 MISSED WINNERS (Biggest Lost Profits)")
    print("="*120)
    print()

    top_winners = sorted([r for r in results if r['estimated_pnl'] > 0],
                         key=lambda x: x['estimated_pnl'], reverse=True)[:20]

    print(f"{'Date':<12} {'Symbol':<20} {'Strategy':<25} {'Rejection Reason':<35} {'Outcome':<20} {'P&L':>10}")
    print("-"*120)

    for r in top_winners:
        print(f"{r['date']:<12} {r['symbol']:<20} {r['strategy']:<25} {r['rejection_reason']:<35} {r['outcome']:<20} Rs.{r['estimated_pnl']:>8.2f}")

    print()

    # Recommendations
    print("="*120)
    print("RECOMMENDATIONS")
    print("="*120)
    print()

    # Find the most profitable rejection reasons (filters to relax)
    profitable_reasons = [(reason, stats) for reason, stats in rejection_stats.items()
                          if stats['pnl'] > 0 and stats['winners'] / stats['count'] >= 0.50]

    profitable_reasons = sorted(profitable_reasons, key=lambda x: x[1]['pnl'], reverse=True)

    if profitable_reasons:
        print("FILTERS TO CONSIDER RELAXING (High win rate + positive P&L):")
        print()
        for reason, stats in profitable_reasons[:5]:
            count = stats['count']
            win_rate = stats['winners'] / count * 100
            pnl = stats['pnl']

            print(f"  {reason}")
            print(f"    → {count} missed trades, {win_rate:.1f}% win rate, Rs.{pnl:.2f} potential profit")
            print()
    else:
        print("No clear filter relaxation opportunities found.")
        print("Current filters are appropriately conservative.")

    # Save detailed results
    output_file = Path("missed_opportunities_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_rejected': len(results),
                'winners': len(winners),
                'win_rate_pct': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / len(results) if len(results) > 0 else 0,
            },
            'by_rejection_reason': dict(rejection_stats),
            'by_strategy': dict(strategy_stats),
            'trades': results
        }, f, indent=2, default=str)

    print()
    print(f"Detailed results saved to: {output_file}")
    print()
    print("="*120)

if __name__ == "__main__":
    main()
