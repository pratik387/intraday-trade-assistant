#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monthly Performance Analysis Across Market Regimes

Analyzes backtest performance by month to identify:
1. Which months/market regimes perform best
2. Setup type performance by month
3. Detector effectiveness by market regime
4. Win rate and PnL trends across different market conditions
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251114-125524_extracted\20251114-125524_full\20251114-125524")

def parse_session_date(session_name):
    """Parse session directory name to get date."""
    try:
        return datetime.strptime(session_name, "%Y-%m-%d").date()
    except:
        return None

def get_year_month(date_obj):
    """Get YYYY-MM string from date."""
    return f"{date_obj.year}-{date_obj.month:02d}"

def main():
    print("=" * 80)
    print("MONTHLY PERFORMANCE ANALYSIS - MARKET REGIME COMPARISON")
    print("=" * 80)

    # Load all analytics (trades)
    all_trades = []
    session_dates = []

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        session_date = parse_session_date(session_dir.name)
        if session_date is None:
            continue

        session_dates.append(session_date)

        analytics_file = session_dir / "analytics.jsonl"
        if analytics_file.exists():
            with open(analytics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        trade = json.loads(line)
                        trade['session_date'] = session_date
                        trade['year_month'] = get_year_month(session_date)
                        all_trades.append(trade)

    print(f"\nLoaded {len(all_trades)} trades from {len(session_dates)} sessions")

    # Identify unique months
    session_dates.sort()
    unique_months = sorted(set([get_year_month(d) for d in session_dates]))

    print(f"Date range: {session_dates[0]} to {session_dates[-1]}")
    print(f"Unique months tested: {len(unique_months)}")
    for ym in unique_months:
        sessions_in_month = [d for d in session_dates if get_year_month(d) == ym]
        print(f"  - {ym}: {len(sessions_in_month)} sessions")

    # Calculate monthly performance
    print(f"\n" + "=" * 80)
    print("MONTHLY PERFORMANCE SUMMARY")
    print("=" * 80)

    monthly_stats = defaultdict(lambda: {
        'trades': 0,
        'winners': 0,
        'losers': 0,
        'total_pnl': 0.0,
        'sessions': set(),
        'gross_profit': 0.0,
        'gross_loss': 0.0,
        'max_win': 0.0,
        'max_loss': 0.0,
    })

    for trade in all_trades:
        ym = trade['year_month']
        session_date = trade['session_date']
        pnl = trade.get('pnl', 0.0)

        monthly_stats[ym]['trades'] += 1
        monthly_stats[ym]['total_pnl'] += pnl
        monthly_stats[ym]['sessions'].add(session_date)

        if pnl > 0:
            monthly_stats[ym]['winners'] += 1
            monthly_stats[ym]['gross_profit'] += pnl
            monthly_stats[ym]['max_win'] = max(monthly_stats[ym]['max_win'], pnl)
        else:
            monthly_stats[ym]['losers'] += 1
            monthly_stats[ym]['gross_loss'] += abs(pnl)
            monthly_stats[ym]['max_loss'] = max(monthly_stats[ym]['max_loss'], abs(pnl))

    print(f"\n{'Month':<10} {'Sessions':>8} {'Trades':>8} {'Win%':>8} {'PnL':>12} {'Avg/Trade':>12} {'Profit Factor':>12}")
    print("-" * 80)

    total_trades = 0
    total_winners = 0
    total_pnl = 0.0
    total_gross_profit = 0.0
    total_gross_loss = 0.0

    for ym in sorted(monthly_stats.keys()):
        stats = monthly_stats[ym]
        sessions = len(stats['sessions'])
        trades = stats['trades']
        winners = stats['winners']
        win_rate = (winners / trades * 100) if trades > 0 else 0
        pnl = stats['total_pnl']
        avg_pnl = pnl / trades if trades > 0 else 0
        profit_factor = (stats['gross_profit'] / stats['gross_loss']) if stats['gross_loss'] > 0 else 0

        print(f"{ym:<10} {sessions:>8} {trades:>8} {win_rate:>7.1f}% ₹{pnl:>10,.0f} ₹{avg_pnl:>10,.0f} {profit_factor:>11.2f}")

        total_trades += trades
        total_winners += winners
        total_pnl += pnl
        total_gross_profit += stats['gross_profit']
        total_gross_loss += stats['gross_loss']

    print("-" * 80)
    total_win_rate = (total_winners / total_trades * 100) if total_trades > 0 else 0
    total_avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    total_profit_factor = (total_gross_profit / total_gross_loss) if total_gross_loss > 0 else 0

    print(f"{'OVERALL':<10} {len(session_dates):>8} {total_trades:>8} {total_win_rate:>7.1f}% ₹{total_pnl:>10,.0f} ₹{total_avg_pnl:>10,.0f} {total_profit_factor:>11.2f}")

    # Analyze by setup type per month
    print(f"\n" + "=" * 80)
    print("SETUP TYPE PERFORMANCE BY MONTH")
    print("=" * 80)

    setup_by_month = defaultdict(lambda: defaultdict(lambda: {
        'trades': 0,
        'winners': 0,
        'pnl': 0.0
    }))

    for trade in all_trades:
        ym = trade['year_month']
        setup = trade.get('setup_type', 'unknown')
        pnl = trade.get('pnl', 0.0)

        setup_by_month[ym][setup]['trades'] += 1
        setup_by_month[ym][setup]['pnl'] += pnl
        if pnl > 0:
            setup_by_month[ym][setup]['winners'] += 1

    for ym in sorted(setup_by_month.keys()):
        print(f"\n{ym}:")
        print(f"{'  Setup Type':<35} {'Trades':>8} {'Win%':>8} {'PnL':>12}")
        print("  " + "-" * 76)

        setups = setup_by_month[ym]
        for setup, stats in sorted(setups.items(), key=lambda x: x[1]['pnl'], reverse=True):
            trades = stats['trades']
            winners = stats['winners']
            win_rate = (winners / trades * 100) if trades > 0 else 0
            pnl = stats['pnl']
            print(f"  {setup:<35} {trades:>8} {win_rate:>7.1f}% ₹{pnl:>10,.0f}")

    # Detector performance by month
    print(f"\n" + "=" * 80)
    print("DETECTOR PERFORMANCE BY MONTH (from decisions)")
    print("=" * 80)

    # Load all decisions to analyze detectors
    detector_by_month = defaultdict(lambda: defaultdict(lambda: {
        'decisions': 0,
        'trades': 0,
        'winners': 0,
        'pnl': 0.0
    }))

    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        session_date = parse_session_date(session_dir.name)
        if session_date is None:
            continue

        ym = get_year_month(session_date)

        events_file = session_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get('type') == 'DECISION':
                            reasons = event.get('decision', {}).get('reasons', '')
                            if 'structure:detector:' in reasons:
                                for part in reasons.split(';'):
                                    if 'structure:detector:' in part:
                                        detector = part.split('structure:detector:')[1].strip()
                                        detector_by_month[ym][detector]['decisions'] += 1

    # Match trades to detectors
    for trade in all_trades:
        ym = trade['year_month']
        # Try to infer detector from setup type
        setup = trade.get('setup_type', '')
        if 'breakout' in setup:
            detector = 'level_breakout_long'
        elif 'failure_fade' in setup:
            detector = 'failure_fade_long'
        elif 'trend' in setup:
            detector = 'trend_pullback_long'
        else:
            detector = 'unknown'

        pnl = trade.get('pnl', 0.0)
        detector_by_month[ym][detector]['trades'] += 1
        detector_by_month[ym][detector]['pnl'] += pnl
        if pnl > 0:
            detector_by_month[ym][detector]['winners'] += 1

    for ym in sorted(detector_by_month.keys()):
        print(f"\n{ym}:")
        print(f"{'  Detector':<35} {'Decisions':>10} {'Trades':>8} {'Win%':>8} {'PnL':>12}")
        print("  " + "-" * 76)

        detectors = detector_by_month[ym]
        for detector, stats in sorted(detectors.items(), key=lambda x: x[1]['decisions'], reverse=True):
            decisions = stats['decisions']
            trades = stats['trades']
            winners = stats['winners']
            win_rate = (winners / trades * 100) if trades > 0 else 0
            pnl = stats['pnl']
            print(f"  {detector:<35} {decisions:>10} {trades:>8} {win_rate:>7.1f}% ₹{pnl:>10,.0f}")

    # Market regime analysis
    print(f"\n" + "=" * 80)
    print("MARKET REGIME INSIGHTS")
    print("=" * 80)

    print(f"\nBest performing month:")
    best_month = max(monthly_stats.items(), key=lambda x: x[1]['total_pnl'])
    best_ym, best_stats = best_month
    print(f"  {best_ym}: ₹{best_stats['total_pnl']:,.0f} ({best_stats['trades']} trades, {best_stats['winners']/best_stats['trades']*100:.1f}% WR)")

    print(f"\nWorst performing month:")
    worst_month = min(monthly_stats.items(), key=lambda x: x[1]['total_pnl'])
    worst_ym, worst_stats = worst_month
    print(f"  {worst_ym}: ₹{worst_stats['total_pnl']:,.0f} ({worst_stats['trades']} trades, {worst_stats['winners']/worst_stats['trades']*100:.1f}% WR)")

    print(f"\nMost active month:")
    most_active = max(monthly_stats.items(), key=lambda x: x[1]['trades'])
    active_ym, active_stats = most_active
    print(f"  {active_ym}: {active_stats['trades']} trades ({len(active_stats['sessions'])} sessions)")

    print(f"\nHighest win rate month:")
    best_wr_month = max(monthly_stats.items(), key=lambda x: x[1]['winners']/x[1]['trades'] if x[1]['trades'] > 0 else 0)
    wr_ym, wr_stats = best_wr_month
    print(f"  {wr_ym}: {wr_stats['winners']/wr_stats['trades']*100:.1f}% ({wr_stats['winners']}/{wr_stats['trades']} trades)")

    # Consistency analysis
    print(f"\n" + "=" * 80)
    print("CONSISTENCY ANALYSIS")
    print("=" * 80)

    profitable_months = sum(1 for stats in monthly_stats.values() if stats['total_pnl'] > 0)
    print(f"\nProfitable months: {profitable_months}/{len(monthly_stats)} ({profitable_months/len(monthly_stats)*100:.1f}%)")

    # Volatility
    monthly_pnls = [stats['total_pnl'] for stats in monthly_stats.values()]
    import statistics
    if len(monthly_pnls) > 1:
        pnl_std = statistics.stdev(monthly_pnls)
        pnl_mean = statistics.mean(monthly_pnls)
        print(f"Monthly PnL std dev: ₹{pnl_std:,.0f}")
        print(f"Monthly PnL mean: ₹{pnl_mean:,.0f}")
        print(f"Consistency ratio (mean/std): {abs(pnl_mean/pnl_std) if pnl_std > 0 else 0:.2f}")

    # Setup diversity by month
    print(f"\n" + "=" * 80)
    print("SETUP DIVERSITY BY MONTH")
    print("=" * 80)

    for ym in sorted(setup_by_month.keys()):
        unique_setups = len(setup_by_month[ym])
        print(f"\n{ym}: {unique_setups} unique setup types")
        setup_counts = [(setup, stats['trades']) for setup, stats in setup_by_month[ym].items()]
        setup_counts.sort(key=lambda x: x[1], reverse=True)
        for setup, count in setup_counts:
            pct = count / sum(s['trades'] for s in setup_by_month[ym].values()) * 100
            print(f"  - {setup}: {count} trades ({pct:.1f}%)")

if __name__ == '__main__':
    main()
