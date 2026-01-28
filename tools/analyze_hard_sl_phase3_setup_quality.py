#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 & 4: Setup Quality and Regime Analysis for hard_sl trades

Analyzes:
1. Setup quality vs professional trading standards
2. Regime detection accuracy (does "chop" actually mean choppy?)
3. Strategy appropriateness (breakout in trend vs fade in chop)
4. Common patterns among failed trades
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

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

    except Exception as e:
        return None

    return None

def load_trade_from_events(session_dir, symbol):
    """Load trade plan and timestamps from events.jsonl"""
    events_file = session_dir / 'events.jsonl'

    if not events_file.exists():
        return None

    decision_event = None
    entry_time = None
    exit_time = None

    with open(events_file, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    event = json.loads(line)
                    if event.get('symbol') != symbol:
                        continue

                    event_type = event.get('type')

                    if event_type == 'DECISION':
                        decision_event = event
                    elif event_type == 'TRIGGER':
                        entry_time = event.get('ts')
                    elif event_type == 'EXIT':
                        exit_event = event.get('exit', {})
                        if exit_event.get('reason') == 'hard_sl':
                            exit_time = event.get('ts')
                except:
                    continue

    if not decision_event or not entry_time:
        return None

    return {
        'decision': decision_event,
        'plan': decision_event.get('plan', {}),
        'entry_time': entry_time,
        'exit_time': exit_time
    }

def calculate_actual_regime(df, decision_time):
    """
    Calculate the ACTUAL market regime using professional standards.

    Uses 1-hour window before decision to classify:
    - TRENDING: Clear directional move with follow-through
    - CHOPPY: Range-bound, no clear direction
    - BREAKOUT: Building for expansion
    """

    try:
        decision_dt = pd.to_datetime(decision_time)
        if decision_dt.tz is None and df['timestamp'].dt.tz is not None:
            decision_dt = decision_dt.tz_localize('Asia/Kolkata')
    except:
        return None

    # Look at 1 hour before decision (opening range)
    start_time = decision_dt - timedelta(hours=1)
    window = df[(df['timestamp'] >= start_time) & (df['timestamp'] < decision_dt)].copy()

    if len(window) < 10:
        return None

    # Calculate metrics
    high = window['high'].max()
    low = window['low'].min()
    open_price = window.iloc[0]['open']
    close_price = window.iloc[-1]['close']

    range_size = high - low
    body_size = abs(close_price - open_price)

    # Calculate directional strength
    upside = high - open_price
    downside = open_price - low

    # Calculate chop via bar overlaps
    overlaps = 0
    for i in range(1, len(window)):
        prev_high = window.iloc[i-1]['high']
        prev_low = window.iloc[i-1]['low']
        curr_high = window.iloc[i]['high']
        curr_low = window.iloc[i]['low']

        # Bars overlap significantly
        if (curr_low < prev_high and curr_high > prev_low):
            overlaps += 1

    overlap_rate = overlaps / (len(window) - 1) if len(window) > 1 else 0

    # Professional regime classification
    actual_regime = 'UNKNOWN'
    regime_confidence = 0

    # TRENDING: Body > 60% of range AND low overlap
    if body_size / range_size >= 0.6 and overlap_rate < 0.5:
        actual_regime = 'TRENDING'
        regime_confidence = min(100, int((body_size / range_size) * 100))

    # CHOPPY: High overlap (>70%) OR small body (<30% of range)
    elif overlap_rate >= 0.7 or (body_size / range_size < 0.3):
        actual_regime = 'CHOPPY'
        regime_confidence = min(100, int(overlap_rate * 100))

    # BREAKOUT SETUP: Tightening range, building energy
    elif range_size < (window['close'].std() * 2):
        actual_regime = 'COILING'
        regime_confidence = 70

    else:
        actual_regime = 'MIXED'
        regime_confidence = 50

    return {
        'actual_regime': actual_regime,
        'confidence': regime_confidence,
        'overlap_rate': overlap_rate * 100,
        'body_pct': (body_size / range_size * 100) if range_size > 0 else 0,
        'range_size': range_size,
        'direction': 'UP' if close_price > open_price else 'DOWN',
    }

def analyze_setup_quality(trade_data, df, actual_regime_data):
    """
    Grade setup quality based on professional trading standards.

    Returns grade: A, B, C, D, F with reasoning
    """

    plan = trade_data['plan']
    decision = trade_data['decision']

    # Extract key info
    strategy = plan.get('strategy', 'unknown')
    detected_regime = plan.get('regime', 'unknown')
    bias = plan.get('bias', 'unknown')

    # Get decision reasoning
    decision_data = decision.get('decision', {})
    reasons = decision_data.get('reasons', '')

    # Quality metrics
    quality = plan.get('quality', {})
    structural_rr = quality.get('structural_rr', 0)
    acceptance = quality.get('acceptance_status', 'unknown')

    # Indicators at decision
    indicators = plan.get('indicators', {})
    adx = indicators.get('adx14', 0) or 0
    rsi = indicators.get('rsi14', 50) or 50

    issues = []
    strengths = []
    grade_score = 100

    # Check 1: Regime alignment
    if actual_regime_data:
        actual = actual_regime_data['actual_regime']

        # CRITICAL: Strategy must match regime
        if actual == 'TRENDING':
            if 'breakout' in strategy:
                strengths.append(f"Breakout in trend ({actual_regime_data['direction']}) ✓")
            elif 'fade' in strategy:
                issues.append("FATAL: Fading a trend (counter-trend) ✗")
                grade_score -= 40

        elif actual == 'CHOPPY':
            if 'fade' in strategy:
                strengths.append("Fade in chop (mean reversion) ✓")
            elif 'breakout' in strategy:
                issues.append("WARNING: Breakout in chop (likely false) ✗")
                grade_score -= 25

        # Detected vs actual mismatch
        if detected_regime != actual.lower():
            if actual == 'CHOPPY' and detected_regime == 'trend_up':
                issues.append(f"Regime WRONG: Detected '{detected_regime}' but actually {actual}")
                grade_score -= 20
            elif actual == 'TRENDING' and 'chop' in detected_regime:
                issues.append(f"Regime WRONG: Detected '{detected_regime}' but actually {actual}")
                grade_score -= 20

    # Check 2: Risk/Reward structure
    if structural_rr < 0.3:
        issues.append(f"Poor R:R structure ({structural_rr:.2f}) - too tight")
        grade_score -= 15
    elif structural_rr >= 0.5:
        strengths.append(f"Good R:R structure ({structural_rr:.2f}) ✓")

    # Check 3: ADX (trend strength)
    if adx < 15:
        issues.append(f"Weak trend strength (ADX={adx:.0f}) - no conviction")
        grade_score -= 10
    elif adx >= 25:
        strengths.append(f"Strong trend (ADX={adx:.0f}) ✓")

    # Check 4: RSI extremes (for fades)
    if 'fade' in strategy:
        if bias == 'long' and rsi > 35:
            issues.append(f"Fade long but RSI={rsi:.0f} (not oversold)")
            grade_score -= 15
        elif bias == 'short' and rsi < 65:
            issues.append(f"Fade short but RSI={rsi:.0f} (not overbought)")
            grade_score -= 15
        else:
            strengths.append(f"RSI extreme ({rsi:.0f}) supports fade ✓")

    # Check 5: Acceptance quality
    if acceptance in ['poor', 'fail']:
        issues.append("Poor acceptance quality")
        grade_score -= 15
    elif acceptance == 'excellent':
        strengths.append("Excellent acceptance ✓")

    # Check 6: Strategy clarity
    if 'breakout' in strategy and 'fade' in strategy:
        issues.append("Confused strategy (both breakout AND fade)")
        grade_score -= 20

    # Grade assignment
    if grade_score >= 85:
        grade = 'A'
        verdict = "INSTITUTIONAL QUALITY"
    elif grade_score >= 70:
        grade = 'B'
        verdict = "ACCEPTABLE SETUP"
    elif grade_score >= 55:
        grade = 'C'
        verdict = "MARGINAL SETUP"
    elif grade_score >= 40:
        grade = 'D'
        verdict = "POOR SETUP"
    else:
        grade = 'F'
        verdict = "RETAIL TRAP"

    return {
        'grade': grade,
        'score': grade_score,
        'verdict': verdict,
        'issues': issues,
        'strengths': strengths,
        'strategy': strategy,
        'detected_regime': detected_regime,
        'actual_regime': actual_regime_data.get('actual_regime') if actual_regime_data else 'UNKNOWN',
        'adx': adx,
        'rsi': rsi,
        'structural_rr': structural_rr,
    }

def main():
    print("="*120)
    print("PHASE 3 & 4: SETUP QUALITY AND REGIME ANALYSIS")
    print("="*120)
    print()

    # Collect all hard_sl trades
    hard_sl_trades = []

    session_dirs = sorted(BACKTEST_DIR.glob('20*'))

    for session_dir in session_dirs:
        analytics_file = session_dir / 'analytics.jsonl'

        if not analytics_file.exists():
            continue

        with open(analytics_file, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get('stage') == 'EXIT' and data.get('reason') == 'hard_sl':
                            hard_sl_trades.append({
                                'date': session_dir.name,
                                'symbol': data.get('symbol'),
                                'session_dir': session_dir,
                            })
                    except:
                        pass

    print(f"Found {len(hard_sl_trades)} hard_sl trades to analyze")
    print()

    # Analyze each trade
    results = []

    for i, trade in enumerate(hard_sl_trades, 1):
        date = trade['date']
        symbol = trade['symbol']
        session_dir = trade['session_dir']

        print(f"[{i}/{len(hard_sl_trades)}] Analyzing {symbol} on {date}...", end=" ")

        # Load 1m data
        df = load_1m_data(symbol, date)

        if df is None or len(df) == 0:
            print("NO CACHE DATA")
            continue

        # Load trade details
        trade_data = load_trade_from_events(session_dir, symbol)

        if not trade_data:
            print("NO EVENTS DATA")
            continue

        decision_time = trade_data['decision'].get('ts')

        # Calculate actual regime
        actual_regime = calculate_actual_regime(df, decision_time)

        # Analyze setup quality
        quality_analysis = analyze_setup_quality(trade_data, df, actual_regime)

        print(f"Grade {quality_analysis['grade']} - {quality_analysis['verdict']}")

        results.append({
            'date': date,
            'symbol': symbol,
            **quality_analysis,
            'regime_overlap_rate': actual_regime.get('overlap_rate') if actual_regime else None,
            'regime_confidence': actual_regime.get('confidence') if actual_regime else None,
        })

    print()
    print("="*120)
    print("RESULTS SUMMARY")
    print("="*120)
    print()

    if len(results) == 0:
        print("No results to analyze.")
        return

    # Grade distribution
    grades = {}
    for grade in ['A', 'B', 'C', 'D', 'F']:
        count = len([r for r in results if r['grade'] == grade])
        grades[grade] = count

    print("SETUP QUALITY DISTRIBUTION:")
    print(f"  Grade A (Institutional): {grades['A']} ({grades['A']/len(results)*100:.1f}%)")
    print(f"  Grade B (Acceptable): {grades['B']} ({grades['B']/len(results)*100:.1f}%)")
    print(f"  Grade C (Marginal): {grades['C']} ({grades['C']/len(results)*100:.1f}%)")
    print(f"  Grade D (Poor): {grades['D']} ({grades['D']/len(results)*100:.1f}%)")
    print(f"  Grade F (Retail Trap): {grades['F']} ({grades['F']/len(results)*100:.1f}%)")
    print()

    # Regime accuracy
    regime_mismatch = [r for r in results if r['detected_regime'] != r['actual_regime'].lower()
                       and r['actual_regime'] != 'UNKNOWN']

    print(f"REGIME DETECTION ACCURACY:")
    print(f"  Mismatched regimes: {len(regime_mismatch)} ({len(regime_mismatch)/len(results)*100:.1f}%)")
    print()

    # Strategy appropriateness
    strategy_errors = []
    for r in results:
        for issue in r['issues']:
            if 'FATAL' in issue or 'Regime WRONG' in issue:
                strategy_errors.append(r)
                break

    print(f"STRATEGY ERRORS:")
    print(f"  Fatal strategy mistakes: {len(strategy_errors)} ({len(strategy_errors)/len(results)*100:.1f}%)")
    print()

    # Show F-grade setups
    f_grades = [r for r in results if r['grade'] == 'F']
    if len(f_grades) > 0:
        print("="*120)
        print("GRADE F SETUPS (RETAIL TRAPS)")
        print("="*120)
        print()
        print(f"{'Date':<12} {'Symbol':<20} {'Strategy':<20} {'Detected':<15} {'Actual':<15} {'Issues':<50}")
        print("-"*120)

        for r in f_grades[:10]:
            issues_str = '; '.join(r['issues'][:2]) if r['issues'] else 'Multiple'
            print(f"{r['date']:<12} {r['symbol']:<20} {r['strategy']:<20} {r['detected_regime']:<15} "
                  f"{r['actual_regime']:<15} {issues_str[:48]}")

        print()

    # Show D-grade setups
    d_grades = [r for r in results if r['grade'] == 'D']
    if len(d_grades) > 0:
        print("="*120)
        print("GRADE D SETUPS (POOR QUALITY)")
        print("="*120)
        print()
        print(f"{'Date':<12} {'Symbol':<20} {'Strategy':<20} {'Main Issue':<60}")
        print("-"*120)

        for r in d_grades[:10]:
            main_issue = r['issues'][0] if r['issues'] else 'Unknown'
            print(f"{r['date']:<12} {r['symbol']:<20} {r['strategy']:<20} {main_issue[:58]}")

        print()

    # Common issues analysis
    print("="*120)
    print("MOST COMMON ISSUES")
    print("="*120)
    print()

    issue_counts = {}
    for r in results:
        for issue in r['issues']:
            # Categorize issue
            if 'Regime WRONG' in issue:
                key = 'Regime misdetection'
            elif 'FATAL: Fading a trend' in issue:
                key = 'Counter-trend fade (fatal)'
            elif 'Breakout in chop' in issue:
                key = 'False breakout in chop'
            elif 'Poor R:R structure' in issue:
                key = 'Poor risk/reward'
            elif 'Weak trend strength' in issue:
                key = 'Low ADX / No conviction'
            elif 'RSI' in issue:
                key = 'RSI not extreme (fade timing)'
            else:
                key = 'Other'

            issue_counts[key] = issue_counts.get(key, 0) + 1

    for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(results) * 100
        print(f"  {issue}: {count} trades ({pct:.1f}%)")

    print()

    # Final verdict
    print("="*120)
    print("VERDICT")
    print("="*120)
    print()

    failing_setups = grades['D'] + grades['F']
    failing_pct = failing_setups / len(results) * 100

    if failing_pct >= 60:
        verdict = "SETUP QUALITY IS THE PRIMARY ISSUE"
        action = "TIGHTEN FILTERS - 60%+ of trades are D/F grade"
    elif failing_pct >= 40:
        verdict = "SETUP QUALITY IS A MAJOR ISSUE"
        action = "IMPROVE FILTERS - 40%+ of trades are poor quality"
    elif len(regime_mismatch) / len(results) >= 0.5:
        verdict = "REGIME DETECTION IS BROKEN"
        action = "FIX REGIME CLASSIFIER - 50%+ misclassified"
    else:
        verdict = "SETUP QUALITY ACCEPTABLE BUT CAN IMPROVE"
        action = "REFINE FILTERS for marginal improvements"

    print(f"{verdict}")
    print(f"→ {failing_pct:.1f}% of trades are D/F grade (poor/trap setups)")
    print(f"→ {len(regime_mismatch)/len(results)*100:.1f}% have regime detection errors")
    print(f"→ {len(strategy_errors)/len(results)*100:.1f}% have fatal strategy errors")
    print()
    print(f"RECOMMENDATION: {action}")
    print()

    # P&L impact
    hard_sl_avg_loss = 493.75
    fixable_count = failing_setups

    avoided_losses = fixable_count * hard_sl_avg_loss

    print(f"P&L IMPACT ESTIMATE (if setup quality fixed):")
    print(f"  Fixable trades (D/F grade): {fixable_count}")
    print(f"  Avoided losses: Rs.{avoided_losses:.2f}")
    print(f"  These trades should NEVER have been taken")
    print()

    # Save results
    output_file = Path("phase3_setup_quality_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_analyzed': len(results),
                'grade_distribution': grades,
                'failing_setups': failing_setups,
                'failing_pct': failing_pct,
                'regime_mismatches': len(regime_mismatch),
                'fatal_strategy_errors': len(strategy_errors),
                'estimated_avoidable_losses': avoided_losses,
            },
            'trades': results
        }, f, indent=2, default=str)

    print(f"Detailed results saved to: {output_file}")
    print()
    print("="*120)

if __name__ == "__main__":
    main()
