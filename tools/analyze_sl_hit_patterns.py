#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep SL Hit Pattern Analysis
Forensic analysis of hard_sl trades to identify common indicators/filters/parameters
that predict stop loss hits vs winning trades.
"""

import json
import sys
import io
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def load_trade_data(logs_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Load all trades and separate into SL hits vs winners.
    Returns: (sl_hits, winners)
    """

    all_trades = []

    for session_dir in sorted(logs_dir.glob("2*")):
        if not session_dir.is_dir():
            continue

        events_file = session_dir / "events.jsonl"
        if not events_file.exists():
            continue

        session_date = session_dir.name

        # Track trades by trade_id
        trades = {}

        with open(events_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line)
                    event_type = event.get('type')
                    trade_id = event.get('trade_id')

                    if event_type == 'DECISION':
                        decision = event.get('decision', {})
                        plan = event.get('plan', {})
                        features = event.get('features', {})
                        indicators = plan.get('indicators', {})

                        trades[trade_id] = {
                            'session': session_date,
                            'symbol': event.get('symbol', '').replace('NSE:', ''),
                            'trade_id': trade_id,
                            'timestamp': event.get('ts'),
                            'setup_type': decision.get('setup_type', ''),
                            'side': plan.get('bias', ''),
                            'regime': decision.get('regime', ''),
                            'rank_score': features.get('rank_score', 0),
                            'confidence': 0,  # Not directly available

                            # Plan data
                            'entry': plan.get('entry', {}).get('reference', 0),
                            'hard_sl': plan.get('stop', {}).get('hard', 0),
                            't1_target': plan.get('targets', [{}])[0].get('level', 0) if plan.get('targets') else 0,
                            't2_target': plan.get('targets', [{}])[1].get('level', 0) if len(plan.get('targets', [])) > 1 else 0,

                            # Quality metrics
                            'structural_rr': plan.get('quality', {}).get('structural_rr', 0),
                            'win_probability': 0,  # Not available
                            'overall_score': 0,  # Not available
                            'acceptance_status': plan.get('quality', {}).get('acceptance_status', ''),

                            # Indicators (from plan.indicators)
                            'vwap': indicators.get('vwap'),
                            'ema20': indicators.get('ema20'),
                            'ema50': indicators.get('ema50'),
                            'rsi': indicators.get('rsi14'),
                            'adx': indicators.get('adx14'),
                            'atr': indicators.get('atr'),
                            'volume_ratio': indicators.get('vol_ratio'),
                            'distance_from_vwap': None,  # Calculate if needed
                            'macd_hist': indicators.get('macd_hist'),

                            # Outcome (will be filled by EXIT event)
                            'exit_reason': None,
                            'pnl': 0,
                            'exit_ts': None
                        }

                    elif event_type == 'TRIGGER':
                        # Track actual entry price if different from plan
                        if trade_id in trades:
                            trades[trade_id]['triggered'] = True

                    elif event_type == 'EXIT':
                        if trade_id in trades:
                            exit_info = event.get('exit', {})
                            trades[trade_id]['exit_reason'] = exit_info.get('reason', 'unknown')
                            trades[trade_id]['pnl'] = exit_info.get('pnl', 0)
                            trades[trade_id]['exit_ts'] = event.get('ts')

                except Exception as e:
                    continue

        # Add completed trades to all_trades
        for trade_id, trade in trades.items():
            if trade.get('exit_reason'):
                all_trades.append(trade)

    # Separate into SL hits vs winners
    sl_hits = [t for t in all_trades if t['exit_reason'] == 'hard_sl']
    winners = [t for t in all_trades if t['pnl'] > 0]

    return sl_hits, winners

def analyze_indicator_patterns(sl_hits: List[Dict], winners: List[Dict]):
    """Compare indicator distributions between SL hits and winners."""

    print("=" * 100)
    print("INDICATOR PATTERN ANALYSIS - SL HITS VS WINNERS")
    print("=" * 100)
    print()

    indicators = ['vwap', 'ema20', 'ema50', 'rsi', 'adx', 'atr', 'volume_ratio',
                  'distance_from_vwap', 'rank_score', 'confidence', 'structural_rr']

    print(f"Sample Size: {len(sl_hits)} SL hits, {len(winners)} winners")
    print()

    for indicator in indicators:
        # Extract values (filter None)
        sl_values = [t[indicator] for t in sl_hits if t.get(indicator) is not None]
        winner_values = [t[indicator] for t in winners if t.get(indicator) is not None]

        if not sl_values or not winner_values:
            continue

        # Calculate statistics
        sl_mean = np.mean(sl_values)
        sl_median = np.median(sl_values)
        sl_std = np.std(sl_values)

        winner_mean = np.mean(winner_values)
        winner_median = np.median(winner_values)
        winner_std = np.std(winner_values)

        # Calculate difference
        mean_diff = ((sl_mean - winner_mean) / winner_mean * 100) if winner_mean != 0 else 0

        print(f"{indicator.upper()}:")
        print(f"  SL Hits:  Mean={sl_mean:8.2f}, Median={sl_median:8.2f}, StdDev={sl_std:8.2f}")
        print(f"  Winners:  Mean={winner_mean:8.2f}, Median={winner_median:8.2f}, StdDev={winner_std:8.2f}")
        print(f"  Difference: {mean_diff:+6.1f}%")

        # Flag significant differences
        if abs(mean_diff) > 15:
            if mean_diff > 0:
                print(f"  ⚠️  SL HITS have HIGHER {indicator} → Consider LOWERING threshold")
            else:
                print(f"  ⚠️  SL HITS have LOWER {indicator} → Consider RAISING threshold")

        print()

def analyze_setup_patterns(sl_hits: List[Dict], winners: List[Dict]):
    """Analyze which setups have highest SL rate."""

    print("=" * 100)
    print("SETUP-SPECIFIC SL RATE ANALYSIS")
    print("=" * 100)
    print()

    # Count by setup
    setup_sl_count = defaultdict(int)
    setup_winner_count = defaultdict(int)
    setup_total = defaultdict(int)

    for trade in sl_hits:
        setup = trade['setup_type']
        setup_sl_count[setup] += 1
        setup_total[setup] += 1

    for trade in winners:
        setup = trade['setup_type']
        setup_winner_count[setup] += 1
        setup_total[setup] += 1

    # Calculate SL rates
    setup_stats = []
    for setup in setup_total.keys():
        total = setup_total[setup]
        sl_count = setup_sl_count[setup]
        winner_count = setup_winner_count[setup]
        sl_rate = (sl_count / total * 100) if total > 0 else 0

        setup_stats.append({
            'setup': setup,
            'total': total,
            'sl_count': sl_count,
            'winner_count': winner_count,
            'sl_rate': sl_rate
        })

    # Sort by SL rate
    setup_stats.sort(key=lambda x: x['sl_rate'], reverse=True)

    print("Setups Ranked by SL Hit Rate:")
    print()

    for i, stats in enumerate(setup_stats, 1):
        if stats['total'] < 3:
            continue

        print(f"{i}. {stats['setup']}")
        print(f"   Total: {stats['total']}, SL Hits: {stats['sl_count']}, Winners: {stats['winner_count']}")
        print(f"   SL Rate: {stats['sl_rate']:.1f}%")

        if stats['sl_rate'] > 50:
            print(f"   ⚠️  CRITICAL - More than half hit SL!")
        elif stats['sl_rate'] > 40:
            print(f"   ⚠️  HIGH - Needs investigation")

        print()

def analyze_regime_patterns(sl_hits: List[Dict], winners: List[Dict]):
    """Analyze SL rate by regime."""

    print("=" * 100)
    print("REGIME-SPECIFIC SL RATE ANALYSIS")
    print("=" * 100)
    print()

    regime_sl_count = defaultdict(int)
    regime_winner_count = defaultdict(int)
    regime_total = defaultdict(int)

    for trade in sl_hits:
        regime = trade['regime']
        regime_sl_count[regime] += 1
        regime_total[regime] += 1

    for trade in winners:
        regime = trade['regime']
        regime_winner_count[regime] += 1
        regime_total[regime] += 1

    for regime in sorted(regime_total.keys()):
        total = regime_total[regime]
        sl_count = regime_sl_count[regime]
        winner_count = regime_winner_count[regime]
        sl_rate = (sl_count / total * 100) if total > 0 else 0

        print(f"{regime.upper()}:")
        print(f"  Total: {total}, SL Hits: {sl_count}, Winners: {winner_count}")
        print(f"  SL Rate: {sl_rate:.1f}%")

        if sl_rate > 45:
            print(f"  ⚠️  HIGH SL RATE - Consider blocking setups in this regime")

        print()

def analyze_timing_patterns(sl_hits: List[Dict], winners: List[Dict]):
    """Analyze decision hour patterns for SL hits."""

    print("=" * 100)
    print("TIMING PATTERN ANALYSIS - DECISION HOUR")
    print("=" * 100)
    print()

    hourly_sl = defaultdict(int)
    hourly_winner = defaultdict(int)
    hourly_total = defaultdict(int)

    for trade in sl_hits:
        try:
            hour = pd.to_datetime(trade['timestamp']).hour
            hourly_sl[hour] += 1
            hourly_total[hour] += 1
        except:
            continue

    for trade in winners:
        try:
            hour = pd.to_datetime(trade['timestamp']).hour
            hourly_winner[hour] += 1
            hourly_total[hour] += 1
        except:
            continue

    for hour in sorted(hourly_total.keys()):
        total = hourly_total[hour]
        sl_count = hourly_sl[hour]
        winner_count = hourly_winner[hour]
        sl_rate = (sl_count / total * 100) if total > 0 else 0

        print(f"{hour:02d}:00-{hour+1:02d}:00:")
        print(f"  Total: {total}, SL Hits: {sl_count}, Winners: {winner_count}")
        print(f"  SL Rate: {sl_rate:.1f}%")

        if sl_rate > 50:
            print(f"  ⚠️  TOXIC HOUR - >50% SL rate")

        print()

def analyze_rank_score_calibration(sl_hits: List[Dict], winners: List[Dict]):
    """Check if rank_score predicts SL vs winners."""

    print("=" * 100)
    print("RANK SCORE CALIBRATION - DOES IT PREDICT SL AVOIDANCE?")
    print("=" * 100)
    print()

    # Bucket by rank_score
    buckets = {
        '<2.0': [],
        '2.0-2.5': [],
        '2.5-3.0': [],
        '3.0-3.5': [],
        '3.5-4.0': [],
        '>4.0': []
    }

    def get_bucket(score):
        if score < 2.0:
            return '<2.0'
        elif score < 2.5:
            return '2.0-2.5'
        elif score < 3.0:
            return '2.5-3.0'
        elif score < 3.5:
            return '3.0-3.5'
        elif score < 4.0:
            return '3.5-4.0'
        else:
            return '>4.0'

    for trade in sl_hits:
        bucket = get_bucket(trade['rank_score'])
        buckets[bucket].append(('sl', trade))

    for trade in winners:
        bucket = get_bucket(trade['rank_score'])
        buckets[bucket].append(('winner', trade))

    print("Rank Score Distribution:")
    print()

    for bucket_name in ['<2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0', '>4.0']:
        trades = buckets[bucket_name]
        if not trades:
            continue

        total = len(trades)
        sl_count = sum(1 for t in trades if t[0] == 'sl')
        winner_count = sum(1 for t in trades if t[0] == 'winner')
        sl_rate = (sl_count / total * 100) if total > 0 else 0

        print(f"Rank Score {bucket_name}:")
        print(f"  Total: {total}, SL Hits: {sl_count}, Winners: {winner_count}")
        print(f"  SL Rate: {sl_rate:.1f}%")

        if sl_rate > 45:
            print(f"  ⚠️  HIGH SL RATE - Rank score not filtering well in this range")

        print()

def analyze_risk_reward_patterns(sl_hits: List[Dict], winners: List[Dict]):
    """Analyze structural_rr distribution."""

    print("=" * 100)
    print("RISK:REWARD (structural_rr) ANALYSIS")
    print("=" * 100)
    print()

    # Bucket by structural_rr
    buckets = {
        '<2.0': {'sl': 0, 'winner': 0},
        '2.0-2.5': {'sl': 0, 'winner': 0},
        '2.5-3.0': {'sl': 0, 'winner': 0},
        '3.0-4.0': {'sl': 0, 'winner': 0},
        '>4.0': {'sl': 0, 'winner': 0}
    }

    def get_rr_bucket(rr):
        if rr < 2.0:
            return '<2.0'
        elif rr < 2.5:
            return '2.0-2.5'
        elif rr < 3.0:
            return '2.5-3.0'
        elif rr < 4.0:
            return '3.0-4.0'
        else:
            return '>4.0'

    for trade in sl_hits:
        rr = trade.get('structural_rr', 0)
        if rr > 0:
            bucket = get_rr_bucket(rr)
            buckets[bucket]['sl'] += 1

    for trade in winners:
        rr = trade.get('structural_rr', 0)
        if rr > 0:
            bucket = get_rr_bucket(rr)
            buckets[bucket]['winner'] += 1

    print("Structural R:R Distribution:")
    print()

    for bucket_name in ['<2.0', '2.0-2.5', '2.5-3.0', '3.0-4.0', '>4.0']:
        stats = buckets[bucket_name]
        total = stats['sl'] + stats['winner']

        if total == 0:
            continue

        sl_rate = (stats['sl'] / total * 100)

        print(f"R:R {bucket_name}:")
        print(f"  Total: {total}, SL Hits: {stats['sl']}, Winners: {stats['winner']}")
        print(f"  SL Rate: {sl_rate:.1f}%")

        if sl_rate > 50 and total > 5:
            print(f"  ⚠️  PARADOX - High R:R but high SL rate (stop too tight?)")

        print()

def find_common_patterns(sl_hits: List[Dict]):
    """Find common combinations that frequently lead to SL."""

    print("=" * 100)
    print("COMMON SL HIT PATTERNS (Multi-Factor Analysis)")
    print("=" * 100)
    print()

    # Pattern: Low ADX + High Volume Ratio
    pattern1 = []
    for trade in sl_hits:
        adx = trade.get('adx')
        vol_ratio = trade.get('volume_ratio')
        if adx and vol_ratio and adx < 20 and vol_ratio > 4:
            pattern1.append(trade)

    if pattern1:
        print(f"PATTERN 1: Low ADX (<20) + High Volume Ratio (>4)")
        print(f"  Found in {len(pattern1)} SL hits ({len(pattern1)/len(sl_hits)*100:.1f}% of all SL hits)")
        print(f"  → INSIGHT: Choppy market with volume spike = False breakout")
        print()

    # Pattern: Extreme RSI + Low Rank Score
    pattern2 = []
    for trade in sl_hits:
        rsi = trade.get('rsi')
        rank_score = trade.get('rank_score', 0)
        if rsi and (rsi > 70 or rsi < 30) and rank_score < 2.5:
            pattern2.append(trade)

    if pattern2:
        print(f"PATTERN 2: Extreme RSI (>70 or <30) + Low Rank Score (<2.5)")
        print(f"  Found in {len(pattern2)} SL hits ({len(pattern2)/len(sl_hits)*100:.1f}% of all SL hits)")
        print(f"  → INSIGHT: Overbought/oversold condition with weak setup = Reversal risk")
        print()

    # Pattern: Morning trades (10:00-12:00) + Chop regime
    pattern3 = []
    for trade in sl_hits:
        try:
            hour = pd.to_datetime(trade['timestamp']).hour
            regime = trade.get('regime', '')
            if 10 <= hour < 12 and regime == 'chop':
                pattern3.append(trade)
        except:
            continue

    if pattern3:
        print(f"PATTERN 3: Morning Hours (10:00-12:00) + Chop Regime")
        print(f"  Found in {len(pattern3)} SL hits ({len(pattern3)/len(sl_hits)*100:.1f}% of all SL hits)")
        print(f"  → INSIGHT: Morning chop = Range-bound, avoid directional trades")
        print()

    # Pattern: High structural_rr (>3.5) + Hard SL
    pattern4 = []
    for trade in sl_hits:
        rr = trade.get('structural_rr', 0)
        if rr > 3.5:
            pattern4.append(trade)

    if pattern4:
        print(f"PATTERN 4: High R:R (>3.5) + SL Hit")
        print(f"  Found in {len(pattern4)} SL hits ({len(pattern4)/len(sl_hits)*100:.1f}% of all SL hits)")
        print(f"  → INSIGHT: Wide targets with tight stops = Stop too close to entry")
        print()

def generate_recommendations(sl_hits: List[Dict], winners: List[Dict]):
    """Generate actionable recommendations based on patterns."""

    print("=" * 100)
    print("ACTIONABLE RECOMMENDATIONS TO REDUCE SL HITS")
    print("=" * 100)
    print()

    # Calculate some key metrics
    sl_adx_values = [t['adx'] for t in sl_hits if t.get('adx') is not None]
    winner_adx_values = [t['adx'] for t in winners if t.get('adx') is not None]

    sl_vol_ratio = [t['volume_ratio'] for t in sl_hits if t.get('volume_ratio') is not None]
    winner_vol_ratio = [t['volume_ratio'] for t in winners if t.get('volume_ratio') is not None]

    sl_rank_scores = [t['rank_score'] for t in sl_hits if t.get('rank_score') is not None]
    winner_rank_scores = [t['rank_score'] for t in winners if t.get('rank_score') is not None]

    recommendations = []

    # ADX threshold
    if sl_adx_values and winner_adx_values:
        sl_adx_mean = np.mean(sl_adx_values)
        winner_adx_mean = np.mean(winner_adx_values)

        if sl_adx_mean < winner_adx_mean * 0.85:
            recommendations.append({
                'priority': 'HIGH',
                'finding': f'SL hits have lower ADX (avg: {sl_adx_mean:.1f}) vs winners (avg: {winner_adx_mean:.1f})',
                'action': f'Add filter: ADX > {winner_adx_mean * 0.9:.1f} for all setups',
                'expected_impact': f'Filter out ~{len([v for v in sl_adx_values if v < winner_adx_mean * 0.9])}/{len(sl_hits)} SL hits'
            })

    # Volume ratio threshold
    if sl_vol_ratio and winner_vol_ratio:
        sl_vol_mean = np.mean(sl_vol_ratio)
        winner_vol_mean = np.mean(winner_vol_ratio)

        if sl_vol_mean > winner_vol_mean * 1.15:
            recommendations.append({
                'priority': 'MEDIUM',
                'finding': f'SL hits have higher volume spikes (avg: {sl_vol_mean:.1f}x) vs winners (avg: {winner_vol_mean:.1f}x)',
                'action': f'Add filter: volume_ratio < {winner_vol_mean * 1.2:.1f} (avoid extreme spikes)',
                'expected_impact': f'Filter out ~{len([v for v in sl_vol_ratio if v > winner_vol_mean * 1.2])}/{len(sl_hits)} SL hits'
            })

    # Rank score threshold
    if sl_rank_scores and winner_rank_scores:
        sl_rank_mean = np.mean(sl_rank_scores)
        winner_rank_mean = np.mean(winner_rank_scores)

        if sl_rank_mean < winner_rank_mean * 0.90:
            recommendations.append({
                'priority': 'HIGH',
                'finding': f'SL hits have lower rank_scores (avg: {sl_rank_mean:.2f}) vs winners (avg: {winner_rank_mean:.2f})',
                'action': f'Raise min_rank_score threshold from 2.0 to {winner_rank_mean * 0.95:.2f}',
                'expected_impact': f'Filter out ~{len([v for v in sl_rank_scores if v < winner_rank_mean * 0.95])}/{len(sl_hits)} SL hits'
            })

    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    recommendations.sort(key=lambda x: priority_order[x['priority']])

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['priority']}] {rec['finding']}")
        print(f"   Action: {rec['action']}")
        print(f"   Expected Impact: {rec['expected_impact']}")
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_sl_hit_patterns.py <logs_dir>")
        print("Example: python tools/analyze_sl_hit_patterns.py backtest_20251119-082113_extracted/20251119-082113_full/20251119-082113")
        sys.exit(1)

    logs_dir = Path(sys.argv[1])

    if not logs_dir.exists():
        print(f"Error: Directory {logs_dir} does not exist")
        sys.exit(1)

    print(f"Loading trade data from: {logs_dir}")
    print()

    sl_hits, winners = load_trade_data(logs_dir)

    print(f"Loaded {len(sl_hits)} SL hits and {len(winners)} winners")
    print()

    # Run all analyses
    analyze_indicator_patterns(sl_hits, winners)
    analyze_setup_patterns(sl_hits, winners)
    analyze_regime_patterns(sl_hits, winners)
    analyze_timing_patterns(sl_hits, winners)
    analyze_rank_score_calibration(sl_hits, winners)
    analyze_risk_reward_patterns(sl_hits, winners)
    find_common_patterns(sl_hits)
    generate_recommendations(sl_hits, winners)

if __name__ == '__main__':
    main()
