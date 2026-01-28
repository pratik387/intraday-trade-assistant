"""
Rank Score Calibrator - Validates if ranking system predicts trade outcomes

This script:
1. Loads ranking.jsonl with rank_scores for each symbol
2. Loads analytics.jsonl to get actual PnL outcomes
3. Calculates correlation between rank_score and PnL
4. Identifies optimal rank_exec_threshold
5. Validates if "excellent" scores actually perform better
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_ranking_data(session_dir):
    """Load ranking data from ranking.jsonl"""
    ranking_file = session_dir / 'ranking.jsonl'

    if not ranking_file.exists():
        return []

    rankings = []
    with open(ranking_file) as f:
        for line in f:
            try:
                event = json.loads(line)
                rankings.append({
                    'symbol': event.get('symbol'),
                    'rank_score': event.get('rank_score'),
                    'quality_rating': event.get('quality_rating'),
                    'setup_type': event.get('setup_type'),
                    'timestamp': event.get('timestamp')
                })
            except Exception:
                continue

    return rankings


def load_analytics_data(session_dir):
    """Load actual trade outcomes from analytics.jsonl"""
    analytics_file = session_dir / 'analytics.jsonl'

    if not analytics_file.exists():
        return []

    outcomes = []
    with open(analytics_file) as f:
        for line in f:
            try:
                event = json.loads(line)
                if event.get('stage') == 'EXIT':
                    outcomes.append({
                        'symbol': event.get('symbol'),
                        'pnl': event.get('pnl'),
                        'exit_reason': event.get('reason'),
                        'was_winner': event.get('pnl', 0) > 0,
                        'entry_time': event.get('timestamp')
                    })
            except Exception:
                continue

    return outcomes


def merge_rankings_with_outcomes(rankings, outcomes):
    """Merge ranking data with actual outcomes"""

    # Convert to DataFrames
    df_rank = pd.DataFrame(rankings)
    df_outcome = pd.DataFrame(outcomes)

    if df_rank.empty or df_outcome.empty:
        return pd.DataFrame()

    # Merge on symbol (assumes 1 trade per symbol per session)
    # For multiple trades on same symbol, this takes the first ranking
    df_merged = pd.merge(df_rank, df_outcome, on='symbol', how='inner')

    return df_merged


def calculate_correlation(df):
    """Calculate correlation between rank_score and PnL"""

    if df.empty or 'rank_score' not in df.columns or 'pnl' not in df.columns:
        return None

    # Remove NaN values
    df_clean = df[['rank_score', 'pnl']].dropna()

    if len(df_clean) < 3:  # Need at least 3 points for correlation
        return None

    correlation = df_clean['rank_score'].corr(df_clean['pnl'])

    return correlation


def analyze_by_quality_rating(df):
    """Analyze performance by quality rating (Excellent, Good, Fair, Poor)"""

    if df.empty or 'quality_rating' not in df.columns:
        return {}

    ratings = ['Excellent', 'Good', 'Fair', 'Poor']
    results = {}

    for rating in ratings:
        rating_trades = df[df['quality_rating'] == rating]

        if len(rating_trades) == 0:
            continue

        results[rating] = {
            'count': len(rating_trades),
            'win_rate': (rating_trades['was_winner'].sum() / len(rating_trades) * 100),
            'avg_pnl': rating_trades['pnl'].mean(),
            'total_pnl': rating_trades['pnl'].sum(),
            'avg_rank_score': rating_trades['rank_score'].mean()
        }

    return results


def find_optimal_threshold(df):
    """Find optimal rank_score threshold that maximizes win rate"""

    if df.empty or 'rank_score' not in df.columns:
        return None

    # Try different thresholds
    thresholds = np.arange(0, 10, 0.5)
    best_threshold = None
    best_win_rate = 0
    best_profit_factor = 0

    results = []

    for threshold in thresholds:
        above_threshold = df[df['rank_score'] >= threshold]

        if len(above_threshold) < 5:  # Need minimum sample size
            continue

        win_rate = (above_threshold['was_winner'].sum() / len(above_threshold) * 100)

        # Calculate profit factor
        winners = above_threshold[above_threshold['pnl'] > 0]['pnl'].sum()
        losers = abs(above_threshold[above_threshold['pnl'] < 0]['pnl'].sum())
        profit_factor = winners / losers if losers > 0 else 0

        results.append({
            'threshold': threshold,
            'trade_count': len(above_threshold),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_pnl': above_threshold['pnl'].mean(),
            'total_pnl': above_threshold['pnl'].sum()
        })

        # Track best by win rate
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_threshold = threshold
            best_profit_factor = profit_factor

    return {
        'best_threshold': best_threshold,
        'best_win_rate': best_win_rate,
        'best_profit_factor': best_profit_factor,
        'all_results': results
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate ranking system predictive power')
    parser.add_argument('run_prefix', help='Run prefix (e.g., run_5cd6bb35)')
    parser.add_argument('--show-thresholds', action='store_true', help='Show all threshold analysis')
    args = parser.parse_args()

    # Find all sessions
    logs_dir = ROOT / 'logs'
    sessions = sorted(logs_dir.glob(f'{args.run_prefix}_*'))

    if not sessions:
        print(f"No sessions found for {args.run_prefix}")
        return

    print(f"\n{'='*80}")
    print(f"RANK SCORE CALIBRATION REPORT - {args.run_prefix}")
    print(f"{'='*80}\n")

    # Aggregate data from all sessions
    all_data = []

    for session_dir in sessions:
        rankings = load_ranking_data(session_dir)
        outcomes = load_analytics_data(session_dir)

        if rankings and outcomes:
            merged = merge_rankings_with_outcomes(rankings, outcomes)
            if not merged.empty:
                all_data.append(merged)

    if not all_data:
        print("No data found with both rankings and outcomes")
        return

    # Combine all data
    df_combined = pd.concat(all_data, ignore_index=True)

    print(f"Total ranked trades with outcomes: {len(df_combined)}")
    print(f"Sessions analyzed: {len(sessions)}\n")

    # 1. Correlation Analysis
    correlation = calculate_correlation(df_combined)

    print("CORRELATION ANALYSIS:")
    print("-" * 80)
    if correlation is not None:
        print(f"Rank Score vs PnL Correlation: {correlation:.3f}")

        if correlation > 0.5:
            print("STRONG positive correlation - ranking system is working well")
        elif correlation > 0.3:
            print("MODERATE correlation - ranking system has some predictive power")
        elif correlation > 0:
            print("WEAK correlation - ranking system barely predicts outcomes")
        else:
            print("NEGATIVE correlation - ranking system is INVERTED!")

        # R-squared (explained variance)
        r_squared = correlation ** 2
        print(f"R² (variance explained): {r_squared:.3f} ({r_squared*100:.1f}%)")
    else:
        print("Could not calculate correlation (insufficient data)")

    # 2. Quality Rating Analysis
    print(f"\n{'='*80}")
    print("PERFORMANCE BY QUALITY RATING:")
    print("-" * 80)

    quality_results = analyze_by_quality_rating(df_combined)

    for rating in ['Excellent', 'Good', 'Fair', 'Poor']:
        if rating in quality_results:
            data = quality_results[rating]
            print(f"\n{rating.upper()} (Rank Score {data['avg_rank_score']:.2f})")
            print(f"  Trades: {data['count']}")
            print(f"  Win Rate: {data['win_rate']:.1f}%")
            print(f"  Avg PnL: Rs.{data['avg_pnl']:.2f}")
            print(f"  Total PnL: Rs.{data['total_pnl']:.2f}")

    # Check if ratings are predictive
    print(f"\n{'='*80}")
    print("QUALITY RATING VALIDATION:")
    print("-" * 80)

    if len(quality_results) >= 2:
        # Check if "Excellent" beats "Poor"
        if 'Excellent' in quality_results and 'Poor' in quality_results:
            excellent_wr = quality_results['Excellent']['win_rate']
            poor_wr = quality_results['Poor']['win_rate']

            if excellent_wr > poor_wr + 10:
                print(f"WORKING: Excellent ({excellent_wr:.1f}%) beats Poor ({poor_wr:.1f}%)")
            elif excellent_wr > poor_wr:
                print(f"WEAK: Excellent ({excellent_wr:.1f}%) only slightly better than Poor ({poor_wr:.1f}%)")
            else:
                print(f"BROKEN: Excellent ({excellent_wr:.1f}%) WORSE than Poor ({poor_wr:.1f}%)!")

    # 3. Optimal Threshold Analysis
    print(f"\n{'='*80}")
    print("OPTIMAL THRESHOLD ANALYSIS:")
    print("-" * 80)

    threshold_results = find_optimal_threshold(df_combined)

    if threshold_results:
        print(f"Optimal Rank Score Threshold: {threshold_results['best_threshold']:.1f}")
        print(f"  Win Rate at threshold: {threshold_results['best_win_rate']:.1f}%")
        print(f"  Profit Factor: {threshold_results['best_profit_factor']:.2f}")

        current_threshold = 1.0  # Assuming default is 1.0
        print(f"\nCurrent threshold (assumed): {current_threshold}")

        # Find current threshold performance
        current_perf = next((r for r in threshold_results['all_results'] if r['threshold'] == current_threshold), None)
        if current_perf:
            print(f"  Win Rate: {current_perf['win_rate']:.1f}%")
            print(f"  Trades: {current_perf['trade_count']}")

        if threshold_results['best_threshold'] != current_threshold:
            improvement = threshold_results['best_win_rate'] - (current_perf['win_rate'] if current_perf else 0)
            print(f"\nRecommendation: Change threshold from {current_threshold} to {threshold_results['best_threshold']:.1f}")
            print(f"   Expected improvement: +{improvement:.1f}% win rate")

        # Show threshold table if requested
        if args.show_thresholds:
            print(f"\nFull Threshold Analysis:")
            print(f"{'Threshold':<12} {'Trades':<10} {'Win Rate':<12} {'Profit Factor':<15} {'Avg PnL':<12}")
            print("-" * 70)
            for result in threshold_results['all_results']:
                print(f"{result['threshold']:<12.1f} {result['trade_count']:<10} {result['win_rate']:<12.1f}% {result['profit_factor']:<15.2f} Rs.{result['avg_pnl']:<12.2f}")

    # 4. Critical Issues
    print(f"\n{'='*80}")
    print("CRITICAL ISSUES:")
    print("-" * 80)

    issues_found = False

    if correlation is not None and correlation < 0.3:
        print(f"CRITICAL: Rank score correlation {correlation:.3f} < 0.3")
        print(f"   Root cause: Ranking formula not predictive of outcomes")
        print(f"   Action: Rebuild ranking system with walk-forward validation")
        issues_found = True

    if 'Excellent' in quality_results and 'Good' in quality_results:
        if quality_results['Excellent']['win_rate'] < quality_results['Good']['win_rate']:
            print(f"CRITICAL: 'Excellent' ({quality_results['Excellent']['win_rate']:.1f}%) worse than 'Good' ({quality_results['Good']['win_rate']:.1f}%)")
            print(f"   Root cause: Quality rating thresholds inverted or broken")
            issues_found = True

    if not issues_found:
        print("No critical ranking system issues found")

    # Save results
    output_file = ROOT / f'rank_calibration_{args.run_prefix}.json'

    output_data = {
        'correlation': float(correlation) if correlation is not None else None,
        'r_squared': float(correlation ** 2) if correlation is not None else None,
        'quality_ratings': quality_results,
        'optimal_threshold': threshold_results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
