"""
Regime Validator - Validates if market regime classification matches actual ADX/BB values

This script:
1. Loads trade_report.csv with market_regime column and indicator values
2. Validates if regime classification logic is working correctly
3. Identifies misclassifications (e.g., "chop" with ADX > 25)
4. Provides statistics on regime classification accuracy
"""

import sys
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Expected regime thresholds (from config)
REGIME_THRESHOLDS = {
    'chop': {
        'adx_max': 20,
        'bb_width_max': 0.025,
        'description': 'Low ADX (<20), narrow BB'
    },
    'trend_up': {
        'adx_min': 25,
        'rsi_min': 50,
        'description': 'High ADX (>25), RSI > 50, price > SMA'
    },
    'trend_down': {
        'adx_min': 25,
        'rsi_max': 50,
        'description': 'High ADX (>25), RSI < 50, price < SMA'
    },
    'squeeze': {
        'bb_width_max': 0.015,
        'adx_max': 15,
        'description': 'Very narrow BB (<0.015), very low ADX'
    }
}


def validate_regime_classification(df):
    """Validate if market_regime matches actual indicator values"""

    results = {
        'total_trades': len(df),
        'regimes': {},
        'misclassifications': [],
        'accuracy': {}
    }

    for regime in ['chop', 'trend_up', 'trend_down', 'squeeze']:
        regime_trades = df[df['regime'] == regime]

        if len(regime_trades) == 0:
            continue

        regime_result = {
            'count': len(regime_trades),
            'adx_range': (regime_trades['adx5'].min(), regime_trades['adx5'].max()),
            'adx_mean': regime_trades['adx5'].mean(),
            'bb_width_range': (regime_trades['bb_width_proxy'].min(), regime_trades['bb_width_proxy'].max()),
            'bb_width_mean': regime_trades['bb_width_proxy'].mean(),
            'valid_count': 0,
            'invalid_count': 0,
            'invalid_trades': []
        }

        # Validate each trade in this regime
        for idx, row in regime_trades.iterrows():
            is_valid = validate_single_regime(regime, row)

            if is_valid:
                regime_result['valid_count'] += 1
            else:
                regime_result['invalid_count'] += 1
                regime_result['invalid_trades'].append({
                    'symbol': row['symbol'],
                    'entry_ts': str(row.get('entry_ts', row.get('decision_ts', 'N/A'))),
                    'adx': row['adx5'],
                    'bb_width': row['bb_width_proxy'],
                    'rsi': None,  # Not available in trade_report
                    'reason': get_misclassification_reason(regime, row)
                })

        # Calculate accuracy
        accuracy = (regime_result['valid_count'] / len(regime_trades) * 100) if len(regime_trades) > 0 else 0
        regime_result['accuracy'] = accuracy

        results['regimes'][regime] = regime_result
        results['accuracy'][regime] = accuracy

    # Overall accuracy
    total_valid = sum(r['valid_count'] for r in results['regimes'].values())
    results['overall_accuracy'] = (total_valid / results['total_trades'] * 100) if results['total_trades'] > 0 else 0

    return results


def validate_single_regime(regime, row):
    """Check if a single trade's regime classification is correct"""

    adx = row['adx5']
    bb_width = row['bb_width_proxy']
    # RSI not available in trade_report, so we can't validate trend direction

    if regime == 'chop':
        # CHOP should have ADX < 20
        return adx < REGIME_THRESHOLDS['chop']['adx_max']

    elif regime == 'trend_up':
        # TREND_UP should have ADX > 25 (can't check RSI without data)
        return adx >= REGIME_THRESHOLDS['trend_up']['adx_min']

    elif regime == 'trend_down':
        # TREND_DOWN should have ADX > 25 (can't check RSI without data)
        return adx >= REGIME_THRESHOLDS['trend_down']['adx_min']

    elif regime == 'squeeze':
        # SQUEEZE should have very narrow BB and low ADX
        return bb_width < REGIME_THRESHOLDS['squeeze']['bb_width_max'] and adx < REGIME_THRESHOLDS['squeeze']['adx_max']

    return True  # Unknown regime, assume valid


def get_misclassification_reason(regime, row):
    """Get human-readable reason for misclassification"""

    adx = row['adx5']
    bb_width = row['bb_width_proxy']

    if regime == 'chop':
        if adx >= REGIME_THRESHOLDS['chop']['adx_max']:
            return f"ADX {adx:.1f} >= 20 (should be < 20 for chop)"

    elif regime == 'trend_up':
        if adx < REGIME_THRESHOLDS['trend_up']['adx_min']:
            return f"ADX {adx:.1f} < 25 (should be >= 25 for trend)"

    elif regime == 'trend_down':
        if adx < REGIME_THRESHOLDS['trend_down']['adx_min']:
            return f"ADX {adx:.1f} < 25 (should be >= 25 for trend)"

    elif regime == 'squeeze':
        if bb_width >= REGIME_THRESHOLDS['squeeze']['bb_width_max']:
            return f"BB width {bb_width:.4f} >= 0.015 (should be < 0.015 for squeeze)"
        if adx >= REGIME_THRESHOLDS['squeeze']['adx_max']:
            return f"ADX {adx:.1f} >= 15 (should be < 15 for squeeze)"

    return "Unknown reason"


def find_trade_reports(run_prefix):
    """Find all trade_report.csv files for a run prefix"""
    logs_dir = ROOT / 'logs'
    sessions = sorted(logs_dir.glob(f'{run_prefix}_*'))

    reports = []
    for session_dir in sessions:
        report_file = session_dir / 'trade_report.csv'
        if report_file.exists():
            reports.append(report_file)

    return reports


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate market regime classification accuracy')
    parser.add_argument('run_prefix', help='Run prefix (e.g., run_5cd6bb35)')
    parser.add_argument('--show-invalid', action='store_true', help='Show all invalid classifications')
    args = parser.parse_args()

    # Find all trade reports
    report_files = find_trade_reports(args.run_prefix)

    if not report_files:
        print(f"No trade reports found for {args.run_prefix}")
        return

    # Load all trade reports
    all_trades = []
    for report_file in report_files:
        try:
            df = pd.read_csv(report_file)
            if not df.empty:
                all_trades.append(df)
        except Exception as e:
            print(f"Warning: Could not load {report_file}: {e}")

    if not all_trades:
        print(f"No valid trade data found for {args.run_prefix}")
        return

    # Combine all trades
    df_combined = pd.concat(all_trades, ignore_index=True)

    print(f"\n{'='*80}")
    print(f"REGIME VALIDATION REPORT - {args.run_prefix}")
    print(f"{'='*80}\n")

    print(f"Total trades analyzed: {len(df_combined)}")
    print(f"Sessions processed: {len(report_files)}\n")

    # Validate regimes
    results = validate_regime_classification(df_combined)

    # Print results
    print(f"OVERALL ACCURACY: {results['overall_accuracy']:.1f}%\n")

    print("REGIME BREAKDOWN:")
    print("-" * 80)

    for regime, data in results['regimes'].items():
        print(f"\n{regime.upper()} ({data['count']} trades)")
        print(f"  Accuracy: {data['accuracy']:.1f}% ({data['valid_count']}/{data['count']})")
        print(f"  ADX range: {data['adx_range'][0]:.1f} - {data['adx_range'][1]:.1f} (mean: {data['adx_mean']:.1f})")
        print(f"  BB width range: {data['bb_width_range'][0]:.4f} - {data['bb_width_range'][1]:.4f} (mean: {data['bb_width_mean']:.4f})")

        # Expected thresholds
        if regime in REGIME_THRESHOLDS:
            print(f"  Expected: {REGIME_THRESHOLDS[regime]['description']}")

        # Warning if accuracy < 80%
        if data['accuracy'] < 80:
            print(f"  WARNING: Low accuracy - regime detection may be broken!")

        # Show invalid trades if requested
        if args.show_invalid and data['invalid_trades']:
            print(f"\n  Invalid classifications ({data['invalid_count']}):")
            for invalid in data['invalid_trades'][:5]:  # Show first 5
                print(f"    - {invalid['symbol']} @ {invalid['entry_ts']}: {invalid['reason']}")
            if data['invalid_count'] > 5:
                print(f"    ... and {data['invalid_count'] - 5} more")

    # Critical issues
    print(f"\n{'='*80}")
    print("CRITICAL ISSUES:")
    print("-" * 80)

    issues_found = False

    for regime, data in results['regimes'].items():
        if data['accuracy'] < 50:
            print(f"CRITICAL: {regime} regime has {data['accuracy']:.1f}% accuracy (< 50%)")
            print(f"   Root cause: Regime detection thresholds likely incorrect")
            issues_found = True
        elif data['accuracy'] < 80:
            print(f"WARNING: {regime} regime has {data['accuracy']:.1f}% accuracy (< 80%)")
            print(f"   Recommendation: Review regime detection parameters")
            issues_found = True

    if not issues_found:
        print("No critical regime classification issues found")

    # Save detailed results
    output_file = ROOT / f'regime_validation_{args.run_prefix}.json'

    # Convert results to JSON-serializable format
    def convert_to_serializable(obj):
        """Convert numpy/pandas types to Python native types"""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj

    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
