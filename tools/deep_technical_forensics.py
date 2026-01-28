#!/usr/bin/env python3
"""
Deep Technical Forensics - Analyze WHY setups fail at mechanical level
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_setup_technically(sessions_root, setup_name):
    """Deep dive into a single setup's technical characteristics"""

    decisions = []
    triggered_trades = []

    # Collect all decisions and triggered trades
    for session_dir in Path(sessions_root).iterdir():
        if not session_dir.is_dir():
            continue

        events_file = session_dir / 'events.jsonl'
        if not events_file.exists():
            continue

        with open(events_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)

                if event.get('type') == 'DECISION' and event.get('decision', {}).get('setup_type') == setup_name:
                    event['session'] = session_dir.name
                    decisions.append(event)

                elif event.get('type') == 'TRIGGER' and event.get('trigger', {}).get('strategy') == setup_name:
                    triggered_trades.append(event)

    if not decisions:
        return None

    # Load actual trade results
    trade_results = {}
    for session_dir in Path(sessions_root).iterdir():
        if not session_dir.is_dir():
            continue
        analytics_file = session_dir / 'analytics.jsonl'
        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    trade = json.loads(line)
                    if trade.get('stage') == 'EXIT' and trade.get('setup_type') == setup_name:
                        trade_results[trade.get('trade_id')] = trade

    triggered_ids = {t['trade_id'] for t in triggered_trades}

    # Separate into triggered vs non-triggered
    triggered_decisions = [d for d in decisions if d['trade_id'] in triggered_ids]
    non_triggered_decisions = [d for d in decisions if d['trade_id'] not in triggered_ids]

    # Analyze technical characteristics
    analysis = {
        'setup_name': setup_name,
        'total_decisions': len(decisions),
        'triggered': len(triggered_decisions),
        'non_triggered': len(non_triggered_decisions),
        'trigger_rate': len(triggered_decisions) / len(decisions) * 100 if decisions else 0,
        'triggered_characteristics': analyze_technical_params(triggered_decisions),
        'non_triggered_characteristics': analyze_technical_params(non_triggered_decisions),
        'trade_results': analyze_trade_outcomes(triggered_decisions, trade_results),
        'sample_trades': get_sample_trades(triggered_decisions, trade_results, limit=5)
    }

    return analysis


def analyze_technical_params(decisions):
    """Extract technical indicator statistics"""
    if not decisions:
        return {}

    adx_values = []
    rsi_values = []
    vol_ratios = []
    rank_scores = []
    macd_hists = []
    price_vs_vwaps = []
    structural_rrs = []

    acceptance_counts = defaultdict(int)
    regime_counts = defaultdict(int)

    for dec in decisions:
        plan = dec.get('plan', {})
        indicators = plan.get('indicators', {})
        quality = plan.get('quality', {})
        features = dec.get('features', {})

        if indicators.get('adx14'):
            adx_values.append(indicators['adx14'])
        if indicators.get('rsi14'):
            rsi_values.append(indicators['rsi14'])
        if indicators.get('vol_ratio'):
            vol_ratios.append(indicators['vol_ratio'])
        if indicators.get('macd_hist'):
            macd_hists.append(indicators['macd_hist'])
        if indicators.get('vwap') and plan.get('entry', {}).get('reference'):
            price = plan['entry']['reference']
            vwap = indicators['vwap']
            price_vs_vwaps.append((price - vwap) / vwap * 100)

        if features.get('rank_score'):
            rank_scores.append(features['rank_score'])
        if quality.get('structural_rr'):
            structural_rrs.append(quality['structural_rr'])
        if quality.get('acceptance_status'):
            acceptance_counts[quality['acceptance_status']] += 1
        if plan.get('regime'):
            regime_counts[plan['regime']] += 1

    def safe_stats(values):
        if not values:
            return {'count': 0}
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0
        }

    return {
        'adx': safe_stats(adx_values),
        'rsi': safe_stats(rsi_values),
        'vol_ratio': safe_stats(vol_ratios),
        'rank_score': safe_stats(rank_scores),
        'macd_hist': safe_stats(macd_hists),
        'price_vs_vwap_pct': safe_stats(price_vs_vwaps),
        'structural_rr': safe_stats(structural_rrs),
        'acceptance_status': dict(acceptance_counts),
        'regime': dict(regime_counts)
    }


def analyze_trade_outcomes(triggered_decisions, trade_results):
    """Analyze actual trade outcomes"""
    if not triggered_decisions:
        return {}

    outcomes = {
        'total': len(triggered_decisions),
        'with_results': 0,
        'winners': 0,
        'losers': 0,
        'pnls': [],
        'exit_reasons': defaultdict(int)
    }

    for dec in triggered_decisions:
        trade_id = dec['trade_id']
        if trade_id in trade_results:
            trade = trade_results[trade_id]
            outcomes['with_results'] += 1
            pnl = trade.get('pnl', 0)
            outcomes['pnls'].append(pnl)
            if pnl > 0:
                outcomes['winners'] += 1
            else:
                outcomes['losers'] += 1
            if trade.get('exit_reason'):
                outcomes['exit_reasons'][trade['exit_reason']] += 1

    if outcomes['pnls']:
        outcomes['total_pnl'] = sum(outcomes['pnls'])
        outcomes['avg_pnl'] = statistics.mean(outcomes['pnls'])
        outcomes['win_rate'] = outcomes['winners'] / len(outcomes['pnls']) * 100

    return outcomes


def get_sample_trades(triggered_decisions, trade_results, limit=5):
    """Get detailed sample trades"""
    samples = []

    for dec in triggered_decisions[:limit]:
        trade_id = dec['trade_id']
        plan = dec.get('plan', {})
        indicators = plan.get('indicators', {})

        sample = {
            'symbol': dec.get('symbol'),
            'session': dec.get('session'),
            'entry': plan.get('entry', {}).get('reference'),
            'stop': plan.get('stop', {}).get('hard'),
            'regime': plan.get('regime'),
            'adx': indicators.get('adx14'),
            'rsi': indicators.get('rsi14'),
            'vol_ratio': indicators.get('vol_ratio'),
            'rank_score': dec.get('features', {}).get('rank_score'),
            'acceptance': plan.get('quality', {}).get('acceptance_status')
        }

        if trade_id in trade_results:
            trade = trade_results[trade_id]
            sample['pnl'] = trade.get('pnl')
            sample['exit_reason'] = trade.get('exit_reason')
            sample['outcome'] = 'WIN' if trade.get('pnl', 0) > 0 else 'LOSS'

        samples.append(sample)

    return samples


def compare_triggered_vs_nontriggered(analysis):
    """Compare technical characteristics between triggered and non-triggered"""
    trig = analysis['triggered_characteristics']
    non_trig = analysis['non_triggered_characteristics']

    print(f"\n{'='*100}")
    print(f"TRIGGERED vs NON-TRIGGERED COMPARISON")
    print(f"{'='*100}\n")

    print(f"Trigger Rate: {analysis['trigger_rate']:.1f}% ({analysis['triggered']}/{analysis['total_decisions']} decisions)\n")

    # Compare key metrics
    metrics = ['adx', 'rsi', 'vol_ratio', 'rank_score', 'structural_rr', 'price_vs_vwap_pct']

    print(f"{'Metric':<25} {'Triggered':<30} {'Non-Triggered':<30} {'Difference':<20}")
    print(f"{'-'*100}")

    for metric in metrics:
        if trig.get(metric, {}).get('count', 0) > 0 and non_trig.get(metric, {}).get('count', 0) > 0:
            trig_mean = trig[metric]['mean']
            non_trig_mean = non_trig[metric]['mean']
            diff = trig_mean - non_trig_mean
            diff_pct = (diff / non_trig_mean * 100) if non_trig_mean != 0 else 0

            print(f"{metric:<25} {trig_mean:>8.2f} (n={trig[metric]['count']:<3}) {non_trig_mean:>12.2f} (n={non_trig[metric]['count']:<3}) {diff:>8.2f} ({diff_pct:>+5.1f}%)")

    # Acceptance status comparison
    print(f"\n{'Acceptance Status':<25} {'Triggered':<30} {'Non-Triggered':<30}")
    print(f"{'-'*80}")
    all_statuses = set(list(trig.get('acceptance_status', {}).keys()) + list(non_trig.get('acceptance_status', {}).keys()))
    for status in sorted(all_statuses):
        trig_count = trig.get('acceptance_status', {}).get(status, 0)
        non_trig_count = non_trig.get('acceptance_status', {}).get(status, 0)
        print(f"{status:<25} {trig_count:<30} {non_trig_count:<30}")

    # Regime comparison
    print(f"\n{'Regime':<25} {'Triggered':<30} {'Non-Triggered':<30}")
    print(f"{'-'*80}")
    all_regimes = set(list(trig.get('regime', {}).keys()) + list(non_trig.get('regime', {}).keys()))
    for regime in sorted(all_regimes):
        trig_count = trig.get('regime', {}).get(regime, 0)
        non_trig_count = non_trig.get('regime', {}).get(status, 0)
        print(f"{regime:<25} {trig_count:<30} {non_trig_count:<30}")


def print_trade_outcomes(analysis):
    """Print trade outcome analysis"""
    outcomes = analysis['trade_results']

    if not outcomes or outcomes.get('total', 0) == 0:
        print("\nNo trade results available")
        return

    print(f"\n{'='*100}")
    print(f"TRADE OUTCOMES")
    print(f"{'='*100}\n")

    print(f"Total Triggered: {outcomes['total']}")
    print(f"With Results: {outcomes['with_results']}")
    print(f"Winners: {outcomes.get('winners', 0)} | Losers: {outcomes.get('losers', 0)}")
    print(f"Win Rate: {outcomes.get('win_rate', 0):.1f}%")
    print(f"Total PnL: Rs.{outcomes.get('total_pnl', 0):.2f}")
    print(f"Avg PnL: Rs.{outcomes.get('avg_pnl', 0):.2f}")

    print(f"\nExit Reasons:")
    for reason, count in sorted(outcomes.get('exit_reasons', {}).items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")


def print_sample_trades(analysis):
    """Print sample trades with full details"""
    samples = analysis.get('sample_trades', [])

    if not samples:
        return

    print(f"\n{'='*100}")
    print(f"SAMPLE TRADES (First 5)")
    print(f"{'='*100}\n")

    for i, sample in enumerate(samples, 1):
        print(f"Trade {i}: {sample['symbol']} on {sample['session']}")
        print(f"  Entry: {sample['entry']} | Stop: {sample['stop']} | Regime: {sample['regime']}")
        print(f"  ADX: {sample.get('adx', 'N/A')} | RSI: {sample.get('rsi', 'N/A')} | Vol Ratio: {sample.get('vol_ratio', 'N/A')}")
        print(f"  Rank: {sample.get('rank_score', 'N/A')} | Acceptance: {sample.get('acceptance', 'N/A')}")
        if 'pnl' in sample:
            print(f"  Outcome: {sample.get('outcome', 'UNKNOWN')} | PnL: Rs.{sample.get('pnl', 0):.2f} | Exit: {sample.get('exit_reason', 'N/A')}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python deep_technical_forensics.py <sessions_root> [setup_name]")
        print("\nAnalyzes technical characteristics of setups to identify fixable issues")
        print("\nExample:")
        print("  python deep_technical_forensics.py backtest_extracted/sessions break_of_structure_long")
        sys.exit(1)

    sessions_root = sys.argv[1]

    # Setups to analyze (default to known failing ones)
    setups_to_analyze = [
        'break_of_structure_long',
        'break_of_structure_short',
        'change_of_character_long',
        'fair_value_gap_long',
        # Compare with successful ones
        'orb_breakdown_short',
        'premium_zone_short',
        'volume_spike_reversal_long'
    ]

    if len(sys.argv) >= 3:
        setups_to_analyze = [sys.argv[2]]

    for setup_name in setups_to_analyze:
        print(f"\n{'#'*100}")
        print(f"# DEEP TECHNICAL FORENSICS: {setup_name.upper()}")
        print(f"{'#'*100}")

        analysis = analyze_setup_technically(sessions_root, setup_name)

        if not analysis:
            print(f"\nNo decisions found for {setup_name}")
            continue

        compare_triggered_vs_nontriggered(analysis)
        print_trade_outcomes(analysis)
        print_sample_trades(analysis)

        print(f"\n{'='*100}\n")


if __name__ == '__main__':
    main()
