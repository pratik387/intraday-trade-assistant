"""
Analyze regime distribution and ADX values for all failing setups.

Based on vwap_reclaim_long findings, check if other failing setups
have similar regime/ADX mismatches.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_setup_regime_distribution(backtest_dir: str, setup_name: str):
    """Analyze regime distribution and ADX for a specific setup."""

    base_path = Path(backtest_dir)
    setup_events = []

    for session_dir in sorted(base_path.iterdir()):
        if not session_dir.is_dir():
            continue

        events_file = session_dir / 'events.jsonl'
        if not events_file.exists():
            continue

        with open(events_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())

                    if (event.get('type') == 'DECISION' and
                        event.get('decision', {}).get('setup_type') == setup_name):

                        record = {
                            'session_date': session_dir.name,
                            'symbol': event.get('symbol'),
                            'regime': event.get('decision', {}).get('regime'),
                            'adx': event.get('bar5', {}).get('adx'),
                        }
                        setup_events.append(record)
                except:
                    continue

    if len(setup_events) == 0:
        return None

    df = pd.DataFrame(setup_events)

    # Load trades to get P&L by regime
    trades = []
    for session_dir in sorted(base_path.iterdir()):
        if not session_dir.is_dir():
            continue

        analytics_file = session_dir / 'analytics.jsonl'
        if not analytics_file.exists():
            continue

        with open(analytics_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    trade = json.loads(line.strip())
                    if trade.get('setup_type') == setup_name:
                        trade['session_date'] = session_dir.name
                        trades.append(trade)
                except:
                    continue

    trades_df = pd.DataFrame(trades)

    return {
        'setup_name': setup_name,
        'total_trades': len(df),
        'regime_distribution': df['regime'].value_counts().to_dict(),
        'adx_stats': {
            'mean': df['adx'].mean(),
            'median': df['adx'].median(),
            'min': df['adx'].min(),
            'max': df['adx'].max(),
        },
        'adx_by_regime': df.groupby('regime')['adx'].mean().to_dict(),
        'trades_pnl': trades_df['pnl'].sum() if len(trades_df) > 0 else 0,
        'pnl_by_regime': trades_df.groupby('regime')['pnl'].sum().to_dict() if len(trades_df) > 0 else {}
    }

def main():
    backtest_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251117-021749_extracted\20251117-021749_full\20251117-021749"

    # Top failing setups from the report
    failing_setups = [
        'order_block_long',           # Rs -2,259 (R:R 0.42)
        'fair_value_gap_long',        # Rs -2,158 (R:R 0.62)
        'flag_continuation_long',     # Rs -881 (R:R 0.85)
        'breakout_long',              # Rs -768
        'orb_breakout_long',          # Rs -688
        'liquidity_sweep_long',       # Rs -673 (ICT)
        'break_of_structure_long',    # Rs -580 (ICT)
        'trend_pullback_long',        # Rs -496
    ]

    print("="*80)
    print("REGIME & ADX ANALYSIS FOR FAILING SETUPS")
    print("="*80)
    print()

    results = {}

    for setup in failing_setups:
        result = analyze_setup_regime_distribution(backtest_dir, setup)

        if result is None:
            print(f"\n{setup}: No trades found")
            continue

        results[setup] = result

        print(f"\n{'='*80}")
        print(f"{setup.upper()}")
        print(f"{'='*80}")

        print(f"\nTotal Trades: {result['total_trades']}")
        print(f"Total P&L: Rs {result['trades_pnl']:,.2f}")

        print(f"\nADX Statistics:")
        print(f"  Mean: {result['adx_stats']['mean']:.1f}")
        print(f"  Median: {result['adx_stats']['median']:.1f}")
        print(f"  Range: {result['adx_stats']['min']:.1f} - {result['adx_stats']['max']:.1f}")

        print(f"\nRegime Distribution:")
        for regime, count in sorted(result['regime_distribution'].items(), key=lambda x: -x[1]):
            pct = count / result['total_trades'] * 100
            avg_adx = result['adx_by_regime'].get(regime, 0)
            pnl = result['pnl_by_regime'].get(regime, 0)
            print(f"  {regime:<15} {count:>3} trades ({pct:>5.1f}%)  ADX: {avg_adx:>5.1f}  P&L: Rs {pnl:>10,.2f}")

        # Check for regime mismatch (similar to vwap_reclaim_long issue)
        chop_pct = result['regime_distribution'].get('chop', 0) / result['total_trades'] * 100
        chop_adx = result['adx_by_regime'].get('chop', 0)

        if chop_pct >= 50 and chop_adx >= 25:
            print(f"\n  >>> WARNING: {chop_pct:.0f}% in CHOP but ADX {chop_adx:.1f} (>= 25)!")
            print(f"  >>> Similar to vwap_reclaim_long issue - check if this is a trend setup!")

        # Check for inverted R:R issue (like order_block_long)
        if 'order_block' in setup or 'fair_value_gap' in setup:
            print(f"\n  >>> ICT pattern - check if R:R is inverted (losses > wins)")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY - SETUPS THAT NEED REGIME FIX")
    print(f"{'='*80}")

    for setup, result in results.items():
        chop_pct = result['regime_distribution'].get('chop', 0) / result['total_trades'] * 100
        chop_adx = result['adx_by_regime'].get('chop', 0)
        chop_pnl = result['pnl_by_regime'].get('chop', 0)

        if chop_pct >= 50 and chop_adx >= 25 and chop_pnl < -500:
            print(f"\n{setup}:")
            print(f"  {chop_pct:.0f}% in CHOP, ADX {chop_adx:.1f}, P&L Rs {chop_pnl:,.2f}")
            print(f"  >>> RECOMMENDED: Remove from chop regime allowlist")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
