"""
Analyze failing setups to understand:
1. What would make them profitable
2. What percentage are close to profitable
3. How to improve entry/exit to turn losers into winners
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_setup_improvement_potential(backtest_dir: str, setup_name: str):
    """Deep analysis of a failing setup to find improvement opportunities."""

    base_path = Path(backtest_dir)

    # Load all trades for this setup
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
                        trade['session'] = session_dir.name
                        trades.append(trade)
                except:
                    continue

    if len(trades) == 0:
        print(f"No trades found for {setup_name}")
        return

    df = pd.DataFrame(trades)

    print("="*80)
    print(f"IMPROVEMENT ANALYSIS: {setup_name.upper()}")
    print("="*80)

    # Overall stats
    total_pnl = df['pnl'].sum()
    win_rate = (df['pnl'] > 0).sum() / len(df) * 100
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0
    avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if len(df[df['pnl'] <= 0]) > 0 else 0

    print(f"\nCURRENT PERFORMANCE:")
    print(f"  Total Trades: {len(df)}")
    print(f"  Total P&L: Rs {total_pnl:,.2f}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg Win: Rs {avg_win:.2f}")
    print(f"  Avg Loss: Rs {avg_loss:.2f}")
    print(f"  R:R Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  R:R Ratio: N/A")

    # Exit reason analysis
    print(f"\nEXIT REASONS:")
    exit_stats = df.groupby('exit_reason').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    exit_stats.columns = ['count', 'total_pnl', 'avg_pnl']

    for exit_reason, row in exit_stats.iterrows():
        pct = row['count'] / len(df) * 100
        print(f"  {exit_reason:<20} {int(row['count']):>3} ({pct:>5.1f}%)  Avg: Rs {row['avg_pnl']:>8,.2f}  Total: Rs {row['total_pnl']:>10,.2f}")

    # CRITICAL: How many were CLOSE to profitable?
    print(f"\n{'='*80}")
    print("NEAR-MISS ANALYSIS")
    print(f"{'='*80}")

    # Trades that hit stop loss
    stop_loss_trades = df[df['exit_reason'].str.contains('stop_loss', case=False, na=False)]

    if len(stop_loss_trades) > 0:
        print(f"\nStop Loss trades: {len(stop_loss_trades)}")
        print(f"  Total loss: Rs {stop_loss_trades['pnl'].sum():,.2f}")

        # Check how many would have been profitable with wider stops
        # This requires MFE (Max Favorable Excursion) data which might not be in analytics
        print(f"\n  >>> OPPORTUNITY: If stops were wider, some might have recovered")

    # Trades that timed out
    timeout_trades = df[df['exit_reason'].str.contains('timeout|eod', case=False, na=False)]

    if len(timeout_trades) > 0:
        print(f"\nTimeout/EOD trades: {len(timeout_trades)}")
        print(f"  Total P&L: Rs {timeout_trades['pnl'].sum():,.2f}")
        print(f"  Avg P&L: Rs {timeout_trades['pnl'].mean():,.2f}")

        profitable_timeouts = timeout_trades[timeout_trades['pnl'] > 0]
        losing_timeouts = timeout_trades[timeout_trades['pnl'] < 0]

        print(f"\n  Profitable timeouts: {len(profitable_timeouts)} (could have hit targets with more time)")
        print(f"  Losing timeouts: {len(losing_timeouts)} (correct to exit)")

        if len(losing_timeouts) > 0:
            # How bad were the losses?
            small_losses = losing_timeouts[losing_timeouts['pnl'] > -100]
            print(f"    Small losses (< Rs 100): {len(small_losses)} (could break even with better entry)")

    # Target analysis
    target_trades = df[df['exit_reason'].str.contains('target', case=False, na=False)]

    if len(target_trades) > 0:
        print(f"\nTarget Hit trades: {len(target_trades)}")
        print(f"  Total profit: Rs {target_trades['pnl'].sum():,.2f}")
        print(f"  Avg profit: Rs {target_trades['pnl'].mean():,.2f}")
        print(f"  Success rate: {len(target_trades)/len(df)*100:.1f}%")

    # Regime performance
    print(f"\n{'='*80}")
    print("REGIME PERFORMANCE")
    print(f"{'='*80}")

    regime_stats = df.groupby('regime').agg({
        'pnl': ['count', 'sum', lambda x: (x > 0).sum() / len(x) * 100]
    }).round(2)
    regime_stats.columns = ['trades', 'total_pnl', 'win_rate']

    for regime, row in regime_stats.iterrows():
        status = "✅ WORKS" if row['total_pnl'] > 0 else "❌ FAILS"
        print(f"\n  {regime:<15} {status}")
        print(f"    Trades: {int(row['trades'])}")
        print(f"    Total P&L: Rs {row['total_pnl']:,.2f}")
        print(f"    Win Rate: {row['win_rate']:.1f}%")

        if row['total_pnl'] > 0:
            print(f"    >>> KEEP in {regime} regime")
        else:
            print(f"    >>> REMOVE from {regime} regime")

    # IMPROVEMENT RECOMMENDATIONS
    print(f"\n{'='*80}")
    print("IMPROVEMENT RECOMMENDATIONS")
    print(f"{'='*80}")

    recommendations = []

    # 1. Regime filtering
    for regime, row in regime_stats.iterrows():
        if row['total_pnl'] < -500:
            impact = abs(row['total_pnl'])
            recommendations.append({
                'priority': 1,
                'action': f'Remove from {regime} regime',
                'impact': f'+Rs {impact:,.2f}',
                'reason': f'Losing Rs {row["total_pnl"]:,.2f} in {regime}'
            })

    # 2. R:R improvement
    if avg_loss != 0:
        current_rr = abs(avg_win/avg_loss)
        if current_rr < 1.5:
            # Need to improve R:R
            target_rr = 1.5
            # Option 1: Tighter stops
            new_avg_loss = avg_win / target_rr
            stop_improvement = abs(avg_loss) - abs(new_avg_loss)

            recommendations.append({
                'priority': 2,
                'action': f'Tighten stops by Rs {stop_improvement:.2f} (from Rs {abs(avg_loss):.2f} to Rs {abs(new_avg_loss):.2f})',
                'impact': f'Improve R:R from {current_rr:.2f} to {target_rr:.2f}',
                'reason': f'Current R:R {current_rr:.2f} too low'
            })

            # Option 2: Wider targets
            new_avg_win = abs(avg_loss) * target_rr
            target_improvement = new_avg_win - avg_win

            recommendations.append({
                'priority': 2,
                'action': f'Widen targets by Rs {target_improvement:.2f} (from Rs {avg_win:.2f} to Rs {new_avg_win:.2f})',
                'impact': f'Improve R:R from {current_rr:.2f} to {target_rr:.2f}',
                'reason': f'Current R:R {current_rr:.2f} too low'
            })

    # 3. Win rate improvement
    target_win_rate = 45  # Minimum viable
    if win_rate < target_win_rate:
        deficit = target_win_rate - win_rate
        trades_needed = int(len(df) * deficit / 100)

        recommendations.append({
            'priority': 3,
            'action': f'Improve win rate by {deficit:.1f}% (need {trades_needed} more winners)',
            'impact': 'Add quality filters',
            'reason': f'Win rate {win_rate:.1f}% below viable {target_win_rate}%'
        })

    # 4. Stop loss analysis
    if len(stop_loss_trades) > len(df) * 0.5:  # More than 50% hit stop
        pct = len(stop_loss_trades) / len(df) * 100
        recommendations.append({
            'priority': 3,
            'action': f'Reduce stop loss hits (currently {pct:.1f}%)',
            'impact': 'Better entry timing or wider stops',
            'reason': f'{len(stop_loss_trades)} out of {len(df)} trades hit stop'
        })

    # Print recommendations
    if len(recommendations) > 0:
        print("\nTop recommendations to make this setup profitable:\n")
        for i, rec in enumerate(sorted(recommendations, key=lambda x: x['priority']), 1):
            print(f"{i}. [{rec['priority']}] {rec['action']}")
            print(f"   Impact: {rec['impact']}")
            print(f"   Reason: {rec['reason']}")
            print()

    # SIMULATION: What if we fix the issues?
    print(f"{'='*80}")
    print("SIMULATION: Potential After Fixes")
    print(f"{'='*80}")

    # Scenario 1: Remove from worst regime
    worst_regime = regime_stats['total_pnl'].idxmin()
    worst_regime_pnl = regime_stats.loc[worst_regime, 'total_pnl']

    if worst_regime_pnl < 0:
        new_pnl = total_pnl - worst_regime_pnl
        print(f"\nScenario 1: Remove from {worst_regime} regime")
        print(f"  Current P&L: Rs {total_pnl:,.2f}")
        print(f"  After removal: Rs {new_pnl:,.2f}")
        print(f"  Improvement: Rs {-worst_regime_pnl:,.2f}")

    # Scenario 2: Improve R:R to 1.5
    if avg_loss != 0 and abs(avg_win/avg_loss) < 1.5:
        # Assume we tighten stops to achieve R:R 1.5
        target_rr = 1.5
        new_avg_loss = avg_win / target_rr

        # Winners stay same, losers reduced
        winners_pnl = df[df['pnl'] > 0]['pnl'].sum()
        losers_count = len(df[df['pnl'] <= 0])
        new_losers_pnl = losers_count * new_avg_loss

        new_pnl_rr = winners_pnl + new_losers_pnl

        print(f"\nScenario 2: Improve R:R to 1.5 (tighter stops)")
        print(f"  Current P&L: Rs {total_pnl:,.2f}")
        print(f"  After R:R fix: Rs {new_pnl_rr:,.2f}")
        print(f"  Improvement: Rs {new_pnl_rr - total_pnl:,.2f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    backtest_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251117-021749_extracted\20251117-021749_full\20251117-021749"

    # Analyze top failing setups
    failing_setups = [
        'vwap_reclaim_long',
        'order_block_long',
        'fair_value_gap_long',
        'flag_continuation_long',
    ]

    for setup in failing_setups:
        analyze_setup_improvement_potential(backtest_dir, setup)
        print("\n\n")
