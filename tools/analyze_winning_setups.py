"""
Analyze winning setups to understand:
1. What's working well (high win rate + profitable)
2. Why they work (regime, ADX, conditions)
3. How to get MORE of these winning trades
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def analyze_winning_setups(backtest_dir: str):
    """Analyze all setups to find winners and understand why they work."""

    base_path = Path(backtest_dir)

    # Load all trades
    all_trades = []
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
                    trade['session'] = session_dir.name
                    all_trades.append(trade)
                except:
                    continue

    df = pd.DataFrame(all_trades)

    print("="*80)
    print("WINNING SETUPS ANALYSIS")
    print("="*80)
    print(f"\nTotal trades analyzed: {len(df)}")
    print(f"Total P&L: Rs {df['pnl'].sum():,.2f}")

    # Analyze by setup type
    setup_stats = df.groupby('setup_type').agg({
        'pnl': ['count', 'sum', 'mean'],
        'exit_reason': lambda x: (x == 'target').sum()  # Count target hits
    }).round(2)

    setup_stats.columns = ['trades', 'total_pnl', 'avg_pnl', 'target_hits']
    setup_stats['win_rate'] = (df.groupby('setup_type')['pnl'].apply(lambda x: (x > 0).sum() / len(x) * 100)).round(1)
    setup_stats = setup_stats.sort_values('total_pnl', ascending=False)

    print("\n" + "="*80)
    print("ALL SETUPS RANKED BY PROFITABILITY")
    print("="*80)

    for setup, row in setup_stats.iterrows():
        status = "✅ WINNER" if row['total_pnl'] > 500 else ("⚠️ SMALL PROFIT" if row['total_pnl'] > 0 else "❌ LOSER")
        print(f"\n{setup} {status}")
        print(f"  Trades: {int(row['trades'])}")
        print(f"  Total P&L: Rs {row['total_pnl']:,.2f}")
        print(f"  Avg P&L: Rs {row['avg_pnl']:,.2f}")
        print(f"  Win Rate: {row['win_rate']:.1f}%")
        print(f"  Target Hits: {int(row['target_hits'])}/{int(row['trades'])} ({int(row['target_hits'])/int(row['trades'])*100:.1f}%)")

    # Deep dive into top 3 winners
    print("\n\n" + "="*80)
    print("DEEP DIVE: TOP 3 WINNING SETUPS")
    print("="*80)

    top_winners = setup_stats.head(3).index.tolist()

    for setup in top_winners:
        setup_trades = df[df['setup_type'] == setup]

        print(f"\n{'='*80}")
        print(f"{setup.upper()}")
        print(f"{'='*80}")

        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Trades: {len(setup_trades)}")
        print(f"  Total P&L: Rs {setup_trades['pnl'].sum():,.2f}")
        print(f"  Avg P&L: Rs {setup_trades['pnl'].mean():,.2f}")
        print(f"  Win Rate: {(setup_trades['pnl'] > 0).sum() / len(setup_trades) * 100:.1f}%")

        # Regime distribution
        print(f"\n  BY REGIME:")
        regime_stats = setup_trades.groupby('regime').agg({
            'pnl': ['count', 'sum', lambda x: (x > 0).sum() / len(x) * 100]
        }).round(2)
        regime_stats.columns = ['trades', 'pnl', 'win_rate']

        for regime, row in regime_stats.iterrows():
            print(f"    {regime:<15} {int(row['trades']):>3} trades  WR: {row['win_rate']:>5.1f}%  P&L: Rs {row['pnl']:>10,.2f}")

        # Exit reasons
        print(f"\n  EXIT REASONS:")
        exit_stats = setup_trades['exit_reason'].value_counts()
        for exit_reason, count in exit_stats.items():
            pct = count / len(setup_trades) * 100
            avg_pnl = setup_trades[setup_trades['exit_reason'] == exit_reason]['pnl'].mean()
            print(f"    {exit_reason:<20} {count:>3} ({pct:>5.1f}%)  Avg P&L: Rs {avg_pnl:>8,.2f}")

    # Decision to trade conversion for winners
    print("\n\n" + "="*80)
    print("DECISION → TRADE CONVERSION FOR WINNERS")
    print("="*80)

    for setup in top_winners:
        # Count decisions
        decisions = 0
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
                            event.get('decision', {}).get('setup_type') == setup):
                            decisions += 1
                    except:
                        continue

        setup_trades = df[df['setup_type'] == setup]
        conversion = len(setup_trades) / decisions * 100 if decisions > 0 else 0

        print(f"\n{setup}:")
        print(f"  DECISION events: {decisions}")
        print(f"  Completed trades: {len(setup_trades)}")
        print(f"  Conversion rate: {conversion:.1f}%")

        if conversion > 50:
            print(f"  ✅ GOOD CONVERSION - Most decisions trigger")
        elif conversion > 30:
            print(f"  ⚠️ OK CONVERSION - Some decisions don't trigger")
        else:
            print(f"  ❌ POOR CONVERSION - Many decisions wasted")

    # Find opportunity: Setups with good performance but LOW trade count
    print("\n\n" + "="*80)
    print("OPPORTUNITY ANALYSIS: Good Setups with Low Volume")
    print("="*80)
    print("\nSetups that work well but we're not trading enough:")

    for setup, row in setup_stats.iterrows():
        # Good performance = positive P&L + >45% win rate
        # Low volume = < 30 trades
        if row['total_pnl'] > 0 and row['win_rate'] > 45 and row['trades'] < 30:
            print(f"\n{setup}:")
            print(f"  Trades: {int(row['trades'])} (LOW!)")
            print(f"  Total P&L: Rs {row['total_pnl']:,.2f}")
            print(f"  Win Rate: {row['win_rate']:.1f}%")
            print(f"  >>> OPPORTUNITY: Need to detect MORE of these!")

    # What conditions make winners win?
    print("\n\n" + "="*80)
    print("SUCCESS FACTORS: What Makes Winners Win?")
    print("="*80)

    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] <= 0]

    print(f"\nWinning trades ({len(winning_trades)}):")
    print(f"  Most common regimes: {winning_trades['regime'].value_counts().head(3).to_dict()}")
    print(f"  Most common exit: {winning_trades['exit_reason'].value_counts().head(1).index[0]}")

    print(f"\nLosing trades ({len(losing_trades)}):")
    print(f"  Most common regimes: {losing_trades['regime'].value_counts().head(3).to_dict()}")
    print(f"  Most common exit: {losing_trades['exit_reason'].value_counts().head(1).index[0]}")

    print("\n" + "="*80)

if __name__ == "__main__":
    backtest_dir = r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251117-021749_extracted\20251117-021749_full\20251117-021749"
    analyze_winning_setups(backtest_dir)
