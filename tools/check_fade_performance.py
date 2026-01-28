"""Check failure_fade performance in current vs backtesting1 branch."""

import json
from pathlib import Path

# Current backtest
backtest_dir = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251117-021749_extracted\20251117-021749_full\20251117-021749")

fade_trades = []

for session_dir in sorted(backtest_dir.iterdir()):
    if not session_dir.is_dir():
        continue

    analytics_file = session_dir / 'analytics.jsonl'
    if not analytics_file.exists():
        continue

    with open(analytics_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                trade = json.loads(line.strip())
                if 'fade' in trade.get('setup_type', '').lower():
                    fade_trades.append(trade)
            except:
                continue

print("="*80)
print("FAILURE FADE PERFORMANCE - CURRENT BACKTEST")
print("="*80)

if len(fade_trades) == 0:
    print("\nNo failure_fade trades found!")
else:
    # Group by setup_type
    by_setup = {}
    for trade in fade_trades:
        setup = trade.get('setup_type')
        if setup not in by_setup:
            by_setup[setup] = []
        by_setup[setup].append(trade)

    for setup, trades in by_setup.items():
        total = len(trades)
        wins = len([t for t in trades if t.get('pnl', 0) > 0])
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        win_rate = wins / total * 100

        print(f"\n{setup}:")
        print(f"  Trades: {total}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total P&L: Rs {total_pnl:,.2f}")

print("\n" + "="*80)
