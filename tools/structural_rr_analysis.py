"""
Analyze structural_rr impact on trade outcomes
"""
import json
from pathlib import Path
from collections import defaultdict
import statistics

BACKTEST_DIR = Path("backtest_20251205-102858_extracted/20251205-102858_full/20251205-102858")

def load_all_data():
    """Load all events and analytics data"""
    all_decisions = []
    all_exits = []

    sessions = [d for d in BACKTEST_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()]

    for session_dir in sessions:
        events_file = session_dir / "events.jsonl"
        analytics_file = session_dir / "analytics.jsonl"

        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('type') == 'DECISION':
                            all_decisions.append(event)
                    except:
                        pass

        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('stage') == 'EXIT' and event.get('is_final_exit', False):
                            all_exits.append(event)
                    except:
                        pass

    return all_decisions, all_exits

def analyze_structural_rr(decisions, exits):
    """Analyze impact of structural_rr on trade outcomes"""
    print("="*80)
    print("STRUCTURAL RR ANALYSIS - Can we filter low RR trades at entry?")
    print("="*80)

    decision_lookup = {d.get('trade_id'): d for d in decisions if d.get('trade_id')}

    all_trades = []
    for e in exits:
        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})

        structural_rr = d.get('plan', {}).get('quality', {}).get('structural_rr') if d else None
        if structural_rr is None:
            continue

        all_trades.append({
            'trade_id': trade_id,
            'setup': e.get('setup_type', ''),
            'pnl': e.get('total_trade_pnl', e.get('pnl', 0)),
            'structural_rr': structural_rr,
            'exit_reason': e.get('reason', ''),
            'regime': e.get('regime', '')
        })

    print(f"\nTotal trades with structural_rr data: {len(all_trades)}")
    current_pnl = sum(t['pnl'] for t in all_trades)
    print(f"Total PnL: Rs {current_pnl:.2f}")

    # Analyze by structural_rr buckets
    print("\n" + "-"*60)
    print("Performance by Structural RR bucket:")
    print("-"*60)

    buckets = [
        ("< 1.5", lambda rr: rr < 1.5),
        ("1.5-2.0", lambda rr: 1.5 <= rr < 2.0),
        ("2.0-2.5", lambda rr: 2.0 <= rr < 2.5),
        ("2.5-3.0", lambda rr: 2.5 <= rr < 3.0),
        ("3.0+", lambda rr: rr >= 3.0),
    ]

    for bucket_name, bucket_fn in buckets:
        bucket_trades = [t for t in all_trades if bucket_fn(t['structural_rr'])]
        if not bucket_trades:
            continue

        wins = len([t for t in bucket_trades if t['pnl'] > 0])
        losses = len([t for t in bucket_trades if t['pnl'] <= 0])
        total = len(bucket_trades)
        wr = wins / total * 100 if total > 0 else 0
        total_pnl = sum(t['pnl'] for t in bucket_trades)
        avg_pnl = total_pnl / total if total > 0 else 0

        print(f"  {bucket_name:10} W={wins:3}, L={losses:3}, WR={wr:5.1f}%, PnL={total_pnl:>10.0f}, Avg={avg_pnl:>8.0f}")

    # Specific analysis: What if we blocked structural_rr < 1.5?
    print("\n" + "-"*60)
    print("FILTER SIMULATION: Block trades with structural_rr < 1.5")
    print("-"*60)

    low_rr_trades = [t for t in all_trades if t['structural_rr'] < 1.5]
    remaining_trades = [t for t in all_trades if t['structural_rr'] >= 1.5]

    print(f"\nTrades with structural_rr < 1.5: {len(low_rr_trades)}")
    print(f"PnL from these trades: Rs {sum(t['pnl'] for t in low_rr_trades):.0f}")
    print(f"Remaining trades: {len(remaining_trades)}")
    print(f"Remaining PnL: Rs {sum(t['pnl'] for t in remaining_trades):.0f}")
    print(f"Improvement: Rs {-sum(t['pnl'] for t in low_rr_trades):.0f}")

    # By setup
    print("\nLow RR (< 1.5) trades by setup:")
    by_setup = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for t in low_rr_trades:
        by_setup[t['setup']]['count'] += 1
        by_setup[t['setup']]['pnl'] += t['pnl']

    for setup, data in sorted(by_setup.items(), key=lambda x: x[1]['pnl']):
        print(f"  {setup}: {data['count']} trades, Rs {data['pnl']:.0f}")

    # Analyze hard_sl relationship with structural_rr
    print("\n" + "-"*60)
    print("Hard SL vs Structural RR relationship:")
    print("-"*60)

    hard_sl = [t for t in all_trades if t['exit_reason'] == 'hard_sl']
    non_hard_sl = [t for t in all_trades if t['exit_reason'] != 'hard_sl']

    hard_sl_rr = [t['structural_rr'] for t in hard_sl]
    non_hard_sl_rr = [t['structural_rr'] for t in non_hard_sl]

    if hard_sl_rr and non_hard_sl_rr:
        print(f"Hard SL trades ({len(hard_sl_rr)}): avg structural_rr = {statistics.mean(hard_sl_rr):.2f}")
        print(f"Non-hard SL trades ({len(non_hard_sl_rr)}): avg structural_rr = {statistics.mean(non_hard_sl_rr):.2f}")

        # Count hard_sl by RR bucket
        print("\nHard SL distribution by structural_rr:")
        for bucket_name, bucket_fn in buckets:
            hard_sl_in_bucket = len([t for t in hard_sl if bucket_fn(t['structural_rr'])])
            total_in_bucket = len([t for t in all_trades if bucket_fn(t['structural_rr'])])
            if total_in_bucket > 0:
                pct = hard_sl_in_bucket / total_in_bucket * 100
                print(f"  {bucket_name:10} {hard_sl_in_bucket:3}/{total_in_bucket:3} = {pct:5.1f}% hard_sl rate")

    # Additional analysis: What about structural_rr < 1.7?
    print("\n" + "-"*60)
    print("FILTER SIMULATION: Block trades with structural_rr < 1.7")
    print("-"*60)

    low_rr_trades_17 = [t for t in all_trades if t['structural_rr'] < 1.7]
    print(f"Trades blocked: {len(low_rr_trades_17)}")
    print(f"PnL blocked: Rs {sum(t['pnl'] for t in low_rr_trades_17):.0f}")
    print(f"Improvement: Rs {-sum(t['pnl'] for t in low_rr_trades_17):.0f}")

def main():
    decisions, exits = load_all_data()
    analyze_structural_rr(decisions, exits)

if __name__ == "__main__":
    main()
