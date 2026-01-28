"""
Simulate trading with capital constraints.
Skip trades that would exceed the capital limit.
"""
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BACKTEST_DIRS = [
    "backtest_20251219-121728_extracted",  # 2023-H1
    "backtest_20251219-190525_extracted",  # 2023-H2
    "backtest_20251220-124532_extracted",  # 2024-H1
    "backtest_20251220-174553_extracted",  # 2024-H2
    "backtest_20251220-203904_extracted",  # 2025-H1
    "backtest_20251221-094125_extracted",  # 2025-H2
]

def parse_timestamp(ts_str):
    """Parse timestamp string to datetime."""
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None

def simulate_session(session_path, capital_limit):
    """
    Simulate a single trading session with capital constraint.
    Returns: (trades_taken, trades_skipped, pnl_taken, pnl_skipped)
    """
    events_file = session_path / "events.jsonl"
    analytics_file = session_path / "analytics.jsonl"

    if not events_file.exists() or not analytics_file.exists():
        return None

    # Collect trade info from DECISION events
    trades = {}
    with open(events_file, encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line)
                if event.get("type") == "DECISION":
                    trade_id = event.get("trade_id")
                    if trade_id:
                        notional = event.get("plan", {}).get("sizing", {}).get("notional", 0)
                        trades[trade_id] = {
                            "notional": notional,
                            "entry_time": None,
                            "exit_time": None,
                            "pnl": None,
                            "setup": event.get("plan", {}).get("strategy", "unknown")
                        }

                if event.get("type") == "TRIGGER":
                    trade_id = event.get("trade_id")
                    if trade_id and trade_id in trades:
                        trades[trade_id]["entry_time"] = parse_timestamp(event.get("ts"))
            except:
                continue

    # Get exit times and PnL from analytics
    with open(analytics_file, encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line)
                trade_id = event.get("trade_id")
                if trade_id and trade_id in trades and event.get("is_final_exit"):
                    trades[trade_id]["exit_time"] = parse_timestamp(event.get("timestamp"))
                    trades[trade_id]["pnl"] = event.get("total_trade_pnl", 0)
            except:
                continue

    # Filter to triggered trades with valid data
    valid_trades = [
        (tid, t) for tid, t in trades.items()
        if t["entry_time"] and t["exit_time"] and t["pnl"] is not None and t["notional"] > 0
    ]

    if not valid_trades:
        return None

    # Sort by entry time
    valid_trades.sort(key=lambda x: x[1]["entry_time"])

    # Simulate with capital constraint
    # Track active positions: list of (exit_time, notional)
    active_positions = []
    trades_taken = []
    trades_skipped = []

    for trade_id, trade in valid_trades:
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        notional = trade["notional"]
        pnl = trade["pnl"]

        # Remove positions that have exited before this entry
        active_positions = [(et, n) for et, n in active_positions if et > entry_time]

        # Calculate current capital in use
        current_capital = sum(n for _, n in active_positions)

        # Can we take this trade?
        if current_capital + notional <= capital_limit:
            # Take the trade
            active_positions.append((exit_time, notional))
            trades_taken.append({
                "trade_id": trade_id,
                "notional": notional,
                "pnl": pnl,
                "setup": trade["setup"]
            })
        else:
            # Skip the trade
            trades_skipped.append({
                "trade_id": trade_id,
                "notional": notional,
                "pnl": pnl,
                "setup": trade["setup"],
                "capital_needed": current_capital + notional
            })

    return {
        "date": session_path.name,
        "trades_taken": len(trades_taken),
        "trades_skipped": len(trades_skipped),
        "pnl_taken": sum(t["pnl"] for t in trades_taken),
        "pnl_skipped": sum(t["pnl"] for t in trades_skipped),
        "skipped_details": trades_skipped
    }

def run_simulation(capital_limit, label):
    """Run simulation across all sessions."""
    print(f"\n{'='*80}")
    print(f"SIMULATION: {label} CAPITAL (Rs {capital_limit:,.0f})")
    print("="*80)

    all_results = []
    total_pnl_taken = 0
    total_pnl_skipped = 0
    total_trades_taken = 0
    total_trades_skipped = 0
    skipped_winners = 0
    skipped_losers = 0
    skipped_winner_pnl = 0
    skipped_loser_pnl = 0

    for backtest_dir in BACKTEST_DIRS:
        base_path = Path(backtest_dir)
        if not base_path.exists():
            continue

        for session_path in sorted(base_path.iterdir()):
            if not session_path.is_dir():
                continue

            result = simulate_session(session_path, capital_limit)
            if result:
                all_results.append(result)
                total_pnl_taken += result["pnl_taken"]
                total_pnl_skipped += result["pnl_skipped"]
                total_trades_taken += result["trades_taken"]
                total_trades_skipped += result["trades_skipped"]

                # Analyze skipped trades
                for skip in result["skipped_details"]:
                    if skip["pnl"] > 0:
                        skipped_winners += 1
                        skipped_winner_pnl += skip["pnl"]
                    else:
                        skipped_losers += 1
                        skipped_loser_pnl += skip["pnl"]

    print(f"\nSessions analyzed: {len(all_results)}")
    print(f"\n--- TRADES ---")
    print(f"Trades TAKEN: {total_trades_taken}")
    print(f"Trades SKIPPED: {total_trades_skipped}")
    print(f"Skip Rate: {total_trades_skipped / (total_trades_taken + total_trades_skipped) * 100:.1f}%")

    print(f"\n--- PnL IMPACT ---")
    print(f"PnL from TAKEN trades: Rs {total_pnl_taken:,.2f}")
    print(f"PnL from SKIPPED trades: Rs {total_pnl_skipped:,.2f}")
    print(f"  - Skipped Winners: {skipped_winners} trades, Rs {skipped_winner_pnl:,.2f}")
    print(f"  - Skipped Losers: {skipped_losers} trades, Rs {skipped_loser_pnl:,.2f}")

    print(f"\n--- COMPARISON TO UNLIMITED ---")
    unlimited_pnl = total_pnl_taken + total_pnl_skipped
    print(f"Unlimited capital PnL: Rs {unlimited_pnl:,.2f}")
    print(f"Constrained PnL: Rs {total_pnl_taken:,.2f}")
    print(f"Difference: Rs {total_pnl_taken - unlimited_pnl:,.2f} ({(total_pnl_taken/unlimited_pnl - 1)*100:+.1f}%)")

    # Days with skipped trades
    days_with_skips = [r for r in all_results if r["trades_skipped"] > 0]
    print(f"\n--- DAYS AFFECTED ---")
    print(f"Days with skipped trades: {len(days_with_skips)} / {len(all_results)} ({len(days_with_skips)/len(all_results)*100:.1f}%)")

    if days_with_skips:
        # Show top 5 days with most skipped PnL
        days_with_skips.sort(key=lambda x: x["pnl_skipped"], reverse=True)
        print(f"\nTop 5 days with highest skipped PnL:")
        print(f"{'Date':<12} {'Taken':>8} {'Skipped':>8} {'PnL Taken':>12} {'PnL Skipped':>12}")
        print("-" * 55)
        for r in days_with_skips[:5]:
            print(f"{r['date']:<12} {r['trades_taken']:>8} {r['trades_skipped']:>8} Rs {r['pnl_taken']:>10,.0f} Rs {r['pnl_skipped']:>10,.0f}")

    return {
        "capital_limit": capital_limit,
        "trades_taken": total_trades_taken,
        "trades_skipped": total_trades_skipped,
        "pnl_taken": total_pnl_taken,
        "pnl_skipped": total_pnl_skipped,
        "skipped_winners": skipped_winners,
        "skipped_winner_pnl": skipped_winner_pnl,
        "skipped_losers": skipped_losers,
        "skipped_loser_pnl": skipped_loser_pnl,
    }

def main():
    print("=" * 80)
    print("CAPITAL CONSTRAINT SIMULATION")
    print("=" * 80)
    print("\nThis simulation checks which trades would be skipped due to")
    print("insufficient capital when multiple trades run concurrently.")

    # Run simulations
    results = {}
    for limit, label in [(300000, "3 LAKH"), (500000, "5 LAKH"), (600000, "6 LAKH"), (1000000, "10 LAKH")]:
        results[label] = run_simulation(limit, label)

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Capital':<12} {'Trades':<12} {'Skipped':<10} {'PnL':>15} {'vs Unlimited':>15}")
    print("-" * 70)

    unlimited_pnl = results["10 LAKH"]["pnl_taken"] + results["10 LAKH"]["pnl_skipped"]

    for label in ["3 LAKH", "5 LAKH", "6 LAKH", "10 LAKH"]:
        r = results[label]
        total_trades = r["trades_taken"] + r["trades_skipped"]
        diff_pct = (r["pnl_taken"] / unlimited_pnl - 1) * 100 if unlimited_pnl else 0
        print(f"Rs {label:<10} {r['trades_taken']:>5}/{total_trades:<5} {r['trades_skipped']:>8} Rs {r['pnl_taken']:>12,.0f} {diff_pct:>+13.1f}%")

    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print("="*80)
    r3 = results["3 LAKH"]
    r5 = results["5 LAKH"]
    print(f"\nWith Rs 3L capital:")
    print(f"  - You capture Rs {r3['pnl_taken']:,.0f} ({r3['pnl_taken']/unlimited_pnl*100:.1f}% of total)")
    print(f"  - You miss {r3['trades_skipped']} trades worth Rs {r3['pnl_skipped']:,.0f}")

    print(f"\nWith Rs 5L capital:")
    print(f"  - You capture Rs {r5['pnl_taken']:,.0f} ({r5['pnl_taken']/unlimited_pnl*100:.1f}% of total)")
    print(f"  - You miss {r5['trades_skipped']} trades worth Rs {r5['pnl_skipped']:,.0f}")

    print(f"\nIncrease from 3L to 5L:")
    extra_pnl = r5['pnl_taken'] - r3['pnl_taken']
    print(f"  - Extra PnL captured: Rs {extra_pnl:,.0f}")
    print(f"  - ROI on extra 2L capital: {extra_pnl/200000*100:.1f}% over 3 years")

if __name__ == "__main__":
    main()
