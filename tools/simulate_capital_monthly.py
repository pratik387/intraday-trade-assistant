"""
Simulate trading with capital constraints - Monthly and Yearly breakdown.
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
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None

def simulate_session(session_path, capital_limit):
    """Simulate a single session with capital constraint."""
    events_file = session_path / "events.jsonl"
    analytics_file = session_path / "analytics.jsonl"

    if not events_file.exists() or not analytics_file.exists():
        return None

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
                        }
                if event.get("type") == "TRIGGER":
                    trade_id = event.get("trade_id")
                    if trade_id and trade_id in trades:
                        trades[trade_id]["entry_time"] = parse_timestamp(event.get("ts"))
            except:
                continue

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

    valid_trades = [
        (tid, t) for tid, t in trades.items()
        if t["entry_time"] and t["exit_time"] and t["pnl"] is not None and t["notional"] > 0
    ]

    if not valid_trades:
        return None

    valid_trades.sort(key=lambda x: x[1]["entry_time"])

    active_positions = []
    pnl_taken = 0
    pnl_skipped = 0
    trades_taken = 0
    trades_skipped = 0

    for trade_id, trade in valid_trades:
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        notional = trade["notional"]
        pnl = trade["pnl"]

        active_positions = [(et, n) for et, n in active_positions if et > entry_time]
        current_capital = sum(n for _, n in active_positions)

        if current_capital + notional <= capital_limit:
            active_positions.append((exit_time, notional))
            pnl_taken += pnl
            trades_taken += 1
        else:
            pnl_skipped += pnl
            trades_skipped += 1

    return {
        "date": session_path.name,
        "trades_taken": trades_taken,
        "trades_skipped": trades_skipped,
        "pnl_taken": pnl_taken,
        "pnl_skipped": pnl_skipped,
    }

def run_simulation(capital_limit):
    """Run simulation and return monthly/yearly breakdown."""
    monthly = defaultdict(lambda: {"trades_taken": 0, "trades_skipped": 0, "pnl_taken": 0, "pnl_skipped": 0, "sessions": 0})

    for backtest_dir in BACKTEST_DIRS:
        base_path = Path(backtest_dir)
        if not base_path.exists():
            continue

        for session_path in sorted(base_path.iterdir()):
            if not session_path.is_dir():
                continue

            result = simulate_session(session_path, capital_limit)
            if result:
                month = result["date"][:7]  # YYYY-MM
                monthly[month]["trades_taken"] += result["trades_taken"]
                monthly[month]["trades_skipped"] += result["trades_skipped"]
                monthly[month]["pnl_taken"] += result["pnl_taken"]
                monthly[month]["pnl_skipped"] += result["pnl_skipped"]
                monthly[month]["sessions"] += 1

    return monthly

def main():
    print("=" * 100)
    print("MONTHLY & YEARLY BREAKDOWN - CAPITAL CONSTRAINT SIMULATION")
    print("=" * 100)

    # Run for both capital limits
    monthly_3L = run_simulation(300000)
    monthly_5L = run_simulation(500000)

    # Monthly breakdown
    print(f"\n{'='*100}")
    print("MONTH-BY-MONTH COMPARISON")
    print("="*100)
    print(f"{'Month':<10} {'Sessions':>8} | {'3L Trades':>10} {'3L PnL':>12} | {'5L Trades':>10} {'5L PnL':>12} | {'Diff':>10}")
    print("-" * 100)

    all_months = sorted(set(monthly_3L.keys()) | set(monthly_5L.keys()))

    yearly_3L = defaultdict(lambda: {"pnl": 0, "trades": 0, "sessions": 0})
    yearly_5L = defaultdict(lambda: {"pnl": 0, "trades": 0, "sessions": 0})

    for month in all_months:
        m3 = monthly_3L.get(month, {"trades_taken": 0, "pnl_taken": 0, "sessions": 0})
        m5 = monthly_5L.get(month, {"trades_taken": 0, "pnl_taken": 0, "sessions": 0})

        diff = m5["pnl_taken"] - m3["pnl_taken"]
        sessions = max(m3["sessions"], m5["sessions"])

        print(f"{month:<10} {sessions:>8} | {m3['trades_taken']:>10} {m3['pnl_taken']:>12,.0f} | {m5['trades_taken']:>10} {m5['pnl_taken']:>12,.0f} | {diff:>+10,.0f}")

        year = month[:4]
        yearly_3L[year]["pnl"] += m3["pnl_taken"]
        yearly_3L[year]["trades"] += m3["trades_taken"]
        yearly_3L[year]["sessions"] += m3["sessions"]
        yearly_5L[year]["pnl"] += m5["pnl_taken"]
        yearly_5L[year]["trades"] += m5["trades_taken"]
        yearly_5L[year]["sessions"] += m5["sessions"]

    # Yearly breakdown
    print(f"\n{'='*100}")
    print("YEARLY BREAKDOWN")
    print("="*100)
    print(f"{'Year':<6} {'Sessions':>8} | {'3L Trades':>10} {'3L PnL':>15} {'3L ROI%':>10} | {'5L Trades':>10} {'5L PnL':>15} {'5L ROI%':>10} | {'Diff':>12}")
    print("-" * 120)

    total_3L = {"pnl": 0, "trades": 0, "sessions": 0}
    total_5L = {"pnl": 0, "trades": 0, "sessions": 0}

    for year in sorted(yearly_3L.keys()):
        y3 = yearly_3L[year]
        y5 = yearly_5L[year]

        roi_3L = (y3["pnl"] / 300000) * 100
        roi_5L = (y5["pnl"] / 500000) * 100
        diff = y5["pnl"] - y3["pnl"]

        print(f"{year:<6} {y3['sessions']:>8} | {y3['trades']:>10} {y3['pnl']:>15,.0f} {roi_3L:>9.1f}% | {y5['trades']:>10} {y5['pnl']:>15,.0f} {roi_5L:>9.1f}% | {diff:>+12,.0f}")

        total_3L["pnl"] += y3["pnl"]
        total_3L["trades"] += y3["trades"]
        total_3L["sessions"] += y3["sessions"]
        total_5L["pnl"] += y5["pnl"]
        total_5L["trades"] += y5["trades"]
        total_5L["sessions"] += y5["sessions"]

    print("-" * 120)
    roi_3L_total = (total_3L["pnl"] / 300000 / 3) * 100  # annualized
    roi_5L_total = (total_5L["pnl"] / 500000 / 3) * 100  # annualized
    diff_total = total_5L["pnl"] - total_3L["pnl"]
    print(f"{'TOTAL':<6} {total_3L['sessions']:>8} | {total_3L['trades']:>10} {total_3L['pnl']:>15,.0f} {roi_3L_total:>8.1f}%/y | {total_5L['trades']:>10} {total_5L['pnl']:>15,.0f} {roi_5L_total:>8.1f}%/y | {diff_total:>+12,.0f}")

    # Summary stats
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print("="*100)

    print("\n3 LAKH CAPITAL:")
    print(f"  Total PnL: Rs {total_3L['pnl']:,.0f}")
    print(f"  Total Trades: {total_3L['trades']}")
    print(f"  Avg Monthly PnL: Rs {total_3L['pnl']/len(all_months):,.0f}")
    print(f"  Avg Yearly PnL: Rs {total_3L['pnl']/3:,.0f}")
    print(f"  Annual ROI: {roi_3L_total:.1f}%")

    print("\n5 LAKH CAPITAL:")
    print(f"  Total PnL: Rs {total_5L['pnl']:,.0f}")
    print(f"  Total Trades: {total_5L['trades']}")
    print(f"  Avg Monthly PnL: Rs {total_5L['pnl']/len(all_months):,.0f}")
    print(f"  Avg Yearly PnL: Rs {total_5L['pnl']/3:,.0f}")
    print(f"  Annual ROI: {roi_5L_total:.1f}%")

    # Best and worst months
    print(f"\n{'='*100}")
    print("BEST & WORST MONTHS (5L Capital)")
    print("="*100)

    sorted_months = sorted(monthly_5L.items(), key=lambda x: x[1]["pnl_taken"], reverse=True)

    print("\nTop 5 Best Months:")
    for month, data in sorted_months[:5]:
        print(f"  {month}: Rs {data['pnl_taken']:>10,.0f} ({data['trades_taken']} trades)")

    print("\nTop 5 Worst Months:")
    for month, data in sorted_months[-5:]:
        print(f"  {month}: Rs {data['pnl_taken']:>10,.0f} ({data['trades_taken']} trades)")

    # Winning vs losing months
    winning_months = [m for m, d in monthly_5L.items() if d["pnl_taken"] > 0]
    losing_months = [m for m, d in monthly_5L.items() if d["pnl_taken"] <= 0]

    print(f"\n{'='*100}")
    print("MONTHLY WIN RATE")
    print("="*100)
    print(f"Winning Months: {len(winning_months)} / {len(all_months)} ({len(winning_months)/len(all_months)*100:.1f}%)")
    print(f"Losing Months: {len(losing_months)} / {len(all_months)} ({len(losing_months)/len(all_months)*100:.1f}%)")

if __name__ == "__main__":
    main()
