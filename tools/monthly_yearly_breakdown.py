"""
Extract month-by-month and yearly breakdown from backtest results.
No capital constraints - actual backtest performance.
"""
import json
from pathlib import Path
from collections import defaultdict

BACKTEST_DIRS = [
    "backtest_20251219-121728_extracted",  # 2023-H1
    "backtest_20251219-190525_extracted",  # 2023-H2
    "backtest_20251220-124532_extracted",  # 2024-H1
    "backtest_20251220-174553_extracted",  # 2024-H2
    "backtest_20251220-203904_extracted",  # 2025-H1
    "backtest_20251221-094125_extracted",  # 2025-H2
]

def extract_session_data(session_path):
    """Extract trades and PnL from a session."""
    analytics_file = session_path / "analytics.jsonl"

    if not analytics_file.exists():
        return None

    trades = {}
    with open(analytics_file, encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line)
                trade_id = event.get("trade_id")
                if trade_id and event.get("is_final_exit"):
                    trades[trade_id] = {
                        "pnl": event.get("total_trade_pnl", 0),
                        "setup": event.get("setup_type", "unknown"),
                        "regime": event.get("regime", "unknown"),
                    }
            except:
                continue

    if not trades:
        return None

    total_pnl = sum(t["pnl"] for t in trades.values())
    winners = sum(1 for t in trades.values() if t["pnl"] > 0)
    losers = sum(1 for t in trades.values() if t["pnl"] <= 0)

    return {
        "date": session_path.name,
        "trades": len(trades),
        "pnl": total_pnl,
        "winners": winners,
        "losers": losers,
    }

def main():
    print("=" * 90)
    print("BACKTEST PERFORMANCE - MONTHLY & YEARLY BREAKDOWN")
    print("=" * 90)

    # Collect all session data
    monthly = defaultdict(lambda: {"sessions": 0, "trades": 0, "pnl": 0, "winners": 0, "losers": 0})
    daily_data = []

    for backtest_dir in BACKTEST_DIRS:
        base_path = Path(backtest_dir)
        if not base_path.exists():
            continue

        for session_path in sorted(base_path.iterdir()):
            if not session_path.is_dir():
                continue

            result = extract_session_data(session_path)
            if result:
                month = result["date"][:7]  # YYYY-MM
                monthly[month]["sessions"] += 1
                monthly[month]["trades"] += result["trades"]
                monthly[month]["pnl"] += result["pnl"]
                monthly[month]["winners"] += result["winners"]
                monthly[month]["losers"] += result["losers"]
                daily_data.append(result)

    # Monthly breakdown
    print(f"\n{'='*90}")
    print("MONTH-BY-MONTH BREAKDOWN")
    print("="*90)
    print(f"{'Month':<10} {'Sessions':>8} {'Trades':>8} {'Winners':>8} {'Losers':>8} {'Win%':>8} {'PnL':>15} {'Avg/Trade':>12}")
    print("-" * 90)

    yearly = defaultdict(lambda: {"sessions": 0, "trades": 0, "pnl": 0, "winners": 0, "losers": 0})

    for month in sorted(monthly.keys()):
        m = monthly[month]
        win_rate = (m["winners"] / m["trades"] * 100) if m["trades"] > 0 else 0
        avg_pnl = m["pnl"] / m["trades"] if m["trades"] > 0 else 0

        print(f"{month:<10} {m['sessions']:>8} {m['trades']:>8} {m['winners']:>8} {m['losers']:>8} {win_rate:>7.1f}% {m['pnl']:>15,.2f} {avg_pnl:>12,.2f}")

        year = month[:4]
        yearly[year]["sessions"] += m["sessions"]
        yearly[year]["trades"] += m["trades"]
        yearly[year]["pnl"] += m["pnl"]
        yearly[year]["winners"] += m["winners"]
        yearly[year]["losers"] += m["losers"]

    # Yearly breakdown
    print(f"\n{'='*90}")
    print("YEARLY BREAKDOWN")
    print("="*90)
    print(f"{'Year':<6} {'Sessions':>10} {'Trades':>10} {'Winners':>10} {'Losers':>10} {'Win%':>8} {'Total PnL':>18} {'Avg/Trade':>12} {'Avg/Session':>14}")
    print("-" * 110)

    total = {"sessions": 0, "trades": 0, "pnl": 0, "winners": 0, "losers": 0}

    for year in sorted(yearly.keys()):
        y = yearly[year]
        win_rate = (y["winners"] / y["trades"] * 100) if y["trades"] > 0 else 0
        avg_pnl = y["pnl"] / y["trades"] if y["trades"] > 0 else 0
        avg_session = y["pnl"] / y["sessions"] if y["sessions"] > 0 else 0

        print(f"{year:<6} {y['sessions']:>10} {y['trades']:>10} {y['winners']:>10} {y['losers']:>10} {win_rate:>7.1f}% {y['pnl']:>18,.2f} {avg_pnl:>12,.2f} {avg_session:>14,.2f}")

        total["sessions"] += y["sessions"]
        total["trades"] += y["trades"]
        total["pnl"] += y["pnl"]
        total["winners"] += y["winners"]
        total["losers"] += y["losers"]

    print("-" * 110)
    win_rate = (total["winners"] / total["trades"] * 100) if total["trades"] > 0 else 0
    avg_pnl = total["pnl"] / total["trades"] if total["trades"] > 0 else 0
    avg_session = total["pnl"] / total["sessions"] if total["sessions"] > 0 else 0
    print(f"{'TOTAL':<6} {total['sessions']:>10} {total['trades']:>10} {total['winners']:>10} {total['losers']:>10} {win_rate:>7.1f}% {total['pnl']:>18,.2f} {avg_pnl:>12,.2f} {avg_session:>14,.2f}")

    # Best and worst months
    print(f"\n{'='*90}")
    print("TOP 5 BEST MONTHS")
    print("="*90)
    sorted_months = sorted(monthly.items(), key=lambda x: x[1]["pnl"], reverse=True)
    for month, data in sorted_months[:5]:
        win_rate = (data["winners"] / data["trades"] * 100) if data["trades"] > 0 else 0
        print(f"  {month}: Rs {data['pnl']:>12,.2f} | {data['trades']} trades | {win_rate:.1f}% win rate")

    print(f"\n{'='*90}")
    print("TOP 5 WORST MONTHS")
    print("="*90)
    for month, data in sorted_months[-5:]:
        win_rate = (data["winners"] / data["trades"] * 100) if data["trades"] > 0 else 0
        print(f"  {month}: Rs {data['pnl']:>12,.2f} | {data['trades']} trades | {win_rate:.1f}% win rate")

    # Best and worst days
    print(f"\n{'='*90}")
    print("TOP 10 BEST DAYS")
    print("="*90)
    sorted_days = sorted(daily_data, key=lambda x: x["pnl"], reverse=True)
    for day in sorted_days[:10]:
        print(f"  {day['date']}: Rs {day['pnl']:>10,.2f} | {day['trades']} trades | {day['winners']}W/{day['losers']}L")

    print(f"\n{'='*90}")
    print("TOP 10 WORST DAYS")
    print("="*90)
    for day in sorted_days[-10:]:
        print(f"  {day['date']}: Rs {day['pnl']:>10,.2f} | {day['trades']} trades | {day['winners']}W/{day['losers']}L")

    # Monthly statistics
    print(f"\n{'='*90}")
    print("MONTHLY STATISTICS")
    print("="*90)
    monthly_pnls = [m["pnl"] for m in monthly.values()]
    winning_months = len([p for p in monthly_pnls if p > 0])
    losing_months = len([p for p in monthly_pnls if p <= 0])

    print(f"Total Months: {len(monthly_pnls)}")
    print(f"Winning Months: {winning_months} ({winning_months/len(monthly_pnls)*100:.1f}%)")
    print(f"Losing Months: {losing_months} ({losing_months/len(monthly_pnls)*100:.1f}%)")
    print(f"Best Month: Rs {max(monthly_pnls):,.2f}")
    print(f"Worst Month: Rs {min(monthly_pnls):,.2f}")
    print(f"Avg Monthly PnL: Rs {sum(monthly_pnls)/len(monthly_pnls):,.2f}")

    # Daily statistics
    print(f"\n{'='*90}")
    print("DAILY STATISTICS")
    print("="*90)
    daily_pnls = [d["pnl"] for d in daily_data]
    winning_days = len([p for p in daily_pnls if p > 0])
    losing_days = len([p for p in daily_pnls if p <= 0])
    zero_days = len([d for d in daily_data if d["trades"] == 0])

    print(f"Trading Days: {len(daily_pnls)}")
    print(f"Winning Days: {winning_days} ({winning_days/len(daily_pnls)*100:.1f}%)")
    print(f"Losing Days: {losing_days} ({losing_days/len(daily_pnls)*100:.1f}%)")
    print(f"Best Day: Rs {max(daily_pnls):,.2f}")
    print(f"Worst Day: Rs {min(daily_pnls):,.2f}")
    print(f"Avg Daily PnL: Rs {sum(daily_pnls)/len(daily_pnls):,.2f}")

if __name__ == "__main__":
    main()
