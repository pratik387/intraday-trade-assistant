"""
Analyze daily capital requirements based on concurrent trades.
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

def analyze_session(session_path):
    """Analyze a single trading session for capital requirements."""
    events_file = session_path / "events.jsonl"
    analytics_file = session_path / "analytics.jsonl"

    if not events_file.exists() or not analytics_file.exists():
        return None

    # Collect trade info: trade_id -> {entry_time, exit_time, notional}
    trades = {}

    # Get entry times and notional from TRIGGER events
    with open(events_file, encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line)
                if event.get("type") == "DECISION":
                    trade_id = event.get("trade_id")
                    if trade_id:
                        notional = event.get("plan", {}).get("sizing", {}).get("notional", 0)
                        trades[trade_id] = {"notional": notional, "entry_time": None, "exit_time": None}

                if event.get("type") == "TRIGGER":
                    trade_id = event.get("trade_id")
                    if trade_id and trade_id in trades:
                        trades[trade_id]["entry_time"] = parse_timestamp(event.get("ts"))
            except:
                continue

    # Get exit times from analytics (final exit)
    with open(analytics_file, encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line)
                trade_id = event.get("trade_id")
                if trade_id and trade_id in trades and event.get("is_final_exit"):
                    trades[trade_id]["exit_time"] = parse_timestamp(event.get("timestamp"))
            except:
                continue

    # Filter to only triggered trades with valid times
    active_trades = {
        tid: t for tid, t in trades.items()
        if t["entry_time"] and t["exit_time"] and t["notional"] > 0
    }

    if not active_trades:
        return None

    # Calculate peak concurrent capital
    # Create events list: (time, +notional for entry, -notional for exit)
    events = []
    for tid, t in active_trades.items():
        events.append((t["entry_time"], t["notional"], "entry", tid))
        events.append((t["exit_time"], -t["notional"], "exit", tid))

    # Sort by time
    events.sort(key=lambda x: x[0])

    # Calculate peak
    current_capital = 0
    peak_capital = 0
    peak_trades = 0
    current_trades = 0

    for time, notional_change, event_type, tid in events:
        if event_type == "entry":
            current_capital += notional_change
            current_trades += 1
        else:
            current_capital += notional_change  # negative
            current_trades -= 1

        if current_capital > peak_capital:
            peak_capital = current_capital
            peak_trades = current_trades

    return {
        "date": session_path.name,
        "total_trades": len(active_trades),
        "peak_capital": peak_capital,
        "peak_concurrent_trades": peak_trades,
        "avg_notional": sum(t["notional"] for t in active_trades.values()) / len(active_trades) if active_trades else 0,
    }

def main():
    print("=" * 80)
    print("DAILY CAPITAL REQUIREMENTS ANALYSIS")
    print("=" * 80)

    all_sessions = []

    for backtest_dir in BACKTEST_DIRS:
        base_path = Path(backtest_dir)
        if not base_path.exists():
            continue

        for session_path in sorted(base_path.iterdir()):
            if not session_path.is_dir():
                continue

            result = analyze_session(session_path)
            if result:
                all_sessions.append(result)

    if not all_sessions:
        print("No sessions found!")
        return

    # Sort by peak capital
    all_sessions.sort(key=lambda x: x["peak_capital"], reverse=True)

    # Overall stats
    total_sessions = len(all_sessions)
    peak_capitals = [s["peak_capital"] for s in all_sessions]
    peak_trades = [s["peak_concurrent_trades"] for s in all_sessions]

    print(f"\nTotal Sessions Analyzed: {total_sessions}")

    print(f"\n{'='*80}")
    print("PEAK CAPITAL STATISTICS")
    print("="*80)
    print(f"Maximum Peak Capital (single day): Rs {max(peak_capitals):,.2f}")
    print(f"Average Peak Capital: Rs {sum(peak_capitals)/len(peak_capitals):,.2f}")
    print(f"Median Peak Capital: Rs {sorted(peak_capitals)[len(peak_capitals)//2]:,.2f}")
    print(f"Minimum Peak Capital: Rs {min(peak_capitals):,.2f}")

    print(f"\n{'='*80}")
    print("CONCURRENT TRADES STATISTICS")
    print("="*80)
    print(f"Maximum Concurrent Trades (single day): {max(peak_trades)}")
    print(f"Average Peak Concurrent Trades: {sum(peak_trades)/len(peak_trades):.1f}")
    print(f"Median Peak Concurrent Trades: {sorted(peak_trades)[len(peak_trades)//2]}")

    # Distribution of peak capital
    print(f"\n{'='*80}")
    print("PEAK CAPITAL DISTRIBUTION")
    print("="*80)

    buckets = [
        (0, 50000, "< 50K"),
        (50000, 100000, "50K-1L"),
        (100000, 200000, "1L-2L"),
        (200000, 300000, "2L-3L"),
        (300000, 500000, "3L-5L"),
        (500000, 1000000, "5L-10L"),
        (1000000, float('inf'), "> 10L"),
    ]

    for low, high, label in buckets:
        count = len([c for c in peak_capitals if low <= c < high])
        pct = count / total_sessions * 100
        bar = "#" * int(pct / 2)
        print(f"{label:>10}: {count:>4} sessions ({pct:>5.1f}%) {bar}")

    # Top 10 highest capital days
    print(f"\n{'='*80}")
    print("TOP 10 HIGHEST CAPITAL REQUIREMENT DAYS")
    print("="*80)
    print(f"{'Date':<12} {'Peak Capital':>15} {'Concurrent':>12} {'Total Trades':>12}")
    print("-" * 55)
    for s in all_sessions[:10]:
        print(f"{s['date']:<12} Rs {s['peak_capital']:>12,.0f} {s['peak_concurrent_trades']:>12} {s['total_trades']:>12}")

    # Recommendation
    print(f"\n{'='*80}")
    print("CAPITAL RECOMMENDATION")
    print("="*80)

    # 95th percentile
    p95_idx = int(total_sessions * 0.95)
    p95_capital = sorted(peak_capitals)[p95_idx]
    p99_idx = int(total_sessions * 0.99)
    p99_capital = sorted(peak_capitals)[p99_idx]

    print(f"50th percentile (median): Rs {sorted(peak_capitals)[len(peak_capitals)//2]:,.0f}")
    print(f"75th percentile: Rs {sorted(peak_capitals)[int(len(peak_capitals)*0.75)]:,.0f}")
    print(f"90th percentile: Rs {sorted(peak_capitals)[int(len(peak_capitals)*0.90)]:,.0f}")
    print(f"95th percentile: Rs {p95_capital:,.0f}")
    print(f"99th percentile: Rs {p99_capital:,.0f}")
    print(f"Maximum: Rs {max(peak_capitals):,.0f}")

    print(f"\nRECOMMENDATION:")
    print(f"  - Conservative (95th %ile): Rs {p95_capital:,.0f}")
    print(f"  - Safe (99th %ile): Rs {p99_capital:,.0f}")
    print(f"  - Maximum coverage: Rs {max(peak_capitals):,.0f}")

    # Monthly breakdown
    print(f"\n{'='*80}")
    print("MONTHLY AVERAGE PEAK CAPITAL")
    print("="*80)

    monthly = defaultdict(list)
    for s in all_sessions:
        month = s["date"][:7]  # YYYY-MM
        monthly[month].append(s["peak_capital"])

    print(f"{'Month':<10} {'Avg Peak Capital':>15} {'Max Peak':>15} {'Sessions':>10}")
    print("-" * 55)
    for month in sorted(monthly.keys()):
        caps = monthly[month]
        print(f"{month:<10} Rs {sum(caps)/len(caps):>12,.0f} Rs {max(caps):>12,.0f} {len(caps):>10}")

if __name__ == "__main__":
    main()
