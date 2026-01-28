"""
Simulate Impact of Time-Based Filters on MODERATE Profile
Shows before/after comparison before implementing changes.
"""

import json
from pathlib import Path
from collections import defaultdict

BACKTEST_FOLDERS = [
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-141442_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-185203_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-194823_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251223-111540_extracted",
]

BROKERAGE_PER_ORDER = 20
STT_RATE = 0.00025
EXCHANGE_TXN_RATE = 0.0000345
GST_RATE = 0.18
SEBI_RATE = 0.000001
STAMP_DUTY_RATE = 0.00003


def calculate_charges(entry_price, exit_price, qty, direction):
    buy_value = entry_price * qty if direction == "LONG" else exit_price * qty
    sell_value = exit_price * qty if direction == "LONG" else entry_price * qty
    turnover = buy_value + sell_value
    brokerage = min(BROKERAGE_PER_ORDER * 2, turnover * 0.0003)
    stt = sell_value * STT_RATE
    exchange_txn = turnover * EXCHANGE_TXN_RATE
    gst = (brokerage + exchange_txn) * GST_RATE
    sebi = turnover * SEBI_RATE
    stamp_duty = buy_value * STAMP_DUTY_RATE
    return brokerage + stt + exchange_txn + gst + sebi + stamp_duty


def load_trades():
    trades = []
    seen_ids = set()

    for folder in BACKTEST_FOLDERS:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue

        for date_dir in sorted(folder_path.iterdir()):
            if not date_dir.is_dir():
                continue

            events_file = date_dir / "events.jsonl"
            analytics_file = date_dir / "analytics.jsonl"

            if not events_file.exists() or not analytics_file.exists():
                continue

            decisions = {}
            with open(events_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get("type") == "DECISION":
                            trade_id = event.get("trade_id")
                            if trade_id:
                                plan = event.get("plan", {})
                                decision = event.get("decision", {})
                                sizing = plan.get("sizing", {})
                                indicators = plan.get("indicators", {})
                                ranking = plan.get("ranking", {})

                                decisions[trade_id] = {
                                    "setup": decision.get("setup_type", plan.get("strategy", "unknown")),
                                    "direction": plan.get("bias", "long").upper(),
                                    "cap_segment": sizing.get("cap_segment", "unknown"),
                                    "regime": plan.get("regime", "unknown"),
                                    "decision_ts": plan.get("decision_ts", event.get("ts", "")),
                                    "date": str(date_dir.name),
                                    "adx": indicators.get("adx", 0),
                                    "rsi": indicators.get("rsi", 50),
                                    "rank_score": ranking.get("score", 0),
                                    "entry_price": plan.get("entry_ref_price", 0),
                                    "qty": sizing.get("qty", 0),
                                }
                    except:
                        continue

            analytics_by_trade = defaultdict(list)
            with open(analytics_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        analytics = json.loads(line)
                        trade_id = analytics.get("trade_id")
                        if trade_id:
                            analytics_by_trade[trade_id].append(analytics)
                    except:
                        continue

            for trade_id, decision in decisions.items():
                if trade_id in seen_ids:
                    continue

                trade_analytics = analytics_by_trade.get(trade_id, [])
                if not trade_analytics:
                    continue

                final_exit = None
                for a in trade_analytics:
                    if a.get("is_final_exit"):
                        final_exit = a
                        break

                if not final_exit:
                    continue

                seen_ids.add(trade_id)

                entry_hour = 10
                entry_minute = 0
                try:
                    ts = decision.get("decision_ts", "")
                    if ts and len(ts) >= 16:
                        entry_hour = int(ts[11:13])
                        entry_minute = int(ts[14:16])
                except:
                    pass

                exit_reason = final_exit.get("reason", "unknown")
                pnl = final_exit.get("total_trade_pnl", 0)
                exit_price = final_exit.get("exit_price", 0)
                entry_price = final_exit.get("actual_entry_price", decision["entry_price"])
                qty = final_exit.get("qty", decision["qty"])

                is_sl_hit = "hard_sl" in exit_reason.lower() or "stop" in exit_reason.lower()

                charges = 0
                if entry_price > 0 and exit_price > 0 and qty > 0:
                    charges = calculate_charges(entry_price, exit_price, qty, decision["direction"])

                trades.append({
                    "trade_id": trade_id,
                    "date": decision["date"],
                    "setup": decision["setup"],
                    "direction": decision["direction"],
                    "cap_segment": decision["cap_segment"],
                    "regime": decision["regime"],
                    "entry_hour": entry_hour,
                    "entry_minute": entry_minute,
                    "exit_reason": exit_reason,
                    "is_sl_hit": is_sl_hit,
                    "gross_pnl": pnl,
                    "charges": charges,
                    "net_pnl": pnl - charges,
                    "adx": decision["adx"],
                    "rsi": decision["rsi"],
                    "rank_score": decision["rank_score"],
                })

    return trades


def apply_moderate_filter(trade):
    """Base MODERATE profile filter (no time constraints)"""
    setup = trade["setup"]
    filters = {
        "range_bounce_short": {"allowed_caps": ["micro_cap", "small_cap"]},
        "resistance_bounce_short": {"allowed_caps": ["micro_cap", "small_cap"]},
        "volume_spike_reversal_long": {"blocked_regimes": ["chop"]},
        "squeeze_release_long": {},
        "orb_pullback_long": {"blocked_caps": ["large_cap"]},
        "orb_pullback_short": {},
        "break_of_structure_long": {"blocked_regimes": ["trend_up"]},
        "support_bounce_long": {"allowed_caps": ["micro_cap"], "rank_min": 1.5},
        "range_bounce_long": {"rsi_min": 55},
        "breakout_long": {"allowed_regimes": ["trend_down"]},
        "premium_zone_short": {"adx_min": 20, "adx_max": 20},
        "order_block_short": {},
        "orb_breakout_long": {"blocked": True},
        "first_hour_momentum_long": {"blocked": True},
        "discount_zone_long": {"blocked": True},
        "failure_fade_short": {"blocked": True},
        "order_block_long": {"blocked": True},
    }

    if setup not in filters:
        return False

    config = filters[setup]

    if config.get("blocked"):
        return False

    if "allowed_caps" in config:
        if trade["cap_segment"] not in config["allowed_caps"]:
            return False

    if "blocked_caps" in config:
        if trade["cap_segment"] in config["blocked_caps"]:
            return False

    if "allowed_regimes" in config:
        if trade["regime"] not in config["allowed_regimes"]:
            return False

    if "blocked_regimes" in config:
        if trade["regime"] in config["blocked_regimes"]:
            return False

    if "adx_min" in config:
        if trade["adx"] < config["adx_min"]:
            return False

    if "adx_max" in config:
        if trade["adx"] > config["adx_max"]:
            return False

    if "rsi_min" in config:
        if trade["rsi"] < config["rsi_min"]:
            return False

    if "rank_min" in config:
        if trade["rank_score"] < config["rank_min"]:
            return False

    return True


def calc_stats(trades_list):
    if not trades_list:
        return {"count": 0, "sl_count": 0, "sl_pct": 0, "gross_pnl": 0, "net_pnl": 0, "avg_net": 0, "win_rate": 0}

    sl_count = sum(1 for t in trades_list if t["is_sl_hit"])
    winners = sum(1 for t in trades_list if t["net_pnl"] > 0)
    gross_pnl = sum(t["gross_pnl"] for t in trades_list)
    net_pnl = sum(t["net_pnl"] for t in trades_list)

    return {
        "count": len(trades_list),
        "sl_count": sl_count,
        "sl_pct": sl_count / len(trades_list) * 100,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "avg_net": net_pnl / len(trades_list),
        "win_rate": winners / len(trades_list) * 100,
    }


def main():
    print("=" * 120)
    print("TIME FILTER SIMULATION - MODERATE PROFILE")
    print("=" * 120)

    all_trades = load_trades()
    moderate_trades = [t for t in all_trades if apply_moderate_filter(t)]

    print(f"Total trades: {len(all_trades)}")
    print(f"MODERATE trades: {len(moderate_trades)}")

    # Define time filter options to test
    time_filter_options = [
        {
            "name": "BASELINE (No Time Filter)",
            "description": "Current MODERATE profile without time constraints",
            "filters": {}
        },
        {
            "name": "OPTION 1: Block 10:00 hour for all LONG",
            "description": "Block all LONG entries during 10:00-10:59",
            "filters": {"long_blocked_hours": [10]}
        },
        {
            "name": "OPTION 2: Block 10:00 hour for range_bounce_long only",
            "description": "Block range_bounce_long during 10:00-10:59 (53% SL)",
            "filters": {"setup_blocked_hours": {"range_bounce_long": [10]}}
        },
        {
            "name": "OPTION 3: Block 10:00-10:30 for all LONG",
            "description": "Block all LONG entries during first 30 mins after 10:00",
            "filters": {"long_blocked_windows": ["10:00"]}
        },
        {
            "name": "OPTION 4: Prefer afternoon (13:00+) for all",
            "description": "Only allow entries from 13:00 onwards",
            "filters": {"allowed_hours": [13, 14]}
        },
        {
            "name": "OPTION 5: Block lunch (12:00-13:00)",
            "description": "Block all entries during lunch hour",
            "filters": {"blocked_hours": [12]}
        },
        {
            "name": "OPTION 6: Optimal windows only",
            "description": "Allow only 11:00-12:00 and 13:00-14:30 (best PnL windows)",
            "filters": {"allowed_hours": [11, 13, 14]}
        },
    ]

    results = []

    for option in time_filter_options:
        filters = option["filters"]

        # Apply time filter to MODERATE trades
        filtered = []
        for t in moderate_trades:
            # Check if blocked
            blocked = False

            # Global blocked hours
            if "blocked_hours" in filters:
                if t["entry_hour"] in filters["blocked_hours"]:
                    blocked = True

            # Global allowed hours
            if "allowed_hours" in filters and not blocked:
                if t["entry_hour"] not in filters["allowed_hours"]:
                    blocked = True

            # Direction-specific blocked hours
            if "long_blocked_hours" in filters and t["direction"] == "LONG":
                if t["entry_hour"] in filters["long_blocked_hours"]:
                    blocked = True

            # Direction-specific blocked windows (30-min)
            if "long_blocked_windows" in filters and t["direction"] == "LONG":
                window = f"{t['entry_hour']:02d}:{0 if t['entry_minute'] < 30 else 30:02d}"
                if window in filters["long_blocked_windows"]:
                    blocked = True

            # Setup-specific blocked hours
            if "setup_blocked_hours" in filters:
                setup_hours = filters["setup_blocked_hours"].get(t["setup"], [])
                if t["entry_hour"] in setup_hours:
                    blocked = True

            if not blocked:
                filtered.append(t)

        stats = calc_stats(filtered)
        baseline_stats = calc_stats(moderate_trades)

        # Calculate improvement
        trades_blocked = baseline_stats["count"] - stats["count"]
        pnl_change = stats["net_pnl"] - baseline_stats["net_pnl"]
        sl_change = stats["sl_pct"] - baseline_stats["sl_pct"]

        results.append({
            "option": option["name"],
            "description": option["description"],
            "stats": stats,
            "trades_blocked": trades_blocked,
            "pnl_change": pnl_change,
            "sl_change": sl_change,
        })

    # Print results
    print("\n" + "=" * 120)
    print("SIMULATION RESULTS")
    print("=" * 120)

    print(f"\n{'Option':<50} {'Trades':>10} {'Blocked':>10} {'SL%':>8} {'dSL%':>8} {'Net PnL':>15} {'dPNL':>15} {'Avg':>10}")
    print("-" * 130)

    for r in results:
        s = r["stats"]
        print(f"{r['option']:<50} {s['count']:>10,} {r['trades_blocked']:>10,} {s['sl_pct']:>7.1f}% {r['sl_change']:>+7.1f}% Rs {s['net_pnl']:>12,.0f} Rs {r['pnl_change']:>+12,.0f} Rs {s['avg_net']:>7,.0f}")

    # Detailed breakdown for each option
    print("\n" + "=" * 120)
    print("DETAILED BREAKDOWN")
    print("=" * 120)

    for r in results:
        s = r["stats"]
        print(f"\n>>> {r['option']}")
        print(f"    {r['description']}")
        print(f"    Trades: {s['count']:,} (blocked {r['trades_blocked']:,})")
        print(f"    SL Rate: {s['sl_pct']:.1f}% (change {r['sl_change']:+.1f}%)")
        print(f"    Win Rate: {s['win_rate']:.1f}%")
        print(f"    Net PnL: Rs {s['net_pnl']:,.0f} (change Rs {r['pnl_change']:+,.0f})")
        print(f"    Avg PnL: Rs {s['avg_net']:.0f}")

    # Financial impact with 3L capital
    print("\n" + "=" * 120)
    print("FINANCIAL IMPACT (Rs 3L Capital, 3 Years)")
    print("=" * 120)

    capital = 300000
    years = 3

    print(f"\n{'Option':<50} {'Net PnL':>15} {'Tax':>12} {'Final':>15} {'Ann ROI':>10} {'Monthly':>12}")
    print("-" * 120)

    for r in results:
        s = r["stats"]
        net = s["net_pnl"]
        tax = net * 0.312 if net > 0 else 0
        final = net - tax
        annual_roi = (final / years / capital) * 100
        monthly = final / 36

        print(f"{r['option']:<50} Rs {net:>12,.0f} Rs {tax:>9,.0f} Rs {final:>12,.0f} {annual_roi:>9.1f}% Rs {monthly:>9,.0f}")

    # Recommendation
    print("\n" + "=" * 120)
    print("RECOMMENDATION")
    print("=" * 120)

    # Find best option (highest net PnL improvement without losing too many trades)
    best = max(results[1:], key=lambda x: x["pnl_change"])  # Skip baseline

    print(f"""
Based on the simulation:

BEST OPTION: {best['option']}
  - {best['description']}
  - Blocks {best['trades_blocked']:,} trades
  - Reduces SL rate by {-best['sl_change']:.1f}%
  - Changes PnL by Rs {best['pnl_change']:+,.0f}

NOTE: Negative PnL change means the blocked trades were actually profitable on average.
      Positive PnL change means blocking those trades improves overall performance.

Do you want me to implement any of these time filters?
""")


if __name__ == "__main__":
    main()
