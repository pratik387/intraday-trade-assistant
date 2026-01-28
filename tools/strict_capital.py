"""Calculate capital for original STRICT profile (235 trades)"""
import json
from pathlib import Path
from collections import defaultdict

BACKTEST_FOLDERS = [
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-141442_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-185203_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-194823_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251223-111540_extracted",
]

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
                                    "cap_segment": sizing.get("cap_segment", "unknown"),
                                    "regime": plan.get("regime", "unknown"),
                                    "entry_price": plan.get("entry_ref_price", 0),
                                    "qty": sizing.get("qty", 0),
                                    "notional": sizing.get("notional", 0),
                                    "decision_ts": plan.get("decision_ts", event.get("ts", "")),
                                    "date": str(date_dir.name),
                                    "adx": indicators.get("adx", 0),
                                    "rsi": indicators.get("rsi", 50),
                                    "rank_score": ranking.get("score", 0),
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
                entry_time = decision["decision_ts"]
                exit_time = None
                actual_entry_price = decision["entry_price"]
                total_pnl = 0
                for a in trade_analytics:
                    if a.get("is_final_exit"):
                        exit_time = a.get("timestamp")
                        total_pnl = a.get("total_trade_pnl", 0)
                        if a.get("actual_entry_price"):
                            actual_entry_price = a.get("actual_entry_price")
                if not exit_time:
                    continue
                seen_ids.add(trade_id)
                trades.append({
                    "trade_id": trade_id,
                    "date": decision["date"],
                    "setup": decision["setup"],
                    "cap_segment": decision["cap_segment"],
                    "regime": decision["regime"],
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": actual_entry_price,
                    "qty": decision["qty"],
                    "notional": decision["notional"] or (actual_entry_price * decision["qty"]),
                    "pnl": total_pnl,
                    "adx": decision["adx"],
                    "rsi": decision["rsi"],
                    "rank_score": decision["rank_score"],
                })
    return trades


def apply_original_strict_filter(trade):
    """Original STRICT filters from v3 analysis that gave 235 trades"""
    setup = trade["setup"]

    # Exact filters from setup_filter_optimizer_v3.py STRICT profile
    filters = {
        "range_bounce_short": {"allowed_caps": ["micro_cap"], "rank_min": 2.0},
        "resistance_bounce_short": {"adx_min": 20, "rsi_min": 40, "rsi_max": 40},
        "volume_spike_reversal_long": {"allowed_caps": ["micro_cap"], "blocked_regimes": ["chop"]},
        "squeeze_release_long": {"allowed_caps": ["mid_cap"]},
        "orb_pullback_long": {"blocked_caps": ["large_cap"]},
        "orb_pullback_short": {"rsi_min": 40},
        "break_of_structure_long": {"allowed_caps": ["small_cap"], "blocked_regimes": ["trend_up"]},
        "support_bounce_long": {"allowed_caps": ["micro_cap"], "rank_min": 2.0},
        "range_bounce_long": {"adx_min": 15, "adx_max": 25, "rsi_min": 40, "rsi_max": 50},
        "order_block_short": {},
    }

    if setup not in filters:
        return False

    config = filters[setup]

    if "allowed_caps" in config:
        if trade["cap_segment"] not in config["allowed_caps"]:
            return False

    if "blocked_caps" in config:
        if trade["cap_segment"] in config["blocked_caps"]:
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

    if "rsi_max" in config:
        if trade["rsi"] > config["rsi_max"]:
            return False

    if "rank_min" in config:
        if trade["rank_score"] < config["rank_min"]:
            return False

    return True


def calculate_concurrent_capital(trades):
    by_date = defaultdict(list)
    for t in trades:
        by_date[t["date"]].append(t)

    daily_stats = []
    for date, day_trades in sorted(by_date.items()):
        if not day_trades:
            continue

        events = []
        for t in day_trades:
            events.append((t["entry_time"], "entry", t["notional"], t))
            events.append((t["exit_time"], "exit", t["notional"], t))
        events.sort(key=lambda x: x[0])

        current_capital = 0
        max_capital = 0
        max_positions = 0
        current_positions = 0

        for ts, event_type, notional, trade in events:
            if event_type == "entry":
                current_capital += notional
                current_positions += 1
            else:
                current_capital -= notional
                current_positions -= 1
            if current_capital > max_capital:
                max_capital = current_capital
                max_positions = current_positions

        total_pnl = sum(t["pnl"] for t in day_trades)
        daily_stats.append({
            "date": date,
            "trades": len(day_trades),
            "max_positions": max_positions,
            "max_capital": max_capital,
            "pnl": total_pnl,
        })

    return daily_stats


def main():
    print("="*100)
    print("STRICT PROFILE (Original 235-trade) - CAPITAL REQUIREMENTS")
    print("="*100)

    all_trades = load_trades()
    strict_trades = [t for t in all_trades if apply_original_strict_filter(t)]

    print(f"\nTotal trades after STRICT filter: {len(strict_trades)}")

    if len(strict_trades) == 0:
        print("No trades found")
        return

    daily_stats = calculate_concurrent_capital(strict_trades)
    days_with_trades = [d for d in daily_stats if d["trades"] > 0]

    if not days_with_trades:
        print("No trading days found")
        return

    max_capital = max(d["max_capital"] for d in days_with_trades)
    sorted_caps = sorted(d["max_capital"] for d in days_with_trades)
    p95_idx = int(len(sorted_caps) * 0.95)
    p95_cap = sorted_caps[p95_idx] if sorted_caps else 0
    avg_cap = sum(d["max_capital"] for d in days_with_trades) / len(days_with_trades)
    max_positions = max(d["max_positions"] for d in days_with_trades)
    avg_positions = sum(d["max_positions"] for d in days_with_trades) / len(days_with_trades)
    total_pnl = sum(d["pnl"] for d in days_with_trades)
    total_trades = sum(d["trades"] for d in days_with_trades)

    print(f"""
Summary:
  Total Trades: {total_trades}
  Trading Days with trades: {len(days_with_trades)}
  Trades per day (avg): {total_trades / 719:.2f}
  Total Gross P&L: Rs {total_pnl:,.0f}

Capital Requirements:
  Max Capital Ever Needed: Rs {max_capital:,.0f}
  95th Percentile Capital: Rs {p95_cap:,.0f}
  Average Daily Max Capital: Rs {avg_cap:,.0f}

Concurrent Positions:
  Max Positions Ever: {max_positions}
  Average Max Positions: {avg_positions:.1f}

Margin Requirement (20% for intraday):
  Max Margin Needed: Rs {max_capital * 0.20:,.0f}
  95th Percentile Margin: Rs {p95_cap * 0.20:,.0f}
  Average Daily Margin: Rs {avg_cap * 0.20:,.0f}
""")

    # Financial calculations
    # From original: 235 trades, Rs 1.30L net, Rs 0.90L final
    gross_pnl = total_pnl
    est_charges = total_trades * 55  # ~Rs 55 per trade avg
    net_after_charges = gross_pnl - est_charges
    tax = net_after_charges * 0.312 if net_after_charges > 0 else 0
    final_net = net_after_charges - tax

    margin = p95_cap * 0.20
    annual_roi = (final_net / 3 / margin) * 100 if margin > 0 else 0

    print(f"""Financial Summary:
  Gross P&L: Rs {gross_pnl:,.0f}
  Est. Trading Charges: Rs {est_charges:,.0f}
  Net P&L (after charges): Rs {net_after_charges:,.0f}
  Income Tax (31.2%): Rs {tax:,.0f}
  FINAL NET PROFIT (3 years): Rs {final_net:,.0f}

  Annual Net Profit: Rs {final_net/3:,.0f}
  Monthly Net Profit: Rs {final_net/36:,.0f}

  P95 Margin Required: Rs {margin:,.0f}
  Annual ROI (after tax): {annual_roi:.1f}%
""")

    print("Top 5 Highest Capital Days:")
    print(f"  {'Date':<12} {'Trades':>6} {'Pos':>5} {'Max Capital':>15} {'P&L':>12}")
    print("  " + "-"*55)
    top_days = sorted(days_with_trades, key=lambda x: x["max_capital"], reverse=True)[:5]
    for d in top_days:
        print(f"  {d['date']:<12} {d['trades']:>6} {d['max_positions']:>5} Rs {d['max_capital']:>12,.0f} Rs {d['pnl']:>10,.0f}")


if __name__ == "__main__":
    main()
