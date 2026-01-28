"""
Optimal Filter Calculation with MODERATE Profile + Additional Optimizations
Goal: 4-5 trades/day, Rs 300-400/trade average for profitable setups
"""

import json
from pathlib import Path
from collections import defaultdict

BACKTEST_FOLDERS = [
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-141442_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-185203_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251222-194823_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251223-111540_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251223-201604_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251224-073333_extracted",
    r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251224-105804_extracted",
]

# Cost model
BROKERAGE_PER_ORDER = 20
STT_RATE = 0.00025
EXCHANGE_TXN_RATE = 0.0000345
GST_RATE = 0.18
SEBI_RATE = 0.000001
STAMP_DUTY_RATE = 0.00003
TAX_RATE = 0.312


def calculate_charges(entry_price, exit_price, qty, direction):
    if entry_price <= 0 or exit_price <= 0 or qty <= 0:
        return 0
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


def load_all_trades():
    """Load all trades with full metadata."""
    trades = []
    seen_ids = set()
    sessions = 0

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

            sessions += 1
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
                    "gross_pnl": pnl,
                    "charges": charges,
                    "net_pnl": pnl - charges,
                    "adx": decision["adx"],
                    "rsi": decision["rsi"],
                    "rank_score": decision["rank_score"],
                })

    return trades, sessions


# CURRENT MODERATE PROFILE (from level_config.json)
CURRENT_MODERATE = {
    "range_bounce_short": {"allowed_caps": ["micro_cap", "small_cap"]},
    "resistance_bounce_short": {"allowed_caps": ["micro_cap", "small_cap"]},
    "volume_spike_reversal_long": {"blocked_regimes": ["chop"]},
    "squeeze_release_long": {},
    "orb_pullback_long": {"blocked_caps": ["large_cap"]},
    "orb_pullback_short": {},
    "break_of_structure_long": {"blocked_regimes": ["trend_up"]},
    "support_bounce_long": {"allowed_caps": ["micro_cap"], "min_rank_score": 1.5},
    "range_bounce_long": {"min_rsi": 55},
    "breakout_long": {"allowed_regimes": ["trend_down"]},
    "premium_zone_short": {"min_adx": 20, "max_adx": 20},
    "order_block_short": {},
    "fair_value_gap_short": {},
    "order_block_long": {"blocked": True},
    "orb_breakout_long": {"blocked": True},
    "first_hour_momentum_long": {"blocked": True},
    "discount_zone_long": {"blocked": True},
    "failure_fade_short": {"blocked": True},
}

# PROPOSED OPTIMIZED PROFILE (based on analysis)
OPTIMIZED_MODERATE = {
    # TOP PERFORMERS - keep and optimize
    "range_bounce_short": {
        "allowed_caps": ["micro_cap", "small_cap"],  # mid_cap has Rs 12/trade avg
        "blocked_hours": [],  # 10:00 has Rs 92/trade vs Rs 133 at 11:00 but still profitable
    },
    "resistance_bounce_short": {
        "allowed_caps": ["micro_cap", "small_cap"],  # large_cap: -Rs 43/trade, mid_cap: -Rs 2/trade
        "blocked_hours": [14],  # 14:00 has Rs -9/trade
    },
    "volume_spike_reversal_long": {
        "blocked_regimes": ["chop"],  # chop has -Rs 215/trade
        "blocked_hours": [10],  # 10:00 has -Rs 75/trade
        "min_adx": 20,  # ADX < 20 has -Rs 368/trade
    },
    "squeeze_release_long": {
        "blocked_regimes": ["trend_down"],  # trend_down has -Rs 83/trade
        "allowed_caps": ["mid_cap", "large_cap"],  # small_cap has -Rs 109/trade
    },
    "order_block_short": {
        "blocked_caps": ["mid_cap"],  # mid_cap has -Rs 1583/trade
        "min_adx": 15,  # ADX < 15 has -Rs 902/trade
    },

    # MARGINAL/LOSING - block or heavily filter
    "support_bounce_long": {"blocked": True},  # -Rs 141/trade avg
    "range_bounce_long": {"blocked": True},  # -Rs 237/trade avg
    "premium_zone_short": {"blocked": True},  # -Rs 41/trade avg
    "breakout_long": {"blocked": True},  # -Rs 7/trade avg
    "break_of_structure_long": {"blocked": True},  # -Rs 267/trade avg
    "orb_pullback_long": {"blocked": True},  # -Rs 23/trade avg
    "orb_pullback_short": {"blocked": True},  # -Rs 106/trade avg
    "failure_fade_short": {"blocked": True},
    "orb_breakout_long": {"blocked": True},
    "first_hour_momentum_long": {"blocked": True},
    "discount_zone_long": {"blocked": True},
    "order_block_long": {"blocked": True},
    "fair_value_gap_short": {},  # Only 1 trade, Rs 1003 profit - keep
}


def apply_filter(trade, profile):
    """Apply profile filter to a trade."""
    setup = trade["setup"]

    if setup not in profile:
        return False  # Unknown setup = blocked

    config = profile[setup]

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

    if "blocked_hours" in config:
        if trade["entry_hour"] in config["blocked_hours"]:
            return False

    if "min_adx" in config:
        if trade["adx"] < config["min_adx"]:
            return False

    if "max_adx" in config:
        if trade["adx"] > config["max_adx"]:
            return False

    if "min_rsi" in config:
        if trade["rsi"] < config["min_rsi"]:
            return False

    if "min_rank_score" in config:
        if trade["rank_score"] < config["min_rank_score"]:
            return False

    return True


def calc_stats(trades_list):
    if not trades_list:
        return {"count": 0, "net_pnl": 0, "avg_net": 0, "win_rate": 0, "charges": 0}

    winners = [t for t in trades_list if t["net_pnl"] > 0]
    net_pnl = sum(t["net_pnl"] for t in trades_list)
    charges = sum(t["charges"] for t in trades_list)

    return {
        "count": len(trades_list),
        "net_pnl": net_pnl,
        "charges": charges,
        "avg_net": net_pnl / len(trades_list),
        "win_rate": len(winners) / len(trades_list) * 100,
    }


def main():
    print("=" * 120)
    print("FILTER OPTIMIZATION CALCULATION")
    print("=" * 120)

    trades, sessions = load_all_trades()
    print(f"\nLoaded {len(trades):,} trades from {sessions} sessions")
    print(f"Target: 4-5 trades/day = {4*sessions:,} to {5*sessions:,} total trades")

    # Apply current MODERATE profile
    current_trades = [t for t in trades if apply_filter(t, CURRENT_MODERATE)]
    current_stats = calc_stats(current_trades)

    print(f"\n{'='*80}")
    print("CURRENT MODERATE PROFILE")
    print(f"{'='*80}")
    print(f"Trades: {current_stats['count']:,} ({current_stats['count']/sessions:.1f}/day)")
    print(f"Net PnL: Rs {current_stats['net_pnl']:,.0f}")
    print(f"Post-Tax: Rs {current_stats['net_pnl'] * (1-TAX_RATE):,.0f}")
    print(f"Avg Net/Trade: Rs {current_stats['avg_net']:.0f}")
    print(f"Win Rate: {current_stats['win_rate']:.1f}%")

    # Breakdown by setup
    print(f"\nBy Setup:")
    print(f"{'Setup':<35} {'Trades':>8} {'Net PnL':>15} {'Avg Net':>10}")
    print("-" * 70)

    by_setup = defaultdict(list)
    for t in current_trades:
        by_setup[t["setup"]].append(t)

    for setup, setup_trades in sorted(by_setup.items(), key=lambda x: -calc_stats(x[1])["net_pnl"]):
        s = calc_stats(setup_trades)
        print(f"{setup:<35} {s['count']:>8,} Rs {s['net_pnl']:>12,.0f} Rs {s['avg_net']:>7,.0f}")

    # Apply OPTIMIZED profile
    optimized_trades = [t for t in trades if apply_filter(t, OPTIMIZED_MODERATE)]
    optimized_stats = calc_stats(optimized_trades)

    print(f"\n{'='*80}")
    print("OPTIMIZED MODERATE PROFILE")
    print(f"{'='*80}")
    print(f"Trades: {optimized_stats['count']:,} ({optimized_stats['count']/sessions:.1f}/day)")
    print(f"Net PnL: Rs {optimized_stats['net_pnl']:,.0f}")
    print(f"Post-Tax: Rs {optimized_stats['net_pnl'] * (1-TAX_RATE):,.0f}")
    print(f"Avg Net/Trade: Rs {optimized_stats['avg_net']:.0f}")
    print(f"Win Rate: {optimized_stats['win_rate']:.1f}%")

    # Breakdown by setup
    print(f"\nBy Setup:")
    print(f"{'Setup':<35} {'Trades':>8} {'Net PnL':>15} {'Avg Net':>10}")
    print("-" * 70)

    by_setup_opt = defaultdict(list)
    for t in optimized_trades:
        by_setup_opt[t["setup"]].append(t)

    for setup, setup_trades in sorted(by_setup_opt.items(), key=lambda x: -calc_stats(x[1])["net_pnl"]):
        s = calc_stats(setup_trades)
        print(f"{setup:<35} {s['count']:>8,} Rs {s['net_pnl']:>12,.0f} Rs {s['avg_net']:>7,.0f}")

    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    trade_reduction = current_stats["count"] - optimized_stats["count"]
    pnl_change = optimized_stats["net_pnl"] - current_stats["net_pnl"]
    avg_change = optimized_stats["avg_net"] - current_stats["avg_net"]

    print(f"\n{'Metric':<30} {'Current':>15} {'Optimized':>15} {'Change':>15}")
    print("-" * 80)
    print(f"{'Trades':<30} {current_stats['count']:>15,} {optimized_stats['count']:>15,} {-trade_reduction:>+15,}")
    print(f"{'Trades/Day':<30} {current_stats['count']/sessions:>15.1f} {optimized_stats['count']/sessions:>15.1f} {(optimized_stats['count']-current_stats['count'])/sessions:>+15.1f}")
    print(f"{'Net PnL':<30} Rs {current_stats['net_pnl']:>12,.0f} Rs {optimized_stats['net_pnl']:>12,.0f} Rs {pnl_change:>+12,.0f}")
    print(f"{'Post-Tax (31.2%)':<30} Rs {current_stats['net_pnl']*(1-TAX_RATE):>12,.0f} Rs {optimized_stats['net_pnl']*(1-TAX_RATE):>12,.0f} Rs {pnl_change*(1-TAX_RATE):>+12,.0f}")
    print(f"{'Avg Net/Trade':<30} Rs {current_stats['avg_net']:>12,.0f} Rs {optimized_stats['avg_net']:>12,.0f} Rs {avg_change:>+12,.0f}")
    print(f"{'Win Rate':<30} {current_stats['win_rate']:>14.1f}% {optimized_stats['win_rate']:>14.1f}% {optimized_stats['win_rate']-current_stats['win_rate']:>+14.1f}%")

    # Further optimization: What if we want exactly 4-5 trades/day?
    print(f"\n{'='*80}")
    print("TARGETED TRADE COUNT SCENARIOS")
    print(f"{'='*80}")

    # Try rank_score filters to reduce further
    rank_thresholds = [0, 0.5, 1.0, 1.5, 2.0]

    print(f"\nWith additional min_rank_score filter on optimized profile:")
    print(f"{'Rank Threshold':<20} {'Trades':>10} {'Trades/Day':>12} {'Net PnL':>15} {'Avg Net':>10}")
    print("-" * 70)

    for threshold in rank_thresholds:
        filtered = [t for t in optimized_trades if t["rank_score"] >= threshold]
        s = calc_stats(filtered)
        if s["count"] > 0:
            print(f"min_rank_score={threshold:<8} {s['count']:>10,} {s['count']/sessions:>11.1f} Rs {s['net_pnl']:>12,.0f} Rs {s['avg_net']:>7,.0f}")

    # Final recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    # Find best rank threshold for 4-5 trades/day target
    target_min = 4 * sessions
    target_max = 5 * sessions

    best_threshold = 0
    best_trades = None

    for threshold in [x/10 for x in range(0, 30)]:  # 0 to 3.0 in 0.1 steps
        filtered = [t for t in optimized_trades if t["rank_score"] >= threshold]
        if target_min <= len(filtered) <= target_max:
            s = calc_stats(filtered)
            if best_trades is None or s["avg_net"] > calc_stats(best_trades)["avg_net"]:
                best_threshold = threshold
                best_trades = filtered
        elif len(filtered) < target_min and best_trades is None:
            best_threshold = threshold - 0.1
            best_trades = [t for t in optimized_trades if t["rank_score"] >= best_threshold]
            break

    if best_trades:
        best_stats = calc_stats(best_trades)
        print(f"""
OPTIMAL CONFIGURATION:
  - Apply OPTIMIZED_MODERATE profile (block losing setups)
  - Add min_rank_score = {best_threshold:.1f} filter

EXPECTED RESULTS:
  - Trades: {best_stats['count']:,} ({best_stats['count']/sessions:.1f}/day)
  - Net PnL: Rs {best_stats['net_pnl']:,.0f}
  - Post-Tax (31.2%): Rs {best_stats['net_pnl']*(1-TAX_RATE):,.0f}
  - Avg Net/Trade: Rs {best_stats['avg_net']:.0f}
  - Win Rate: {best_stats['win_rate']:.1f}%

CHANGES FROM CURRENT:
  - Trades reduced by {current_stats['count'] - best_stats['count']:,} ({(current_stats['count'] - best_stats['count'])/current_stats['count']*100:.0f}%)
  - Avg PnL increased by Rs {best_stats['avg_net'] - current_stats['avg_net']:.0f}/trade
""")

    # Detail the exact changes needed
    print(f"\n{'='*80}")
    print("EXACT FILTER CHANGES TO IMPLEMENT")
    print(f"{'='*80}")

    print("""
LEVEL PIPELINE (level_config.json):
  range_bounce_short:
    - allowed_caps: ["micro_cap", "small_cap"]  # KEEP

  resistance_bounce_short:
    - allowed_caps: ["micro_cap", "small_cap"]  # KEEP
    - blocked_hours: [14]                        # ADD - blocks -Rs 9/trade hour

  support_bounce_long:
    - blocked: true                              # CHANGE - was allowed for micro_cap

REVERSION PIPELINE (reversion_config.json):
  volume_spike_reversal_long:
    - blocked_regimes: ["chop"]                  # KEEP
    - blocked_hours: [10]                        # ADD - blocks -Rs 75/trade hour
    - min_adx: 20                                # ADD - blocks -Rs 368/trade trades

  range_bounce_long:
    - blocked: true                              # CHANGE - was min_rsi: 55 (still losing)

BREAKOUT PIPELINE (breakout_config.json):
  squeeze_release_long:
    - blocked_regimes: ["trend_down"]            # ADD - blocks -Rs 83/trade trades
    - allowed_caps: ["mid_cap", "large_cap"]     # ADD - small_cap has -Rs 109/trade

  breakout_long:
    - blocked: true                              # CHANGE - was allowed for trend_down

  order_block_short:
    - blocked_caps: ["mid_cap"]                  # ADD - blocks -Rs 1583/trade trades
    - min_adx: 15                                # ADD - blocks -Rs 902/trade trades

MOMENTUM PIPELINE (momentum_config.json):
  - All setups already blocked                   # KEEP
    (first_hour_momentum_long, orb_breakout_long, orb_pullback_*, etc.)

  premium_zone_short:
    - blocked: true                              # CHANGE - was min_adx: 20

BLOCKED SETUPS (NO TRADES):
  - range_bounce_long (was -Rs 237/trade)
  - support_bounce_long (was -Rs 141/trade)
  - premium_zone_short (was -Rs 41/trade)
  - breakout_long (was -Rs 7/trade)
  - break_of_structure_long (was -Rs 267/trade)
  - first_hour_momentum_long (was -Rs 138/trade)
  - discount_zone_long (was -Rs 248/trade)
  - orb_breakout_long (was -Rs 78/trade)
  - orb_pullback_long (was -Rs 23/trade)
  - orb_pullback_short (was -Rs 106/trade)
  - order_block_long (was -Rs 56/trade)
  - failure_fade_short (was -Rs 791/trade)

ACTIVE SETUPS (5 setups):
  1. range_bounce_short      - Rs 109/trade avg
  2. resistance_bounce_short - Rs 72/trade avg (will improve with hour filter)
  3. volume_spike_reversal_long - Rs 204/trade avg (will improve with filters)
  4. squeeze_release_long    - Rs 269/trade avg (will improve with filters)
  5. order_block_short       - Rs 638/trade avg (will improve with filters)
""")


if __name__ == "__main__":
    main()
