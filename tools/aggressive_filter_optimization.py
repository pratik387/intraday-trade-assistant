"""
Aggressive Filter Optimization to reach Rs 300-400/trade target
Goal: 4-5 trades/day with Rs 300-400 avg net per trade
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
                try:
                    ts = decision.get("decision_ts", "")
                    if ts and len(ts) >= 16:
                        entry_hour = int(ts[11:13])
                except:
                    pass

                exit_reason = final_exit.get("reason", "unknown")
                pnl = final_exit.get("total_trade_pnl", 0)
                exit_price = final_exit.get("exit_price", 0)
                entry_price = final_exit.get("actual_entry_price", decision["entry_price"])
                qty = final_exit.get("qty", decision["qty"])

                charges = calculate_charges(entry_price, exit_price, qty, decision["direction"])

                trades.append({
                    "setup": decision["setup"],
                    "direction": decision["direction"],
                    "cap_segment": decision["cap_segment"],
                    "regime": decision["regime"],
                    "entry_hour": entry_hour,
                    "gross_pnl": pnl,
                    "charges": charges,
                    "net_pnl": pnl - charges,
                    "adx": decision["adx"],
                    "rsi": decision["rsi"],
                    "rank_score": decision["rank_score"],
                })

    return trades, sessions


def calc_stats(trades_list):
    if not trades_list:
        return {"count": 0, "net_pnl": 0, "avg_net": 0, "win_rate": 0}
    winners = [t for t in trades_list if t["net_pnl"] > 0]
    net_pnl = sum(t["net_pnl"] for t in trades_list)
    return {
        "count": len(trades_list),
        "net_pnl": net_pnl,
        "avg_net": net_pnl / len(trades_list),
        "win_rate": len(winners) / len(trades_list) * 100,
    }


def main():
    print("=" * 120)
    print("AGGRESSIVE FILTER OPTIMIZATION - TARGET Rs 300-400/trade")
    print("=" * 120)

    trades, sessions = load_all_trades()
    print(f"\nLoaded {len(trades):,} trades from {sessions} sessions")
    print(f"Target: 4-5 trades/day = {4*sessions:,} to {5*sessions:,} trades")
    print(f"Target avg PnL: Rs 300-400/trade")

    # Focus on the 2 main setups that drive most volume
    range_bounce_short = [t for t in trades if t["setup"] == "range_bounce_short"
                          and t["cap_segment"] in ["micro_cap", "small_cap"]]
    resistance_bounce_short = [t for t in trades if t["setup"] == "resistance_bounce_short"
                               and t["cap_segment"] in ["micro_cap", "small_cap"]]
    volume_spike = [t for t in trades if t["setup"] == "volume_spike_reversal_long"
                    and t["regime"] != "chop"]
    squeeze = [t for t in trades if t["setup"] == "squeeze_release_long"]
    order_block = [t for t in trades if t["setup"] == "order_block_short"]
    fvg = [t for t in trades if t["setup"] == "fair_value_gap_short"]

    print(f"\n{'='*80}")
    print("CURRENT BASELINE (with cap filters only)")
    print(f"{'='*80}")

    base = range_bounce_short + resistance_bounce_short + volume_spike + squeeze + order_block + fvg
    base_stats = calc_stats(base)
    print(f"Total: {base_stats['count']:,} trades, Rs {base_stats['avg_net']:.0f}/trade")

    # Analyze each setup's high-value segments
    print(f"\n{'='*80}")
    print("DEEP DIVE: RANGE_BOUNCE_SHORT")
    print(f"{'='*80}")

    rbs_stats = calc_stats(range_bounce_short)
    print(f"Base: {rbs_stats['count']:,} trades, Rs {rbs_stats['avg_net']:.0f}/trade")

    # Combination filters
    print("\nCombination filter analysis:")
    print(f"{'Filter Combo':<60} {'Trades':>8} {'Avg Net':>10} {'Net PnL':>15}")
    print("-" * 95)

    combos = [
        ("micro_cap only", lambda t: t["cap_segment"] == "micro_cap"),
        ("micro_cap + rank >= 1.0", lambda t: t["cap_segment"] == "micro_cap" and t["rank_score"] >= 1.0),
        ("micro_cap + rank >= 1.5", lambda t: t["cap_segment"] == "micro_cap" and t["rank_score"] >= 1.5),
        ("micro_cap + RSI 50-70", lambda t: t["cap_segment"] == "micro_cap" and 50 <= t["rsi"] < 70),
        ("micro_cap + hour 11-12", lambda t: t["cap_segment"] == "micro_cap" and t["entry_hour"] in [11, 12]),
        ("rank >= 1.5", lambda t: t["rank_score"] >= 1.5),
        ("rank >= 2.0", lambda t: t["rank_score"] >= 2.0),
        ("rank >= 1.5 + hour 11-13", lambda t: t["rank_score"] >= 1.5 and t["entry_hour"] in [11, 12, 13]),
        ("RSI 50-65", lambda t: 50 <= t["rsi"] < 65),
        ("RSI 50-60 + rank >= 1.0", lambda t: 50 <= t["rsi"] < 60 and t["rank_score"] >= 1.0),
        ("hour 11-12 + rank >= 1.0", lambda t: t["entry_hour"] in [11, 12] and t["rank_score"] >= 1.0),
        ("BEST: micro + rank >= 1.5 + RSI 50-70", lambda t: t["cap_segment"] == "micro_cap" and t["rank_score"] >= 1.5 and 50 <= t["rsi"] < 70),
    ]

    for name, filter_func in combos:
        filtered = [t for t in range_bounce_short if filter_func(t)]
        s = calc_stats(filtered)
        if s["count"] > 0:
            print(f"{name:<60} {s['count']:>8,} Rs {s['avg_net']:>7,.0f} Rs {s['net_pnl']:>12,.0f}")

    print(f"\n{'='*80}")
    print("DEEP DIVE: RESISTANCE_BOUNCE_SHORT")
    print(f"{'='*80}")

    res_stats = calc_stats(resistance_bounce_short)
    print(f"Base: {res_stats['count']:,} trades, Rs {res_stats['avg_net']:.0f}/trade")

    print("\nCombination filter analysis:")
    print(f"{'Filter Combo':<60} {'Trades':>8} {'Avg Net':>10} {'Net PnL':>15}")
    print("-" * 95)

    for name, filter_func in combos:
        filtered = [t for t in resistance_bounce_short if filter_func(t)]
        s = calc_stats(filtered)
        if s["count"] > 0:
            print(f"{name:<60} {s['count']:>8,} Rs {s['avg_net']:>7,.0f} Rs {s['net_pnl']:>12,.0f}")

    # Additional filter: block 14:00
    res_no_14 = [t for t in resistance_bounce_short if t["entry_hour"] != 14]
    res_no_14_stats = calc_stats(res_no_14)
    print(f"{'block hour 14':<60} {res_no_14_stats['count']:>8,} Rs {res_no_14_stats['avg_net']:>7,.0f} Rs {res_no_14_stats['net_pnl']:>12,.0f}")

    # Block RSI > 70
    res_rsi_cap = [t for t in resistance_bounce_short if t["rsi"] <= 70]
    res_rsi_stats = calc_stats(res_rsi_cap)
    print(f"{'block RSI > 70':<60} {res_rsi_stats['count']:>8,} Rs {res_rsi_stats['avg_net']:>7,.0f} Rs {res_rsi_stats['net_pnl']:>12,.0f}")

    # Combined best filters
    res_optimized = [t for t in resistance_bounce_short if t["entry_hour"] != 14 and t["rsi"] <= 70]
    res_opt_stats = calc_stats(res_optimized)
    print(f"{'block hour 14 + RSI > 70':<60} {res_opt_stats['count']:>8,} Rs {res_opt_stats['avg_net']:>7,.0f} Rs {res_opt_stats['net_pnl']:>12,.0f}")

    print(f"\n{'='*80}")
    print("FINAL OPTIMIZATION SCENARIOS")
    print(f"{'='*80}")

    # Create several optimization scenarios
    scenarios = []

    # Scenario 1: Conservative - minor filters
    s1_rbs = [t for t in range_bounce_short if t["rank_score"] >= 1.5]
    s1_res = [t for t in resistance_bounce_short if t["entry_hour"] != 14 and t["rsi"] <= 70]
    s1_vol = [t for t in volume_spike if t["entry_hour"] != 10 and t["adx"] >= 20]
    s1_sqz = [t for t in squeeze if t["regime"] != "trend_down" and t["cap_segment"] != "small_cap"]
    s1_ob = [t for t in order_block if t["cap_segment"] != "mid_cap" and t["adx"] >= 15]
    s1 = s1_rbs + s1_res + s1_vol + s1_sqz + s1_ob + fvg
    scenarios.append(("Conservative", s1))

    # Scenario 2: Moderate - rank >= 1.5 on all
    s2_rbs = [t for t in range_bounce_short if t["rank_score"] >= 1.5]
    s2_res = [t for t in resistance_bounce_short if t["rank_score"] >= 1.5 and t["entry_hour"] != 14 and t["rsi"] <= 70]
    s2_vol = [t for t in volume_spike if t["entry_hour"] != 10 and t["adx"] >= 20]
    s2_sqz = [t for t in squeeze if t["regime"] != "trend_down" and t["cap_segment"] != "small_cap"]
    s2_ob = [t for t in order_block if t["cap_segment"] != "mid_cap" and t["adx"] >= 15]
    s2 = s2_rbs + s2_res + s2_vol + s2_sqz + s2_ob + fvg
    scenarios.append(("Moderate (rank>=1.5)", s2))

    # Scenario 3: Aggressive - rank >= 2.0 on main, micro only for rbs
    s3_rbs = [t for t in range_bounce_short if t["cap_segment"] == "micro_cap" and t["rank_score"] >= 1.5]
    s3_res = [t for t in resistance_bounce_short if t["rank_score"] >= 1.5 and t["entry_hour"] != 14 and t["rsi"] <= 70]
    s3_vol = [t for t in volume_spike if t["entry_hour"] != 10 and t["adx"] >= 20]
    s3_sqz = squeeze  # keep all
    s3_ob = order_block  # keep all
    s3 = s3_rbs + s3_res + s3_vol + s3_sqz + s3_ob + fvg
    scenarios.append(("Aggressive (micro_cap focus)", s3))

    # Scenario 4: Ultra-aggressive - high rank everywhere
    s4_rbs = [t for t in range_bounce_short if t["rank_score"] >= 2.0]
    s4_res = [t for t in resistance_bounce_short if t["rank_score"] >= 2.0]
    s4_vol = [t for t in volume_spike if t["rank_score"] >= 1.0]
    s4_sqz = squeeze
    s4_ob = order_block
    s4 = s4_rbs + s4_res + s4_vol + s4_sqz + s4_ob + fvg
    scenarios.append(("Ultra (rank>=2.0 on main)", s4))

    # Scenario 5: Only high-performers (volume_spike, squeeze, order_block)
    s5 = volume_spike + squeeze + order_block + fvg
    scenarios.append(("High-performers only", s5))

    # Scenario 6: Target exactly Rs 300+/trade
    s6_rbs = [t for t in range_bounce_short if t["cap_segment"] == "micro_cap" and t["rank_score"] >= 2.0 and 50 <= t["rsi"] < 70]
    s6_res = [t for t in resistance_bounce_short if t["cap_segment"] == "micro_cap" and t["rank_score"] >= 1.5]
    s6_vol = [t for t in volume_spike if t["entry_hour"] != 10 and t["cap_segment"] == "micro_cap"]
    s6_sqz = [t for t in squeeze if t["cap_segment"] in ["mid_cap", "large_cap"]]
    s6_ob = order_block
    s6 = s6_rbs + s6_res + s6_vol + s6_sqz + s6_ob + fvg
    scenarios.append(("Target Rs 300+", s6))

    print(f"\n{'Scenario':<35} {'Trades':>10} {'Trades/Day':>12} {'Net PnL':>15} {'Avg Net':>10} {'Win%':>8}")
    print("-" * 95)

    for name, trades_list in scenarios:
        s = calc_stats(trades_list)
        print(f"{name:<35} {s['count']:>10,} {s['count']/sessions:>11.1f} Rs {s['net_pnl']:>12,.0f} Rs {s['avg_net']:>7,.0f} {s['win_rate']:>7.1f}%")

    # Breakdown for best scenarios
    print(f"\n{'='*80}")
    print("SCENARIO BREAKDOWN")
    print(f"{'='*80}")

    for name, trades_list in scenarios:
        s = calc_stats(trades_list)
        if 4 <= s["count"]/sessions <= 8:  # Reasonable range
            print(f"\n>>> {name} ({s['count']/sessions:.1f} trades/day, Rs {s['avg_net']:.0f}/trade)")

            by_setup = defaultdict(list)
            for t in trades_list:
                by_setup[t["setup"]].append(t)

            for setup, setup_trades in sorted(by_setup.items(), key=lambda x: -calc_stats(x[1])["net_pnl"]):
                ss = calc_stats(setup_trades)
                print(f"    {setup:<35} {ss['count']:>6,} trades, Rs {ss['avg_net']:>6,.0f}/trade, Rs {ss['net_pnl']:>10,.0f} total")

    # Financial summary for best option
    print(f"\n{'='*80}")
    print("FINANCIAL PROJECTIONS (Rs 3L Capital, 3 Years)")
    print(f"{'='*80}")

    capital = 300000
    years = 3

    print(f"\n{'Scenario':<35} {'Net PnL':>15} {'Tax (31.2%)':>15} {'Post-Tax':>15} {'Annual ROI':>12}")
    print("-" * 95)

    for name, trades_list in scenarios:
        s = calc_stats(trades_list)
        tax = max(0, s["net_pnl"] * TAX_RATE)
        post_tax = s["net_pnl"] - tax
        annual_roi = (post_tax / years / capital) * 100

        print(f"{name:<35} Rs {s['net_pnl']:>12,.0f} Rs {tax:>12,.0f} Rs {post_tax:>12,.0f} {annual_roi:>11.1f}%")


if __name__ == "__main__":
    main()
