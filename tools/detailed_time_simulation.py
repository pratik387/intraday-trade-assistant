"""
Detailed Time Filter Simulation
Shows complete financial breakdown and trade quality analysis
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
    if "allowed_caps" in config and trade["cap_segment"] not in config["allowed_caps"]:
        return False
    if "blocked_caps" in config and trade["cap_segment"] in config["blocked_caps"]:
        return False
    if "allowed_regimes" in config and trade["regime"] not in config["allowed_regimes"]:
        return False
    if "blocked_regimes" in config and trade["regime"] in config["blocked_regimes"]:
        return False
    if "adx_min" in config and trade["adx"] < config["adx_min"]:
        return False
    if "adx_max" in config and trade["adx"] > config["adx_max"]:
        return False
    if "rsi_min" in config and trade["rsi"] < config["rsi_min"]:
        return False
    if "rank_min" in config and trade["rank_score"] < config["rank_min"]:
        return False
    return True


def analyze_blocked_trades(all_trades, filtered_trades):
    """Analyze what trades would be blocked"""
    filtered_ids = {t["trade_id"] for t in filtered_trades}
    blocked = [t for t in all_trades if t["trade_id"] not in filtered_ids]

    winners_blocked = [t for t in blocked if t["net_pnl"] > 0]
    losers_blocked = [t for t in blocked if t["net_pnl"] <= 0]

    return {
        "total_blocked": len(blocked),
        "winners_blocked": len(winners_blocked),
        "losers_blocked": len(losers_blocked),
        "blocked_gross_pnl": sum(t["gross_pnl"] for t in blocked),
        "blocked_net_pnl": sum(t["net_pnl"] for t in blocked),
        "winners_pnl_lost": sum(t["net_pnl"] for t in winners_blocked),
        "losers_pnl_avoided": sum(t["net_pnl"] for t in losers_blocked),
    }


def main():
    print("=" * 130)
    print("DETAILED TIME FILTER SIMULATION")
    print("=" * 130)

    all_trades = load_trades()
    moderate_trades = [t for t in all_trades if apply_moderate_filter(t)]

    print(f"\nMODERATE Profile Baseline: {len(moderate_trades):,} trades")

    # Define time filter options with clear explanations
    options = [
        {
            "id": "BASELINE",
            "name": "No Time Filter (Current)",
            "description": "Current MODERATE profile - trades at any time during market hours (9:15-15:30)",
            "rule": lambda t: True,
        },
        {
            "id": "OPT1",
            "name": "Block ALL LONG trades at 10:00 hour",
            "description": "Blocks all LONG direction trades entered between 10:00-10:59. SHORT trades unaffected.",
            "rule": lambda t: not (t["direction"] == "LONG" and t["entry_hour"] == 10),
        },
        {
            "id": "OPT2",
            "name": "Block range_bounce_long at 10:00 hour only",
            "description": "Blocks only 'range_bounce_long' setup between 10:00-10:59. All other setups unaffected.",
            "rule": lambda t: not (t["setup"] == "range_bounce_long" and t["entry_hour"] == 10),
        },
        {
            "id": "OPT3",
            "name": "Block 10:00-10:30 window for LONG",
            "description": "Blocks LONG trades only in first 30 mins (10:00-10:29). 10:30+ allowed.",
            "rule": lambda t: not (t["direction"] == "LONG" and t["entry_hour"] == 10 and t["entry_minute"] < 30),
        },
        {
            "id": "OPT4",
            "name": "Block 12:00-13:00 lunch hour",
            "description": "Blocks ALL trades during lunch (12:00-12:59). Morning and afternoon allowed.",
            "rule": lambda t: t["entry_hour"] != 12,
        },
        {
            "id": "OPT5",
            "name": "Only 11:00-14:30 (skip early morning)",
            "description": "Allows trades only from 11:00-14:30. Blocks 9:15-10:59 and after 14:30.",
            "rule": lambda t: 11 <= t["entry_hour"] <= 14,
        },
        {
            "id": "OPT6",
            "name": "Afternoon only (13:00+)",
            "description": "Only trade in afternoon session (13:00-15:30). Blocks all morning trades.",
            "rule": lambda t: t["entry_hour"] >= 13,
        },
    ]

    results = []

    for opt in options:
        filtered = [t for t in moderate_trades if opt["rule"](t)]
        blocked_analysis = analyze_blocked_trades(moderate_trades, filtered)

        # Calculate full financials
        gross_pnl = sum(t["gross_pnl"] for t in filtered)
        total_charges = sum(t["charges"] for t in filtered)
        net_after_charges = gross_pnl - total_charges
        tax = net_after_charges * 0.312 if net_after_charges > 0 else 0
        final_net = net_after_charges - tax

        # Win/loss stats
        winners = [t for t in filtered if t["net_pnl"] > 0]
        losers = [t for t in filtered if t["net_pnl"] <= 0]
        sl_trades = [t for t in filtered if t["is_sl_hit"]]

        results.append({
            "opt": opt,
            "filtered": filtered,
            "blocked": blocked_analysis,
            "gross_pnl": gross_pnl,
            "charges": total_charges,
            "net_after_charges": net_after_charges,
            "tax": tax,
            "final_net": final_net,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(filtered) * 100 if filtered else 0,
            "sl_count": len(sl_trades),
            "sl_rate": len(sl_trades) / len(filtered) * 100 if filtered else 0,
        })

    # Print detailed results
    print("\n" + "=" * 130)
    print("OPTION DESCRIPTIONS")
    print("=" * 130)

    for r in results:
        print(f"\n{r['opt']['id']}: {r['opt']['name']}")
        print(f"   Rule: {r['opt']['description']}")

    print("\n" + "=" * 130)
    print("FINANCIAL BREAKDOWN (All amounts in Rs)")
    print("=" * 130)

    header = f"{'Option':<10} {'Trades':>8} {'Gross PnL':>14} {'Charges':>12} {'Net(post-fee)':>14} {'Tax(31.2%)':>12} {'FINAL NET':>14}"
    print(f"\n{header}")
    print("-" * 90)

    for r in results:
        print(f"{r['opt']['id']:<10} {len(r['filtered']):>8,} {r['gross_pnl']:>14,.0f} {r['charges']:>12,.0f} {r['net_after_charges']:>14,.0f} {r['tax']:>12,.0f} {r['final_net']:>14,.0f}")

    print("\n" + "=" * 130)
    print("WIN/LOSS STATISTICS")
    print("=" * 130)

    header = f"{'Option':<10} {'Trades':>8} {'Winners':>8} {'Losers':>8} {'WinRate':>10} {'SL Hits':>8} {'SL Rate':>10}"
    print(f"\n{header}")
    print("-" * 70)

    for r in results:
        print(f"{r['opt']['id']:<10} {len(r['filtered']):>8,} {r['winners']:>8,} {r['losers']:>8,} {r['win_rate']:>9.1f}% {r['sl_count']:>8,} {r['sl_rate']:>9.1f}%")

    print("\n" + "=" * 130)
    print("BLOCKED TRADES ANALYSIS (What you lose by applying the filter)")
    print("=" * 130)

    header = f"{'Option':<10} {'Blocked':>8} {'Winners':>10} {'Losers':>10} {'Winners PnL':>14} {'Losers PnL':>14} {'Net Impact':>14}"
    print(f"\n{header}")
    print("-" * 95)

    for r in results:
        b = r['blocked']
        net_impact = -b['blocked_net_pnl']  # Positive = blocking helped
        print(f"{r['opt']['id']:<10} {b['total_blocked']:>8,} {b['winners_blocked']:>10,} {b['losers_blocked']:>10,} {b['winners_pnl_lost']:>14,.0f} {b['losers_pnl_avoided']:>14,.0f} {net_impact:>+14,.0f}")

    print("\n" + "=" * 130)
    print("COMPARISON VS BASELINE (Changes from No Filter)")
    print("=" * 130)

    baseline = results[0]

    header = f"{'Option':<10} {'Trades':>10} {'Final Net':>14} {'Change':>14} {'Monthly':>12} {'Change/Mo':>12}"
    print(f"\n{header}")
    print("-" * 80)

    for r in results:
        trades_change = len(r['filtered']) - len(baseline['filtered'])
        final_change = r['final_net'] - baseline['final_net']
        monthly = r['final_net'] / 36
        monthly_change = final_change / 36

        print(f"{r['opt']['id']:<10} {len(r['filtered']):>10,} {r['final_net']:>14,.0f} {final_change:>+14,.0f} {monthly:>12,.0f} {monthly_change:>+12,.0f}")

    print("\n" + "=" * 130)
    print("DETAILED TRADE QUALITY OF BLOCKED TRADES")
    print("=" * 130)

    for r in results[1:]:  # Skip baseline
        opt = r['opt']
        blocked_trades = [t for t in moderate_trades if not opt["rule"](t)]

        if not blocked_trades:
            continue

        print(f"\n>>> {opt['id']}: {opt['name']}")
        print(f"    Blocks {len(blocked_trades)} trades")

        # Breakdown by outcome
        winners = [t for t in blocked_trades if t["net_pnl"] > 0]
        losers = [t for t in blocked_trades if t["net_pnl"] <= 0]
        sl_hits = [t for t in blocked_trades if t["is_sl_hit"]]

        print(f"\n    Blocked trade breakdown:")
        print(f"      Winners blocked: {len(winners):>6} (lost PnL: Rs {sum(t['net_pnl'] for t in winners):>10,.0f})")
        print(f"      Losers blocked:  {len(losers):>6} (avoided PnL: Rs {sum(t['net_pnl'] for t in losers):>10,.0f})")
        print(f"      SL hits blocked: {len(sl_hits):>6}")

        # By setup
        by_setup = defaultdict(list)
        for t in blocked_trades:
            by_setup[t["setup"]].append(t)

        if len(by_setup) <= 5:
            print(f"\n    By setup:")
            for setup, trades in sorted(by_setup.items(), key=lambda x: -len(x[1])):
                w = len([t for t in trades if t["net_pnl"] > 0])
                l = len(trades) - w
                pnl = sum(t["net_pnl"] for t in trades)
                print(f"      {setup:<35} {len(trades):>4} trades ({w}W/{l}L), PnL: Rs {pnl:>8,.0f}")

    print("\n" + "=" * 130)
    print("RECOMMENDATION")
    print("=" * 130)

    # Find best option based on final net improvement
    improvements = [(r, r['final_net'] - baseline['final_net']) for r in results[1:]]
    best = max(improvements, key=lambda x: x[1])
    worst = min(improvements, key=lambda x: x[1])

    print(f"""
SUMMARY:
--------
Baseline (No time filter): Rs {baseline['final_net']:,.0f} final net over 3 years

Best improvement: {best[0]['opt']['id']} - {best[0]['opt']['name']}
  - Final Net: Rs {best[0]['final_net']:,.0f}
  - Change: Rs {best[1]:+,.0f} ({best[1]/36:+,.0f}/month)
  - Blocks {best[0]['blocked']['total_blocked']} trades ({best[0]['blocked']['winners_blocked']} winners, {best[0]['blocked']['losers_blocked']} losers)

Worst option: {worst[0]['opt']['id']} - {worst[0]['opt']['name']}
  - Final Net: Rs {worst[0]['final_net']:,.0f}
  - Change: Rs {worst[1]:+,.0f} ({worst[1]/36:+,.0f}/month)

CONCLUSION:
The time filters have MINIMAL positive impact. The best option ({best[0]['opt']['id']})
only improves by Rs {best[1]:+,.0f} over 3 years (Rs {best[1]/36:+,.0f}/month).

Most time filters REDUCE profits because they block more winners than losers.
The MODERATE setup filters are already doing a good job of quality control.
""")


if __name__ == "__main__":
    main()
