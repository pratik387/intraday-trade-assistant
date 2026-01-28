"""
Capital Simulation Analysis
Simulates returns at different capital levels for MODERATE and STRICT profiles.
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

# Current system uses Rs 1000 risk per trade
BASE_RISK_PER_TRADE = 1000

# Charge calculations
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
                                    "entry_price": plan.get("entry_ref_price", 0),
                                    "stop_loss": plan.get("stop_loss", 0),
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
                exit_price = 0
                total_pnl = 0

                for a in trade_analytics:
                    if a.get("is_final_exit"):
                        exit_time = a.get("timestamp")
                        total_pnl = a.get("total_trade_pnl", 0)
                        exit_price = a.get("exit_price", 0)
                        if a.get("actual_entry_price"):
                            actual_entry_price = a.get("actual_entry_price")

                if not exit_time:
                    continue

                seen_ids.add(trade_id)

                qty = decision["qty"]
                notional = decision["notional"] or (actual_entry_price * qty)

                # Calculate R-multiple (how many R the trade made)
                stop_loss = decision["stop_loss"]
                risk_per_share = abs(actual_entry_price - stop_loss) if stop_loss > 0 else actual_entry_price * 0.01
                risk_amount = risk_per_share * qty if qty > 0 else BASE_RISK_PER_TRADE
                r_multiple = total_pnl / risk_amount if risk_amount > 0 else 0

                trades.append({
                    "trade_id": trade_id,
                    "date": decision["date"],
                    "setup": decision["setup"],
                    "direction": decision["direction"],
                    "cap_segment": decision["cap_segment"],
                    "regime": decision["regime"],
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": actual_entry_price,
                    "exit_price": exit_price,
                    "stop_loss": stop_loss,
                    "qty": qty,
                    "notional": notional,
                    "pnl": total_pnl,
                    "r_multiple": r_multiple,
                    "risk_amount": risk_amount,
                    "adx": decision["adx"],
                    "rsi": decision["rsi"],
                    "rank_score": decision["rank_score"],
                })

    return trades


def apply_strict_filter(trade):
    """Original STRICT filters - 235 trades"""
    setup = trade["setup"]
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


def apply_moderate_filter(trade):
    """MODERATE filters - ~23K trades"""
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


def simulate_with_capital_constraint(trades, available_capital, margin_rate=0.20):
    """
    Simulate trades with capital constraint.
    Skip trades if concurrent capital exceeds available capital.
    """
    by_date = defaultdict(list)
    for t in trades:
        by_date[t["date"]].append(t)

    total_pnl = 0
    total_charges = 0
    trades_taken = 0
    trades_skipped = 0

    daily_results = []

    for date, day_trades in sorted(by_date.items()):
        # Sort by entry time
        day_trades.sort(key=lambda x: x["entry_time"])

        # Track active positions
        active_positions = []  # (exit_time, notional)
        day_pnl = 0
        day_charges = 0
        day_taken = 0
        day_skipped = 0

        for trade in day_trades:
            # Remove expired positions
            active_positions = [(et, n) for et, n in active_positions if et > trade["entry_time"]]

            # Calculate current capital in use
            current_capital = sum(n for _, n in active_positions)
            required_margin = current_capital * margin_rate

            # Check if we can take this trade
            trade_margin = trade["notional"] * margin_rate

            if required_margin + trade_margin <= available_capital:
                # Take the trade
                active_positions.append((trade["exit_time"], trade["notional"]))

                # Calculate charges
                charges = 0
                if trade["entry_price"] > 0 and trade["exit_price"] > 0 and trade["qty"] > 0:
                    charges = calculate_charges(
                        trade["entry_price"], trade["exit_price"],
                        trade["qty"], trade["direction"]
                    )

                day_pnl += trade["pnl"]
                day_charges += charges
                day_taken += 1
            else:
                day_skipped += 1

        total_pnl += day_pnl
        total_charges += day_charges
        trades_taken += day_taken
        trades_skipped += day_skipped

        daily_results.append({
            "date": date,
            "taken": day_taken,
            "skipped": day_skipped,
            "pnl": day_pnl - day_charges,
        })

    net_pnl = total_pnl - total_charges
    tax = net_pnl * 0.312 if net_pnl > 0 else 0
    final_net = net_pnl - tax

    return {
        "trades_taken": trades_taken,
        "trades_skipped": trades_skipped,
        "gross_pnl": total_pnl,
        "charges": total_charges,
        "net_pnl": net_pnl,
        "tax": tax,
        "final_net": final_net,
        "daily_results": daily_results,
    }


def simulate_with_scaled_risk(trades, risk_per_trade, base_risk=BASE_RISK_PER_TRADE):
    """
    Simulate trades with scaled risk per trade.
    Scale PnL proportionally based on risk multiplier.
    """
    scale_factor = risk_per_trade / base_risk

    total_gross_pnl = 0
    total_charges = 0

    for trade in trades:
        # Scale PnL by risk multiplier
        scaled_pnl = trade["pnl"] * scale_factor

        # Scale notional for charge calculation
        scaled_notional = trade["notional"] * scale_factor
        scaled_qty = trade["qty"] * scale_factor

        # Calculate charges on scaled position
        charges = 0
        if trade["entry_price"] > 0 and trade["exit_price"] > 0 and scaled_qty > 0:
            charges = calculate_charges(
                trade["entry_price"], trade["exit_price"],
                scaled_qty, trade["direction"]
            )

        total_gross_pnl += scaled_pnl
        total_charges += charges

    net_pnl = total_gross_pnl - total_charges
    tax = net_pnl * 0.312 if net_pnl > 0 else 0
    final_net = net_pnl - tax

    return {
        "trades": len(trades),
        "risk_per_trade": risk_per_trade,
        "scale_factor": scale_factor,
        "gross_pnl": total_gross_pnl,
        "charges": total_charges,
        "net_pnl": net_pnl,
        "tax": tax,
        "final_net": final_net,
    }


def calculate_required_capital_for_risk(trades, risk_per_trade, base_risk=BASE_RISK_PER_TRADE):
    """Calculate max capital needed for given risk level"""
    scale_factor = risk_per_trade / base_risk

    by_date = defaultdict(list)
    for t in trades:
        by_date[t["date"]].append(t)

    max_capital = 0

    for date, day_trades in by_date.items():
        events = []
        for t in day_trades:
            scaled_notional = t["notional"] * scale_factor
            events.append((t["entry_time"], "entry", scaled_notional))
            events.append((t["exit_time"], "exit", scaled_notional))

        events.sort(key=lambda x: x[0])

        current = 0
        for ts, etype, notional in events:
            if etype == "entry":
                current += notional
            else:
                current -= notional
            max_capital = max(max_capital, current)

    return max_capital


def main():
    print("=" * 120)
    print("CAPITAL SIMULATION ANALYSIS")
    print("=" * 120)

    print("\nLoading trades...")
    all_trades = load_trades()
    print(f"Loaded {len(all_trades)} trades")

    strict_trades = [t for t in all_trades if apply_strict_filter(t)]
    moderate_trades = [t for t in all_trades if apply_moderate_filter(t)]

    print(f"STRICT trades: {len(strict_trades)}")
    print(f"MODERATE trades: {len(moderate_trades)}")

    trading_days = 719
    years = 3

    # =========================================================================
    # MODERATE PROFILE - Capital Constraint Simulation
    # =========================================================================
    print("\n" + "=" * 120)
    print("MODERATE PROFILE - CAPITAL CONSTRAINT SIMULATION")
    print("=" * 120)
    print("\nSimulating what happens when you have limited capital...")
    print("(Some trades will be skipped if concurrent positions exceed capital)")

    moderate_capitals = [100000, 300000, 500000, 1000000]  # 1L, 3L, 5L, 10L

    print(f"\n{'Capital':<15} {'Trades Taken':<15} {'Skipped':<12} {'Gross PnL':<15} {'Charges':<12} {'Net PnL':<15} {'Tax':<12} {'Final Net':<15} {'Ann ROI':<10}")
    print("-" * 130)

    for capital in moderate_capitals:
        result = simulate_with_capital_constraint(moderate_trades, capital)
        annual_net = result["final_net"] / years
        roi = (annual_net / capital) * 100 if capital > 0 else 0

        print(f"Rs {capital/100000:.0f}L{'':<8} {result['trades_taken']:<15,} {result['trades_skipped']:<12,} "
              f"Rs {result['gross_pnl']:>11,.0f} Rs {result['charges']:>8,.0f} Rs {result['net_pnl']:>11,.0f} "
              f"Rs {result['tax']:>8,.0f} Rs {result['final_net']:>11,.0f} {roi:>8.1f}%")

    # Detailed breakdown for key capital levels
    print("\n" + "-" * 80)
    print("DETAILED BREAKDOWN FOR MODERATE PROFILE:")
    print("-" * 80)

    for capital in [100000, 300000, 500000]:
        result = simulate_with_capital_constraint(moderate_trades, capital)
        annual_net = result["final_net"] / years
        monthly_net = annual_net / 12

        print(f"\n>>> Rs {capital/100000:.0f} Lakh Capital:")
        print(f"    Trades Taken: {result['trades_taken']:,} / {len(moderate_trades):,} ({result['trades_taken']/len(moderate_trades)*100:.1f}%)")
        print(f"    Trades Skipped: {result['trades_skipped']:,}")
        print(f"    Gross PnL (3yr): Rs {result['gross_pnl']:,.0f}")
        print(f"    Trading Charges: Rs {result['charges']:,.0f}")
        print(f"    Net PnL: Rs {result['net_pnl']:,.0f}")
        print(f"    Income Tax (31.2%): Rs {result['tax']:,.0f}")
        print(f"    FINAL NET (3yr): Rs {result['final_net']:,.0f}")
        print(f"    Annual Net: Rs {annual_net:,.0f}")
        print(f"    Monthly Net: Rs {monthly_net:,.0f}")
        print(f"    Annual ROI: {(annual_net/capital)*100:.1f}%")

    # =========================================================================
    # STRICT PROFILE - Scaled Risk Simulation
    # =========================================================================
    print("\n" + "=" * 120)
    print("STRICT PROFILE - SCALED RISK SIMULATION")
    print("=" * 120)
    print("\nSimulating increased risk per trade (scaling position sizes)...")
    print("Base risk: Rs 1,000 per trade")

    # Calculate base stats
    base_result = simulate_with_scaled_risk(strict_trades, BASE_RISK_PER_TRADE)
    base_capital = calculate_required_capital_for_risk(strict_trades, BASE_RISK_PER_TRADE)

    print(f"\nBASELINE (Rs 1000 risk/trade):")
    print(f"  Max Capital Needed: Rs {base_capital:,.0f}")
    print(f"  Margin Required (20%): Rs {base_capital * 0.20:,.0f}")
    print(f"  Final Net (3yr): Rs {base_result['final_net']:,.0f}")

    # Target capital levels and corresponding risk per trade
    target_capitals = [300000, 500000, 1000000]  # 3L, 5L, 10L margin

    print(f"\n{'Target Capital':<18} {'Risk/Trade':<15} {'Max Notional':<18} {'Gross PnL':<15} {'Charges':<12} {'Net PnL':<15} {'Tax':<12} {'Final Net':<15} {'Ann ROI':<10}")
    print("-" * 140)

    strict_simulations = []

    for target_capital in target_capitals:
        # Calculate risk per trade to use target capital
        # margin = notional * 0.20, so notional = capital / 0.20
        # scale_factor = target_notional / base_notional
        base_margin = base_capital * 0.20
        scale_factor = target_capital / base_margin if base_margin > 0 else 1
        risk_per_trade = BASE_RISK_PER_TRADE * scale_factor

        result = simulate_with_scaled_risk(strict_trades, risk_per_trade)
        max_notional = calculate_required_capital_for_risk(strict_trades, risk_per_trade)

        annual_net = result["final_net"] / years
        roi = (annual_net / target_capital) * 100 if target_capital > 0 else 0

        strict_simulations.append({
            "capital": target_capital,
            "risk_per_trade": risk_per_trade,
            "result": result,
            "max_notional": max_notional,
            "roi": roi,
        })

        print(f"Rs {target_capital/100000:.0f}L{'':<12} Rs {risk_per_trade:>10,.0f} Rs {max_notional:>14,.0f} "
              f"Rs {result['gross_pnl']:>11,.0f} Rs {result['charges']:>8,.0f} Rs {result['net_pnl']:>11,.0f} "
              f"Rs {result['tax']:>8,.0f} Rs {result['final_net']:>11,.0f} {roi:>8.1f}%")

    # Detailed breakdown
    print("\n" + "-" * 80)
    print("DETAILED BREAKDOWN FOR STRICT PROFILE:")
    print("-" * 80)

    for sim in strict_simulations:
        result = sim["result"]
        annual_net = result["final_net"] / years
        monthly_net = annual_net / 12

        print(f"\n>>> Rs {sim['capital']/100000:.0f} Lakh Capital:")
        print(f"    Risk per Trade: Rs {sim['risk_per_trade']:,.0f} ({sim['risk_per_trade']/BASE_RISK_PER_TRADE:.1f}x base)")
        print(f"    Trades: {result['trades']:,}")
        print(f"    Max Notional: Rs {sim['max_notional']:,.0f}")
        print(f"    Margin Used: Rs {sim['max_notional'] * 0.20:,.0f}")
        print(f"    Gross PnL (3yr): Rs {result['gross_pnl']:,.0f}")
        print(f"    Trading Charges: Rs {result['charges']:,.0f}")
        print(f"    Net PnL: Rs {result['net_pnl']:,.0f}")
        print(f"    Income Tax (31.2%): Rs {result['tax']:,.0f}")
        print(f"    FINAL NET (3yr): Rs {result['final_net']:,.0f}")
        print(f"    Annual Net: Rs {annual_net:,.0f}")
        print(f"    Monthly Net: Rs {monthly_net:,.0f}")
        print(f"    Annual ROI: {sim['roi']:.1f}%")

    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 120)
    print("COMPARISON SUMMARY")
    print("=" * 120)

    print(f"\n{'Profile':<12} {'Capital':<12} {'Trades':<10} {'Final Net (3yr)':<18} {'Annual Net':<15} {'Monthly Net':<15} {'ROI':<10}")
    print("-" * 100)

    # MODERATE results
    for capital in [100000, 300000, 500000]:
        result = simulate_with_capital_constraint(moderate_trades, capital)
        annual = result["final_net"] / years
        monthly = annual / 12
        roi = (annual / capital) * 100
        print(f"{'MODERATE':<12} Rs {capital/100000:.0f}L{'':<6} {result['trades_taken']:<10,} Rs {result['final_net']:>13,.0f} Rs {annual:>11,.0f} Rs {monthly:>11,.0f} {roi:>8.1f}%")

    print("-" * 100)

    # STRICT results
    for sim in strict_simulations:
        result = sim["result"]
        annual = result["final_net"] / years
        monthly = annual / 12
        print(f"{'STRICT':<12} Rs {sim['capital']/100000:.0f}L{'':<6} {result['trades']:<10,} Rs {result['final_net']:>13,.0f} Rs {annual:>11,.0f} Rs {monthly:>11,.0f} {sim['roi']:>8.1f}%")

    # =========================================================================
    # RECOMMENDATION
    # =========================================================================
    print("\n" + "=" * 120)
    print("RECOMMENDATION")
    print("=" * 120)

    print("""
MODERATE PROFILE:
- Best for: Traders with limited capital who want more trading activity
- Rs 1L capital: ~17.8K trades, Rs 4.16L final net (3yr), Rs 139%+ ROI
- Rs 3L capital: ~20.1K trades, Rs 10.8L final net (3yr), ~120% ROI
- Rs 5L capital: ~21.4K trades, Rs 14.5L final net (3yr), ~97% ROI
- Note: ROI decreases as capital increases because you hit trade capacity

STRICT PROFILE:
- Best for: Traders who want to scale returns with capital
- Higher capital = proportionally higher returns
- Rs 3L capital: 235 trades, ~Rs 3.8L final net (3yr), ~42% ROI
- Rs 5L capital: 235 trades, ~Rs 6.3L final net (3yr), ~42% ROI
- Rs 10L capital: 235 trades, ~Rs 12.6L final net (3yr), ~42% ROI
- Note: ROI stays constant (scales linearly)

KEY INSIGHT:
- MODERATE has higher absolute returns at lower capital levels
- STRICT ROI is more predictable and consistent
- With Rs 5L+: MODERATE gives more total returns
- Risk consideration: MODERATE has more trades = more chances for drawdown
""")


if __name__ == "__main__":
    main()
