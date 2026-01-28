"""
Comprehensive Setup-by-Setup Filter Analysis
Goal: 4-5 trades/day, profitable setups Rs 300-400/trade, losing setups Rs 100/trade min
"""

import json
from pathlib import Path
from collections import defaultdict
import statistics

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
TAX_RATE = 0.312  # 31.2% tax on profits


def calculate_charges(entry_price, exit_price, qty, direction):
    """Calculate Zerodha charges for a trade."""
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
    """Load all trades from all backtest folders."""
    trades = []
    seen_ids = set()
    sessions_processed = 0

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

            sessions_processed += 1

            # Load decisions
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
                                    "atr_pct": indicators.get("atr_pct", 0),
                                    "rvol": indicators.get("rvol", 1),
                                    "rank_score": ranking.get("score", 0),
                                    "entry_price": plan.get("entry_ref_price", 0),
                                    "qty": sizing.get("qty", 0),
                                    "symbol": event.get("symbol", ""),
                                }
                    except:
                        continue

            # Load analytics
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

            # Match trades
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

                # Parse entry time
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
                mfe = final_exit.get("mfe", 0)
                mae = final_exit.get("mae", 0)

                is_sl_hit = "hard_sl" in exit_reason.lower() or "stop" in exit_reason.lower()
                is_t1_hit = "t1" in exit_reason.lower() or "target" in exit_reason.lower()
                is_t2_hit = "t2" in exit_reason.lower()
                is_t3_hit = "t3" in exit_reason.lower()

                charges = calculate_charges(entry_price, exit_price, qty, decision["direction"])
                net_pnl = pnl - charges

                trades.append({
                    "trade_id": trade_id,
                    "date": decision["date"],
                    "symbol": decision["symbol"],
                    "setup": decision["setup"],
                    "direction": decision["direction"],
                    "cap_segment": decision["cap_segment"],
                    "regime": decision["regime"],
                    "entry_hour": entry_hour,
                    "entry_minute": entry_minute,
                    "exit_reason": exit_reason,
                    "is_sl_hit": is_sl_hit,
                    "is_t1_hit": is_t1_hit,
                    "is_t2_hit": is_t2_hit,
                    "is_t3_hit": is_t3_hit,
                    "gross_pnl": pnl,
                    "charges": charges,
                    "net_pnl": net_pnl,
                    "mfe": mfe,
                    "mae": mae,
                    "adx": decision["adx"],
                    "rsi": decision["rsi"],
                    "atr_pct": decision["atr_pct"],
                    "rvol": decision["rvol"],
                    "rank_score": decision["rank_score"],
                    "entry_price": entry_price,
                    "qty": qty,
                })

    return trades, sessions_processed


def calc_stats(trades_list, label=""):
    """Calculate comprehensive statistics for a list of trades."""
    if not trades_list:
        return {
            "count": 0, "sl_count": 0, "sl_pct": 0, "win_rate": 0,
            "gross_pnl": 0, "charges": 0, "net_pnl": 0,
            "avg_gross": 0, "avg_net": 0, "avg_winner": 0, "avg_loser": 0,
            "profit_factor": 0
        }

    sl_count = sum(1 for t in trades_list if t["is_sl_hit"])
    winners = [t for t in trades_list if t["net_pnl"] > 0]
    losers = [t for t in trades_list if t["net_pnl"] <= 0]

    gross_pnl = sum(t["gross_pnl"] for t in trades_list)
    charges = sum(t["charges"] for t in trades_list)
    net_pnl = sum(t["net_pnl"] for t in trades_list)

    gross_wins = sum(t["net_pnl"] for t in winners) if winners else 0
    gross_losses = abs(sum(t["net_pnl"] for t in losers)) if losers else 1

    return {
        "count": len(trades_list),
        "sl_count": sl_count,
        "sl_pct": sl_count / len(trades_list) * 100 if trades_list else 0,
        "win_rate": len(winners) / len(trades_list) * 100 if trades_list else 0,
        "gross_pnl": gross_pnl,
        "charges": charges,
        "net_pnl": net_pnl,
        "avg_gross": gross_pnl / len(trades_list) if trades_list else 0,
        "avg_net": net_pnl / len(trades_list) if trades_list else 0,
        "avg_winner": sum(t["net_pnl"] for t in winners) / len(winners) if winners else 0,
        "avg_loser": sum(t["net_pnl"] for t in losers) / len(losers) if losers else 0,
        "profit_factor": gross_wins / gross_losses if gross_losses > 0 else 0,
    }


def analyze_by_dimension(trades, dim_name, dim_getter):
    """Analyze trades broken down by a specific dimension."""
    by_dim = defaultdict(list)
    for t in trades:
        val = dim_getter(t)
        by_dim[val].append(t)

    results = []
    for val, val_trades in sorted(by_dim.items(), key=lambda x: -calc_stats(x[1])["net_pnl"]):
        stats = calc_stats(val_trades)
        results.append({
            "value": val,
            "stats": stats,
        })

    return results


def analyze_setup(setup_name, setup_trades, all_trades):
    """Deep analysis of a single setup."""
    print(f"\n{'='*120}")
    print(f"SETUP: {setup_name}")
    print(f"{'='*120}")

    stats = calc_stats(setup_trades)

    print(f"\nOVERALL STATS:")
    print(f"  Trades: {stats['count']:,}")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    print(f"  SL Rate: {stats['sl_pct']:.1f}%")
    print(f"  Gross PnL: Rs {stats['gross_pnl']:,.0f}")
    print(f"  Charges: Rs {stats['charges']:,.0f}")
    print(f"  Net PnL: Rs {stats['net_pnl']:,.0f}")
    print(f"  Avg Net/Trade: Rs {stats['avg_net']:.0f}")
    print(f"  Avg Winner: Rs {stats['avg_winner']:.0f}")
    print(f"  Avg Loser: Rs {stats['avg_loser']:.0f}")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")

    # By Hour
    print(f"\n--- BY HOUR ---")
    print(f"{'Hour':<8} {'Trades':>8} {'WinRate':>8} {'SL%':>8} {'Net PnL':>15} {'Avg Net':>10}")
    print("-" * 60)
    hour_analysis = analyze_by_dimension(setup_trades, "hour", lambda t: t["entry_hour"])
    for r in hour_analysis:
        s = r["stats"]
        print(f"{r['value']:>6}:00 {s['count']:>8,} {s['win_rate']:>7.1f}% {s['sl_pct']:>7.1f}% Rs {s['net_pnl']:>12,.0f} Rs {s['avg_net']:>7,.0f}")

    # By Cap Segment
    print(f"\n--- BY CAP SEGMENT ---")
    print(f"{'Segment':<15} {'Trades':>8} {'WinRate':>8} {'SL%':>8} {'Net PnL':>15} {'Avg Net':>10}")
    print("-" * 70)
    cap_analysis = analyze_by_dimension(setup_trades, "cap", lambda t: t["cap_segment"])
    for r in cap_analysis:
        s = r["stats"]
        print(f"{r['value']:<15} {s['count']:>8,} {s['win_rate']:>7.1f}% {s['sl_pct']:>7.1f}% Rs {s['net_pnl']:>12,.0f} Rs {s['avg_net']:>7,.0f}")

    # By Regime
    print(f"\n--- BY REGIME ---")
    print(f"{'Regime':<15} {'Trades':>8} {'WinRate':>8} {'SL%':>8} {'Net PnL':>15} {'Avg Net':>10}")
    print("-" * 70)
    regime_analysis = analyze_by_dimension(setup_trades, "regime", lambda t: t["regime"])
    for r in regime_analysis:
        s = r["stats"]
        print(f"{r['value']:<15} {s['count']:>8,} {s['win_rate']:>7.1f}% {s['sl_pct']:>7.1f}% Rs {s['net_pnl']:>12,.0f} Rs {s['avg_net']:>7,.0f}")

    # By ADX ranges
    print(f"\n--- BY ADX RANGE ---")
    print(f"{'ADX Range':<15} {'Trades':>8} {'WinRate':>8} {'SL%':>8} {'Net PnL':>15} {'Avg Net':>10}")
    print("-" * 70)

    def get_adx_range(t):
        adx = t["adx"]
        if adx < 15:
            return "0-15 (weak)"
        elif adx < 20:
            return "15-20 (moderate)"
        elif adx < 25:
            return "20-25 (strong)"
        elif adx < 30:
            return "25-30 (v.strong)"
        else:
            return "30+ (extreme)"

    adx_analysis = analyze_by_dimension(setup_trades, "adx", get_adx_range)
    for r in adx_analysis:
        s = r["stats"]
        print(f"{r['value']:<15} {s['count']:>8,} {s['win_rate']:>7.1f}% {s['sl_pct']:>7.1f}% Rs {s['net_pnl']:>12,.0f} Rs {s['avg_net']:>7,.0f}")

    # By RSI ranges
    print(f"\n--- BY RSI RANGE ---")
    print(f"{'RSI Range':<15} {'Trades':>8} {'WinRate':>8} {'SL%':>8} {'Net PnL':>15} {'Avg Net':>10}")
    print("-" * 70)

    def get_rsi_range(t):
        rsi = t["rsi"]
        if rsi < 30:
            return "0-30 (oversold)"
        elif rsi < 40:
            return "30-40"
        elif rsi < 50:
            return "40-50"
        elif rsi < 60:
            return "50-60"
        elif rsi < 70:
            return "60-70"
        else:
            return "70+ (overbought)"

    rsi_analysis = analyze_by_dimension(setup_trades, "rsi", get_rsi_range)
    for r in rsi_analysis:
        s = r["stats"]
        print(f"{r['value']:<15} {s['count']:>8,} {s['win_rate']:>7.1f}% {s['sl_pct']:>7.1f}% Rs {s['net_pnl']:>12,.0f} Rs {s['avg_net']:>7,.0f}")

    # By Rank Score ranges
    print(f"\n--- BY RANK SCORE ---")
    print(f"{'Rank Score':<15} {'Trades':>8} {'WinRate':>8} {'SL%':>8} {'Net PnL':>15} {'Avg Net':>10}")
    print("-" * 70)

    def get_rank_range(t):
        score = t["rank_score"]
        if score < 1.0:
            return "0-1.0 (low)"
        elif score < 1.5:
            return "1.0-1.5"
        elif score < 2.0:
            return "1.5-2.0"
        elif score < 2.5:
            return "2.0-2.5"
        else:
            return "2.5+ (high)"

    rank_analysis = analyze_by_dimension(setup_trades, "rank", get_rank_range)
    for r in rank_analysis:
        s = r["stats"]
        print(f"{r['value']:<15} {s['count']:>8,} {s['win_rate']:>7.1f}% {s['sl_pct']:>7.1f}% Rs {s['net_pnl']:>12,.0f} Rs {s['avg_net']:>7,.0f}")

    # By Exit Type
    print(f"\n--- BY EXIT TYPE ---")
    print(f"{'Exit':<20} {'Trades':>8} {'%':>6} {'Avg Net':>12}")
    print("-" * 50)

    def get_exit_type(t):
        if t["is_sl_hit"]:
            return "STOP_LOSS"
        elif t["is_t3_hit"]:
            return "T3 (full target)"
        elif t["is_t2_hit"]:
            return "T2 (2nd target)"
        elif t["is_t1_hit"]:
            return "T1 (1st target)"
        else:
            return "OTHER"

    exit_analysis = analyze_by_dimension(setup_trades, "exit", get_exit_type)
    for r in exit_analysis:
        s = r["stats"]
        pct = s["count"] / len(setup_trades) * 100
        print(f"{r['value']:<20} {s['count']:>8,} {pct:>5.1f}% Rs {s['avg_net']:>9,.0f}")

    return stats


def simulate_filter(setup_trades, filter_func, filter_name):
    """Simulate applying a filter and show impact."""
    kept = [t for t in setup_trades if filter_func(t)]
    blocked = [t for t in setup_trades if not filter_func(t)]

    kept_stats = calc_stats(kept)
    blocked_stats = calc_stats(blocked)
    baseline_stats = calc_stats(setup_trades)

    print(f"\n>>> FILTER: {filter_name}")
    print(f"    Kept: {kept_stats['count']:,} trades, Avg Net: Rs {kept_stats['avg_net']:.0f}")
    print(f"    Blocked: {blocked_stats['count']:,} trades, Avg Net: Rs {blocked_stats['avg_net']:.0f}")

    if kept_stats["count"] > 0 and blocked_stats["count"] > 0:
        pnl_change = kept_stats["net_pnl"] - baseline_stats["net_pnl"]
        avg_change = kept_stats["avg_net"] - baseline_stats["avg_net"]

        if blocked_stats["avg_net"] < 0:
            recommendation = "APPLY - blocking unprofitable trades"
        elif kept_stats["avg_net"] > baseline_stats["avg_net"] * 1.1:
            recommendation = "CONSIDER - improves avg by 10%+"
        else:
            recommendation = "SKIP - blocks profitable trades"

        print(f"    Impact: PnL {'+' if pnl_change > 0 else ''}{pnl_change:,.0f}, Avg {'+' if avg_change > 0 else ''}{avg_change:.0f}")
        print(f"    Recommendation: {recommendation}")

    return kept_stats, blocked_stats


def main():
    print("=" * 120)
    print("COMPREHENSIVE SETUP-BY-SETUP FILTER ANALYSIS")
    print("=" * 120)
    print("\nLoading all trades from backtest folders...")

    trades, sessions = load_all_trades()

    print(f"\nLoaded {len(trades):,} trades from {sessions} sessions")
    print(f"Average trades/session: {len(trades)/sessions:.1f}")
    print(f"Target: 4-5 trades/day = {4*sessions} to {5*sessions} total trades")

    # Overall stats
    overall_stats = calc_stats(trades)
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*80}")
    print(f"Total Trades: {overall_stats['count']:,}")
    print(f"Net PnL (after charges): Rs {overall_stats['net_pnl']:,.0f}")
    print(f"Post-Tax PnL (31.2%): Rs {overall_stats['net_pnl'] * (1 - TAX_RATE):,.0f}")
    print(f"Avg Net/Trade: Rs {overall_stats['avg_net']:.0f}")

    # Group by setup
    by_setup = defaultdict(list)
    for t in trades:
        by_setup[t["setup"]].append(t)

    # Sort by net PnL descending
    sorted_setups = sorted(by_setup.items(), key=lambda x: -calc_stats(x[1])["net_pnl"])

    print(f"\n{'='*80}")
    print("SETUP SUMMARY (sorted by Net PnL)")
    print(f"{'='*80}")
    print(f"{'Setup':<35} {'Trades':>8} {'Win%':>7} {'Net PnL':>15} {'Avg Net':>10} {'Status':<10}")
    print("-" * 90)

    for setup, setup_trades in sorted_setups:
        stats = calc_stats(setup_trades)
        status = "PROFIT" if stats["net_pnl"] > 0 else "LOSS"
        print(f"{setup:<35} {stats['count']:>8,} {stats['win_rate']:>6.1f}% Rs {stats['net_pnl']:>12,.0f} Rs {stats['avg_net']:>7,.0f} {status:<10}")

    # Deep dive into each setup
    print("\n" + "=" * 120)
    print("DETAILED SETUP ANALYSIS")
    print("=" * 120)

    for setup, setup_trades in sorted_setups:
        if len(setup_trades) < 10:
            print(f"\n>>> {setup}: Only {len(setup_trades)} trades - skipping detailed analysis")
            continue

        analyze_setup(setup, setup_trades, trades)

        # Suggest filters based on patterns
        print(f"\n>>> FILTER SIMULATIONS FOR {setup}")

        # Hour-based filters
        if any(t["entry_hour"] == 10 for t in setup_trades):
            simulate_filter(setup_trades, lambda t: t["entry_hour"] != 10, "Block 10:00 hour")

        if any(t["entry_hour"] >= 14 for t in setup_trades):
            simulate_filter(setup_trades, lambda t: t["entry_hour"] < 14, "Block after 14:00")

        # Cap-based filters
        caps = set(t["cap_segment"] for t in setup_trades)
        if "large_cap" in caps:
            simulate_filter(setup_trades, lambda t: t["cap_segment"] != "large_cap", "Block large_cap")
        if "mid_cap" in caps:
            simulate_filter(setup_trades, lambda t: t["cap_segment"] != "mid_cap", "Block mid_cap")

        # Regime-based filters
        regimes = set(t["regime"] for t in setup_trades)
        for regime in regimes:
            regime_trades = [t for t in setup_trades if t["regime"] == regime]
            regime_stats = calc_stats(regime_trades)
            if regime_stats["avg_net"] < 0 and len(regime_trades) > 10:
                simulate_filter(setup_trades, lambda t, r=regime: t["regime"] != r, f"Block regime={regime}")

        # Rank score filters
        simulate_filter(setup_trades, lambda t: t["rank_score"] >= 1.5, "min_rank_score=1.5")
        simulate_filter(setup_trades, lambda t: t["rank_score"] >= 2.0, "min_rank_score=2.0")

        # ADX filters
        simulate_filter(setup_trades, lambda t: t["adx"] >= 15, "min_adx=15")
        simulate_filter(setup_trades, lambda t: t["adx"] >= 20, "min_adx=20")

        print("-" * 120)


if __name__ == "__main__":
    main()
