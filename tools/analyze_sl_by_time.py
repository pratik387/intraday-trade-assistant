"""
Analyze SL Hit Patterns by Time of Day
Find optimal trading windows to reduce stop loss hits.
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


def load_trades_with_exit_info():
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

                # Parse entry hour and minute
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

                # Classify exit
                is_sl_hit = "hard_sl" in exit_reason.lower() or "stop" in exit_reason.lower()
                is_t1_hit = "t1" in exit_reason.lower()
                is_t2_hit = "t2" in exit_reason.lower()
                is_eod = "eod" in exit_reason.lower() or "end_of_day" in exit_reason.lower()

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
                    "is_t1_hit": is_t1_hit,
                    "is_t2_hit": is_t2_hit,
                    "is_eod": is_eod,
                    "pnl": pnl,
                    "adx": decision["adx"],
                    "rsi": decision["rsi"],
                    "rank_score": decision["rank_score"],
                })

    return trades


def apply_moderate_filter(trade):
    """MODERATE profile filter"""
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


def main():
    print("=" * 120)
    print("SL HIT PATTERN ANALYSIS BY TIME OF DAY")
    print("=" * 120)

    all_trades = load_trades_with_exit_info()
    print(f"Loaded {len(all_trades)} trades")

    moderate_trades = [t for t in all_trades if apply_moderate_filter(t)]
    print(f"MODERATE filtered trades: {len(moderate_trades)}")

    # Analysis by entry hour
    print("\n" + "=" * 120)
    print("ANALYSIS BY ENTRY HOUR (MODERATE Profile)")
    print("=" * 120)

    by_hour = defaultdict(list)
    for t in moderate_trades:
        by_hour[t["entry_hour"]].append(t)

    print(f"\n{'Hour':<8} {'Trades':>8} {'SL%':>8} {'T1%':>8} {'T2%':>8} {'EOD%':>8} {'AvgPnL':>12} {'TotalPnL':>14}")
    print("-" * 90)

    hour_stats = []
    for hour in sorted(by_hour.keys()):
        trades = by_hour[hour]
        total = len(trades)
        sl_count = sum(1 for t in trades if t["is_sl_hit"])
        t1_count = sum(1 for t in trades if t["is_t1_hit"])
        t2_count = sum(1 for t in trades if t["is_t2_hit"])
        eod_count = sum(1 for t in trades if t["is_eod"])
        total_pnl = sum(t["pnl"] for t in trades)
        avg_pnl = total_pnl / total if total > 0 else 0

        sl_pct = sl_count / total * 100 if total > 0 else 0
        t1_pct = t1_count / total * 100 if total > 0 else 0
        t2_pct = t2_count / total * 100 if total > 0 else 0
        eod_pct = eod_count / total * 100 if total > 0 else 0

        hour_stats.append({
            "hour": hour,
            "trades": total,
            "sl_pct": sl_pct,
            "t1_pct": t1_pct,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
        })

        print(f"{hour:02d}:00{'':<3} {total:>8} {sl_pct:>7.1f}% {t1_pct:>7.1f}% {t2_pct:>7.1f}% {eod_pct:>7.1f}% Rs {avg_pnl:>9,.0f} Rs {total_pnl:>11,.0f}")

    # Analysis by 30-minute windows
    print("\n" + "=" * 120)
    print("ANALYSIS BY 30-MIN WINDOWS (MODERATE Profile)")
    print("=" * 120)

    by_window = defaultdict(list)
    for t in moderate_trades:
        window = f"{t['entry_hour']:02d}:{0 if t['entry_minute'] < 30 else 30:02d}"
        by_window[window].append(t)

    print(f"\n{'Window':<10} {'Trades':>8} {'SL%':>8} {'T1%':>8} {'WinRate':>10} {'AvgPnL':>12} {'TotalPnL':>14}")
    print("-" * 80)

    window_stats = []
    for window in sorted(by_window.keys()):
        trades = by_window[window]
        total = len(trades)
        sl_count = sum(1 for t in trades if t["is_sl_hit"])
        t1_count = sum(1 for t in trades if t["is_t1_hit"])
        winners = sum(1 for t in trades if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in trades)
        avg_pnl = total_pnl / total if total > 0 else 0

        sl_pct = sl_count / total * 100 if total > 0 else 0
        t1_pct = t1_count / total * 100 if total > 0 else 0
        win_rate = winners / total * 100 if total > 0 else 0

        window_stats.append({
            "window": window,
            "trades": total,
            "sl_pct": sl_pct,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
        })

        print(f"{window:<10} {total:>8} {sl_pct:>7.1f}% {t1_pct:>7.1f}% {win_rate:>9.1f}% Rs {avg_pnl:>9,.0f} Rs {total_pnl:>11,.0f}")

    # Per-setup time analysis
    print("\n" + "=" * 120)
    print("SETUP-SPECIFIC TIME ANALYSIS (High SL setups)")
    print("=" * 120)

    by_setup = defaultdict(list)
    for t in moderate_trades:
        by_setup[t["setup"]].append(t)

    # Find setups with high SL rates
    high_sl_setups = []
    for setup, trades in by_setup.items():
        if len(trades) >= 20:
            sl_count = sum(1 for t in trades if t["is_sl_hit"])
            sl_pct = sl_count / len(trades) * 100
            if sl_pct > 35:
                high_sl_setups.append((setup, trades, sl_pct))

    for setup, trades, overall_sl in sorted(high_sl_setups, key=lambda x: -x[2]):
        print(f"\n{setup} (Overall SL: {overall_sl:.1f}%, {len(trades)} trades)")
        print("-" * 80)

        by_hour_setup = defaultdict(list)
        for t in trades:
            by_hour_setup[t["entry_hour"]].append(t)

        print(f"  {'Hour':<8} {'Trades':>8} {'SL%':>8} {'WR%':>8} {'AvgPnL':>12}")
        for hour in sorted(by_hour_setup.keys()):
            h_trades = by_hour_setup[hour]
            if len(h_trades) < 5:
                continue
            sl_count = sum(1 for t in h_trades if t["is_sl_hit"])
            winners = sum(1 for t in h_trades if t["pnl"] > 0)
            sl_pct = sl_count / len(h_trades) * 100
            win_rate = winners / len(h_trades) * 100
            avg_pnl = sum(t["pnl"] for t in h_trades) / len(h_trades)
            print(f"  {hour:02d}:00{'':<3} {len(h_trades):>8} {sl_pct:>7.1f}% {win_rate:>7.1f}% Rs {avg_pnl:>9,.0f}")

    # Recommendations
    print("\n" + "=" * 120)
    print("TIME-BASED FILTER RECOMMENDATIONS")
    print("=" * 120)

    # Find bad windows (high SL, low avg PnL)
    bad_windows = [w for w in window_stats if w["sl_pct"] > 40 and w["avg_pnl"] < 50]
    good_windows = [w for w in window_stats if w["sl_pct"] < 35 and w["avg_pnl"] > 100]

    print("\n>>> WINDOWS TO AVOID (High SL%, Low Avg PnL):")
    for w in sorted(bad_windows, key=lambda x: x["sl_pct"], reverse=True):
        print(f"    {w['window']}: SL {w['sl_pct']:.1f}%, Avg PnL Rs {w['avg_pnl']:.0f}, {w['trades']} trades")

    print("\n>>> BEST WINDOWS (Low SL%, High Avg PnL):")
    for w in sorted(good_windows, key=lambda x: x["avg_pnl"], reverse=True):
        print(f"    {w['window']}: SL {w['sl_pct']:.1f}%, Avg PnL Rs {w['avg_pnl']:.0f}, {w['trades']} trades")

    # Simulate impact of blocking bad windows
    print("\n" + "=" * 120)
    print("SIMULATION: Impact of Time Filters")
    print("=" * 120)

    # Define windows to test blocking
    windows_to_block = [w["window"] for w in bad_windows if w["trades"] >= 100]

    if windows_to_block:
        blocked_trades = [t for t in moderate_trades
                         if f"{t['entry_hour']:02d}:{0 if t['entry_minute'] < 30 else 30:02d}" not in windows_to_block]

        original_pnl = sum(t["pnl"] for t in moderate_trades)
        original_sl = sum(1 for t in moderate_trades if t["is_sl_hit"]) / len(moderate_trades) * 100

        new_pnl = sum(t["pnl"] for t in blocked_trades)
        new_sl = sum(1 for t in blocked_trades if t["is_sl_hit"]) / len(blocked_trades) * 100 if blocked_trades else 0

        trades_blocked = len(moderate_trades) - len(blocked_trades)
        pnl_diff = new_pnl - original_pnl

        print(f"\nBlocking windows: {windows_to_block}")
        print(f"  Original: {len(moderate_trades):,} trades, SL {original_sl:.1f}%, PnL Rs {original_pnl:,.0f}")
        print(f"  After:    {len(blocked_trades):,} trades, SL {new_sl:.1f}%, PnL Rs {new_pnl:,.0f}")
        print(f"  Impact:   Blocked {trades_blocked:,} trades, PnL change Rs {pnl_diff:+,.0f}")

    # Direction-specific time analysis
    print("\n" + "=" * 120)
    print("LONG vs SHORT TIME ANALYSIS")
    print("=" * 120)

    for direction in ["LONG", "SHORT"]:
        dir_trades = [t for t in moderate_trades if t["direction"] == direction]
        if not dir_trades:
            continue

        print(f"\n{direction} trades ({len(dir_trades)} total):")
        print(f"  {'Hour':<8} {'Trades':>8} {'SL%':>8} {'WR%':>8} {'AvgPnL':>12}")

        by_hour_dir = defaultdict(list)
        for t in dir_trades:
            by_hour_dir[t["entry_hour"]].append(t)

        for hour in sorted(by_hour_dir.keys()):
            trades = by_hour_dir[hour]
            if len(trades) < 10:
                continue
            sl_pct = sum(1 for t in trades if t["is_sl_hit"]) / len(trades) * 100
            win_rate = sum(1 for t in trades if t["pnl"] > 0) / len(trades) * 100
            avg_pnl = sum(t["pnl"] for t in trades) / len(trades)
            print(f"  {hour:02d}:00{'':<3} {len(trades):>8} {sl_pct:>7.1f}% {win_rate:>7.1f}% Rs {avg_pnl:>9,.0f}")


if __name__ == "__main__":
    main()
