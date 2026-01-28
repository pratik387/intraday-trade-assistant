"""
Deep Analysis of SL Hit Patterns and Cap Segment Performance.

Key questions:
1. What's causing the massive hard_sl losses (-Rs 393L)?
2. Why is micro_cap profitable while others aren't?
3. What parameters predict SL hits vs target hits?
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

# All backtest folders
BACKTEST_FOLDERS = [
    'backtest_20251223-111540_extracted',
    'backtest_20251222-194823_extracted',
    'backtest_20251222-185203_extracted',
    'backtest_20251222-141442_extracted',
]

BROKERAGE_PER_ORDER = 20
STT_RATE = 0.00025
EXCHANGE_RATE = 0.0000345
SEBI_RATE = 0.000001
STAMP_DUTY_RATE = 0.00003
GST_RATE = 0.18


def calculate_charges(entry_value, exit_value, num_exits=1):
    brokerage = BROKERAGE_PER_ORDER * (1 + num_exits)
    stt = exit_value * STT_RATE
    exchange = (entry_value + exit_value) * EXCHANGE_RATE
    sebi = (entry_value + exit_value) * SEBI_RATE
    stamp = entry_value * STAMP_DUTY_RATE
    gst = (brokerage + exchange) * GST_RATE
    return brokerage + stt + exchange + sebi + stamp + gst


def parse_all_data(base_dir):
    decisions = {}
    outcomes = {}

    base_path = Path(base_dir)

    for folder_name in BACKTEST_FOLDERS:
        folder_path = base_path / folder_name
        if not folder_path.exists():
            continue

        date_folders = sorted([d for d in folder_path.iterdir() if d.is_dir() and d.name.startswith("20")])

        for date_folder in date_folders:
            events_file = date_folder / "events.jsonl"
            if events_file.exists():
                with open(events_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            if event.get("type") == "DECISION":
                                trade_id = event.get("trade_id")
                                if trade_id:
                                    event["_date"] = date_folder.name
                                    decisions[trade_id] = event
                        except:
                            continue

            analytics_file = date_folder / "analytics.jsonl"
            if analytics_file.exists():
                with open(analytics_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            trade_id = event.get("trade_id")
                            if trade_id and event.get("is_final_exit"):
                                outcomes[trade_id] = event
                        except:
                            continue

    return decisions, outcomes


def extract_trade(decision, outcome):
    plan = decision.get("plan", {})
    dec = decision.get("decision", {})

    ranking = plan.get("ranking", {})
    if isinstance(ranking, list):
        ranking = {}
    components = ranking.get("components", {})

    sizing = plan.get("sizing", {})
    if isinstance(sizing, list):
        sizing = {}

    quality = plan.get("quality", {})
    if isinstance(quality, list):
        quality = {}

    indicators = plan.get("indicators", {})
    if isinstance(indicators, list):
        indicators = {}

    entry_price = outcome.get("actual_entry_price", 0) or outcome.get("entry_reference", 0)
    exit_price = outcome.get("exit_price", 0)
    qty = outcome.get("qty", 0)
    total_exits = outcome.get("total_exits", 1)

    entry_value = entry_price * qty if entry_price and qty else 0
    exit_value = exit_price * qty if exit_price and qty else 0
    charges = calculate_charges(entry_value, exit_value, total_exits) if entry_value > 0 else 50

    gross_pnl = outcome.get("total_trade_pnl", 0)
    net_pnl = gross_pnl - charges
    exit_reason = outcome.get("reason", "unknown")

    # Classify exit type
    if exit_reason == "hard_sl":
        exit_type = "hard_sl"
    elif exit_reason.startswith("target_t"):
        exit_type = "target"
    elif "sl_post_t1" in exit_reason:
        exit_type = "sl_post_t1"
    elif exit_reason.startswith("eod"):
        exit_type = "eod"
    else:
        exit_type = "other"

    return {
        "trade_id": decision.get("trade_id"),
        "symbol": decision.get("symbol"),
        "date": decision.get("_date"),
        "setup_type": dec.get("setup_type", plan.get("strategy", "unknown")),
        "regime": dec.get("regime", plan.get("regime", "unknown")),
        "category": plan.get("category", "unknown"),
        "bias": plan.get("bias", "unknown"),
        "cap_segment": sizing.get("cap_segment", "unknown"),
        "rank_score": ranking.get("score", 0),
        "comp_volume": components.get("volume", 0),
        "comp_adx": components.get("adx", 0),
        "comp_rsi": components.get("rsi", 0),
        "structural_rr": quality.get("structural_rr", 0),
        "ind_adx": indicators.get("adx", 0),
        "ind_rsi": indicators.get("rsi", 50),
        "ind_atr": indicators.get("atr", 0),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "exit_type": exit_type,
        "gross_pnl": gross_pnl,
        "charges": charges,
        "net_pnl": net_pnl,
        "is_winner": net_pnl > 0,
    }


def analyze_sl_patterns(trades):
    """Analyze what causes hard_sl hits."""
    print("\n" + "=" * 80)
    print("HARD SL HIT ANALYSIS")
    print("=" * 80)

    sl_trades = [t for t in trades if t["exit_type"] == "hard_sl"]
    non_sl_trades = [t for t in trades if t["exit_type"] != "hard_sl"]

    print(f"\nTotal trades: {len(trades):,}")
    print(f"Hard SL hits: {len(sl_trades):,} ({len(sl_trades)/len(trades)*100:.1f}%)")
    print(f"Non-SL exits: {len(non_sl_trades):,}")

    # Compare parameters
    print(f"\n{'Parameter':<25} {'SL Hit Avg':>12} {'Non-SL Avg':>12} {'Diff%':>10}")
    print("-" * 65)

    params = ["rank_score", "comp_volume", "comp_adx", "comp_rsi", "structural_rr", "ind_adx", "ind_rsi", "ind_atr"]

    for param in params:
        sl_vals = [t[param] for t in sl_trades if t.get(param) is not None and isinstance(t[param], (int, float))]
        non_sl_vals = [t[param] for t in non_sl_trades if t.get(param) is not None and isinstance(t[param], (int, float))]

        if not sl_vals or not non_sl_vals:
            continue

        sl_avg = statistics.mean(sl_vals)
        non_sl_avg = statistics.mean(non_sl_vals)
        diff = ((sl_avg - non_sl_avg) / abs(non_sl_avg) * 100) if non_sl_avg != 0 else 0

        print(f"{param:<25} {sl_avg:>12.3f} {non_sl_avg:>12.3f} {diff:>+9.1f}%")

    # SL by setup type
    print(f"\n{'Setup':<35} {'Total':>8} {'SL Hits':>8} {'SL%':>8} {'SL Loss':>12}")
    print("-" * 80)

    by_setup = defaultdict(lambda: {"total": 0, "sl": 0, "sl_loss": 0})
    for t in trades:
        setup = t["setup_type"]
        by_setup[setup]["total"] += 1
        if t["exit_type"] == "hard_sl":
            by_setup[setup]["sl"] += 1
            by_setup[setup]["sl_loss"] += t["net_pnl"]

    for setup, stats in sorted(by_setup.items(), key=lambda x: x[1]["sl_loss"]):
        if stats["total"] < 50:
            continue
        sl_pct = stats["sl"] / stats["total"] * 100
        print(f"{setup[:35]:<35} {stats['total']:>8} {stats['sl']:>8} {sl_pct:>7.1f}% {stats['sl_loss']:>12,.0f}")

    # SL by regime
    print(f"\n{'Regime':<20} {'Total':>8} {'SL Hits':>8} {'SL%':>8} {'SL Loss':>12}")
    print("-" * 60)

    by_regime = defaultdict(lambda: {"total": 0, "sl": 0, "sl_loss": 0})
    for t in trades:
        regime = t["regime"]
        by_regime[regime]["total"] += 1
        if t["exit_type"] == "hard_sl":
            by_regime[regime]["sl"] += 1
            by_regime[regime]["sl_loss"] += t["net_pnl"]

    for regime, stats in sorted(by_regime.items(), key=lambda x: x[1]["sl_loss"]):
        sl_pct = stats["sl"] / stats["total"] * 100
        print(f"{regime:<20} {stats['total']:>8} {stats['sl']:>8} {sl_pct:>7.1f}% {stats['sl_loss']:>12,.0f}")

    # SL by cap segment
    print(f"\n{'Cap Segment':<20} {'Total':>8} {'SL Hits':>8} {'SL%':>8} {'SL Loss':>12}")
    print("-" * 60)

    by_cap = defaultdict(lambda: {"total": 0, "sl": 0, "sl_loss": 0})
    for t in trades:
        cap = t["cap_segment"]
        by_cap[cap]["total"] += 1
        if t["exit_type"] == "hard_sl":
            by_cap[cap]["sl"] += 1
            by_cap[cap]["sl_loss"] += t["net_pnl"]

    for cap, stats in sorted(by_cap.items(), key=lambda x: x[1]["sl_loss"]):
        sl_pct = stats["sl"] / stats["total"] * 100
        print(f"{cap:<20} {stats['total']:>8} {stats['sl']:>8} {sl_pct:>7.1f}% {stats['sl_loss']:>12,.0f}")


def analyze_bias_performance(trades):
    """Analyze LONG vs SHORT performance."""
    print("\n" + "=" * 80)
    print("LONG vs SHORT BIAS ANALYSIS")
    print("=" * 80)

    longs = [t for t in trades if t["bias"] == "long"]
    shorts = [t for t in trades if t["bias"] == "short"]

    print(f"\nLONG trades: {len(longs):,}")
    print(f"  Win rate: {sum(1 for t in longs if t['is_winner'])/len(longs)*100:.1f}%")
    print(f"  Avg Net PnL: Rs {statistics.mean([t['net_pnl'] for t in longs]):.0f}")
    print(f"  Total Net PnL: Rs {sum(t['net_pnl'] for t in longs):,.0f}")
    print(f"  SL hit rate: {sum(1 for t in longs if t['exit_type']=='hard_sl')/len(longs)*100:.1f}%")

    print(f"\nSHORT trades: {len(shorts):,}")
    print(f"  Win rate: {sum(1 for t in shorts if t['is_winner'])/len(shorts)*100:.1f}%")
    print(f"  Avg Net PnL: Rs {statistics.mean([t['net_pnl'] for t in shorts]):.0f}")
    print(f"  Total Net PnL: Rs {sum(t['net_pnl'] for t in shorts):,.0f}")
    print(f"  SL hit rate: {sum(1 for t in shorts if t['exit_type']=='hard_sl')/len(shorts)*100:.1f}%")

    # Long by regime
    print(f"\n{'LONG by Regime':<25} {'Trades':>8} {'Win%':>8} {'AvgNet':>10} {'TotalNet':>12}")
    print("-" * 70)

    by_regime = defaultdict(list)
    for t in longs:
        by_regime[t["regime"]].append(t)

    for regime, rtrades in sorted(by_regime.items(), key=lambda x: sum(t["net_pnl"] for t in x[1]), reverse=True):
        wins = sum(1 for t in rtrades if t["is_winner"])
        avg = statistics.mean([t["net_pnl"] for t in rtrades])
        total = sum(t["net_pnl"] for t in rtrades)
        print(f"{regime:<25} {len(rtrades):>8} {wins/len(rtrades)*100:>7.1f}% {avg:>10.0f} {total:>12,.0f}")

    # Short by regime
    print(f"\n{'SHORT by Regime':<25} {'Trades':>8} {'Win%':>8} {'AvgNet':>10} {'TotalNet':>12}")
    print("-" * 70)

    by_regime = defaultdict(list)
    for t in shorts:
        by_regime[t["regime"]].append(t)

    for regime, rtrades in sorted(by_regime.items(), key=lambda x: sum(t["net_pnl"] for t in x[1]), reverse=True):
        wins = sum(1 for t in rtrades if t["is_winner"])
        avg = statistics.mean([t["net_pnl"] for t in rtrades])
        total = sum(t["net_pnl"] for t in rtrades)
        print(f"{regime:<25} {len(rtrades):>8} {wins/len(rtrades)*100:>7.1f}% {avg:>10.0f} {total:>12,.0f}")


def analyze_cap_segment_deep(trades):
    """Deep dive into cap segment performance."""
    print("\n" + "=" * 80)
    print("CAP SEGMENT DEEP DIVE")
    print("=" * 80)

    for cap in ["micro_cap", "small_cap", "mid_cap", "large_cap"]:
        cap_trades = [t for t in trades if t["cap_segment"] == cap]
        if not cap_trades:
            continue

        print(f"\n{cap.upper()} ({len(cap_trades):,} trades)")
        print("-" * 50)

        total = sum(t["net_pnl"] for t in cap_trades)
        avg = statistics.mean([t["net_pnl"] for t in cap_trades])
        wins = sum(1 for t in cap_trades if t["is_winner"])
        sl_hits = sum(1 for t in cap_trades if t["exit_type"] == "hard_sl")

        print(f"  Total Net PnL: Rs {total:,.0f}")
        print(f"  Avg Net PnL: Rs {avg:.0f}")
        print(f"  Win Rate: {wins/len(cap_trades)*100:.1f}%")
        print(f"  SL Hit Rate: {sl_hits/len(cap_trades)*100:.1f}%")

        # By setup
        by_setup = defaultdict(list)
        for t in cap_trades:
            by_setup[t["setup_type"]].append(t)

        print(f"\n  {'Setup':<30} {'Trades':>7} {'AvgNet':>9} {'TotalNet':>12}")
        for setup, strades in sorted(by_setup.items(), key=lambda x: sum(t["net_pnl"] for t in x[1]), reverse=True)[:5]:
            avg = statistics.mean([t["net_pnl"] for t in strades])
            total = sum(t["net_pnl"] for t in strades)
            print(f"  {setup[:30]:<30} {len(strades):>7} {avg:>9.0f} {total:>12,.0f}")


def find_profitable_filters(trades):
    """Find filters that result in profitable subsets."""
    print("\n" + "=" * 80)
    print("SEARCHING FOR PROFITABLE SUBSETS")
    print("=" * 80)

    baseline_avg = statistics.mean([t["net_pnl"] for t in trades])
    print(f"\nBaseline: {len(trades):,} trades, Avg: Rs {baseline_avg:.0f}")

    filters = [
        # Bias filters
        ("bias == short", lambda t: t["bias"] == "short"),
        ("bias == short AND cap == micro", lambda t: t["bias"] == "short" and t["cap_segment"] == "micro_cap"),
        ("bias == short AND cap == small", lambda t: t["bias"] == "short" and t["cap_segment"] == "small_cap"),

        # Setup filters for shorts
        ("range_bounce_short", lambda t: t["setup_type"] == "range_bounce_short"),
        ("resistance_bounce_short", lambda t: t["setup_type"] == "resistance_bounce_short"),

        # ADX filters
        ("comp_adx >= 0.2", lambda t: t.get("comp_adx", 0) >= 0.2),
        ("comp_adx >= 0.3", lambda t: t.get("comp_adx", 0) >= 0.3),
        ("comp_adx >= 0.4", lambda t: t.get("comp_adx", 0) >= 0.4),

        # Volume filters
        ("comp_volume >= 1.2", lambda t: t.get("comp_volume", 0) >= 1.2),
        ("comp_volume >= 1.4", lambda t: t.get("comp_volume", 0) >= 1.4),

        # Combined
        ("short AND adx >= 0.3", lambda t: t["bias"] == "short" and t.get("comp_adx", 0) >= 0.3),
        ("short AND volume >= 1.2", lambda t: t["bias"] == "short" and t.get("comp_volume", 0) >= 1.2),
        ("short AND adx >= 0.2 AND volume >= 1.2", lambda t: t["bias"] == "short" and t.get("comp_adx", 0) >= 0.2 and t.get("comp_volume", 0) >= 1.2),

        # Cap segment filters
        ("micro_cap only", lambda t: t["cap_segment"] == "micro_cap"),
        ("micro_cap AND short", lambda t: t["cap_segment"] == "micro_cap" and t["bias"] == "short"),

        # Structural RR (lower is better based on earlier findings)
        ("structural_rr <= 3.0", lambda t: t.get("structural_rr", 10) <= 3.0),
        ("structural_rr <= 2.5", lambda t: t.get("structural_rr", 10) <= 2.5),

        # RSI zone (lower is better)
        ("comp_rsi <= 0.1", lambda t: t.get("comp_rsi", 1) <= 0.1),
    ]

    print(f"\n{'Filter':<45} {'Trades':>8} {'AvgNet':>10} {'TotalNet':>12} {'Win%':>8}")
    print("-" * 90)

    profitable = []
    for name, func in filters:
        filtered = [t for t in trades if func(t)]
        if len(filtered) < 100:
            continue

        avg = statistics.mean([t["net_pnl"] for t in filtered])
        total = sum(t["net_pnl"] for t in filtered)
        wins = sum(1 for t in filtered if t["is_winner"])
        win_rate = wins / len(filtered) * 100

        if avg > 0:
            profitable.append((name, len(filtered), avg, total, win_rate))

        status = "***" if avg > 0 else ""
        print(f"{name:<45} {len(filtered):>8} {avg:>10.0f} {total:>12,.0f} {win_rate:>7.1f}% {status}")

    if profitable:
        print("\n" + "=" * 80)
        print("PROFITABLE FILTERS FOUND!")
        print("=" * 80)
        for name, count, avg, total, win_rate in sorted(profitable, key=lambda x: x[2], reverse=True):
            print(f"  {name}: {count:,} trades, Avg: Rs {avg:.0f}, Total: Rs {total:,.0f}, Win: {win_rate:.1f}%")


def main():
    print("=" * 80)
    print("SL AND CAP SEGMENT DEEP ANALYSIS")
    print("=" * 80)

    base_path = Path(__file__).parent.parent

    print("\nParsing data...")
    decisions, outcomes = parse_all_data(str(base_path))

    trades = []
    for trade_id, decision in decisions.items():
        if trade_id in outcomes:
            trades.append(extract_trade(decision, outcomes[trade_id]))

    print(f"Loaded {len(trades):,} trades")

    analyze_sl_patterns(trades)
    analyze_bias_performance(trades)
    analyze_cap_segment_deep(trades)
    find_profitable_filters(trades)


if __name__ == "__main__":
    main()
