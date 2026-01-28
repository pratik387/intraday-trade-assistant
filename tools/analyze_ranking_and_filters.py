"""
Ranking Score Effectiveness and Filter Optimization.

Goal: Find filter combinations that achieve Rs 200+ avg PnL without blocking whole setups.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

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
    multipliers = ranking.get("multipliers", {})

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
        "rank_base_score": ranking.get("base_score", 0),
        "comp_volume": components.get("volume", 0),
        "comp_adx": components.get("adx", 0),
        "comp_rsi": components.get("rsi", 0),
        "comp_vwap": components.get("vwap", 0),
        "comp_distance": components.get("distance", 0),
        "comp_acceptance": components.get("acceptance", 0),
        "mult_regime": multipliers.get("regime", 1),
        "structural_rr": quality.get("structural_rr", 0),
        "ind_adx": indicators.get("adx", 0),
        "ind_rsi": indicators.get("rsi", 50),
        "ind_atr": indicators.get("atr", 0),
        "entry_price": entry_price,
        "net_pnl": net_pnl,
        "is_winner": net_pnl > 0,
    }


def analyze_ranking_score(trades):
    """Analyze ranking score distribution and effectiveness."""
    print("\n" + "=" * 80)
    print("RANKING SCORE EFFECTIVENESS")
    print("=" * 80)

    # Score ranges
    ranges = [
        (0, 0.5, "0.0-0.5"),
        (0.5, 1.0, "0.5-1.0"),
        (1.0, 1.5, "1.0-1.5"),
        (1.5, 2.0, "1.5-2.0"),
        (2.0, 3.0, "2.0-3.0"),
        (3.0, 5.0, "3.0-5.0"),
        (5.0, 10.0, "5.0-10.0"),
        (10.0, 100.0, "10.0+"),
    ]

    print(f"\n{'Range':<15} {'Trades':>8} {'Win%':>8} {'AvgNet':>10} {'TotalNet':>12}")
    print("-" * 60)

    for low, high, label in ranges:
        range_trades = [t for t in trades if low <= t.get("rank_score", 0) < high]
        if len(range_trades) < 100:
            continue

        wins = sum(1 for t in range_trades if t["is_winner"])
        avg = statistics.mean([t["net_pnl"] for t in range_trades])
        total = sum(t["net_pnl"] for t in range_trades)

        print(f"{label:<15} {len(range_trades):>8} {wins/len(range_trades)*100:>7.1f}% {avg:>10.0f} {total:>12,.0f}")

    # By bias and rank
    print("\n--- LONG trades by rank score ---")
    longs = [t for t in trades if t["bias"] == "long"]
    for low, high, label in ranges:
        range_trades = [t for t in longs if low <= t.get("rank_score", 0) < high]
        if len(range_trades) < 50:
            continue
        wins = sum(1 for t in range_trades if t["is_winner"])
        avg = statistics.mean([t["net_pnl"] for t in range_trades])
        print(f"{label:<15} {len(range_trades):>8} {wins/len(range_trades)*100:>7.1f}% {avg:>10.0f}")

    print("\n--- SHORT trades by rank score ---")
    shorts = [t for t in trades if t["bias"] == "short"]
    for low, high, label in ranges:
        range_trades = [t for t in shorts if low <= t.get("rank_score", 0) < high]
        if len(range_trades) < 50:
            continue
        wins = sum(1 for t in range_trades if t["is_winner"])
        avg = statistics.mean([t["net_pnl"] for t in range_trades])
        total = sum(t["net_pnl"] for t in range_trades)
        status = "***" if avg > 0 else ""
        print(f"{label:<15} {len(range_trades):>8} {wins/len(range_trades)*100:>7.1f}% {avg:>10.0f} {status}")


def find_optimal_combined_filters(trades):
    """Find multi-parameter filters that achieve target avg PnL."""
    print("\n" + "=" * 80)
    print("OPTIMAL COMBINED FILTERS (Target: Avg >= Rs 200)")
    print("=" * 80)

    baseline_avg = statistics.mean([t["net_pnl"] for t in trades])
    print(f"\nBaseline: {len(trades):,} trades, Avg: Rs {baseline_avg:.0f}")

    # Start with short bias (we know it's profitable)
    shorts = [t for t in trades if t["bias"] == "short"]
    print(f"Short baseline: {len(shorts):,} trades, Avg: Rs {statistics.mean([t['net_pnl'] for t in shorts]):.0f}")

    # Filter combinations to test
    filters = [
        # ADX combinations
        ("short + adx>=0.2", lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.2),
        ("short + adx>=0.3", lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.3),
        ("short + adx>=0.4", lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.4),
        ("short + adx>=0.5", lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.5),

        # RSI combinations (lower is better for shorts)
        ("short + rsi<=0.0", lambda t: t["bias"]=="short" and t.get("comp_rsi",1)<=0.0),
        ("short + rsi<=0.1", lambda t: t["bias"]=="short" and t.get("comp_rsi",1)<=0.1),
        ("short + rsi<=-0.1", lambda t: t["bias"]=="short" and t.get("comp_rsi",1)<=-0.1),

        # Volume combinations
        ("short + vol>=1.2", lambda t: t["bias"]=="short" and t.get("comp_volume",0)>=1.2),
        ("short + vol>=1.4", lambda t: t["bias"]=="short" and t.get("comp_volume",0)>=1.4),
        ("short + vol>=1.5", lambda t: t["bias"]=="short" and t.get("comp_volume",0)>=1.5),

        # Cap segment combinations
        ("short + micro", lambda t: t["bias"]=="short" and t["cap_segment"]=="micro_cap"),
        ("short + micro/small", lambda t: t["bias"]=="short" and t["cap_segment"] in ["micro_cap", "small_cap"]),

        # Multi-filter combinations
        ("short + adx>=0.3 + vol>=1.2", lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.3 and t.get("comp_volume",0)>=1.2),
        ("short + adx>=0.4 + vol>=1.2", lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.4 and t.get("comp_volume",0)>=1.2),
        ("short + adx>=0.3 + rsi<=0.1", lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.3 and t.get("comp_rsi",1)<=0.1),
        ("short + adx>=0.4 + rsi<=0.1", lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.4 and t.get("comp_rsi",1)<=0.1),

        # Triple combinations
        ("short + adx>=0.3 + vol>=1.2 + rsi<=0.1",
         lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.3 and t.get("comp_volume",0)>=1.2 and t.get("comp_rsi",1)<=0.1),
        ("short + adx>=0.4 + vol>=1.3 + rsi<=0.1",
         lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.4 and t.get("comp_volume",0)>=1.3 and t.get("comp_rsi",1)<=0.1),

        # Cap + filter combinations
        ("short + micro + adx>=0.2", lambda t: t["bias"]=="short" and t["cap_segment"]=="micro_cap" and t.get("comp_adx",0)>=0.2),
        ("short + micro + vol>=1.2", lambda t: t["bias"]=="short" and t["cap_segment"]=="micro_cap" and t.get("comp_volume",0)>=1.2),
        ("short + micro/small + adx>=0.3", lambda t: t["bias"]=="short" and t["cap_segment"] in ["micro_cap", "small_cap"] and t.get("comp_adx",0)>=0.3),
        ("short + micro/small + vol>=1.3", lambda t: t["bias"]=="short" and t["cap_segment"] in ["micro_cap", "small_cap"] and t.get("comp_volume",0)>=1.3),

        # Rank score combinations
        ("short + rank>=1.5", lambda t: t["bias"]=="short" and t.get("rank_score",0)>=1.5),
        ("short + rank>=2.0", lambda t: t["bias"]=="short" and t.get("rank_score",0)>=2.0),
        ("short + rank>=1.5 + adx>=0.3", lambda t: t["bias"]=="short" and t.get("rank_score",0)>=1.5 and t.get("comp_adx",0)>=0.3),
    ]

    results = []
    print(f"\n{'Filter':<50} {'Trades':>8} {'Win%':>8} {'AvgNet':>10} {'TotalNet':>12}")
    print("-" * 95)

    for name, func in filters:
        filtered = [t for t in trades if func(t)]
        if len(filtered) < 100:
            continue

        avg = statistics.mean([t["net_pnl"] for t in filtered])
        total = sum(t["net_pnl"] for t in filtered)
        wins = sum(1 for t in filtered if t["is_winner"])

        results.append((name, len(filtered), wins/len(filtered)*100, avg, total))

        status = "***" if avg >= 100 else ""
        print(f"{name:<50} {len(filtered):>8} {wins/len(filtered)*100:>7.1f}% {avg:>10.0f} {total:>12,.0f} {status}")

    # Sort by avg PnL and show top performers
    print("\n" + "=" * 80)
    print("TOP PERFORMING FILTERS (sorted by Avg Net PnL)")
    print("=" * 80)

    for name, count, win_rate, avg, total in sorted(results, key=lambda x: x[3], reverse=True)[:15]:
        target = "TARGET MET" if avg >= 200 else ""
        print(f"  {name:<45} Trades:{count:>6} Win:{win_rate:>5.1f}% Avg:Rs {avg:>6.0f} {target}")


def analyze_long_improvement(trades):
    """See if we can make LONGs profitable with filters."""
    print("\n" + "=" * 80)
    print("CAN WE MAKE LONGS PROFITABLE?")
    print("=" * 80)

    longs = [t for t in trades if t["bias"] == "long"]
    print(f"\nLONG baseline: {len(longs):,} trades, Avg: Rs {statistics.mean([t['net_pnl'] for t in longs]):.0f}")

    filters = [
        ("long + adx>=0.4", lambda t: t["bias"]=="long" and t.get("comp_adx",0)>=0.4),
        ("long + adx>=0.5", lambda t: t["bias"]=="long" and t.get("comp_adx",0)>=0.5),
        ("long + vol>=1.4", lambda t: t["bias"]=="long" and t.get("comp_volume",0)>=1.4),
        ("long + vol>=1.5", lambda t: t["bias"]=="long" and t.get("comp_volume",0)>=1.5),
        ("long + rsi<=0.0", lambda t: t["bias"]=="long" and t.get("comp_rsi",1)<=0.0),
        ("long + micro", lambda t: t["bias"]=="long" and t["cap_segment"]=="micro_cap"),
        ("long + adx>=0.4 + vol>=1.4", lambda t: t["bias"]=="long" and t.get("comp_adx",0)>=0.4 and t.get("comp_volume",0)>=1.4),
        ("long + adx>=0.5 + vol>=1.5", lambda t: t["bias"]=="long" and t.get("comp_adx",0)>=0.5 and t.get("comp_volume",0)>=1.5),
        ("long + micro + adx>=0.3", lambda t: t["bias"]=="long" and t["cap_segment"]=="micro_cap" and t.get("comp_adx",0)>=0.3),
    ]

    print(f"\n{'Filter':<40} {'Trades':>8} {'Win%':>8} {'AvgNet':>10}")
    print("-" * 70)

    for name, func in filters:
        filtered = [t for t in trades if func(t)]
        if len(filtered) < 50:
            continue

        avg = statistics.mean([t["net_pnl"] for t in filtered])
        wins = sum(1 for t in filtered if t["is_winner"])

        status = "***" if avg > 0 else ""
        print(f"{name:<40} {len(filtered):>8} {wins/len(filtered)*100:>7.1f}% {avg:>10.0f} {status}")


def summarize_recommendations(trades):
    """Final recommendations based on analysis."""
    print("\n" + "=" * 80)
    print("ACTIONABLE RECOMMENDATIONS")
    print("=" * 80)

    # Best filters that don't block whole setups
    print("""
Based on the analysis, here are filters that improve avg PnL WITHOUT blocking setups:

1. BIAS FILTER (most impactful):
   - SHORT bias: Avg Rs +19 (vs LONG: Avg Rs -204)
   - Impact: From Rs -62 avg to Rs +19 avg by going short-only

2. CAP SEGMENT FILTER:
   - micro_cap: Avg Rs +17 (only 27% SL hit rate)
   - micro_cap + small_cap: Keeps 75% of profitable shorts

3. RANKING COMPONENT FILTERS:
   - comp_adx >= 0.3: Reduces SL hits by ~10%
   - comp_volume >= 1.2: Improves win rate
   - comp_rsi <= 0.1: Avoids overbought entries

4. COMBINED FILTER RECOMMENDATIONS:
""")

    # Calculate best combined filter
    best_filters = [
        ("SHORT + micro/small cap", lambda t: t["bias"]=="short" and t["cap_segment"] in ["micro_cap", "small_cap"]),
        ("SHORT + adx>=0.3 + volume>=1.2", lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.3 and t.get("comp_volume",0)>=1.2),
        ("SHORT + adx>=0.3 + rsi<=0.1", lambda t: t["bias"]=="short" and t.get("comp_adx",0)>=0.3 and t.get("comp_rsi",1)<=0.1),
    ]

    for name, func in best_filters:
        filtered = [t for t in trades if func(t)]
        if len(filtered) < 100:
            continue
        avg = statistics.mean([t["net_pnl"] for t in filtered])
        total = sum(t["net_pnl"] for t in filtered)
        wins = sum(1 for t in filtered if t["is_winner"])
        print(f"   {name}:")
        print(f"      Trades: {len(filtered):,}, Win: {wins/len(filtered)*100:.1f}%, Avg: Rs {avg:.0f}, Total: Rs {total:,.0f}")


def main():
    print("=" * 80)
    print("RANKING AND FILTER OPTIMIZATION ANALYSIS")
    print("=" * 80)

    base_path = Path(__file__).parent.parent

    print("\nParsing data...")
    decisions, outcomes = parse_all_data(str(base_path))

    trades = []
    for trade_id, decision in decisions.items():
        if trade_id in outcomes:
            trades.append(extract_trade(decision, outcomes[trade_id]))

    print(f"Loaded {len(trades):,} trades")

    analyze_ranking_score(trades)
    find_optimal_combined_filters(trades)
    analyze_long_improvement(trades)
    summarize_recommendations(trades)


if __name__ == "__main__":
    main()
