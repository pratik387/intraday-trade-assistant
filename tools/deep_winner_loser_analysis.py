"""
Deep Winner vs Loser Analysis Script

Analyzes trades to find discriminating patterns between winners and losers.
Extracts from:
- events.jsonl: DECISION events with ranking, indicators, sizing, quality
- analytics.jsonl: Trade outcomes (PnL, exit reason)
- planning.jsonl: Planning phase data

Goal: Find filters that can increase avg PnL to Rs 200-300+ without time-based filters
or blocking entire setup types.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import statistics

# Zerodha charges for net PnL calculation
BROKERAGE_PER_ORDER = 20
STT_RATE = 0.00025
EXCHANGE_RATE = 0.0000345
SEBI_RATE = 0.000001
STAMP_DUTY_RATE = 0.00003
GST_RATE = 0.18

# All backtest folders to process
BACKTEST_FOLDERS = [
    'backtest_20251223-111540_extracted',
    'backtest_20251222-194823_extracted',
    'backtest_20251222-185203_extracted',
    'backtest_20251222-141442_extracted',
]


def calculate_trade_charges(entry_value: float, exit_value: float, num_exits: int = 1) -> float:
    """Calculate total trading charges for a trade."""
    # Brokerage: Rs 20 per order (1 entry + num_exits exits)
    brokerage = BROKERAGE_PER_ORDER * (1 + num_exits)

    # STT: 0.025% on sell side only
    stt = exit_value * STT_RATE

    # Exchange charges on both sides
    exchange = (entry_value + exit_value) * EXCHANGE_RATE

    # SEBI charges on both sides
    sebi = (entry_value + exit_value) * SEBI_RATE

    # Stamp duty on buy side only
    stamp = entry_value * STAMP_DUTY_RATE

    # GST on brokerage + exchange charges
    gst = (brokerage + exchange) * GST_RATE

    return brokerage + stt + exchange + sebi + stamp + gst


def parse_all_data(base_dir: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Parse all DECISION events and trade outcomes from all backtest folders.

    Returns:
        decisions: Dict mapping trade_id to decision data (from events.jsonl)
        outcomes: Dict mapping trade_id to outcome data (from analytics.jsonl)
    """
    decisions = {}
    outcomes = {}

    base_path = Path(base_dir)

    for folder_name in BACKTEST_FOLDERS:
        folder_path = base_path / folder_name
        if not folder_path.exists():
            print(f"  Skipping {folder_name} - not found")
            continue

        # Find all date folders
        date_folders = sorted([d for d in folder_path.iterdir() if d.is_dir() and d.name.startswith("20")])
        print(f"  {folder_name}: {len(date_folders)} trading days")

        for date_folder in date_folders:
            # Parse events.jsonl for DECISION data
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
                                    event["_folder"] = folder_name
                                    decisions[trade_id] = event
                        except json.JSONDecodeError:
                            continue

            # Parse analytics.jsonl for outcomes
            analytics_file = date_folder / "analytics.jsonl"
            if analytics_file.exists():
                with open(analytics_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            trade_id = event.get("trade_id")
                            if trade_id and event.get("is_final_exit"):
                                outcomes[trade_id] = event
                        except json.JSONDecodeError:
                            continue

    print(f"\nParsed {len(decisions)} DECISION events")
    print(f"Parsed {len(outcomes)} final trade outcomes")

    return decisions, outcomes


def extract_flat_features(decision: Dict, outcome: Dict) -> Dict:
    """
    Extract all features from a decision/outcome pair into a flat dict.
    """
    plan = decision.get("plan", {})
    dec = decision.get("decision", {})
    features = decision.get("features", {})

    # Get nested structures
    ranking = plan.get("ranking", {})
    if isinstance(ranking, list):
        ranking = {}
    components = ranking.get("components", {})
    multipliers = ranking.get("multipliers", {})
    adjustments = ranking.get("universal_adjustments", {})

    sizing = plan.get("sizing", {})
    if isinstance(sizing, list):
        sizing = {}

    quality = plan.get("quality", {})
    if isinstance(quality, list):
        quality = {}

    indicators = plan.get("indicators", {})
    if isinstance(indicators, list):
        indicators = {}

    entry = plan.get("entry", {})
    if isinstance(entry, list):
        entry = {}

    stop = plan.get("stop", {})
    if isinstance(stop, list):
        stop = {}

    # Calculate charges
    entry_price = outcome.get("actual_entry_price", 0) or outcome.get("entry_reference", 0)
    exit_price = outcome.get("exit_price", 0)
    qty = outcome.get("qty", 0)
    total_exits = outcome.get("total_exits", 1)

    entry_value = entry_price * qty if entry_price and qty else 0
    exit_value = exit_price * qty if exit_price and qty else 0
    charges = calculate_trade_charges(entry_value, exit_value, total_exits) if entry_value > 0 else 50

    gross_pnl = outcome.get("total_trade_pnl", 0)
    net_pnl = gross_pnl - charges

    return {
        # Identifiers
        "trade_id": decision.get("trade_id"),
        "symbol": decision.get("symbol"),
        "date": decision.get("_date"),

        # Decision fields
        "setup_type": dec.get("setup_type", plan.get("strategy", "unknown")),
        "regime": dec.get("regime", plan.get("regime", "unknown")),
        "category": plan.get("category", "unknown"),
        "bias": plan.get("bias", "unknown"),

        # Ranking
        "rank_score": ranking.get("score", features.get("rank_score", 0)),
        "rank_base_score": ranking.get("base_score", 0),

        # Ranking components
        "comp_volume": components.get("volume", 0),
        "comp_rsi": components.get("rsi", 0),
        "comp_adx": components.get("adx", 0),
        "comp_vwap": components.get("vwap", 0),
        "comp_distance": components.get("distance", 0),
        "comp_squeeze": components.get("squeeze", 0),
        "comp_acceptance": components.get("acceptance", 0),

        # Multipliers
        "mult_regime": multipliers.get("regime", 1),
        "mult_score_scale": multipliers.get("score_scale", 1),

        # Universal adjustments
        "adj_time_of_day": adjustments.get("time_of_day_mult", 1),
        "adj_htf_15m": adjustments.get("htf_15m_mult", 1),
        "adj_daily_trend": adjustments.get("daily_trend_mult", 1),
        "adj_multi_tf_daily": adjustments.get("multi_tf_daily_mult", 1),
        "adj_rr_penalty": adjustments.get("rr_penalty", 0),

        # Sizing
        "qty": sizing.get("qty", qty),
        "notional": sizing.get("notional", 0),
        "risk_rupees": sizing.get("risk_rupees", 0),
        "risk_per_share": sizing.get("risk_per_share", 0),
        "size_mult": sizing.get("size_mult", dec.get("size_mult", 1)),
        "cap_segment": sizing.get("cap_segment", "unknown"),
        "cap_size_mult": sizing.get("cap_size_mult", 1),

        # Quality
        "structural_rr": quality.get("structural_rr", 0),
        "quality_status": quality.get("status", "unknown"),

        # Indicators
        "ind_atr": indicators.get("atr", 0),
        "ind_adx": indicators.get("adx", 0),
        "ind_rsi": indicators.get("rsi", 50),
        "ind_vwap": indicators.get("vwap", 0),

        # Entry/Stop
        "entry_ref_price": plan.get("entry_ref_price", entry_price),
        "stop_hard": stop.get("hard", 0),

        # Outcomes
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": outcome.get("reason", "unknown"),
        "gross_pnl": gross_pnl,
        "charges": charges,
        "net_pnl": net_pnl,
        "total_exits": total_exits,

        # Win/Loss
        "is_winner": net_pnl > 0,
    }


def link_decisions_to_outcomes(decisions: Dict, outcomes: Dict) -> List[Dict]:
    """
    Link DECISION events to their final trade outcomes.
    """
    linked_trades = []

    for trade_id, decision in decisions.items():
        if trade_id not in outcomes:
            continue

        outcome = outcomes[trade_id]
        trade = extract_flat_features(decision, outcome)
        linked_trades.append(trade)

    return linked_trades


def analyze_parameter_by_outcome(trades: List[Dict], param: str, bins: int = 8) -> None:
    """Analyze how a parameter correlates with outcomes."""
    values = [(t[param], t["net_pnl"], t["is_winner"]) for t in trades if t.get(param) is not None]

    if not values:
        return

    # Check if numeric
    sample_val = values[0][0]
    if not isinstance(sample_val, (int, float)):
        # Categorical
        groups = defaultdict(list)
        for val, pnl, win in values:
            groups[str(val)].append((pnl, win))

        print(f"\n{param}:")
        print("-" * 70)
        print(f"{'Value':<30} {'Trades':>8} {'Win%':>8} {'AvgPnL':>10} {'TotalPnL':>12}")

        for group_name in sorted(groups.keys()):
            group_data = groups[group_name]
            count = len(group_data)
            if count < 10:
                continue
            wins = sum(1 for _, w in group_data if w)
            win_rate = wins / count * 100
            avg_pnl = statistics.mean([p for p, _ in group_data])
            total_pnl = sum(p for p, _ in group_data)
            print(f"{group_name[:30]:<30} {count:>8} {win_rate:>7.1f}% {avg_pnl:>10.0f} {total_pnl:>12,.0f}")
    else:
        # Numeric - create bins
        numeric_vals = [v for v, _, _ in values if isinstance(v, (int, float)) and v == v]
        if not numeric_vals:
            return

        min_val = min(numeric_vals)
        max_val = max(numeric_vals)

        if min_val == max_val:
            return

        bin_size = (max_val - min_val) / bins
        bin_edges = [min_val + i * bin_size for i in range(bins + 1)]

        binned = defaultdict(list)
        for val, pnl, win in values:
            if not isinstance(val, (int, float)) or val != val:
                continue
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= val < bin_edges[i + 1] or (i == len(bin_edges) - 2 and val == bin_edges[-1]):
                    label = f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
                    binned[label].append((pnl, win))
                    break

        print(f"\n{param}:")
        print("-" * 70)
        print(f"{'Range':<25} {'Trades':>8} {'Win%':>8} {'AvgPnL':>10} {'TotalPnL':>12}")

        for i in range(len(bin_edges) - 1):
            label = f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
            if label not in binned or len(binned[label]) < 10:
                continue
            group_data = binned[label]
            count = len(group_data)
            wins = sum(1 for _, w in group_data if w)
            win_rate = wins / count * 100
            avg_pnl = statistics.mean([p for p, _ in group_data])
            total_pnl = sum(p for p, _ in group_data)
            print(f"{label:<25} {count:>8} {win_rate:>7.1f}% {avg_pnl:>10.0f} {total_pnl:>12,.0f}")


def find_discriminating_patterns(trades: List[Dict]) -> None:
    """Find parameters that best discriminate winners from losers."""
    winners = [t for t in trades if t["is_winner"]]
    losers = [t for t in trades if not t["is_winner"]]

    print(f"\n{'='*80}")
    print(f"WINNER VS LOSER COMPARISON")
    print(f"{'='*80}")
    print(f"Total: {len(trades):,} | Winners: {len(winners):,} ({len(winners)/len(trades)*100:.1f}%) | Losers: {len(losers):,}")

    # Parameters to compare
    numeric_params = [
        "rank_score", "rank_base_score",
        "comp_volume", "comp_rsi", "comp_adx", "comp_vwap", "comp_distance", "comp_squeeze", "comp_acceptance",
        "mult_regime", "adj_htf_15m", "adj_daily_trend",
        "structural_rr", "ind_adx", "ind_rsi", "ind_atr",
        "size_mult", "cap_size_mult", "risk_rupees",
    ]

    print(f"\n{'Parameter':<25} {'Winner Avg':>12} {'Loser Avg':>12} {'Diff%':>10} {'Direction':<15}")
    print("-" * 80)

    significant = []
    for param in numeric_params:
        winner_vals = [t[param] for t in winners if t.get(param) is not None and isinstance(t[param], (int, float))]
        loser_vals = [t[param] for t in losers if t.get(param) is not None and isinstance(t[param], (int, float))]

        if not winner_vals or not loser_vals:
            continue

        w_avg = statistics.mean(winner_vals)
        l_avg = statistics.mean(loser_vals)
        diff_pct = ((w_avg - l_avg) / abs(l_avg) * 100) if l_avg != 0 else 0

        direction = "HIGHER=BETTER" if w_avg > l_avg else "LOWER=BETTER"
        if abs(diff_pct) > 5:
            significant.append((param, diff_pct, direction))

        print(f"{param:<25} {w_avg:>12.3f} {l_avg:>12.3f} {diff_pct:>+9.1f}% {direction:<15}")

    print(f"\n{'='*80}")
    print("SIGNIFICANT DISCRIMINATORS (>5% difference)")
    print(f"{'='*80}")
    for param, diff, direction in sorted(significant, key=lambda x: abs(x[1]), reverse=True):
        print(f"  {param}: {diff:+.1f}% ({direction})")


def analyze_by_category(trades: List[Dict]) -> None:
    """Analyze performance by categorical variables."""
    categories = ["setup_type", "regime", "category", "cap_segment", "exit_reason"]

    for cat in categories:
        print(f"\n{'='*80}")
        print(f"ANALYSIS BY {cat.upper()}")
        print(f"{'='*80}")

        groups = defaultdict(list)
        for t in trades:
            val = t.get(cat, "unknown")
            groups[val].append(t)

        # Sort by total net PnL
        sorted_groups = sorted(groups.items(), key=lambda x: sum(t["net_pnl"] for t in x[1]), reverse=True)

        print(f"{'Value':<35} {'Trades':>8} {'Win%':>8} {'AvgNet':>10} {'TotalNet':>12}")
        print("-" * 80)

        for val, group_trades in sorted_groups:
            if len(group_trades) < 10:
                continue
            wins = sum(1 for t in group_trades if t["is_winner"])
            win_rate = wins / len(group_trades) * 100
            avg_net = statistics.mean([t["net_pnl"] for t in group_trades])
            total_net = sum(t["net_pnl"] for t in group_trades)
            print(f"{str(val)[:35]:<35} {len(group_trades):>8} {win_rate:>7.1f}% {avg_net:>10.0f} {total_net:>12,.0f}")


def analyze_ranking_components(trades: List[Dict]) -> None:
    """Deep dive into ranking component effectiveness."""
    print(f"\n{'='*80}")
    print("RANKING COMPONENT ANALYSIS")
    print(f"{'='*80}")

    components = ["comp_volume", "comp_rsi", "comp_adx", "comp_vwap", "comp_distance", "comp_squeeze", "comp_acceptance"]

    for comp in components:
        analyze_parameter_by_outcome(trades, comp, bins=6)


def find_optimal_filters(trades: List[Dict]) -> None:
    """Find filter combinations that achieve target avg PnL."""
    print(f"\n{'='*80}")
    print("OPTIMAL FILTER SEARCH (Target: Avg Net PnL >= Rs 200)")
    print(f"{'='*80}")

    baseline = len(trades)
    baseline_avg = statistics.mean([t["net_pnl"] for t in trades])
    baseline_total = sum(t["net_pnl"] for t in trades)

    print(f"\nBaseline: {baseline:,} trades, Avg: Rs {baseline_avg:.0f}, Total: Rs {baseline_total:,.0f}")

    # Filter tests
    filters = [
        ("rank_score >= 2.0", lambda t: t.get("rank_score", 0) >= 2.0),
        ("rank_score >= 3.0", lambda t: t.get("rank_score", 0) >= 3.0),
        ("rank_score >= 4.0", lambda t: t.get("rank_score", 0) >= 4.0),
        ("rank_score >= 5.0", lambda t: t.get("rank_score", 0) >= 5.0),
        ("comp_volume >= 1.0", lambda t: t.get("comp_volume", 0) >= 1.0),
        ("comp_volume >= 1.5", lambda t: t.get("comp_volume", 0) >= 1.5),
        ("comp_adx >= 0.5", lambda t: t.get("comp_adx", 0) >= 0.5),
        ("comp_adx >= 1.0", lambda t: t.get("comp_adx", 0) >= 1.0),
        ("ind_adx >= 20", lambda t: t.get("ind_adx", 0) >= 20),
        ("ind_adx >= 25", lambda t: t.get("ind_adx", 0) >= 25),
        ("ind_adx >= 30", lambda t: t.get("ind_adx", 0) >= 30),
        ("structural_rr >= 1.5", lambda t: t.get("structural_rr", 0) >= 1.5),
        ("structural_rr >= 2.0", lambda t: t.get("structural_rr", 0) >= 2.0),
        ("adj_htf_15m >= 1.0", lambda t: t.get("adj_htf_15m", 1) >= 1.0),
        ("rank >= 3 AND adx >= 25", lambda t: t.get("rank_score", 0) >= 3.0 and t.get("ind_adx", 0) >= 25),
        ("rank >= 4 AND volume >= 1.0", lambda t: t.get("rank_score", 0) >= 4.0 and t.get("comp_volume", 0) >= 1.0),
        ("rank >= 3 AND rr >= 1.5", lambda t: t.get("rank_score", 0) >= 3.0 and t.get("structural_rr", 0) >= 1.5),
        ("rank >= 4 AND adx >= 25 AND rr >= 1.5", lambda t: t.get("rank_score", 0) >= 4.0 and t.get("ind_adx", 0) >= 25 and t.get("structural_rr", 0) >= 1.5),
    ]

    print(f"\n{'Filter':<45} {'Trades':>8} {'%Kept':>7} {'AvgNet':>10} {'TotalNet':>12} {'Target':>7}")
    print("-" * 95)

    for name, func in filters:
        filtered = [t for t in trades if func(t)]
        if len(filtered) < 100:
            continue

        avg = statistics.mean([t["net_pnl"] for t in filtered])
        total = sum(t["net_pnl"] for t in filtered)
        pct = len(filtered) / baseline * 100
        target = "YES" if avg >= 200 else "NO"

        print(f"{name:<45} {len(filtered):>8} {pct:>6.1f}% {avg:>10.0f} {total:>12,.0f} {target:>7}")


def analyze_setup_regime_combo(trades: List[Dict]) -> None:
    """Analyze setup + regime combinations."""
    print(f"\n{'='*80}")
    print("SETUP + REGIME COMBINATIONS (Top 20 / Bottom 20)")
    print(f"{'='*80}")

    combos = defaultdict(list)
    for t in trades:
        combo = f"{t.get('setup_type', 'unk')} | {t.get('regime', 'unk')}"
        combos[combo].append(t)

    sorted_combos = sorted(combos.items(), key=lambda x: sum(t["net_pnl"] for t in x[1]), reverse=True)

    print(f"\n{'Combo':<50} {'Trades':>7} {'Win%':>7} {'AvgNet':>9} {'TotalNet':>12}")
    print("-" * 90)

    print("\nTOP 20:")
    for combo, combo_trades in sorted_combos[:20]:
        if len(combo_trades) < 10:
            continue
        wins = sum(1 for t in combo_trades if t["is_winner"])
        win_rate = wins / len(combo_trades) * 100
        avg = statistics.mean([t["net_pnl"] for t in combo_trades])
        total = sum(t["net_pnl"] for t in combo_trades)
        print(f"{combo[:50]:<50} {len(combo_trades):>7} {win_rate:>6.1f}% {avg:>9.0f} {total:>12,.0f}")

    print("\nBOTTOM 20:")
    for combo, combo_trades in sorted_combos[-20:]:
        if len(combo_trades) < 10:
            continue
        wins = sum(1 for t in combo_trades if t["is_winner"])
        win_rate = wins / len(combo_trades) * 100
        avg = statistics.mean([t["net_pnl"] for t in combo_trades])
        total = sum(t["net_pnl"] for t in combo_trades)
        print(f"{combo[:50]:<50} {len(combo_trades):>7} {win_rate:>6.1f}% {avg:>9.0f} {total:>12,.0f}")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("DEEP WINNER VS LOSER ANALYSIS")
    print("=" * 80)

    base_path = Path(__file__).parent.parent

    print("\n[1/4] Parsing all backtest data...")
    decisions, outcomes = parse_all_data(str(base_path))

    print("\n[2/4] Linking decisions to outcomes...")
    trades = link_decisions_to_outcomes(decisions, outcomes)
    print(f"Linked {len(trades):,} trades with complete data")

    if not trades:
        print("No trades to analyze!")
        return

    # Overall stats
    total_net = sum(t["net_pnl"] for t in trades)
    avg_net = statistics.mean([t["net_pnl"] for t in trades])
    win_rate = sum(1 for t in trades if t["is_winner"]) / len(trades) * 100

    print(f"\nOVERALL (Net of Charges):")
    print(f"  Trades: {len(trades):,}")
    print(f"  Total Net PnL: Rs {total_net:,.0f}")
    print(f"  Avg Net PnL: Rs {avg_net:.0f}")
    print(f"  Win Rate: {win_rate:.1f}%")

    print("\n[3/4] Analyzing patterns...")
    find_discriminating_patterns(trades)
    analyze_by_category(trades)
    analyze_ranking_components(trades)
    analyze_setup_regime_combo(trades)

    print("\n[4/4] Finding optimal filters...")
    find_optimal_filters(trades)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
