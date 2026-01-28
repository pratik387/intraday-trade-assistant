"""
Deep Forensic Analysis of First Hour Momentum (FHM) Trades

Analyzes:
1. Winners vs Losers parameter comparison (RVOL, price_move, volume, ADX, entry timing)
2. Stop loss analysis - would tighter/wider stops help?
3. MFE/MAE analysis - how far did price go in our favor before reversing?
4. Entry timing analysis - which 5-min bars work best?
5. RVOL threshold analysis - what's the optimal RVOL cutoff?
6. Exit reason breakdown - hard_sl, t1, t2, eod?
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

# Backtest directory
BACKTEST_DIR = Path(r"C:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251208-040138_extracted")

def load_fhm_trades():
    """Load all FHM trades from backtest sessions."""
    fhm_decisions = []
    fhm_triggers = []
    fhm_exits = []

    # Walk through all session folders
    for session_dir in BACKTEST_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        events_file = session_dir / "events.jsonl"
        analytics_file = session_dir / "analytics.jsonl"

        # Load decisions and triggers from events.jsonl
        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if "first_hour_momentum" in str(event.get("decision", {}).get("setup_type", "") or event.get("trigger", {}).get("strategy", "")):
                            if event.get("type") == "DECISION":
                                fhm_decisions.append(event)
                            elif event.get("type") == "TRIGGER":
                                fhm_triggers.append(event)
                            elif event.get("type") == "EXIT":
                                fhm_exits.append(event)
                    except:
                        pass

        # Load exits from analytics.jsonl
        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if "first_hour_momentum" in str(event.get("setup_type", "")):
                            fhm_exits.append(event)
                    except:
                        pass

    return fhm_decisions, fhm_triggers, fhm_exits


def analyze_winners_vs_losers(decisions, exits):
    """Compare parameters between winning and losing trades."""

    # Create lookup by trade_id
    exit_lookup = {}
    for ex in exits:
        trade_id = ex.get("trade_id")
        if trade_id:
            pnl = ex.get("exit", {}).get("pnl") or ex.get("pnl", 0)
            exit_lookup[trade_id] = {
                "pnl": pnl,
                "exit_reason": ex.get("exit", {}).get("reason") or ex.get("exit_reason", ""),
                "mfe": ex.get("exit", {}).get("diagnostics", {}).get("mfe") or ex.get("mfe"),
                "mae": ex.get("exit", {}).get("diagnostics", {}).get("mae") or ex.get("mae"),
            }

    winners = {"long": [], "short": []}
    losers = {"long": [], "short": []}

    for dec in decisions:
        trade_id = dec.get("trade_id")
        if trade_id not in exit_lookup:
            continue

        exit_info = exit_lookup[trade_id]
        pnl = exit_info["pnl"]
        if pnl is None:
            continue

        setup_type = dec.get("decision", {}).get("setup_type", "")
        plan = dec.get("plan", {})
        indicators = plan.get("indicators", {})
        bar5 = dec.get("bar5", {})

        bias = "long" if "_long" in setup_type else "short"

        trade_data = {
            "trade_id": trade_id,
            "symbol": dec.get("symbol"),
            "pnl": pnl,
            "exit_reason": exit_info["exit_reason"],
            "mfe": exit_info["mfe"],
            "mae": exit_info["mae"],
            # Entry parameters
            "entry_ref_price": plan.get("entry_ref_price"),
            "stop_hard": plan.get("stop", {}).get("hard"),
            "risk_per_share": plan.get("stop", {}).get("risk_per_share"),
            # Indicators at decision time
            "atr": indicators.get("atr"),
            "adx": indicators.get("adx"),
            "rsi": indicators.get("rsi"),
            "vwap": indicators.get("vwap"),
            # Bar5 data
            "bar5_volume": bar5.get("volume"),
            "bar5_adx": bar5.get("adx"),
            "bar5_vwap": bar5.get("vwap"),
            # Quality metrics
            "structural_rr": plan.get("quality", {}).get("structural_rr"),
            "rank_score": plan.get("ranking", {}).get("score"),
            # Targets
            "t1_level": plan.get("targets", [{}])[0].get("level") if plan.get("targets") else None,
            "t1_rr": plan.get("targets", [{}])[0].get("rr") if plan.get("targets") else None,
            # Regime
            "regime": plan.get("regime") or dec.get("decision", {}).get("regime"),
            # Timing
            "decision_ts": dec.get("ts"),
            # Pipeline reasons (contains RVOL info)
            "pipeline_reasons": plan.get("pipeline_reasons", []),
        }

        # Extract RVOL from pipeline_reasons
        for reason in trade_data["pipeline_reasons"]:
            if "fhm_volume_ok:" in reason:
                try:
                    rvol_str = reason.split(":")[1].replace("x", "")
                    trade_data["rvol"] = float(rvol_str)
                except:
                    pass

        if pnl > 0:
            winners[bias].append(trade_data)
        else:
            losers[bias].append(trade_data)

    return winners, losers


def print_comparison(winners, losers, field, label):
    """Print comparison of a field between winners and losers."""
    w_vals = [t[field] for t in winners if t.get(field) is not None]
    l_vals = [t[field] for t in losers if t.get(field) is not None]

    if not w_vals or not l_vals:
        print(f"  {label}: Insufficient data")
        return

    w_avg = statistics.mean(w_vals)
    l_avg = statistics.mean(l_vals)

    diff_pct = ((w_avg - l_avg) / l_avg * 100) if l_avg != 0 else 0

    print(f"  {label}:")
    print(f"    Winners: {w_avg:.2f} (n={len(w_vals)})")
    print(f"    Losers:  {l_avg:.2f} (n={len(l_vals)})")
    print(f"    Diff:    {diff_pct:+.1f}%")


def analyze_sl_optimization(losers):
    """Analyze if tighter/wider stops would help."""
    print("\n" + "="*60)
    print("STOP LOSS OPTIMIZATION ANALYSIS")
    print("="*60)

    for bias in ["long", "short"]:
        if not losers[bias]:
            continue

        print(f"\n{bias.upper()} LOSERS:")

        sl_hit_trades = [t for t in losers[bias] if t.get("exit_reason") == "hard_sl"]
        print(f"  Total SL hits: {len(sl_hit_trades)}")

        if not sl_hit_trades:
            continue

        # Analyze MAE (Maximum Adverse Excursion)
        mae_vals = [abs(t["mae"]) for t in sl_hit_trades if t.get("mae") is not None]
        mfe_vals = [t["mfe"] for t in sl_hit_trades if t.get("mfe") is not None]
        risk_vals = [t["risk_per_share"] for t in sl_hit_trades if t.get("risk_per_share") is not None]

        if mae_vals:
            print(f"\n  MAE Analysis (how far price went against us):")
            print(f"    Avg MAE: {statistics.mean(mae_vals):.2f}")
            print(f"    Max MAE: {max(mae_vals):.2f}")
            print(f"    Min MAE: {min(mae_vals):.2f}")

        if mfe_vals:
            print(f"\n  MFE Analysis (how far price went in our favor before SL):")
            positive_mfe = [m for m in mfe_vals if m and m > 0]
            if positive_mfe:
                print(f"    Trades that went positive first: {len(positive_mfe)}/{len(sl_hit_trades)}")
                print(f"    Avg positive MFE before SL: {statistics.mean(positive_mfe):.2f}")

        if risk_vals:
            print(f"\n  Risk per share (stop distance):")
            print(f"    Avg risk: {statistics.mean(risk_vals):.2f}")

        # Spike test: what if stop was tighter?
        print(f"\n  SPIKE TEST - Tighter Stops:")
        for mult in [0.5, 0.75]:
            saved = 0
            for t in sl_hit_trades:
                if t.get("mae") is not None and t.get("risk_per_share"):
                    tighter_risk = t["risk_per_share"] * mult
                    if abs(t["mae"]) <= tighter_risk:
                        # Would have still hit SL
                        pass
                    else:
                        # Would have avoided this SL
                        saved += 1
            print(f"    {mult}x risk: Would avoid {saved}/{len(sl_hit_trades)} SL hits")


def analyze_exit_reasons(winners, losers):
    """Analyze exit reasons."""
    print("\n" + "="*60)
    print("EXIT REASON BREAKDOWN")
    print("="*60)

    for bias in ["long", "short"]:
        print(f"\n{bias.upper()}:")

        all_trades = winners[bias] + losers[bias]
        exit_reasons = defaultdict(lambda: {"count": 0, "pnl": 0})

        for t in all_trades:
            reason = t.get("exit_reason", "unknown")
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["pnl"] += t.get("pnl", 0)

        for reason, data in sorted(exit_reasons.items(), key=lambda x: x[1]["count"], reverse=True):
            avg_pnl = data["pnl"] / data["count"] if data["count"] > 0 else 0
            print(f"  {reason}: {data['count']} trades, Total PnL: {data['pnl']:.2f}, Avg: {avg_pnl:.2f}")


def analyze_timing(winners, losers):
    """Analyze entry timing (which 5-min bar works best)."""
    print("\n" + "="*60)
    print("ENTRY TIMING ANALYSIS")
    print("="*60)

    for bias in ["long", "short"]:
        print(f"\n{bias.upper()}:")

        all_trades = winners[bias] + losers[bias]
        timing_buckets = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})

        for t in all_trades:
            ts = t.get("decision_ts", "")
            if ts:
                # Extract time (HH:MM)
                try:
                    time_part = ts.split(" ")[1][:5]
                    timing_buckets[time_part]["pnl"] += t.get("pnl", 0)
                    if t.get("pnl", 0) > 0:
                        timing_buckets[time_part]["wins"] += 1
                    else:
                        timing_buckets[time_part]["losses"] += 1
                except:
                    pass

        for time, data in sorted(timing_buckets.items()):
            total = data["wins"] + data["losses"]
            wr = data["wins"] / total * 100 if total > 0 else 0
            avg_pnl = data["pnl"] / total if total > 0 else 0
            print(f"  {time}: {total} trades, WR: {wr:.0f}%, Avg PnL: {avg_pnl:.2f}")


def analyze_rvol_threshold(winners, losers):
    """Find optimal RVOL threshold."""
    print("\n" + "="*60)
    print("RVOL THRESHOLD ANALYSIS")
    print("="*60)

    for bias in ["long", "short"]:
        print(f"\n{bias.upper()}:")

        all_trades = winners[bias] + losers[bias]
        rvol_buckets = {
            "2.0-2.5x": {"wins": 0, "losses": 0, "pnl": 0},
            "2.5-3.0x": {"wins": 0, "losses": 0, "pnl": 0},
            "3.0-4.0x": {"wins": 0, "losses": 0, "pnl": 0},
            "4.0+x": {"wins": 0, "losses": 0, "pnl": 0},
        }

        for t in all_trades:
            rvol = t.get("rvol")
            if rvol is None:
                continue

            if rvol >= 4.0:
                bucket = "4.0+x"
            elif rvol >= 3.0:
                bucket = "3.0-4.0x"
            elif rvol >= 2.5:
                bucket = "2.5-3.0x"
            else:
                bucket = "2.0-2.5x"

            rvol_buckets[bucket]["pnl"] += t.get("pnl", 0)
            if t.get("pnl", 0) > 0:
                rvol_buckets[bucket]["wins"] += 1
            else:
                rvol_buckets[bucket]["losses"] += 1

        for bucket, data in rvol_buckets.items():
            total = data["wins"] + data["losses"]
            if total == 0:
                continue
            wr = data["wins"] / total * 100
            avg_pnl = data["pnl"] / total
            print(f"  {bucket}: {total} trades, WR: {wr:.0f}%, Avg PnL: {avg_pnl:.2f}, Total: {data['pnl']:.2f}")


def analyze_regime(winners, losers):
    """Analyze performance by regime."""
    print("\n" + "="*60)
    print("REGIME ANALYSIS")
    print("="*60)

    for bias in ["long", "short"]:
        print(f"\n{bias.upper()}:")

        all_trades = winners[bias] + losers[bias]
        regime_buckets = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})

        for t in all_trades:
            regime = t.get("regime", "unknown")
            regime_buckets[regime]["pnl"] += t.get("pnl", 0)
            if t.get("pnl", 0) > 0:
                regime_buckets[regime]["wins"] += 1
            else:
                regime_buckets[regime]["losses"] += 1

        for regime, data in sorted(regime_buckets.items(), key=lambda x: x[1]["pnl"]):
            total = data["wins"] + data["losses"]
            if total == 0:
                continue
            wr = data["wins"] / total * 100
            avg_pnl = data["pnl"] / total
            print(f"  {regime}: {total} trades, WR: {wr:.0f}%, Avg PnL: {avg_pnl:.2f}, Total: {data['pnl']:.2f}")


def main():
    print("="*60)
    print("DEEP FORENSIC ANALYSIS: FIRST HOUR MOMENTUM (FHM)")
    print("="*60)

    print("\nLoading FHM trades from backtest...")
    decisions, triggers, exits = load_fhm_trades()

    print(f"Found: {len(decisions)} decisions, {len(triggers)} triggers, {len(exits)} exits")

    # Separate by setup type
    long_decisions = [d for d in decisions if "long" in d.get("decision", {}).get("setup_type", "")]
    short_decisions = [d for d in decisions if "short" in d.get("decision", {}).get("setup_type", "")]

    print(f"FHM_LONG decisions: {len(long_decisions)}")
    print(f"FHM_SHORT decisions: {len(short_decisions)}")

    winners, losers = analyze_winners_vs_losers(decisions, exits)

    print(f"\nTriggered & Completed:")
    print(f"  FHM_LONG: {len(winners['long'])} winners, {len(losers['long'])} losers")
    print(f"  FHM_SHORT: {len(winners['short'])} winners, {len(losers['short'])} losers")

    # Parameter comparison
    print("\n" + "="*60)
    print("WINNERS vs LOSERS PARAMETER COMPARISON")
    print("="*60)

    for bias in ["long", "short"]:
        if not winners[bias] or not losers[bias]:
            continue

        print(f"\n{bias.upper()}:")
        print_comparison(winners[bias], losers[bias], "bar5_volume", "Bar5 Volume")
        print_comparison(winners[bias], losers[bias], "bar5_adx", "Bar5 ADX")
        print_comparison(winners[bias], losers[bias], "structural_rr", "Structural R:R")
        print_comparison(winners[bias], losers[bias], "rank_score", "Rank Score")
        print_comparison(winners[bias], losers[bias], "risk_per_share", "Risk per Share")
        print_comparison(winners[bias], losers[bias], "atr", "ATR")

    # SL Optimization
    analyze_sl_optimization(losers)

    # Exit reasons
    analyze_exit_reasons(winners, losers)

    # Timing analysis
    analyze_timing(winners, losers)

    # RVOL threshold
    analyze_rvol_threshold(winners, losers)

    # Regime analysis
    analyze_regime(winners, losers)

    # Print specific losing trades for inspection
    print("\n" + "="*60)
    print("TOP 10 BIGGEST LOSERS (for inspection)")
    print("="*60)

    all_losers = losers["long"] + losers["short"]
    all_losers.sort(key=lambda x: x.get("pnl", 0))

    for t in all_losers[:10]:
        print(f"\n{t['symbol']} ({t.get('decision_ts', '')})")
        print(f"  PnL: {t['pnl']:.2f}")
        print(f"  Exit: {t['exit_reason']}")
        print(f"  Entry: {t['entry_ref_price']}, Stop: {t['stop_hard']}, Risk: {t['risk_per_share']}")
        print(f"  Regime: {t['regime']}")
        print(f"  ADX: {t.get('adx')}, Volume: {t.get('bar5_volume')}")
        print(f"  MAE: {t.get('mae')}, MFE: {t.get('mfe')}")


if __name__ == "__main__":
    main()
