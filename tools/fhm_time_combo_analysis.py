"""
FHM Time + VWAP Combo Analysis
Test combinations of time cutoff and VWAP distance filters
"""
import json
import pandas as pd
from pathlib import Path

BACKTEST_DIR = Path(r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251209-060217_extracted")

def extract_fhm_trades():
    trades = []
    for day_dir in sorted(BACKTEST_DIR.iterdir()):
        if not day_dir.is_dir():
            continue
        events_file = day_dir / "events.jsonl"
        analytics_file = day_dir / "analytics.jsonl"
        if not events_file.exists():
            continue

        decisions, triggers, exits = {}, {}, {}
        with open(events_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                except:
                    continue
                trade_id = event.get("trade_id")
                event_type = event.get("type")
                if event_type == "DECISION":
                    setup_type = event.get("decision", {}).get("setup_type", "")
                    if "first_hour_momentum" in setup_type:
                        decisions[trade_id] = event
                elif event_type == "TRIGGER":
                    setup_type = event.get("trigger", {}).get("strategy", "")
                    if "first_hour_momentum" in setup_type:
                        triggers[trade_id] = event

        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                    except:
                        continue
                    if event.get("stage") != "EXIT" or "first_hour_momentum" not in event.get("setup_type", ""):
                        continue
                    trade_id = event.get("trade_id")
                    if trade_id not in exits:
                        exits[trade_id] = []
                    exits[trade_id].append(event)

        for trade_id, decision in decisions.items():
            if trade_id not in triggers or trade_id not in exits:
                continue
            exit_events = exits[trade_id]
            total_pnl = 0
            for e in exit_events:
                if e.get("is_final_exit"):
                    total_pnl = e.get("total_trade_pnl", e.get("pnl", 0)) or 0
                    break
            if total_pnl == 0:
                total_pnl = sum(e.get("pnl", 0) or 0 for e in exit_events)

            dec = decision.get("decision", {})
            plan = decision.get("plan", {})
            bar5 = decision.get("bar5", {})
            sizing = plan.get("sizing", {})
            indicators = plan.get("indicators", {})
            timectx = decision.get("timectx", {})

            price_vs_vwap = None
            if bar5.get("close") and indicators.get("vwap") and indicators.get("vwap") > 0:
                price_vs_vwap = ((bar5["close"] - indicators["vwap"]) / indicators["vwap"]) * 100

            trades.append({
                "trade_id": trade_id,
                "bias": plan.get("bias"),
                "regime": dec.get("regime"),
                "cap_segment": sizing.get("cap_segment"),
                "total_pnl": total_pnl,
                "is_winner": total_pnl > 0,
                "bar5_volume": bar5.get("volume"),
                "minute_of_day": timectx.get("minute_of_day"),
                "price_vs_vwap_pct": price_vs_vwap,
            })
    return trades


def test_combos(df, name, baseline_pnl, baseline_avg):
    """Test time + VWAP combos"""
    print(f"\n--- {name} COMBO FILTERS ---")

    combos = []
    # Time cutoffs + VWAP distance
    for max_min in [570, 585, 600, 615]:  # 09:30, 09:45, 10:00, 10:15
        for vwap_dist in [0.3, 0.5, 1.0, 1.5]:
            mask = (df['minute_of_day'] <= max_min) & (df['price_vs_vwap_pct'].abs() <= vwap_dist)
            subset = df[mask]
            if len(subset) >= 5:
                w = subset['is_winner'].sum()
                p = subset['total_pnl'].sum()
                wr = w / len(subset) * 100
                avg = p / len(subset)
                hour = max_min // 60
                minute = max_min % 60
                combos.append({
                    "filter": f"<{hour:02d}:{minute:02d} + |VWAP|<={vwap_dist}%",
                    "trades": len(subset),
                    "wr": wr,
                    "pnl": p,
                    "avg": avg,
                })

    combos.sort(key=lambda x: x['avg'], reverse=True)
    print(f"BASELINE: {len(df)} trades, PnL={baseline_pnl:,.0f}, Avg={baseline_avg:.1f}/trade\n")
    print(f"{'Filter':<30} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'Avg':>10} {'Imp':>8}")
    print("-" * 75)
    for c in combos[:10]:
        imp = ((c['avg'] - baseline_avg) / abs(baseline_avg) * 100) if baseline_avg != 0 else 0
        status = "BETTER" if c['avg'] > baseline_avg else ""
        print(f"{c['filter']:<30} {c['trades']:>7} {c['wr']:>6.1f}% {c['pnl']:>9,.0f} {c['avg']:>9,.1f} {imp:>7.0f}% {status}")


def main():
    print("="*70)
    print("FHM TIME + VWAP COMBINATION ANALYSIS")
    print("="*70)

    trades = extract_fhm_trades()
    df = pd.DataFrame(trades)

    # LONG MID_CAP
    long_midcap = df[(df['bias'] == 'long') & (df['cap_segment'] == 'mid_cap')]
    if len(long_midcap) >= 10:
        baseline_pnl = long_midcap['total_pnl'].sum()
        baseline_avg = baseline_pnl / len(long_midcap)
        test_combos(long_midcap, "LONG MID_CAP", baseline_pnl, baseline_avg)

    # SHORTS
    shorts = df[df['bias'] == 'short']
    if len(shorts) >= 10:
        baseline_pnl = shorts['total_pnl'].sum()
        baseline_avg = baseline_pnl / len(shorts)
        test_combos(shorts, "SHORTS (ALL)", baseline_pnl, baseline_avg)

    # SHORTS CHOP ONLY (what the config should enforce)
    shorts_chop = shorts[shorts['regime'] == 'chop']
    if len(shorts_chop) >= 10:
        baseline_pnl = shorts_chop['total_pnl'].sum()
        baseline_avg = baseline_pnl / len(shorts_chop)
        test_combos(shorts_chop, "SHORTS CHOP ONLY", baseline_pnl, baseline_avg)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - EXPECTED RESULTS WITH FILTERS")
    print("="*70)

    # LONG MID_CAP with time filter
    lm_time = long_midcap[long_midcap['minute_of_day'] <= 600]  # Before 10:00
    print(f"\nLONG MID_CAP (before 10:00): {len(lm_time)} trades, PnL={lm_time['total_pnl'].sum():,.0f} Rs")

    # SHORTS CHOP with time filter
    sc_time = shorts_chop[shorts_chop['minute_of_day'] <= 600]
    print(f"SHORTS CHOP (before 10:00): {len(sc_time)} trades, PnL={sc_time['total_pnl'].sum():,.0f} Rs")

    # Combined
    combined = len(lm_time) + len(sc_time)
    combined_pnl = lm_time['total_pnl'].sum() + sc_time['total_pnl'].sum()
    print(f"\nCOMBINED: {combined} trades, PnL={combined_pnl:,.0f} Rs, Avg={combined_pnl/combined:.1f}/trade")
    print(f"vs BASELINE: 438 trades, PnL=-27,794 Rs")


if __name__ == "__main__":
    main()
