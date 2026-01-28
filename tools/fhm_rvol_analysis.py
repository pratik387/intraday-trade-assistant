"""
FHM RVOL-Based Filter Analysis
Since FHM is RVOL-based (not ADX), analyze filters that align with RVOL momentum:
- RVOL threshold
- Volume absolute
- Price move %
- Time within window
- VWAP position
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

BACKTEST_DIR = Path(r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251209-060217_extracted")

def extract_fhm_trades():
    """Extract all FHM trades with RVOL-specific parameters"""
    trades = []

    for day_dir in sorted(BACKTEST_DIR.iterdir()):
        if not day_dir.is_dir():
            continue

        events_file = day_dir / "events.jsonl"
        analytics_file = day_dir / "analytics.jsonl"
        if not events_file.exists():
            continue

        decisions = {}
        triggers = {}
        exits = {}

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
                    if event.get("stage") != "EXIT":
                        continue
                    setup_type = event.get("setup_type", "")
                    if "first_hour_momentum" not in setup_type:
                        continue
                    trade_id = event.get("trade_id")
                    if trade_id not in exits:
                        exits[trade_id] = []
                    exits[trade_id].append(event)

        for trade_id, decision in decisions.items():
            if trade_id not in triggers or trade_id not in exits:
                continue
            trigger = triggers[trade_id]
            exit_events = exits[trade_id]

            # Get total PnL
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
            features = decision.get("features", {})
            sizing = plan.get("sizing", {})
            indicators = plan.get("indicators", {})
            timectx = decision.get("timectx", {})

            # Get RVOL-specific data
            trade = {
                "trade_id": trade_id,
                "date": day_dir.name,
                "symbol": decision.get("symbol"),
                "setup_type": dec.get("setup_type"),
                "bias": plan.get("bias"),
                "regime": dec.get("regime"),
                "cap_segment": sizing.get("cap_segment"),
                "total_pnl": total_pnl,
                "is_winner": total_pnl > 0,

                # RVOL-specific
                "rvol": features.get("fhm_rvol"),
                "price_move_pct": features.get("fhm_price_move_pct"),

                # Bar5 data (at decision time)
                "bar5_volume": bar5.get("volume"),
                "bar5_close": bar5.get("close"),
                "bar5_open": bar5.get("open"),
                "bar5_high": bar5.get("high"),
                "bar5_low": bar5.get("low"),

                # Indicators
                "vwap": indicators.get("vwap"),
                "atr": indicators.get("atr"),
                "rsi": indicators.get("rsi"),

                # Price vs VWAP
                "price_vs_vwap_pct": None,

                # Timing
                "minute_of_day": timectx.get("minute_of_day"),

                # Entry/SL
                "entry_price": plan.get("entry_ref_price"),
                "hard_sl": plan.get("stop", {}).get("hard"),
                "structural_rr": plan.get("quality", {}).get("structural_rr"),
            }

            # Calculate price vs VWAP %
            if trade["bar5_close"] and trade["vwap"] and trade["vwap"] > 0:
                trade["price_vs_vwap_pct"] = ((trade["bar5_close"] - trade["vwap"]) / trade["vwap"]) * 100

            trades.append(trade)

    return trades


def analyze_rvol_filters(df, segment_name):
    """Analyze RVOL-specific filters"""
    print(f"\n{'='*70}")
    print(f"RVOL-BASED FILTER ANALYSIS: {segment_name}")
    print(f"{'='*70}")

    total = len(df)
    winners = df['is_winner'].sum()
    pnl = df['total_pnl'].sum()
    wr = winners / total * 100 if total > 0 else 0
    avg_pnl = pnl / total if total > 0 else 0

    print(f"\nBASELINE: {total} trades, {winners}W, WR={wr:.1f}%, PnL={pnl:,.0f} Rs, Avg={avg_pnl:,.1f}/trade")

    filters = []

    # RVOL filters (since FHM is RVOL-based, this is KEY)
    print("\n--- RVOL FILTERS (Core FHM metric) ---")
    for rvol_min in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        mask = df['rvol'] >= rvol_min
        subset = df[mask]
        if len(subset) >= 5:
            w = subset['is_winner'].sum()
            p = subset['total_pnl'].sum()
            wr = w / len(subset) * 100
            avg = p / len(subset)
            status = "BETTER" if avg > avg_pnl else ""
            print(f"  RVOL >= {rvol_min}: {len(subset):>4} trades, WR={wr:>5.1f}%, PnL={p:>10,.0f}, Avg={avg:>8,.1f}/trade {status}")
            filters.append({"filter": f"RVOL >= {rvol_min}", "trades": len(subset), "wr": wr, "pnl": p, "avg": avg})

    # Price move % filters
    print("\n--- PRICE MOVE % FILTERS ---")
    for move_min in [1.0, 1.5, 2.0, 2.5, 3.0]:
        mask = df['price_move_pct'].abs() >= move_min
        subset = df[mask]
        if len(subset) >= 5:
            w = subset['is_winner'].sum()
            p = subset['total_pnl'].sum()
            wr = w / len(subset) * 100
            avg = p / len(subset)
            status = "BETTER" if avg > avg_pnl else ""
            print(f"  Move >= {move_min}%: {len(subset):>4} trades, WR={wr:>5.1f}%, PnL={p:>10,.0f}, Avg={avg:>8,.1f}/trade {status}")
            filters.append({"filter": f"Move >= {move_min}%", "trades": len(subset), "wr": wr, "pnl": p, "avg": avg})

    # Volume absolute filters
    print("\n--- ABSOLUTE VOLUME FILTERS ---")
    for vol_min in [50000, 100000, 150000, 200000, 300000, 500000]:
        mask = df['bar5_volume'] >= vol_min
        subset = df[mask]
        if len(subset) >= 5:
            w = subset['is_winner'].sum()
            p = subset['total_pnl'].sum()
            wr = w / len(subset) * 100
            avg = p / len(subset)
            status = "BETTER" if avg > avg_pnl else ""
            print(f"  Vol >= {vol_min/1000:.0f}k: {len(subset):>4} trades, WR={wr:>5.1f}%, PnL={p:>10,.0f}, Avg={avg:>8,.1f}/trade {status}")
            filters.append({"filter": f"Vol >= {vol_min/1000:.0f}k", "trades": len(subset), "wr": wr, "pnl": p, "avg": avg})

    # Time of day filters (minute_of_day: 555 = 09:15, 630 = 10:30)
    print("\n--- TIME FILTERS (within FHM window 09:20-10:30) ---")
    for max_minute in [570, 585, 600, 615, 630]:  # 09:30, 09:45, 10:00, 10:15, 10:30
        hour = max_minute // 60
        minute = max_minute % 60
        mask = df['minute_of_day'] <= max_minute
        subset = df[mask]
        if len(subset) >= 5:
            w = subset['is_winner'].sum()
            p = subset['total_pnl'].sum()
            wr = w / len(subset) * 100
            avg = p / len(subset)
            status = "BETTER" if avg > avg_pnl else ""
            print(f"  Before {hour:02d}:{minute:02d}: {len(subset):>4} trades, WR={wr:>5.1f}%, PnL={p:>10,.0f}, Avg={avg:>8,.1f}/trade {status}")
            filters.append({"filter": f"Before {hour:02d}:{minute:02d}", "trades": len(subset), "wr": wr, "pnl": p, "avg": avg})

    # VWAP distance filters (for longs: close to VWAP is better for pullback entry)
    print("\n--- VWAP DISTANCE FILTERS ---")
    for dist_max in [0.5, 1.0, 1.5, 2.0, 3.0]:
        mask = df['price_vs_vwap_pct'].abs() <= dist_max
        subset = df[mask]
        if len(subset) >= 5:
            w = subset['is_winner'].sum()
            p = subset['total_pnl'].sum()
            wr = w / len(subset) * 100
            avg = p / len(subset)
            status = "BETTER" if avg > avg_pnl else ""
            print(f"  |VWAP dist| <= {dist_max}%: {len(subset):>4} trades, WR={wr:>5.1f}%, PnL={p:>10,.0f}, Avg={avg:>8,.1f}/trade {status}")
            filters.append({"filter": f"|VWAP dist| <= {dist_max}%", "trades": len(subset), "wr": wr, "pnl": p, "avg": avg})

    return filters


def test_combo_filters(df, segment_name):
    """Test combinations of best RVOL-based filters"""
    print(f"\n{'='*70}")
    print(f"COMBINATION FILTERS: {segment_name}")
    print(f"{'='*70}")

    baseline_avg = df['total_pnl'].sum() / len(df) if len(df) > 0 else 0

    combos = []

    # Test RVOL + Volume combos
    for rvol in [2.0, 2.5, 3.0, 3.5]:
        for vol in [100000, 150000, 200000]:
            mask = (df['rvol'] >= rvol) & (df['bar5_volume'] >= vol)
            subset = df[mask]
            if len(subset) >= 5:
                w = subset['is_winner'].sum()
                p = subset['total_pnl'].sum()
                wr = w / len(subset) * 100
                avg = p / len(subset)
                if avg > baseline_avg:
                    combos.append({
                        "filters": f"RVOL>={rvol} + Vol>={vol/1000:.0f}k",
                        "trades": len(subset),
                        "wr": wr,
                        "pnl": p,
                        "avg": avg
                    })

    # Test RVOL + Time combos
    for rvol in [2.0, 2.5, 3.0]:
        for max_min in [585, 600, 615]:
            mask = (df['rvol'] >= rvol) & (df['minute_of_day'] <= max_min)
            subset = df[mask]
            if len(subset) >= 5:
                w = subset['is_winner'].sum()
                p = subset['total_pnl'].sum()
                wr = w / len(subset) * 100
                avg = p / len(subset)
                if avg > baseline_avg:
                    hour = max_min // 60
                    minute = max_min % 60
                    combos.append({
                        "filters": f"RVOL>={rvol} + Before {hour:02d}:{minute:02d}",
                        "trades": len(subset),
                        "wr": wr,
                        "pnl": p,
                        "avg": avg
                    })

    # Test RVOL + Move % combos
    for rvol in [2.0, 2.5, 3.0]:
        for move in [1.5, 2.0, 2.5]:
            mask = (df['rvol'] >= rvol) & (df['price_move_pct'].abs() >= move)
            subset = df[mask]
            if len(subset) >= 5:
                w = subset['is_winner'].sum()
                p = subset['total_pnl'].sum()
                wr = w / len(subset) * 100
                avg = p / len(subset)
                if avg > baseline_avg:
                    combos.append({
                        "filters": f"RVOL>={rvol} + Move>={move}%",
                        "trades": len(subset),
                        "wr": wr,
                        "pnl": p,
                        "avg": avg
                    })

    # Sort by avg PnL
    combos.sort(key=lambda x: x['avg'], reverse=True)

    print(f"\nBASELINE: {len(df)} trades, Avg={baseline_avg:.1f}/trade")
    print(f"\n{'Filters':<40} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'Avg':>10}")
    print("-" * 80)
    for c in combos[:15]:
        print(f"{c['filters']:<40} {c['trades']:>7} {c['wr']:>6.1f}% {c['pnl']:>9,.0f} {c['avg']:>9,.1f}")

    return combos


def main():
    print("="*70)
    print("FHM RVOL-BASED FILTER ANALYSIS")
    print("Finding filters that align with RVOL-based momentum structure")
    print("="*70)

    trades = extract_fhm_trades()
    df = pd.DataFrame(trades)

    print(f"\nTotal FHM trades: {len(df)}")
    print(f"Total PnL: {df['total_pnl'].sum():,.0f} Rs")

    # Check RVOL data availability
    rvol_available = df['rvol'].notna().sum()
    print(f"Trades with RVOL data: {rvol_available}")

    if rvol_available < len(df) * 0.5:
        print("\nWARNING: Less than 50% of trades have RVOL data!")
        print("RVOL might not be logged in decision events.")
        print("\nLet me check what features are available:")
        print(df.columns.tolist())

        # Check bar5 volume as proxy
        print(f"\nbar5_volume available: {df['bar5_volume'].notna().sum()}")
        print(f"Sample bar5_volume: {df['bar5_volume'].head(10).tolist()}")

    # Split by direction
    longs = df[df['bias'] == 'long']
    shorts = df[df['bias'] == 'short']

    print(f"\nLONGS: {len(longs)} trades, PnL: {longs['total_pnl'].sum():,.0f} Rs")
    print(f"SHORTS: {len(shorts)} trades, PnL: {shorts['total_pnl'].sum():,.0f} Rs")

    # Analyze LONG MID_CAP (our profitable segment)
    long_midcap = longs[longs['cap_segment'] == 'mid_cap']
    print(f"\nLONG MID_CAP: {len(long_midcap)} trades, PnL: {long_midcap['total_pnl'].sum():,.0f} Rs")

    if len(long_midcap) >= 10:
        analyze_rvol_filters(long_midcap, "LONG MID_CAP")
        test_combo_filters(long_midcap, "LONG MID_CAP")

    # Analyze SHORTS
    if len(shorts) >= 10:
        analyze_rvol_filters(shorts, "SHORTS")
        test_combo_filters(shorts, "SHORTS")

    # Regime breakdown for shorts (to verify chop filter)
    print("\n" + "="*70)
    print("SHORTS REGIME BREAKDOWN (verify chop filter)")
    print("="*70)
    for regime in shorts['regime'].dropna().unique():
        subset = shorts[shorts['regime'] == regime]
        w = subset['is_winner'].sum()
        p = subset['total_pnl'].sum()
        wr = w / len(subset) * 100 if len(subset) > 0 else 0
        avg = p / len(subset) if len(subset) > 0 else 0
        print(f"{regime:<15}: {len(subset):>4} trades, WR={wr:>5.1f}%, PnL={p:>10,.0f} Rs, Avg={avg:>8,.1f}/trade")


if __name__ == "__main__":
    main()
