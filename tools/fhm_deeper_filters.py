"""
FHM Deeper Filter Analysis - Find additional filters within profitable segments
to reduce trade count while maximizing profit
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

BACKTEST_DIR = Path(r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251209-060217_extracted")

def extract_fhm_trades():
    """Extract all FHM trades with ALL parameters"""
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

            # Calculate total PnL from ALL exits
            total_pnl = sum(e.get("total_trade_pnl", e.get("pnl", 0)) or 0 for e in exit_events if e.get("is_final_exit"))
            if total_pnl == 0:
                total_pnl = sum(e.get("pnl", 0) or 0 for e in exit_events)

            # Determine if winner
            is_winner = total_pnl > 0

            # Get all parameters
            dec = decision.get("decision", {})
            plan = decision.get("plan", {})
            sizing = plan.get("sizing", {})
            indicators = plan.get("indicators", {})
            quality = plan.get("quality", {})
            ranking = plan.get("ranking", {})

            # Regime diagnostics
            regime_diag = dec.get("regime_diagnostics") or {}

            trade = {
                "trade_id": trade_id,
                "date": day_dir.name,
                "symbol": decision.get("symbol"),
                "setup_type": dec.get("setup_type"),
                "bias": plan.get("bias"),
                "regime": dec.get("regime"),
                "cap_segment": sizing.get("cap_segment"),
                "total_pnl": total_pnl,
                "is_winner": is_winner,

                # Key indicators
                "adx": indicators.get("adx"),
                "rsi": indicators.get("rsi"),
                "atr": indicators.get("atr"),
                "vwap": indicators.get("vwap"),

                # Quality
                "structural_rr": quality.get("structural_rr"),
                "quality_status": quality.get("status"),

                # Volume
                "volume": decision.get("bar5", {}).get("volume"),

                # Ranking
                "rank_score": ranking.get("score"),

                # Regime diagnostics
                "adx_trend": regime_diag.get("adx_trend"),
                "squeeze_pctile": regime_diag.get("squeeze_pctile"),
                "bb_width_pct": regime_diag.get("bb_width_pct"),
            }

            trades.append(trade)

    return trades


def analyze_filters(trades_df, segment_name):
    """Analyze various filter combinations within a segment"""
    print(f"\n{'='*60}")
    print(f"DEEP FILTER ANALYSIS: {segment_name}")
    print(f"{'='*60}")

    total = len(trades_df)
    winners = trades_df['is_winner'].sum()
    pnl = trades_df['total_pnl'].sum()
    wr = winners / total * 100 if total > 0 else 0
    avg_pnl = pnl / total if total > 0 else 0

    print(f"\nBASELINE: {total} trades, {winners}W, WR={wr:.1f}%, PnL={pnl:,.0f} Rs, Avg={avg_pnl:,.1f}/trade")

    # Test various filters
    filters = []

    # ADX filters
    for adx_min in [15, 20, 25, 30, 35, 40]:
        mask = trades_df['adx'] >= adx_min
        subset = trades_df[mask]
        if len(subset) >= 10:
            w = subset['is_winner'].sum()
            p = subset['total_pnl'].sum()
            wr = w / len(subset) * 100
            avg = p / len(subset)
            filters.append({
                "filter": f"ADX >= {adx_min}",
                "trades": len(subset),
                "winners": w,
                "wr": wr,
                "pnl": p,
                "avg_pnl": avg
            })

    # RSI filters (direction-dependent)
    if "long" in segment_name.lower():
        # For longs: lower RSI is better (not overbought)
        for rsi_max in [50, 55, 60, 65, 70]:
            mask = trades_df['rsi'] <= rsi_max
            subset = trades_df[mask]
            if len(subset) >= 10:
                w = subset['is_winner'].sum()
                p = subset['total_pnl'].sum()
                wr = w / len(subset) * 100
                avg = p / len(subset)
                filters.append({
                    "filter": f"RSI <= {rsi_max}",
                    "trades": len(subset),
                    "winners": w,
                    "wr": wr,
                    "pnl": p,
                    "avg_pnl": avg
                })
    else:
        # For shorts: higher RSI is better (overbought)
        for rsi_min in [40, 45, 50, 55, 60]:
            mask = trades_df['rsi'] >= rsi_min
            subset = trades_df[mask]
            if len(subset) >= 10:
                w = subset['is_winner'].sum()
                p = subset['total_pnl'].sum()
                wr = w / len(subset) * 100
                avg = p / len(subset)
                filters.append({
                    "filter": f"RSI >= {rsi_min}",
                    "trades": len(subset),
                    "winners": w,
                    "wr": wr,
                    "pnl": p,
                    "avg_pnl": avg
                })

    # Volume filters
    for vol_min in [50000, 100000, 150000, 200000, 300000]:
        mask = trades_df['volume'] >= vol_min
        subset = trades_df[mask]
        if len(subset) >= 10:
            w = subset['is_winner'].sum()
            p = subset['total_pnl'].sum()
            wr = w / len(subset) * 100
            avg = p / len(subset)
            filters.append({
                "filter": f"Volume >= {vol_min/1000:.0f}k",
                "trades": len(subset),
                "winners": w,
                "wr": wr,
                "pnl": p,
                "avg_pnl": avg
            })

    # Structural RR filters
    for srr_max in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        mask = trades_df['structural_rr'] <= srr_max
        subset = trades_df[mask]
        if len(subset) >= 10:
            w = subset['is_winner'].sum()
            p = subset['total_pnl'].sum()
            wr = w / len(subset) * 100
            avg = p / len(subset)
            filters.append({
                "filter": f"SRR <= {srr_max}",
                "trades": len(subset),
                "winners": w,
                "wr": wr,
                "pnl": p,
                "avg_pnl": avg
            })

    # Rank score filters
    for rank_min in [0.5, 1.0, 1.5, 2.0, 2.5]:
        mask = trades_df['rank_score'] >= rank_min
        subset = trades_df[mask]
        if len(subset) >= 10:
            w = subset['is_winner'].sum()
            p = subset['total_pnl'].sum()
            wr = w / len(subset) * 100
            avg = p / len(subset)
            filters.append({
                "filter": f"Rank >= {rank_min}",
                "trades": len(subset),
                "winners": w,
                "wr": wr,
                "pnl": p,
                "avg_pnl": avg
            })

    # Sort by average PnL per trade
    filters.sort(key=lambda x: x['avg_pnl'], reverse=True)

    print(f"\n{'Filter':<20} {'Trades':>8} {'Win':>6} {'WR%':>8} {'PnL':>12} {'Avg/trade':>12}")
    print("-" * 70)
    for f in filters[:20]:  # Top 20
        if f['pnl'] > 0:  # Only profitable
            print(f"{f['filter']:<20} {f['trades']:>8} {f['winners']:>6} {f['wr']:>7.1f}% {f['pnl']:>11,.0f} {f['avg_pnl']:>11,.1f}")

    return filters


def test_combinations(trades_df, segment_name):
    """Test combination of best filters"""
    print(f"\n{'='*60}")
    print(f"COMBINATION FILTER ANALYSIS: {segment_name}")
    print(f"{'='*60}")

    combos = []

    # Test combinations of filters
    adx_values = [15, 20, 25, 30]
    vol_values = [50000, 100000, 150000, 200000]
    srr_values = [1.0, 1.5, 2.0, 2.5]

    if "long" in segment_name.lower():
        rsi_values = [50, 55, 60, 65]
        rsi_direction = "max"
    else:
        rsi_values = [45, 50, 55, 60]
        rsi_direction = "min"

    for adx in adx_values:
        for vol in vol_values:
            for srr in srr_values:
                for rsi in rsi_values:
                    if rsi_direction == "max":
                        mask = (
                            (trades_df['adx'] >= adx) &
                            (trades_df['volume'] >= vol) &
                            (trades_df['structural_rr'] <= srr) &
                            (trades_df['rsi'] <= rsi)
                        )
                    else:
                        mask = (
                            (trades_df['adx'] >= adx) &
                            (trades_df['volume'] >= vol) &
                            (trades_df['structural_rr'] <= srr) &
                            (trades_df['rsi'] >= rsi)
                        )

                    subset = trades_df[mask]
                    if len(subset) >= 5:  # Minimum 5 trades
                        w = subset['is_winner'].sum()
                        p = subset['total_pnl'].sum()
                        wr = w / len(subset) * 100 if len(subset) > 0 else 0
                        avg = p / len(subset) if len(subset) > 0 else 0

                        if p > 0:  # Only profitable combos
                            combos.append({
                                "adx": adx,
                                "vol": vol,
                                "srr": srr,
                                "rsi": rsi,
                                "rsi_dir": rsi_direction,
                                "trades": len(subset),
                                "winners": w,
                                "wr": wr,
                                "pnl": p,
                                "avg_pnl": avg
                            })

    # Sort by avg PnL per trade
    combos.sort(key=lambda x: x['avg_pnl'], reverse=True)

    print(f"\n{'ADX':>5} {'Vol':>8} {'SRR':>5} {'RSI':>5} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'Avg':>10}")
    print("-" * 70)
    for c in combos[:15]:  # Top 15
        rsi_str = f"<={c['rsi']}" if c['rsi_dir'] == 'max' else f">={c['rsi']}"
        print(f">={c['adx']:>3} >={c['vol']/1000:>5.0f}k <={c['srr']:>4.1f} {rsi_str:>6} {c['trades']:>7} {c['wr']:>6.1f}% {c['pnl']:>9,.0f} {c['avg_pnl']:>9,.1f}")

    return combos


def main():
    print("="*60)
    print("FHM DEEPER FILTER ANALYSIS")
    print("Finding additional filters to reduce trades while improving profit")
    print("="*60)

    # Extract all trades
    trades = extract_fhm_trades()
    df = pd.DataFrame(trades)

    print(f"\nTotal FHM trades extracted: {len(df)}")
    print(f"Total PnL: {df['total_pnl'].sum():,.0f} Rs")

    # Split by direction
    longs = df[df['bias'] == 'long']
    shorts = df[df['bias'] == 'short']

    print(f"\nLONGS: {len(longs)} trades, PnL: {longs['total_pnl'].sum():,.0f} Rs")
    print(f"SHORTS: {len(shorts)} trades, PnL: {shorts['total_pnl'].sum():,.0f} Rs")

    # Analyze SHORTS (already profitable - can we improve?)
    if len(shorts) >= 10:
        analyze_filters(shorts, "SHORTS")
        test_combinations(shorts, "SHORTS")

    # Analyze LONG MID_CAP (profitable segment)
    long_midcap = longs[longs['cap_segment'] == 'mid_cap']
    print(f"\nLONG MID_CAP: {len(long_midcap)} trades, PnL: {long_midcap['total_pnl'].sum():,.0f} Rs")

    if len(long_midcap) >= 10:
        analyze_filters(long_midcap, "LONG MID_CAP")
        test_combinations(long_midcap, "LONG MID_CAP")

    # Also check other segments
    print("\n" + "="*60)
    print("ALL CAP SEGMENT BREAKDOWN FOR LONGS:")
    print("="*60)
    for cap in longs['cap_segment'].unique():
        subset = longs[longs['cap_segment'] == cap]
        w = subset['is_winner'].sum()
        p = subset['total_pnl'].sum()
        wr = w / len(subset) * 100 if len(subset) > 0 else 0
        avg = p / len(subset) if len(subset) > 0 else 0
        print(f"{cap:<15}: {len(subset):>4} trades, {w:>3}W, WR={wr:>5.1f}%, PnL={p:>10,.0f} Rs, Avg={avg:>8,.1f}/trade")

    # Check regime breakdown for SHORTS
    print("\n" + "="*60)
    print("REGIME BREAKDOWN FOR SHORTS:")
    print("="*60)
    for regime in shorts['regime'].unique():
        subset = shorts[shorts['regime'] == regime]
        w = subset['is_winner'].sum()
        p = subset['total_pnl'].sum()
        wr = w / len(subset) * 100 if len(subset) > 0 else 0
        avg = p / len(subset) if len(subset) > 0 else 0
        print(f"{regime:<15}: {len(subset):>4} trades, {w:>3}W, WR={wr:>5.1f}%, PnL={p:>10,.0f} Rs, Avg={avg:>8,.1f}/trade")


if __name__ == "__main__":
    main()
