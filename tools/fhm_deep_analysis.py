"""
FHM Deep Analysis - Comprehensive extraction and analysis of ALL FHM trades
Analyzes EVERY parameter/indicator to find patterns that separate winners from losers
"""
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz

BACKTEST_DIR = Path(r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251209-060217_extracted")
OHLCV_CACHE = Path(r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\cache\ohlcv_archive")
IST = pytz.timezone('Asia/Kolkata')

def extract_all_fhm_trades():
    """Extract ALL FHM trades with complete data from all days"""
    trades = []

    for day_dir in sorted(BACKTEST_DIR.iterdir()):
        if not day_dir.is_dir():
            continue

        events_file = day_dir / "events.jsonl"
        analytics_file = day_dir / "analytics.jsonl"
        if not events_file.exists():
            continue

        # Load all events for this day
        decisions = {}
        triggers = {}
        exits = {}

        # Load decisions and triggers from events.jsonl
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

        # Load exits from analytics.jsonl (has full setup_type)
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

        # Combine DECISION + TRIGGER + EXIT into complete trades
        for trade_id, decision in decisions.items():
            if trade_id not in triggers:
                continue  # No trigger = no actual trade

            trigger = triggers[trade_id]
            exit_events = exits.get(trade_id, [])

            if not exit_events:
                continue  # No exit = incomplete trade

            # Get the final exit (last one with is_final_exit=True or just last)
            final_exit = None
            for e in exit_events:
                if e.get("is_final_exit"):
                    final_exit = e
                    break
            if not final_exit:
                final_exit = exit_events[-1]

            # Extract ALL data
            trade = extract_complete_trade_data(decision, trigger, exit_events, final_exit, day_dir.name)
            trades.append(trade)

    return trades

def extract_complete_trade_data(decision, trigger, exit_events, final_exit, date):
    """Extract EVERY parameter from a trade"""

    dec = decision.get("decision", {})
    # NOTE: plan is at top level of event, NOT under decision!
    plan = decision.get("plan", {})
    bar5 = decision.get("bar5", {})
    features = decision.get("features", {})
    timectx = decision.get("timectx", {})

    # Basic trade info
    trade = {
        "date": date,
        "trade_id": decision.get("trade_id"),
        "symbol": decision.get("symbol"),
        "setup_type": dec.get("setup_type"),
        "bias": plan.get("bias"),
        "regime": dec.get("regime"),
        "decision_ts": dec.get("reasons", ""),
        "ts": decision.get("ts"),
    }

    # Entry data
    trade["entry_ref_price"] = plan.get("entry_ref_price")
    trade["actual_entry_price"] = trigger.get("trigger", {}).get("actual_price")
    trade["entry_zone_low"] = plan.get("entry", {}).get("zone", [None, None])[0]
    trade["entry_zone_high"] = plan.get("entry", {}).get("zone", [None, None])[1]
    trade["entry_mode"] = plan.get("entry", {}).get("mode")
    trade["entry_trigger"] = plan.get("entry", {}).get("trigger")

    # Trigger timing
    trigger_ts = trigger.get("ts")
    decision_ts = decision.get("ts")
    if trigger_ts and decision_ts:
        trade["trigger_ts"] = trigger_ts
        trade["decision_to_trigger_mins"] = _time_diff_mins(decision_ts, trigger_ts)

    # Stop loss data
    trade["hard_sl"] = plan.get("stop", {}).get("hard")
    trade["risk_per_share"] = plan.get("stop", {}).get("risk_per_share")

    # Targets
    targets = plan.get("targets", [])
    for t in targets:
        name = t.get("name", "").lower()
        trade[f"{name}_level"] = t.get("level")
        trade[f"{name}_rr"] = t.get("rr")
        trade[f"{name}_qty_pct"] = t.get("qty_pct")

    # Quality metrics
    quality = plan.get("quality", {})
    trade["structural_rr"] = quality.get("structural_rr")
    trade["quality_status"] = quality.get("status")
    trade["t1_feasible"] = quality.get("t1_feasible")
    trade["t2_feasible"] = quality.get("t2_feasible")
    quality_metrics = quality.get("metrics", {})
    trade["quality_adx"] = quality_metrics.get("adx")
    trade["quality_adx_score"] = quality_metrics.get("adx_score")
    trade["quality_ema_aligned"] = quality_metrics.get("ema_aligned")
    trade["quality_ema20"] = quality_metrics.get("ema20")
    trade["quality_ema50"] = quality_metrics.get("ema50")

    # Ranking
    ranking = plan.get("ranking", {})
    trade["rank_score"] = ranking.get("score")
    trade["rank_base_score"] = ranking.get("base_score")
    components = ranking.get("components", {})
    for k, v in components.items():
        trade[f"rank_{k}"] = v
    multipliers = ranking.get("multipliers", {})
    trade["regime_multiplier"] = multipliers.get("regime")
    adjustments = ranking.get("universal_adjustments", {})
    for k, v in adjustments.items():
        trade[f"adj_{k}"] = v

    # Sizing
    sizing = plan.get("sizing", {})
    trade["qty"] = sizing.get("qty")
    trade["notional"] = sizing.get("notional")
    trade["risk_rupees"] = sizing.get("risk_rupees")
    trade["size_mult"] = sizing.get("size_mult")
    trade["cap_segment"] = sizing.get("cap_segment")

    # Indicators at decision time
    indicators = plan.get("indicators", {})
    trade["atr"] = indicators.get("atr")
    trade["adx"] = indicators.get("adx")
    trade["rsi"] = indicators.get("rsi")
    trade["vwap"] = indicators.get("vwap")

    # Levels
    levels = plan.get("levels", {})
    trade["PDH"] = levels.get("PDH")
    trade["PDL"] = levels.get("PDL")
    trade["PDC"] = levels.get("PDC")
    trade["ORH"] = levels.get("ORH")
    trade["ORL"] = levels.get("ORL")

    # Bar5 data (OHLCV at decision time)
    trade["bar5_open"] = bar5.get("open")
    trade["bar5_high"] = bar5.get("high")
    trade["bar5_low"] = bar5.get("low")
    trade["bar5_close"] = bar5.get("close")
    trade["bar5_volume"] = bar5.get("volume")
    trade["bar5_vwap"] = bar5.get("vwap")
    trade["bar5_adx"] = bar5.get("adx")
    trade["bar5_bb_width"] = bar5.get("bb_width_proxy")

    # FHM specific context
    fhm_context = plan.get("fhm_context", {})
    trade["fhm_rvol"] = fhm_context.get("rvol")
    trade["fhm_price_move_pct"] = fhm_context.get("price_move_pct")
    trade["fhm_eligible"] = fhm_context.get("eligible")

    # Features
    trade["feature_rank_score"] = features.get("rank_score")
    trade["feature_fhm_rvol"] = features.get("fhm_rvol")
    trade["feature_fhm_price_move_pct"] = features.get("fhm_price_move_pct")

    # Time context
    trade["minute_of_day"] = timectx.get("minute_of_day")
    trade["day_of_week"] = timectx.get("day_of_week")

    # Regime diagnostics
    regime_diag = dec.get("regime_diagnostics") or {}
    daily_diag = regime_diag.get("daily") or {}
    trade["daily_regime"] = daily_diag.get("regime")
    trade["daily_regime_confidence"] = daily_diag.get("confidence")
    trade["daily_trend_strength"] = daily_diag.get("trend_strength")
    daily_metrics = daily_diag.get("metrics") or {}
    trade["daily_price"] = daily_metrics.get("price")
    trade["daily_ema200"] = daily_metrics.get("ema200")
    trade["daily_adx"] = daily_metrics.get("adx")
    trade["daily_bb_width"] = daily_metrics.get("bb_width")
    trade["daily_bb_threshold"] = daily_metrics.get("bb_threshold")
    trade["daily_price_distance_pct"] = daily_metrics.get("price_distance_pct")

    hourly_diag = regime_diag.get("hourly") or {}
    trade["hourly_session_bias"] = hourly_diag.get("session_bias")
    trade["hourly_momentum"] = hourly_diag.get("momentum")

    fivem_diag = regime_diag.get("5m") or {}
    trade["5m_regime"] = fivem_diag.get("regime")

    unified_diag = regime_diag.get("unified") or {}
    trade["unified_regime"] = unified_diag.get("regime")
    trade["unified_confidence"] = unified_diag.get("confidence")

    # Pipeline reasons
    trade["pipeline_reasons"] = plan.get("pipeline_reasons", [])
    trade["cautions"] = plan.get("cautions", [])

    # Exit data (from analytics.jsonl format)
    trade["exit_ts"] = final_exit.get("timestamp")
    trade["exit_price"] = final_exit.get("exit_price")
    trade["exit_reason"] = final_exit.get("reason")
    trade["pnl"] = final_exit.get("pnl")
    trade["total_trade_pnl"] = final_exit.get("total_trade_pnl")
    trade["slippage_bps"] = final_exit.get("slippage_bps")
    trade["exit_sequence"] = final_exit.get("exit_sequence")
    trade["total_exits"] = final_exit.get("total_exits")
    trade["exit_actual_entry_price"] = final_exit.get("actual_entry_price")
    trade["exit_entry_reference"] = final_exit.get("entry_reference")

    # Exit analytics
    exit_analytics = final_exit.get("analytics", {})
    trade["time_decay_factor"] = exit_analytics.get("time_decay_factor")
    trade["execution_probability"] = exit_analytics.get("execution_probability")

    # Derived metrics
    if trade["actual_entry_price"] and trade["hard_sl"]:
        trade["actual_sl_distance"] = abs(trade["actual_entry_price"] - trade["hard_sl"])
        trade["actual_sl_pct"] = trade["actual_sl_distance"] / trade["actual_entry_price"] * 100

    if trade["actual_entry_price"] and trade.get("t1_level"):
        trade["actual_t1_distance"] = abs(trade["t1_level"] - trade["actual_entry_price"])

    # Calculate hold time
    if trade.get("trigger_ts") and trade.get("exit_ts"):
        trade["hold_time_mins"] = _time_diff_mins(trade["trigger_ts"], trade["exit_ts"])

    # Win/Loss classification
    if trade["pnl"] is not None:
        trade["is_winner"] = trade["pnl"] > 0

    # Exit analysis
    trade["num_exits"] = len(exit_events)
    trade["hit_t1"] = any(e.get("reason") == "t1_hit" for e in exit_events)
    trade["hit_t2"] = any(e.get("reason") == "t2_hit" for e in exit_events)
    trade["hit_hard_sl"] = trade["exit_reason"] == "hard_sl"
    trade["hit_trailing_sl"] = trade["exit_reason"] == "trailing_sl"
    trade["eod_exit"] = trade["exit_reason"] == "eod_square_off"

    return trade

def _time_diff_mins(ts1, ts2):
    """Calculate time difference in minutes between two timestamp strings"""
    try:
        t1 = datetime.strptime(ts1, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(ts2, "%Y-%m-%d %H:%M:%S")
        return (t2 - t1).total_seconds() / 60
    except:
        return None

def load_1m_ohlcv(symbol, date):
    """Load 1-minute OHLCV data for a symbol"""
    symbol_clean = symbol.replace("NSE:", "")
    cache_path = OHLCV_CACHE / f"{symbol_clean}.NS" / f"{symbol_clean}.NS_1minutes.feather"

    if not cache_path.exists():
        return None

    df = pd.read_feather(cache_path)

    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Filter for the specific date
    trade_date = pd.to_datetime(date)
    df_day = df[df['date'].dt.date == trade_date.date()].copy()

    return df_day if len(df_day) > 0 else None

def analyze_price_action(trade, df_1m):
    """Analyze 1-minute price action for a trade"""
    if df_1m is None or len(df_1m) == 0:
        return {}

    trigger_ts = trade.get("trigger_ts")
    if not trigger_ts:
        return {}

    try:
        trigger_time = pd.to_datetime(trigger_ts)
        if trigger_time.tzinfo is None:
            trigger_time = IST.localize(trigger_time)
    except:
        return {}

    entry_price = trade.get("actual_entry_price")
    sl_price = trade.get("hard_sl")
    t1_level = trade.get("t1_level")
    bias = trade.get("bias")

    if not all([entry_price, sl_price, t1_level]):
        return {}

    # Get bars after entry
    if df_1m['date'].dt.tz is None:
        df_1m['date'] = df_1m['date'].dt.tz_localize('Asia/Kolkata')

    bars_after = df_1m[df_1m['date'] >= trigger_time].copy()

    if len(bars_after) == 0:
        return {}

    analysis = {}

    # Calculate MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion)
    if bias == "long":
        analysis["mfe"] = (bars_after['high'].max() - entry_price) / entry_price * 100
        analysis["mae"] = (entry_price - bars_after['low'].min()) / entry_price * 100
        analysis["mfe_price"] = bars_after['high'].max()
        analysis["mae_price"] = bars_after['low'].min()
    else:
        analysis["mfe"] = (entry_price - bars_after['low'].min()) / entry_price * 100
        analysis["mae"] = (bars_after['high'].max() - entry_price) / entry_price * 100
        analysis["mfe_price"] = bars_after['low'].min()
        analysis["mae_price"] = bars_after['high'].max()

    # Calculate time to hit SL or T1
    bars_to_sl = None
    bars_to_t1 = None
    sl_first = None
    t1_first = None

    for i, (idx, bar) in enumerate(bars_after.iterrows()):
        if bias == "long":
            if sl_first is None and bar['low'] <= sl_price:
                sl_first = i + 1
                bars_to_sl = i + 1
            if t1_first is None and bar['high'] >= t1_level:
                t1_first = i + 1
                bars_to_t1 = i + 1
        else:
            if sl_first is None and bar['high'] >= sl_price:
                sl_first = i + 1
                bars_to_sl = i + 1
            if t1_first is None and bar['low'] <= t1_level:
                t1_first = i + 1
                bars_to_t1 = i + 1

        if sl_first and t1_first:
            break

    analysis["bars_to_sl"] = bars_to_sl
    analysis["bars_to_t1"] = bars_to_t1
    analysis["sl_hit_first"] = sl_first is not None and (t1_first is None or sl_first < t1_first)
    analysis["t1_hit_first"] = t1_first is not None and (sl_first is None or t1_first < sl_first)

    # Price action in first 5, 10, 15, 30 mins
    for mins in [5, 10, 15, 30, 60]:
        bars_window = bars_after.iloc[:mins] if len(bars_after) >= mins else bars_after
        if len(bars_window) > 0:
            if bias == "long":
                analysis[f"high_{mins}m"] = bars_window['high'].max()
                analysis[f"low_{mins}m"] = bars_window['low'].min()
                analysis[f"close_{mins}m"] = bars_window.iloc[-1]['close']
                analysis[f"pct_move_{mins}m"] = (bars_window.iloc[-1]['close'] - entry_price) / entry_price * 100
            else:
                analysis[f"high_{mins}m"] = bars_window['high'].max()
                analysis[f"low_{mins}m"] = bars_window['low'].min()
                analysis[f"close_{mins}m"] = bars_window.iloc[-1]['close']
                analysis[f"pct_move_{mins}m"] = (entry_price - bars_window.iloc[-1]['close']) / entry_price * 100

    # Volume analysis
    if 'volume' in bars_after.columns:
        first_5_vol = bars_after.iloc[:5]['volume'].mean() if len(bars_after) >= 5 else bars_after['volume'].mean()
        next_10_vol = bars_after.iloc[5:15]['volume'].mean() if len(bars_after) >= 15 else 0
        analysis["vol_first_5m"] = first_5_vol
        analysis["vol_next_10m"] = next_10_vol
        analysis["vol_ratio_5v10"] = first_5_vol / next_10_vol if next_10_vol > 0 else 0

    return analysis

def main():
    print("=" * 80)
    print("FHM DEEP ANALYSIS - EXTRACTING ALL TRADES")
    print("=" * 80)

    # Extract all FHM trades
    trades = extract_all_fhm_trades()
    print(f"\nTotal FHM trades extracted: {len(trades)}")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(trades)

    # Separate winners and losers
    winners = df[df['is_winner'] == True]
    losers = df[df['is_winner'] == False]

    print(f"Winners: {len(winners)} ({len(winners)/len(df)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(df)*100:.1f}%)")
    print(f"Total PnL: {df['pnl'].sum():.2f}")

    # Exit reason breakdown
    print("\n" + "=" * 80)
    print("EXIT REASON BREAKDOWN")
    print("=" * 80)
    print(df['exit_reason'].value_counts())

    # Analyze every numeric column
    print("\n" + "=" * 80)
    print("WINNER vs LOSER COMPARISON - ALL NUMERIC METRICS")
    print("=" * 80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    comparison_data = []
    for col in numeric_cols:
        if col in ['is_winner', 'pnl', 'total_trade_pnl']:
            continue

        winner_vals = winners[col].dropna()
        loser_vals = losers[col].dropna()

        if len(winner_vals) == 0 or len(loser_vals) == 0:
            continue

        winner_mean = winner_vals.mean()
        loser_mean = loser_vals.mean()

        if winner_mean == 0 and loser_mean == 0:
            continue

        # Calculate difference
        if loser_mean != 0:
            pct_diff = (winner_mean - loser_mean) / abs(loser_mean) * 100
        else:
            pct_diff = float('inf') if winner_mean > 0 else float('-inf')

        comparison_data.append({
            "metric": col,
            "winner_mean": winner_mean,
            "winner_std": winner_vals.std(),
            "loser_mean": loser_mean,
            "loser_std": loser_vals.std(),
            "pct_diff": pct_diff
        })

    # Sort by absolute difference
    comparison_data = sorted(comparison_data, key=lambda x: abs(x['pct_diff']), reverse=True)

    print("\n{:<40} {:>12} {:>12} {:>12}".format("METRIC", "WINNER AVG", "LOSER AVG", "% DIFF"))
    print("-" * 80)
    for item in comparison_data[:50]:  # Top 50 differentiating metrics
        print("{:<40} {:>12.4f} {:>12.4f} {:>+12.1f}%".format(
            item['metric'][:40], item['winner_mean'], item['loser_mean'], item['pct_diff']
        ))

    # Long vs Short breakdown
    print("\n" + "=" * 80)
    print("LONG vs SHORT BREAKDOWN")
    print("=" * 80)

    for bias in ['long', 'short']:
        subset = df[df['bias'] == bias]
        if len(subset) == 0:
            print(f"\n{bias.upper()}: No trades")
            continue
        w = subset[subset['is_winner'] == True]
        l = subset[subset['is_winner'] == False]
        print(f"\n{bias.upper()}:")
        print(f"  Total: {len(subset)}, Winners: {len(w)} ({len(w)/len(subset)*100:.1f}%), PnL: {subset['pnl'].sum():.2f}")
        print(f"  Exit reasons: {subset['exit_reason'].value_counts().to_dict()}")

    # Regime breakdown
    print("\n" + "=" * 80)
    print("REGIME BREAKDOWN")
    print("=" * 80)

    for regime in df['regime'].dropna().unique():
        subset = df[df['regime'] == regime]
        if len(subset) == 0:
            continue
        w = subset[subset['is_winner'] == True]
        l = subset[subset['is_winner'] == False]
        print(f"\n{regime}:")
        print(f"  Total: {len(subset)}, Winners: {len(w)} ({len(w)/len(subset)*100:.1f}%), PnL: {subset['pnl'].sum():.2f}")
        if len(w) > 0 and len(l) > 0:
            print(f"  Avg Winner PnL: {w['pnl'].mean():.2f}, Avg Loser PnL: {l['pnl'].mean():.2f}")

    # Time of day analysis
    print("\n" + "=" * 80)
    print("TIME OF DAY ANALYSIS (minute_of_day)")
    print("=" * 80)

    df['hour'] = df['minute_of_day'] // 60
    for hour in sorted(df['hour'].dropna().unique()):
        subset = df[df['hour'] == hour]
        w = subset[subset['is_winner'] == True]
        pnl = subset['pnl'].sum()
        wr = len(w)/len(subset)*100 if len(subset) > 0 else 0
        print(f"Hour {int(hour)}: {len(subset):3d} trades, WR: {wr:5.1f}%, PnL: {pnl:8.2f}")

    # Cap segment breakdown
    print("\n" + "=" * 80)
    print("CAP SEGMENT BREAKDOWN")
    print("=" * 80)

    for seg in df['cap_segment'].dropna().unique():
        subset = df[df['cap_segment'] == seg]
        if len(subset) == 0:
            continue
        w = subset[subset['is_winner'] == True]
        print(f"\n{seg}:")
        print(f"  Total: {len(subset)}, Winners: {len(w)} ({len(w)/len(subset)*100:.1f}%), PnL: {subset['pnl'].sum():.2f}")

    # Now load 1m OHLCV and analyze price action
    print("\n" + "=" * 80)
    print("LOADING 1M OHLCV DATA FOR PRICE ACTION ANALYSIS")
    print("=" * 80)

    price_action_results = []
    for i, trade in enumerate(trades):
        symbol = trade.get("symbol")
        date = trade.get("date")

        df_1m = load_1m_ohlcv(symbol, date)
        if df_1m is not None:
            pa = analyze_price_action(trade, df_1m)
            pa["trade_id"] = trade["trade_id"]
            pa["is_winner"] = trade.get("is_winner")
            pa["pnl"] = trade.get("pnl")
            pa["exit_reason"] = trade.get("exit_reason")
            price_action_results.append(pa)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(trades)} trades")

    print(f"\nPrice action analysis complete for {len(price_action_results)} trades")

    # Price action comparison
    if price_action_results:
        pa_df = pd.DataFrame(price_action_results)
        pa_winners = pa_df[pa_df['is_winner'] == True]
        pa_losers = pa_df[pa_df['is_winner'] == False]

        print("\n" + "=" * 80)
        print("PRICE ACTION: WINNER vs LOSER")
        print("=" * 80)

        for col in ['mfe', 'mae', 'bars_to_sl', 'bars_to_t1', 'pct_move_5m', 'pct_move_15m', 'pct_move_30m', 'vol_ratio_5v10']:
            if col in pa_df.columns:
                w_mean = pa_winners[col].dropna().mean()
                l_mean = pa_losers[col].dropna().mean()
                if not (np.isnan(w_mean) or np.isnan(l_mean)):
                    diff = ((w_mean - l_mean) / abs(l_mean) * 100) if l_mean != 0 else 0
                    print(f"{col:20s}: Winners={w_mean:8.3f}, Losers={l_mean:8.3f}, Diff={diff:+.1f}%")

    # Find the BEST filters
    print("\n" + "=" * 80)
    print("POTENTIAL FILTERS TO SEPARATE WINNERS FROM LOSERS")
    print("=" * 80)

    # Test various thresholds
    filters_to_test = [
        ("bar5_volume", ">", [100000, 150000, 200000, 300000]),
        ("structural_rr", ">", [0.5, 0.7, 1.0, 1.2, 1.5]),
        ("structural_rr", "<", [1.5, 1.8, 2.0, 2.5]),
        ("rank_score", ">", [1.5, 2.0, 2.5, 3.0]),
        ("adx", ">", [15, 20, 25, 30]),
        ("adx", "<", [25, 30, 35, 40]),
        ("rsi", "<", [55, 60, 65, 70]),
        ("rsi", ">", [40, 45, 50, 55]),
        ("minute_of_day", "<", [600, 620, 640, 660]),  # Entry time
        ("actual_sl_pct", "<", [0.8, 1.0, 1.2, 1.5]),
        ("rank_volume", ">", [1.0, 1.5, 2.0]),
        ("daily_trend_strength", ">", [0.3, 0.5, 0.7]),
        ("daily_regime_confidence", ">", [0.5, 0.6, 0.7]),
    ]

    filter_results = []

    for col, op, thresholds in filters_to_test:
        if col not in df.columns:
            continue

        for thresh in thresholds:
            if op == ">":
                filtered = df[df[col] > thresh]
                remaining = df[df[col] <= thresh]
                filter_desc = f"{col} > {thresh}"
            else:
                filtered = df[df[col] < thresh]
                remaining = df[df[col] >= thresh]
                filter_desc = f"{col} < {thresh}"

            if len(filtered) < 10:
                continue

            filtered_winners = filtered[filtered['is_winner'] == True]
            filtered_wr = len(filtered_winners) / len(filtered) * 100
            filtered_pnl = filtered['pnl'].sum()

            remaining_winners = remaining[remaining['is_winner'] == True]
            remaining_wr = len(remaining_winners) / len(remaining) * 100 if len(remaining) > 0 else 0
            remaining_pnl = remaining['pnl'].sum()

            filter_results.append({
                "filter": filter_desc,
                "passed_trades": len(filtered),
                "passed_wr": filtered_wr,
                "passed_pnl": filtered_pnl,
                "blocked_trades": len(remaining),
                "blocked_wr": remaining_wr,
                "blocked_pnl": remaining_pnl,
                "net_benefit": filtered_pnl - (filtered_pnl + remaining_pnl)  # Improvement from blocking bad trades
            })

    # Sort by potential benefit
    filter_results = sorted(filter_results, key=lambda x: x['passed_wr'] - x['blocked_wr'], reverse=True)

    print("\n{:<30} {:>8} {:>8} {:>10} {:>8} {:>8} {:>10}".format(
        "FILTER", "PASSED", "WR%", "PNL", "BLOCKED", "WR%", "PNL"))
    print("-" * 90)
    for f in filter_results[:30]:
        print("{:<30} {:>8} {:>7.1f}% {:>10.0f} {:>8} {:>7.1f}% {:>10.0f}".format(
            f['filter'][:30], f['passed_trades'], f['passed_wr'], f['passed_pnl'],
            f['blocked_trades'], f['blocked_wr'], f['blocked_pnl']
        ))

    # Save full data for further analysis
    output_path = Path("fhm_deep_analysis_output.json")
    output = {
        "summary": {
            "total_trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(df) * 100,
            "total_pnl": df['pnl'].sum(),
            "avg_winner": winners['pnl'].mean(),
            "avg_loser": losers['pnl'].mean()
        },
        "comparison": comparison_data,
        "filter_results": filter_results,
        "trades": trades
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nFull data saved to: {output_path}")

    return df, comparison_data, filter_results

if __name__ == "__main__":
    df, comparison, filters = main()
