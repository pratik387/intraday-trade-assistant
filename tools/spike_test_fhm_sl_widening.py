"""
Spike Test: FHM Stop Loss Widening

For each losing FHM trade:
1. Get entry, stop, exit, MAE, MFE
2. Simulate what happens with widened stops
3. Check if losers would become winners
"""
import json
from pathlib import Path
from collections import defaultdict

backtest_dir = Path(r"c:\Users\pratikhegde\OneDrive - Nagarro\Desktop\Pratik\intraday-trade-assistant\backtest_20251208-040138_extracted")
print(f"Using backtest: {backtest_dir}")

# Collect all trades by trade_id
all_trades = defaultdict(dict)

for session_dir in backtest_dir.iterdir():
    if not session_dir.is_dir():
        continue

    events_file = session_dir / "events.jsonl"
    if not events_file.exists():
        continue

    with open(events_file, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
            except:
                continue

            trade_id = event.get("trade_id")
            event_type = event.get("type")

            if not trade_id:
                continue

            if event_type == "DECISION":
                decision = event.get("decision", {})
                setup_type = decision.get("setup_type", "")
                if "first_hour_momentum" in setup_type:
                    all_trades[trade_id]["decision"] = event
                    all_trades[trade_id]["setup_type"] = setup_type
                    all_trades[trade_id]["symbol"] = event.get("symbol")

            elif event_type == "TRIGGER":
                if trade_id in all_trades:
                    all_trades[trade_id]["trigger"] = event

            elif event_type == "EXIT":
                if trade_id in all_trades:
                    if "exits" not in all_trades[trade_id]:
                        all_trades[trade_id]["exits"] = []
                    all_trades[trade_id]["exits"].append(event)

# Filter to only FHM trades with exits
fhm_trades = []
for trade_id, trade_data in all_trades.items():
    if "exits" in trade_data and "trigger" in trade_data:
        fhm_trades.append(trade_data)

print(f"Total FHM trades with exits: {len(fhm_trades)}")

# Analyze each trade
long_trades = []
short_trades = []

for trade in fhm_trades:
    setup_type = trade["setup_type"]
    symbol = trade["symbol"]

    trigger = trade["trigger"].get("trigger", {})
    entry_price = trigger.get("actual_price")

    # Get decision details for stop loss
    decision = trade["decision"].get("decision", {})
    stop_loss = decision.get("stop_loss") or decision.get("hard_stop")

    # If not in decision, check trade_plan
    if not stop_loss:
        trade_plan = decision.get("trade_plan", {})
        stop_loss = trade_plan.get("hard_sl") or trade_plan.get("stop_loss")

    # Calculate total PnL and get MAE/MFE from exits
    total_pnl = 0
    all_mae = []
    all_mfe = []
    exit_reasons = []
    final_exit_price = None

    for exit_event in trade.get("exits", []):
        exit_data = exit_event.get("exit", {})
        pnl = exit_data.get("pnl", 0)
        total_pnl += pnl
        final_exit_price = exit_data.get("price")

        diag = exit_data.get("diagnostics", {})
        if diag.get("mae") is not None:
            all_mae.append(diag["mae"])
        if diag.get("mfe") is not None:
            all_mfe.append(diag["mfe"])
        exit_reasons.append(exit_data.get("reason", ""))

    trade_info = {
        "trade_id": trade.get("decision", {}).get("trade_id"),
        "symbol": symbol,
        "setup_type": setup_type,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "exit_price": final_exit_price,
        "pnl": total_pnl,
        "mae": min(all_mae) if all_mae else None,  # Worst adverse excursion
        "mfe": max(all_mfe) if all_mfe else None,  # Best favorable excursion
        "exit_reasons": exit_reasons,
        "side": "long" if "_long" in setup_type else "short",
    }

    if "_long" in setup_type:
        long_trades.append(trade_info)
    else:
        short_trades.append(trade_info)

long_winners = [t for t in long_trades if t["pnl"] >= 0]
long_losers = [t for t in long_trades if t["pnl"] < 0]
short_winners = [t for t in short_trades if t["pnl"] >= 0]
short_losers = [t for t in short_trades if t["pnl"] < 0]

print(f"\nFHM_LONG: {len(long_winners)} winners, {len(long_losers)} losers")
print(f"FHM_SHORT: {len(short_winners)} winners, {len(short_losers)} losers")

# Exit reason breakdown
print("\n" + "="*60)
print("EXIT REASON ANALYSIS")
print("="*60)

reason_stats = defaultdict(lambda: {"count": 0, "pnl": 0})
for t in long_trades + short_trades:
    for reason in t["exit_reasons"]:
        reason_stats[reason]["count"] += 1
        reason_stats[reason]["pnl"] += t["pnl"] / len(t["exit_reasons"])  # Divide by exits

for reason, stats in sorted(reason_stats.items(), key=lambda x: x[1]["count"], reverse=True):
    print(f"  {reason}: {stats['count']} exits, PnL contribution: {stats['pnl']:.2f}")

# MAE/MFE Analysis
print("\n" + "="*60)
print("MAE/MFE ANALYSIS (R-multiples from entry)")
print("="*60)

def analyze_mae_mfe(trades, label):
    mae_values = [t["mae"] for t in trades if t["mae"] is not None]
    mfe_values = [t["mfe"] for t in trades if t["mfe"] is not None]

    if mae_values:
        print(f"\n{label} MAE (Max Adverse Excursion):")
        print(f"  Avg MAE: {sum(mae_values)/len(mae_values):.2f}R")
        print(f"  Min MAE: {min(mae_values):.2f}R (worst drawdown)")
        print(f"  Max MAE: {max(mae_values):.2f}R")
    else:
        print(f"\n{label}: No MAE data available")

    if mfe_values:
        print(f"\n{label} MFE (Max Favorable Excursion):")
        print(f"  Avg MFE: {sum(mfe_values)/len(mfe_values):.2f}R")
        print(f"  Max MFE: {max(mfe_values):.2f}R (best run)")
        print(f"  Min MFE: {min(mfe_values):.2f}R")

    return mae_values, mfe_values

print("\n--- LONG LOSERS ---")
long_loser_mae, long_loser_mfe = analyze_mae_mfe(long_losers, "LONG LOSERS")

print("\n--- LONG WINNERS ---")
long_winner_mae, long_winner_mfe = analyze_mae_mfe(long_winners, "LONG WINNERS")

print("\n--- SHORT LOSERS ---")
short_loser_mae, short_loser_mfe = analyze_mae_mfe(short_losers, "SHORT LOSERS")

# SPIKE TEST: SL Widening
print("\n" + "="*60)
print("SPIKE TEST: WOULD WIDENING STOPS HELP?")
print("="*60)

def spike_test_widening(losers, label):
    """
    For each loser:
    - If MAE < 1.0R (didn't breach 1R stop), check if it hit T1 (MFE >= 1.0R)
    - If yes, trade would have been a winner with 1R stop
    """
    if not losers:
        print(f"\n{label}: No losers to analyze")
        return

    trades_with_data = [t for t in losers if t["mae"] is not None and t["mfe"] is not None]

    if not trades_with_data:
        print(f"\n{label}: No MAE/MFE data available for simulation")
        return

    print(f"\n{label} ({len(trades_with_data)} trades with MAE/MFE data):")

    # Test different stop widths (in R-multiples)
    for stop_r in [0.5, 1.0, 1.5, 2.0]:
        conversions = 0
        still_losers = 0
        new_total_pnl = 0

        for t in trades_with_data:
            mae = abs(t["mae"])  # MAE is usually negative for longs
            mfe = t["mfe"]

            if mae < stop_r:
                # Trade survived the adverse move
                if mfe >= 1.0:
                    # Would have hit T1
                    conversions += 1
                    new_total_pnl += 0.6  # T1 at 1R, 60% exit
                else:
                    # Survived but didn't hit T1
                    still_losers += 1
                    new_total_pnl += t["pnl"] * 0.5  # Assume smaller loss
            else:
                # Still hit stop
                still_losers += 1
                new_total_pnl += -stop_r  # Full stop loss at widened level

        original_pnl = sum(t["pnl"] for t in trades_with_data)

        print(f"  Stop at {stop_r}R: {conversions} conversions to winner, {still_losers} still losers")
        print(f"    Original PnL: {original_pnl:.2f} -> Simulated: {new_total_pnl:.2f}")

spike_test_widening(long_losers, "LONG LOSERS")
spike_test_widening(short_losers, "SHORT LOSERS")

# Detailed look at biggest losers
print("\n" + "="*60)
print("TOP 10 LONG LOSERS - DETAILED ANALYSIS")
print("="*60)

sorted_losers = sorted(long_losers, key=lambda x: x["pnl"])[:10]
for t in sorted_losers:
    print(f"\n{t['symbol']}")
    print(f"  PnL: {t['pnl']:.2f}")
    print(f"  Entry: {t['entry_price']}, Stop: {t['stop_loss']}, Exit: {t['exit_price']}")
    print(f"  MAE: {t['mae']}R, MFE: {t['mfe']}R")
    print(f"  Exit reasons: {t['exit_reasons']}")

    if t['entry_price'] and t['stop_loss']:
        original_risk = abs(t['entry_price'] - t['stop_loss'])
        print(f"  Original risk/share: {original_risk:.2f}")

        # Check what happened
        if t['mae'] is not None and t['mfe'] is not None:
            if abs(t['mae']) < 1.0:
                print(f"  -> Did NOT hit 1R stop (MAE={t['mae']:.2f}R)")
                if t['mfe'] >= 1.0:
                    print(f"  -> BUT reached T1 (MFE={t['mfe']:.2f}R)! Would have WON with wider stop")
                else:
                    print(f"  -> And didn't reach T1 either (MFE={t['mfe']:.2f}R)")
            else:
                print(f"  -> Hit full stop (MAE={t['mae']:.2f}R)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if long_loser_mae:
    mae_under_1r = sum(1 for m in long_loser_mae if abs(m) < 1.0)
    mae_under_1_5r = sum(1 for m in long_loser_mae if abs(m) < 1.5)
    total = len(long_loser_mae)

    print(f"\nLONG LOSERS that did NOT hit 1R stop: {mae_under_1r}/{total} ({100*mae_under_1r/total:.0f}%)")
    print(f"LONG LOSERS that did NOT hit 1.5R stop: {mae_under_1_5r}/{total} ({100*mae_under_1_5r/total:.0f}%)")

    # Check how many of those would have hit T1
    potential_winners = 0
    for t in long_losers:
        if t["mae"] is not None and t["mfe"] is not None:
            if abs(t["mae"]) < 1.0 and t["mfe"] >= 1.0:
                potential_winners += 1

    print(f"\nPotential winners with 1R stop (MAE < 1R AND MFE >= 1R): {potential_winners}")
