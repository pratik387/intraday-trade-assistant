"""
PRO TRADER FORENSIC ANALYSIS
- Load actual 1-minute OHLCV data
- Analyze what happened AFTER exits
- Calculate MAE/MFE for each trade
- Find the "almost winners" that got stopped out
- Time-in-trade analysis
- Correlation with market (Nifty)
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import statistics

BACKTEST_DIR = Path("backtest_20251205-102858_extracted/20251205-102858_full/20251205-102858")
OHLCV_CACHE = Path("cache/ohlcv_archive")

def load_ohlcv(symbol, date_str):
    """Load 1-minute OHLCV data for a symbol on a given date"""
    # Symbol format: NSE:SYMBOL -> SYMBOL.NS
    clean_symbol = symbol.replace("NSE:", "") + ".NS"

    # Try different path patterns
    paths_to_try = [
        OHLCV_CACHE / clean_symbol / f"{date_str}.feather",
        OHLCV_CACHE / clean_symbol / f"{date_str.replace('-', '')}.feather",
    ]

    for path in paths_to_try:
        if path.exists():
            try:
                df = pd.read_feather(path)
                return df
            except:
                pass
    return None

def load_all_trade_data():
    """Load all trade data from backtest"""
    all_decisions = []
    all_triggers = []
    all_exits = []

    sessions = [d for d in BACKTEST_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()]

    for session_dir in sessions:
        date_str = session_dir.name
        events_file = session_dir / "events.jsonl"
        analytics_file = session_dir / "analytics.jsonl"

        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        event['_date'] = date_str
                        if event.get('type') == 'DECISION':
                            all_decisions.append(event)
                        elif event.get('type') == 'TRIGGER':
                            all_triggers.append(event)
                        elif event.get('type') == 'EXIT':
                            all_exits.append(event)
                    except:
                        pass

        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        event['_date'] = date_str
                        if event.get('stage') == 'EXIT':
                            all_exits.append(event)
                    except:
                        pass

    return all_decisions, all_triggers, all_exits

def analyze_what_happened_after_sl(decisions, triggers, exits):
    """
    CRITICAL: For each hard_sl trade, what happened AFTER we got stopped out?
    - Did price reverse and go to our target?
    - How much did we miss?
    """
    print("\n" + "="*100)
    print("FORENSIC ANALYSIS: What happened AFTER hard_sl exits?")
    print("="*100)

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    trigger_lookup = {t.get('trade_id'): t for t in triggers}

    # Get final exits only
    final_exits = {}
    for e in exits:
        trade_id = e.get('trade_id')
        if e.get('is_final_exit') or e.get('stage') == 'EXIT':
            if trade_id not in final_exits or e.get('is_final_exit'):
                final_exits[trade_id] = e

    hard_sl_exits = [e for e in final_exits.values() if e.get('reason') == 'hard_sl']

    print(f"\nAnalyzing {len(hard_sl_exits)} hard_sl trades...")

    would_have_won = []
    correct_sl = []

    for e in hard_sl_exits[:50]:  # Sample first 50
        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})
        t = trigger_lookup.get(trade_id, {})

        if not d:
            continue

        symbol = e.get('symbol', d.get('symbol'))
        date_str = e.get('_date', d.get('_date'))

        if not symbol or not date_str:
            continue

        # Load OHLCV data
        ohlcv = load_ohlcv(symbol, date_str)
        if ohlcv is None or len(ohlcv) == 0:
            continue

        # Get trade parameters
        plan = d.get('plan', {})
        bias = plan.get('bias', 'long')
        entry_price = t.get('trigger', {}).get('actual_price') if t else e.get('actual_entry_price')
        sl_price = plan.get('stop', {}).get('hard')
        targets = plan.get('targets', [])
        t1_price = targets[0].get('level') if targets else None
        t2_price = targets[1].get('level') if len(targets) > 1 else None

        if not entry_price or not sl_price:
            continue

        # Find exit time
        exit_ts = e.get('timestamp', e.get('ts'))
        if not exit_ts:
            continue

        try:
            exit_time = pd.to_datetime(exit_ts)
        except:
            continue

        # Get price data AFTER exit
        ohlcv['timestamp'] = pd.to_datetime(ohlcv['timestamp'])
        after_exit = ohlcv[ohlcv['timestamp'] > exit_time]

        if len(after_exit) == 0:
            continue

        # What was the best price after exit?
        if bias == 'long':
            best_price_after = after_exit['high'].max()
            worst_price_after = after_exit['low'].min()
            would_hit_t1 = t1_price and best_price_after >= t1_price
            would_hit_t2 = t2_price and best_price_after >= t2_price
        else:
            best_price_after = after_exit['low'].min()
            worst_price_after = after_exit['high'].max()
            would_hit_t1 = t1_price and best_price_after <= t1_price
            would_hit_t2 = t2_price and best_price_after <= t2_price

        trade_info = {
            'trade_id': trade_id,
            'symbol': symbol,
            'setup': e.get('setup_type'),
            'bias': bias,
            'entry_price': entry_price,
            'sl_price': sl_price,
            't1_price': t1_price,
            't2_price': t2_price,
            'exit_price': e.get('exit_price'),
            'pnl': e.get('total_trade_pnl', e.get('pnl')),
            'best_after': best_price_after,
            'worst_after': worst_price_after,
            'would_hit_t1': would_hit_t1,
            'would_hit_t2': would_hit_t2,
            'bars_after_exit': len(after_exit)
        }

        if would_hit_t1 or would_hit_t2:
            would_have_won.append(trade_info)
        else:
            correct_sl.append(trade_info)

    print(f"\n--- ANALYSIS OF {len(would_have_won) + len(correct_sl)} HARD_SL TRADES ---")
    print(f"\nWOULD HAVE WON (price hit our target AFTER we got stopped): {len(would_have_won)}")
    print(f"CORRECT STOP (price continued against us): {len(correct_sl)}")

    if would_have_won:
        print(f"\n{'='*80}")
        print("TRADES WHERE WE GOT STOPPED OUT BUT WOULD HAVE WON:")
        print(f"{'='*80}")

        total_missed_pnl = sum(t['pnl'] for t in would_have_won)
        print(f"\nTotal PnL lost from premature stops: Rs {total_missed_pnl:.0f}")

        # Group by setup
        by_setup = defaultdict(list)
        for t in would_have_won:
            by_setup[t['setup']].append(t)

        print(f"\nBy Setup:")
        for setup, trades in sorted(by_setup.items(), key=lambda x: sum(t['pnl'] for t in x[1])):
            total_pnl = sum(t['pnl'] for t in trades)
            t1_count = sum(1 for t in trades if t['would_hit_t1'])
            t2_count = sum(1 for t in trades if t['would_hit_t2'])
            print(f"  {setup}: {len(trades)} trades, Rs {total_pnl:.0f} lost")
            print(f"    Would hit T1: {t1_count}, Would hit T2: {t2_count}")

        # Show examples
        print(f"\nExamples of 'Almost Winners':")
        for t in would_have_won[:5]:
            print(f"\n  {t['symbol']} ({t['setup']}):")
            print(f"    Entry: {t['entry_price']:.2f}, SL: {t['sl_price']:.2f}")
            print(f"    T1: {t['t1_price']:.2f}, T2: {t['t2_price']:.2f}")
            print(f"    Got stopped at: {t['exit_price']:.2f}")
            print(f"    Best price AFTER stop: {t['best_after']:.2f}")
            print(f"    PnL lost: Rs {t['pnl']:.0f}")
            if t['would_hit_t2']:
                print(f"    --> WOULD HAVE HIT T2!")
            elif t['would_hit_t1']:
                print(f"    --> Would have hit T1")

def analyze_mae_mfe(decisions, triggers, exits):
    """
    MAE (Maximum Adverse Excursion) - How far did price go against us?
    MFE (Maximum Favorable Excursion) - How far did price go in our favor?

    This tells us:
    - Are we getting stopped out at the worst moment?
    - Are we leaving money on the table?
    """
    print("\n" + "="*100)
    print("MAE/MFE ANALYSIS - Where are we leaving money?")
    print("="*100)

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    trigger_lookup = {t.get('trade_id'): t for t in triggers}

    final_exits = {}
    for e in exits:
        trade_id = e.get('trade_id')
        if e.get('is_final_exit'):
            final_exits[trade_id] = e

    mae_mfe_data = []

    for trade_id, e in list(final_exits.items())[:100]:  # Sample
        d = decision_lookup.get(trade_id, {})
        t = trigger_lookup.get(trade_id, {})

        if not d:
            continue

        symbol = e.get('symbol', d.get('symbol'))
        date_str = e.get('_date', d.get('_date'))

        if not symbol or not date_str:
            continue

        ohlcv = load_ohlcv(symbol, date_str)
        if ohlcv is None or len(ohlcv) == 0:
            continue

        plan = d.get('plan', {})
        bias = plan.get('bias', 'long')
        entry_price = t.get('trigger', {}).get('actual_price') if t else e.get('actual_entry_price')

        if not entry_price:
            continue

        # Get entry and exit times
        entry_ts = t.get('trigger', {}).get('ts') if t else d.get('ts')
        exit_ts = e.get('timestamp', e.get('ts'))

        if not entry_ts or not exit_ts:
            continue

        try:
            entry_time = pd.to_datetime(entry_ts)
            exit_time = pd.to_datetime(exit_ts)
        except:
            continue

        ohlcv['timestamp'] = pd.to_datetime(ohlcv['timestamp'])
        during_trade = ohlcv[(ohlcv['timestamp'] >= entry_time) & (ohlcv['timestamp'] <= exit_time)]

        if len(during_trade) == 0:
            continue

        if bias == 'long':
            mae = entry_price - during_trade['low'].min()  # Max drop
            mfe = during_trade['high'].max() - entry_price  # Max rise
        else:
            mae = during_trade['high'].max() - entry_price  # Max rise (bad for short)
            mfe = entry_price - during_trade['low'].min()  # Max drop (good for short)

        pnl = e.get('total_trade_pnl', e.get('pnl', 0))
        atr = d.get('plan', {}).get('indicators', {}).get('atr', 1)

        mae_mfe_data.append({
            'trade_id': trade_id,
            'setup': e.get('setup_type'),
            'bias': bias,
            'pnl': pnl,
            'mae': mae,
            'mfe': mfe,
            'mae_atr': mae / atr if atr else 0,
            'mfe_atr': mfe / atr if atr else 0,
            'exit_reason': e.get('reason'),
            'is_winner': pnl > 0
        })

    if not mae_mfe_data:
        print("No MAE/MFE data could be calculated")
        return

    winners = [t for t in mae_mfe_data if t['is_winner']]
    losers = [t for t in mae_mfe_data if not t['is_winner']]

    print(f"\nAnalyzed {len(mae_mfe_data)} trades")
    print(f"Winners: {len(winners)}, Losers: {len(losers)}")

    if winners:
        print(f"\n--- WINNERS ---")
        print(f"Avg MAE (drawdown before winning): {statistics.mean([t['mae_atr'] for t in winners]):.2f} ATR")
        print(f"Avg MFE (max favorable): {statistics.mean([t['mfe_atr'] for t in winners]):.2f} ATR")

    if losers:
        print(f"\n--- LOSERS ---")
        print(f"Avg MAE: {statistics.mean([t['mae_atr'] for t in losers]):.2f} ATR")
        print(f"Avg MFE (they DID go in our favor by): {statistics.mean([t['mfe_atr'] for t in losers]):.2f} ATR")

        # How many losers actually went positive first?
        went_positive = [t for t in losers if t['mfe'] > 0]
        print(f"\nLosers that went positive first: {len(went_positive)} ({len(went_positive)/len(losers)*100:.0f}%)")
        if went_positive:
            print(f"  Avg MFE before reversal: {statistics.mean([t['mfe_atr'] for t in went_positive]):.2f} ATR")

def analyze_time_in_trade(decisions, triggers, exits):
    """
    How long are trades lasting?
    - Are losers dying quickly?
    - Are winners being held long enough?
    """
    print("\n" + "="*100)
    print("TIME IN TRADE ANALYSIS")
    print("="*100)

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    trigger_lookup = {t.get('trade_id'): t for t in triggers}

    final_exits = {e.get('trade_id'): e for e in exits if e.get('is_final_exit')}

    time_data = []

    for trade_id, e in final_exits.items():
        t = trigger_lookup.get(trade_id, {})

        entry_ts = t.get('ts') if t else None
        exit_ts = e.get('timestamp')

        if not entry_ts or not exit_ts:
            continue

        try:
            entry_time = pd.to_datetime(entry_ts)
            exit_time = pd.to_datetime(exit_ts)
            duration_mins = (exit_time - entry_time).total_seconds() / 60
        except:
            continue

        if duration_mins < 0 or duration_mins > 400:  # Filter bad data
            continue

        time_data.append({
            'trade_id': trade_id,
            'setup': e.get('setup_type'),
            'duration_mins': duration_mins,
            'pnl': e.get('total_trade_pnl', e.get('pnl', 0)),
            'exit_reason': e.get('reason'),
            'is_winner': e.get('total_trade_pnl', e.get('pnl', 0)) > 0
        })

    if not time_data:
        print("No time data available")
        return

    winners = [t for t in time_data if t['is_winner']]
    losers = [t for t in time_data if not t['is_winner']]
    hard_sl = [t for t in time_data if t['exit_reason'] == 'hard_sl']

    print(f"\nTotal trades analyzed: {len(time_data)}")

    print(f"\n--- TIME IN TRADE ---")
    print(f"All trades avg duration: {statistics.mean([t['duration_mins'] for t in time_data]):.0f} mins")

    if winners:
        print(f"Winners avg duration: {statistics.mean([t['duration_mins'] for t in winners]):.0f} mins")
    if losers:
        print(f"Losers avg duration: {statistics.mean([t['duration_mins'] for t in losers]):.0f} mins")
    if hard_sl:
        print(f"Hard_SL avg duration: {statistics.mean([t['duration_mins'] for t in hard_sl]):.0f} mins")

    # Distribution
    print(f"\n--- DURATION DISTRIBUTION ---")
    buckets = [(0, 30), (30, 60), (60, 120), (120, 240), (240, 999)]

    for low, high in buckets:
        bucket_trades = [t for t in time_data if low <= t['duration_mins'] < high]
        if bucket_trades:
            wins = len([t for t in bucket_trades if t['is_winner']])
            total = len(bucket_trades)
            total_pnl = sum(t['pnl'] for t in bucket_trades)
            print(f"  {low}-{high} mins: {total} trades, WR={wins/total*100:.0f}%, PnL=Rs {total_pnl:.0f}")

def analyze_daily_patterns(decisions, triggers, exits):
    """
    Are there specific days where we lose more?
    Cluster analysis of losses
    """
    print("\n" + "="*100)
    print("DAILY PATTERN ANALYSIS - Where do losses cluster?")
    print("="*100)

    final_exits = {e.get('trade_id'): e for e in exits if e.get('is_final_exit')}

    # Group by date
    by_date = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0, 'hard_sl': 0})

    for e in final_exits.values():
        date_str = e.get('_date', '')
        if not date_str:
            continue

        pnl = e.get('total_trade_pnl', e.get('pnl', 0))
        by_date[date_str]['trades'] += 1
        by_date[date_str]['pnl'] += pnl
        if pnl > 0:
            by_date[date_str]['wins'] += 1
        if e.get('reason') == 'hard_sl':
            by_date[date_str]['hard_sl'] += 1

    # Find worst days
    sorted_days = sorted(by_date.items(), key=lambda x: x[1]['pnl'])

    print(f"\n--- WORST 10 DAYS ---")
    for date_str, data in sorted_days[:10]:
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        print(f"  {date_str}: {data['trades']} trades, WR={wr:.0f}%, PnL=Rs {data['pnl']:.0f}, Hard_SL={data['hard_sl']}")

    print(f"\n--- BEST 10 DAYS ---")
    for date_str, data in sorted_days[-10:]:
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        print(f"  {date_str}: {data['trades']} trades, WR={wr:.0f}%, PnL=Rs {data['pnl']:.0f}, Hard_SL={data['hard_sl']}")

def analyze_entry_quality(decisions, triggers, exits):
    """
    Are we entering at bad prices?
    - Distance from entry zone to actual entry
    - Slippage patterns
    """
    print("\n" + "="*100)
    print("ENTRY QUALITY ANALYSIS - Are we entering at bad prices?")
    print("="*100)

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    trigger_lookup = {t.get('trade_id'): t for t in triggers}
    final_exits = {e.get('trade_id'): e for e in exits if e.get('is_final_exit')}

    entry_data = []

    for trade_id, e in final_exits.items():
        d = decision_lookup.get(trade_id, {})
        t = trigger_lookup.get(trade_id, {})

        if not d:
            continue

        plan = d.get('plan', {})
        entry_zone = plan.get('entry', {}).get('zone', [])
        entry_ref = plan.get('entry_ref_price')
        actual_entry = t.get('trigger', {}).get('actual_price') if t else e.get('actual_entry_price')
        atr = plan.get('indicators', {}).get('atr', 1)
        bias = plan.get('bias')

        if not entry_ref or not actual_entry or not atr:
            continue

        # Calculate slippage
        if bias == 'long':
            slippage = actual_entry - entry_ref  # Positive = bad for long
        else:
            slippage = entry_ref - actual_entry  # Positive = bad for short

        slippage_atr = slippage / atr

        entry_data.append({
            'trade_id': trade_id,
            'setup': e.get('setup_type'),
            'bias': bias,
            'slippage': slippage,
            'slippage_atr': slippage_atr,
            'pnl': e.get('total_trade_pnl', e.get('pnl', 0)),
            'exit_reason': e.get('reason'),
            'is_winner': e.get('total_trade_pnl', e.get('pnl', 0)) > 0
        })

    if not entry_data:
        print("No entry data available")
        return

    winners = [t for t in entry_data if t['is_winner']]
    losers = [t for t in entry_data if not t['is_winner']]

    print(f"\nAnalyzed {len(entry_data)} trades")

    print(f"\n--- ENTRY SLIPPAGE (in ATR) ---")
    print(f"All trades: avg slippage = {statistics.mean([t['slippage_atr'] for t in entry_data]):.3f} ATR")
    if winners:
        print(f"Winners: avg slippage = {statistics.mean([t['slippage_atr'] for t in winners]):.3f} ATR")
    if losers:
        print(f"Losers: avg slippage = {statistics.mean([t['slippage_atr'] for t in losers]):.3f} ATR")

    # High slippage trades
    high_slip = [t for t in entry_data if t['slippage_atr'] > 0.3]
    if high_slip:
        print(f"\n--- HIGH SLIPPAGE TRADES (>0.3 ATR) ---")
        print(f"Count: {len(high_slip)}")
        print(f"Avg PnL: Rs {statistics.mean([t['pnl'] for t in high_slip]):.0f}")
        print(f"Total PnL: Rs {sum(t['pnl'] for t in high_slip):.0f}")

        # By setup
        by_setup = defaultdict(list)
        for t in high_slip:
            by_setup[t['setup']].append(t)
        print(f"\nHigh slippage by setup:")
        for setup, trades in sorted(by_setup.items(), key=lambda x: -len(x[1])):
            print(f"  {setup}: {len(trades)} trades, avg slip {statistics.mean([t['slippage_atr'] for t in trades]):.2f} ATR")

def analyze_consecutive_patterns(decisions, triggers, exits):
    """
    Are losses coming in clusters?
    Streak analysis
    """
    print("\n" + "="*100)
    print("CONSECUTIVE PATTERN ANALYSIS - Loss streaks")
    print("="*100)

    final_exits = [e for e in exits if e.get('is_final_exit')]
    final_exits.sort(key=lambda x: (x.get('_date', ''), x.get('timestamp', '')))

    # Calculate streaks
    current_streak = 0
    max_loss_streak = 0
    max_win_streak = 0
    streaks = []

    for e in final_exits:
        pnl = e.get('total_trade_pnl', e.get('pnl', 0))

        if pnl > 0:
            if current_streak < 0:
                streaks.append(current_streak)
                current_streak = 1
            else:
                current_streak += 1
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if current_streak > 0:
                streaks.append(current_streak)
                current_streak = -1
            else:
                current_streak -= 1
            max_loss_streak = min(max_loss_streak, current_streak)

    if current_streak != 0:
        streaks.append(current_streak)

    print(f"\nMax win streak: {max_win_streak}")
    print(f"Max loss streak: {abs(max_loss_streak)}")

    # Analyze what happens after loss streaks
    loss_streaks = [s for s in streaks if s < 0]
    if loss_streaks:
        print(f"\nLoss streak distribution:")
        for streak_len in range(1, 6):
            count = len([s for s in loss_streaks if s == -streak_len])
            if count > 0:
                print(f"  {streak_len} losses in a row: {count} times")

def main():
    print("="*100)
    print("PRO TRADER FORENSIC ANALYSIS")
    print("="*100)

    print("\nLoading all trade data...")
    decisions, triggers, exits = load_all_trade_data()
    print(f"Loaded {len(decisions)} decisions, {len(triggers)} triggers, {len(exits)} exits")

    analyze_what_happened_after_sl(decisions, triggers, exits)
    analyze_mae_mfe(decisions, triggers, exits)
    analyze_time_in_trade(decisions, triggers, exits)
    analyze_daily_patterns(decisions, triggers, exits)
    analyze_entry_quality(decisions, triggers, exits)
    analyze_consecutive_patterns(decisions, triggers, exits)

if __name__ == "__main__":
    main()
