"""
DEEP ANALYSIS: Hard SL, Exit Management, and High-Probability Setups
"""
import json
from pathlib import Path
from collections import defaultdict
import statistics

BACKTEST_DIR = Path("backtest_20251205-102858_extracted/20251205-102858_full/20251205-102858")

def load_all_data():
    """Load ALL events including triggers and partial exits"""
    all_decisions = []
    all_exits = []  # ALL exits including partials
    all_triggers = []

    sessions = [d for d in BACKTEST_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()]
    print(f"Found {len(sessions)} sessions")

    for session_dir in sessions:
        events_file = session_dir / "events.jsonl"
        analytics_file = session_dir / "analytics.jsonl"

        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('type') == 'DECISION':
                            event['_session'] = session_dir.name
                            all_decisions.append(event)
                        elif event.get('type') == 'TRIGGER':
                            event['_session'] = session_dir.name
                            all_triggers.append(event)
                    except:
                        pass

        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('stage') == 'EXIT':
                            event['_session'] = session_dir.name
                            all_exits.append(event)
                    except:
                        pass

    return all_decisions, all_triggers, all_exits

def analyze_hard_sl_deep(decisions, triggers, exits):
    """
    DEEP DIVE: Why are trades hitting hard SL?
    - SL distance in ATR multiples
    - Entry slippage effect
    - Time in trade before SL
    - MAE (Maximum Adverse Excursion) analysis
    - MFE (Maximum Favorable Excursion) before SL
    """
    print("\n" + "="*100)
    print("HARD SL DEEP ANALYSIS - Why are 89 trades losing Rs 45,558?")
    print("="*100)

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    trigger_lookup = {t.get('trade_id'): t for t in triggers}

    # Get only final exits
    final_exits = [e for e in exits if e.get('is_final_exit', False)]
    hard_sl_exits = [e for e in final_exits if e.get('reason') == 'hard_sl']

    print(f"\nTotal hard_sl trades: {len(hard_sl_exits)}")
    total_loss = sum(e.get('total_trade_pnl', e.get('pnl', 0)) for e in hard_sl_exits)
    print(f"Total loss: Rs {total_loss:.2f}")

    # Detailed analysis
    hard_sl_details = []
    for e in hard_sl_exits:
        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})
        t = trigger_lookup.get(trade_id, {})

        if not d:
            continue

        plan = d.get('plan', {})
        indicators = plan.get('indicators', {})
        quality = plan.get('quality', {})
        sizing = plan.get('sizing', {})
        stop = plan.get('stop', {})

        entry_ref_price = plan.get('entry_ref_price', 0)
        actual_entry = t.get('trigger', {}).get('actual_price', 0) if t else e.get('actual_entry_price', 0)
        hard_sl_price = stop.get('hard', 0)
        atr = indicators.get('atr', 0)

        # Calculate SL distance
        if entry_ref_price and hard_sl_price and atr:
            sl_distance = abs(entry_ref_price - hard_sl_price)
            sl_atr_mult = sl_distance / atr if atr > 0 else 0

            # Entry slippage
            slippage = abs(actual_entry - entry_ref_price) if actual_entry else 0
            slippage_atr = slippage / atr if atr > 0 else 0

            # Effective SL distance after slippage
            if plan.get('bias') == 'long':
                effective_sl_distance = actual_entry - hard_sl_price if actual_entry else sl_distance
            else:
                effective_sl_distance = hard_sl_price - actual_entry if actual_entry else sl_distance

            effective_sl_atr = effective_sl_distance / atr if atr > 0 else 0
        else:
            sl_atr_mult = 0
            slippage_atr = 0
            effective_sl_atr = 0

        hard_sl_details.append({
            'trade_id': trade_id,
            'setup': e.get('setup_type'),
            'pnl': e.get('total_trade_pnl', e.get('pnl', 0)),
            'regime': e.get('regime'),
            'entry_hour': d.get('ts', '')[11:13] if d.get('ts') else '',
            'sl_atr_mult': sl_atr_mult,
            'slippage_atr': slippage_atr,
            'effective_sl_atr': effective_sl_atr,
            'slippage_bps': e.get('slippage_bps', 0),
            'atr': atr,
            'adx': indicators.get('adx'),
            'structural_rr': quality.get('structural_rr'),
            'risk_per_share': sizing.get('risk_per_share', 0),
            'entry_ref': entry_ref_price,
            'actual_entry': actual_entry,
            'sl_price': hard_sl_price,
            'bias': plan.get('bias')
        })

    # Analysis 1: SL Distance in ATR
    print("\n" + "-"*80)
    print("1. SL DISTANCE ANALYSIS (in ATR multiples)")
    print("-"*80)

    sl_atr_values = [t['sl_atr_mult'] for t in hard_sl_details if t['sl_atr_mult'] > 0]
    effective_sl_values = [t['effective_sl_atr'] for t in hard_sl_details if t['effective_sl_atr'] > 0]

    if sl_atr_values:
        print(f"\nPlanned SL distance (ATR):")
        print(f"  Avg: {statistics.mean(sl_atr_values):.2f} ATR")
        print(f"  Median: {statistics.median(sl_atr_values):.2f} ATR")
        print(f"  Min: {min(sl_atr_values):.2f} ATR")
        print(f"  Max: {max(sl_atr_values):.2f} ATR")

        # Bucket analysis
        print(f"\n  Distribution:")
        buckets = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 999)]
        for low, high in buckets:
            count = len([x for x in sl_atr_values if low <= x < high])
            if count > 0:
                avg_pnl = statistics.mean([t['pnl'] for t in hard_sl_details if low <= t['sl_atr_mult'] < high])
                print(f"    {low:.1f}-{high:.1f} ATR: {count} trades, avg loss Rs {avg_pnl:.0f}")

    if effective_sl_values:
        print(f"\nEffective SL distance AFTER slippage:")
        print(f"  Avg: {statistics.mean(effective_sl_values):.2f} ATR")
        print(f"  Median: {statistics.median(effective_sl_values):.2f} ATR")

    # Analysis 2: Slippage Impact
    print("\n" + "-"*80)
    print("2. SLIPPAGE IMPACT ON HARD SL")
    print("-"*80)

    slippage_values = [t['slippage_bps'] for t in hard_sl_details if t['slippage_bps']]
    slippage_atr_values = [t['slippage_atr'] for t in hard_sl_details if t['slippage_atr'] > 0]

    if slippage_values:
        print(f"\nSlippage (BPS):")
        print(f"  Avg: {statistics.mean(slippage_values):.0f} bps")
        print(f"  Median: {statistics.median(slippage_values):.0f} bps")
        print(f"  Max: {max(slippage_values):.0f} bps")

        # High slippage trades
        high_slip = [t for t in hard_sl_details if t['slippage_bps'] and t['slippage_bps'] > 100]
        if high_slip:
            print(f"\n  High slippage (>100 bps): {len(high_slip)} trades")
            print(f"  PnL from high slippage hard_sl: Rs {sum(t['pnl'] for t in high_slip):.0f}")

    if slippage_atr_values:
        print(f"\nSlippage as % of SL distance:")
        print(f"  Avg slippage: {statistics.mean(slippage_atr_values):.2f} ATR")
        print(f"  This reduces effective SL room by ~{statistics.mean(slippage_atr_values)/statistics.mean(sl_atr_values)*100:.0f}%" if sl_atr_values else "")

    # Analysis 3: By Setup Type - Which setups have tight SL?
    print("\n" + "-"*80)
    print("3. HARD SL BY SETUP - SL Tightness Analysis")
    print("-"*80)

    by_setup = defaultdict(list)
    for t in hard_sl_details:
        by_setup[t['setup']].append(t)

    print(f"\n{'Setup':<25} {'Count':>6} {'Avg Loss':>10} {'Avg SL ATR':>10} {'Avg Slip':>10}")
    print("-"*70)

    for setup in sorted(by_setup.keys(), key=lambda s: sum(t['pnl'] for t in by_setup[s])):
        trades = by_setup[setup]
        count = len(trades)
        avg_loss = statistics.mean([t['pnl'] for t in trades])
        avg_sl_atr = statistics.mean([t['sl_atr_mult'] for t in trades if t['sl_atr_mult'] > 0]) if any(t['sl_atr_mult'] > 0 for t in trades) else 0
        avg_slip = statistics.mean([t['slippage_bps'] for t in trades if t['slippage_bps']]) if any(t['slippage_bps'] for t in trades) else 0
        total_pnl = sum(t['pnl'] for t in trades)
        print(f"{setup:<25} {count:>6} {avg_loss:>10.0f} {avg_sl_atr:>10.2f} {avg_slip:>10.0f}  Total: Rs {total_pnl:.0f}")

    # Analysis 4: By ADX - Are low ADX trades getting stopped out more?
    print("\n" + "-"*80)
    print("4. HARD SL BY ADX - Volatility Impact")
    print("-"*80)

    adx_buckets = [(0, 20), (20, 25), (25, 30), (30, 40), (40, 100)]
    for low, high in adx_buckets:
        trades = [t for t in hard_sl_details if t['adx'] and low <= t['adx'] < high]
        if trades:
            avg_sl_atr = statistics.mean([t['sl_atr_mult'] for t in trades if t['sl_atr_mult'] > 0]) if any(t['sl_atr_mult'] > 0 for t in trades) else 0
            total_pnl = sum(t['pnl'] for t in trades)
            print(f"ADX {low}-{high}: {len(trades)} trades, avg SL {avg_sl_atr:.2f} ATR, total loss Rs {total_pnl:.0f}")

    # Analysis 5: Recommendations
    print("\n" + "-"*80)
    print("5. SL IMPROVEMENT RECOMMENDATIONS")
    print("-"*80)

    # Find trades where SL was very tight
    very_tight_sl = [t for t in hard_sl_details if 0 < t['sl_atr_mult'] < 0.8]
    if very_tight_sl:
        print(f"\nVery tight SL (<0.8 ATR): {len(very_tight_sl)} trades")
        print(f"  Total loss: Rs {sum(t['pnl'] for t in very_tight_sl):.0f}")
        print(f"  Setups affected:")
        setup_counts = defaultdict(int)
        for t in very_tight_sl:
            setup_counts[t['setup']] += 1
        for setup, count in sorted(setup_counts.items(), key=lambda x: -x[1]):
            print(f"    {setup}: {count}")

    # Find trades where slippage ate most of SL buffer
    high_slip_ratio = [t for t in hard_sl_details if t['slippage_atr'] > 0 and t['sl_atr_mult'] > 0 and t['slippage_atr'] / t['sl_atr_mult'] > 0.3]
    if high_slip_ratio:
        print(f"\nSlippage ate >30% of SL buffer: {len(high_slip_ratio)} trades")
        print(f"  Total loss: Rs {sum(t['pnl'] for t in high_slip_ratio):.0f}")

def analyze_exit_management(decisions, triggers, exits):
    """
    DEEP DIVE: Are we exiting too early?
    - T1 exits - could we have gotten more?
    - T2 exits - what was the potential beyond?
    - Time in trade analysis
    """
    print("\n" + "="*100)
    print("EXIT MANAGEMENT ANALYSIS - Are we leaving money on the table?")
    print("="*100)

    decision_lookup = {d.get('trade_id'): d for d in decisions}

    # Group exits by trade_id
    exits_by_trade = defaultdict(list)
    for e in exits:
        exits_by_trade[e.get('trade_id')].append(e)

    # Analyze T1 and T2 exits
    t1_exits = []
    t2_exits = []
    eod_exits = []
    sl_post_t1_exits = []

    for trade_id, trade_exits in exits_by_trade.items():
        # Sort by timestamp
        trade_exits.sort(key=lambda x: x.get('timestamp', ''))

        for e in trade_exits:
            reason = e.get('reason', '')
            pnl = e.get('pnl', 0)
            d = decision_lookup.get(trade_id, {})

            exit_data = {
                'trade_id': trade_id,
                'reason': reason,
                'pnl': pnl,
                'setup': e.get('setup_type'),
                'regime': e.get('regime'),
                'is_final': e.get('is_final_exit', False),
                'total_pnl': e.get('total_trade_pnl', pnl),
                'exit_sequence': e.get('exit_sequence', 1),
                'r_multiple': e.get('analytics', {}).get('r_multiple') if isinstance(e.get('analytics'), dict) else None
            }

            if 't1' in reason.lower():
                t1_exits.append(exit_data)
            elif 't2' in reason.lower() or 'target_t2' in reason.lower():
                t2_exits.append(exit_data)
            elif 'eod' in reason.lower():
                eod_exits.append(exit_data)
            elif 'sl_post_t1' in reason.lower():
                sl_post_t1_exits.append(exit_data)

    print(f"\nExit breakdown:")
    print(f"  T1 partial exits: {len(t1_exits)}")
    print(f"  T2 full exits: {len(t2_exits)}")
    print(f"  SL post T1: {len(sl_post_t1_exits)}")
    print(f"  EOD exits: {len(eod_exits)}")

    # T1 Analysis
    print("\n" + "-"*80)
    print("T1 EXIT ANALYSIS - 60% of position exited at T1")
    print("-"*80)

    if t1_exits:
        t1_pnl = sum(e['pnl'] for e in t1_exits)
        print(f"\nT1 exits: {len(t1_exits)}")
        print(f"Total PnL from T1 exits: Rs {t1_pnl:.0f}")
        print(f"Avg PnL per T1 exit: Rs {t1_pnl/len(t1_exits):.0f}")

        # What happened after T1?
        t1_trade_ids = set(e['trade_id'] for e in t1_exits)
        t1_final_outcomes = []
        for trade_id in t1_trade_ids:
            trade_exits = exits_by_trade.get(trade_id, [])
            final_exit = [e for e in trade_exits if e.get('is_final_exit', False)]
            if final_exit:
                t1_final_outcomes.append({
                    'trade_id': trade_id,
                    'final_reason': final_exit[0].get('reason'),
                    'total_pnl': final_exit[0].get('total_trade_pnl', 0)
                })

        # Breakdown of what happened after T1
        print("\nWhat happened to remaining 40% after T1:")
        outcome_counts = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for o in t1_final_outcomes:
            reason = o['final_reason']
            if 't2' in reason.lower() or 'target_t2' in reason.lower():
                key = 'Hit T2'
            elif 'sl_post_t1' in reason.lower():
                key = 'SL post T1'
            elif 'eod' in reason.lower():
                key = 'EOD exit'
            else:
                key = reason
            outcome_counts[key]['count'] += 1
            outcome_counts[key]['pnl'] += o['total_pnl']

        for outcome, data in sorted(outcome_counts.items(), key=lambda x: -x[1]['count']):
            print(f"  {outcome}: {data['count']} trades, total PnL Rs {data['pnl']:.0f}")

    # T2 Analysis
    print("\n" + "-"*80)
    print("T2 EXIT ANALYSIS - Trades that hit T2")
    print("-"*80)

    if t2_exits:
        t2_pnl = sum(e['total_pnl'] for e in t2_exits if e['is_final'])
        print(f"\nT2 full exits: {len([e for e in t2_exits if e['is_final']])}")
        print(f"Total PnL from T2 trades: Rs {t2_pnl:.0f}")

        # By setup
        print("\nT2 hits by setup:")
        by_setup = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for e in t2_exits:
            if e['is_final']:
                by_setup[e['setup']]['count'] += 1
                by_setup[e['setup']]['pnl'] += e['total_pnl']
        for setup, data in sorted(by_setup.items(), key=lambda x: -x[1]['count']):
            print(f"  {setup}: {data['count']} trades, Rs {data['pnl']:.0f}")

    # SL Post T1 Analysis
    print("\n" + "-"*80)
    print("SL POST T1 ANALYSIS - Trades that hit T1 but then reversed")
    print("-"*80)

    if sl_post_t1_exits:
        sl_post_t1_pnl = sum(e['total_pnl'] for e in sl_post_t1_exits if e['is_final'])
        print(f"\nSL post T1 trades: {len([e for e in sl_post_t1_exits if e['is_final']])}")
        print(f"Total PnL: Rs {sl_post_t1_pnl:.0f}")
        print("(Note: These are BE or small winners since T1 was hit)")

        # By setup
        print("\nSL post T1 by setup:")
        by_setup = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for e in sl_post_t1_exits:
            if e['is_final']:
                by_setup[e['setup']]['count'] += 1
                by_setup[e['setup']]['pnl'] += e['total_pnl']
        for setup, data in sorted(by_setup.items(), key=lambda x: -x[1]['pnl']):
            print(f"  {setup}: {data['count']} trades, Rs {data['pnl']:.0f}")

    # EOD Analysis
    print("\n" + "-"*80)
    print("EOD EXIT ANALYSIS - Trades held till end of day")
    print("-"*80)

    if eod_exits:
        # Final EOD exits only
        eod_final = [e for e in eod_exits if e['is_final']]
        eod_pnl = sum(e['total_pnl'] for e in eod_final)
        print(f"\nEOD exits: {len(eod_final)}")
        print(f"Total PnL from EOD exits: Rs {eod_pnl:.0f}")

        # Positive vs negative EOD
        positive_eod = [e for e in eod_final if e['total_pnl'] > 0]
        negative_eod = [e for e in eod_final if e['total_pnl'] <= 0]
        print(f"\n  Positive EOD: {len(positive_eod)} trades, Rs {sum(e['total_pnl'] for e in positive_eod):.0f}")
        print(f"  Negative EOD: {len(negative_eod)} trades, Rs {sum(e['total_pnl'] for e in negative_eod):.0f}")

def analyze_high_probability_setups(decisions, triggers, exits):
    """
    DEEP DIVE: What makes the winning setups work?
    - orb_breakout_long: 61% WR
    - support_bounce_long: 100% WR
    - vwap_reclaim_long: 100% WR
    """
    print("\n" + "="*100)
    print("HIGH-PROBABILITY SETUP ANALYSIS - What makes winners work?")
    print("="*100)

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    final_exits = [e for e in exits if e.get('is_final_exit', False)]

    high_prob_setups = ['orb_breakout_long', 'support_bounce_long', 'vwap_reclaim_long']

    for setup in high_prob_setups:
        setup_exits = [e for e in final_exits if e.get('setup_type') == setup]
        if not setup_exits:
            continue

        print(f"\n{'='*80}")
        print(f"SETUP: {setup}")
        print(f"{'='*80}")

        winners = [e for e in setup_exits if e.get('total_trade_pnl', 0) > 0]
        losers = [e for e in setup_exits if e.get('total_trade_pnl', 0) <= 0]

        total_pnl = sum(e.get('total_trade_pnl', 0) for e in setup_exits)
        print(f"\nTotal trades: {len(setup_exits)}")
        print(f"Winners: {len(winners)}, Losers: {len(losers)}")
        print(f"Win Rate: {len(winners)/len(setup_exits)*100:.1f}%")
        print(f"Total PnL: Rs {total_pnl:.0f}")
        print(f"Avg PnL per trade: Rs {total_pnl/len(setup_exits):.0f}")

        if winners:
            print(f"\nWinners analysis:")
            print(f"  Avg winner: Rs {sum(e.get('total_trade_pnl', 0) for e in winners)/len(winners):.0f}")
            print(f"  Best trade: Rs {max(e.get('total_trade_pnl', 0) for e in winners):.0f}")

        if losers:
            print(f"\nLosers analysis:")
            print(f"  Avg loser: Rs {sum(e.get('total_trade_pnl', 0) for e in losers)/len(losers):.0f}")
            print(f"  Worst trade: Rs {min(e.get('total_trade_pnl', 0) for e in losers):.0f}")

        # Indicator analysis
        print("\nIndicator characteristics at entry:")
        adx_values = []
        rsi_values = []
        vol_values = []
        rr_values = []

        for e in setup_exits:
            d = decision_lookup.get(e.get('trade_id'), {})
            if d:
                indicators = d.get('plan', {}).get('indicators', {})
                quality = d.get('plan', {}).get('quality', {})
                if indicators.get('adx'):
                    adx_values.append(indicators['adx'])
                if indicators.get('rsi'):
                    rsi_values.append(indicators['rsi'])
                if d.get('bar5', {}).get('volume'):
                    vol_values.append(d['bar5']['volume'])
                if quality.get('structural_rr'):
                    rr_values.append(quality['structural_rr'])

        if adx_values:
            print(f"  ADX: avg={statistics.mean(adx_values):.1f}, range=[{min(adx_values):.1f}, {max(adx_values):.1f}]")
        if rsi_values:
            print(f"  RSI: avg={statistics.mean(rsi_values):.1f}, range=[{min(rsi_values):.1f}, {max(rsi_values):.1f}]")
        if vol_values:
            print(f"  Volume: avg={statistics.mean(vol_values)/1000:.0f}k")
        if rr_values:
            print(f"  Structural RR: avg={statistics.mean(rr_values):.2f}")

        # Regime analysis
        print("\nRegime breakdown:")
        by_regime = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
        for e in setup_exits:
            regime = e.get('regime', 'unknown')
            pnl = e.get('total_trade_pnl', 0)
            if pnl > 0:
                by_regime[regime]['wins'] += 1
            else:
                by_regime[regime]['losses'] += 1
            by_regime[regime]['pnl'] += pnl

        for regime, data in sorted(by_regime.items(), key=lambda x: -x[1]['pnl']):
            total = data['wins'] + data['losses']
            wr = data['wins'] / total * 100 if total > 0 else 0
            print(f"  {regime}: W={data['wins']}, L={data['losses']}, WR={wr:.0f}%, PnL=Rs {data['pnl']:.0f}")

        # Entry hour analysis
        print("\nEntry hour breakdown:")
        by_hour = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
        for e in setup_exits:
            d = decision_lookup.get(e.get('trade_id'), {})
            hour = d.get('ts', '')[11:13] if d and len(d.get('ts', '')) >= 13 else ''
            if not hour:
                continue
            pnl = e.get('total_trade_pnl', 0)
            if pnl > 0:
                by_hour[hour]['wins'] += 1
            else:
                by_hour[hour]['losses'] += 1
            by_hour[hour]['pnl'] += pnl

        for hour in sorted(by_hour.keys()):
            data = by_hour[hour]
            total = data['wins'] + data['losses']
            wr = data['wins'] / total * 100 if total > 0 else 0
            print(f"  Hour {hour}: W={data['wins']}, L={data['losses']}, WR={wr:.0f}%, PnL=Rs {data['pnl']:.0f}")

        # Exit reason analysis
        print("\nExit reason breakdown:")
        by_exit = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for e in setup_exits:
            reason = e.get('reason', 'unknown')
            # Simplify reason
            if 't1' in reason.lower():
                key = 't1_partial'
            elif 't2' in reason.lower():
                key = 'target_t2_full'
            elif 'hard_sl' in reason.lower():
                key = 'hard_sl'
            elif 'sl_post_t1' in reason.lower():
                key = 'sl_post_t1'
            elif 'eod' in reason.lower():
                key = 'eod_squareoff'
            else:
                key = reason

            by_exit[key]['count'] += 1
            by_exit[key]['pnl'] += e.get('total_trade_pnl', 0)

        for exit_reason, data in sorted(by_exit.items(), key=lambda x: -x[1]['count']):
            print(f"  {exit_reason}: {data['count']} trades, Rs {data['pnl']:.0f}")

def analyze_big_winners_characteristics(decisions, triggers, exits):
    """
    DEEP DIVE: What makes big winners (>500 Rs)?
    """
    print("\n" + "="*100)
    print("BIG WINNERS (>500 Rs) DEEP ANALYSIS")
    print("="*100)

    decision_lookup = {d.get('trade_id'): d for d in decisions}
    final_exits = [e for e in exits if e.get('is_final_exit', False)]

    big_winners = [e for e in final_exits if e.get('total_trade_pnl', 0) > 500]

    print(f"\nTotal big winners: {len(big_winners)}")
    total_pnl = sum(e.get('total_trade_pnl', 0) for e in big_winners)
    print(f"Total PnL from big winners: Rs {total_pnl:.0f}")
    print(f"Avg big winner: Rs {total_pnl/len(big_winners):.0f}")

    # Common characteristics
    print("\n" + "-"*80)
    print("COMMON CHARACTERISTICS OF BIG WINNERS")
    print("-"*80)

    adx_values = []
    rsi_values = []
    rr_values = []
    rank_scores = []
    slippage_values = []

    for e in big_winners:
        d = decision_lookup.get(e.get('trade_id'), {})
        if d:
            indicators = d.get('plan', {}).get('indicators', {})
            quality = d.get('plan', {}).get('quality', {})
            ranking = d.get('plan', {}).get('ranking', {})

            if indicators.get('adx'):
                adx_values.append(indicators['adx'])
            if indicators.get('rsi'):
                rsi_values.append(indicators['rsi'])
            if quality.get('structural_rr'):
                rr_values.append(quality['structural_rr'])
            if ranking.get('score'):
                rank_scores.append(ranking['score'])

        if e.get('slippage_bps'):
            slippage_values.append(e['slippage_bps'])

    print(f"\nADX: avg={statistics.mean(adx_values):.1f}, median={statistics.median(adx_values):.1f}")
    print(f"RSI: avg={statistics.mean(rsi_values):.1f}, median={statistics.median(rsi_values):.1f}")
    print(f"Structural RR: avg={statistics.mean(rr_values):.2f}, median={statistics.median(rr_values):.2f}")
    print(f"Rank Score: avg={statistics.mean(rank_scores):.2f}, median={statistics.median(rank_scores):.2f}")
    if slippage_values:
        print(f"Slippage: avg={statistics.mean(slippage_values):.0f} bps")

    # Exit reason for big winners
    print("\nExit reasons for big winners:")
    by_exit = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for e in big_winners:
        reason = e.get('reason', '')
        if 't2' in reason.lower():
            key = 'T2 full exit'
        elif 'eod' in reason.lower():
            key = 'EOD exit'
        elif 'sl_post_t1' in reason.lower():
            key = 'SL post T1'
        else:
            key = reason
        by_exit[key]['count'] += 1
        by_exit[key]['pnl'] += e.get('total_trade_pnl', 0)

    for exit_reason, data in sorted(by_exit.items(), key=lambda x: -x[1]['count']):
        print(f"  {exit_reason}: {data['count']} trades, Rs {data['pnl']:.0f}")

    # Regime for big winners
    print("\nRegime for big winners:")
    by_regime = defaultdict(int)
    for e in big_winners:
        by_regime[e.get('regime', 'unknown')] += 1
    for regime, count in sorted(by_regime.items(), key=lambda x: -x[1]):
        print(f"  {regime}: {count}")

    # Entry hour for big winners
    print("\nEntry hour for big winners:")
    by_hour = defaultdict(int)
    for e in big_winners:
        d = decision_lookup.get(e.get('trade_id'), {})
        hour = d.get('ts', '')[11:13] if d and len(d.get('ts', '')) >= 13 else ''
        if hour:
            by_hour[hour] += 1
    for hour in sorted(by_hour.keys()):
        print(f"  Hour {hour}: {by_hour[hour]}")

def main():
    print("Loading all backtest data (including all exits)...")
    decisions, triggers, exits = load_all_data()
    print(f"Loaded {len(decisions)} decisions, {len(triggers)} triggers, {len(exits)} exits")

    analyze_hard_sl_deep(decisions, triggers, exits)
    analyze_exit_management(decisions, triggers, exits)
    analyze_high_probability_setups(decisions, triggers, exits)
    analyze_big_winners_characteristics(decisions, triggers, exits)

if __name__ == "__main__":
    main()
