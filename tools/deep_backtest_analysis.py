"""
Deep backtest analysis - comprehensive examination of all trades
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

BACKTEST_DIR = Path("backtest_20251205-102858_extracted/20251205-102858_full/20251205-102858")

def load_all_data():
    """Load all events and analytics data from all sessions"""
    all_decisions = []
    all_exits = []

    # Sessions are organized by date (2023-12-01, 2024-01-02, etc.)
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
                    except:
                        pass

        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('stage') == 'EXIT' and event.get('is_final_exit', False):
                            event['_session'] = session_dir.name
                            all_exits.append(event)
                    except:
                        pass

    return all_decisions, all_exits

def analyze_winners_vs_losers(decisions, exits):
    """Compare indicator values between winners and losers by setup"""
    print("\n" + "="*80)
    print("WINNERS VS LOSERS - DETAILED INDICATOR ANALYSIS")
    print("="*80)

    # Create decision lookup by trade_id
    decision_lookup = {}
    for d in decisions:
        trade_id = d.get('trade_id')
        if trade_id:
            decision_lookup[trade_id] = d

    # Group by setup using analytics data
    setup_trades = defaultdict(lambda: {'winners': [], 'losers': []})

    for e in exits:
        trade_id = e.get('trade_id')
        pnl = e.get('total_trade_pnl', e.get('pnl', 0))
        setup = e.get('setup_type', 'unknown')
        regime = e.get('regime', 'unknown')

        # Get indicators from decision if available
        indicators = {}
        entry_hour = None
        structural_rr = None
        rank_score = None
        volume = None
        slippage_bps = e.get('slippage_bps', 0)

        if trade_id in decision_lookup:
            d = decision_lookup[trade_id]
            indicators = d.get('plan', {}).get('indicators', {})
            structural_rr = d.get('plan', {}).get('quality', {}).get('structural_rr')
            rank_score = d.get('plan', {}).get('ranking', {}).get('score')
            volume = d.get('bar5', {}).get('volume')
            ts = d.get('ts', '')
            if ts and len(ts) >= 13:
                entry_hour = ts[11:13]

        entry_data = {
            'trade_id': trade_id,
            'pnl': pnl,
            'adx': indicators.get('adx'),
            'rsi': indicators.get('rsi'),
            'volume': volume,
            'volume_ratio': indicators.get('volume_ratio'),
            'atr_pct': indicators.get('atr_pct'),
            'regime': regime,
            'entry_hour': entry_hour,
            'structural_rr': structural_rr,
            'rank_score': rank_score,
            'exit_reason': e.get('reason', 'unknown'),
            'slippage_bps': slippage_bps
        }

        if pnl > 0:
            setup_trades[setup]['winners'].append(entry_data)
        else:
            setup_trades[setup]['losers'].append(entry_data)

    # Analyze each setup
    for setup in sorted(setup_trades.keys()):
        winners = setup_trades[setup]['winners']
        losers = setup_trades[setup]['losers']

        if len(winners) < 2 or len(losers) < 2:
            continue

        print(f"\n{'='*60}")
        print(f"SETUP: {setup}")
        print(f"Winners: {len(winners)}, Losers: {len(losers)}, WR: {len(winners)/(len(winners)+len(losers))*100:.1f}%")
        print(f"Total W PnL: Rs {sum(t['pnl'] for t in winners):.0f}, Total L PnL: Rs {sum(t['pnl'] for t in losers):.0f}")
        print(f"{'='*60}")

        # ADX Analysis
        w_adx = [t['adx'] for t in winners if t['adx'] is not None]
        l_adx = [t['adx'] for t in losers if t['adx'] is not None]
        if w_adx and l_adx:
            print(f"\nADX: Winners avg={statistics.mean(w_adx):.1f}, Losers avg={statistics.mean(l_adx):.1f}")
            print(f"     Winners median={statistics.median(w_adx):.1f}, Losers median={statistics.median(l_adx):.1f}")

        # RSI Analysis
        w_rsi = [t['rsi'] for t in winners if t['rsi'] is not None]
        l_rsi = [t['rsi'] for t in losers if t['rsi'] is not None]
        if w_rsi and l_rsi:
            print(f"\nRSI: Winners avg={statistics.mean(w_rsi):.1f}, Losers avg={statistics.mean(l_rsi):.1f}")

        # Structural RR Analysis
        w_rr = [t['structural_rr'] for t in winners if t['structural_rr'] is not None]
        l_rr = [t['structural_rr'] for t in losers if t['structural_rr'] is not None]
        if w_rr and l_rr:
            print(f"\nStructural RR: Winners avg={statistics.mean(w_rr):.2f}, Losers avg={statistics.mean(l_rr):.2f}")

        # Rank Score Analysis
        w_rank = [t['rank_score'] for t in winners if t['rank_score'] is not None]
        l_rank = [t['rank_score'] for t in losers if t['rank_score'] is not None]
        if w_rank and l_rank:
            print(f"\nRank Score: Winners avg={statistics.mean(w_rank):.2f}, Losers avg={statistics.mean(l_rank):.2f}")

        # Slippage Analysis
        w_slip = [t['slippage_bps'] for t in winners if t['slippage_bps']]
        l_slip = [t['slippage_bps'] for t in losers if t['slippage_bps']]
        if w_slip and l_slip:
            print(f"\nSlippage BPS: Winners avg={statistics.mean(w_slip):.0f}, Losers avg={statistics.mean(l_slip):.0f}")

        # Regime Analysis
        print("\nRegime breakdown:")
        for regime in ['trend_up', 'trend_down', 'chop', 'squeeze']:
            w_count = len([t for t in winners if t['regime'] == regime])
            l_count = len([t for t in losers if t['regime'] == regime])
            total = w_count + l_count
            if total > 0:
                wr = w_count / total * 100
                w_pnl = sum(t['pnl'] for t in winners if t['regime'] == regime)
                l_pnl = sum(t['pnl'] for t in losers if t['regime'] == regime)
                print(f"  {regime}: W={w_count}, L={l_count}, WR={wr:.0f}%, PnL={w_pnl+l_pnl:.0f}")

        # Entry Hour Analysis
        print("\nEntry hour breakdown:")
        for hour in ['09', '10', '11', '12', '13', '14', '15']:
            w_count = len([t for t in winners if t['entry_hour'] == hour])
            l_count = len([t for t in losers if t['entry_hour'] == hour])
            total = w_count + l_count
            if total > 0:
                wr = w_count / total * 100
                w_pnl = sum(t['pnl'] for t in winners if t['entry_hour'] == hour)
                l_pnl = sum(t['pnl'] for t in losers if t['entry_hour'] == hour)
                print(f"  Hour {hour}: W={w_count}, L={l_count}, WR={wr:.0f}%, PnL={w_pnl+l_pnl:.0f}")

        # Exit reason analysis
        print("\nExit reasons (Losers only):")
        exit_counts = defaultdict(int)
        for t in losers:
            exit_counts[t['exit_reason']] += 1
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {reason}: {count}")

def analyze_hard_sl_characteristics(decisions, exits):
    """Deep dive into hard_sl exits"""
    print("\n" + "="*80)
    print("HARD_SL DEEP ANALYSIS - What's causing stop losses?")
    print("="*80)

    # Create decision lookup
    decision_lookup = {d.get('trade_id'): d for d in decisions if d.get('trade_id')}

    hard_sl_trades = []
    for e in exits:
        if e.get('reason') != 'hard_sl':
            continue

        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})
        indicators = d.get('plan', {}).get('indicators', {}) if d else {}

        hard_sl_trades.append({
            'trade_id': trade_id,
            'setup': e.get('setup_type'),
            'pnl': e.get('total_trade_pnl', e.get('pnl', 0)),
            'adx': indicators.get('adx'),
            'rsi': indicators.get('rsi'),
            'regime': e.get('regime'),
            'entry_hour': d.get('ts', '')[11:13] if d and len(d.get('ts', '')) >= 13 else None,
            'structural_rr': d.get('plan', {}).get('quality', {}).get('structural_rr') if d else None,
            'rank_score': d.get('plan', {}).get('ranking', {}).get('score') if d else None,
            'symbol': e.get('symbol')
        })

    if not hard_sl_trades:
        print("No hard_sl trades found")
        return

    print(f"\nTotal hard_sl exits: {len(hard_sl_trades)}")
    total_loss = sum(t['pnl'] for t in hard_sl_trades)
    print(f"Total loss from hard_sl: Rs {total_loss:.2f}")
    print(f"Avg loss per hard_sl: Rs {total_loss/len(hard_sl_trades):.2f}")

    # By Setup
    print("\nHard SL by Setup:")
    by_setup = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for t in hard_sl_trades:
        by_setup[t['setup']]['count'] += 1
        by_setup[t['setup']]['pnl'] += t['pnl']

    for setup, data in sorted(by_setup.items(), key=lambda x: x[1]['pnl']):
        print(f"  {setup}: {data['count']} trades, Rs {data['pnl']:.0f}")

    # By Regime
    print("\nHard SL by Regime:")
    by_regime = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for t in hard_sl_trades:
        by_regime[t['regime']]['count'] += 1
        by_regime[t['regime']]['pnl'] += t['pnl']

    for regime, data in sorted(by_regime.items(), key=lambda x: x[1]['pnl']):
        print(f"  {regime}: {data['count']} trades, Rs {data['pnl']:.0f}")

    # By Entry Hour
    print("\nHard SL by Entry Hour:")
    by_hour = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for t in hard_sl_trades:
        if t['entry_hour']:
            by_hour[t['entry_hour']]['count'] += 1
            by_hour[t['entry_hour']]['pnl'] += t['pnl']

    for hour, data in sorted(by_hour.items()):
        print(f"  Hour {hour}: {data['count']} trades, Rs {data['pnl']:.0f}")

    # ADX Analysis
    adx_values = [t['adx'] for t in hard_sl_trades if t['adx'] is not None]
    if adx_values:
        print(f"\nADX for hard_sl trades:")
        print(f"  Avg: {statistics.mean(adx_values):.1f}")
        print(f"  < 20: {len([x for x in adx_values if x < 20])}")
        print(f"  20-25: {len([x for x in adx_values if 20 <= x < 25])}")
        print(f"  25-30: {len([x for x in adx_values if 25 <= x < 30])}")
        print(f"  >= 30: {len([x for x in adx_values if x >= 30])}")

def analyze_regime_setup_matrix(decisions, exits):
    """Setup x Regime matrix analysis"""
    print("\n" + "="*80)
    print("REGIME x SETUP MATRIX - Which setups work in which regimes?")
    print("="*80)

    # Use exits directly since they have setup_type and regime
    matrix = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0}))

    for e in exits:
        pnl = e.get('total_trade_pnl', e.get('pnl', 0))
        setup = e.get('setup_type', 'unknown')
        regime = e.get('regime', 'unknown')

        if pnl > 0:
            matrix[setup][regime]['wins'] += 1
        else:
            matrix[setup][regime]['losses'] += 1
        matrix[setup][regime]['pnl'] += pnl

    # Print matrix
    regimes = ['trend_up', 'trend_down', 'chop', 'squeeze']

    for setup in sorted(matrix.keys()):
        total_pnl = sum(matrix[setup][r]['pnl'] for r in regimes)
        total_trades = sum(matrix[setup][r]['wins'] + matrix[setup][r]['losses'] for r in regimes)
        print(f"\n{setup} (Total: {total_trades} trades, Rs {total_pnl:.0f}):")
        print(f"  {'Regime':<12} {'Wins':>6} {'Losses':>6} {'WR%':>6} {'PnL':>10}")
        print(f"  {'-'*44}")

        for regime in regimes:
            data = matrix[setup][regime]
            total = data['wins'] + data['losses']
            if total > 0:
                wr = data['wins'] / total * 100
                marker = " <-- AVOID" if wr < 40 and total >= 3 else ""
                print(f"  {regime:<12} {data['wins']:>6} {data['losses']:>6} {wr:>5.0f}% {data['pnl']:>10.0f}{marker}")

def analyze_entry_timing_granular(decisions, exits):
    """Granular entry timing analysis by setup"""
    print("\n" + "="*80)
    print("ENTRY TIMING ANALYSIS - When do setups work best?")
    print("="*80)

    # Create decision lookup
    decision_lookup = {d.get('trade_id'): d for d in decisions if d.get('trade_id')}

    # Setup x Hour matrix
    setup_hour_data = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0}))

    for e in exits:
        trade_id = e.get('trade_id')
        pnl = e.get('total_trade_pnl', e.get('pnl', 0))
        setup = e.get('setup_type', 'unknown')

        d = decision_lookup.get(trade_id, {})
        entry_hour = d.get('ts', '')[11:13] if d and len(d.get('ts', '')) >= 13 else None
        if not entry_hour:
            continue

        if pnl > 0:
            setup_hour_data[setup][entry_hour]['wins'] += 1
        else:
            setup_hour_data[setup][entry_hour]['losses'] += 1
        setup_hour_data[setup][entry_hour]['pnl'] += pnl

    # Print matrix for key setups
    for setup in sorted(setup_hour_data.keys()):
        total_pnl = sum(setup_hour_data[setup][h]['pnl'] for h in setup_hour_data[setup])
        if abs(total_pnl) < 500:  # Skip setups with minimal PnL
            continue

        print(f"\n{setup}:")
        print(f"  {'Hour':<6} {'Wins':>6} {'Losses':>6} {'WR%':>6} {'PnL':>10}")
        print(f"  {'-'*38}")

        for hour in ['09', '10', '11', '12', '13', '14', '15']:
            data = setup_hour_data[setup][hour]
            total = data['wins'] + data['losses']
            if total > 0:
                wr = data['wins'] / total * 100
                marker = " <-- AVOID" if wr < 40 and total >= 3 else ""
                print(f"  {hour:6} {data['wins']:>6} {data['losses']:>6} {wr:>5.0f}% {data['pnl']:>10.0f}{marker}")

def analyze_filter_opportunities(decisions, exits):
    """Identify specific filter opportunities"""
    print("\n" + "="*80)
    print("FILTER OPPORTUNITY ANALYSIS")
    print("="*80)

    # Create decision lookup
    decision_lookup = {d.get('trade_id'): d for d in decisions if d.get('trade_id')}

    all_trades = []
    for e in exits:
        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})
        indicators = d.get('plan', {}).get('indicators', {}) if d else {}

        all_trades.append({
            'trade_id': trade_id,
            'setup': e.get('setup_type', ''),
            'pnl': e.get('total_trade_pnl', e.get('pnl', 0)),
            'adx': indicators.get('adx'),
            'rsi': indicators.get('rsi'),
            'volume': d.get('bar5', {}).get('volume') if d else None,
            'regime': e.get('regime', ''),
            'entry_hour': d.get('ts', '')[11:13] if d and len(d.get('ts', '')) >= 13 else '',
            'structural_rr': d.get('plan', {}).get('quality', {}).get('structural_rr') if d else None,
            'exit_reason': e.get('reason', ''),
            'rank_score': d.get('plan', {}).get('ranking', {}).get('score') if d else None,
            'slippage_bps': e.get('slippage_bps', 0)
        })

    current_pnl = sum(t['pnl'] for t in all_trades)
    print(f"Current Total PnL: Rs {current_pnl:.2f}")
    print(f"Total Trades: {len(all_trades)}")
    print(f"Target: Rs 50,000")
    print(f"Gap: Rs {50000 - current_pnl:.2f}")

    # Test various filters
    filters = [
        ("vwap_lose_short in trend_up", lambda t: t['setup'] == 'vwap_lose_short' and t['regime'] == 'trend_up'),
        ("vwap_lose_short with ADX < 30", lambda t: t['setup'] == 'vwap_lose_short' and (t['adx'] or 100) < 30),
        ("vwap_lose_short entirely", lambda t: t['setup'] == 'vwap_lose_short'),
        ("discount_zone_long entirely", lambda t: t['setup'] == 'discount_zone_long'),
        ("resistance_bounce_short at hour 11", lambda t: t['setup'] == 'resistance_bounce_short' and t['entry_hour'] == '11'),
        ("resistance_bounce_short at hour 13", lambda t: t['setup'] == 'resistance_bounce_short' and t['entry_hour'] == '13'),
        ("premium_zone_short at hour 11", lambda t: t['setup'] == 'premium_zone_short' and t['entry_hour'] == '11'),
        ("premium_zone_short in trend_up", lambda t: t['setup'] == 'premium_zone_short' and t['regime'] == 'trend_up'),
        ("orb_breakout_long at hour 11", lambda t: t['setup'] == 'orb_breakout_long' and t['entry_hour'] == '11'),
        ("range_bounce_short in chop", lambda t: t['setup'] == 'range_bounce_short' and t['regime'] == 'chop'),
        ("any setup with ADX < 15", lambda t: (t['adx'] or 100) < 15),
        ("any SHORT with ADX < 20", lambda t: t['setup'] and 'short' in t['setup'] and (t['adx'] or 100) < 20),
        ("any setup with slippage > 200 bps", lambda t: (t['slippage_bps'] or 0) > 200),
        ("hard_sl trades with structural_rr < 1.5", lambda t: t['exit_reason'] == 'hard_sl' and (t['structural_rr'] or 100) < 1.5),
    ]

    print(f"\n{'Filter':<50} {'Blocked':>8} {'PnL Blocked':>12} {'New Total':>12} {'Delta':>12}")
    print("-" * 100)

    for name, filter_fn in filters:
        try:
            blocked = [t for t in all_trades if filter_fn(t)]
            blocked_pnl = sum(t['pnl'] for t in blocked)
            remaining_pnl = current_pnl - blocked_pnl
            improvement = -blocked_pnl  # Blocking losses improves PnL

            print(f"{name:<50} {len(blocked):>8} {blocked_pnl:>12.0f} {remaining_pnl:>12.0f} {improvement:>+12.0f}")
        except Exception as ex:
            print(f"{name:<50} ERROR: {ex}")

def simulate_combined_filters(decisions, exits):
    """Simulate combined filter application"""
    print("\n" + "="*80)
    print("COMBINED FILTER SIMULATION")
    print("="*80)

    # Create decision lookup
    decision_lookup = {d.get('trade_id'): d for d in decisions if d.get('trade_id')}

    all_trades = []
    for e in exits:
        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})
        indicators = d.get('plan', {}).get('indicators', {}) if d else {}

        all_trades.append({
            'trade_id': trade_id,
            'setup': e.get('setup_type', ''),
            'pnl': e.get('total_trade_pnl', e.get('pnl', 0)),
            'adx': indicators.get('adx'),
            'regime': e.get('regime', ''),
            'entry_hour': d.get('ts', '')[11:13] if d and len(d.get('ts', '')) >= 13 else '',
            'structural_rr': d.get('plan', {}).get('quality', {}).get('structural_rr') if d else None,
        })

    current_pnl = sum(t['pnl'] for t in all_trades)
    current_trades = len(all_trades)

    # Define winning filter combinations based on analysis
    filters_to_apply = [
        ("Block vwap_lose_short in trend_up", lambda t: t['setup'] == 'vwap_lose_short' and t['regime'] == 'trend_up'),
        ("Block vwap_lose_short with ADX < 30", lambda t: t['setup'] == 'vwap_lose_short' and (t['adx'] or 100) < 30),
        ("Block discount_zone_long", lambda t: t['setup'] == 'discount_zone_long'),
    ]

    remaining = all_trades.copy()
    print(f"\nStarting: {len(remaining)} trades, Rs {current_pnl:.0f}")

    for name, filter_fn in filters_to_apply:
        blocked = [t for t in remaining if filter_fn(t)]
        remaining = [t for t in remaining if not filter_fn(t)]
        blocked_pnl = sum(t['pnl'] for t in blocked)
        new_pnl = sum(t['pnl'] for t in remaining)
        print(f"\nAfter: {name}")
        print(f"  Blocked: {len(blocked)} trades worth Rs {blocked_pnl:.0f}")
        print(f"  Remaining: {len(remaining)} trades, Rs {new_pnl:.0f}")

    final_pnl = sum(t['pnl'] for t in remaining)
    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"  Original: {current_trades} trades, Rs {current_pnl:.0f}")
    print(f"  After filters: {len(remaining)} trades, Rs {final_pnl:.0f}")
    print(f"  Improvement: Rs {final_pnl - current_pnl:.0f}")
    print(f"  Still need: Rs {50000 - final_pnl:.0f} to reach target")

def analyze_big_winners(decisions, exits):
    """Analyze characteristics of big winners (>500 Rs)"""
    print("\n" + "="*80)
    print("BIG WINNERS ANALYSIS (>500 Rs) - What makes trades succeed big?")
    print("="*80)

    decision_lookup = {d.get('trade_id'): d for d in decisions if d.get('trade_id')}

    big_winners = []
    for e in exits:
        pnl = e.get('total_trade_pnl', e.get('pnl', 0))
        if pnl <= 500:
            continue

        trade_id = e.get('trade_id')
        d = decision_lookup.get(trade_id, {})
        indicators = d.get('plan', {}).get('indicators', {}) if d else {}

        big_winners.append({
            'trade_id': trade_id,
            'setup': e.get('setup_type'),
            'pnl': pnl,
            'adx': indicators.get('adx'),
            'rsi': indicators.get('rsi'),
            'regime': e.get('regime'),
            'entry_hour': d.get('ts', '')[11:13] if d and len(d.get('ts', '')) >= 13 else None,
            'structural_rr': d.get('plan', {}).get('quality', {}).get('structural_rr') if d else None,
            'rank_score': d.get('plan', {}).get('ranking', {}).get('score') if d else None,
            'symbol': e.get('symbol')
        })

    print(f"\nTotal big winners: {len(big_winners)}")
    total_big_win_pnl = sum(t['pnl'] for t in big_winners)
    print(f"Total PnL from big winners: Rs {total_big_win_pnl:.2f}")

    # By Setup
    print("\nBig Winners by Setup:")
    by_setup = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for t in big_winners:
        by_setup[t['setup']]['count'] += 1
        by_setup[t['setup']]['pnl'] += t['pnl']

    for setup, data in sorted(by_setup.items(), key=lambda x: -x[1]['count']):
        print(f"  {setup}: {data['count']} trades, Rs {data['pnl']:.0f}")

    # Common characteristics
    if big_winners:
        print("\nCommon characteristics of big winners:")
        adx_values = [t['adx'] for t in big_winners if t['adx'] is not None]
        if adx_values:
            print(f"  ADX: avg={statistics.mean(adx_values):.1f}, median={statistics.median(adx_values):.1f}")

        rr_values = [t['structural_rr'] for t in big_winners if t['structural_rr'] is not None]
        if rr_values:
            print(f"  Structural RR: avg={statistics.mean(rr_values):.2f}, median={statistics.median(rr_values):.2f}")

        # By Regime
        print("\nBig Winners by Regime:")
        by_regime = defaultdict(int)
        for t in big_winners:
            by_regime[t['regime']] += 1
        for regime, count in sorted(by_regime.items(), key=lambda x: -x[1]):
            print(f"  {regime}: {count}")

        # By Entry Hour
        print("\nBig Winners by Entry Hour:")
        by_hour = defaultdict(int)
        for t in big_winners:
            if t['entry_hour']:
                by_hour[t['entry_hour']] += 1
        for hour, count in sorted(by_hour.items()):
            print(f"  Hour {hour}: {count}")

def main():
    print("Loading all backtest data...")
    decisions, exits = load_all_data()
    print(f"Loaded {len(decisions)} decisions, {len(exits)} final exits")

    total_pnl = sum(e.get('total_trade_pnl', e.get('pnl', 0)) for e in exits)
    print(f"Total PnL from all exits: Rs {total_pnl:.2f}")

    # Run all analyses
    analyze_winners_vs_losers(decisions, exits)
    analyze_hard_sl_characteristics(decisions, exits)
    analyze_regime_setup_matrix(decisions, exits)
    analyze_entry_timing_granular(decisions, exits)
    analyze_big_winners(decisions, exits)
    analyze_filter_opportunities(decisions, exits)
    simulate_combined_filters(decisions, exits)

if __name__ == "__main__":
    main()
