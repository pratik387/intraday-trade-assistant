"""
Final comprehensive filter analysis - find all viable filters
"""
import json
from pathlib import Path
from collections import defaultdict

BACKTEST_DIR = Path("backtest_20251205-102858_extracted/20251205-102858_full/20251205-102858")

def load_all_data():
    """Load all events and analytics data"""
    all_decisions = []
    all_exits = []

    sessions = [d for d in BACKTEST_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()]

    for session_dir in sessions:
        events_file = session_dir / "events.jsonl"
        analytics_file = session_dir / "analytics.jsonl"

        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('type') == 'DECISION':
                            all_decisions.append(event)
                    except:
                        pass

        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get('stage') == 'EXIT' and event.get('is_final_exit', False):
                            all_exits.append(event)
                    except:
                        pass

    return all_decisions, all_exits

def main():
    decisions, exits = load_all_data()
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
            'regime': e.get('regime', ''),
            'entry_hour': d.get('ts', '')[11:13] if d and len(d.get('ts', '')) >= 13 else '',
            'structural_rr': d.get('plan', {}).get('quality', {}).get('structural_rr') if d else None,
            'exit_reason': e.get('reason', ''),
            'slippage_bps': e.get('slippage_bps', 0)
        })

    current_pnl = sum(t['pnl'] for t in all_trades)
    print(f"Current Total PnL: Rs {current_pnl:.2f}")
    print(f"Total Trades: {len(all_trades)}")
    print(f"Target: Rs 50,000")
    print(f"Gap: Rs {50000 - current_pnl:.2f}")

    # Comprehensive filter testing
    filters = [
        # vwap_lose_short filters
        ("F1: vwap_lose_short entirely", lambda t: t['setup'] == 'vwap_lose_short'),
        ("F2: vwap_lose_short in trend_up", lambda t: t['setup'] == 'vwap_lose_short' and t['regime'] == 'trend_up'),
        ("F3: vwap_lose_short with ADX < 30", lambda t: t['setup'] == 'vwap_lose_short' and (t['adx'] or 100) < 30),
        ("F4: vwap_lose_short in chop", lambda t: t['setup'] == 'vwap_lose_short' and t['regime'] == 'chop'),

        # resistance_bounce_short filters
        ("F5: resistance_bounce_short at hour 13", lambda t: t['setup'] == 'resistance_bounce_short' and t['entry_hour'] == '13'),
        ("F6: resistance_bounce_short in trend_up", lambda t: t['setup'] == 'resistance_bounce_short' and t['regime'] == 'trend_up'),
        ("F7: resistance_bounce_short in chop", lambda t: t['setup'] == 'resistance_bounce_short' and t['regime'] == 'chop'),

        # premium_zone_short filters
        ("F8: premium_zone_short with structural_rr < 1.5", lambda t: t['setup'] == 'premium_zone_short' and (t['structural_rr'] or 10) < 1.5),
        ("F9: premium_zone_short in trend_down", lambda t: t['setup'] == 'premium_zone_short' and t['regime'] == 'trend_down'),
        ("F10: premium_zone_short with ADX > 45", lambda t: t['setup'] == 'premium_zone_short' and (t['adx'] or 0) > 45),

        # Other filters
        ("F11: discount_zone_long entirely", lambda t: t['setup'] == 'discount_zone_long'),
        ("F12: range_bounce_short in chop", lambda t: t['setup'] == 'range_bounce_short' and t['regime'] == 'chop'),

        # General ADX filters
        ("F13: ANY short setup with ADX < 18", lambda t: t['setup'] and 'short' in t['setup'] and (t['adx'] or 100) < 18),
        ("F14: ANY short setup with ADX < 20", lambda t: t['setup'] and 'short' in t['setup'] and (t['adx'] or 100) < 20),
    ]

    print(f"\n{'='*100}")
    print("ALL FILTER OPPORTUNITIES (sorted by improvement)")
    print(f"{'='*100}")
    print(f"{'Filter':<55} {'Blocked':>8} {'PnL Blocked':>12} {'New Total':>12} {'Delta':>12}")
    print("-" * 100)

    results = []
    for name, filter_fn in filters:
        try:
            blocked = [t for t in all_trades if filter_fn(t)]
            blocked_pnl = sum(t['pnl'] for t in blocked)
            remaining_pnl = current_pnl - blocked_pnl
            improvement = -blocked_pnl

            results.append((name, len(blocked), blocked_pnl, remaining_pnl, improvement))
        except Exception as ex:
            print(f"{name:<55} ERROR: {ex}")

    # Sort by improvement (descending)
    results.sort(key=lambda x: -x[4])

    for name, count, blocked_pnl, remaining_pnl, improvement in results:
        marker = " ***" if improvement > 500 else ""
        print(f"{name:<55} {count:>8} {blocked_pnl:>12.0f} {remaining_pnl:>12.0f} {improvement:>+12.0f}{marker}")

    print(f"\n{'='*100}")
    print("OPTIMAL COMBINED FILTER SIMULATION")
    print(f"{'='*100}")

    # Apply only positive-impact filters sequentially (avoid overlap)
    optimal_filters = [
        ("Block vwap_lose_short entirely", lambda t: t['setup'] == 'vwap_lose_short'),
        ("Block resistance_bounce_short at hour 13", lambda t: t['setup'] == 'resistance_bounce_short' and t['entry_hour'] == '13'),
        ("Block discount_zone_long entirely", lambda t: t['setup'] == 'discount_zone_long'),
        ("Block premium_zone_short with structural_rr < 1.5", lambda t: t['setup'] == 'premium_zone_short' and (t['structural_rr'] or 10) < 1.5),
        ("Block resistance_bounce_short in chop", lambda t: t['setup'] == 'resistance_bounce_short' and t['regime'] == 'chop'),
        ("Block resistance_bounce_short in trend_up", lambda t: t['setup'] == 'resistance_bounce_short' and t['regime'] == 'trend_up'),
    ]

    remaining = all_trades.copy()
    print(f"\nStarting: {len(remaining)} trades, Rs {sum(t['pnl'] for t in remaining):.0f}")

    for name, filter_fn in optimal_filters:
        blocked = [t for t in remaining if filter_fn(t)]
        blocked_pnl = sum(t['pnl'] for t in blocked)

        # Only apply if it improves PnL
        if blocked_pnl < 0:
            remaining = [t for t in remaining if not filter_fn(t)]
            new_pnl = sum(t['pnl'] for t in remaining)
            print(f"\n✓ {name}")
            print(f"  Blocked: {len(blocked)} trades worth Rs {blocked_pnl:.0f}")
            print(f"  Remaining: {len(remaining)} trades, Rs {new_pnl:.0f}")
        else:
            print(f"\n✗ {name} - SKIPPED (blocks Rs {blocked_pnl:.0f} in profits)")

    final_pnl = sum(t['pnl'] for t in remaining)
    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"  Original: {len(all_trades)} trades, Rs {current_pnl:.0f}")
    print(f"  After filters: {len(remaining)} trades, Rs {final_pnl:.0f}")
    print(f"  Improvement: Rs {final_pnl - current_pnl:.0f}")
    print(f"  Still need: Rs {50000 - final_pnl:.0f} to reach target")

    # Summary of filters to implement
    print(f"\n{'='*60}")
    print("FILTERS TO IMPLEMENT:")
    print(f"{'='*60}")
    print("""
1. vwap_lose_short: BLOCK ENTIRELY
   - Evidence: 16 trades, -3,282 Rs
   - Why: 25% WR, all winners have ADX > 37, most losers have ADX < 30
   - 0% WR in trend_up (4 trades, all losses)

2. resistance_bounce_short at hour 13: BLOCK
   - Evidence: 13 trades, -1,966 Rs
   - Why: 15% WR (2W, 11L) - terrible timing

3. discount_zone_long: BLOCK ENTIRELY
   - Evidence: 1 trade, -500 Rs
   - Why: 0% WR

4. premium_zone_short with structural_rr < 1.5: BLOCK
   - Evidence: 32 trades, -1,658 Rs
   - Why: Low structural RR means unfavorable risk/reward

5. resistance_bounce_short in chop: BLOCK (if profitable)
   - Evidence: 3 trades, -422 Rs
   - Why: 33% WR in chop regime

6. resistance_bounce_short in trend_up: CONSIDER BLOCKING
   - Evidence: 14 trades, -882 Rs
   - Why: 43% WR - counter-trend in uptrend
""")

if __name__ == "__main__":
    main()
