#!/usr/bin/env python
"""
Analyze complete filtering funnel using ALL stage logs.

Stage logs available:
1. scanning.jsonl - Scanner stage (momentum/volume scoring)
2. screening.jsonl - Screener stage (ORB level computation)
3. ranking.jsonl - Ranking stage (rank_score calculation)
4. events.jsonl - Decision gate stage (trade plans)

Flow:
nse_all.json (1,992) → scanning.jsonl → screening.jsonl → ranking.jsonl → events.jsonl

Find: Where exactly do big movers get filtered out?
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_complete_funnel():
    """Trace big movers through ALL log stages."""

    base_path = Path("backtest_20251110-132748_extracted/20251110-132748_full/20251110-132748/2023-12-22")

    # Big movers from previous analysis
    big_movers_raw = ['AAVAS.NS', 'ABSLAMC.NS', 'ADANIENSOL.NS', 'AIAENG.NS',
                      'ALKYLAMINE.NS', 'ANGELONE.NS', 'ASAHIINDIA.NS',
                      'ASHIANA.NS', 'AUTOAXLES.NS', 'BASF.NS']

    big_movers = [f"NSE:{s.replace('.NS', '')}" for s in big_movers_raw]

    print("="*80)
    print("COMPLETE FILTERING FUNNEL ANALYSIS")
    print("="*80)
    print(f"\nTarget Date: 2023-12-22")
    print(f"Big Movers to Track: {len(big_movers)}")
    print()

    # Track each big mover through all stages
    mover_status = {}
    for mover in big_movers:
        mover_status[mover] = {
            'scanned_accept': False,
            'scanned_reject': False,
            'screened': False,
            'ranked': False,
            'event': False,
            'scan_details': [],
            'screen_details': [],
            'rank_details': [],
            'event_details': []
        }

    # ===== STAGE 1: SCANNING.JSONL =====
    print("="*80)
    print("STAGE 1: SCANNING (scanning.jsonl)")
    print("="*80)

    scanning_file = base_path / "scanning.jsonl"
    scan_accept_count = 0
    scan_reject_count = 0
    all_scanned_symbols = set()

    if scanning_file.exists():
        with open(scanning_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        symbol = entry.get("symbol")
                        action = entry.get("action")

                        all_scanned_symbols.add(symbol)

                        if action == "accept":
                            scan_accept_count += 1
                        elif action == "reject":
                            scan_reject_count += 1

                        # Check if it's a big mover
                        if symbol in big_movers:
                            if action == "accept":
                                mover_status[symbol]['scanned_accept'] = True
                            elif action == "reject":
                                mover_status[symbol]['scanned_reject'] = True

                            mover_status[symbol]['scan_details'].append({
                                'action': action,
                                'timestamp': entry.get('timestamp'),
                                'bias': entry.get('bias'),
                                'category': entry.get('category'),
                                'score_long': entry.get('score_long'),
                                'score_short': entry.get('score_short'),
                                'rank_long': entry.get('rank_long'),
                                'rank_short': entry.get('rank_short')
                            })
                    except:
                        pass

        print(f"Total scanning entries: {scan_accept_count + scan_reject_count}")
        print(f"  Accept: {scan_accept_count}")
        print(f"  Reject: {scan_reject_count}")
        print(f"Unique symbols scanned: {len(all_scanned_symbols)}")
    else:
        print(f"ERROR: {scanning_file} not found")

    # ===== STAGE 2: SCREENING.JSONL =====
    print("\n" + "="*80)
    print("STAGE 2: SCREENING (screening.jsonl)")
    print("="*80)

    screening_file = base_path / "screening.jsonl"
    screened_symbols = set()

    if screening_file.exists():
        with open(screening_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        symbol = entry.get("symbol")
                        action = entry.get("action")

                        if action == "accept":
                            screened_symbols.add(symbol)

                        if symbol in big_movers and action == "accept":
                            mover_status[symbol]['screened'] = True
                            mover_status[symbol]['screen_details'].append({
                                'timestamp': entry.get('timestamp'),
                                'action_type': entry.get('action_type'),
                                'levels_count': entry.get('levels_count'),
                                'ORH': entry.get('ORH'),
                                'ORL': entry.get('ORL')
                            })
                    except:
                        pass

        print(f"Unique symbols screened: {len(screened_symbols)}")
    else:
        print(f"ERROR: {screening_file} not found")

    # ===== STAGE 3: RANKING.JSONL =====
    print("\n" + "="*80)
    print("STAGE 3: RANKING (ranking.jsonl)")
    print("="*80)

    ranking_file = base_path / "ranking.jsonl"
    ranked_symbols = set()

    if ranking_file.exists():
        with open(ranking_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        symbol = entry.get("symbol")
                        ranked_symbols.add(symbol)

                        if symbol in big_movers:
                            mover_status[symbol]['ranked'] = True
                            mover_status[symbol]['rank_details'].append({
                                'rank_score': entry.get('rank_score'),
                                'passed_rank_threshold': entry.get('passed_rank_threshold')
                            })
                    except:
                        pass

        print(f"Unique symbols ranked: {len(ranked_symbols)}")
    else:
        print(f"ERROR: {ranking_file} not found")

    # ===== STAGE 4: EVENTS.JSONL =====
    print("\n" + "="*80)
    print("STAGE 4: EVENTS (events.jsonl)")
    print("="*80)

    events_file = base_path / "events.jsonl"
    event_symbols = set()

    if events_file.exists():
        with open(events_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        symbol = entry.get("symbol")
                        event_symbols.add(symbol)

                        if symbol in big_movers:
                            mover_status[symbol]['event'] = True
                            mover_status[symbol]['event_details'].append({
                                'setup_type': entry.get('decision', {}).get('setup_type'),
                                'regime': entry.get('decision', {}).get('regime'),
                                'rank_score': entry.get('features', {}).get('rank_score')
                            })
                    except:
                        pass

        print(f"Unique symbols in events: {len(event_symbols)}")
    else:
        print(f"ERROR: {events_file} not found")

    # ===== FUNNEL SUMMARY =====
    print("\n" + "="*80)
    print("FILTERING FUNNEL SUMMARY")
    print("="*80)
    print(f"\n1,992 stocks in universe")
    print(f"  ↓")
    print(f"{len(all_scanned_symbols):,} symbols scanned ({scan_accept_count:,} accept, {scan_reject_count:,} reject)")
    print(f"  ↓")
    print(f"{len(screened_symbols):,} symbols screened (ORB levels computed)")
    print(f"  ↓")
    print(f"{len(ranked_symbols):,} symbols ranked")
    print(f"  ↓")
    print(f"{len(event_symbols):,} symbols in events (trade plans)")

    # Calculate dropout rates
    if len(all_scanned_symbols) > 0:
        scan_to_screen_rate = len(screened_symbols) / len(all_scanned_symbols) * 100
    else:
        scan_to_screen_rate = 0

    if len(screened_symbols) > 0:
        screen_to_rank_rate = len(ranked_symbols) / len(screened_symbols) * 100
    else:
        screen_to_rank_rate = 0

    if len(ranked_symbols) > 0:
        rank_to_event_rate = len(event_symbols) / len(ranked_symbols) * 100
    else:
        rank_to_event_rate = 0

    print(f"\nDropout Rates:")
    print(f"  Scanning → Screening: {100-scan_to_screen_rate:.1f}% dropped")
    print(f"  Screening → Ranking:  {100-screen_to_rank_rate:.1f}% dropped ← BIGGEST BOTTLENECK")
    print(f"  Ranking → Events:     {100-rank_to_event_rate:.1f}% dropped")

    # ===== BIG MOVER TRACKING =====
    print("\n" + "="*80)
    print("BIG MOVER TRACKING THROUGH FUNNEL")
    print("="*80)

    for mover in big_movers:
        status = mover_status[mover]

        print(f"\n{mover}:")

        # Stage 1: Scanning
        if status['scanned_accept']:
            print(f"  [✓] SCANNED (ACCEPT)")
            if status['scan_details']:
                detail = status['scan_details'][0]
                print(f"      Category: {detail['category']}, Bias: {detail['bias']}")
                print(f"      Score Long: {detail.get('score_long', 0):.3f}, Rank: {detail.get('rank_long')}")
        elif status['scanned_reject']:
            print(f"  [✗] SCANNED (REJECT) ← FILTERED HERE")
            if status['scan_details']:
                detail = status['scan_details'][0]
                print(f"      Category: {detail['category']}, Bias: {detail['bias']}")
        else:
            print(f"  [✗] NOT SCANNED ← FILTERED HERE")

        # Stage 2: Screening
        if status['screened']:
            print(f"  [✓] SCREENED")
            if status['screen_details']:
                detail = status['screen_details'][0]
                print(f"      ORB Levels: ORH={detail.get('ORH')}, ORL={detail.get('ORL')}")
        else:
            if status['scanned_accept']:
                print(f"  [✗] NOT SCREENED ← FILTERED HERE")

        # Stage 3: Ranking
        if status['ranked']:
            print(f"  [✓] RANKED")
            if status['rank_details']:
                detail = status['rank_details'][0]
                print(f"      Rank Score: {detail.get('rank_score'):.2f}")
        else:
            if status['screened']:
                print(f"  [✗] NOT RANKED ← FILTERED HERE")

        # Stage 4: Events
        if status['event']:
            print(f"  [✓] TRADE PLAN CREATED")
            if status['event_details']:
                detail = status['event_details'][0]
                print(f"      Setup: {detail.get('setup_type')}, Regime: {detail.get('regime')}")
        else:
            if status['ranked']:
                print(f"  [✗] NO TRADE PLAN ← FILTERED HERE")

    # ===== KEY FINDINGS =====
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Count where big movers get filtered
    filtered_at_scan = sum(1 for s in mover_status.values() if not s['scanned_accept'])
    filtered_at_screen = sum(1 for s in mover_status.values() if s['scanned_accept'] and not s['screened'])
    filtered_at_rank = sum(1 for s in mover_status.values() if s['screened'] and not s['ranked'])
    filtered_at_event = sum(1 for s in mover_status.values() if s['ranked'] and not s['event'])

    print(f"\nBig Movers Filtered At Each Stage:")
    print(f"  Scanning:  {filtered_at_scan}/{len(big_movers)} ({filtered_at_scan/len(big_movers)*100:.1f}%)")
    print(f"  Screening: {filtered_at_screen}/{len(big_movers)} ({filtered_at_screen/len(big_movers)*100:.1f}%)")
    print(f"  Ranking:   {filtered_at_rank}/{len(big_movers)} ({filtered_at_rank/len(big_movers)*100:.1f}%)")
    print(f"  Events:    {filtered_at_event}/{len(big_movers)} ({filtered_at_event/len(big_movers)*100:.1f}%)")

    print(f"\n★ PRIMARY BOTTLENECK: {100-screen_to_rank_rate:.1f}% dropout from Screening → Ranking")
    print(f"★ BIG MOVERS: {filtered_at_rank} out of {len(big_movers)} filtered at Ranking stage")

if __name__ == '__main__':
    analyze_complete_funnel()
