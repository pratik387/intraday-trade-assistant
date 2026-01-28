#!/usr/bin/env python
"""
DEEP FORENSIC ANALYSIS - Institutional Breakout Filter Investigation

This tool performs a comprehensive analysis of blocked breakout trades:
1. Parses 57 blocked trades from FORENSIC_BREAKOUT_ANALYSIS.md
2. Searches filtered backtest logs to find WHY each trade was rejected
3. Identifies which specific filter rejected each trade (timing/conviction/accumulation/cleanness)
4. Runs spike tests using 1m OHLC data to validate decisions
5. Categorizes by rejection reason and P&L impact
6. Generates actionable recommendations for filter tuning

Filters analyzed:
- TIMING: Pre-institutional hours (9:15-9:45am retail noise)
- CONVICTION: Weak candle close position (<70% for longs, >30% for shorts)
- ACCUMULATION: Insufficient volume buildup (need 3/5 bars with vol_z>1.0)
- CLEANNESS: Over-tested levels (>3 touches in 20 bars)
"""

from __future__ import annotations
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

BASELINE_BACKTEST = "backtest_20251108-034930"
FILTERED_BACKTEST = "backtest_20251108-124615"
FILTERED_LOGS_DIR = ROOT / "backtest_20251108-124615_extracted" / "20251108-124615_full" / "20251108-124615"
OHLCV_ARCHIVE = ROOT / "ohlcv_archive"
FORENSIC_INPUT = ROOT / "FORENSIC_BREAKOUT_ANALYSIS.md"
OUTPUT_FILE = ROOT / "DEEP_FORENSIC_ANALYSIS.md"

# Filter keywords to search in logs
FILTER_PATTERNS = {
    "timing": [
        r"Pre-institutional hours",
        r"9:15-9:45am retail noise",
        r"Rejected.*timing",
    ],
    "conviction": [
        r"Weak.*candle",
        r"close at.*range.*need",
        r"Doji candle",
        r"no conviction",
    ],
    "accumulation": [
        r"No volume accumulation",
        r"vol_z>1\.0.*need",
        r"institutional volume.*buildup",
    ],
    "cleanness": [
        r"Over-tested level",
        r"touches.*bars",
        r"support/resistance zone",
    ],
    "other": [
        r"REJECT",
        r"blocked",
        r"filter",
    ]
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class BlockedTrade:
    """Represents a blocked breakout trade from baseline backtest"""
    def __init__(self, date: str, symbol: str, strategy: str, exit_reason: str, pnl: float):
        self.date = date
        self.symbol = symbol
        self.strategy = strategy
        self.exit_reason = exit_reason
        self.pnl = pnl
        self.is_winner = pnl > 0
        self.is_hard_sl = exit_reason == "hard_sl"

        # Will be populated by log analysis
        self.rejection_filter: Optional[str] = None
        self.rejection_reason: Optional[str] = None
        self.spike_test_result: Optional[Dict] = None

    def __repr__(self):
        return f"BlockedTrade({self.date}, {self.symbol}, {self.strategy}, P&L={self.pnl:.2f})"


class SpikeTestResult:
    """Results from simulating a trade using 1m OHLC data"""
    def __init__(self, symbol: str, date: str):
        self.symbol = symbol
        self.date = date
        self.entry_price: Optional[float] = None
        self.sl_price: Optional[float] = None
        self.t1_price: Optional[float] = None
        self.t2_price: Optional[float] = None
        self.exit_price: Optional[float] = None
        self.exit_reason: Optional[str] = None
        self.pnl: Optional[float] = None
        self.success: bool = False
        self.error: Optional[str] = None


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_forensic_report() -> List[BlockedTrade]:
    """Parse FORENSIC_BREAKOUT_ANALYSIS.md to extract all 57 blocked trades"""
    trades = []

    if not FORENSIC_INPUT.exists():
        print(f"[ERROR] Forensic input file not found: {FORENSIC_INPUT}")
        return trades

    with open(FORENSIC_INPUT, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find tables with blocked trades
    # Format: | Date | Symbol | Strategy | Exit Reason | P&L |
    pattern = r'\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*NSE:(\w+)\s*\|\s*(\w+)\s*\|\s*([^\|]+)\s*\|\s*Rs\.([-\d.]+)\s*\|'

    matches = re.findall(pattern, content)
    print(f"[PARSER] Found {len(matches)} blocked trades in forensic report")

    for match in matches:
        date, symbol, strategy, exit_reason, pnl_str = match
        pnl = float(pnl_str.replace(',', ''))
        exit_reason = exit_reason.strip()

        trade = BlockedTrade(date, symbol, strategy, exit_reason, pnl)
        trades.append(trade)

    print(f"[PARSER] Parsed {len(trades)} blocked trades")
    print(f"  - Winners: {sum(1 for t in trades if t.is_winner)}")
    print(f"  - Losers: {sum(1 for t in trades if not t.is_winner)}")
    print(f"  - Hard SL: {sum(1 for t in trades if t.is_hard_sl)}")

    return trades


def infer_filter_from_baseline_logs(trade: BlockedTrade) -> Tuple[str, str]:
    """
    Since filtered backtest has 0 breakouts (all blocked at structure level),
    we need to infer the rejection filter by analyzing the baseline backtest logs
    to understand the trade characteristics, then match against filter thresholds.

    Returns (filter_type, inferred_reason)
    """
    # For now, since we can't access baseline logs easily, classify all as
    # "institutional_filters" - the generic category.
    # The spike tests will tell us if the filters were correct.

    # Future enhancement: Load baseline events.jsonl to get entry_time, candle data, etc.
    return "institutional_filters", "Blocked by one of: timing/conviction/accumulation/cleanness filters"


def find_rejection_reason(trade: BlockedTrade) -> Tuple[Optional[str], Optional[str]]:
    """
    Search filtered backtest logs for rejection reason.

    Strategy:
    1. Search events_decisions.jsonl for rejected events with matching symbol + strategy
    2. Fall back to searching agent.log for rejection messages
    3. If not found, infer from baseline logs
    4. Analyze rejection_reason field to classify filter type

    Returns (filter_type, rejection_reason_text)
    """
    # Find log file for this date
    date_dir = FILTERED_LOGS_DIR / trade.date

    if not date_dir.exists():
        # Logs not found - infer from baseline
        return infer_filter_from_baseline_logs(trade)

    # First, search events_decisions.jsonl for rejected events
    decisions_file = date_dir / "events_decisions.jsonl"

    if decisions_file.exists():
        try:
            with open(decisions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line)

                        # Check if this is a rejection for our symbol/strategy
                        if (event.get('action') == 'reject' and
                            trade.symbol in event.get('symbol', '') and
                            trade.strategy in event.get('strategy_type', '')):

                            rejection_reason = event.get('rejection_reason', 'Unknown')

                            # Classify filter type based on rejection reason
                            for filter_type, patterns in FILTER_PATTERNS.items():
                                for pattern in patterns:
                                    if re.search(pattern, rejection_reason, re.IGNORECASE):
                                        return filter_type, rejection_reason

                            # If no pattern match, return as "other"
                            return "other", rejection_reason

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"[WARN] Error reading {decisions_file}: {e}")

    # Fallback: Search agent.log for rejection messages
    log_files = [
        date_dir / "agent.log",
        date_dir / "trade_logs.log"
    ]

    for log_file in log_files:
        if not log_file.exists():
            continue

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Search for symbol mentions with rejection keywords
            symbol_pattern = rf'{trade.symbol}.*(?:REJECT|reject|blocked|filter)'
            symbol_matches = re.finditer(symbol_pattern, content, re.IGNORECASE)

            for match in symbol_matches:
                # Extract surrounding context (500 chars)
                start = max(0, match.start() - 250)
                end = min(len(content), match.end() + 250)
                context = content[start:end]

                # Check each filter type
                for filter_type, patterns in FILTER_PATTERNS.items():
                    for pattern in patterns:
                        if re.search(pattern, context, re.IGNORECASE):
                            # Extract the rejection line
                            lines = context.split('\n')
                            for line in lines:
                                if re.search(pattern, line, re.IGNORECASE):
                                    return filter_type, line.strip()

        except Exception as e:
            print(f"[WARN] Error reading {log_file}: {e}")

    # Not found in logs - infer from baseline data
    return infer_filter_from_baseline_logs(trade)


# ============================================================================
# SPIKE TEST SIMULATION
# ============================================================================

def load_1m_ohlc(symbol: str, date: str) -> Optional[pd.DataFrame]:
    """Load 1-minute OHLC data from archive"""
    # Remove NSE: prefix if present
    clean_symbol = symbol.replace('NSE:', '')

    # Try multiple path patterns
    patterns = [
        OHLCV_ARCHIVE / f"{clean_symbol}_1m_{date}.csv",
        OHLCV_ARCHIVE / f"{clean_symbol}_1min_{date}.csv",
        OHLCV_ARCHIVE / date / f"{clean_symbol}_1m.csv",
        OHLCV_ARCHIVE / date / f"{clean_symbol}_1min.csv",
    ]

    for path in patterns:
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=['timestamp'])
                if len(df) > 0:
                    return df
            except Exception as e:
                print(f"[WARN] Error loading {path}: {e}")

    return None


def simulate_trade(trade: BlockedTrade, df_1m: pd.DataFrame, entry_time: str = "10:00") -> SpikeTestResult:
    """
    Simulate the trade using 1m OHLC data.

    Assumptions:
    - Entry: First breakout bar after entry_time (default 10:00 to avoid ORB)
    - SL: 2.0 ATR below entry (standard config)
    - T1: 1.5R (standard breakout target)
    - T2: 3.0R (standard breakout target)
    - Exit: Hard SL, T1, T2, or EOD squareoff at 15:15
    """
    result = SpikeTestResult(trade.symbol, trade.date)

    try:
        # Calculate 14-period ATR on 5m timeframe (resample 1m data)
        df_5m = df_1m.set_index('timestamp').resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        if len(df_5m) < 14:
            result.error = "Insufficient data for ATR calculation"
            return result

        # Calculate ATR
        df_5m['tr'] = np.maximum(
            df_5m['high'] - df_5m['low'],
            np.maximum(
                abs(df_5m['high'] - df_5m['close'].shift(1)),
                abs(df_5m['low'] - df_5m['close'].shift(1))
            )
        )
        atr = df_5m['tr'].tail(14).mean()

        # Find entry point (first bar after entry_time)
        df_1m = df_1m.copy()
        df_1m['time'] = pd.to_datetime(df_1m['timestamp']).dt.time
        entry_mask = df_1m['time'] >= pd.to_datetime(entry_time).time()

        if entry_mask.sum() == 0:
            result.error = f"No data after {entry_time}"
            return result

        entry_idx = entry_mask.idxmax()
        entry_bar = df_1m.loc[entry_idx]
        entry_price = float(entry_bar['close'])

        # Set levels based on strategy direction
        is_long = trade.strategy.endswith('_long')

        if is_long:
            sl_price = entry_price - (2.0 * atr)
            t1_price = entry_price + (1.5 * 2.0 * atr)  # 1.5R
            t2_price = entry_price + (3.0 * 2.0 * atr)  # 3.0R
        else:
            sl_price = entry_price + (2.0 * atr)
            t1_price = entry_price - (1.5 * 2.0 * atr)
            t2_price = entry_price - (3.0 * 2.0 * atr)

        result.entry_price = entry_price
        result.sl_price = sl_price
        result.t1_price = t1_price
        result.t2_price = t2_price

        # Simulate trade bar by bar
        remaining_bars = df_1m.loc[entry_idx+1:]
        position_size = 1.0  # Normalized to 100%

        for idx, bar in remaining_bars.iterrows():
            bar_time = pd.to_datetime(bar['timestamp']).time()
            bar_high = float(bar['high'])
            bar_low = float(bar['low'])
            bar_close = float(bar['close'])

            # Check exits in order: SL -> T1 -> T2 -> EOD

            # 1. Hard SL
            if is_long and bar_low <= sl_price:
                result.exit_price = sl_price
                result.exit_reason = "hard_sl"
                result.pnl = (sl_price - entry_price) * position_size
                result.success = True
                return result
            elif not is_long and bar_high >= sl_price:
                result.exit_price = sl_price
                result.exit_reason = "hard_sl"
                result.pnl = (entry_price - sl_price) * position_size
                result.success = True
                return result

            # 2. T1 (40% exit)
            if is_long and bar_high >= t1_price:
                # Partial exit at T1
                t1_pnl = (t1_price - entry_price) * 0.4
                position_size = 0.6
                result.pnl = t1_pnl  # Track T1 P&L
                # Continue for T2...
            elif not is_long and bar_low <= t1_price:
                t1_pnl = (entry_price - t1_price) * 0.4
                position_size = 0.6
                result.pnl = t1_pnl

            # 3. T2 (40% exit, 20% remains)
            if position_size == 0.6:  # T1 already hit
                if is_long and bar_high >= t2_price:
                    t2_pnl = (t2_price - entry_price) * 0.4
                    result.pnl += t2_pnl
                    position_size = 0.2
                    # Trail remaining 20%...
                elif not is_long and bar_low <= t2_price:
                    t2_pnl = (entry_price - t2_price) * 0.4
                    result.pnl += t2_pnl
                    position_size = 0.2

            # 4. EOD squareoff at 15:15
            if bar_time >= pd.to_datetime("15:15").time():
                result.exit_price = bar_close
                if result.pnl is None:
                    # No T1/T2 hit, full position exit
                    if is_long:
                        result.pnl = (bar_close - entry_price) * position_size
                    else:
                        result.pnl = (entry_price - bar_close) * position_size
                else:
                    # Add remaining position P&L
                    if is_long:
                        result.pnl += (bar_close - entry_price) * position_size
                    else:
                        result.pnl += (entry_price - bar_close) * position_size

                r_multiple = result.pnl / (abs(entry_price - sl_price))
                result.exit_reason = f"eod_squareoff_15:15 ({r_multiple:.2f}R)"
                result.success = True
                return result

        # If we get here, trade didn't exit (data ended)
        result.error = "Trade did not exit (incomplete data)"
        return result

    except Exception as e:
        result.error = f"Simulation error: {e}"
        return result


def run_spike_tests(trades: List[BlockedTrade]) -> Dict[str, SpikeTestResult]:
    """Run spike tests for all blocked trades"""
    results = {}

    print(f"\n[SPIKE TEST] Running simulations for {len(trades)} blocked trades...")

    for i, trade in enumerate(trades, 1):
        print(f"  [{i}/{len(trades)}] {trade.symbol} on {trade.date}...", end=' ')

        # Load 1m data
        df_1m = load_1m_ohlc(trade.symbol, trade.date)

        if df_1m is None:
            print("X No 1m data")
            result = SpikeTestResult(trade.symbol, trade.date)
            result.error = "1m OHLC data not found"
            results[f"{trade.date}_{trade.symbol}"] = result
            continue

        # Run simulation
        result = simulate_trade(trade, df_1m)

        if result.success:
            print(f"OK P&L: {result.pnl:.2f}, Exit: {result.exit_reason}")
        else:
            print(f"ERROR {result.error}")

        results[f"{trade.date}_{trade.symbol}"] = result
        trade.spike_test_result = result.__dict__

    return results


# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

def analyze_by_filter(trades: List[BlockedTrade]) -> Dict[str, Dict]:
    """Group trades by rejection filter and calculate statistics"""
    filter_stats = defaultdict(lambda: {
        'count': 0,
        'winners': [],
        'losers': [],
        'hard_sl_losers': [],
        'total_pnl_blocked': 0.0,
        'winner_pnl': 0.0,
        'loser_pnl': 0.0,
        'spike_test_validated': 0,
        'spike_test_failed': 0,
    })

    for trade in trades:
        filter_type = trade.rejection_filter or "unknown"
        stats = filter_stats[filter_type]

        stats['count'] += 1
        stats['total_pnl_blocked'] += trade.pnl

        if trade.is_winner:
            stats['winners'].append(trade)
            stats['winner_pnl'] += trade.pnl
        else:
            stats['losers'].append(trade)
            stats['loser_pnl'] += trade.pnl
            if trade.is_hard_sl:
                stats['hard_sl_losers'].append(trade)

        # Check spike test validation
        if trade.spike_test_result:
            spike_pnl = trade.spike_test_result.get('pnl')
            if spike_pnl is not None:
                # Validated if spike test matches baseline outcome
                if (spike_pnl > 0) == (trade.pnl > 0):
                    stats['spike_test_validated'] += 1
                else:
                    stats['spike_test_failed'] += 1

    return dict(filter_stats)


def generate_report(trades: List[BlockedTrade], filter_stats: Dict) -> str:
    """Generate comprehensive markdown report"""

    report = []
    report.append("# DEEP FORENSIC ANALYSIS - Institutional Breakout Filters")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Baseline Backtest**: {BASELINE_BACKTEST} (102 trades, 57 breakouts)")
    report.append(f"**Filtered Backtest**: {FILTERED_BACKTEST} (84 trades, 0 breakouts)")
    report.append(f"**Blocked Trades Analyzed**: {len(trades)}")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## EXECUTIVE SUMMARY")
    report.append("")
    total_winners = sum(1 for t in trades if t.is_winner)
    total_losers = sum(1 for t in trades if not t.is_winner)
    total_hard_sl = sum(1 for t in trades if t.is_hard_sl)
    total_pnl = sum(t.pnl for t in trades)

    report.append(f"**Breakdown by Outcome**:")
    report.append(f"- Winners blocked: {total_winners} trades (Rs.{sum(t.pnl for t in trades if t.is_winner):.2f})")
    report.append(f"- Losers blocked: {total_losers} trades (Rs.{sum(t.pnl for t in trades if not t.is_winner):.2f})")
    report.append(f"- Hard SL losers: {total_hard_sl} trades")
    report.append(f"- **Net P&L Impact**: Rs.{total_pnl:.2f}")
    report.append("")

    # Filter-by-Filter Breakdown
    report.append("---")
    report.append("")
    report.append("## FILTER-BY-FILTER BREAKDOWN")
    report.append("")

    for filter_type, stats in sorted(filter_stats.items(), key=lambda x: -abs(x[1]['total_pnl_blocked'])):
        report.append(f"### {filter_type.upper()} Filter")
        report.append("")
        report.append(f"**Rejection Count**: {stats['count']} trades")
        report.append(f"**Winners Blocked**: {len(stats['winners'])} (Rs.{stats['winner_pnl']:.2f})")
        report.append(f"**Losers Blocked**: {len(stats['losers'])} (Rs.{stats['loser_pnl']:.2f})")
        report.append(f"**Hard SL Losers**: {len(stats['hard_sl_losers'])}")
        report.append(f"**Net P&L Impact**: Rs.{stats['total_pnl_blocked']:.2f}")
        report.append("")

        # Spike test validation
        total_spike_tests = stats['spike_test_validated'] + stats['spike_test_failed']
        if total_spike_tests > 0:
            validation_rate = stats['spike_test_validated'] / total_spike_tests * 100
            report.append(f"**Spike Test Validation**: {stats['spike_test_validated']}/{total_spike_tests} ({validation_rate:.1f}%)")
            report.append("")

        # Top blocked winners
        if stats['winners']:
            report.append(f"#### Top 5 Blocked Winners ({filter_type})")
            report.append("")
            report.append("| Date | Symbol | P&L | Exit Reason | Rejection Reason |")
            report.append("|------|--------|-----|-------------|------------------|")

            for trade in sorted(stats['winners'], key=lambda t: -t.pnl)[:5]:
                reason = (trade.rejection_reason or "Unknown")[:80]
                report.append(f"| {trade.date} | {trade.symbol} | Rs.{trade.pnl:.2f} | {trade.exit_reason} | {reason} |")
            report.append("")

        # Top blocked losers
        if stats['losers']:
            report.append(f"#### Top 5 Blocked Losers ({filter_type})")
            report.append("")
            report.append("| Date | Symbol | P&L | Exit Reason | Rejection Reason |")
            report.append("|------|--------|-----|-------------|------------------|")

            for trade in sorted(stats['losers'], key=lambda t: t.pnl)[:5]:
                reason = (trade.rejection_reason or "Unknown")[:80]
                report.append(f"| {trade.date} | {trade.symbol} | Rs.{trade.pnl:.2f} | {trade.exit_reason} | {reason} |")
            report.append("")

        report.append("")

    # Spike Test Results
    report.append("---")
    report.append("")
    report.append("## SPIKE TEST VALIDATION")
    report.append("")
    report.append("Top 10 blocked winners - Would they have actually been profitable?")
    report.append("")
    report.append("| Date | Symbol | Baseline P&L | Spike Test P&L | Match? | Exit Reason |")
    report.append("|------|--------|--------------|----------------|--------|-------------|")

    top_winners = sorted([t for t in trades if t.is_winner], key=lambda t: -t.pnl)[:10]
    for trade in top_winners:
        if trade.spike_test_result and trade.spike_test_result.get('success'):
            spike_pnl = trade.spike_test_result.get('pnl', 0.0)
            match = "YES" if (spike_pnl > 0) == (trade.pnl > 0) else "NO"
            exit_reason = trade.spike_test_result.get('exit_reason', 'Unknown')
            report.append(f"| {trade.date} | {trade.symbol} | Rs.{trade.pnl:.2f} | Rs.{spike_pnl:.2f} | {match} | {exit_reason} |")
        else:
            error = trade.spike_test_result.get('error', 'No data') if trade.spike_test_result else 'Not run'
            report.append(f"| {trade.date} | {trade.symbol} | Rs.{trade.pnl:.2f} | N/A | - | {error} |")

    report.append("")
    report.append("Top 10 blocked losers - Were filters correct to block them?")
    report.append("")
    report.append("| Date | Symbol | Baseline P&L | Spike Test P&L | Match? | Exit Reason |")
    report.append("|------|--------|--------------|----------------|--------|-------------|")

    top_losers = sorted([t for t in trades if not t.is_winner], key=lambda t: t.pnl)[:10]
    for trade in top_losers:
        if trade.spike_test_result and trade.spike_test_result.get('success'):
            spike_pnl = trade.spike_test_result.get('pnl', 0.0)
            match = "YES" if (spike_pnl > 0) == (trade.pnl > 0) else "NO"
            exit_reason = trade.spike_test_result.get('exit_reason', 'Unknown')
            report.append(f"| {trade.date} | {trade.symbol} | Rs.{trade.pnl:.2f} | Rs.{spike_pnl:.2f} | {match} | {exit_reason} |")
        else:
            error = trade.spike_test_result.get('error', 'No data') if trade.spike_test_result else 'Not run'
            report.append(f"| {trade.date} | {trade.symbol} | Rs.{trade.pnl:.2f} | N/A | - | {error} |")

    report.append("")

    # Recommendations
    report.append("---")
    report.append("")
    report.append("## ACTIONABLE RECOMMENDATIONS")
    report.append("")

    # Analyze which filter needs relaxing
    for filter_type, stats in sorted(filter_stats.items(), key=lambda x: -x[1]['winner_pnl']):
        if len(stats['winners']) == 0:
            continue

        win_rate = len(stats['hard_sl_losers']) / stats['count'] if stats['count'] > 0 else 0
        avg_winner = stats['winner_pnl'] / len(stats['winners']) if stats['winners'] else 0
        avg_loser = stats['loser_pnl'] / len(stats['losers']) if stats['losers'] else 0

        report.append(f"### {filter_type.upper()} Filter")
        report.append("")
        report.append(f"**Current Impact**:")
        report.append(f"- Blocks {len(stats['winners'])} winners (Rs.{stats['winner_pnl']:.2f}, avg Rs.{avg_winner:.2f}/trade)")
        report.append(f"- Blocks {len(stats['losers'])} losers (Rs.{stats['loser_pnl']:.2f}, avg Rs.{avg_loser:.2f}/trade)")
        report.append(f"- Hard SL prevention rate: {len(stats['hard_sl_losers'])}/{stats['count']} ({win_rate*100:.1f}%)")
        report.append("")

        # Recommendation logic
        if stats['winner_pnl'] > abs(stats['loser_pnl']) * 1.5:
            report.append(f"**Recommendation**: [WARNING] **RELAX THIS FILTER**")
            report.append(f"- Opportunity cost (Rs.{stats['winner_pnl']:.2f}) >> savings (Rs.{abs(stats['loser_pnl']):.2f})")
            report.append(f"- Suggested threshold adjustment: Analyze example trades below")
            report.append("")
        elif len(stats['hard_sl_losers']) / stats['count'] > 0.6:
            report.append(f"**Recommendation**: [OK] **FILTER WORKING WELL**")
            report.append(f"- Prevents {len(stats['hard_sl_losers'])}/{stats['count']} hard SL hits ({win_rate*100:.1f}%)")
            report.append(f"- Keep current thresholds")
            report.append("")
        else:
            report.append(f"**Recommendation**: [REVIEW] **NEEDS INVESTIGATION**")
            report.append(f"- Mixed results - review example trades below")
            report.append("")

        report.append("")

    report.append("---")
    report.append("")
    report.append("## DETAILED TRADE LIST")
    report.append("")
    report.append("All blocked trades with rejection reasons:")
    report.append("")
    report.append("| Date | Symbol | Strategy | P&L | Filter | Rejection Reason |")
    report.append("|------|--------|----------|-----|--------|------------------|")

    for trade in sorted(trades, key=lambda t: (t.rejection_filter or "zzz", -abs(t.pnl))):
        filter_type = trade.rejection_filter or "unknown"
        reason = (trade.rejection_reason or "Unknown")[:100]
        report.append(f"| {trade.date} | {trade.symbol} | {trade.strategy} | Rs.{trade.pnl:.2f} | {filter_type} | {reason} |")

    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by deep_forensic_analysis.py*")

    return "\n".join(report)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("DEEP FORENSIC ANALYSIS - Institutional Breakout Filters")
    print("=" * 80)
    print()

    # Step 1: Parse blocked trades from forensic report
    print("[STEP 1] Parsing FORENSIC_BREAKOUT_ANALYSIS.md...")
    trades = parse_forensic_report()

    if not trades:
        print("[ERROR] No trades found in forensic report. Exiting.")
        return

    print(f"[SUCCESS] Loaded {len(trades)} blocked trades")
    print()

    # Step 2: Find rejection reasons in logs
    print("[STEP 2] Analyzing filtered backtest logs for rejection reasons...")

    for i, trade in enumerate(trades, 1):
        print(f"  [{i}/{len(trades)}] {trade.symbol} on {trade.date}...", end=' ')

        filter_type, reason = find_rejection_reason(trade)
        trade.rejection_filter = filter_type
        trade.rejection_reason = reason

        print(f"{filter_type}")

    print()

    # Step 3: Run spike tests
    print("[STEP 3] Running spike tests to validate filter decisions...")
    spike_results = run_spike_tests(trades)
    print()

    # Step 4: Analyze results
    print("[STEP 4] Analyzing results by filter type...")
    filter_stats = analyze_by_filter(trades)

    for filter_type, stats in sorted(filter_stats.items()):
        print(f"  {filter_type}: {stats['count']} trades, "
              f"{len(stats['winners'])} winners (Rs.{stats['winner_pnl']:.2f}), "
              f"{len(stats['losers'])} losers (Rs.{stats['loser_pnl']:.2f})")
    print()

    # Step 5: Generate report
    print("[STEP 5] Generating comprehensive report...")
    report = generate_report(trades, filter_stats)

    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[SUCCESS] Report written to: {OUTPUT_FILE}")
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
