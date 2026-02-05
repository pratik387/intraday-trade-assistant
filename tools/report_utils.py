"""
Shared reporting utilities — single source of truth for post-processing calculations.

Used by:
  - tools/generate_backtest_report.py  (client/investor reports)
  - tools/calculate_net_pnl.py         (quick CLI PnL calculator)

All fee constants, tax rates, MIS leverage lookups, and trade data loaders
live here. NO production code dependencies — this is purely post-processing.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# FEE CONSTANTS — Zerodha Intraday Equity (single source of truth)
# Reference: https://zerodha.com/charges  (post Oct 1, 2024 "true-to-label")
# Verified against Zerodha console statutory charges breakdown Feb 2026
# =============================================================================
BROKERAGE_RATE = 0.0003         # 0.03% of order value
BROKERAGE_CAP = 20.0            # Rs 20 per order cap (whichever is lower)
STT_RATE = 0.00025              # 0.025% on sell side only
EXCHANGE_RATE_NSE = 0.0000297   # 0.00297% NSE transaction charges (total turnover)
SEBI_RATE = 0.000001            # Rs 10 per crore (total turnover)
IPFT_RATE = 0.000001            # Rs 10 per crore NSE Investor Protection Fund (total turnover)
STAMP_DUTY_RATE = 0.00003       # 0.003% on buy side (Maharashtra max rate)
GST_RATE = 0.18                 # 18% on (brokerage + exchange + SEBI + IPFT)

# =============================================================================
# TAX CONSTANTS — Speculative Business Income
# =============================================================================
TAX_BASE_RATE = 0.30            # 30% highest slab (conservative)
TAX_CESS_RATE = 0.04            # 4% health & education cess
EFFECTIVE_TAX_RATE = TAX_BASE_RATE * (1 + TAX_CESS_RATE)  # 31.2%

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NSE_ALL_PATH = PROJECT_ROOT / "nse_all.json"


# =============================================================================
# CHARGE CALCULATIONS
# =============================================================================

def calculate_order_charges(buy_turnover: float, sell_turnover: float,
                            num_orders: int) -> Dict[str, float]:
    """
    Calculate Zerodha intraday equity charges for a set of orders.

    Brokerage: min(0.03% × order_value, Rs 20) per order.
    Approximated using average order value when individual order sizes unavailable.

    Args:
        buy_turnover: Total buy-side turnover (sum of price × qty for buys)
        sell_turnover: Total sell-side turnover (sum of price × qty for sells)
        num_orders: Total number of executed orders (each buy/sell leg counts as 1)

    Returns:
        Dict with brokerage, stt, exchange, ipft, gst, stamp_duty, sebi,
        total_charges, total_orders, total_turnover
    """
    total_turnover = buy_turnover + sell_turnover

    # Brokerage: min(0.03% × avg_order_value, Rs 20) × num_orders
    if num_orders > 0:
        avg_order_value = total_turnover / num_orders
        brokerage = min(BROKERAGE_RATE * avg_order_value, BROKERAGE_CAP) * num_orders
    else:
        brokerage = 0.0

    stt = sell_turnover * STT_RATE
    exchange = total_turnover * EXCHANGE_RATE_NSE
    sebi = total_turnover * SEBI_RATE
    ipft = total_turnover * IPFT_RATE
    stamp_duty = buy_turnover * STAMP_DUTY_RATE
    gst = (brokerage + exchange + sebi + ipft) * GST_RATE
    total = brokerage + stt + exchange + ipft + gst + stamp_duty + sebi

    return {
        'brokerage': round(brokerage, 2),
        'stt': round(stt, 2),
        'exchange': round(exchange, 2),
        'sebi': round(sebi, 2),
        'ipft': round(ipft, 2),
        'stamp_duty': round(stamp_duty, 2),
        'gst': round(gst, 2),
        'total_charges': round(total, 2),
        'total_orders': num_orders,
        'total_turnover': round(total_turnover, 2),
    }


# =============================================================================
# TAX CALCULATIONS
# =============================================================================

def calculate_income_tax(net_profit: float) -> Dict[str, float]:
    """
    Calculate income tax on speculative business income.
    30% base + 4% cess = 31.2% effective rate.
    Only applies on positive profit.

    Args:
        net_profit: Net profit after all trading fees

    Returns:
        Dict with taxable_income, base_tax, cess, total_tax, net_after_tax
    """
    if net_profit <= 0:
        return {
            'taxable_income': 0,
            'base_tax': 0,
            'cess': 0,
            'total_tax': 0,
            'net_after_tax': net_profit,
        }

    base_tax = net_profit * TAX_BASE_RATE
    cess = base_tax * TAX_CESS_RATE
    total_tax = base_tax + cess

    return {
        'taxable_income': round(net_profit, 2),
        'base_tax': round(base_tax, 2),
        'cess': round(cess, 2),
        'total_tax': round(total_tax, 2),
        'net_after_tax': round(net_profit - total_tax, 2),
    }


# =============================================================================
# MIS LEVERAGE — from nse_all.json
# =============================================================================

_nse_all_cache: Optional[Dict] = None


def load_nse_all() -> Dict[str, dict]:
    """
    Load nse_all.json and return dict keyed by multiple symbol formats.
    Caches after first load.

    Keys: "AARTIIND.NS", "AARTIIND", "NSE:AARTIIND"
    """
    global _nse_all_cache
    if _nse_all_cache is not None:
        return _nse_all_cache

    if not NSE_ALL_PATH.exists():
        print(f"      WARNING: nse_all.json not found at {NSE_ALL_PATH}")
        _nse_all_cache = {}
        return _nse_all_cache

    with open(NSE_ALL_PATH, 'r') as f:
        data = json.load(f)

    result = {}
    for entry in data:
        sym = entry.get('symbol', '')
        result[sym] = entry
        # AARTIIND.NS → AARTIIND
        base = sym.replace('.NS', '').replace('.BO', '')
        result[base] = entry
        # NSE:AARTIIND
        result[f"NSE:{base}"] = entry

    _nse_all_cache = result
    return result


def get_mis_leverage_for_symbol(symbol: str, nse_all: Optional[Dict] = None) -> float:
    """
    Get MIS leverage for a symbol from nse_all.json.

    Args:
        symbol: Trading symbol (e.g. "NSE:AARTIIND", "AARTIIND.NS", "AARTIIND")
        nse_all: Pre-loaded nse_all dict. If None, loads from file.

    Returns:
        MIS leverage multiplier (e.g. 5.0). Returns 1.0 if not found/disabled.
    """
    if nse_all is None:
        nse_all = load_nse_all()

    if not nse_all:
        return 1.0

    clean = symbol.strip()
    candidates = [clean]

    # NSE:SYMBOL → SYMBOL
    if ':' in clean:
        base = clean.split(':')[-1]
        candidates.extend([base, f"{base}.NS"])

    # SYMBOL.NS → SYMBOL
    if '.NS' in clean or '.BO' in clean:
        candidates.append(clean.replace('.NS', '').replace('.BO', ''))

    # SYMBOL_uid → SYMBOL (trade_id format like NSE:SOLARA_fec08ea9)
    for c in list(candidates):
        if '_' in c:
            root = c.split('_')[0]
            candidates.extend([root, f"NSE:{root}", f"{root}.NS"])

    for c in candidates:
        entry = nse_all.get(c)
        if entry:
            lev = entry.get('mis_leverage')
            if lev and float(lev) > 0:
                return float(lev)

    return 1.0


# =============================================================================
# MIS LEVERAGE — from trade_report.csv (per-trade, preferred source)
# =============================================================================

def load_mis_from_trade_reports(backtest_dirs) -> Dict[str, float]:
    """
    Read mis_leverage per trade_id from trade_report.csv files.

    This is the preferred MIS source because it captures the actual leverage
    used at the time of each trade (from nse_all.json at backtest run time).

    Args:
        backtest_dirs: List of (dir_path, period, start, end, run_id) tuples
                       OR list of directory path strings/Paths

    Returns:
        Dict mapping trade_id -> mis_leverage (float)
    """
    mis_map = {}

    for item in backtest_dirs:
        # Handle both tuple format (from discover_backtest_sources) and plain paths
        if isinstance(item, (list, tuple)):
            dir_path = item[0]
        else:
            dir_path = item

        base_path = Path(dir_path)
        if not base_path.exists():
            continue

        for session_dir in sorted(base_path.iterdir()):
            if not session_dir.is_dir():
                continue
            csv_path = session_dir / "trade_report.csv"
            if not csv_path.exists():
                continue
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tid = row.get('trade_id', '')
                        mis = row.get('mis_leverage', '')
                        if tid and mis:
                            try:
                                mis_map[tid] = float(mis)
                            except (ValueError, TypeError):
                                pass
            except Exception:
                continue

    return mis_map


def get_trade_mis_leverage(trade: dict, mis_from_csv: Dict[str, float],
                           nse_all: Optional[Dict] = None) -> float:
    """
    Get MIS leverage for a single trade, with fallback chain:
      1. Per-trade mis_leverage from trade_report.csv
      2. Symbol lookup from nse_all.json
      3. Default 1.0

    Args:
        trade: Trade dict with 'trade_id' and 'symbol' keys
        mis_from_csv: Pre-loaded dict from load_mis_from_trade_reports()
        nse_all: Pre-loaded nse_all dict (optional, loaded on demand)

    Returns:
        MIS leverage multiplier (float)
    """
    # 1. Per-trade from CSV (most accurate)
    tid = trade.get('trade_id', '')
    if tid and tid in mis_from_csv:
        lev = mis_from_csv[tid]
        if lev > 1.0:
            return lev

    # 2. Symbol lookup from nse_all.json
    symbol = trade.get('symbol', '')
    if symbol:
        return get_mis_leverage_for_symbol(symbol, nse_all)

    return 1.0


# =============================================================================
# DAILY SUMMARIES — from performance.json
# =============================================================================

def load_daily_summaries(backtest_dirs) -> List[dict]:
    """
    Load per-day summaries from performance.json files.

    Args:
        backtest_dirs: List of (dir_path, ...) tuples or path strings

    Returns:
        List of dicts: {date, pnl, trades, wins, losses, fees, ...}
    """
    summaries = []

    for item in backtest_dirs:
        if isinstance(item, (list, tuple)):
            dir_path = item[0]
        else:
            dir_path = item

        base_path = Path(dir_path)
        if not base_path.exists():
            continue

        for session_dir in sorted(base_path.iterdir()):
            if not session_dir.is_dir():
                continue
            perf_path = session_dir / "performance.json"
            if not perf_path.exists():
                continue
            try:
                with open(perf_path, 'r') as f:
                    perf = json.load(f)
                summary = perf.get('summary', {})
                summaries.append({
                    'date': session_dir.name,
                    'pnl': summary.get('total_pnl', 0),
                    'trades': summary.get('completed_trades', 0),
                    'wins': summary.get('wins', 0),
                    'losses': summary.get('losses', 0),
                    'breakevens': summary.get('breakevens', 0),
                    'win_rate': summary.get('win_rate', 0),
                    'fees': perf.get('execution', {}).get('total_fees', 0),
                })
            except Exception:
                continue

    return summaries
