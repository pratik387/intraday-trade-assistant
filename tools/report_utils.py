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
# TAX CONSTANTS — Speculative Business Income (Section 43(5), Section 73)
# Indian FY runs April 1 to March 31. Tax on net annual speculative income.
# Speculative losses carry forward up to 4 years (only vs speculative income).
# =============================================================================
TAX_BASE_RATE = 0.30            # 30% highest slab (conservative)
TAX_CESS_RATE = 0.04            # 4% health & education cess
EFFECTIVE_TAX_RATE = TAX_BASE_RATE * (1 + TAX_CESS_RATE)  # 31.2%
SPECULATIVE_LOSS_CARRY_FORWARD_YEARS = 4  # Section 73

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

def get_financial_year(date_str: str) -> str:
    """
    Get Indian financial year label from a date string.

    Indian FY runs April 1 to March 31.
    Example: "2024-05-15" → "FY2024-25", "2025-02-10" → "FY2024-25"

    Args:
        date_str: Date in YYYY-MM-DD format (or any format starting with YYYY-MM-DD)

    Returns:
        FY label like "FY2024-25"
    """
    # Extract YYYY-MM-DD from the start
    date_part = date_str[:10]
    year = int(date_part[:4])
    month = int(date_part[5:7])

    if month >= 4:  # Apr-Dec: FY starts this year
        fy_start = year
    else:           # Jan-Mar: FY started previous year
        fy_start = year - 1

    fy_end = fy_start + 1
    return f"FY{fy_start}-{str(fy_end)[-2:]}"


def calculate_annual_tax(fy_nets: Dict[str, float]) -> Dict[str, object]:
    """
    Calculate income tax on speculative business income per financial year
    with loss carry-forward (Section 73, up to 4 years).

    Tax is on NET annual speculative income. Within a FY, all intraday
    profits and losses offset each other. Net losses carry forward up to
    4 years, only against speculative income.

    Args:
        fy_nets: Dict mapping FY label → net income after charges for that FY.
                 Example: {"FY2022-23": 150000, "FY2023-24": -50000, "FY2024-25": 200000}

    Returns:
        Dict with:
          total_tax, total_net_after_tax,
          per_fy: [{fy, gross_net, carried_loss, taxable, tax, net_after_tax}],
          total_loss_carried_forward (unexpired at end)
    """
    # Remove UNKNOWN FY — if present, treat as last FY (best-effort)
    unknown_net = fy_nets.pop('UNKNOWN', 0.0)

    # Sort FYs chronologically
    sorted_fys = sorted(fy_nets.keys())

    # Append unknown to last FY if any trades had no date
    if unknown_net != 0.0:
        if sorted_fys:
            fy_nets[sorted_fys[-1]] = fy_nets.get(sorted_fys[-1], 0.0) + unknown_net
        else:
            # All trades unknown — create a placeholder FY
            sorted_fys = ['FY0000-01']
            fy_nets['FY0000-01'] = unknown_net

    per_fy = []
    total_tax = 0.0
    total_net_after_tax = 0.0
    # Track carried losses: list of (fy_of_loss, amount_remaining)
    carried_losses: List[Tuple[str, float]] = []

    for fy in sorted_fys:
        gross_net = fy_nets[fy]

        # Expire losses older than 4 years
        fy_start_year = int(fy[2:6])
        carried_losses = [
            (loss_fy, amt) for loss_fy, amt in carried_losses
            if int(loss_fy[2:6]) + SPECULATIVE_LOSS_CARRY_FORWARD_YEARS >= fy_start_year
            and amt > 0
        ]

        # Apply carried losses against positive income
        loss_offset = 0.0
        if gross_net > 0 and carried_losses:
            remaining_income = gross_net
            updated_losses = []
            for loss_fy, loss_amt in carried_losses:
                if remaining_income <= 0:
                    updated_losses.append((loss_fy, loss_amt))
                    continue
                offset = min(remaining_income, loss_amt)
                remaining_income -= offset
                loss_offset += offset
                if loss_amt - offset > 0:
                    updated_losses.append((loss_fy, loss_amt - offset))
            carried_losses = updated_losses

        taxable = max(0, gross_net - loss_offset)

        # If net is negative this FY, add to carry-forward pool
        if gross_net < 0:
            carried_losses.append((fy, abs(gross_net)))

        # Tax on taxable amount
        if taxable > 0:
            base_tax = taxable * TAX_BASE_RATE
            cess = base_tax * TAX_CESS_RATE
            fy_tax = base_tax + cess
        else:
            fy_tax = 0.0

        fy_net_after_tax = gross_net - loss_offset - fy_tax
        # If gross_net was negative, net_after_tax is just the loss (no tax)
        if gross_net <= 0:
            fy_net_after_tax = gross_net

        total_tax += fy_tax
        total_net_after_tax += fy_net_after_tax

        per_fy.append({
            'fy': fy,
            'gross_net': round(gross_net, 2),
            'loss_offset': round(loss_offset, 2),
            'taxable': round(taxable, 2),
            'tax': round(fy_tax, 2),
            'net_after_tax': round(fy_net_after_tax, 2),
        })

    # Sum remaining carried losses
    remaining_carry = sum(amt for _, amt in carried_losses)

    return {
        'total_tax': round(total_tax, 2),
        'total_net_after_tax': round(total_net_after_tax, 2),
        'per_fy': per_fy,
        'total_loss_carried_forward': round(remaining_carry, 2),
    }


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
# PER-TRADE FINAL NET PNL — correct MIS + charges + tax calculation
# =============================================================================

def calculate_single_trade_charges(entry_price: float, exit_price: float,
                                    qty: int) -> Dict[str, float]:
    """
    Calculate Zerodha intraday charges for a single trade (1 buy + 1 sell order).

    Uses exact per-order brokerage cap instead of averaging.

    Args:
        entry_price: Entry price per share
        exit_price: Exit price per share
        qty: Number of shares

    Returns:
        Dict with brokerage, stt, exchange, sebi, ipft, stamp_duty, gst,
        total_charges
    """
    buy_turnover = entry_price * qty
    sell_turnover = exit_price * qty
    total_turnover = buy_turnover + sell_turnover

    # Per-order brokerage with Rs 20 cap
    buy_brokerage = min(BROKERAGE_RATE * buy_turnover, BROKERAGE_CAP)
    sell_brokerage = min(BROKERAGE_RATE * sell_turnover, BROKERAGE_CAP)
    brokerage = buy_brokerage + sell_brokerage

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
    }


def calculate_per_trade_final_pnl(trades: List[dict],
                                   mis_from_csv: Optional[Dict[str, float]] = None,
                                   nse_all: Optional[Dict] = None) -> Dict[str, float]:
    """
    Calculate final net PnL with per-trade MIS/charges and annual tax:
      1. MIS leverage × trade PnL (both profits AND losses) — per trade
      2. Subtract per-trade charges — per trade
      3. Tax on NET annual speculative income per FY (Section 73)
         Losses offset profits within same FY; net losses carry forward 4 years.

    Each trade dict needs keys:
      Required: 'pnl' (or 'realized_pnl'), 'symbol', 'trade_id'
      Required for tax: 'date' (YYYY-MM-DD) or 'session' (containing date)
      Optional: 'fees' (pre-calculated dict with 'total_fees'),
                'entry_price', 'exit_price', 'qty' (for charge fallback)

    Returns:
        Dict with gross_pnl, mis_pnl_total, total_charges, net_before_tax,
        total_tax, final_net_pnl, avg_multiplier, trade_count, tax_by_fy
    """
    if mis_from_csv is None:
        mis_from_csv = {}
    if nse_all is None:
        nse_all = load_nse_all()

    gross_pnl = 0.0
    mis_pnl_total = 0.0
    total_charges = 0.0
    multipliers = []
    trade_count = 0
    # Collect per-trade net (after MIS & charges) grouped by FY
    fy_nets: Dict[str, float] = {}

    for trade in trades:
        pnl = trade.get('pnl') or trade.get('realized_pnl') or 0.0
        gross_pnl += pnl

        # 1. Per-trade MIS leverage
        multiplier = get_trade_mis_leverage(trade, mis_from_csv, nse_all)
        multipliers.append(multiplier)
        trade_mis_pnl = pnl * multiplier
        mis_pnl_total += trade_mis_pnl

        # 2. Per-trade charges
        trade_charges = 0.0
        fees = trade.get('fees')
        if isinstance(fees, dict) and fees.get('total_fees', 0) > 0:
            trade_charges = fees['total_fees']
        elif isinstance(fees, (int, float)) and fees > 0:
            trade_charges = fees
        else:
            entry_p = trade.get('entry_price') or trade.get('actual_entry_price') or 0
            exit_p = trade.get('exit_price', 0)
            qty = trade.get('qty', 0)
            if entry_p > 0 and exit_p > 0 and qty > 0:
                charges = calculate_single_trade_charges(entry_p, exit_p, qty)
                trade_charges = charges['total_charges']
        total_charges += trade_charges

        # 3. Accumulate trade net into FY bucket for annual tax
        trade_net = trade_mis_pnl - trade_charges
        # Extract date for FY grouping
        date_str = trade.get('date', '')
        if not date_str:
            # Try session column (comprehensive_run_analyzer format)
            session = trade.get('session', '')
            if session and len(session) >= 10:
                # Extract YYYY-MM-DD from session name (e.g., "run_xxx_2024-01-15")
                import re as _re
                date_match = _re.search(r'\d{4}-\d{2}-\d{2}', session)
                date_str = date_match.group() if date_match else ''
        if date_str and len(date_str) >= 10:
            fy = get_financial_year(date_str)
        else:
            fy = 'UNKNOWN'
        fy_nets[fy] = fy_nets.get(fy, 0.0) + trade_net

        trade_count += 1

    avg_multiplier = sum(multipliers) / len(multipliers) if multipliers else 1.0
    net_before_tax = mis_pnl_total - total_charges

    # 3. Annual tax with loss carry-forward (Section 73)
    tax_result = calculate_annual_tax(fy_nets)
    total_tax = tax_result['total_tax']
    final_net_pnl = net_before_tax - total_tax

    return {
        'gross_pnl': round(gross_pnl, 2),
        'mis_pnl_total': round(mis_pnl_total, 2),
        'avg_multiplier': round(avg_multiplier, 2),
        'total_charges': round(total_charges, 2),
        'net_before_tax': round(net_before_tax, 2),
        'total_tax': round(total_tax, 2),
        'final_net_pnl': round(final_net_pnl, 2),
        'trade_count': trade_count,
        'tax_by_fy': tax_result['per_fy'],
        'loss_carried_forward': tax_result['total_loss_carried_forward'],
    }


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
