"""
Comprehensive Backtest Report Generator
Generates both high-level executive summary and detailed analysis reports.

Usage:
    # Auto-discover all backtest_*.zip and backtest_*_extracted/ directories
    python tools/generate_backtest_report.py

    # Specify specific files/directories
    python tools/generate_backtest_report.py backtest_20251217-143652_extracted
    python tools/generate_backtest_report.py backtest_20251201.zip backtest_20251215.zip
    python tools/generate_backtest_report.py backtest_*_extracted  # glob patterns work

Output Structure:
    analysis/reports/3year_backtest/run_YYYYMMDD_HHMMSS/
        - executive_summary.txt   (high-level customer-ready)
        - detailed_report.txt     (full detailed analysis)
        - data.json               (machine-readable data)
"""
import json
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

# ============================================================================
# CONFIGURATION
# ============================================================================

import zipfile
import glob


def extract_run_id_from_path(path_str: str) -> str:
    """
    Extract the backtest run ID from a path.

    Examples:
        - 20251228-075841_full -> 20251228-075841
        - backtest_20251228-075841_extracted -> 20251228-075841
        - backtest_20251228-120054.zip -> 20251228-120054
    """
    # Pattern: YYYYMMDD-HHMMSS (timestamp format)
    pattern = r'(\d{8}-\d{6})'
    match = re.search(pattern, path_str)
    if match:
        return match.group(1)
    return None

# ============================================================================
# ZERODHA INTRADAY EQUITY CHARGES
# ============================================================================
BROKERAGE_PER_ORDER = 20  # Rs 20 per executed order (flat)
STT_RATE = 0.00025  # 0.025% on SELL side only
EXCHANGE_RATE = 0.0000345  # NSE charges ~0.00345%
SEBI_RATE = 0.000001  # 0.0001%
STAMP_DUTY_RATE = 0.00003  # 0.003% on BUY side
GST_RATE = 0.18  # 18% on brokerage + exchange charges

# ============================================================================
# INCOME TAX ON SPECULATIVE BUSINESS INCOME
# ============================================================================
TAX_RATE = 0.30  # 30% (highest slab for speculative income)
CESS_RATE = 0.04  # 4% health and education cess on tax

# ============================================================================
# MIS LEVERAGE (Zerodha Margin Intraday Square-off)
# ============================================================================
NRML_MARGIN_PCT = 0.50  # NRML margin is typically 50% for most stocks

# Try to load pandas for MIS margin file reading
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def load_mis_margins(mis_file=None):
    """
    Load MIS margin data from Excel file.

    Args:
        mis_file: Path to Excel file with MIS margin percentages.
                  If None, searches common locations.

    Returns:
        Dict mapping symbol -> MIS margin percentage (e.g., 0.20 for 20%)
    """
    if not PANDAS_AVAILABLE:
        return {}

    # Find MIS file if not provided
    if not mis_file:
        # Project root is parent of tools directory
        project_root = Path(__file__).parent.parent
        candidates = [
            project_root / 'zerodha_mis_margin.xlsx',  # E:\Codebase\intraday-trade-assistant\zerodha_mis_margin.xlsx
            Path.cwd() / 'zerodha_mis_margin.xlsx',
            project_root / 'config' / 'zerodha_mis_margin.xlsx',
            Path.home() / 'zerodha_mis_margin.xlsx',
        ]
        for candidate in candidates:
            if candidate.exists():
                mis_file = str(candidate)
                print(f"      Found MIS file: {mis_file}")
                break

    if not mis_file or not Path(mis_file).exists():
        return {}

    try:
        df = pd.read_excel(mis_file)
        margins = {}

        # Find symbol and margin columns
        symbol_col = None
        margin_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'scrip' in col_lower or 'symbol' in col_lower or 'tradingsymbol' in col_lower:
                symbol_col = col
            elif 'mis' in col_lower:
                margin_col = col
            elif 'margin' in col_lower and margin_col is None:
                margin_col = col

        if symbol_col and margin_col:
            for _, row in df.iterrows():
                symbol = str(row[symbol_col]).upper().strip()
                if symbol in ('NAN', 'SCRIP', 'SYMBOL', '') or pd.isna(row[symbol_col]):
                    continue

                margin = row[margin_col]
                if isinstance(margin, str) and 'margin' in margin.lower():
                    continue

                try:
                    if isinstance(margin, str):
                        margin = float(margin.replace('%', '').replace('x', '').strip())
                    else:
                        margin = float(margin)

                    # If value > 1, assume it's a percentage (e.g., 20 = 20%)
                    if margin > 1:
                        margin = margin / 100

                    if margin > 0:
                        margins[symbol] = margin
                        # Also add without exchange prefix
                        if ':' in symbol:
                            margins[symbol.split(':')[-1]] = margin
                except (ValueError, TypeError):
                    continue

        print(f"      Loaded MIS margins for {len(margins)} symbols")
        return margins

    except Exception as e:
        print(f"      Warning: Could not load MIS data: {e}")
        return {}


def get_mis_multiplier(symbol, mis_margins):
    """
    Get MIS leverage multiplier for a symbol.

    Multiplier = NRML_MARGIN / MIS_MARGIN
    Example: NRML 50% / MIS 20% = 2.5x leverage
    """
    clean_symbol = symbol.upper().strip()
    if ':' in clean_symbol:
        clean_symbol = clean_symbol.split(':')[-1]
    if '_' in clean_symbol:
        clean_symbol = clean_symbol.split('_')[0]

    mis_margin = mis_margins.get(clean_symbol, NRML_MARGIN_PCT)

    if mis_margin <= 0:
        return 1.0

    return NRML_MARGIN_PCT / mis_margin


def calculate_income_tax(profit_after_fees):
    """
    Calculate income tax on trading profits.

    For intraday trading, profits are classified as "Speculative Business Income"
    and taxed at individual's applicable slab rate.
    Uses highest slab (30%) for conservative estimate.

    Args:
        profit_after_fees: Net profit after all trading fees

    Returns:
        Dict with tax breakdown
    """
    if profit_after_fees <= 0:
        return {
            'taxable_income': 0,
            'base_tax': 0,
            'cess': 0,
            'total_tax': 0,
            'net_after_tax': profit_after_fees
        }

    base_tax = profit_after_fees * TAX_RATE
    cess = base_tax * CESS_RATE
    total_tax = base_tax + cess

    return {
        'taxable_income': round(profit_after_fees, 2),
        'base_tax': round(base_tax, 2),
        'cess': round(cess, 2),
        'total_tax': round(total_tax, 2),
        'net_after_tax': round(profit_after_fees - total_tax, 2)
    }


def discover_backtest_sources(specific_paths=None):
    """
    Auto-discover backtest directories and zip files.
    Extracts zips if needed and returns list of (dir_path, period, start, end).

    Args:
        specific_paths: Optional list of specific paths/patterns to use instead of auto-discovery.
                       Supports glob patterns (e.g., "backtest_*_extracted").
    """
    sources = []

    if specific_paths:
        # Use provided paths instead of auto-discovery
        zips = []
        dirs = []
        for pattern in specific_paths:
            # Expand glob patterns
            matches = glob.glob(pattern)
            if not matches:
                # Try as literal path
                if Path(pattern).exists():
                    matches = [pattern]
                else:
                    print(f"      WARNING: No match for '{pattern}'")
                    continue
            for match in matches:
                if match.endswith('.zip'):
                    zips.append(match)
                elif Path(match).is_dir():
                    dirs.append(match)
        zips = sorted(set(zips))
        dirs = sorted(set(dirs))
    else:
        # Auto-discover all backtest zips and extracted directories
        zips = sorted(glob.glob("backtest_*.zip"))
        dirs = sorted(glob.glob("backtest_*_extracted")) + sorted([
            d for d in glob.glob("backtest_2025*")
            if Path(d).is_dir() and not d.endswith('.zip')
        ])

    # Extract any zips that don't have corresponding extracted dirs
    for zip_path in zips:
        extract_dir = zip_path.replace(".zip", "_extracted")
        if not Path(extract_dir).exists():
            print(f"      Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            dirs.append(extract_dir)

    # Remove duplicates
    dirs = sorted(set(dirs))

    # For each directory, detect date range from session folders
    for dir_path in dirs:
        if not Path(dir_path).exists():
            continue

        # Get all date folders (format: YYYY-MM-DD)
        date_folders = []
        for item in Path(dir_path).iterdir():
            if item.is_dir() and len(item.name) == 10 and item.name[4] == '-':
                try:
                    datetime.strptime(item.name, "%Y-%m-%d")
                    date_folders.append(item.name)
                except ValueError:
                    pass

        if date_folders:
            date_folders.sort()
            start_date = date_folders[0]
            end_date = date_folders[-1]
            # Create period label from date range
            start_year = start_date[:4]
            end_year = end_date[:4]
            if start_year == end_year:
                period = f"{start_year}"
            else:
                period = f"{start_year}-{end_year}"

            # Extract run ID from directory name
            run_id = extract_run_id_from_path(dir_path)

            sources.append((dir_path, period, start_date, end_date, run_id))

    return sources

# Will be populated at runtime
BACKTEST_DIRS = []

OUTPUT_DIR = Path("analysis/reports")


def parse_timestamp(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None


def extract_all_data():
    """Extract all trade data from backtest directories."""
    trades = []
    sessions = []

    for dir_name, period, start_date, end_date, run_id in BACKTEST_DIRS:
        base_path = Path(dir_name)
        if not base_path.exists():
            continue

        for session_path in sorted(base_path.iterdir()):
            if not session_path.is_dir():
                continue

            date = session_path.name
            session_trades = []
            decisions = {}
            triggers = {}

            events_file = session_path / "events.jsonl"
            if events_file.exists():
                with open(events_file, encoding='utf-8') as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                            trade_id = event.get("trade_id")
                            if event.get("type") == "DECISION" and trade_id:
                                decisions[trade_id] = {
                                    "entry_time": parse_timestamp(event.get("ts")),
                                    "notional": event.get("plan", {}).get("sizing", {}).get("notional", 0),
                                    "setup": event.get("plan", {}).get("strategy", "unknown"),
                                    "regime": event.get("plan", {}).get("regime", "unknown"),
                                }
                            if event.get("type") == "TRIGGER" and trade_id:
                                triggers[trade_id] = {
                                    "trigger_time": parse_timestamp(event.get("ts")),
                                }
                        except:
                            continue

            analytics_file = session_path / "analytics.jsonl"
            if analytics_file.exists():
                with open(analytics_file, encoding='utf-8') as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                            trade_id = event.get("trade_id")
                            if trade_id and event.get("is_final_exit"):
                                decision = decisions.get(trade_id, {})
                                trigger = triggers.get(trade_id, {})
                                # Get fees from analytics (calculated by TradingLogger)
                                fees_data = event.get("fees", {})
                                total_fees = fees_data.get("total_fees", 0) if fees_data else 0

                                trade = {
                                    "trade_id": trade_id,
                                    "date": date,
                                    "period": period,
                                    "symbol": event.get("symbol", ""),  # For MIS leverage calculation
                                    "pnl": event.get("total_trade_pnl", 0),
                                    "fees": total_fees,  # Actual fees from TradingLogger
                                    "net_pnl": event.get("net_pnl", 0),  # PnL after fees
                                    "exit_reason": event.get("reason", "unknown"),
                                    "exit_time": parse_timestamp(event.get("timestamp")),
                                    "setup": event.get("setup_type", decision.get("setup", "unknown")),
                                    "regime": event.get("regime", decision.get("regime", "unknown")),
                                    "notional": decision.get("notional", 0),
                                    "entry_time": trigger.get("trigger_time") or decision.get("entry_time"),
                                }
                                trades.append(trade)
                                session_trades.append(trade)
                        except:
                            continue

            if session_trades:
                sessions.append({
                    "date": date,
                    "period": period,
                    "trades": len(session_trades),
                    "pnl": sum(t["pnl"] for t in session_trades),
                    "winners": sum(1 for t in session_trades if t["pnl"] > 0),
                    "losers": sum(1 for t in session_trades if t["pnl"] <= 0),
                })

    return trades, sessions


def extract_order_data():
    """Extract order-level data for charges calculation from analytics files."""
    orders = []

    for dir_name, period, start_date, end_date, run_id in BACKTEST_DIRS:
        base_path = Path(dir_name)
        if not base_path.exists():
            continue

        for session_path in sorted(base_path.iterdir()):
            if not session_path.is_dir():
                continue

            analytics_file = session_path / "analytics.jsonl"
            if analytics_file.exists():
                with open(analytics_file, encoding='utf-8') as f:
                    for line in f:
                        try:
                            ev = json.loads(line)
                            exit_price = ev.get('exit_price', 0)
                            qty = ev.get('qty', 0)
                            entry_price = ev.get('entry_reference', 0) or ev.get('actual_entry_price', 0)
                            exit_seq = ev.get('exit_sequence', 1)
                            setup = ev.get('setup_type', 'unknown')

                            if exit_price <= 0 or qty <= 0:
                                continue

                            # Extract pre-calculated fees if available
                            fees = ev.get('fees', {})

                            orders.append({
                                'exit_price': exit_price,
                                'entry_price': entry_price if entry_price > 0 else exit_price,
                                'qty': qty,
                                'exit_sequence': exit_seq,
                                'is_final': ev.get('is_final_exit', False),
                                'gross_pnl': ev.get('total_trade_pnl', 0) if ev.get('is_final_exit') else 0,
                                'net_pnl': ev.get('net_pnl', 0) if ev.get('is_final_exit') else 0,
                                'setup': setup,
                                # Pre-calculated fees from analytics (if available)
                                'fees': fees,
                                'has_precalc_fees': bool(fees),
                            })
                        except:
                            continue
    return orders


def calculate_charges(orders):
    """
    Calculate Zerodha intraday equity charges from order data.

    Uses pre-calculated fees from analytics.jsonl when available (from TradingLogger).
    Falls back to local calculation for backward compatibility with older data.
    """
    total_brokerage = 0
    total_stt = 0
    total_exchange = 0
    total_gst = 0
    total_stamp = 0
    total_sebi = 0
    total_turnover = 0
    total_orders = 0
    precalc_count = 0

    for order in orders:
        exit_turnover = order['exit_price'] * order['qty']
        entry_turnover = order['entry_price'] * order['qty']

        # Use pre-calculated fees if available
        if order.get('has_precalc_fees') and order.get('fees'):
            fees = order['fees']
            total_brokerage += fees.get('brokerage', 0)
            total_stt += fees.get('stt', 0)
            total_exchange += fees.get('exchange', 0)
            total_gst += fees.get('gst', 0)
            total_stamp += fees.get('stamp_duty', 0)
            total_sebi += fees.get('sebi', 0)
            total_turnover += entry_turnover + exit_turnover
            total_orders += 2 if order['exit_sequence'] == 1 else 1
            precalc_count += 1
        else:
            # Fallback: calculate locally for backward compatibility
            # Entry order counted only once per trade (on first exit)
            if order['exit_sequence'] == 1:
                total_orders += 1
                total_brokerage += BROKERAGE_PER_ORDER
                total_stamp += entry_turnover * STAMP_DUTY_RATE
                total_turnover += entry_turnover

            # Exit order (each partial exit is separate)
            total_orders += 1
            total_brokerage += BROKERAGE_PER_ORDER
            total_stt += exit_turnover * STT_RATE
            total_exchange += exit_turnover * EXCHANGE_RATE
            total_sebi += exit_turnover * SEBI_RATE
            total_turnover += exit_turnover

    # GST on brokerage + exchange (only for fallback-calculated orders)
    # Pre-calculated fees already include GST
    if precalc_count < len(orders):
        # Some orders used fallback calculation, need to add GST for those
        fallback_brokerage = sum(
            BROKERAGE_PER_ORDER * (2 if o['exit_sequence'] == 1 else 1)
            for o in orders if not o.get('has_precalc_fees')
        )
        fallback_exchange = sum(
            o['exit_price'] * o['qty'] * EXCHANGE_RATE
            for o in orders if not o.get('has_precalc_fees')
        )
        total_gst += (fallback_brokerage + fallback_exchange) * GST_RATE

    total_charges = total_brokerage + total_stt + total_exchange + total_gst + total_stamp + total_sebi

    return {
        'total_orders': total_orders,
        'total_turnover': total_turnover,
        'brokerage': total_brokerage,
        'stt': total_stt,
        'exchange': total_exchange,
        'gst': total_gst,
        'stamp_duty': total_stamp,
        'sebi': total_sebi,
        'total_charges': total_charges,
        'precalc_count': precalc_count,
        'fallback_count': len(orders) - precalc_count,
    }


def calculate_setup_charges(orders):
    """Calculate charges per setup type for profitability analysis."""
    by_setup = defaultdict(lambda: {
        'orders': [],
        'gross_pnl': 0,
        'trades': 0
    })

    for order in orders:
        setup = order['setup']
        by_setup[setup]['orders'].append(order)
        if order['is_final']:
            by_setup[setup]['gross_pnl'] += order['gross_pnl']
            by_setup[setup]['trades'] += 1

    result = {}
    for setup, data in by_setup.items():
        charges = calculate_charges(data['orders'])
        net_pnl = data['gross_pnl'] - charges['total_charges']
        result[setup] = {
            'trades': data['trades'],
            'gross_pnl': data['gross_pnl'],
            'charges': charges['total_charges'],
            'net_pnl': net_pnl,
            'avg_gross': data['gross_pnl'] / data['trades'] if data['trades'] > 0 else 0,
            'avg_net': net_pnl / data['trades'] if data['trades'] > 0 else 0,
            'profitable': net_pnl > 0,
        }

    return dict(sorted(result.items(), key=lambda x: x[1]['net_pnl'], reverse=True))


def calculate_performance_summary(trades, sessions):
    total_pnl = sum(t["pnl"] for t in trades)
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in winners)
    gross_loss = abs(sum(t["pnl"] for t in losers))
    pnls = [t["pnl"] for t in trades]

    return {
        "total_sessions": len(sessions),
        "total_trades": len(trades),
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": total_pnl / len(trades) if trades else 0,
        "avg_pnl_per_session": total_pnl / len(sessions) if sessions else 0,
        "win_rate": len(winners) / len(trades) * 100 if trades else 0,
        "winners": len(winners),
        "losers": len(losers),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 999.99,
        "avg_winner": gross_profit / len(winners) if winners else 0,
        "avg_loser": -gross_loss / len(losers) if losers else 0,
        "best_trade": max(pnls) if pnls else 0,
        "worst_trade": min(pnls) if pnls else 0,
        "median_pnl": statistics.median(pnls) if pnls else 0,
        "std_dev": statistics.stdev(pnls) if len(pnls) > 1 else 0,
    }


def calculate_yearly_breakdown(trades, sessions):
    yearly = defaultdict(lambda: {"sessions": 0, "trades": 0, "pnl": 0, "winners": 0, "losers": 0})
    for s in sessions:
        year = s["date"][:4]
        yearly[year]["sessions"] += 1
        yearly[year]["trades"] += s["trades"]
        yearly[year]["pnl"] += s["pnl"]
        yearly[year]["winners"] += s["winners"]
        yearly[year]["losers"] += s["losers"]

    result = {}
    for year in sorted(yearly.keys()):
        y = yearly[year]
        result[year] = {
            "sessions": y["sessions"],
            "trades": y["trades"],
            "pnl": y["pnl"],
            "winners": y["winners"],
            "losers": y["losers"],
            "win_rate": y["winners"] / y["trades"] * 100 if y["trades"] > 0 else 0,
            "avg_pnl_per_trade": y["pnl"] / y["trades"] if y["trades"] > 0 else 0,
            "avg_pnl_per_session": y["pnl"] / y["sessions"] if y["sessions"] > 0 else 0,
        }
    return result


def calculate_monthly_breakdown(trades, sessions):
    monthly = defaultdict(lambda: {"sessions": 0, "trades": 0, "pnl": 0, "winners": 0, "losers": 0})
    for s in sessions:
        month = s["date"][:7]
        monthly[month]["sessions"] += 1
        monthly[month]["trades"] += s["trades"]
        monthly[month]["pnl"] += s["pnl"]
        monthly[month]["winners"] += s["winners"]
        monthly[month]["losers"] += s["losers"]

    result = {}
    for month in sorted(monthly.keys()):
        m = monthly[month]
        result[month] = {
            "sessions": m["sessions"],
            "trades": m["trades"],
            "pnl": m["pnl"],
            "winners": m["winners"],
            "losers": m["losers"],
            "win_rate": m["winners"] / m["trades"] * 100 if m["trades"] > 0 else 0,
            "avg_pnl_per_trade": m["pnl"] / m["trades"] if m["trades"] > 0 else 0,
        }
    return result


def calculate_setup_performance(trades):
    by_setup = defaultdict(list)
    for t in trades:
        by_setup[t["setup"]].append(t)

    result = {}
    for setup, setup_trades in by_setup.items():
        pnl = sum(t["pnl"] for t in setup_trades)
        winners = [t for t in setup_trades if t["pnl"] > 0]
        result[setup] = {
            "trades": len(setup_trades),
            "pnl": pnl,
            "win_rate": len(winners) / len(setup_trades) * 100 if setup_trades else 0,
            "avg_pnl": pnl / len(setup_trades) if setup_trades else 0,
            "winners": len(winners),
            "losers": len(setup_trades) - len(winners),
        }
    return dict(sorted(result.items(), key=lambda x: x[1]["pnl"], reverse=True))


def calculate_regime_performance(trades):
    by_regime = defaultdict(list)
    for t in trades:
        by_regime[t["regime"]].append(t)

    result = {}
    for regime, regime_trades in by_regime.items():
        pnl = sum(t["pnl"] for t in regime_trades)
        winners = [t for t in regime_trades if t["pnl"] > 0]
        result[regime] = {
            "trades": len(regime_trades),
            "pnl": pnl,
            "win_rate": len(winners) / len(regime_trades) * 100 if regime_trades else 0,
            "avg_pnl": pnl / len(regime_trades) if regime_trades else 0,
        }
    return dict(sorted(result.items(), key=lambda x: x[1]["pnl"], reverse=True))


def calculate_capital_requirements(trades):
    by_date = defaultdict(list)
    for t in trades:
        if t["entry_time"] and t["exit_time"] and t["notional"] > 0:
            by_date[t["date"]].append(t)

    peak_capitals = []
    peak_concurrent = []

    for date, day_trades in by_date.items():
        day_trades.sort(key=lambda x: x["entry_time"])
        events = []
        for t in day_trades:
            events.append((t["entry_time"], t["notional"], "entry"))
            events.append((t["exit_time"], -t["notional"], "exit"))
        events.sort(key=lambda x: x[0])

        current_capital = 0
        current_trades = 0
        max_capital = 0
        max_trades = 0

        for time, change, event_type in events:
            if event_type == "entry":
                current_capital += change
                current_trades += 1
            else:
                current_capital += change
                current_trades -= 1
            if current_capital > max_capital:
                max_capital = current_capital
                max_trades = current_trades

        peak_capitals.append(max_capital)
        peak_concurrent.append(max_trades)

    if not peak_capitals:
        return {}

    sorted_caps = sorted(peak_capitals)
    n = len(sorted_caps)

    return {
        "avg_peak_capital": sum(peak_capitals) / n,
        "median_peak_capital": sorted_caps[n // 2],
        "max_peak_capital": max(peak_capitals),
        "min_peak_capital": min(peak_capitals),
        "percentile_75": sorted_caps[int(n * 0.75)],
        "percentile_90": sorted_caps[int(n * 0.90)],
        "percentile_95": sorted_caps[int(n * 0.95)],
        "percentile_99": sorted_caps[int(n * 0.99)] if n > 100 else sorted_caps[-1],
        "avg_concurrent_trades": sum(peak_concurrent) / len(peak_concurrent),
        "max_concurrent_trades": max(peak_concurrent),
    }


def simulate_capital_constraint(trades, capital_limit, mis_margins=None):
    """
    Simulate trading with a capital constraint.

    Args:
        trades: List of trade dicts with entry_time, exit_time, notional, pnl, symbol, fees
        capital_limit: Maximum capital to deploy at any time
        mis_margins: Dict of symbol -> MIS margin percentage (for leverage calculation)

    Returns:
        Dict with trades_taken, gross_pnl_mis, fees, tax, net_pnl, capture_rate
    """
    by_date = defaultdict(list)
    for t in trades:
        if t["entry_time"] and t["exit_time"] and t["notional"] > 0:
            by_date[t["date"]].append(t)

    trades_taken = 0
    trades_skipped = 0
    pnl_taken = 0
    pnl_skipped = 0
    fees_taken = 0

    for date, day_trades in by_date.items():
        day_trades.sort(key=lambda x: x["entry_time"])
        active = []
        for t in day_trades:
            active = [(et, n) for et, n in active if et > t["entry_time"]]
            current = sum(n for _, n in active)

            # Calculate MIS-adjusted PnL for this trade
            if mis_margins:
                multiplier = get_mis_multiplier(t.get("symbol", ""), mis_margins)
            else:
                multiplier = 1.0
            mis_pnl = t["pnl"] * multiplier

            if current + t["notional"] <= capital_limit:
                active.append((t["exit_time"], t["notional"]))
                pnl_taken += mis_pnl
                fees_taken += t.get("fees", 0)  # Use actual fees from trade
                trades_taken += 1
            else:
                pnl_skipped += mis_pnl
                trades_skipped += 1

    # Net after fees (using actual fees from trades taken)
    net_after_fees = pnl_taken - fees_taken

    # Calculate tax on profit
    tax_result = calculate_income_tax(net_after_fees)
    net_pnl = tax_result['net_after_tax']

    return {
        "trades_taken": trades_taken,
        "trades_skipped": trades_skipped,
        "gross_pnl_mis": pnl_taken,
        "fees": fees_taken,
        "tax": tax_result['total_tax'],
        "net_pnl": net_pnl,
        "capture_rate": trades_taken / (trades_taken + trades_skipped) * 100 if (trades_taken + trades_skipped) > 0 else 0,
    }


def calculate_drawdown_analysis(sessions):
    equity = 0
    peak = 0
    max_drawdown = 0
    max_drawdown_pct = 0
    losing_streaks = []
    current_streak = 0

    for s in sorted(sessions, key=lambda x: x["date"]):
        equity += s["pnl"]
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if peak > 0:
            dd_pct = drawdown / peak * 100
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct

        if s["pnl"] < 0:
            current_streak += 1
        else:
            if current_streak > 0:
                losing_streaks.append(current_streak)
            current_streak = 0

    if current_streak > 0:
        losing_streaks.append(current_streak)

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "max_losing_streak": max(losing_streaks) if losing_streaks else 0,
        "avg_losing_streak": sum(losing_streaks) / len(losing_streaks) if losing_streaks else 0,
        "final_equity": equity,
        "peak_equity": peak,
    }


def calculate_risk_metrics(trades, sessions):
    daily_pnls = [s["pnl"] for s in sessions]
    if len(daily_pnls) < 2:
        return {}

    avg_daily = sum(daily_pnls) / len(daily_pnls)
    std_daily = statistics.stdev(daily_pnls)
    sharpe = (avg_daily * 252) / (std_daily * (252 ** 0.5)) if std_daily > 0 else 0

    negative_pnls = [p for p in daily_pnls if p < 0]
    downside_dev = statistics.stdev(negative_pnls) if len(negative_pnls) > 1 else 0
    sortino = (avg_daily * 252) / (downside_dev * (252 ** 0.5)) if downside_dev > 0 else 0

    drawdown = calculate_drawdown_analysis(sessions)
    # Calculate years from session date range
    session_dates = [s["date"] for s in sessions]
    if len(session_dates) >= 2:
        years_span = (datetime.strptime(max(session_dates), "%Y-%m-%d") -
                      datetime.strptime(min(session_dates), "%Y-%m-%d")).days / 365.25
        years_span = max(years_span, 1)  # At least 1 year for calculation
    else:
        years_span = 1
    calmar = (sum(daily_pnls) / years_span) / drawdown["max_drawdown"] if drawdown["max_drawdown"] > 0 else 0

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "daily_volatility": std_daily,
        "annualized_volatility": std_daily * (252 ** 0.5),
    }


def generate_executive_summary(data, output_path):
    lines = []
    lines.append("=" * 80)
    lines.append("TRADING STRATEGY - EXECUTIVE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Backtest Period: {data['period']['start']} to {data['period']['end']}")
    lines.append(f"Duration: {data['period']['years']} years ({data['period']['months']} months)")
    lines.append("")

    perf = data["performance"]
    chg = data["charges"]
    net_after_charges = perf['total_pnl'] - chg['total_charges']

    lines.append("-" * 80)
    lines.append("KEY PERFORMANCE METRICS (GROSS - before charges)")
    lines.append("-" * 80)
    lines.append(f"  Gross P&L:               Rs {perf['total_pnl']:>15,.2f}")
    lines.append(f"  Total Trades:            {perf['total_trades']:>15,}")
    lines.append(f"  Win Rate:                {perf['win_rate']:>14.1f}%")
    lines.append(f"  Profit Factor:           {perf['profit_factor']:>15.2f}")
    lines.append(f"  Average Trade (Gross):   Rs {perf['avg_pnl_per_trade']:>15,.2f}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("ZERODHA CHARGES BREAKDOWN")
    lines.append("-" * 80)
    lines.append(f"  Brokerage ({chg['total_orders']:,} orders × Rs 20): Rs {chg['brokerage']:>12,.0f}")
    lines.append(f"  STT (0.025% on sell):          Rs {chg['stt']:>12,.0f}")
    lines.append(f"  Exchange (0.00345%):           Rs {chg['exchange']:>12,.0f}")
    lines.append(f"  GST (18% on brkrg+exch):       Rs {chg['gst']:>12,.0f}")
    lines.append(f"  Stamp Duty (0.003% on buy):    Rs {chg['stamp_duty']:>12,.0f}")
    lines.append(f"  SEBI charges:                  Rs {chg['sebi']:>12,.0f}")
    lines.append("  " + "-" * 50)
    lines.append(f"  TOTAL CHARGES:                 Rs {chg['total_charges']:>12,.0f}")
    lines.append("")

    # MIS Leverage section
    mis = data.get("mis_leverage", {})
    tax = data.get("tax", {})

    lines.append("-" * 80)
    lines.append("NET P&L CALCULATION")
    lines.append("-" * 80)
    lines.append(f"  Gross P&L (NRML):        Rs {perf['total_pnl']:>15,.2f}")

    if mis.get('has_mis_data'):
        lines.append(f"  MIS Leverage (avg):      {mis.get('avg_multiplier', 1.0):>14.2f}x")
        lines.append(f"  Gross P&L (with MIS):    Rs {mis.get('gross_pnl_mis', perf['total_pnl']):>15,.2f}")
        lines.append(f"  Less: Trading Charges:   Rs {chg['total_charges']:>15,.2f}")
        lines.append(f"  ────────────────────────────────────────────────")
        lines.append(f"  Net P&L (after fees):    Rs {mis.get('net_after_fees_mis', net_after_charges):>15,.2f}")
    else:
        lines.append(f"  (MIS data not available - showing NRML 1x)")
        lines.append(f"  Less: Trading Charges:   Rs {chg['total_charges']:>15,.2f}")
        lines.append(f"  ────────────────────────────────────────────────")
        lines.append(f"  Net P&L (after fees):    Rs {net_after_charges:>15,.2f}")
    lines.append("")

    # Tax section
    lines.append("-" * 80)
    lines.append("INCOME TAX (Speculative Business Income)")
    lines.append("-" * 80)
    if tax.get('taxable_income', 0) > 0:
        lines.append(f"  Taxable Income:          Rs {tax.get('taxable_income', 0):>15,.2f}")
        lines.append(f"  Base Tax (30%):          Rs {tax.get('base_tax', 0):>15,.2f}")
        lines.append(f"  Health & Ed Cess (4%):   Rs {tax.get('cess', 0):>15,.2f}")
        lines.append(f"  ────────────────────────────────────────────────")
        lines.append(f"  Total Tax:               Rs {tax.get('total_tax', 0):>15,.2f}")
    else:
        lines.append(f"  No tax applicable (no profit)")
    lines.append("")

    # Final net PnL
    final_net = tax.get('net_after_tax', net_after_charges)
    lines.append("-" * 80)
    lines.append("═══════════════════════════════════════════════════════════════════════════════")
    lines.append(f"  FINAL NET P&L (after fees + tax): Rs {final_net:>15,.2f}")
    lines.append(f"  Avg Net per Trade:                Rs {final_net/perf['total_trades']:>15,.2f}")
    status = "PROFITABLE ✓" if final_net > 0 else "UNPROFITABLE ✗"
    lines.append(f"  Status: {status}")
    lines.append("═══════════════════════════════════════════════════════════════════════════════")
    lines.append("")

    lines.append("-" * 80)
    lines.append("ANNUAL PERFORMANCE")
    lines.append("-" * 80)
    lines.append(f"  {'Year':<8} {'Trades':>10} {'Win Rate':>10} {'Net Profit':>18} {'Avg/Trade':>12}")
    lines.append("  " + "-" * 60)
    for year, y in data["yearly"].items():
        lines.append(f"  {year:<8} {y['trades']:>10} {y['win_rate']:>9.1f}% Rs {y['pnl']:>15,.0f} Rs {y['avg_pnl_per_trade']:>9,.0f}")
    lines.append("")

    monthly = data["monthly"]
    winning_months = sum(1 for m in monthly.values() if m["pnl"] > 0)
    total_months = len(monthly)
    lines.append("-" * 80)
    lines.append("MONTHLY STATISTICS")
    lines.append("-" * 80)
    lines.append(f"  Profitable Months:       {winning_months} / {total_months} ({winning_months/total_months*100:.1f}%)")
    lines.append(f"  Best Month:              Rs {max(m['pnl'] for m in monthly.values()):>15,.2f}")
    lines.append(f"  Worst Month:             Rs {min(m['pnl'] for m in monthly.values()):>15,.2f}")
    lines.append(f"  Average Monthly:         Rs {sum(m['pnl'] for m in monthly.values())/len(monthly):>15,.2f}")
    lines.append("")

    risk = data["risk_metrics"]
    dd = data["drawdown"]
    lines.append("-" * 80)
    lines.append("RISK METRICS")
    lines.append("-" * 80)
    lines.append(f"  Sharpe Ratio:            {risk.get('sharpe_ratio', 0):>15.2f}")
    lines.append(f"  Sortino Ratio:           {risk.get('sortino_ratio', 0):>15.2f}")
    lines.append(f"  Max Drawdown:            Rs {dd['max_drawdown']:>15,.2f}")
    lines.append(f"  Max Losing Streak:       {dd['max_losing_streak']:>15} days")
    lines.append("")

    cap = data["capital"]
    lines.append("-" * 80)
    lines.append("CAPITAL REQUIREMENTS")
    lines.append("-" * 80)
    lines.append(f"  Recommended (95%):       Rs {cap['percentile_95']:>15,.0f}")
    lines.append(f"  Conservative (90%):      Rs {cap['percentile_90']:>15,.0f}")
    lines.append(f"  Average Required:        Rs {cap['avg_peak_capital']:>15,.0f}")
    lines.append(f"  Max Concurrent Trades:   {cap['max_concurrent_trades']:>15}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("CAPITAL SCENARIO ANALYSIS (MIS + Fees + Tax)")
    lines.append("-" * 80)
    lines.append(f"  {'Capital':<10} {'Trades':>8} {'Gross MIS':>12} {'Fees':>10} {'Tax':>10} {'Net PnL':>12} {'ROI/yr':>10}")
    lines.append("  " + "-" * 75)
    years = data["period"]["years"] or 1
    for label, sim in data["capital_scenarios"].items():
        cap_val = int(label.replace("L", "")) * 100000
        annual_roi = (sim["net_pnl"] / cap_val / years) * 100
        lines.append(
            f"  Rs {label:<7} {sim['trades_taken']:>8} "
            f"Rs {sim['gross_pnl_mis']:>9,.0f} "
            f"Rs {sim['fees']:>7,.0f} "
            f"Rs {sim['tax']:>7,.0f} "
            f"Rs {sim['net_pnl']:>9,.0f} "
            f"{annual_roi:>9.1f}%"
        )
    lines.append("")

    lines.append("-" * 80)
    lines.append("STRATEGY PROFITABILITY (AFTER CHARGES)")
    lines.append("-" * 80)
    lines.append(f"  {'Setup':<32} {'Trades':>7} {'Gross':>12} {'Charges':>12} {'Net':>12} {'Status':<10}")
    lines.append("  " + "-" * 85)
    setup_chg = data["setup_charges"]
    profitable_count = sum(1 for s in setup_chg.values() if s['profitable'])
    for setup, s in list(setup_chg.items())[:10]:
        status = "✓" if s['profitable'] else "✗"
        lines.append(f"  {setup[:32]:<32} {s['trades']:>7} Rs {s['gross_pnl']:>9,.0f} Rs {s['charges']:>9,.0f} Rs {s['net_pnl']:>9,.0f} {status}")
    lines.append("")
    lines.append(f"  Profitable setups: {profitable_count}/{len(setup_chg)}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF EXECUTIVE SUMMARY")
    lines.append("=" * 80)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def generate_detailed_report(data, output_path):
    lines = []
    lines.append("=" * 100)
    lines.append("TRADING STRATEGY - DETAILED ANALYSIS REPORT")
    lines.append("=" * 100)
    lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Backtest Period: {data['period']['start']} to {data['period']['end']}")
    lines.append("")

    # SECTION 1
    perf = data["performance"]
    chg = data["charges"]
    net_after_charges = perf['total_pnl'] - chg['total_charges']

    lines.append("=" * 100)
    lines.append("SECTION 1: PERFORMANCE SUMMARY (GROSS)")
    lines.append("=" * 100)
    lines.append(f"  Total Trading Sessions:       {perf['total_sessions']:>12}")
    lines.append(f"  Total Trades Executed:        {perf['total_trades']:>12}")
    lines.append(f"  Winning Trades:               {perf['winners']:>12}")
    lines.append(f"  Losing Trades:                {perf['losers']:>12}")
    lines.append(f"  Win Rate:                     {perf['win_rate']:>11.2f}%")
    lines.append(f"  Gross P&L:                    Rs {perf['total_pnl']:>15,.2f}")
    lines.append(f"  Gross Profit (winners):       Rs {perf['gross_profit']:>15,.2f}")
    lines.append(f"  Gross Loss (losers):          Rs {perf['gross_loss']:>15,.2f}")
    lines.append(f"  Profit Factor:                {perf['profit_factor']:>15.2f}")
    lines.append(f"  Average Trade:                Rs {perf['avg_pnl_per_trade']:>15,.2f}")
    lines.append(f"  Average Winner:               Rs {perf['avg_winner']:>15,.2f}")
    lines.append(f"  Average Loser:                Rs {perf['avg_loser']:>15,.2f}")
    lines.append(f"  Best Trade:                   Rs {perf['best_trade']:>15,.2f}")
    lines.append(f"  Worst Trade:                  Rs {perf['worst_trade']:>15,.2f}")
    lines.append("")

    # SECTION 1B - CHARGES
    lines.append("=" * 100)
    lines.append("SECTION 1B: ZERODHA CHARGES BREAKDOWN")
    lines.append("=" * 100)
    lines.append(f"  Total Orders Executed:        {chg['total_orders']:>12,}")
    lines.append(f"  Total Turnover:               Rs {chg['total_turnover']:>15,.2f}")
    lines.append("")
    lines.append("  Charge Components:")
    lines.append(f"    Brokerage (Rs 20/order):    Rs {chg['brokerage']:>15,.2f}")
    lines.append(f"    STT (0.025% on sell):       Rs {chg['stt']:>15,.2f}")
    lines.append(f"    Exchange (0.00345%):        Rs {chg['exchange']:>15,.2f}")
    lines.append(f"    GST (18% on brkrg+exch):    Rs {chg['gst']:>15,.2f}")
    lines.append(f"    Stamp Duty (0.003% on buy): Rs {chg['stamp_duty']:>15,.2f}")
    lines.append(f"    SEBI charges:               Rs {chg['sebi']:>15,.2f}")
    lines.append("    " + "-" * 50)
    lines.append(f"    TOTAL CHARGES:              Rs {chg['total_charges']:>15,.2f}")
    lines.append("")
    lines.append("  NET P&L AFTER ALL CHARGES:")
    lines.append(f"    Gross P&L:                  Rs {perf['total_pnl']:>15,.2f}")
    lines.append(f"    Less: Charges:              Rs {chg['total_charges']:>15,.2f}")
    lines.append("    " + "=" * 50)
    lines.append(f"    NET P&L:                    Rs {net_after_charges:>15,.2f}")
    lines.append(f"    Avg Net per Trade:          Rs {net_after_charges/perf['total_trades']:>15,.2f}")
    status = "PROFITABLE" if net_after_charges > 0 else "UNPROFITABLE"
    lines.append(f"    Status:                     {status:>15}")
    lines.append("")

    # SECTION 2
    lines.append("=" * 100)
    lines.append("SECTION 2: YEARLY BREAKDOWN")
    lines.append("=" * 100)
    lines.append(f"  {'Year':<6} {'Sessions':>10} {'Trades':>10} {'Winners':>10} {'Losers':>10} {'Win%':>8} {'Total PnL':>18} {'Avg/Trade':>12}")
    lines.append("  " + "-" * 95)
    for year, y in data["yearly"].items():
        lines.append(f"  {year:<6} {y['sessions']:>10} {y['trades']:>10} {y['winners']:>10} {y['losers']:>10} {y['win_rate']:>7.1f}% Rs {y['pnl']:>15,.0f} Rs {y['avg_pnl_per_trade']:>9,.0f}")
    lines.append("")

    # SECTION 3
    lines.append("=" * 100)
    lines.append("SECTION 3: MONTHLY BREAKDOWN")
    lines.append("=" * 100)
    lines.append(f"  {'Month':<10} {'Sessions':>8} {'Trades':>8} {'Win%':>8} {'PnL':>15} {'Avg/Trade':>12}")
    lines.append("  " + "-" * 65)
    for month, m in data["monthly"].items():
        lines.append(f"  {month:<10} {m['sessions']:>8} {m['trades']:>8} {m['win_rate']:>7.1f}% Rs {m['pnl']:>12,.0f} Rs {m['avg_pnl_per_trade']:>9,.0f}")
    lines.append("")

    # SECTION 4
    lines.append("=" * 100)
    lines.append("SECTION 4: STRATEGY/SETUP PERFORMANCE (GROSS)")
    lines.append("=" * 100)
    lines.append(f"  {'Setup':<40} {'Trades':>8} {'Win%':>8} {'Total PnL':>15} {'Avg PnL':>12}")
    lines.append("  " + "-" * 90)
    for setup, s in data["setups"].items():
        lines.append(f"  {setup[:40]:<40} {s['trades']:>8} {s['win_rate']:>7.1f}% Rs {s['pnl']:>12,.0f} Rs {s['avg_pnl']:>9,.0f}")
    lines.append("")

    # SECTION 4B - SETUP PROFITABILITY AFTER CHARGES
    lines.append("=" * 100)
    lines.append("SECTION 4B: STRATEGY PROFITABILITY (AFTER CHARGES)")
    lines.append("=" * 100)
    lines.append(f"  {'Setup':<35} {'Trades':>7} {'Gross PnL':>12} {'Charges':>12} {'Net PnL':>12} {'Avg Net':>10} {'Status':<6}")
    lines.append("  " + "-" * 95)
    setup_chg = data["setup_charges"]
    for setup, s in setup_chg.items():
        status = "OK" if s['profitable'] else "LOSS"
        lines.append(f"  {setup[:35]:<35} {s['trades']:>7} Rs {s['gross_pnl']:>9,.0f} Rs {s['charges']:>9,.0f} Rs {s['net_pnl']:>9,.0f} Rs {s['avg_net']:>7,.0f} {status}")
    lines.append("")
    profitable_count = sum(1 for s in setup_chg.values() if s['profitable'])
    lines.append(f"  Summary: {profitable_count}/{len(setup_chg)} setups profitable after charges")
    lines.append("")

    # SECTION 5
    lines.append("=" * 100)
    lines.append("SECTION 5: MARKET REGIME PERFORMANCE")
    lines.append("=" * 100)
    lines.append(f"  {'Regime':<20} {'Trades':>10} {'Win%':>10} {'Total PnL':>18} {'Avg PnL':>12}")
    lines.append("  " + "-" * 75)
    for regime, r in data["regimes"].items():
        lines.append(f"  {regime:<20} {r['trades']:>10} {r['win_rate']:>9.1f}% Rs {r['pnl']:>15,.0f} Rs {r['avg_pnl']:>9,.0f}")
    lines.append("")

    # SECTION 6
    cap = data["capital"]
    lines.append("=" * 100)
    lines.append("SECTION 6: CAPITAL REQUIREMENTS ANALYSIS")
    lines.append("=" * 100)
    lines.append(f"  Average Peak Capital:      Rs {cap['avg_peak_capital']:>15,.0f}")
    lines.append(f"  Median Peak Capital:       Rs {cap['median_peak_capital']:>15,.0f}")
    lines.append(f"  Maximum Peak Capital:      Rs {cap['max_peak_capital']:>15,.0f}")
    lines.append(f"  75th Percentile:           Rs {cap['percentile_75']:>15,.0f}")
    lines.append(f"  90th Percentile:           Rs {cap['percentile_90']:>15,.0f}")
    lines.append(f"  95th Percentile:           Rs {cap['percentile_95']:>15,.0f}")
    lines.append(f"  Max Concurrent Trades:     {cap['max_concurrent_trades']:>15}")
    lines.append("")
    lines.append("  Capital Constraint Simulation (MIS + Fees + Tax):")
    lines.append(f"    {'Capital':<10} {'Trades':>8} {'Gross MIS':>12} {'Fees':>10} {'Tax':>10} {'Net PnL':>12} {'ROI/Yr':>10}")
    lines.append("    " + "-" * 75)
    years = data["period"]["years"] or 1
    for label, sim in data["capital_scenarios"].items():
        cap_val = int(label.replace("L", "")) * 100000
        annual_roi = (sim["net_pnl"] / cap_val / years) * 100
        lines.append(
            f"    Rs {label:<7} {sim['trades_taken']:>8} "
            f"Rs {sim['gross_pnl_mis']:>9,.0f} "
            f"Rs {sim['fees']:>7,.0f} "
            f"Rs {sim['tax']:>7,.0f} "
            f"Rs {sim['net_pnl']:>9,.0f} "
            f"{annual_roi:>9.1f}%"
        )
    lines.append("")

    # SECTION 7
    risk = data["risk_metrics"]
    dd = data["drawdown"]
    lines.append("=" * 100)
    lines.append("SECTION 7: RISK ANALYSIS")
    lines.append("=" * 100)
    lines.append(f"  Sharpe Ratio:              {risk.get('sharpe_ratio', 0):>15.2f}")
    lines.append(f"  Sortino Ratio:             {risk.get('sortino_ratio', 0):>15.2f}")
    lines.append(f"  Calmar Ratio:              {risk.get('calmar_ratio', 0):>15.2f}")
    lines.append(f"  Daily Volatility:          Rs {risk.get('daily_volatility', 0):>15,.2f}")
    lines.append(f"  Maximum Drawdown:          Rs {dd['max_drawdown']:>15,.2f}")
    lines.append(f"  Max Drawdown %:            {dd['max_drawdown_pct']:>14.2f}%")
    lines.append(f"  Max Losing Streak:         {dd['max_losing_streak']:>15} days")
    lines.append("")

    # SECTION 8
    lines.append("=" * 100)
    lines.append("SECTION 8: BEST & WORST ANALYSIS")
    lines.append("=" * 100)
    sorted_months = sorted(data["monthly"].items(), key=lambda x: x[1]["pnl"], reverse=True)
    lines.append("  Top 5 Best Months:")
    for month, m in sorted_months[:5]:
        lines.append(f"    {month}: Rs {m['pnl']:>12,.2f} | {m['trades']} trades | {m['win_rate']:.1f}%")
    lines.append("")
    lines.append("  Top 5 Worst Months:")
    for month, m in sorted_months[-5:]:
        lines.append(f"    {month}: Rs {m['pnl']:>12,.2f} | {m['trades']} trades | {m['win_rate']:.1f}%")
    lines.append("")

    sorted_sessions = sorted(data["sessions"], key=lambda x: x["pnl"], reverse=True)
    lines.append("  Top 10 Best Days:")
    for s in sorted_sessions[:10]:
        lines.append(f"    {s['date']}: Rs {s['pnl']:>10,.2f} | {s['trades']} trades | {s['winners']}W/{s['losers']}L")
    lines.append("")
    lines.append("  Top 10 Worst Days:")
    for s in sorted_sessions[-10:]:
        lines.append(f"    {s['date']}: Rs {s['pnl']:>10,.2f} | {s['trades']} trades | {s['winners']}W/{s['losers']}L")
    lines.append("")

    lines.append("=" * 100)
    lines.append("END OF DETAILED REPORT")
    lines.append("=" * 100)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def main():
    global BACKTEST_DIRS

    print("=" * 70)
    print("BACKTEST REPORT GENERATOR")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for command-line arguments
    specific_paths = sys.argv[1:] if len(sys.argv) > 1 else None

    if specific_paths:
        print(f"\n[0/6] Using specified sources: {', '.join(specific_paths)}")
    else:
        print("\n[0/6] Auto-discovering backtest sources...")

    BACKTEST_DIRS = discover_backtest_sources(specific_paths)
    if not BACKTEST_DIRS:
        print("      ERROR: No backtest directories or zips found!")
        print("      Expected: backtest_*.zip or backtest_*_extracted/ directories")
        return

    # Collect run IDs from all sources
    backtest_run_ids = []
    for dir_path, period, start, end, run_id in BACKTEST_DIRS:
        run_id_str = f" [run: {run_id}]" if run_id else ""
        print(f"      Found: {dir_path} [{period}] ({start} to {end}){run_id_str}")
        if run_id:
            backtest_run_ids.append(run_id)

    print("\n[1/7] Extracting trade data...")
    trades, sessions = extract_all_data()
    print(f"      Found {len(trades)} trades across {len(sessions)} sessions")

    print("[2/7] Extracting order data for charges...")
    orders = extract_order_data()
    print(f"      Found {len(orders)} order events")

    print("[3/7] Calculating performance metrics...")
    performance = calculate_performance_summary(trades, sessions)
    yearly = calculate_yearly_breakdown(trades, sessions)
    monthly = calculate_monthly_breakdown(trades, sessions)

    print("[4/8] Calculating Zerodha charges...")
    charges = calculate_charges(orders)
    setup_charges = calculate_setup_charges(orders)
    net_after_fees = performance['total_pnl'] - charges['total_charges']
    print(f"      Total charges: Rs {charges['total_charges']:,.0f}")
    print(f"      Net P&L after charges: Rs {net_after_fees:,.0f}")

    print("[5/8] Loading MIS margins and calculating leverage...")
    mis_margins = load_mis_margins()
    if mis_margins:
        # Calculate MIS-adjusted PnL per trade
        multipliers = []
        mis_pnl_total = 0
        for trade in trades:
            multiplier = get_mis_multiplier(trade.get('symbol', ''), mis_margins)
            multipliers.append(multiplier)
            mis_pnl_total += trade.get('pnl', 0) * multiplier
        avg_multiplier = sum(multipliers) / len(multipliers) if multipliers else 1.0
        mis_net_after_fees = mis_pnl_total - charges['total_charges']
        print(f"      Avg MIS multiplier: {avg_multiplier:.2f}x")
        print(f"      Gross PnL with MIS: Rs {mis_pnl_total:,.0f}")
        print(f"      Net PnL with MIS (after fees): Rs {mis_net_after_fees:,.0f}")
    else:
        print(f"      No MIS file found, using 1x leverage (NRML)")
        avg_multiplier = 1.0
        mis_pnl_total = performance['total_pnl']
        mis_net_after_fees = net_after_fees

    # Calculate income tax on profit after fees (using MIS-adjusted if available)
    tax_result = calculate_income_tax(mis_net_after_fees)
    print(f"      Income tax (30% + 4% cess): Rs {tax_result['total_tax']:,.0f}")
    print(f"      Final Net PnL after tax: Rs {tax_result['net_after_tax']:,.0f}")

    print("[6/8] Analyzing setup and regime performance...")
    setups = calculate_setup_performance(trades)
    regimes = calculate_regime_performance(trades)

    print("[7/8] Analyzing capital requirements...")
    capital = calculate_capital_requirements(trades)
    capital_scenarios = {}
    for limit, label in [(300000, "3L"), (500000, "5L"), (600000, "6L"), (1000000, "10L")]:
        capital_scenarios[label] = simulate_capital_constraint(trades, limit, mis_margins)

    print("[8/8] Calculating risk metrics...")
    risk_metrics = calculate_risk_metrics(trades, sessions)
    drawdown = calculate_drawdown_analysis(sessions)

    all_dates = [s["date"] for s in sessions]
    start_date = min(all_dates)
    end_date = max(all_dates)
    years = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days / 365.25
    data = {
        "period": {"start": start_date, "end": end_date, "years": round(years, 1), "months": len(monthly)},
        "performance": performance,
        "yearly": yearly,
        "monthly": monthly,
        "setups": setups,
        "regimes": regimes,
        "capital": capital,
        "capital_scenarios": capital_scenarios,
        "risk_metrics": risk_metrics,
        "drawdown": drawdown,
        "sessions": sessions,
        "charges": charges,
        "setup_charges": setup_charges,
        "mis_leverage": {
            "avg_multiplier": round(avg_multiplier, 2),
            "gross_pnl_nrml": performance['total_pnl'],
            "gross_pnl_mis": mis_pnl_total,
            "net_after_fees_mis": mis_net_after_fees,
            "has_mis_data": bool(mis_margins),
        },
        "tax": tax_result,
    }

    print("[6/6] Generating reports...")

    # Create unique run folder: analysis/reports/3year_backtest/run_YYYYMMDD_HHMMSS/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_year = data["period"]["start"][:4]
    end_year = data["period"]["end"][:4]

    run_dir = OUTPUT_DIR / "3year_backtest" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"      Output folder: {run_dir}")

    exec_path = run_dir / "executive_summary.txt"
    generate_executive_summary(data, exec_path)
    print(f"      Executive Summary: {exec_path.name}")

    detail_path = run_dir / "detailed_report.txt"
    generate_detailed_report(data, detail_path)
    print(f"      Detailed Report: {detail_path.name}")

    json_path = run_dir / "data.json"
    json_data = {k: v for k, v in data.items() if k != "sessions"}
    json_data["metadata"] = {
        "generated_at": timestamp,
        "date_range": f"{start_year}-{end_year}",
        "start_date": data["period"]["start"],
        "end_date": data["period"]["end"],
        "backtest_run_ids": backtest_run_ids,  # OCI job run IDs used for this backtest
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"      JSON Data: {json_path.name}")
    if backtest_run_ids:
        print(f"      Run IDs: {', '.join(backtest_run_ids)}")

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)

    with open(exec_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Replace Unicode box characters that may not print on Windows console
        content = content.replace("═", "=").replace("─", "-").replace("✓", "[OK]").replace("✗", "[X]")
        print("\n" + content)


if __name__ == "__main__":
    main()
