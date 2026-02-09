"""
Calculate Net PnL with Zerodha MIS margins, platform fees, and taxation.

This script calculates the actual net profit/loss from a backtest run considering:
1. MIS (Margin Intraday Square-off) leverage per stock
2. Zerodha platform fees (brokerage, STT, exchange fees, etc.)
3. Income tax on speculative business income (30% + 4% cess)

Uses shared utilities from report_utils.py for fee constants, tax, and MIS lookup.

Usage:
    python tools/calculate_net_pnl.py <backtest_dir>
    python tools/calculate_net_pnl.py <backtest_dir> --no-mis

Example:
    python tools/calculate_net_pnl.py backtest_20251212-020235_extracted
    python tools/calculate_net_pnl.py 20260205-093403_full
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

# Shared reporting utilities (fee constants, tax, MIS leverage)
try:
    from report_utils import (
        BROKERAGE_RATE, BROKERAGE_CAP, STT_RATE, EXCHANGE_RATE_NSE,
        SEBI_RATE, IPFT_RATE, STAMP_DUTY_RATE, GST_RATE,
        calculate_order_charges, calculate_income_tax,
        load_nse_all, get_mis_leverage_for_symbol,
        load_mis_from_trade_reports, get_trade_mis_leverage,
    )
except ImportError:
    from tools.report_utils import (
        BROKERAGE_RATE, BROKERAGE_CAP, STT_RATE, EXCHANGE_RATE_NSE,
        SEBI_RATE, IPFT_RATE, STAMP_DUTY_RATE, GST_RATE,
        calculate_order_charges, calculate_income_tax,
        load_nse_all, get_mis_leverage_for_symbol,
        load_mis_from_trade_reports, get_trade_mis_leverage,
    )


def load_backtest_data(backtest_dir: str) -> Dict:
    """
    Load backtest data and aggregate by trade.

    Returns dict with:
    - trades: list of trade dicts with symbol, pnl, entry_price, entry_qty, exit_values
    - total_pnl: gross PnL
    - trade_count: number of unique trades
    """
    backtest_path = Path(backtest_dir)
    if not backtest_path.exists():
        raise FileNotFoundError(f"Backtest directory not found: {backtest_dir}")

    trade_pnls = defaultdict(float)
    trade_symbols = {}
    trade_entries = {}  # {trade_id: {'price': float, 'qty': int}}
    trade_exits = defaultdict(list)  # {trade_id: [{'price': float, 'qty': int}, ...]}

    for date_dir in sorted(backtest_path.iterdir()):
        if not date_dir.is_dir():
            continue

        # Read analytics.jsonl for PnL data
        analytics_file = date_dir / 'analytics.jsonl'
        if analytics_file.exists():
            with open(analytics_file, 'r') as f:
                for line in f:
                    try:
                        a = json.loads(line)
                        trade_id = a.get('trade_id')
                        pnl = a.get('pnl', 0)
                        symbol = a.get('symbol', 'UNKNOWN')

                        trade_pnls[trade_id] += pnl
                        if trade_id not in trade_symbols:
                            trade_symbols[trade_id] = symbol
                    except Exception:
                        continue

        # Read events.jsonl for entry/exit prices and quantities
        events_file = date_dir / 'events.jsonl'
        if events_file.exists():
            with open(events_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        event_type = event.get('type')
                        trade_id = event.get('trade_id')

                        if event_type == 'TRIGGER':
                            trigger = event.get('trigger', {})
                            entry_price = trigger.get('actual_price', 0)
                            entry_qty = trigger.get('qty', 0)
                            if trade_id and entry_price and entry_qty:
                                trade_entries[trade_id] = {
                                    'price': entry_price,
                                    'qty': entry_qty
                                }

                        elif event_type == 'EXIT':
                            exit_data = event.get('exit', {})
                            exit_price = exit_data.get('price', 0)
                            exit_qty = exit_data.get('qty', 0)
                            if trade_id and exit_price and exit_qty:
                                trade_exits[trade_id].append({
                                    'price': exit_price,
                                    'qty': exit_qty
                                })
                    except Exception:
                        continue

    trades = []
    for trade_id, pnl in trade_pnls.items():
        entry = trade_entries.get(trade_id, {})
        exits = trade_exits.get(trade_id, [])

        # Calculate total buy value (entry) and sell value (exits)
        buy_value = entry.get('price', 0) * entry.get('qty', 0)
        sell_value = sum(e['price'] * e['qty'] for e in exits)

        trades.append({
            'trade_id': trade_id,
            'symbol': trade_symbols.get(trade_id, 'UNKNOWN'),
            'pnl': pnl,
            'entry_price': entry.get('price', 0),
            'entry_qty': entry.get('qty', 0),
            'buy_value': buy_value,
            'sell_value': sell_value
        })

    return {
        'trades': trades,
        'total_pnl': sum(trade_pnls.values()),
        'trade_count': len(trade_pnls)
    }


def calculate_net_pnl(backtest_dir: str, use_mis: bool = True,
                      verbose: bool = True) -> Dict:
    """
    Calculate net PnL with MIS leverage, fees, and taxation.

    MIS leverage sourced from:
      1. trade_report.csv (per-trade, preferred)
      2. nse_all.json (symbol-level fallback)

    Args:
        backtest_dir: Path to extracted backtest directory
        use_mis: Whether to apply MIS leverage
        verbose: Print detailed output

    Returns:
        Dict with complete PnL breakdown
    """
    # Load backtest data
    data = load_backtest_data(backtest_dir)
    trades = data['trades']
    gross_pnl_nrml = data['total_pnl']
    trade_count = data['trade_count']

    if verbose:
        print("=" * 70)
        print("NET PNL CALCULATION WITH MIS, FEES, AND TAXATION")
        print("=" * 70)
        print(f"\nBacktest: {backtest_dir}")
        print(f"Total trades: {trade_count}")
        print(f"Gross PnL (NRML): Rs {gross_pnl_nrml:,.0f}")

    # Calculate MIS-adjusted PnL
    if use_mis:
        # Load MIS from trade_report.csv (preferred) and nse_all.json (fallback)
        mis_from_csv = load_mis_from_trade_reports([backtest_dir])
        nse_all = load_nse_all()

        mis_source = "trade_report.csv" if mis_from_csv else ("nse_all.json" if nse_all else "none")
        if verbose:
            print(f"\nMIS LEVERAGE (source: {mis_source}):")

        mis_pnl_total = 0
        multipliers = []
        for trade in trades:
            multiplier = get_trade_mis_leverage(trade, mis_from_csv, nse_all)
            mis_pnl = trade['pnl'] * multiplier
            mis_pnl_total += mis_pnl
            multipliers.append(multiplier)

        avg_multiplier = sum(multipliers) / len(multipliers) if multipliers else 1.0
        gross_pnl = mis_pnl_total

        if verbose:
            print(f"  Average multiplier: {avg_multiplier:.2f}x")
            print(f"  Gross PnL (with MIS): Rs {gross_pnl:,.0f}")
    else:
        gross_pnl = gross_pnl_nrml
        avg_multiplier = 1.0
        mis_from_csv = {}
        nse_all = {}

    # Calculate fees using actual trade values
    total_fees = 0
    total_buy_value = 0
    total_sell_value = 0
    trades_with_values = 0

    for trade in trades:
        buy_value = trade.get('buy_value', 0)
        sell_value = trade.get('sell_value', 0)

        if buy_value > 0 or sell_value > 0:
            # Use actual trade values — 2 orders per round-trip (buy + sell)
            charges = calculate_order_charges(buy_value, sell_value, num_orders=2)
            total_fees += charges['total_charges']
            total_buy_value += buy_value
            total_sell_value += sell_value
            trades_with_values += 1
        else:
            # Fallback: estimate with Rs 25k avg trade value if no actual data
            fallback_value = 25000
            charges = calculate_order_charges(fallback_value, fallback_value, num_orders=2)
            total_fees += charges['total_charges']

    avg_trade_value = (total_buy_value + total_sell_value) / (2 * trades_with_values) if trades_with_values > 0 else 25000
    total_turnover = total_buy_value + total_sell_value

    if verbose:
        print(f"\nZERODHA FEES:")
        print(f"  Trades: {trade_count}")
        print(f"  Total turnover: Rs {total_turnover:,.0f}")
        print(f"  Avg trade value: Rs {avg_trade_value:,.0f}")
        print(f"  Total fees: Rs {total_fees:,.0f}")
        if trade_count > 0:
            print(f"  Avg fee/trade: Rs {total_fees/trade_count:.0f}")

    # Calculate profit after fees
    profit_after_fees = gross_pnl - total_fees

    if verbose:
        print(f"\nPROFIT AFTER FEES: Rs {profit_after_fees:,.0f}")

    # Calculate tax
    tax_result = calculate_income_tax(profit_after_fees)

    if verbose:
        print(f"\nTAXATION (Speculative Business Income):")
        print(f"  Taxable income: Rs {tax_result['taxable_income']:,.0f}")
        print(f"  Base tax (30%): Rs {tax_result['base_tax']:,.0f}")
        print(f"  Cess (4%): Rs {tax_result['cess']:,.0f}")
        print(f"  Total tax: Rs {tax_result['total_tax']:,.0f}")
        effective_rate = (tax_result['total_tax'] / profit_after_fees * 100) if profit_after_fees > 0 else 0
        print(f"  Effective rate: {effective_rate:.1f}%")

    net_pnl = tax_result['net_after_tax']

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"FINAL NET PNL: Rs {net_pnl:,.0f}")
        print(f"{'=' * 70}")
        print(f"\nSummary:")
        print(f"  Gross (NRML):      Rs {gross_pnl_nrml:>12,.0f}")
        if use_mis:
            print(f"  Gross (MIS):       Rs {gross_pnl:>12,.0f}  ({avg_multiplier:.2f}x leverage)")
        print(f"  - Fees:            Rs {total_fees:>12,.0f}")
        print(f"  - Tax:             Rs {tax_result['total_tax']:>12,.0f}")
        print(f"  = Net:             Rs {net_pnl:>12,.0f}")

    return {
        'backtest_dir': backtest_dir,
        'trade_count': trade_count,
        'gross_pnl_nrml': gross_pnl_nrml,
        'mis_multiplier_avg': avg_multiplier,
        'gross_pnl_mis': gross_pnl if use_mis else gross_pnl_nrml,
        'total_fees': total_fees,
        'profit_after_fees': profit_after_fees,
        'tax': tax_result,
        'net_pnl': net_pnl
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate net PnL with MIS leverage, fees, and taxation'
    )
    parser.add_argument('backtest_dir', help='Path to extracted backtest directory')
    parser.add_argument('--no-mis', action='store_true',
                        help='Skip MIS leverage calculation')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')

    args = parser.parse_args()

    result = calculate_net_pnl(
        args.backtest_dir,
        use_mis=not args.no_mis,
        verbose=not args.quiet
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return result


if __name__ == '__main__':
    main()
