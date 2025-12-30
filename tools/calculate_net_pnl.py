"""
Calculate Net PnL with Zerodha MIS margins, platform fees, and taxation.

This script calculates the actual net profit/loss from a backtest run considering:
1. MIS (Margin Intraday Square-off) leverage per stock
2. Zerodha platform fees (brokerage, STT, exchange fees, etc.)
3. Income tax on speculative business income (30% + 4% cess)

Usage:
    python tools/calculate_net_pnl.py <backtest_dir> [--mis-file <path>]

Example:
    python tools/calculate_net_pnl.py backtest_20251212-020235_extracted
    python tools/calculate_net_pnl.py backtest_20251212-020235_extracted --mis-file zerodha_mis_margin.xlsx
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional

# Try to import pandas/openpyxl for Excel reading
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class ZerodhaFeeCalculator:
    """
    Calculate Zerodha intraday equity trading fees.

    Fee structure for Intraday Equity (as of Dec 2024):
    - Brokerage: Rs 20 per executed order OR 0.03% (whichever is lower)
    - STT: 0.025% on sell side only
    - Exchange Transaction Charges: NSE 0.00297%, BSE 0.00375%
    - SEBI Turnover Fee: Rs 10 per crore (0.0001%)
    - Stamp Duty: 0.003% on buy side (varies by state, using max)
    - GST: 18% on (brokerage + exchange txn charges + SEBI charges)

    Reference: https://zerodha.com/brokerage-calculator
    """

    # Fee structure (as of Dec 2024)
    BROKERAGE_FLAT = 20  # Rs per executed order
    BROKERAGE_PCT = 0.0003  # 0.03%
    STT_PCT = 0.00025  # 0.025% on sell side only (intraday equity)
    EXCHANGE_TXN_NSE_PCT = 0.0000297  # 0.00297% NSE
    EXCHANGE_TXN_BSE_PCT = 0.0000375  # 0.00375% BSE
    SEBI_CHARGES_PCT = 0.000001  # Rs 10 per crore = 0.0001%
    STAMP_DUTY_PCT = 0.00003  # 0.003% on buy side (Maharashtra rate)
    GST_PCT = 0.18  # 18% on brokerage + exchange txn + SEBI

    @classmethod
    def calculate_fees(cls, buy_value: float, sell_value: float, exchange: str = 'NSE') -> Dict[str, float]:
        """
        Calculate all fees for a round-trip intraday trade.

        Args:
            buy_value: Total buy value (price * qty)
            sell_value: Total sell value (price * qty)
            exchange: 'NSE' or 'BSE' (default NSE)

        Returns:
            Dict with breakdown of all fees
        """
        turnover = buy_value + sell_value

        # Select exchange transaction rate
        exchange_txn_pct = cls.EXCHANGE_TXN_NSE_PCT if exchange == 'NSE' else cls.EXCHANGE_TXN_BSE_PCT

        # Brokerage: Rs 20 flat per order (buy + sell = 2 orders)
        # For Zerodha, it's Rs 20 per executed order regardless of quantity
        # So for a round-trip trade: Rs 20 (buy) + Rs 20 (sell) = Rs 40
        brokerage = cls.BROKERAGE_FLAT * 2  # Buy order + Sell order

        # STT (Securities Transaction Tax) - only on sell side for intraday
        stt = sell_value * cls.STT_PCT

        # Exchange transaction charges (on total turnover)
        exchange_txn = turnover * exchange_txn_pct

        # SEBI turnover fee (on total turnover)
        sebi = turnover * cls.SEBI_CHARGES_PCT

        # Stamp duty (only on buy side)
        stamp_duty = buy_value * cls.STAMP_DUTY_PCT

        # GST @ 18% on (brokerage + exchange txn charges + SEBI charges)
        gst = (brokerage + exchange_txn + sebi) * cls.GST_PCT

        total_fees = brokerage + stt + exchange_txn + sebi + stamp_duty + gst

        return {
            'brokerage': round(brokerage, 2),
            'stt': round(stt, 2),
            'exchange_txn': round(exchange_txn, 2),
            'sebi': round(sebi, 2),
            'stamp_duty': round(stamp_duty, 2),
            'gst': round(gst, 2),
            'total_fees': round(total_fees, 2)
        }

    @classmethod
    def estimate_fees_from_pnl(cls, gross_pnl: float, avg_trade_value: float = 50000) -> float:
        """
        Estimate fees from gross PnL when exact trade values aren't available.

        Uses average trade value to estimate turnover and fees.
        For a typical Rs 50k trade, fees are approximately Rs 30-40.
        """
        # Estimate number of trades from gross PnL (rough approximation)
        # This is a simplified estimation
        estimated_trades = max(1, abs(gross_pnl) / 250)  # Assume Rs 250 avg PnL per trade

        # Each trade has buy and sell legs
        total_turnover = estimated_trades * avg_trade_value * 2

        # Simplified fee calculation (approximately 0.05-0.06% of turnover)
        estimated_fee_pct = 0.0005  # 0.05%
        return round(total_turnover * estimated_fee_pct, 2)


class MISLeverageCalculator:
    """Calculate MIS leverage multipliers from Zerodha margin data."""

    NRML_MARGIN_PCT = 0.50  # NRML margin is typically 50% for most stocks

    def __init__(self, mis_file: Optional[str] = None):
        """
        Initialize with MIS margin data file.

        Args:
            mis_file: Path to Excel file with MIS margin percentages
        """
        self.mis_margins = {}
        if mis_file and PANDAS_AVAILABLE:
            self._load_mis_data(mis_file)

    def _load_mis_data(self, filepath: str):
        """Load MIS margin data from Excel file."""
        try:
            df = pd.read_excel(filepath)
            # Expected columns: 'Scrip'/'Symbol' and 'MIS'/'Margin'
            symbol_col = None
            margin_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if 'scrip' in col_lower or 'symbol' in col_lower or 'instrument' in col_lower or 'tradingsymbol' in col_lower:
                    symbol_col = col
                elif 'mis' in col_lower:
                    margin_col = col
                elif 'margin' in col_lower and margin_col is None:
                    margin_col = col

            if symbol_col and margin_col:
                for _, row in df.iterrows():
                    symbol = str(row[symbol_col]).upper().strip()
                    # Skip header rows or invalid entries
                    if symbol in ('NAN', 'SCRIP', 'SYMBOL', '') or pd.isna(row[symbol_col]):
                        continue

                    margin = row[margin_col]
                    # Skip header values
                    if isinstance(margin, str) and 'margin' in margin.lower():
                        continue

                    try:
                        # Handle percentage formats
                        if isinstance(margin, str):
                            margin = float(margin.replace('%', '').replace('x', '').strip())
                        else:
                            margin = float(margin)

                        # If value is > 1, assume it's a percentage (e.g., 20 = 20%)
                        if margin > 1:
                            margin = margin / 100

                        if margin > 0:
                            self.mis_margins[symbol] = margin
                            # Also add without exchange prefix
                            if ':' in symbol:
                                base_symbol = symbol.split(':')[-1]
                                self.mis_margins[base_symbol] = margin
                    except (ValueError, TypeError):
                        continue

            print(f"Loaded MIS margins for {len(self.mis_margins)} symbols")
        except Exception as e:
            print(f"Warning: Could not load MIS data: {e}")

    def get_leverage_multiplier(self, symbol: str) -> float:
        """
        Get MIS leverage multiplier for a symbol.

        Multiplier = NRML_MARGIN / MIS_MARGIN
        Example: NRML 50% / MIS 20% = 2.5x leverage
        """
        # Clean symbol
        clean_symbol = symbol.upper().strip()
        if ':' in clean_symbol:
            clean_symbol = clean_symbol.split(':')[-1]
        # Remove _uid suffix if present
        if '_' in clean_symbol:
            clean_symbol = clean_symbol.split('_')[0]

        mis_margin = self.mis_margins.get(clean_symbol, self.NRML_MARGIN_PCT)

        if mis_margin <= 0:
            return 1.0

        return self.NRML_MARGIN_PCT / mis_margin

    def calculate_mis_pnl(self, nrml_pnl: float, symbol: str) -> float:
        """Calculate PnL with MIS leverage applied."""
        multiplier = self.get_leverage_multiplier(symbol)
        return nrml_pnl * multiplier


class TaxCalculator:
    """Calculate income tax on trading profits."""

    # Speculative business income tax rates (FY 2024-25)
    SPECULATIVE_TAX_RATE = 0.30  # 30% for income above 15 lakhs
    CESS_RATE = 0.04  # 4% health and education cess

    @classmethod
    def calculate_tax(cls, net_profit: float, tax_slab: str = 'highest') -> Dict[str, float]:
        """
        Calculate tax on trading profits.

        For intraday trading, profits are classified as "Speculative Business Income"
        and taxed at individual's applicable slab rate.

        Args:
            net_profit: Net profit after fees
            tax_slab: 'highest' assumes 30% bracket, can be 'lowest', 'middle', etc.

        Returns:
            Dict with tax breakdown
        """
        if net_profit <= 0:
            return {
                'taxable_income': 0,
                'base_tax': 0,
                'cess': 0,
                'total_tax': 0,
                'effective_rate': 0,
                'net_after_tax': net_profit
            }

        # Use highest slab rate (30%) for conservative estimate
        # In reality, user should provide their actual slab
        base_tax = net_profit * cls.SPECULATIVE_TAX_RATE
        cess = base_tax * cls.CESS_RATE
        total_tax = base_tax + cess

        effective_rate = total_tax / net_profit if net_profit > 0 else 0

        return {
            'taxable_income': round(net_profit, 2),
            'base_tax': round(base_tax, 2),
            'cess': round(cess, 2),
            'total_tax': round(total_tax, 2),
            'effective_rate': round(effective_rate * 100, 2),
            'net_after_tax': round(net_profit - total_tax, 2)
        }


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
                    except:
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
                    except:
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


def find_mis_file() -> Optional[str]:
    """Find MIS margin file in common locations."""
    # Get project root (parent of tools directory)
    project_root = Path(__file__).parent.parent
    cwd = Path.cwd()

    candidates = [
        # Current working directory
        cwd / 'zerodha_mis_margin.xlsx',
        cwd / 'mis_margins.xlsx',
        # Project root (relative to this script)
        project_root / 'zerodha_mis_margin.xlsx',
        project_root / 'config' / 'zerodha_mis_margin.xlsx',
        project_root / 'data' / 'zerodha_mis_margin.xlsx',
        # Relative path (if cwd is different)
        'zerodha_mis_margin.xlsx',
        'mis_margins.xlsx',
        # Home directory
        Path.home() / 'zerodha_mis_margin.xlsx',
    ]

    for candidate in candidates:
        try:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return str(candidate_path.resolve())
        except Exception:
            continue

    return None


def calculate_net_pnl(backtest_dir: str, mis_file: Optional[str] = None,
                      use_mis: bool = True, verbose: bool = True) -> Dict:
    """
    Calculate net PnL with MIS leverage, fees, and taxation.

    Args:
        backtest_dir: Path to extracted backtest directory
        mis_file: Optional path to Zerodha MIS margin Excel file. If not provided,
                  will search in common locations.
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

    # Find MIS file if not provided
    if use_mis and not mis_file:
        mis_file = find_mis_file()
        if verbose:
            if mis_file:
                print(f"Found MIS file: {mis_file}")
            else:
                print("WARNING: No MIS file found. Will use 1x leverage.")

    if verbose:
        print("=" * 70)
        print("NET PNL CALCULATION WITH MIS, FEES, AND TAXATION")
        print("=" * 70)
        print(f"\nBacktest: {backtest_dir}")
        print(f"Total trades: {trade_count}")
        print(f"Gross PnL (NRML): Rs {gross_pnl_nrml:,.0f}")
        if use_mis:
            print(f"MIS file path: {mis_file}")
            print(f"Pandas available: {PANDAS_AVAILABLE}")

    # Calculate MIS-adjusted PnL
    mis_calc = MISLeverageCalculator(mis_file) if use_mis else None

    if verbose and use_mis and mis_calc:
        print(f"MIS margins loaded: {len(mis_calc.mis_margins)} symbols")

    if use_mis and mis_calc:
        mis_pnl_total = 0
        multipliers = []
        for trade in trades:
            multiplier = mis_calc.get_leverage_multiplier(trade['symbol'])
            mis_pnl = trade['pnl'] * multiplier
            mis_pnl_total += mis_pnl
            multipliers.append(multiplier)

        avg_multiplier = sum(multipliers) / len(multipliers) if multipliers else 1.0
        gross_pnl = mis_pnl_total

        if verbose:
            print(f"\nMIS LEVERAGE:")
            print(f"  Average multiplier: {avg_multiplier:.2f}x")
            print(f"  Gross PnL (with MIS): Rs {gross_pnl:,.0f}")
    else:
        gross_pnl = gross_pnl_nrml
        avg_multiplier = 1.0

    # Calculate fees using actual trade values
    total_fees = 0
    total_buy_value = 0
    total_sell_value = 0
    trades_with_values = 0

    for trade in trades:
        buy_value = trade.get('buy_value', 0)
        sell_value = trade.get('sell_value', 0)

        if buy_value > 0 or sell_value > 0:
            # Use actual trade values
            fees = ZerodhaFeeCalculator.calculate_fees(buy_value, sell_value)
            total_fees += fees['total_fees']
            total_buy_value += buy_value
            total_sell_value += sell_value
            trades_with_values += 1
        else:
            # Fallback: estimate with Rs 25k avg trade value if no actual data
            fallback_value = 25000
            fees = ZerodhaFeeCalculator.calculate_fees(fallback_value, fallback_value)
            total_fees += fees['total_fees']

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
    tax_result = TaxCalculator.calculate_tax(profit_after_fees)

    if verbose:
        print(f"\nTAXATION (Speculative Business Income):")
        print(f"  Taxable income: Rs {tax_result['taxable_income']:,.0f}")
        print(f"  Base tax (30%): Rs {tax_result['base_tax']:,.0f}")
        print(f"  Cess (4%): Rs {tax_result['cess']:,.0f}")
        print(f"  Total tax: Rs {tax_result['total_tax']:,.0f}")
        print(f"  Effective rate: {tax_result['effective_rate']:.1f}%")

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
    # Find default MIS file
    default_mis = find_mis_file()

    parser = argparse.ArgumentParser(
        description='Calculate net PnL with MIS leverage, fees, and taxation'
    )
    parser.add_argument('backtest_dir', help='Path to extracted backtest directory')
    parser.add_argument('--mis-file', help='Path to Zerodha MIS margin Excel file',
                       default=default_mis)
    parser.add_argument('--no-mis', action='store_true',
                       help='Skip MIS leverage calculation')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Use provided file or auto-detected default
    mis_file = args.mis_file
    if mis_file and not Path(mis_file).exists():
        print(f"Warning: MIS file '{mis_file}' not found. Will search for alternatives.")
        mis_file = find_mis_file()

    if not mis_file and not args.no_mis:
        print("Warning: No MIS file found. Using default 1x leverage (NRML).")

    result = calculate_net_pnl(
        args.backtest_dir,
        mis_file=mis_file,
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
