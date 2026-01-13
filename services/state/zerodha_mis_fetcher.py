"""
Zerodha MIS Fetcher

Fetches the list of MIS-allowed stocks from Zerodha's public Google Sheets.
Used to validate paper trades - prevents taking trades on stocks that don't
support MIS, which would fail in live trading.

Sheet URL: https://docs.google.com/spreadsheets/d/1fLTsNpFJPK349RTjs0GRSXJZD-5soCUkZt9eSMTJ2m4

Sheet format:
ISIN | Symbol | BSE Symbol | MIS Var+ELM+Adhoc margin | MIS Margin(%) | CO MIS Multiplier | ...
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


@dataclass
class MISInfo:
    """MIS eligibility info for a stock."""
    symbol: str
    margin_pct: float      # 20.00 = 20%
    multiplier: float      # 5 = 5x leverage
    isin: Optional[str] = None


class ZerodhaMISFetcher:
    """
    Fetches and caches MIS-allowed stocks from Zerodha's Google Sheets.

    Usage:
        fetcher = ZerodhaMISFetcher()
        if fetcher.load_from_zerodha():
            print(f"Loaded {fetcher.count()} MIS symbols")

        if fetcher.is_mis_allowed("RELIANCE.NS"):
            leverage = fetcher.get_leverage("RELIANCE.NS")
    """

    SHEET_ID = "1fLTsNpFJPK349RTjs0GRSXJZD-5soCUkZt9eSMTJ2m4"
    CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv"

    def __init__(self):
        self._mis_symbols: Dict[str, MISInfo] = {}  # bare symbol -> MISInfo
        self._loaded: bool = False
        self._load_timestamp: Optional[datetime] = None

    def load_from_zerodha(self, timeout_sec: int = 30) -> bool:
        """
        Fetch MIS list from Zerodha's Google Sheets CSV export.

        Returns:
            True if loaded successfully, False on error
        """
        try:
            logger.info(f"MIS_FETCHER | Fetching from Zerodha sheet...")
            response = requests.get(self.CSV_URL, timeout=timeout_sec)
            response.raise_for_status()

            # Parse CSV
            df = pd.read_csv(io.StringIO(response.text))

            # Expected columns (may vary slightly):
            # ISIN, Symbol, BSE Symbol, MIS Var+ELM+Adhoc margin, MIS Margin(%), CO MIS Multiplier, ...
            required_cols = ["Symbol", "MIS Margin(%)", "CO MIS Multiplier"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.error(f"MIS_FETCHER | Missing columns: {missing}. Available: {list(df.columns)}")
                return False

            # Parse each row
            self._mis_symbols.clear()
            for _, row in df.iterrows():
                symbol = str(row.get("Symbol", "")).strip()
                if not symbol or symbol == "nan":
                    continue

                try:
                    margin_pct = float(row.get("MIS Margin(%)", 0))
                    multiplier = float(row.get("CO MIS Multiplier", 5))
                    isin = str(row.get("ISIN", "")) if "ISIN" in df.columns else None
                except (ValueError, TypeError):
                    continue

                # Store with bare symbol (no .NS suffix)
                self._mis_symbols[symbol.upper()] = MISInfo(
                    symbol=symbol.upper(),
                    margin_pct=margin_pct,
                    multiplier=multiplier,
                    isin=isin if isin and isin != "nan" else None
                )

            self._loaded = True
            self._load_timestamp = datetime.now()
            logger.info(f"MIS_FETCHER | Loaded {len(self._mis_symbols)} MIS-allowed symbols")
            return True

        except requests.RequestException as e:
            logger.error(f"MIS_FETCHER | Network error: {e}")
            return False
        except pd.errors.ParserError as e:
            logger.error(f"MIS_FETCHER | CSV parse error: {e}")
            return False
        except Exception as e:
            logger.error(f"MIS_FETCHER | Unexpected error: {e}")
            return False

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to bare format for lookup (remove .NS, NSE: prefix)."""
        bare = symbol.upper()
        bare = bare.replace(".NS", "").replace("NSE:", "")
        return bare

    def is_mis_allowed(self, symbol: str) -> bool:
        """
        Check if a symbol supports MIS trading.

        Args:
            symbol: Symbol in any format (RELIANCE, RELIANCE.NS, NSE:RELIANCE)

        Returns:
            True if MIS allowed, False if not (or not loaded)
        """
        if not self._loaded:
            return True  # Permissive if not loaded

        bare = self._normalize_symbol(symbol)
        return bare in self._mis_symbols

    def get_mis_info(self, symbol: str) -> Optional[MISInfo]:
        """Get full MIS info for a symbol."""
        bare = self._normalize_symbol(symbol)
        return self._mis_symbols.get(bare)

    def get_leverage(self, symbol: str) -> Optional[float]:
        """
        Get MIS leverage multiplier for a symbol.

        Returns:
            Leverage multiplier (e.g., 5.0 for 5x), or None if not found
        """
        info = self.get_mis_info(symbol)
        return info.multiplier if info else None

    def get_margin_pct(self, symbol: str) -> Optional[float]:
        """
        Get MIS margin percentage for a symbol.

        Returns:
            Margin percentage (e.g., 20.0 for 20%), or None if not found
        """
        info = self.get_mis_info(symbol)
        return info.margin_pct if info else None

    def count(self) -> int:
        """Number of MIS-allowed symbols loaded."""
        return len(self._mis_symbols)

    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._loaded

    def get_load_timestamp(self) -> Optional[datetime]:
        """Get timestamp when data was loaded."""
        return self._load_timestamp

    def get_all_symbols(self) -> list:
        """Get list of all MIS-allowed symbols (bare format)."""
        return list(self._mis_symbols.keys())
