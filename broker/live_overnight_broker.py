"""Composite broker for LIVE overnight trading.

Market DATA comes from Upstox (the paper-validated path); ORDERS, fills, and
GTTs go to Kite. The signal pipeline is byte-identical to paper — only the
order sink changes. Constructed by main.py for `--mode overnight` live runs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


class LiveOvernightBroker:
    def __init__(self, data_sdk: Any, kite: Any) -> None:
        # `_data_sdk` name is load-bearing: overnight_handlers reads
        # broker._data_sdk for the async 5m batch + baseline build.
        self._data_sdk = data_sdk
        self._kite = kite

    # NOTE: fetch_candles is intentionally NOT implemented. Live 5m data comes
    # from Upstox via _data_sdk (get_intraday_5m / async batch); the handler's
    # _get_5m_for_symbol_live Kite-fallback is guarded by hasattr and degrades
    # to a symbol-skip, which is correct for this hybrid (Kite is order-only).

    # ---------------- market data (Upstox) ----------------
    def get_intraday_5m(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._data_sdk.get_intraday_5m(symbol)

    def get_daily(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        return self._data_sdk.get_daily(symbol, days=days)

    def list_symbols(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[str]:
        syms = getattr(self._data_sdk, "_equity_instruments", None)
        if syms:
            return list(syms)
        if hasattr(self._data_sdk, "list_symbols"):
            return self._data_sdk.list_symbols(exchange=exchange, instrument_type=instrument_type)
        return []

    def get_symbol_map(self) -> Dict[str, int]:
        return self._data_sdk.get_symbol_map()

    def resolve_tokens(self, symbols) -> List[int]:
        return self._data_sdk.resolve_tokens(symbols)

    # ---------------- orders / GTT (Kite) ----------------
    def place_order(self, **kwargs) -> str:
        return self._kite.place_order(**kwargs)

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        return self._kite.get_order_status(order_id)

    def cancel_order(self, order_id: str, variety: str = "regular") -> bool:
        # Frees the margin pinned by an unfilled entry BUY (2026-07-09
        # incident) so lower-ranked candidates aren't rejected for funds.
        return self._kite.cancel_order(order_id, variety=variety)

    def get_ltp(self, symbol: str) -> float:
        return self._kite.get_ltp(symbol)

    def get_quote(self, symbol: str) -> Dict[str, float]:
        # Circuit bands + last_price from Kite — used by the marketable-LIMIT
        # pricing to stay tick-valid and inside the circuit band.
        return self._kite.get_quote(symbol)

    def place_gtt_stop(self, **kwargs) -> str:
        return self._kite.place_gtt_stop(**kwargs)

    def cancel_gtt(self, gtt_id: str) -> bool:
        return self._kite.cancel_gtt(gtt_id)
