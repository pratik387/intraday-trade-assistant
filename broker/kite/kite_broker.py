from __future__ import annotations
"""
STRICT Zerodha broker adapter (REST ONLY) for intraday.
No backward compatibility or hidden defaults.

Single accepted symbol format throughout the app: **"EXCH:TRADINGSYMBOL"**
  Examples: "NSE:RELIANCE", "NFO:BANKNIFTY24AUGFUT"

API (minimal):
  class KiteBroker:
    - __init__(api_key: str | None = None, access_token: str | None = None)
    - place_order(symbol: str, side: Literal['BUY','SELL'], qty: int,
                  order_type: Literal['MARKET','LIMIT'] = 'MARKET',
                  price: float | None = None,
                  product: Literal['MIS','CNC'] = 'MIS',
                  variety: Literal['regular'] = 'regular',
                  validity: Literal['DAY'] = 'DAY',
                  tag: str | None = None,
                  check_margins: bool = True,
                  auto_fallback_cnc: bool = True) -> str
    - check_mis_availability(symbol: str, side: str, qty: int, price: float = None) -> Tuple[bool, str]
    - get_ltp(symbol: str) -> float
    - get_ltp_batch(symbols: list[str]) -> dict[str, float]

Features:
  - MIS margin pre-check using order_margins API
  - Auto-fallback to CNC if MIS rejected
  - Comprehensive error handling

Requires env OR explicit params: KITE_API_KEY, KITE_ACCESS_TOKEN
Requires: `pip install kiteconnect`
"""
from typing import Dict, List, Optional, Tuple
from config.env_setup import env

from kiteconnect import KiteConnect

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

__all__ = ["KiteBroker"]


def _split_symbol(symbol: str) -> Tuple[str, str]:
    s = (symbol or "").strip().upper()
    if ":" not in s:
        raise ValueError(f"Symbol must be 'EXCH:TSYM' (got: {s!r})")
    exch, tsym = s.split(":", 1)
    if not exch or not tsym:
        raise ValueError(f"Invalid symbol: {s!r}")
    return exch, tsym


class KiteBroker:
    def __init__(self, api_key: Optional[str] = None, access_token: Optional[str] = None) -> None:
        self.api_key = env.KITE_API_KEY
        self.access_token = env.KITE_ACCESS_TOKEN
        if not self.api_key or not self.access_token:
            raise RuntimeError("KiteBroker: KITE_API_KEY and KITE_ACCESS_TOKEN are required")
        self.kc = KiteConnect(api_key=self.api_key)
        self.kc.set_access_token(self.access_token)

        # MIS availability cache (symbol -> is_mis_available)
        self._mis_cache: Dict[str, bool] = {}

    # ------------------------------ MIS Availability Check --------------
    def check_mis_availability(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if MIS is available for a stock using order_margins API.

        Returns:
            (is_available, reason)
            - (True, "MIS available") if MIS order can be placed
            - (False, "reason") if MIS not available with reason
        """
        # Check cache first
        if symbol in self._mis_cache:
            cached = self._mis_cache[symbol]
            if cached:
                return True, "MIS available (cached)"
            else:
                return False, "MIS not available (cached)"

        try:
            exch, tsym = _split_symbol(symbol)
            side_u = side.upper()
            txn = self.kc.TRANSACTION_TYPE_BUY if side_u == "BUY" else self.kc.TRANSACTION_TYPE_SELL

            # Prepare order params for margin check
            order_params = [{
                "exchange": exch,
                "tradingsymbol": tsym,
                "transaction_type": txn,
                "variety": "regular",
                "product": "MIS",
                "order_type": "MARKET" if price is None else "LIMIT",
                "quantity": int(qty),
            }]

            if price is not None:
                order_params[0]["price"] = float(price)

            # Call order_margins API
            margins = self.kc.order_margins(order_params)

            # Check if margins were returned successfully
            if margins and len(margins) > 0:
                margin_data = margins[0]

                # Check for errors
                if margin_data.get("status") == "error":
                    error_msg = margin_data.get("message", "Unknown error")
                    logger.warning(f"MIS not available for {symbol}: {error_msg}")
                    self._mis_cache[symbol] = False
                    return False, f"MIS not available: {error_msg}"

                # MIS is available
                required_margin = margin_data.get("total", 0)
                logger.info(f"MIS available for {symbol} | Required margin: Rs.{required_margin:.2f}")
                self._mis_cache[symbol] = True
                return True, f"MIS available (margin: Rs.{required_margin:.2f})"

            else:
                logger.warning(f"MIS check failed for {symbol}: No margin data returned")
                # Don't cache this - might be temporary API issue
                return False, "MIS check failed: No margin data"

        except Exception as e:
            logger.error(f"MIS availability check failed for {symbol}: {e}")
            # Don't cache errors - might be temporary
            return False, f"MIS check error: {str(e)}"

    # ------------------------------ Orders ------------------------------
    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        product: str = "MIS",
        variety: str = "regular",
        validity: str = "DAY",
        tag: Optional[str] = None,
        check_margins: bool = True,
        auto_fallback_cnc: bool = True,
    ) -> str:
        """
        Place order with MIS/CNC product type.

        Args:
            check_margins: If True, check MIS availability before placing MIS order
            auto_fallback_cnc: If True and MIS not available, fallback to CNC automatically

        Returns:
            order_id as string

        Raises:
            ValueError: Invalid parameters
            RuntimeError: Order placement failed
        """
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY/SELL, got {side}")
        if order_type.upper() not in ("MARKET", "LIMIT"):
            raise ValueError(f"order_type must be MARKET/LIMIT, got {order_type}")
        if product.upper() not in ("MIS", "CNC"):
            raise ValueError("Only MIS/CNC products supported")
        if variety.lower() != "regular":
            raise ValueError("Only 'regular' variety supported")
        if validity != "DAY":
            raise ValueError("Only DAY validity supported")

        # Check MIS availability if requested and product is MIS
        actual_product = product.upper()
        if check_margins and actual_product == "MIS":
            is_available, reason = self.check_mis_availability(symbol, side, qty, price)
            if not is_available:
                if auto_fallback_cnc:
                    logger.warning(f"MIS not available for {symbol}, falling back to CNC | Reason: {reason}")
                    actual_product = "CNC"
                else:
                    raise RuntimeError(f"MIS not available for {symbol}: {reason}")

        exch, tsym = _split_symbol(symbol)
        txn = self.kc.TRANSACTION_TYPE_BUY if side_u == "BUY" else self.kc.TRANSACTION_TYPE_SELL
        ot = self.kc.ORDER_TYPE_LIMIT if order_type.upper() == "LIMIT" else self.kc.ORDER_TYPE_MARKET

        # Map product to KiteConnect constant
        if actual_product == "MIS":
            kc_product = self.kc.PRODUCT_MIS
        elif actual_product == "CNC":
            kc_product = self.kc.PRODUCT_CNC
        else:
            raise ValueError(f"Unsupported product: {actual_product}")

        params = {
            "exchange": exch,
            "tradingsymbol": tsym,
            "transaction_type": txn,
            "quantity": int(qty),
            "product": kc_product,
            "order_type": ot,
            "validity": validity,
            "tag": tag or "",
        }
        if ot == self.kc.ORDER_TYPE_LIMIT:
            if price is None:
                raise ValueError("price is required for LIMIT orders")
            params["price"] = float(price)

        try:
            resp = self.kc.place_order(variety=self.kc.VARIETY_REGULAR, **params)
            order_id = str(resp.get("order_id") or resp["data"]["order_id"])  # type: ignore[index]
            logger.info(f"Order placed: {symbol} | {side} {qty} @ {product} | Order ID: {order_id}")
            return order_id
        except Exception as e:
            # If MIS order failed and auto-fallback is enabled, try CNC
            if actual_product == "MIS" and auto_fallback_cnc and "MIS" in str(e):
                logger.warning(f"MIS order failed for {symbol}, retrying with CNC | Error: {e}")
                params["product"] = self.kc.PRODUCT_CNC
                try:
                    resp = self.kc.place_order(variety=self.kc.VARIETY_REGULAR, **params)
                    order_id = str(resp.get("order_id") or resp["data"]["order_id"])  # type: ignore[index]
                    logger.info(f"Order placed with CNC fallback: {symbol} | Order ID: {order_id}")
                    return order_id
                except Exception as e2:
                    logger.error(f"CNC fallback also failed for {symbol}: {e2}")
                    raise RuntimeError(f"Order placement failed (MIS and CNC): {e2}")
            else:
                logger.error(f"Order placement failed for {symbol}: {e}")
                raise RuntimeError(f"Order placement failed: {e}")

    # ------------------------------ Market data -------------------------
    def get_ltp(self, symbol: str, **kwargs) -> float:
        exch, tsym = _split_symbol(symbol)
        key = f"{exch}:{tsym}"
        data = self.kc.ltp([key])
        node = data.get(key) or {}
        return float(node.get("last_price") or 0.0)

    def get_ltp_batch(self, symbols: List[str]) -> Dict[str, float]:
        if not symbols:
            return {}
        keys = []
        for s in symbols:
            exch, tsym = _split_symbol(s)
            keys.append(f"{exch}:{tsym}")
        data = self.kc.ltp(keys)
        out: Dict[str, float] = {}
        for k, v in data.items():
            out[k] = float(v.get("last_price") or 0.0)
        return out
