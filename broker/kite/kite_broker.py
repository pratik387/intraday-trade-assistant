from __future__ import annotations
"""
STRICT Zerodha broker adapter (REST ONLY) for intraday.
No backward compatibility or hidden defaults.

Single accepted symbol format throughout the app: **"EXCH:TRADINGSYMBOL"**
  Examples: "NSE:RELIANCE", "NFO:BANKNIFTY24AUGFUT"

API (minimal):
  class KiteBroker:
    - __init__(api_key: str | None = None, access_token: str | None = None, dry_run: bool = False)
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
import time
import uuid
from typing import Dict, List, Optional, Tuple
from config.env_setup import env

# Order tagging prefix for identifying app-placed trades
APP_ORDER_TAG_PREFIX = "ITDA_"  # Intraday Trade Assistant

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
    def __init__(self, api_key: Optional[str] = None, access_token: Optional[str] = None, dry_run: bool = False, ltp_cache=None) -> None:
        self.api_key = env.KITE_API_KEY
        self.access_token = env.KITE_ACCESS_TOKEN
        if not self.api_key or not self.access_token:
            raise RuntimeError("KiteBroker: KITE_API_KEY and KITE_ACCESS_TOKEN are required")
        self.kc = KiteConnect(api_key=self.api_key)
        self.kc.set_access_token(self.access_token)

        # MIS availability cache (symbol -> is_mis_available)
        self._mis_cache: Dict[str, bool] = {}

        # Paper trading mode
        self.dry_run = dry_run
        self._paper_orders: List[Dict] = []  # Log orders in paper trading
        self._paper_order_counter = 0

        # Websocket LTP cache - use for instant price lookups (no rate limit)
        # Falls back to REST API if cache miss
        self._ltp_cache = ltp_cache

        # Rate limiting for LTP REST API calls (Zerodha limit: 3 req/sec)
        # Only used when cache miss (symbol not subscribed to websocket)
        self._ltp_rps = 2.5  # Stay under 3 req/sec limit
        self._ltp_min_interval = 1.0 / self._ltp_rps  # ~0.4s between calls
        self._ltp_last_call = 0.0

        if self.dry_run:
            logger.warning("🧪 PAPER TRADING MODE - Orders will be simulated, not sent to Zerodha")
        else:
            logger.warning("⚠️ LIVE TRADING MODE - Orders will be sent to Zerodha!")

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
        trade_id: Optional[str] = None,
        check_margins: bool = True,
        auto_fallback_cnc: bool = True,
    ) -> str:
        """
        Place order with MIS/CNC product type.

        Args:
            check_margins: If True, check MIS availability before placing MIS order
            auto_fallback_cnc: If True and MIS not available, fallback to CNC automatically
            trade_id: Optional trade ID for tagging (generates ITDA_<12chars> tag)

        Returns:
            order_id as string

        Raises:
            ValueError: Invalid parameters
            RuntimeError: Order placement failed
        """
        # Generate order tag from trade_id for trade identification
        if trade_id and not tag:
            # Use last 12 chars of trade_id (trade IDs are like "NSE:RELIANCE_abc123def456")
            tag = f"{APP_ORDER_TAG_PREFIX}{trade_id[-12:]}"
        elif not tag:
            # Generate random tag if no trade_id provided
            tag = f"{APP_ORDER_TAG_PREFIX}{uuid.uuid4().hex[:12]}"
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

        # PAPER TRADING MODE - Simulate order
        if self.dry_run:
            return self._simulate_order(
                symbol=symbol, side=side_u, qty=qty,
                order_type=order_type, price=price, product=actual_product
            )

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
            # kite.place_order() returns order_id string directly (not a dict)
            # See: https://kite.trade/docs/pykiteconnect/v4/
            order_id = self.kc.place_order(variety=self.kc.VARIETY_REGULAR, **params)
            logger.info(f"Order placed: {symbol} | {side} {qty} @ {product} | Order ID: {order_id}")
            return str(order_id)
        except Exception as e:
            # If MIS order failed, check if we can fallback
            if actual_product == "MIS" and "MIS" in str(e):
                # CNC fallback only works for BUY orders
                # For SELL (short) orders, CNC requires holdings which we don't have
                if side.upper() == "SELL":
                    # Don't log here - caller handles logging appropriately
                    raise RuntimeError(f"MIS blocked for {symbol} - shorting not allowed (no CNC fallback for shorts)")

                # For BUY orders, try CNC fallback if enabled
                if auto_fallback_cnc:
                    logger.warning(f"MIS order failed for {symbol}, retrying with CNC | Error: {e}")
                    params["product"] = self.kc.PRODUCT_CNC
                    try:
                        order_id = self.kc.place_order(variety=self.kc.VARIETY_REGULAR, **params)
                        logger.info(f"Order placed with CNC fallback: {symbol} | Order ID: {order_id}")
                        return str(order_id)
                    except Exception as e2:
                        logger.error(f"CNC fallback also failed for {symbol}: {e2}")
                        raise RuntimeError(f"Order placement failed (MIS and CNC): {e2}")

            logger.error(f"Order placement failed for {symbol}: {e}")
            raise RuntimeError(f"Order placement failed: {e}")

    # ------------------------------ Rate limiting -------------------------
    def _rate_limit_ltp(self) -> None:
        """Enforce rate limit for LTP calls (Zerodha: 3 req/sec)."""
        now = time.monotonic()
        elapsed = now - self._ltp_last_call
        if elapsed < self._ltp_min_interval:
            time.sleep(self._ltp_min_interval - elapsed)
        self._ltp_last_call = time.monotonic()

    # ------------------------------ Market data -------------------------
    def get_ltp(self, symbol: str, **kwargs) -> float:
        # 1. Try websocket cache first (instant, no rate limit)
        if self._ltp_cache:
            cached = self._ltp_cache.get_ltp(symbol)
            if cached is not None:
                return cached

        # 2. Fallback to REST API only if cache miss (symbol not in websocket subscription)
        self._rate_limit_ltp()
        exch, tsym = _split_symbol(symbol)
        key = f"{exch}:{tsym}"
        for attempt in range(3):
            try:
                data = self.kc.ltp([key])
                node = data.get(key) or {}
                return float(node.get("last_price") or 0.0)
            except Exception as e:
                if "Too many requests" in str(e) and attempt < 2:
                    logger.warning(f"Rate limited on LTP call, backing off (attempt {attempt+1})")
                    time.sleep(1.0 + attempt * 0.5)  # Back off: 1s, 1.5s
                    continue
                raise
        return 0.0

    def get_ltp_with_level(self, symbol: str, check_level: Optional[float] = None, **kwargs) -> float:
        """
        Get LTP with optional level checking (for exit executor).

        In live/paper mode, check_level is ignored since tick hook handles instant exits.
        Just returns current LTP from Zerodha API.

        Args:
            symbol: Trading symbol (EXCH:TSYM format)
            check_level: Ignored in live mode (tick hook handles exits)

        Returns:
            Current LTP from Zerodha
        """
        # In live mode, just return current LTP (tick hook handles instant exits)
        return self.get_ltp(symbol, **kwargs)

    def get_ltp_batch(self, symbols: List[str]) -> Dict[str, float]:
        if not symbols:
            return {}
        self._rate_limit_ltp()
        keys = []
        for s in symbols:
            exch, tsym = _split_symbol(s)
            keys.append(f"{exch}:{tsym}")
        for attempt in range(3):
            try:
                data = self.kc.ltp(keys)
                out: Dict[str, float] = {}
                for k, v in data.items():
                    out[k] = float(v.get("last_price") or 0.0)
                return out
            except Exception as e:
                if "Too many requests" in str(e) and attempt < 2:
                    logger.warning(f"Rate limited on LTP batch call, backing off (attempt {attempt+1})")
                    time.sleep(1.0 + attempt * 0.5)
                    continue
                raise
        return {}

    # ------------------------------ Position & Order Queries ----------------
    def get_positions(self) -> Dict:
        """Get all intraday positions from broker."""
        if self.dry_run:
            return {"day": [], "net": []}  # Paper trading has no real positions
        try:
            return self.kc.positions()
        except Exception as e:
            logger.error(f"Failed to fetch broker positions: {e}")
            return {"day": [], "net": []}

    def get_orders(self) -> List[Dict]:
        """Get all orders for today."""
        if self.dry_run:
            return self._paper_orders  # Return simulated orders in paper mode
        try:
            return self.kc.orders()
        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            return []

    def get_app_orders(self) -> List[Dict]:
        """Get only orders placed by this app (tagged with ITDA_)."""
        orders = self.get_orders()
        return [o for o in orders if o.get("tag", "").startswith(APP_ORDER_TAG_PREFIX)]

    def get_order_fill_price(self, order_id: str, max_retries: int = 3, retry_delay: float = 0.5) -> Optional[float]:
        """
        Get actual fill price for an order from broker.

        Polls order status until COMPLETE or max retries reached.
        Returns average_price from broker, or None if order not found/not filled.

        In paper/dry_run mode: Returns immediately from simulated orders (no delay).
        In live mode: Polls with retry delay for async order processing.

        Args:
            order_id: Order ID returned from place_order()
            max_retries: Number of times to poll for fill (default 3)
            retry_delay: Seconds between retries (default 0.5s)

        Returns:
            Actual fill price from broker, or None if unavailable
        """
        import time

        # Paper mode: single lookup, no retries needed (orders are instant)
        if self.dry_run:
            max_retries = 1

        for attempt in range(max_retries):
            orders = self.get_orders()
            for o in orders:
                if str(o.get("order_id")) == str(order_id):
                    status = o.get("status", "").upper()
                    if status == "COMPLETE":
                        avg_price = o.get("average_price")
                        filled_qty = o.get("filled_quantity", 0)
                        if avg_price is not None and avg_price > 0 and filled_qty > 0:
                            return float(avg_price)
                    elif status in ("REJECTED", "CANCELLED"):
                        logger.warning(f"Order {order_id} status: {status}")
                        return None

            # Order not complete yet, wait and retry (live mode only)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        if not self.dry_run:
            logger.warning(f"Could not get fill price for order {order_id} after {max_retries} attempts")
        return None

    # ------------------------------ Paper Trading ---------------------------
    def _simulate_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str,
        price: Optional[float],
        product: str
    ) -> str:
        """
        Simulate order execution in paper trading mode.

        Returns a simulated order ID and logs the order details.
        Does NOT call Zerodha API.
        """
        import pandas as pd

        self._paper_order_counter += 1
        order_id = f"PAPER_{self._paper_order_counter:08d}"

        # Get current market price
        try:
            current_ltp = self.get_ltp(symbol)
        except Exception:
            current_ltp = price or 0.0

        # Simulate realistic fill price with slippage
        fill_price = self._calculate_paper_fill_price(symbol, side, current_ltp)

        # Log the simulated order (format matches Zerodha's orders() response)
        paper_order = {
            'order_id': order_id,
            'timestamp': pd.Timestamp.now(),
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'filled_quantity': qty,
            'order_type': order_type,
            'order_price': price,
            'fill_price': fill_price,
            'average_price': fill_price,  # Match Zerodha format for get_order_fill_price()
            'product': product,
            'status': 'COMPLETE'  # Match Zerodha status (not 'FILLED')
        }
        self._paper_orders.append(paper_order)

        logger.info(
            f"[PAPER] Order simulated: {symbol} | {side} {qty} @ Rs.{fill_price:.2f} "
            f"({product}) | Order ID: {order_id}"
        )

        return order_id

    def _calculate_paper_fill_price(self, symbol: str, side: str, ltp: float) -> float:
        """
        Calculate realistic fill price with slippage based on market conditions.

        Slippage depends on:
        - Market cap (liquidity)
        - Order side (buy pays more, sell gets less)
        """
        # Default slippage: 5 bps (0.05%)
        slippage_bps = 5

        # TODO: Get actual market cap from nse_all.json and adjust
        # large_cap: 2 bps, mid_cap: 5 bps, small_cap: 10 bps, micro_cap: 20 bps

        slippage_amount = ltp * (slippage_bps / 10000)

        if side == "BUY":
            return ltp + slippage_amount  # Pay more when buying
        else:
            return ltp - slippage_amount  # Get less when selling

    def get_paper_trading_report(self) -> dict:
        """Get summary of paper trading orders"""
        if not self.dry_run:
            return {}

        return {
            'total_orders': len(self._paper_orders),
            'orders': self._paper_orders,
            'mode': 'paper_trading'
        }
