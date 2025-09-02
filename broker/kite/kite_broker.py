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
                  product: Literal['MIS'] = 'MIS',
                  variety: Literal['regular'] = 'regular',
                  validity: Literal['DAY'] = 'DAY',
                  tag: str | None = None) -> str
    - get_ltp(symbol: str) -> float
    - get_ltp_batch(symbols: list[str]) -> dict[str, float]

Requires env OR explicit params: KITE_API_KEY, KITE_ACCESS_TOKEN
Requires: `pip install kiteconnect`
"""
from typing import Dict, List, Optional, Tuple
from config.env_setup import env

from kiteconnect import KiteConnect

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
    ) -> str:
        side_u = side.upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY/SELL, got {side}")
        if order_type.upper() not in ("MARKET", "LIMIT"):
            raise ValueError(f"order_type must be MARKET/LIMIT, got {order_type}")
        if product.upper() != "MIS":
            raise ValueError("Only MIS product supported in intraday mode")
        if variety.lower() != "regular":
            raise ValueError("Only 'regular' variety supported")
        if validity != "DAY":
            raise ValueError("Only DAY validity supported")

        exch, tsym = _split_symbol(symbol)
        txn = self.kc.TRANSACTION_TYPE_BUY if side_u == "BUY" else self.kc.TRANSACTION_TYPE_SELL
        ot = self.kc.ORDER_TYPE_LIMIT if order_type.upper() == "LIMIT" else self.kc.ORDER_TYPE_MARKET
        params = {
            "exchange": exch,
            "tradingsymbol": tsym,
            "transaction_type": txn,
            "quantity": int(qty),
            "product": self.kc.PRODUCT_MIS,
            "order_type": ot,
            "validity": validity,
            "tag": tag or "",
        }
        if ot == self.kc.ORDER_TYPE_LIMIT:
            if price is None:
                raise ValueError("price is required for LIMIT orders")
            params["price"] = float(price)

        resp = self.kc.place_order(variety=self.kc.VARIETY_REGULAR, **params)
        try:
            return str(resp.get("order_id") or resp["data"]["order_id"])  # type: ignore[index]
        except Exception:
            return str(resp)

    # ------------------------------ Market data -------------------------
    def get_ltp(self, symbol: str) -> float:
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
