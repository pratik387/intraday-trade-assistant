from __future__ import annotations
"""
TickRouter — normalizes broker tick packets → BarBuilder.on_tick(symbol, price, volume, ts)

Usage in ScreenerLive:
    self.router = TickRouter(on_tick=self.agg.on_tick,
                             token_to_symbol=self.sdk.token_to_symbol_map())
    self.ws.on_message(self.router.handle_raw)

Contracts
- `symbol` is always "EXCH:TRADINGSYMBOL" (e.g., "NSE:RELIANCE") in the callback
- `price` is float LTP
- `volume` is *per-tick* traded quantity (not cumulative day volume). If not present,
  we pass 0.0 and let the bar builder accumulate only on known trade prints.
- `ts` is IST-naive datetime (no tzinfo). We convert if packet has tz-aware time.

This module is broker-agnostic but ships with Zerodha Kite-friendly parsing.
Kite tick usually contains keys: instrument_token, last_price, last_traded_quantity,
last_trade_time (tz-aware), volume (cumulative), ohlc, etc.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

try:  # Python 3.9+
    from zoneinfo import ZoneInfo  # type: ignore
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    _IST = None  # fallback: keep whatever tz or naive is provided

logger = logging.getLogger(__name__)

OnTick = Callable[[str, float, float, datetime], None]


@dataclass
class _Maps:
    token_to_symbol: Dict[int, str]


def _to_ist_naive(ts: datetime) -> datetime:
    if ts is None:
        return datetime.now()
    try:
        if ts.tzinfo is not None and _IST is not None:
            return ts.astimezone(_IST).replace(tzinfo=None)
        return ts.replace(tzinfo=None)
    except Exception:
        return ts


class TickRouter:
    """Parse raw tick payloads and emit normalized callbacks to the bar builder.

    You can update the token→symbol map at runtime if your subscription set changes.
    """

    def __init__(
        self,
        *,
        on_tick: Optional[OnTick] = None,
        token_to_symbol: Optional[Mapping[int, str]] = None,
    ) -> None:
        self._on_tick: Optional[OnTick] = on_tick
        self._maps = _Maps(token_to_symbol=dict(token_to_symbol or {}))

        # metrics
        self._miss_no_map = 0
        self._miss_bad_pkt = 0

    # ----------------------------- wiring ---------------------------------
    def set_on_tick(self, cb: OnTick) -> None:
        self._on_tick = cb

    def update_token_map(self, mapping: Mapping[int, str]) -> None:
        self._maps.token_to_symbol = dict(mapping)

    # ----------------------------- routing --------------------------------
    def handle_raw(self, raw: Any) -> None:
        """Entry from WSClient.on_message(raw). Accepts list[dict] | dict | tuple.
        Silently skips malformed packets. Logs only on first few misses to avoid noise.
        """
        try:
            if raw is None:
                return
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    self._handle_one(item)
            elif isinstance(raw, dict):
                self._handle_one(raw)
            else:
                # unknown shape (string/binary) — ignore quietly
                self._miss_bad_pkt += 1
                if self._miss_bad_pkt < 5:
                    logger.debug("tick_router: unexpected payload type=%s", type(raw))
        except Exception as e:  # pragma: no cover
            logger.debug("tick_router: handle_raw error: %s", e)

    # ----------------------------- parsers --------------------------------
    def _handle_one(self, pkt: Any) -> None:
        if pkt is None:
            return
        try:
            if isinstance(pkt, dict):
                self._handle_dict(pkt)
            elif isinstance(pkt, (list, tuple)) and len(pkt) >= 3:
                # (token, ltp, qty, [ts]) — generic tuple support
                token = int(pkt[0])
                price = float(pkt[1])
                qty = float(pkt[2]) if pkt[2] is not None else 0.0
                ts = _to_ist_naive(pkt[3]) if len(pkt) >= 4 and isinstance(pkt[3], datetime) else datetime.now()
                self._emit(token, price, qty, ts)
            else:
                self._miss_bad_pkt += 1
                if self._miss_bad_pkt < 5:
                    logger.debug("tick_router: bad pkt shape=%s", type(pkt))
        except Exception as e:  # pragma: no cover
            self._miss_bad_pkt += 1
            if self._miss_bad_pkt < 5:
                logger.debug("tick_router: _handle_one error: %s", e)

    def _handle_dict(self, t: Dict[str, Any]) -> None:
        # Zerodha Kite style
        token = t.get("instrument_token") or t.get("token") or t.get("tradable")
        if token is None:
            self._miss_bad_pkt += 1
            return
        try:
            token = int(token)
        except Exception:
            self._miss_bad_pkt += 1
            return

        price = t.get("last_price") or t.get("ltp") or t.get("last_traded_price")
        if price is None:
            return  # nothing to do
        try:
            price = float(price)
        except Exception:
            return

        # Prefer per-trade quantity if available; fall back to 0
        qty = t.get("last_traded_quantity") or t.get("last_quantity") or 0.0
        try:
            qty = float(qty)
        except Exception:
            qty = 0.0

        ts = t.get("last_trade_time") or t.get("exchange_timestamp") or datetime.now()
        if isinstance(ts, datetime):
            ts = _to_ist_naive(ts)
        else:
            ts = datetime.now()

        self._emit(token, price, qty, ts)

    # ----------------------------- emit -----------------------------------
    def _emit(self, token: int, price: float, qty: float, ts: datetime) -> None:
        if self._on_tick is None:
            return
        sym = self._maps.token_to_symbol.get(int(token))
        if not sym:
            self._miss_no_map += 1
            if self._miss_no_map <= 3:
                logger.debug("tick_router: no symbol map for token=%s", token)
            return
        try:
            self._on_tick(sym, price, qty, ts)
        except Exception as e:  # pragma: no cover
            logger.debug("tick_router: on_tick callback failed: %s", e)
