"""
UpstoxTickerAdapter — KiteTicker-compatible wrapper around Upstox MarketDataStreamerV3.

Presents the same attribute-assignable callback interface that WSClient expects:
    ticker.on_connect = callback(ws, response)
    ticker.on_ticks   = callback(ws, ticks: list[dict])
    ticker.on_close   = callback(*args, **kwargs)
    ticker.on_error   = callback(*args, **kwargs)
    ticker.subscribe(tokens: list[int])
    ticker.unsubscribe(tokens: list[int])
    ticker.set_mode(mode: str, tokens: list[int])
    ticker.connect()   # blocking
    ticker.close()

Converts Upstox protobuf messages to Zerodha-format tick dicts at the boundary
so that TickRouter, BarBuilder, and all downstream code see no difference.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from config.logging_config import get_agent_logger

# IST as fixed UTC+05:30 offset (for stdlib datetime conversions)
_IST = timezone(timedelta(hours=5, minutes=30))

logger = get_agent_logger()


class UpstoxTickerAdapter:
    """KiteTicker-compatible adapter for Upstox MarketDataStreamerV3."""

    # Attribute-assignable callbacks (WSClient sets these directly)
    on_connect: Optional[Callable] = None
    on_ticks: Optional[Callable] = None
    on_close: Optional[Callable] = None
    on_error: Optional[Callable] = None

    def __init__(
        self,
        access_token: str,
        key_to_int: Dict[str, int],
        int_to_key: Dict[int, str],
        tok_to_sym: Dict[int, str],
    ) -> None:
        self._access_token = access_token
        self._key_to_int = key_to_int   # instrument_key -> int_token
        self._int_to_key = int_to_key   # int_token -> instrument_key
        self._tok_to_sym = tok_to_sym   # int_token -> "NSE:SYMBOL"
        self._streamer: Any = None
        self._stop = threading.Event()
        self._subscribed_keys: set = set()

    def connect(self, **kwargs) -> None:
        """
        Connect to Upstox WebSocket (blocking, like KiteTicker.connect()).
        WSClient._run() calls this in a background thread.
        Accepts **kwargs for compatibility (WSClient passes threaded=True for KiteTicker).
        """
        try:
            import upstox_client
            from upstox_client.rest import ApiException
        except ImportError:
            raise ImportError(
                "upstox-python-sdk not installed. Run: pip install upstox-python-sdk"
            )

        if not self._access_token:
            raise RuntimeError(
                "UpstoxTickerAdapter: UPSTOX_ACCESS_TOKEN not set. "
                "Run broker/upstox/get_upstox_token.py first."
            )

        try:
            # V3 SDK: ApiClient handles auth internally, no separate WS URL fetch
            configuration = upstox_client.Configuration()
            configuration.access_token = self._access_token
            api_client = upstox_client.ApiClient(configuration)

            # MarketDataStreamerV3 handles WS URL, auth, and protobuf decoding
            self._streamer = upstox_client.MarketDataStreamerV3(api_client)

            # Enable auto-reconnect (5s interval, 50 retries)
            self._streamer.auto_reconnect(enable=True, interval=5, retry_count=50)

            # Wire Upstox event emitter callbacks
            self._streamer.on("open", self._on_open)
            self._streamer.on("message", self._on_message)
            self._streamer.on("close", self._on_ws_close)
            self._streamer.on("error", self._on_ws_error)

            logger.info("UPSTOX_WS | Connecting to Upstox WebSocket (v3)...")
            self._streamer.connect()

            # NOTE: on_connect is fired from _on_open() when the socket is actually ready.
            # Do NOT fire it here — connect() returns before the socket is open,
            # and WSClient will flush subscriptions on on_connect which would fail.

            # Block until stop is signaled (matches KiteTicker.connect() behavior)
            while not self._stop.is_set():
                time.sleep(0.5)

        except Exception as e:
            logger.exception(f"UPSTOX_WS | Connection failed: {e}")
            if self.on_error:
                self.on_error(None, e)
            raise

    def close(self) -> None:
        """Disconnect the WebSocket."""
        self._stop.set()
        if self._streamer:
            try:
                self._streamer.disconnect()
            except Exception as e:
                logger.warning(f"UPSTOX_WS | Disconnect error: {e}")
        if self.on_close:
            try:
                self.on_close(None, None)
            except Exception:
                pass

    def subscribe(self, int_tokens: List[int]) -> None:
        """Subscribe to instruments by int token (mapped to Upstox instrument_keys)."""
        if not self._streamer or not int_tokens:
            return

        keys_to_sub = []
        for token in int_tokens:
            key = self._int_to_key.get(token)
            if key and key not in self._subscribed_keys:
                keys_to_sub.append(key)
                self._subscribed_keys.add(key)

        if keys_to_sub:
            try:
                self._streamer.subscribe(keys_to_sub, "full")
                logger.debug(f"UPSTOX_WS | Subscribed {len(keys_to_sub)} instruments (full mode)")
            except Exception as e:
                logger.error(f"UPSTOX_WS | Subscribe failed: {e}")

    def unsubscribe(self, int_tokens: List[int]) -> None:
        """Unsubscribe instruments by int token."""
        if not self._streamer or not int_tokens:
            return

        keys_to_unsub = []
        for token in int_tokens:
            key = self._int_to_key.get(token)
            if key and key in self._subscribed_keys:
                keys_to_unsub.append(key)
                self._subscribed_keys.discard(key)

        if keys_to_unsub:
            try:
                self._streamer.unsubscribe(keys_to_unsub)
                logger.debug(f"UPSTOX_WS | Unsubscribed {len(keys_to_unsub)} instruments")
            except Exception as e:
                logger.error(f"UPSTOX_WS | Unsubscribe failed: {e}")

    def set_mode(self, mode: str, tokens: List[int]) -> None:
        """
        Set subscription mode for tokens.
        Maps Zerodha modes to Upstox modes:
          "full"  -> "full"   (OHLC + depth + LTP)
          "quote" -> "full"   (Upstox 'full' includes quote-level data)
          "ltp"   -> "ltpc"   (LTP + change)
        """
        if not self._streamer or not tokens:
            return

        upstox_mode = "full" if mode in ("full", "quote") else "ltpc"

        keys = [self._int_to_key[t] for t in tokens if t in self._int_to_key]
        if keys:
            try:
                self._streamer.change_mode(keys, upstox_mode)
            except Exception as e:
                logger.debug(f"UPSTOX_WS | set_mode({upstox_mode}) error: {e}")

    # ─── Internal Upstox event handlers ────────────────────────────────────

    def _on_open(self) -> None:
        """Upstox WebSocket opened — fire on_connect so WSClient flushes subscriptions."""
        logger.info("UPSTOX_WS | WebSocket connection opened")
        if self.on_connect:
            self.on_connect(None, {"status": "connected"})

    def _on_message(self, message: Any) -> None:
        """
        Convert Upstox protobuf message to Zerodha-format tick dicts
        and fire on_ticks callback.
        """
        if not self.on_ticks:
            return

        try:
            ticks = self._convert_message_to_ticks(message)
            if ticks:
                self.on_ticks(None, ticks)
        except Exception as e:
            logger.error(f"UPSTOX_WS | Message conversion error: {e}", exc_info=True)

    def _on_ws_close(self) -> None:
        """Upstox WebSocket closed."""
        logger.warning("UPSTOX_WS | WebSocket connection closed")
        if self.on_close:
            self.on_close(None, None)

    def _on_ws_error(self, error: Any) -> None:
        """Upstox WebSocket error."""
        logger.error(f"UPSTOX_WS | WebSocket error: {error}")
        if self.on_error:
            self.on_error(None, error)

    # ─── Message conversion ────────────────────────────────────────────────

    def _convert_message_to_ticks(self, message: Any) -> List[dict]:
        """
        Convert Upstox MarketDataStreamerV3 message to list of Zerodha-format tick dicts.

        Upstox V3 message structure (protobuf decoded):
            message.feeds = {instrument_key: FeedResponse, ...}
            FeedResponse.ff = FullFeed
            FullFeed.market_ff = MarketFullFeed
            MarketFullFeed.ltpc = LTPC (ltp, ltq, ltt, cp)
            MarketFullFeed.market_ohlc = MarketOHLC (ohlc list with interval="1d")

        Zerodha tick dict (what TickRouter expects):
            {
                "instrument_token": int,
                "last_price": float,
                "last_traded_quantity": int,
                "volume_traded": int,       # cumulative day volume
                "last_trade_time": datetime, # IST naive
            }
        """
        ticks = []

        # Handle the protobuf message from MarketDataStreamerV3
        feeds = getattr(message, "feeds", None)
        if not feeds:
            return ticks

        for instrument_key, feed_response in feeds.items():
            int_token = self._key_to_int.get(instrument_key)
            if int_token is None:
                continue

            try:
                ff = getattr(feed_response, "ff", None)
                if ff is None:
                    continue

                market_ff = getattr(ff, "market_ff", None)
                if market_ff is None:
                    continue

                # Extract LTPC (Last Traded Price & Change)
                ltpc = getattr(market_ff, "ltpc", None)
                if ltpc is None:
                    continue

                ltp = getattr(ltpc, "ltp", 0.0)
                ltq = getattr(ltpc, "ltq", 0)
                ltt = getattr(ltpc, "ltt", None)

                # Parse last trade time to IST-naive datetime
                trade_time = None
                if ltt:
                    try:
                        if isinstance(ltt, str):
                            trade_time = datetime.fromisoformat(
                                ltt.replace("Z", "+00:00")
                            )
                            # Convert to IST naive
                            trade_time = trade_time.astimezone(_IST).replace(tzinfo=None)
                        elif isinstance(ltt, (int, float)):
                            # Epoch seconds -> UTC-aware -> IST naive
                            trade_time = datetime.fromtimestamp(ltt, tz=timezone.utc)
                            trade_time = trade_time.astimezone(_IST).replace(tzinfo=None)
                        elif isinstance(ltt, datetime):
                            if ltt.tzinfo:
                                trade_time = ltt.astimezone(_IST).replace(tzinfo=None)
                            else:
                                trade_time = ltt
                    except Exception:
                        trade_time = None

                # Extract cumulative day volume from market OHLC
                cum_volume = 0
                market_ohlc = getattr(market_ff, "market_ohlc", None)
                if market_ohlc:
                    ohlc_list = getattr(market_ohlc, "ohlc", [])
                    for ohlc_item in ohlc_list:
                        interval = getattr(ohlc_item, "interval", "")
                        if interval == "1d":
                            cum_volume = getattr(ohlc_item, "vol", 0) or 0
                            break

                # Build Zerodha-compatible tick dict
                tick = {
                    "instrument_token": int_token,
                    "last_price": float(ltp),
                    "last_traded_quantity": int(ltq),
                    "volume_traded": int(cum_volume),
                }
                if trade_time:
                    tick["last_trade_time"] = trade_time

                ticks.append(tick)

            except Exception as e:
                logger.debug(f"UPSTOX_WS | Tick conversion error for {instrument_key}: {e}")

        return ticks
