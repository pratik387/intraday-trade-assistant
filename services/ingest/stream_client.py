from __future__ import annotations
"""
WebSocket client wrapper for the broker SDK (Kite or Feather via DRY_RUN).

Goals
-----
- Present a small, deterministic API used by SubscriptionManager + ScreenerLive.
- Hide SDK specifics (connect/reconnect, callbacks, batching).
- No config defaults here; caller wires configuration into SubscriptionManager.

Public API (used elsewhere)
---------------------------
ws = WSClient(sdk, on_tick)
ws.on_message(lambda raw: ...)
ws.start()            # non-blocking; spawns reader thread
ws.stop()
ws.subscribe_batch(tokens: list[int])
ws.unsubscribe_batch(tokens: list[int])
ws.set_mode(mode: str, tokens: list[int])  # e.g., "quote" | "ltp" | "full"

Notes
-----
- We pass through raw tick payloads to the router (dicts from the SDK).
- Reconnects: we follow the SDK's built-in reconnect logic if available.
- Threading: background thread owns the SDK event loop; stop() joins it.
"""
import threading
import time
from typing import Callable, Optional, Any, List
from config.logging_config import get_agent_logger

logger = get_agent_logger()

class WSClient:
    """Lightweight wrapper around the broker's ticker/stream.

    Parameters
    ----------
    sdk : Any
        An authenticated broker client that can produce a ticker instance.
        We expect it to expose `create_ticker()` (preferred) or `ticker()` or
        `get_ticker()` returning an object with Zerodha-like methods:
        - .on_ticks(callback)
        - .on_connect(callback)
        - .connect() / .close()
        - .subscribe(list[int]) / .unsubscribe(list[int])
        - .set_mode(mode: str, tokens: list[int])
    """

    def __init__(self, sdk: Any, on_tick: Callable[[dict], None]) -> None:
        self._sdk = sdk
        self._ticker: Any = None
        self._on_message: Optional[Callable[[Any], None]] = None
        self._on_close_cb: Optional[Callable] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.on_tick = on_tick

    # --------------------------- Public Wiring API ---------------------------
    def on_message(self, cb: Callable[[Any], None]) -> None:
        """Register a single raw-message (tick) callback.
        The callback receives whatever payload the SDK emits per tick batch.
        """
        self._on_message = cb

    def on_close(self, cb: Callable) -> None:
        """Register callback for when WS connection closes (replay ends in dry-run)."""
        self._on_close_cb = cb

    # ------------------------------ Lifecycle -------------------------------
    def start(self) -> None:
        """Spawn the reader thread and connect the stream."""
        if self._thread and self._thread.is_alive():
            logger.warning("WSClient.start called twice; ignoring")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="WSClient", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request shutdown and wait for the stream to close."""
        self._stop.set()
        try:
            if self._ticker is not None:
                close = getattr(self._ticker, "close", None)
                if callable(close):
                    close()
        except Exception as e:
            logger.warning(f"WSClient.stop: close failed: {e}")
        if self._thread:
            self._thread.join(timeout=5.0)

    # ------------------------------ Subscriptions ---------------------------
    def subscribe_batch(self, tokens: List[int]) -> None:
        if not tokens or self._ticker is None:
            return
        try:
            self._ticker.subscribe(tokens)
        except Exception as e:
            logger.error(f"WSClient.subscribe_batch failed: {e}")

    def unsubscribe_batch(self, tokens: List[int]) -> None:
        if not tokens or self._ticker is None:
            return
        try:
            self._ticker.unsubscribe(tokens)
        except Exception as e:
            logger.error(f"WSClient.unsubscribe_batch failed: {e}")

    def set_mode(self, mode: str, tokens: List[int]) -> None:
        if not tokens or self._ticker is None:
            return
        try:
            self._ticker.set_mode(mode, tokens)
        except Exception as e:
            logger.error(f"WSClient.set_mode({mode}) failed: {e}")

    # ------------------------------- Internals ------------------------------
    def _run(self) -> None:
        """Thread target: obtain ticker, wire callbacks, connect, loop."""
        try:
            self._ticker = self._sdk.make_ticker()
            self._wire_callbacks(self._ticker)
            logger.info("WSClient connecting...")
            self._connect(self._ticker)
            # Passive loop to keep the thread alive as long as SDK runs
            while not self._stop.is_set():
                time.sleep(0.25)
        except Exception as e:
            logger.exception(f"WSClient thread died: {e}")
        finally:
            try:
                if self._ticker is not None:
                    close = getattr(self._ticker, "close", None)
                    if callable(close):
                        close()
            except Exception:
                pass
            logger.info("WSClient thread exit")

    # -- helpers --
    def _wire_callbacks(self, ticker: Any) -> None:
        # on_connect: useful to log and possibly resubscribe if SDK needs it
        on_connect = getattr(ticker, "on_connect", None)
        if callable(on_connect):
            def _connected(*_args, **_kw):
                logger.info("WS connected")
            on_connect(_connected)

        # on_ticks: Zerodha-style batch callback; pass through to router
        on_ticks = getattr(ticker, "on_ticks", None)
        if callable(on_ticks):
            def _ticks(_ws, ticks):
                if self._on_message:
                    try:
                        for tk in ticks or []:
                            self._on_message(tk)
                    except Exception as e:
                        logger.error(f"on_message callback failed: {e}")
            on_ticks(_ticks)

        # on_close: replay finished or connection lost
        on_close = getattr(ticker, "on_close", None)
        if callable(on_close):
            def _closed(*_args, **_kw):
                logger.warning("WS closed - replay finished or connection lost")
                if self._on_close_cb:
                    try:
                        self._on_close_cb()
                    except Exception as e:
                        logger.error(f"on_close callback failed: {e}")
            on_close(_closed)

    @staticmethod
    def _connect(ticker: Any) -> None:
        try:
            connect = getattr(ticker, "connect", None)
            if not callable(connect):
                raise RuntimeError("WSClient: ticker has no connect()")
            connect()
        except Exception as e:
            raise RuntimeError(f"WSClient: connect() failed: {e}") from e
        
        # --- Compatibility wrappers for SubscriptionManager ---
    def subscribe(self, tokens: list[int]) -> None:
        """Compat: SubscriptionManager expects ws.subscribe(list[int])."""
        try:
            self.subscribe_batch(tokens)
        except Exception as e:
            logger.error(f"WSClient.subscribe failed: {e}")

    def unsubscribe(self, tokens: list[int]) -> None:
        """Compat: SubscriptionManager expects ws.unsubscribe(list[int])."""
        try:
            self.unsubscribe_batch(tokens)
        except Exception as e:
            logger.error(f"WSClient.unsubscribe failed: {e}")
