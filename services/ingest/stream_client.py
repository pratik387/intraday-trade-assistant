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

        # Buffer for subscriptions before WebSocket connects
        self._connected = threading.Event()
        self._pending_subscriptions: List[List[int]] = []
        self._pending_lock = threading.Lock()

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
        if not tokens:
            return

        # If WebSocket not connected yet, buffer the subscription
        if not self._connected.is_set():
            with self._pending_lock:
                self._pending_subscriptions.append(tokens)
            return

        # WebSocket is connected, send subscription
        if self._ticker is None:
            logger.error(f"WSClient: Ticker is None, dropping {len(tokens)} subscriptions")
            return

        try:
            self._ticker.subscribe(tokens)
        except Exception as e:
            logger.error(f"WSClient: Subscribe failed for {len(tokens)} tokens: {e}")

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
            except Exception as e:
                logger.warning(f"WSClient: Error closing ticker: {e}")

    # -- helpers --
    def _wire_callbacks(self, ticker: Any) -> None:
        # KiteTicker uses attribute assignment pattern, not method calls
        # e.g., ticker.on_connect = callback (not ticker.on_connect(callback))

        # on_connect: useful to log and possibly resubscribe if SDK needs it
        if hasattr(ticker, "on_connect"):
            def _connected(ws, response):
                logger.info("[DEBUG] on_connect callback started")
                logger.info("WebSocket connected")

                # Mark as connected
                logger.info("[DEBUG] Setting connected event...")
                self._connected.set()
                logger.info("[DEBUG] Connected event set")

                # Flush all buffered subscriptions
                logger.info("[DEBUG] Acquiring pending lock...")
                with self._pending_lock:
                    logger.info("[DEBUG] Lock acquired, checking pending subscriptions...")
                    if self._pending_subscriptions:
                        total_tokens = sum(len(batch) for batch in self._pending_subscriptions)
                        logger.info(f"Flushing {total_tokens} buffered subscription tokens...")

                        for batch in self._pending_subscriptions:
                            try:
                                self._ticker.subscribe(batch)
                            except Exception as e:
                                logger.error(f"Failed to subscribe buffered tokens: {e}")

                        self._pending_subscriptions.clear()
                        logger.info("All buffered subscriptions sent")
                    else:
                        logger.info("[DEBUG] No pending subscriptions to flush")

                logger.info("[DEBUG] on_connect callback completed")

            ticker.on_connect = _connected
        else:
            logger.warning("Ticker has no on_connect attribute")

        # on_ticks: Zerodha-style batch callback; pass through to router
        if hasattr(ticker, "on_ticks"):
            def _ticks(_ws, ticks):
                if self._on_message:
                    try:
                        for tk in ticks or []:
                            self._on_message(tk)
                    except Exception as e:
                        logger.error(f"Tick processing failed: {e}", exc_info=True)
                else:
                    logger.warning(f"on_ticks fired but _on_message is None! Dropping {len(ticks) if ticks else 0} ticks")
            ticker.on_ticks = _ticks
        else:
            logger.warning("Ticker has no on_ticks attribute")

        # on_close: replay finished or connection lost
        if hasattr(ticker, "on_close"):
            def _closed(*_args, **_kw):
                logger.warning("WebSocket disconnected")
                # Clear connected flag
                self._connected.clear()
                if self._on_close_cb:
                    try:
                        self._on_close_cb()
                    except Exception as e:
                        logger.error(f"on_close callback failed: {e}")
            ticker.on_close = _closed
        else:
            logger.warning("Ticker has no on_close attribute")

        # on_error: capture WebSocket errors
        if hasattr(ticker, "on_error"):
            def _error(*_args, **_kw):
                logger.error(f"WebSocket error: {_args} {_kw}")
            ticker.on_error = _error
        else:
            logger.warning("Ticker has no on_error attribute")

    @staticmethod
    def _connect(ticker: Any) -> None:
        try:
            connect = getattr(ticker, "connect", None)
            if not callable(connect):
                raise RuntimeError("WSClient: ticker has no connect()")
            connect()
        except Exception as e:
            logger.exception(f"WebSocket connect() failed: {e}", exc_info=True)
            raise RuntimeError(f"WSClient: connect() failed: {e}") from e
        
        # --- Compatibility wrappers for SubscriptionManager ---
    def subscribe(self, tokens: list[int]) -> None:
        """Compat: SubscriptionManager expects ws.subscribe(list[int])."""
        try:
            self.subscribe_batch(tokens)
        except Exception as e:
            logger.exception(f"WSClient.subscribe failed: {e}")

    def unsubscribe(self, tokens: list[int]) -> None:
        """Compat: SubscriptionManager expects ws.unsubscribe(list[int])."""
        try:
            self.unsubscribe_batch(tokens)
        except Exception as e:
            logger.error(f"WSClient.unsubscribe failed: {e}")
