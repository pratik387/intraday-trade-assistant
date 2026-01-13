"""
WebSocket server for real-time dashboard updates.

Runs alongside HTTP API on separate port (default 8081).
Broadcasts position updates, LTP changes, closed trades, and status changes
to connected dashboard clients.

Usage:
    ws_server = WebSocketServer(port=8081)
    ws_server.start()

    # Broadcast from any thread:
    ws_server.broadcast("positions", {"positions": [...]})
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Set, Dict, Any, Optional, Callable

try:
    # Use the new asyncio API (websockets >= 13.0)
    from websockets.asyncio.server import serve
    from websockets.exceptions import ConnectionClosed
except ImportError:
    try:
        # Fallback to legacy API for older versions
        from websockets.server import serve
        from websockets.exceptions import ConnectionClosed
    except ImportError:
        # Graceful fallback if websockets not installed
        serve = None  # type: ignore
        ConnectionClosed = Exception

from config.logging_config import get_agent_logger

logger = get_agent_logger()


class WebSocketServer:
    """
    WebSocket server for real-time dashboard updates.

    Thread-safe broadcasting from any thread to all connected clients.
    Runs async event loop in dedicated daemon thread.
    """

    def __init__(self, port: int = 8081, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._clients: Set[Any] = set()  # Set of WebSocket connections
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._stop_event: Optional[asyncio.Event] = None
        self._server = None  # WebSocket server instance

    async def _handler(self, websocket: Any):
        """Handle new WebSocket connection."""
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"[WS] Client connected: {client_info}")

        with self._lock:
            self._clients.add(websocket)

        try:
            # Keep connection alive, handle any incoming messages
            async for message in websocket:
                # Currently no client -> server messages needed
                # Future: could handle subscription requests
                try:
                    data = json.loads(message)
                    logger.debug(f"[WS] Received from {client_info}: {data}")
                except json.JSONDecodeError:
                    pass
        except ConnectionClosed:
            pass
        finally:
            with self._lock:
                self._clients.discard(websocket)
            logger.info(f"[WS] Client disconnected: {client_info}")

    def broadcast(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcast message to all connected clients.

        Thread-safe - can be called from any thread.

        Args:
            event_type: Message type (e.g., "positions", "ltp", "status")
            data: Data payload to send
        """
        with self._lock:
            if not self._clients or not self._loop:
                return
            client_count = len(self._clients)

        message = json.dumps({"type": event_type, "data": data}, default=str)

        # Schedule broadcast on async event loop
        asyncio.run_coroutine_threadsafe(
            self._broadcast_async(message),
            self._loop
        )
        logger.debug(f"[WS] Broadcast {event_type} to {client_count} clients")

    async def _broadcast_async(self, message: str):
        """Send message to all connected clients asynchronously."""
        with self._lock:
            clients = list(self._clients)

        if not clients:
            return

        # Send to all clients, ignore individual failures
        results = await asyncio.gather(
            *[self._safe_send(client, message) for client in clients],
            return_exceptions=True
        )

        # Count failures for logging
        failures = sum(1 for r in results if r is False)
        if failures > 0:
            logger.debug(f"[WS] Broadcast failed for {failures}/{len(clients)} clients")

    async def _safe_send(self, client: Any, message: str) -> bool:
        """Send message to single client, return False on failure."""
        try:
            await client.send(message)
            return True
        except Exception:
            return False

    def start(self):
        """Start WebSocket server in background thread."""
        if serve is None:
            logger.warning("[WS] websockets library not installed - WebSocket server disabled")
            return

        if self._running:
            logger.warning("[WS] Server already running")
            return

        self._thread = threading.Thread(
            target=self._run,
            name="WebSocketServer",
            daemon=True
        )
        self._thread.start()

        # Wait briefly for server to start
        time.sleep(0.1)
        if self._running:
            logger.info(f"[WS] WebSocket server started on ws://{self.host}:{self.port}")

    def _run(self):
        """Run async event loop in dedicated thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_event = asyncio.Event()
        self._running = True

        async def main():
            try:
                self._server = await serve(self._handler, self.host, self.port)
                logger.debug(f"[WS] Server listening on {self.host}:{self.port}")
                # Wait until stop is signaled
                await self._stop_event.wait()
            except OSError as e:
                logger.error(f"[WS] Failed to start server on port {self.port}: {e}")
                self._running = False
            finally:
                # Graceful cleanup
                if self._server:
                    self._server.close()
                    await self._server.wait_closed()

        try:
            self._loop.run_until_complete(main())
        except Exception as e:
            if self._running:  # Only log if not intentional shutdown
                logger.error(f"[WS] Server error: {e}")
        finally:
            self._running = False
            # Clean up the event loop
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()
            self._loop = None

    def stop(self):
        """Stop WebSocket server gracefully."""
        if not self._running:
            return

        self._running = False

        # Signal the server to stop
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)

        # Wait for thread to finish (with timeout)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        logger.info("[WS] WebSocket server stopped")

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        with self._lock:
            return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


class LTPBatcher:
    """
    Batches LTP updates to reduce WebSocket message frequency.

    Collects price updates and broadcasts them in batches every interval_ms.
    Prevents flooding clients with per-tick updates.

    Usage:
        batcher = LTPBatcher(ws_server.broadcast, interval_ms=500)
        batcher.update("NSE:RELIANCE", 2505.50)  # Called on every tick
        # Broadcast happens automatically every 500ms with all pending updates
    """

    def __init__(self, broadcast_fn: Callable[[str, Dict], None], interval_ms: int = 500):
        self._pending: Dict[str, Dict[str, Any]] = {}  # symbol -> {price, ts}
        self._broadcast = broadcast_fn
        self._interval = interval_ms / 1000
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="LTPBatcher", daemon=True)
        self._thread.start()

    def update(self, symbol: str, price: float, timestamp: str = None):
        """Queue an LTP update for batched broadcast."""
        with self._lock:
            self._pending[symbol] = {
                "price": price,
                "ts": timestamp or ""
            }

    def _run(self):
        """Background thread that broadcasts pending updates periodically."""
        while self._running:
            time.sleep(self._interval)

            with self._lock:
                if not self._pending:
                    continue
                batch = self._pending.copy()
                self._pending.clear()

            # Broadcast batched LTP updates
            self._broadcast("ltp_batch", {"prices": batch})

    def stop(self):
        """Stop the batcher thread."""
        self._running = False
