"""
HTTP API Server for Trading Engine.

Provides a clean routing interface for registering endpoints:
    @server.get("/path")
    def handler(ctx): ...

    @server.post("/path", auth_required=True)
    def handler(ctx, body): ...
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Callable, Dict, List, Any, Tuple
from functools import partial
from queue import Queue, Empty

from config.logging_config import get_agent_logger
from api.state import SessionState

logger = get_agent_logger()


class RequestContext:
    """Context passed to route handlers."""
    def __init__(self, server: "APIServer", headers: dict):
        self.server = server
        self.headers = headers


class APIServer:
    """
    HTTP API server with declarative routing.

    Usage:
        server = APIServer(port=8080)

        @server.get("/status")
        def get_status(ctx):
            return {"status": "ok"}, 200

        @server.post("/exit", auth_required=True)
        def queue_exit(ctx, body):
            return {"queued": True}, 200

        server.start()
    """

    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Route registry: {method: {path: (handler, auth_required)}}
        self._routes: Dict[str, Dict[str, Tuple[Callable, bool]]] = {
            "GET": {},
            "POST": {}
        }

        # State tracking
        self._state = SessionState.INITIALIZING
        self._state_since = datetime.now()
        self._start_time = datetime.now()
        self._last_heartbeat = datetime.now()
        self._paused_reason: Optional[str] = None  # Reason for pause (when PAUSED)

        # External references (set by main.py)
        self._position_store = None
        self._capital_manager = None
        self._ltp_cache = None
        self._kite_client = None  # For broker API calls (funds, etc.)
        self._shutdown_callback: Optional[Callable] = None

        # Auth token (from env var or explicit set)
        self._auth_token: Optional[str] = os.getenv("API_AUTH_TOKEN") or os.getenv("ADMIN_TOKEN")

        # Queue for exit requests - consumed by ExitExecutor
        self._exit_queue: Queue = Queue()

        # Metrics
        self._metrics = {
            "trades_entered": 0,
            "trades_exited": 0,
            "errors": 0,
            "user_actions": 0,
        }
        self._lock = threading.RLock()

    # ==================== Route Registration ====================

    def get(self, path: str, auth_required: bool = False):
        """Decorator to register a GET route."""
        def decorator(handler: Callable):
            self._routes["GET"][path] = (handler, auth_required)
            return handler
        return decorator

    def post(self, path: str, auth_required: bool = False):
        """Decorator to register a POST route."""
        def decorator(handler: Callable):
            self._routes["POST"][path] = (handler, auth_required)
            return handler
        return decorator

    def get_route(self, method: str, path: str) -> Optional[Tuple[Callable, bool]]:
        """Get handler for a route."""
        return self._routes.get(method, {}).get(path)

    # ==================== State Management ====================

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str):
        """Update session state."""
        with self._lock:
            self._state = state
            self._state_since = datetime.now()
        logger.info(f"[API] State: {state}")

    def heartbeat(self):
        """Called periodically to signal process is alive."""
        with self._lock:
            self._last_heartbeat = datetime.now()

    def increment(self, name: str, value: int = 1):
        """Increment a metric counter."""
        with self._lock:
            self._metrics[name] = self._metrics.get(name, 0) + value

    # ==================== External References ====================

    def set_position_store(self, store):
        self._position_store = store

    def set_capital_manager(self, manager):
        self._capital_manager = manager

    def set_ltp_cache(self, cache):
        self._ltp_cache = cache

    def set_kite_client(self, client):
        """Set Kite client for broker API calls (funds, etc.)."""
        self._kite_client = client

    def set_auth_token(self, token: Optional[str]):
        """Set auth token for protected endpoints."""
        if token:
            self._auth_token = token

    def set_shutdown_callback(self, callback: Callable):
        self._shutdown_callback = callback

    # ==================== Auth ====================

    def verify_auth(self, token: Optional[str]) -> bool:
        """Verify auth token. Returns False if no token configured."""
        if not self._auth_token:
            return False
        return token == self._auth_token

    # ==================== Exit Queue (for ExitExecutor) ====================

    def get_pending_exits(self) -> List[Dict[str, Any]]:
        """
        Get all pending exit requests (non-blocking).
        Called by ExitExecutor in its run loop.
        """
        requests = []
        while True:
            try:
                req = self._exit_queue.get_nowait()
                requests.append(req)
            except Empty:
                break
        return requests

    # ==================== Server Lifecycle ====================

    def start(self):
        """Start the API server in a background thread."""
        if self._running:
            return

        # Register routes from route modules
        from api.routes import register_all_routes
        register_all_routes(self)

        # Register shutdown route directly (special case)
        @self.post("/shutdown")
        def shutdown(ctx, body):
            if self._shutdown_callback:
                threading.Thread(target=self._shutdown_callback, daemon=True).start()
                return {"status": "shutdown initiated"}, 200
            return {"error": "Shutdown not configured"}, 500

        try:
            handler = partial(_RequestHandler, self)
            self._server = HTTPServer((self.host, self.port), handler)
            self._running = True

            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="APIServer",
                daemon=True
            )
            self._thread.start()
            auth_status = "enabled" if self._auth_token else "disabled"
            logger.info(f"[API] Server started on http://{self.host}:{self.port} (auth: {auth_status})")
        except OSError as e:
            logger.warning(f"[API] Could not start server on port {self.port}: {e}")

    def stop(self):
        """Stop the API server."""
        if not self._running:
            return
        self._running = False
        if self._server:
            self._server.shutdown()
        logger.info("[API] Server stopped")


class _RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler that routes to registered handlers."""

    def __init__(self, api_server: APIServer, *args, **kwargs):
        self.api_server = api_server
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Auth-Token, X-Admin-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Optional[dict]:
        """Read and parse JSON body from request."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                return {}
            body = self.rfile.read(content_length)
            return json.loads(body.decode('utf-8'))
        except Exception as e:
            logger.warning(f"[API] Failed to parse JSON body: {e}")
            return None

    def _get_auth_token(self) -> Optional[str]:
        """Get auth token from headers (supports both header names)."""
        return self.headers.get('X-Auth-Token') or self.headers.get('X-Admin-Token')

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Auth-Token, X-Admin-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        self._handle_request("GET")

    def do_POST(self):
        self._handle_request("POST")

    def _handle_request(self, method: str):
        try:
            route = self.api_server.get_route(method, self.path)
            if not route:
                self._send_json({"error": "Not found"}, 404)
                return

            handler, auth_required = route

            # Check auth if required
            if auth_required:
                token = self._get_auth_token()
                if not self.api_server.verify_auth(token):
                    self._send_json({"error": "Unauthorized. Provide X-Auth-Token header."}, 401)
                    return

            # Build context
            ctx = RequestContext(self.api_server, dict(self.headers))

            # Call handler
            if method == "GET":
                result = handler(ctx)
            else:  # POST
                body = self._read_json_body()
                if body is None:
                    self._send_json({"error": "Invalid JSON body"}, 400)
                    return
                result = handler(ctx, body)

            # Handle response
            if isinstance(result, tuple):
                data, status = result
            else:
                data, status = result, 200

            self._send_json(data, status)

        except Exception as e:
            logger.exception(f"[API] Error handling {method} {self.path}: {e}")
            self._send_json({"error": str(e)}, 500)


# Singleton
_instance: Optional[APIServer] = None


def get_api_server(port: int = 8080) -> APIServer:
    """Get or create the singleton API server instance."""
    global _instance
    if _instance is None:
        _instance = APIServer(port=port)
    return _instance
