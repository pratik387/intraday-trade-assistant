"""
Health Server for Production Monitoring

Provides HTTP endpoints for:
- Health checks (systemd watchdog, load balancer)
- Status monitoring (positions, P&L, state)
- Remote control (graceful shutdown)
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Callable, Any
from functools import partial

from config.logging_config import get_agent_logger

logger = get_agent_logger()


class SessionState:
    """Trading session lifecycle states."""
    INITIALIZING = "initializing"
    RECOVERING = "recovering"
    WARMING = "warming"
    TRADING = "trading"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"


class HealthServer:
    """
    Lightweight HTTP server for health checks and monitoring.

    Usage:
        health = HealthServer(port=8080)
        health.set_position_store(positions)
        health.start()
    """

    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # State tracking
        self._state = SessionState.INITIALIZING
        self._state_since = datetime.now()
        self._start_time = datetime.now()
        self._last_heartbeat = datetime.now()

        # External references (set by main.py)
        self._position_store = None
        self._capital_manager = None
        self._ltp_cache = None
        self._shutdown_callback: Optional[Callable] = None

        # Metrics
        self._metrics = {
            "trades_entered": 0,
            "trades_exited": 0,
            "errors": 0,
        }
        self._lock = threading.RLock()

    def set_state(self, state: str):
        """Update session state."""
        with self._lock:
            self._state = state
            self._state_since = datetime.now()
        logger.info(f"[HEALTH] State: {state}")

    def set_position_store(self, store):
        self._position_store = store

    def set_capital_manager(self, manager):
        self._capital_manager = manager

    def set_ltp_cache(self, cache):
        self._ltp_cache = cache

    def set_shutdown_callback(self, callback: Callable):
        self._shutdown_callback = callback

    def heartbeat(self):
        """Called periodically to signal process is alive."""
        with self._lock:
            self._last_heartbeat = datetime.now()

    def increment(self, name: str, value: int = 1):
        """Increment a metric counter."""
        with self._lock:
            self._metrics[name] = self._metrics.get(name, 0) + value

    def get_status(self) -> dict:
        """Get full system status."""
        with self._lock:
            now = datetime.now()
            uptime = (now - self._start_time).total_seconds()
            state_duration = (now - self._state_since).total_seconds()
            heartbeat_age = (now - self._last_heartbeat).total_seconds()

            # Position info with current P&L
            positions = []
            total_unrealized_pnl = 0.0

            if self._position_store:
                for pos in self._position_store.all():
                    pos_dict = {
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "qty": pos.qty,
                        "entry": pos.avg_price,
                    }

                    # Get current LTP for P&L calculation
                    if self._ltp_cache:
                        ltp = self._ltp_cache.get_ltp(pos.symbol)
                        if ltp:
                            pos_dict["ltp"] = round(ltp, 2)
                            if pos.side == "BUY":
                                pnl = (ltp - pos.avg_price) * pos.qty
                            else:
                                pnl = (pos.avg_price - ltp) * pos.qty
                            pos_dict["pnl"] = round(pnl, 2)
                            total_unrealized_pnl += pnl

                    # Add targets/stops from plan
                    if hasattr(pos, 'plan') and pos.plan:
                        pos_dict["sl"] = pos.plan.get("sl")
                        pos_dict["t1"] = pos.plan.get("t1")
                        pos_dict["t2"] = pos.plan.get("t2")
                        state = pos.plan.get("_state", {})
                        if state.get("t1_done"):
                            pos_dict["t1_done"] = True

                    positions.append(pos_dict)

            # Capital info
            capital = {}
            if self._capital_manager and self._capital_manager.enabled:
                stats = self._capital_manager.get_stats()
                capital = {
                    "available": round(stats.get("available_capital", 0), 2),
                    "margin_used": round(stats.get("margin_used", 0), 2),
                    "total": round(stats.get("total_capital", 0), 2),
                    "positions": stats.get("positions_count", 0),
                }

            return {
                "status": "ok" if self._state == SessionState.TRADING else self._state,
                "state": self._state,
                "uptime_seconds": int(uptime),
                "state_duration_seconds": int(state_duration),
                "heartbeat_age_seconds": int(heartbeat_age),
                "positions": positions,
                "positions_count": len(positions),
                "unrealized_pnl": round(total_unrealized_pnl, 2),
                "capital": capital,
                "metrics": dict(self._metrics),
                "timestamp": now.isoformat(),
            }

    def start(self):
        """Start the health server in a background thread."""
        if self._running:
            return

        try:
            handler = partial(_HealthHandler, self)
            self._server = HTTPServer((self.host, self.port), handler)
            self._running = True

            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="HealthServer",
                daemon=True
            )
            self._thread.start()
            logger.info(f"[HEALTH] Server started on http://{self.host}:{self.port}")
        except OSError as e:
            logger.warning(f"[HEALTH] Could not start server on port {self.port}: {e}")

    def stop(self):
        """Stop the health server."""
        if not self._running:
            return
        self._running = False
        if self._server:
            self._server.shutdown()
        logger.info("[HEALTH] Server stopped")


class _HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health endpoints."""

    def __init__(self, health_server: HealthServer, *args, **kwargs):
        self.health_server = health_server
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path == "/health" or self.path == "/":
                status = self.health_server.get_status()
                is_healthy = status["state"] in [
                    SessionState.TRADING, SessionState.WARMING, SessionState.RECOVERING
                ]
                self._send_json(
                    {"status": "ok" if is_healthy else "unhealthy", "state": status["state"]},
                    200 if is_healthy else 503
                )

            elif self.path == "/status":
                self._send_json(self.health_server.get_status())

            elif self.path == "/positions":
                status = self.health_server.get_status()
                self._send_json({
                    "positions": status["positions"],
                    "count": status["positions_count"],
                    "unrealized_pnl": status["unrealized_pnl"]
                })

            else:
                self._send_json({"error": "Not found"}, 404)
        except Exception as e:
            logger.exception(f"[HEALTH] Error handling GET {self.path}: {e}")
            self._send_json({"error": str(e)}, 500)

    def do_POST(self):
        if self.path == "/shutdown":
            callback = self.health_server._shutdown_callback
            if callback:
                self._send_json({"status": "shutdown initiated"})
                threading.Thread(target=callback, daemon=True).start()
            else:
                self._send_json({"error": "Shutdown not configured"}, 500)
        else:
            self._send_json({"error": "Not found"}, 404)


# Singleton
_instance: Optional[HealthServer] = None


def get_health_server(port: int = 8080) -> HealthServer:
    global _instance
    if _instance is None:
        _instance = HealthServer(port=port)
    return _instance
