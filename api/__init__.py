"""
API module for Trading Engine.

Provides HTTP endpoints for monitoring and control:
- Health checks (systemd watchdog, load balancer)
- Status monitoring (positions, P&L, state)
- Position exits (queued for ExitExecutor)
- Capital management (via CapitalManager)

Usage:
    from api import get_api_server, SessionState

    server = get_api_server(port=8080)
    server.set_position_store(positions)
    server.set_capital_manager(capital_manager)
    server.start()
"""

from api.server import APIServer, get_api_server
from api.state import SessionState

__all__ = ["APIServer", "get_api_server", "SessionState"]
