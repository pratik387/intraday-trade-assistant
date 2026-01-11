"""
Route handlers for the API server.

Each module provides handlers for a specific domain:
- health: Health check endpoints
- status: Status and position monitoring
- exits: Position exit requests
- capital: Capital and MIS management
- trading: Trading control (pause/resume)
"""

from api.routes.health import register_health_routes
from api.routes.status import register_status_routes
from api.routes.exits import register_exit_routes
from api.routes.capital import register_capital_routes
from api.routes.trading import register_trading_routes


def register_all_routes(server):
    """Register all route handlers with the server."""
    register_health_routes(server)
    register_status_routes(server)
    register_exit_routes(server)
    register_capital_routes(server)
    register_trading_routes(server)
