"""
Health check endpoints.

GET /health - Basic health check for load balancers/systemd
GET /       - Alias for /health
"""

from api.state import SessionState


def register_health_routes(server):
    """Register health check routes."""

    @server.get("/health")
    @server.get("/")
    def health_check(ctx):
        """
        Basic health check endpoint.
        Returns 200 if healthy, 503 if unhealthy.
        """
        state = ctx.server.state
        is_healthy = state in [
            SessionState.TRADING,
            SessionState.WARMING,
            SessionState.RECOVERING,
            SessionState.PAUSED  # Paused is healthy - data collection continues
        ]
        return {
            "status": "ok" if is_healthy else "unhealthy",
            "state": state
        }, 200 if is_healthy else 503
