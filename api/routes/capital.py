"""
Capital management endpoints.

POST /admin/capital - Update capital allocation
POST /admin/mis     - Toggle MIS (margin intraday) mode

These endpoints delegate to CapitalManager for actual changes.
"""

from config.logging_config import get_agent_logger

logger = get_agent_logger()


def register_capital_routes(server):
    """Register capital management routes."""

    @server.post("/admin/capital", auth_required=True)
    def set_capital(ctx, body):
        """
        Update capital allocation.

        Body:
            capital: float - New capital amount

        Returns:
            status: "ok" on success
            old_capital: previous value
            new_capital: new value
        """
        capital = body.get("capital")
        if capital is None:
            return {"error": "Missing 'capital' field"}, 400

        result = _set_capital(ctx.server, float(capital))
        status_code = 200 if "error" not in result else 400
        return result, status_code

    @server.post("/admin/mis", auth_required=True)
    def toggle_mis(ctx, body):
        """
        Toggle MIS (margin intraday) mode.

        Body:
            enabled: bool - Enable or disable MIS

        Returns:
            status: "ok" on success
            old_value: previous value
            new_value: new value
        """
        enabled = body.get("enabled")
        if enabled is None:
            return {"error": "Missing 'enabled' field"}, 400

        result = _toggle_mis(ctx.server, bool(enabled))
        status_code = 200 if "error" not in result else 400
        return result, status_code


def _set_capital(server, capital: float) -> dict:
    """Set capital allocation via CapitalManager."""
    if not server._capital_manager:
        return {"error": "Capital manager not configured"}

    old_capital = server._capital_manager.total_capital
    server._capital_manager.total_capital = capital
    server.increment("user_actions")

    logger.info(f"[API] Capital changed: {old_capital} -> {capital}")
    return {
        "status": "ok",
        "old_capital": old_capital,
        "new_capital": capital
    }


def _toggle_mis(server, enabled: bool) -> dict:
    """Enable/disable MIS mode via CapitalManager."""
    if not server._capital_manager:
        return {"error": "Capital manager not configured"}

    if not hasattr(server._capital_manager, 'mis_enabled'):
        return {"error": "MIS toggle not supported by capital manager"}

    old_value = server._capital_manager.mis_enabled
    server._capital_manager.mis_enabled = enabled
    server.increment("user_actions")

    logger.info(f"[API] MIS mode changed: {old_value} -> {enabled}")
    return {
        "status": "ok",
        "old_value": old_value,
        "new_value": enabled
    }
