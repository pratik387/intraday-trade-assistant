"""
Capital management endpoints.

GET  /funds         - Get broker account funds (from Kite DMAT)
POST /admin/capital - Update capital allocation
POST /admin/mis     - Toggle MIS (margin intraday) mode

These endpoints delegate to CapitalManager for actual changes.
"""

from config.logging_config import get_agent_logger

logger = get_agent_logger()


def register_capital_routes(server):
    """Register capital management routes."""

    @server.get("/funds")
    def get_funds(ctx):
        """
        Get broker account funds from Kite DMAT.
        Returns available cash, margin, etc.
        No auth required - read-only.
        """
        if not ctx.server._kite_client:
            return {"error": "Kite client not configured", "funds": None}, 200

        try:
            funds = ctx.server._kite_client.get_funds()
            return {"status": "ok", "funds": funds}, 200
        except Exception as e:
            logger.error(f"[API] Failed to get funds: {e}")
            return {"error": str(e), "funds": None}, 200

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

    cm = server._capital_manager
    old_capital = cm.total_capital

    # Calculate margin currently in use
    margin_used = sum(p['margin_used'] for p in cm.positions.values())

    # Update total capital and recalculate available
    cm.total_capital = capital
    cm.available_capital = capital - margin_used

    server.increment("user_actions")

    logger.info(f"[API] Capital changed: {old_capital} -> {capital} (available: {cm.available_capital})")
    return {
        "status": "ok",
        "old_capital": old_capital,
        "new_capital": capital,
        "available_capital": cm.available_capital
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
