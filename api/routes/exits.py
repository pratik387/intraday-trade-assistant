"""
Position exit endpoints.

POST /admin/exit      - Queue exit request for a single position
POST /admin/exit-all  - Queue exit requests for all open positions

Exit requests are queued and processed by ExitExecutor to ensure
proper handling (logging, persistence, capital release).
"""

from datetime import datetime

from config.logging_config import get_agent_logger

logger = get_agent_logger()


def register_exit_routes(server):
    """Register exit routes."""

    @server.post("/admin/exit", auth_required=True)
    def queue_exit(ctx, body):
        """
        Queue an exit request for ExitExecutor to process.

        Body:
            symbol: str - Position symbol to exit
            qty: int (optional) - Quantity to exit, None for full exit
            reason: str (optional) - Exit reason for logging

        Returns:
            status: "queued" on success
            error: message on failure
        """
        symbol = body.get("symbol")
        if not symbol:
            return {"error": "Missing 'symbol' field"}, 400

        qty = body.get("qty")  # None means full exit
        reason = body.get("reason", "manual_exit")

        result = _queue_exit_request(ctx.server, symbol, qty, reason)
        status_code = 200 if "error" not in result else 400
        return result, status_code

    @server.post("/admin/exit-all", auth_required=True)
    def queue_exit_all(ctx, body):
        """
        Queue exit requests for all open positions.

        Body:
            reason: str (optional) - Exit reason for logging

        Returns:
            status: "queued" on success
            queued: list of symbols queued
            count: number of positions queued
        """
        reason = body.get("reason", "manual_exit_all")
        result = _queue_exit_all(ctx.server, reason)
        status_code = 200 if "error" not in result else 400
        return result, status_code


def _queue_exit_request(server, symbol: str, qty=None, reason: str = "manual_exit") -> dict:
    """Queue a single exit request."""
    if not server._position_store:
        return {"error": "Position store not configured"}

    position = server._position_store.get(symbol)
    if not position:
        return {"error": f"No position found for {symbol}"}

    # Validate qty: must be None (full exit) or positive integer
    if qty is not None:
        if qty <= 0:
            return {"error": f"Exit qty must be positive, got {qty}"}
        if qty > position.qty:
            return {"error": f"Exit qty {qty} exceeds position qty {position.qty}"}

    request = {
        "symbol": symbol,
        "qty": qty,  # None means full exit
        "reason": reason,
        "requested_at": datetime.now().isoformat()
    }
    server._exit_queue.put(request)
    server.increment("user_actions")

    logger.info(f"[API] Exit request queued: {symbol} qty={qty or 'full'} reason={reason}")
    return {
        "status": "queued",
        "symbol": symbol,
        "qty": qty or position.qty,
        "message": "Exit request queued for processing"
    }


def _queue_exit_all(server, reason: str = "manual_exit_all") -> dict:
    """Queue exit requests for all open positions."""
    if not server._position_store:
        return {"error": "Position store not configured"}

    positions = server._position_store.all()
    if not positions:
        return {"status": "ok", "message": "No positions to exit", "queued": []}

    queued = []
    for pos in positions:
        request = {
            "symbol": pos.symbol,
            "qty": None,  # Full exit
            "reason": reason,
            "requested_at": datetime.now().isoformat()
        }
        server._exit_queue.put(request)
        queued.append(pos.symbol)

    server.increment("user_actions")
    logger.info(f"[API] Exit-all queued: {len(queued)} positions, reason={reason}")

    return {
        "status": "queued",
        "reason": reason,
        "queued": queued,
        "count": len(queued)
    }
