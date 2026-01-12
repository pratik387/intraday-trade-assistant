"""
Status and position monitoring endpoints.

GET /status    - Full system status with positions, P&L, metrics
GET /positions - Open positions with current P&L
"""

from datetime import datetime
from api.state import SessionState


def register_status_routes(server):
    """Register status monitoring routes."""

    @server.get("/status")
    def get_status(ctx):
        """
        Get full system status.
        Includes state, positions, P&L, capital, and metrics.
        """
        return _build_status(ctx.server), 200

    @server.get("/positions")
    def get_positions(ctx):
        """
        Get open positions with current P&L.
        Lighter endpoint than /status for position-only queries.
        """
        status = _build_status(ctx.server)
        return {
            "positions": status["positions"],
            "count": status["positions_count"],
            "unrealized_pnl": status["unrealized_pnl"]
        }, 200

    @server.get("/closed")
    def get_closed_trades(ctx):
        """
        Get closed trades for this session.
        Returns list of trades that have been exited.
        """
        closed = ctx.server.get_closed_trades()
        total_pnl = sum(t.get("pnl", 0) for t in closed)
        winners = sum(1 for t in closed if t.get("pnl", 0) > 0)
        losers = sum(1 for t in closed if t.get("pnl", 0) < 0)
        return {
            "trades": closed,
            "count": len(closed),
            "total_pnl": round(total_pnl, 2),
            "winners": winners,
            "losers": losers,
            "win_rate": round(winners / len(closed) * 100, 1) if closed else 0
        }, 200


def _build_status(server) -> dict:
    """Build full status response."""
    now = datetime.now()
    uptime = (now - server._start_time).total_seconds()
    state_duration = (now - server._state_since).total_seconds()
    heartbeat_age = (now - server._last_heartbeat).total_seconds()

    # Position info with current P&L
    positions = []
    total_unrealized_pnl = 0.0

    if server._position_store:
        for pos in server._position_store.all():
            pos_dict = {
                "symbol": pos.symbol,
                "side": pos.side,
                "qty": pos.qty,
                "entry": pos.avg_price,
            }

            # Get current LTP for P&L calculation
            if server._ltp_cache:
                ltp = server._ltp_cache.get_ltp(pos.symbol)
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
    if server._capital_manager and server._capital_manager.enabled:
        stats = server._capital_manager.get_stats()
        capital = {
            "available": round(stats.get("available_capital", 0), 2),
            "margin_used": round(stats.get("margin_used", 0), 2),
            "total": round(stats.get("total_capital", 0), 2),
            "positions": stats.get("positions_count", 0),
            "mis_enabled": getattr(server._capital_manager, 'mis_enabled', False),
        }

    # Pause state info
    is_paused = server.state == SessionState.PAUSED
    paused_reason = server._paused_reason if is_paused else None

    return {
        "status": "ok" if server.state == SessionState.TRADING else server.state,
        "state": server.state,
        "paused": is_paused,
        "paused_reason": paused_reason,
        "uptime_seconds": int(uptime),
        "state_duration_seconds": int(state_duration),
        "heartbeat_age_seconds": int(heartbeat_age),
        "positions": positions,
        "positions_count": len(positions),
        "unrealized_pnl": round(total_unrealized_pnl, 2),
        "capital": capital,
        "metrics": dict(server._metrics),
        "pending_exit_requests": server._exit_queue.qsize(),
        "timestamp": now.isoformat(),
        "auth_enabled": bool(server._auth_token),
    }
