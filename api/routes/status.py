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
        Excludes shadow trades (simulated trades for analysis only).
        """
        all_closed = ctx.server.get_closed_trades()
        # Filter out shadow trades - they're for internal analysis only
        closed = [t for t in all_closed if not t.get("shadow", False)]
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
    """Build full status response. Excludes shadow trades from dashboard."""
    now = datetime.now()
    uptime = (now - server._start_time).total_seconds()
    state_duration = (now - server._state_since).total_seconds()
    heartbeat_age = (now - server._last_heartbeat).total_seconds()

    # Position info with current P&L
    # Filter out shadow trades - they're for internal analysis only
    positions = []
    total_unrealized_pnl = 0.0

    if server._position_store:
        for pos in server._position_store.all():
            # Skip shadow trades - simulated positions that don't consume capital
            if hasattr(pos, 'plan') and pos.plan and pos.plan.get("shadow", False):
                continue
            pos_dict = {
                "symbol": pos.symbol,
                "side": pos.side,
                "qty": pos.qty,
                "entry": pos.avg_price,
            }

            # Get current LTP for P&L calculation
            if server._ltp_cache:
                # Try both with and without NSE: prefix since tick router may use different format
                ltp = server._ltp_cache.get_ltp(pos.symbol)
                if ltp is None and pos.symbol.startswith("NSE:"):
                    # Try without prefix
                    ltp = server._ltp_cache.get_ltp(pos.symbol.replace("NSE:", ""))
                if ltp is None and not pos.symbol.startswith("NSE:"):
                    # Try with prefix
                    ltp = server._ltp_cache.get_ltp(f"NSE:{pos.symbol}")
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
                plan = pos.plan
                # SL is stored under stop.hard
                stop_data = plan.get("stop", {})
                if isinstance(stop_data, dict):
                    pos_dict["sl"] = stop_data.get("hard")
                else:
                    pos_dict["sl"] = plan.get("sl")

                # Targets are stored in targets array
                targets = plan.get("targets", [])
                if targets and len(targets) > 0:
                    pos_dict["t1"] = targets[0].get("level")
                else:
                    pos_dict["t1"] = plan.get("t1")

                if targets and len(targets) > 1:
                    pos_dict["t2"] = targets[1].get("level")
                else:
                    pos_dict["t2"] = plan.get("t2")

                state = plan.get("_state", {})
                if state.get("t1_done"):
                    pos_dict["t1_done"] = True
                    # Only full exit available after T1 taken
                    pos_dict["exit_options"] = ["full"]
                    # Add booked PnL and T1 exit time if available
                    pos_dict["booked_pnl"] = state.get("t1_profit", 0)
                    pos_dict["t1_exit_time"] = state.get("t1_exit_time")
                else:
                    # Both 50% and full exit available
                    pos_dict["exit_options"] = ["partial", "full"]
                    pos_dict["booked_pnl"] = 0

                # Entry time from plan
                pos_dict["entry_time"] = plan.get("entry_ts") or plan.get("trigger_ts")

                # Setup type for display
                pos_dict["setup"] = plan.get("setup_type", "unknown")

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
