"""
Trading control endpoints.

POST /admin/pause   - Pause new entries (data collection continues)
POST /admin/resume  - Resume taking new entries

When paused:
- Tick processing continues (bars keep building)
- ORB detection continues (levels calculated)
- Existing position monitoring continues
- NO new entries are taken

This allows users to:
- Lock in profits without stopping data collection
- React to news by pausing entries
- Resume trading when conditions improve
"""

from api.state import SessionState
from config.logging_config import get_agent_logger

logger = get_agent_logger()


def register_trading_routes(server):
    """Register trading control routes."""

    @server.post("/admin/pause", auth_required=True)
    def pause_trading(ctx, body):
        """
        Pause new trade entries.

        Data collection and position monitoring continue.
        Use /admin/resume to resume trading.

        Body:
            reason: str (optional) - Reason for pausing

        Returns:
            status: "ok" on success
            state: new session state
            was_trading: whether it was actively trading before
        """
        reason = body.get("reason", "manual_pause")

        was_trading = ctx.server.state == SessionState.TRADING

        if ctx.server.state == SessionState.PAUSED:
            return {
                "status": "ok",
                "message": "Already paused",
                "state": SessionState.PAUSED,
                "was_trading": False
            }, 200

        if ctx.server.state not in [SessionState.TRADING, SessionState.WARMING]:
            return {
                "error": f"Cannot pause from state: {ctx.server.state}",
                "state": ctx.server.state
            }, 400

        ctx.server.set_state(SessionState.PAUSED)
        ctx.server._paused_reason = reason
        ctx.server.increment("user_actions")

        logger.info(f"[API] Trading paused: reason={reason}")
        return {
            "status": "ok",
            "message": "Trading paused - no new entries will be taken",
            "state": SessionState.PAUSED,
            "was_trading": was_trading,
            "reason": reason
        }, 200

    @server.post("/admin/resume", auth_required=True)
    def resume_trading(ctx, body):
        """
        Resume taking new trade entries.

        Body: (none required)

        Returns:
            status: "ok" on success
            state: new session state
            was_paused: whether it was paused before
        """
        was_paused = ctx.server.state == SessionState.PAUSED

        if ctx.server.state == SessionState.TRADING:
            return {
                "status": "ok",
                "message": "Already trading",
                "state": SessionState.TRADING,
                "was_paused": False
            }, 200

        if ctx.server.state != SessionState.PAUSED:
            return {
                "error": f"Cannot resume from state: {ctx.server.state}",
                "state": ctx.server.state
            }, 400

        ctx.server.set_state(SessionState.TRADING)
        ctx.server._paused_reason = None
        ctx.server.increment("user_actions")

        logger.info("[API] Trading resumed")
        return {
            "status": "ok",
            "message": "Trading resumed - new entries enabled",
            "state": SessionState.TRADING,
            "was_paused": was_paused
        }, 200
