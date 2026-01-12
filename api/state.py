"""
Session state management for the trading engine.
"""


class SessionState:
    """Trading session lifecycle states."""
    INITIALIZING = "initializing"
    RECOVERING = "recovering"
    WARMING = "warming"
    TRADING = "trading"
    PAUSED = "paused"  # No new entries, data collection continues
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"
