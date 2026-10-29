"""Symbols blocked from NEW intraday entries for the rest of the process (= the day).

Set when the system is unsure what the account holds in a symbol: an entry whose
outcome could not be read, an exit that flipped the position, a restart that found
the broker on the other side or an app position with no record. Trading on top of
an unknown position is how a mistake becomes two. Exits are never blocked.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

_blocked: Dict[str, str] = {}
_all: Optional[str] = None
_lock = threading.Lock()


def block(symbol: str, reason: str) -> None:
    with _lock:
        _blocked.setdefault(symbol, reason)


def block_all(reason: str) -> None:
    """No new entries in ANY symbol for the rest of the day (daily loss stop)."""
    global _all
    with _lock:
        if _all is None:
            _all = reason


def blocked_reason(symbol: str) -> Optional[str]:
    with _lock:
        return _all or _blocked.get(symbol)


def clear_all() -> None:
    """Tests only."""
    global _all
    with _lock:
        _blocked.clear()
        _all = None
