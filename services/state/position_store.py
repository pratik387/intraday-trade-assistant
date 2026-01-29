"""
PositionStore — thread-safe in-memory position store.

Shared by TriggerAwareExecutor (writer) and ExitExecutor (reader/updater).
Broadcasts position changes to WebSocket clients for real-time dashboard updates.
"""
from __future__ import annotations

import threading
from typing import Optional

from services.execution.trade_executor import Position


class PositionStore:
    """
    Thread-safe in-memory store shared by TriggerAwareExecutor (writer) and ExitExecutor (reader/updater).
    Broadcasts position changes to WebSocket clients for real-time dashboard updates.
    """
    def __init__(self, api_server=None) -> None:
        self._by_sym: dict[str, Position] = {}
        self._lock = threading.RLock()
        self._api_server = api_server

    def set_api_server(self, api_server):
        """Set API server for WebSocket broadcasts."""
        self._api_server = api_server

    def upsert(self, p: Position) -> None:
        with self._lock:
            self._by_sym[p.symbol] = p
        self._broadcast_positions()

    def get(self, sym: str) -> Optional[Position]:
        with self._lock:
            return self._by_sym.get(sym)

    def all(self) -> list[Position]:
        with self._lock:
            return list(self._by_sym.values())

    # --- required by ExitExecutor ---
    def list_open(self) -> dict[str, Position]:
        with self._lock:
            return dict(self._by_sym)

    def close(self, sym: str) -> None:
        with self._lock:
            self._by_sym.pop(sym, None)
        self._broadcast_positions()

    def reduce(self, sym: str, qty_exit: int) -> None:
        """Reduce qty for partial exits; remove if goes to zero."""
        with self._lock:
            p = self._by_sym.get(sym)
            if not p:
                return
            new_qty = int(p.qty) - int(qty_exit)
            if new_qty <= 0:
                self._by_sym.pop(sym, None)
            else:
                p.qty = new_qty
                self._by_sym[sym] = p
        self._broadcast_positions()

    def _broadcast_positions(self):
        """Broadcast current positions to WebSocket clients.
        Excludes shadow trades - they're for internal analysis only.
        """
        if not self._api_server:
            return
        positions = []
        with self._lock:
            for p in self._by_sym.values():
                # Skip shadow trades - simulated positions that don't consume capital
                if hasattr(p, 'plan') and p.plan and p.plan.get("shadow", False):
                    continue
                pos_dict = {
                    "symbol": p.symbol,
                    "side": p.side,
                    "qty": p.qty,
                    "entry": p.avg_price,
                }
                # Include plan data if available
                if hasattr(p, 'plan') and p.plan:
                    plan = p.plan
                    stop_data = plan.get("stop", {})
                    pos_dict["sl"] = stop_data.get("hard") if isinstance(stop_data, dict) else plan.get("sl")
                    targets = plan.get("targets", [])
                    if targets and len(targets) > 0:
                        pos_dict["t1"] = targets[0].get("level")
                    if targets and len(targets) > 1:
                        pos_dict["t2"] = targets[1].get("level")
                    pos_dict["setup"] = plan.get("setup_type", "unknown")
                    pos_dict["entry_time"] = plan.get("entry_ts") or plan.get("trigger_ts")
                    state = plan.get("_state", {})
                    pos_dict["t1_done"] = state.get("t1_done", False)
                    # Include all partial profits: T1 + T2 + manual API partials
                    t1_profit = state.get("t1_profit", 0) or 0
                    t2_profit = state.get("t2_profit", 0) or 0
                    manual_profit = state.get("manual_partial_profit", 0) or 0
                    pos_dict["booked_pnl"] = t1_profit + t2_profit + manual_profit
                positions.append(pos_dict)
        self._api_server.broadcast_ws("positions", {"positions": positions})
