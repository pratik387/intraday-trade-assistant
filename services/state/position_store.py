"""
PositionStore — thread-safe in-memory position store.

Shared by TriggerAwareExecutor (writer) and ExitExecutor (reader/updater).
Broadcasts position changes to WebSocket clients for real-time dashboard updates.

Keyed by trade_id (not symbol) so wide_open_mode capture runs can hold multiple
concurrent open positions on the same symbol without one overwriting another.
Each plan carries its own trade_id from the orchestrator. Smoke 23 evidence:
keying by symbol caused 88% of orb_15 / 78% of narrow_cpr_breakout positions to
be silently overwritten by later same-symbol setups, leaving them untracked
through SL/T1/T2/EOD-flat.
"""
from __future__ import annotations

import threading
from typing import Optional

from services.execution.models import Position


def _trade_id_of(p: Position) -> str:
    """Extract trade_id from a Position. Plans always carry one (orchestrator
    sets `trade_id` in the plan). Fall back to symbol-only key as a last resort
    so a missing trade_id reverts to legacy behaviour rather than crashing."""
    tid = (p.plan or {}).get("trade_id") if hasattr(p, "plan") else None
    return str(tid) if tid else f"sym:{p.symbol}"


class PositionStore:
    """
    Thread-safe in-memory store shared by TriggerAwareExecutor (writer) and ExitExecutor (reader/updater).
    Broadcasts position changes to WebSocket clients for real-time dashboard updates.
    """
    def __init__(self, api_server=None) -> None:
        # Keyed by trade_id so multiple concurrent positions on the same symbol
        # (sub7/sub8 wide_open_mode) coexist without overwrite.
        self._by_tid: dict[str, Position] = {}
        self._lock = threading.RLock()
        self._api_server = api_server

    def set_api_server(self, api_server):
        """Set API server for WebSocket broadcasts."""
        self._api_server = api_server

    def upsert(self, p: Position) -> None:
        with self._lock:
            self._by_tid[_trade_id_of(p)] = p
        self._broadcast_positions()

    def get_by_trade_id(self, trade_id: str) -> Optional[Position]:
        with self._lock:
            return self._by_tid.get(str(trade_id))

    def get(self, sym: str) -> Optional[Position]:
        """Legacy lookup by symbol — returns the FIRST open position on `sym`.
        Ambiguous when multiple positions exist on the same symbol; new code
        should call `get_by_trade_id` or `list_open_by_symbol` instead."""
        with self._lock:
            for p in self._by_tid.values():
                if p.symbol == sym:
                    return p
            return None

    def all(self) -> list[Position]:
        with self._lock:
            return list(self._by_tid.values())

    # --- required by ExitExecutor ---
    def list_open(self) -> dict[str, Position]:
        """Return a snapshot of all open positions keyed by trade_id."""
        with self._lock:
            return dict(self._by_tid)

    def list_open_by_symbol(self, sym: str) -> list[Position]:
        """All open positions on a given symbol (may be multiple under wide_open_mode)."""
        with self._lock:
            return [p for p in self._by_tid.values() if p.symbol == sym]

    def close(self, trade_id: str) -> None:
        with self._lock:
            self._by_tid.pop(str(trade_id), None)
        self._broadcast_positions()

    def reduce(self, trade_id: str, qty_exit: int) -> None:
        """Reduce qty for partial exits; remove if goes to zero."""
        with self._lock:
            p = self._by_tid.get(str(trade_id))
            if not p:
                return
            new_qty = int(p.qty) - int(qty_exit)
            if new_qty <= 0:
                self._by_tid.pop(str(trade_id), None)
            else:
                p.qty = new_qty
                self._by_tid[str(trade_id)] = p
        self._broadcast_positions()

    def _broadcast_positions(self):
        """Broadcast current positions to WebSocket clients.
        Excludes shadow trades - they're for internal analysis only.
        """
        if not self._api_server:
            return
        positions = []
        with self._lock:
            for p in self._by_tid.values():
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
                    if state.get("eod_scale_out_first_done"):
                        pos_dict["eod_partial_done"] = True
                    if state.get("manual_partial_qty", 0):
                        pos_dict["manual_partial_done"] = True
                    # Include all partial profits: T1 + T2 + manual + EOD
                    t1_profit = state.get("t1_profit", 0) or 0
                    t2_profit = state.get("t2_profit", 0) or 0
                    manual_profit = state.get("manual_partial_profit", 0) or 0
                    eod_profit = state.get("eod_partial_profit", 0) or 0
                    pos_dict["booked_pnl"] = t1_profit + t2_profit + manual_profit + eod_profit
                positions.append(pos_dict)
        self._api_server.broadcast_ws("positions", {"positions": positions})
