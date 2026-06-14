"""
Position Persistence Layer

Provides persistence for position state to survive server restarts.
Supports both paper trading and live trading modes.

Key Features:
- Atomic file writes (write to temp, then rename)
- Event-driven persistence (save on entry/exit, not periodic)
- Fallback recovery from events.jsonl if snapshot corrupted
- Thread-safe operations
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

try:
    from utils.time_util import _now_naive_ist
except Exception:
    _now_naive_ist = None


def _now_ist():
    """Current IST-naive datetime (logging + snapshot timestamp).

    The snapshot timestamp is load-critical — `_load_from_file` compares its
    date against `_today_ist()` to decide freshness — so it MUST be written on
    the same (IST) clock the comparison reads, otherwise a UTC-hosted server
    crossing IST midnight could misjudge staleness.
    """
    if _now_naive_ist is not None:
        return _now_naive_ist()
    return datetime.now()


def _today_ist():
    """Current IST-naive date for snapshot-staleness decisions (restart recovery)."""
    return _now_ist().date()


def _parse_date(s):
    """Parse a YYYY-MM-DD (or ISO) string to a date; None on failure."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        try:
            return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None


@dataclass
class PersistedPosition:
    """Serializable position state for persistence."""
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: int
    avg_price: float
    trade_id: str
    order_id: Optional[str] = None
    order_tag: Optional[str] = None
    entry_time: Optional[str] = None
    plan: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)  # t1_done, t2_done, trailing_sl, etc.
    # Multi-day hold (CNC/MTF) fields. None for legacy intraday / 1-night
    # overnight positions (close_dn) — those keep the original behavior exactly.
    entry_date: Optional[str] = None      # YYYY-MM-DD (IST)
    exit_on_date: Optional[str] = None    # YYYY-MM-DD (IST); exit when today >= this
    product: Optional[str] = None         # "MIS" | "CNC" | "MTF"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersistedPosition":
        """Create from dict."""
        return cls(
            symbol=data.get("symbol", ""),
            side=data.get("side", "BUY"),
            qty=int(data.get("qty", 0)),
            avg_price=float(data.get("avg_price", 0.0)),
            trade_id=data.get("trade_id", ""),
            order_id=data.get("order_id"),
            order_tag=data.get("order_tag"),
            entry_time=data.get("entry_time"),
            plan=data.get("plan", {}),
            state=data.get("state", {}),
            entry_date=data.get("entry_date"),
            exit_on_date=data.get("exit_on_date"),
            product=data.get("product"),
        )


class PositionPersistence:
    """
    Handles position state persistence to survive restarts.

    Usage:
        persistence = PositionPersistence(log_dir)

        # Save on entry
        persistence.save_position(pos, order_id, order_tag)

        # Update on partial exit
        persistence.update_position(symbol, new_qty, state_updates)

        # Remove on full exit
        persistence.remove_position(symbol)

        # Load on startup
        positions = persistence.load_snapshot()
    """

    def __init__(self, state_dir: Path) -> None:
        """
        Initialize persistence layer.

        Args:
            state_dir: Directory to store snapshot file (usually log_dir)
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "positions_snapshot.json"
        self._lock = threading.Lock()
        self._positions: Dict[str, PersistedPosition] = {}

        # Try to load existing snapshot on init
        self._load_from_file()

    def _load_from_file(self) -> None:
        """Load positions from snapshot file if it exists.

        Two recovery regimes, decided by snapshot freshness:

        - Today's snapshot: load every position verbatim (intraday + multi-day
          alike). This is the normal same-day restart path and is unchanged.

        - Stale snapshot (a previous day): intraday and 1-night overnight
          positions would have been EOD/next-open squared off, so they are
          dropped exactly as before — a legacy position has no `exit_on_date`,
          so it is never salvaged (close_dn behavior is byte-identical). A
          multi-day CNC/MTF hold, however, is still open if its `exit_on_date`
          is today or later; those are SALVAGED so a restart mid-hold does not
          orphan a real, leveraged delivery position. Multi-day holds whose
          `exit_on_date` has already passed are dropped (they should have been
          exited and will be reconciled against the broker).
        """
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            positions_raw = data.get("positions", {})

            # Determine snapshot freshness. An absent or unparseable timestamp
            # is treated as STALE (not fresh): that forces the per-position
            # salvage/expiry path, which drops legacy intraday positions (the
            # safe default) instead of blindly loading a snapshot of unknown age.
            snapshot_ts = data.get("timestamp")
            snapshot_date = _parse_date(snapshot_ts)
            if snapshot_date is None and snapshot_ts:
                logger.warning(
                    f"[PERSIST] Could not parse snapshot timestamp: {snapshot_ts}, "
                    f"treating as stale (per-position recovery)"
                )
            is_stale = snapshot_date != _today_ist()

            if not is_stale:
                for sym, pos_data in positions_raw.items():
                    self._positions[sym] = PersistedPosition.from_dict(pos_data)
                logger.info(
                    f"[PERSIST] Loaded {len(self._positions)} positions from snapshot "
                    f"(today's date validated)"
                )
                return

            # Stale snapshot: salvage only still-open multi-day holds.
            today = _today_ist()
            salvaged, dropped = 0, 0
            for sym, pos_data in positions_raw.items():
                pos = PersistedPosition.from_dict(pos_data)
                exit_on = _parse_date(pos.exit_on_date)
                if exit_on is not None and exit_on >= today:
                    self._positions[sym] = pos
                    salvaged += 1
                else:
                    dropped += 1
            logger.warning(
                f"[PERSIST] Stale snapshot ({snapshot_ts}): salvaged {salvaged} "
                f"still-open multi-day hold(s); dropped {dropped} intraday/1-night/"
                f"expired position(s) (EOD squared off)."
            )
        except Exception as e:
            logger.error(f"Failed to load position snapshot: {e}")

    def _save_to_file(self) -> None:
        """Atomic save of all positions to file."""
        try:
            data = {
                "timestamp": _now_ist().isoformat(),
                "positions": {
                    sym: pos.to_dict()
                    for sym, pos in self._positions.items()
                }
            }
            # Atomic write: write to temp, then rename
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.state_file)  # Atomic on most systems
        except Exception as e:
            logger.error(f"Failed to save position snapshot: {e}")

    # -------------------- Public API --------------------

    def save_position(
        self,
        symbol: str,
        side: str,
        qty: int,
        avg_price: float,
        trade_id: str,
        order_id: Optional[str] = None,
        order_tag: Optional[str] = None,
        plan: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
        entry_date: Optional[str] = None,
        exit_on_date: Optional[str] = None,
        product: Optional[str] = None,
    ) -> None:
        """
        Save a new position (on entry).

        Args:
            symbol: Trading symbol (EXCH:TSYM)
            side: "BUY" or "SELL"
            qty: Position quantity
            avg_price: Entry price
            trade_id: Unique trade identifier
            order_id: Broker order ID
            order_tag: Order tag (ITDA_xxx)
            plan: Trade plan dict (targets, stops, etc.)
            state: Initial state dict
            entry_date: YYYY-MM-DD (IST) entry date for multi-day holds. None
                for legacy intraday / 1-night overnight (close_dn) positions.
            exit_on_date: YYYY-MM-DD (IST); the position is held until today >=
                this date. None => legacy behavior (no multi-day salvage).
            product: "MIS" | "CNC" | "MTF". None for legacy positions.
        """
        with self._lock:
            self._positions[symbol] = PersistedPosition(
                symbol=symbol,
                side=side,
                qty=qty,
                avg_price=avg_price,
                trade_id=trade_id,
                order_id=order_id,
                order_tag=order_tag,
                entry_time=_now_ist().isoformat(),
                plan=plan or {},
                state=state or {},
                entry_date=entry_date,
                exit_on_date=exit_on_date,
                product=product,
            )
            self._save_to_file()
            logger.debug(
                f"Position saved: {symbol} {side} {qty}@{avg_price}"
                + (f" exit_on={exit_on_date}" if exit_on_date else "")
            )

    def update_position(
        self,
        symbol: str,
        new_qty: Optional[int] = None,
        state_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update position (on partial exit or state change).

        Args:
            symbol: Trading symbol
            new_qty: Updated quantity (after partial exit)
            state_updates: State updates (t1_done, trailing_sl, etc.)
        """
        with self._lock:
            pos = self._positions.get(symbol)
            if not pos:
                logger.warning(f"Cannot update non-existent position: {symbol}")
                return

            if new_qty is not None:
                pos.qty = new_qty

            if state_updates:
                pos.state.update(state_updates)

            self._save_to_file()
            logger.debug(f"Position updated: {symbol} qty={pos.qty} state={pos.state}")

    def remove_position(self, symbol: str) -> None:
        """
        Remove position (on full exit).

        Args:
            symbol: Trading symbol
        """
        with self._lock:
            if symbol in self._positions:
                del self._positions[symbol]
                self._save_to_file()
                logger.debug(f"Position removed: {symbol}")

    def load_snapshot(self) -> Dict[str, PersistedPosition]:
        """
        Load all positions from snapshot.

        Returns:
            Dict mapping symbol to PersistedPosition
        """
        with self._lock:
            return dict(self._positions)

    def get_position(self, symbol: str) -> Optional[PersistedPosition]:
        """
        Get a specific position.

        Args:
            symbol: Trading symbol

        Returns:
            PersistedPosition or None if not found
        """
        with self._lock:
            return self._positions.get(symbol)

    def clear_all(self) -> None:
        """Clear all positions (use with caution)."""
        with self._lock:
            self._positions.clear()
            self._save_to_file()
            logger.warning("All positions cleared from persistence")

    def recover_from_events(self, events_file: Path) -> Dict[str, PersistedPosition]:
        """
        Reconstruct positions from event log (backup recovery).

        This is a fallback when snapshot is corrupted or missing.
        Reads TRIGGER and EXIT events to reconstruct current positions.

        Args:
            events_file: Path to events.jsonl file

        Returns:
            Dict mapping symbol to PersistedPosition
        """
        positions: Dict[str, PersistedPosition] = {}

        if not events_file.exists():
            logger.warning(f"Events file not found: {events_file}")
            return positions

        try:
            with open(events_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        self._apply_event(positions, event)
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.debug(f"Error processing event: {e}")
                        continue

            # Update internal state
            with self._lock:
                self._positions = positions
                self._save_to_file()

            logger.info(f"Recovered {len(positions)} positions from events.jsonl")
            return positions

        except Exception as e:
            logger.error(f"Failed to recover from events: {e}")
            return {}

    def _apply_event(self, positions: Dict[str, PersistedPosition], event: Dict[str, Any]) -> None:
        """Apply a single event to reconstruct position state."""
        event_type = event.get("type")

        if event_type == "TRIGGER":
            # Entry event - create position
            trigger = event.get("trigger", {})
            symbol = event.get("symbol", "")
            trade_id = event.get("trade_id", "")

            if symbol and trade_id:
                # Determine side from trigger
                side = trigger.get("side", "BUY")

                positions[symbol] = PersistedPosition(
                    symbol=symbol,
                    side=side,
                    qty=int(trigger.get("qty", 0)),
                    avg_price=float(trigger.get("actual_price", 0)),
                    trade_id=trade_id,
                    order_id=trigger.get("order_id"),
                    entry_time=event.get("ts"),
                    plan={},  # Plan not available in TRIGGER event
                    state={},
                    # Forward multi-day fields if the TRIGGER event carries them,
                    # so events.jsonl recovery does not downgrade a still-open
                    # CNC/MTF hold to legacy status (which would orphan it on the
                    # next stale-snapshot load).
                    entry_date=trigger.get("entry_date"),
                    exit_on_date=trigger.get("exit_on_date"),
                    product=trigger.get("product"),
                )

        elif event_type == "EXIT" or event.get("stage") == "EXIT":
            # Exit event - update or remove position
            symbol = event.get("symbol", "")
            is_final = event.get("is_final_exit", False)

            if symbol in positions:
                if is_final:
                    # Full exit - remove position
                    del positions[symbol]
                else:
                    # Partial exit - update quantity
                    qty_exited = int(event.get("qty", 0))
                    pos = positions[symbol]
                    pos.qty = max(0, pos.qty - qty_exited)
                    if pos.qty <= 0:
                        del positions[symbol]
