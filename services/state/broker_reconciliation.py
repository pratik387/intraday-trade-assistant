"""
Broker Reconciliation Layer

Syncs app position state with actual broker positions on startup.
Only used for live trading mode (paper mode skips reconciliation).

Key Features:
- Compares persisted positions with broker positions
- Detects manual trades (not placed by app)
- Identifies externally closed positions
- Handles qty mismatches
- Trusts broker as source of truth
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from broker.kite.kite_broker import APP_ORDER_TAG_PREFIX

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


@dataclass
class BrokerPosition:
    """Represents a position as reported by broker."""
    symbol: str
    qty: int
    avg_price: float
    pnl: float = 0.0
    product: str = "MIS"  # MIS or CNC


@dataclass
class ReconciliationResult:
    """Result of reconciliation between app and broker positions."""
    # Positions that exist in both app and broker with matching qty
    matched: Dict[str, Any] = field(default_factory=dict)

    # Positions in app but not in broker (externally closed)
    orphaned_app: Dict[str, Any] = field(default_factory=dict)

    # Qty mismatch between app and broker
    qty_mismatch: Dict[str, Tuple[Any, BrokerPosition]] = field(default_factory=dict)

    # Manual trades in broker (no ITDA_ tag)
    manual_trades: Dict[str, BrokerPosition] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary string."""
        parts = []
        if self.matched:
            parts.append(f"matched={len(self.matched)}")
        if self.orphaned_app:
            parts.append(f"orphaned_app={len(self.orphaned_app)}")
        if self.qty_mismatch:
            parts.append(f"qty_mismatch={len(self.qty_mismatch)}")
        if self.manual_trades:
            parts.append(f"manual_trades={len(self.manual_trades)}")
        return ", ".join(parts) if parts else "no_positions"


class BrokerReconciliation:
    """
    Reconciles app positions with broker positions on startup.

    Usage:
        reconciliation = BrokerReconciliation(broker)
        result = reconciliation.reconcile(persisted_positions)

        # Handle results
        for sym, pos in result.matched.items():
            position_store.upsert(pos)

        for sym in result.orphaned_app:
            logger.warning(f"Position {sym} was closed externally")

        for sym, (app_pos, broker_pos) in result.qty_mismatch.items():
            # Trust broker qty
            app_pos.qty = broker_pos.qty
            position_store.upsert(app_pos)
    """

    def __init__(self, broker) -> None:
        """
        Initialize reconciliation layer.

        Args:
            broker: KiteBroker instance
        """
        self.broker = broker

    def reconcile(self, persisted: Dict[str, Any]) -> ReconciliationResult:
        """
        Compare persisted positions with broker positions.

        Args:
            persisted: Dict mapping symbol to PersistedPosition

        Returns:
            ReconciliationResult with categorized positions
        """
        result = ReconciliationResult()

        # Get broker positions
        broker_positions = self._get_broker_positions()
        if broker_positions is None:
            logger.warning("Could not fetch broker positions, using persisted state only")
            # Fall back to trusting persisted state
            for sym, pos in persisted.items():
                result.matched[sym] = pos
            return result

        # Get app orders (tagged with ITDA_)
        app_order_symbols = self._get_app_order_symbols()

        # Reconcile each persisted position
        for sym, persisted_pos in persisted.items():
            broker_pos = broker_positions.get(sym)

            if broker_pos:
                # Position exists in both - verify qty matches
                persisted_qty = int(getattr(persisted_pos, "qty", 0))
                if broker_pos.qty == persisted_qty:
                    result.matched[sym] = persisted_pos
                else:
                    # Qty mismatch - trust broker
                    result.qty_mismatch[sym] = (persisted_pos, broker_pos)
                    logger.warning(
                        f"QTY MISMATCH: {sym} app_qty={persisted_qty} broker_qty={broker_pos.qty}"
                    )
            else:
                # Position was closed externally
                result.orphaned_app[sym] = persisted_pos
                logger.warning(f"Position closed externally: {sym}")

        # Find manual trades (broker positions not in our persisted state)
        for sym, broker_pos in broker_positions.items():
            if sym not in persisted:
                # Check if this is an app trade by looking at order tags
                if sym not in app_order_symbols:
                    result.manual_trades[sym] = broker_pos
                    logger.info(f"Manual trade detected (not managed): {sym}")

        logger.info(f"Reconciliation complete: {result.summary()}")
        return result

    def _get_broker_positions(self) -> Optional[Dict[str, BrokerPosition]]:
        """
        Fetch current positions from Zerodha.

        Returns:
            Dict mapping symbol to BrokerPosition, or None on error
        """
        try:
            positions = self.broker.get_positions()
            day_positions = positions.get("day", [])

            result: Dict[str, BrokerPosition] = {}
            for p in day_positions:
                qty = int(p.get("quantity", 0))
                if qty == 0:
                    continue  # Skip closed positions

                # Build symbol in EXCH:TSYM format
                exchange = p.get("exchange", "NSE")
                tsym = p.get("tradingsymbol", "")
                symbol = f"{exchange}:{tsym}"

                result[symbol] = BrokerPosition(
                    symbol=symbol,
                    qty=abs(qty),  # Zerodha uses negative for short
                    avg_price=float(p.get("average_price", 0)),
                    pnl=float(p.get("pnl", 0)),
                    product=p.get("product", "MIS"),
                )

            return result

        except Exception as e:
            logger.error(f"Failed to fetch broker positions: {e}")
            return None

    def _get_app_order_symbols(self) -> set:
        """
        Get symbols for all orders placed by this app.

        Returns:
            Set of symbols that have app-placed orders
        """
        try:
            app_orders = self.broker.get_app_orders()
            symbols = set()
            for order in app_orders:
                exchange = order.get("exchange", "NSE")
                tsym = order.get("tradingsymbol", "")
                symbols.add(f"{exchange}:{tsym}")
            return symbols
        except Exception as e:
            logger.error(f"Failed to fetch app orders: {e}")
            return set()

    def adjust_for_broker(
        self,
        persisted_pos: Any,
        broker_pos: BrokerPosition
    ) -> Any:
        """
        Adjust persisted position to match broker state.

        Args:
            persisted_pos: PersistedPosition from app
            broker_pos: BrokerPosition from broker

        Returns:
            Updated persisted position
        """
        # Trust broker qty
        persisted_pos.qty = broker_pos.qty
        persisted_pos.avg_price = broker_pos.avg_price

        # Add reconciliation marker to state
        if hasattr(persisted_pos, "state") and isinstance(persisted_pos.state, dict):
            persisted_pos.state["reconciled_with_broker"] = True
            persisted_pos.state["broker_qty"] = broker_pos.qty
            persisted_pos.state["broker_avg_price"] = broker_pos.avg_price

        return persisted_pos
