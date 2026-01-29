"""
Startup Recovery — restore position state on process start.

Extracted from main.py so both the main process and child execution process
can perform startup recovery independently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from config.logging_config import get_agent_logger
from services.execution.trade_executor import Position
from services.state import PositionPersistence, BrokerReconciliation, validate_paper_position_on_recovery
from services.state.position_store import PositionStore

logger = get_agent_logger()


def _merge_persisted_state_into_plan(pers_pos) -> dict:
    """
    Merge PersistedPosition.state into plan["_state"] for proper recovery.

    The exit_executor reads state from plan["_state"], but persistence stores
    updates in a separate 'state' field. This merges them back together.
    """
    plan = dict(pers_pos.plan) if pers_pos.plan else {}
    if pers_pos.state:
        # Merge persisted state into plan's _state
        plan_state = plan.get("_state", {})
        plan_state.update(pers_pos.state)
        plan["_state"] = plan_state
    return plan


def startup_recovery(
    broker,
    is_live_mode: bool,
    is_paper_mode: bool,
    log_dir,
    position_store: PositionStore,
    trading_logger_instance=None
) -> Optional[PositionPersistence]:
    """
    Recover position state on startup.

    For live mode: Reconciles with broker positions
    For paper mode: Validates positions against current price (SL/T1/T2 checks)
    For dry-run (backtest): Returns None (no persistence needed)

    Args:
        broker: KiteBroker instance
        is_live_mode: True if live trading
        is_paper_mode: True if paper trading
        log_dir: Directory for position snapshot
        position_store: PositionStore to populate
        trading_logger_instance: TradingLogger for phantom exit logging

    Returns:
        PositionPersistence instance (or None for backtests)
    """
    # Backtests don't need persistence
    if not is_live_mode and not is_paper_mode:
        logger.info("[RECOVERY] Backtest mode - persistence disabled")
        return None

    # Use a dedicated state directory for persistence (not session-specific)
    # This allows recovery across sessions
    state_dir = Path(__file__).resolve().parents[1] / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    persistence = PositionPersistence(state_dir)
    persisted = persistence.load_snapshot()

    if not persisted:
        logger.info("[RECOVERY] No persisted positions found")
        return persistence

    logger.info(f"[RECOVERY] Found {len(persisted)} persisted positions")

    if is_live_mode:
        # Live mode: Reconcile with broker
        try:
            reconciliation = BrokerReconciliation(broker)
            result = reconciliation.reconcile(persisted)

            # Log results
            if result.orphaned_app:
                logger.warning(f"[RECOVERY] Positions closed externally: {list(result.orphaned_app.keys())}")
            if result.manual_trades:
                logger.info(f"[RECOVERY] Manual trades detected (not managed): {list(result.manual_trades.keys())}")
            if result.qty_mismatch:
                logger.warning(f"[RECOVERY] QTY MISMATCH - adjusting to broker state: {list(result.qty_mismatch.keys())}")

            # Restore matched positions
            for sym, pers_pos in result.matched.items():
                merged_plan = _merge_persisted_state_into_plan(pers_pos)
                position_store.upsert(Position(
                    symbol=pers_pos.symbol,
                    side=pers_pos.side,
                    qty=pers_pos.qty,
                    avg_price=pers_pos.avg_price,
                    plan=merged_plan,
                ))
                logger.info(f"[RECOVERY] Restored position: {sym} {pers_pos.side} {pers_pos.qty}@{pers_pos.avg_price} state={pers_pos.state}")

            # Handle qty mismatches - trust broker
            for sym, (pers_pos, broker_pos) in result.qty_mismatch.items():
                adjusted_pos = reconciliation.adjust_for_broker(pers_pos, broker_pos)
                merged_plan = _merge_persisted_state_into_plan(adjusted_pos)
                position_store.upsert(Position(
                    symbol=adjusted_pos.symbol,
                    side=adjusted_pos.side,
                    qty=adjusted_pos.qty,
                    avg_price=adjusted_pos.avg_price,
                    plan=merged_plan,
                ))
                # Update persistence with adjusted qty
                persistence.update_position(sym, new_qty=adjusted_pos.qty)
                logger.info(f"[RECOVERY] Adjusted position (broker qty): {sym} {adjusted_pos.qty} state={adjusted_pos.state}")

            # Remove orphaned positions from persistence
            for sym in result.orphaned_app:
                persistence.remove_position(sym)

            logger.info(f"[RECOVERY] Recovered {len(result.matched) + len(result.qty_mismatch)} positions")

        except Exception as e:
            logger.error(f"[RECOVERY] Broker reconciliation failed: {e}")
            # Fall back to persisted state only
            for sym, pers_pos in persisted.items():
                merged_plan = _merge_persisted_state_into_plan(pers_pos)
                position_store.upsert(Position(
                    symbol=pers_pos.symbol,
                    side=pers_pos.side,
                    qty=pers_pos.qty,
                    avg_price=pers_pos.avg_price,
                    plan=merged_plan,
                ))
            logger.warning(f"[RECOVERY] Restored {len(persisted)} positions from snapshot (no broker verification)")

    else:  # Paper mode
        # Paper mode: Validate positions against current price before restoring
        # This handles positions where SL/T1/T2 would have been hit while offline
        restored_count = 0
        phantom_exit_count = 0

        for sym, pers_pos in persisted.items():
            # Get current price to validate position
            try:
                current_price = broker.get_ltp(sym)
                if current_price is None:
                    logger.warning(f"[RECOVERY] Could not get LTP for {sym}, skipping validation")
                    current_price = pers_pos.avg_price  # Fallback to entry price
            except Exception as e:
                logger.warning(f"[RECOVERY] LTP fetch failed for {sym}: {e}, using entry price")
                current_price = pers_pos.avg_price

            # Validate position against current price (SL/T1/T2 checks)
            should_restore, state_updates, phantom_logged = validate_paper_position_on_recovery(
                pers_pos=pers_pos,
                current_price=current_price,
                trading_logger=trading_logger_instance,
                persistence=persistence
            )

            if phantom_logged:
                phantom_exit_count += 1

            if not should_restore:
                continue  # Position was stopped out or T2 hit - already handled

            # Apply state updates (e.g., t1_done=True if T1 was hit)
            if state_updates:
                if pers_pos.state is None:
                    pers_pos.state = {}
                pers_pos.state.update(state_updates)

                # Update quantity if T1 was hit (partial exit happened)
                if "_remaining_qty" in state_updates:
                    pers_pos.qty = state_updates["_remaining_qty"]

                # Update persistence with new state
                persistence.update_position(sym, new_qty=pers_pos.qty, state_updates=state_updates)

            merged_plan = _merge_persisted_state_into_plan(pers_pos)
            position_store.upsert(Position(
                symbol=pers_pos.symbol,
                side=pers_pos.side,
                qty=pers_pos.qty,
                avg_price=pers_pos.avg_price,
                plan=merged_plan,
            ))
            restored_count += 1
            logger.info(
                f"[RECOVERY] Restored position (paper): {sym} {pers_pos.side} {pers_pos.qty}@{pers_pos.avg_price} "
                f"state={pers_pos.state} current_price={current_price:.2f}"
            )

        logger.info(
            f"[RECOVERY] Paper mode recovery complete: restored={restored_count} "
            f"phantom_exits={phantom_exit_count} total_persisted={len(persisted)}"
        )

    return persistence
