"""
Paper Trading Reconciliation Layer

Validates persisted positions against current market prices on startup.
Only used for paper trading mode (live mode uses broker_reconciliation.py).

Key Features:
- Checks if SL would have been hit while offline
- Checks if T1/T2 targets were reached
- Logs phantom exits for accurate PnL tracking
- Updates position state before restoring
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

try:
    from config.logging_config import get_agent_logger
    logger = get_agent_logger()
    if logger is None:
        raise ValueError("get_agent_logger returned None")
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


def validate_paper_position_on_recovery(
    pers_pos,
    current_price: float,
    trading_logger,
    persistence
) -> Tuple[bool, Dict[str, Any], bool]:
    """
    Validate a persisted position against current price for paper trading recovery.

    Args:
        pers_pos: PersistedPosition object
        current_price: Current market price for the symbol
        trading_logger: TradingLogger instance for phantom exit logging
        persistence: PositionPersistence instance

    Returns:
        (should_restore, state_updates, phantom_exit_logged)
        - should_restore: True if position should be restored
        - state_updates: Dict of state updates (e.g., {"t1_done": True})
        - phantom_exit_logged: True if we logged a phantom exit
    """
    plan = pers_pos.plan or {}
    side = pers_pos.side  # "BUY" or "SELL"
    entry_price = pers_pos.avg_price
    qty = pers_pos.qty
    symbol = pers_pos.symbol

    # Get stop and targets from plan
    stop_data = plan.get("stop", {})
    hard_sl = stop_data.get("hard")
    targets = plan.get("targets", [])

    t1_level = targets[0].get("level") if len(targets) > 0 else None
    t2_level = targets[1].get("level") if len(targets) > 1 else None
    t1_qty_pct = targets[0].get("qty_pct", 0.6) if len(targets) > 0 else 0.6
    # Note: t2_qty computed as qty - t1_qty (remainder)

    is_long = side == "BUY"
    state_updates = {}

    # Check if SL was hit (position would have been stopped out)
    if hard_sl is not None:
        sl_hit = (is_long and current_price <= hard_sl) or (not is_long and current_price >= hard_sl)
        if sl_hit:
            # Calculate loss at SL
            if is_long:
                sl_loss = (hard_sl - entry_price) * qty
            else:
                sl_loss = (entry_price - hard_sl) * qty

            logger.warning(
                f"[RECOVERY] Position {symbol} SKIPPED - SL would have been hit. "
                f"Entry={entry_price:.2f} SL={hard_sl:.2f} Current={current_price:.2f} Loss={sl_loss:.2f}"
            )

            # Log phantom SL exit to analytics
            if trading_logger:
                # Calculate R-multiple (will be -1.0 for SL hit)
                risk_per_share = abs(entry_price - hard_sl)
                r_multiple = sl_loss / (qty * risk_per_share) if risk_per_share > 0 else None

                trading_logger.log_exit({
                    "symbol": symbol,
                    "trade_id": pers_pos.trade_id,
                    "timestamp": datetime.now().isoformat(),
                    "qty": qty,
                    "exit_price": hard_sl,
                    "reason": "phantom_hard_sl_recovery",
                    "pnl": sl_loss,
                    "bias": "long" if is_long else "short",
                    "strategy": plan.get("strategy", ""),
                    "setup_type": plan.get("setup_type", ""),
                    "actual_entry_price": entry_price,
                    "slippage_bps": 0,  # No slippage for phantom exits
                    "exit_sequence": 1,
                    "total_exits": 1,
                    "is_final_exit": True,
                    "total_trade_pnl": sl_loss,
                    "diagnostics": {
                        "r_multiple": r_multiple,
                        "mae_pct": None,  # Not available for phantom exits
                        "mfe_pct": None,
                        "time_in_trade_mins": None,  # Unknown - server was offline
                        "remaining_qty": 0,
                    },
                    "meta": {
                        "regime": plan.get("regime"),
                        "setup_type": plan.get("setup_type"),
                        "acceptance_status": plan.get("quality", {}).get("acceptance_status"),
                    },
                })

            # Remove from persistence since it would have been exited
            if persistence:
                persistence.remove_position(symbol)
            return (False, {}, True)

    # Check if T2 was hit (full exit, need to log phantom profits)
    if t2_level is not None:
        t2_hit = (is_long and current_price >= t2_level) or (not is_long and current_price <= t2_level)
        if t2_hit:
            # Calculate T1 profit + T2 profit
            t1_qty = int(qty * t1_qty_pct)
            t2_qty = qty - t1_qty

            if is_long:
                t1_profit = (t1_level - entry_price) * t1_qty if t1_level else 0
                t2_profit = (t2_level - entry_price) * t2_qty
            else:
                t1_profit = (entry_price - t1_level) * t1_qty if t1_level else 0
                t2_profit = (entry_price - t2_level) * t2_qty

            total_profit = t1_profit + t2_profit

            logger.info(
                f"[RECOVERY] Position {symbol} SKIPPED - T2 hit while offline. "
                f"Entry={entry_price:.2f} T1={t1_level} T2={t2_level:.2f} Current={current_price:.2f} "
                f"T1_profit={t1_profit:.2f} T2_profit={t2_profit:.2f} Total={total_profit:.2f}"
            )

            # Log phantom exits to analytics
            if trading_logger:
                timestamp = datetime.now().isoformat()
                risk_per_share = abs(entry_price - hard_sl) if hard_sl else None

                # Log T1 phantom exit
                if t1_level:
                    t1_r_multiple = t1_profit / (t1_qty * risk_per_share) if risk_per_share else None
                    trading_logger.log_exit({
                        "symbol": symbol,
                        "trade_id": pers_pos.trade_id,
                        "timestamp": timestamp,
                        "qty": t1_qty,
                        "exit_price": t1_level,
                        "reason": "phantom_t1_recovery",
                        "pnl": t1_profit,
                        "bias": "long" if is_long else "short",
                        "strategy": plan.get("strategy", ""),
                        "setup_type": plan.get("setup_type", ""),
                        "actual_entry_price": entry_price,
                        "slippage_bps": 0,
                        "exit_sequence": 1,
                        "total_exits": 2,
                        "is_final_exit": False,
                        "diagnostics": {
                            "r_multiple": t1_r_multiple,
                            "mae_pct": None,
                            "mfe_pct": None,
                            "time_in_trade_mins": None,
                            "remaining_qty": t2_qty,
                        },
                        "meta": {
                            "regime": plan.get("regime"),
                            "setup_type": plan.get("setup_type"),
                            "acceptance_status": plan.get("quality", {}).get("acceptance_status"),
                        },
                    })

                # Log T2 phantom exit
                t2_r_multiple = t2_profit / (t2_qty * risk_per_share) if risk_per_share else None
                trading_logger.log_exit({
                    "symbol": symbol,
                    "trade_id": pers_pos.trade_id,
                    "timestamp": timestamp,
                    "qty": t2_qty,
                    "exit_price": t2_level,
                    "reason": "phantom_t2_recovery",
                    "pnl": t2_profit,
                    "bias": "long" if is_long else "short",
                    "strategy": plan.get("strategy", ""),
                    "setup_type": plan.get("setup_type", ""),
                    "actual_entry_price": entry_price,
                    "slippage_bps": 0,
                    "exit_sequence": 2,
                    "total_exits": 2,
                    "is_final_exit": True,
                    "total_trade_pnl": total_profit,
                    "diagnostics": {
                        "r_multiple": t2_r_multiple,
                        "mae_pct": None,
                        "mfe_pct": None,
                        "time_in_trade_mins": None,
                        "remaining_qty": 0,
                    },
                    "meta": {
                        "regime": plan.get("regime"),
                        "setup_type": plan.get("setup_type"),
                        "acceptance_status": plan.get("quality", {}).get("acceptance_status"),
                    },
                })

            # Remove from persistence since it's fully exited
            if persistence:
                persistence.remove_position(symbol)
            return (False, {}, True)

    # Check if T1 was hit (partial exit, mark t1_done=True)
    if t1_level is not None:
        t1_hit = (is_long and current_price >= t1_level) or (not is_long and current_price <= t1_level)
        if t1_hit:
            logger.info(
                f"[RECOVERY] Position {symbol} - T1 hit while offline, marking t1_done=True. "
                f"Entry={entry_price:.2f} T1={t1_level:.2f} Current={current_price:.2f}"
            )
            state_updates["t1_done"] = True

            # Also update the remaining quantity (T1 partial was taken)
            t1_qty = int(qty * t1_qty_pct)
            remaining_qty = qty - t1_qty
            state_updates["_remaining_qty"] = remaining_qty

            # Log phantom T1 exit
            if trading_logger:
                if is_long:
                    t1_profit = (t1_level - entry_price) * t1_qty
                else:
                    t1_profit = (entry_price - t1_level) * t1_qty

                risk_per_share = abs(entry_price - hard_sl) if hard_sl else None
                t1_r_multiple = t1_profit / (t1_qty * risk_per_share) if risk_per_share else None

                trading_logger.log_exit({
                    "symbol": symbol,
                    "trade_id": pers_pos.trade_id,
                    "timestamp": datetime.now().isoformat(),
                    "qty": t1_qty,
                    "exit_price": t1_level,
                    "reason": "phantom_t1_recovery",
                    "pnl": t1_profit,
                    "bias": "long" if is_long else "short",
                    "strategy": plan.get("strategy", ""),
                    "setup_type": plan.get("setup_type", ""),
                    "actual_entry_price": entry_price,
                    "slippage_bps": 0,
                    "exit_sequence": 1,
                    "total_exits": 2,
                    "is_final_exit": False,
                    "diagnostics": {
                        "r_multiple": t1_r_multiple,
                        "mae_pct": None,
                        "mfe_pct": None,
                        "time_in_trade_mins": None,
                        "remaining_qty": remaining_qty,
                    },
                    "meta": {
                        "regime": plan.get("regime"),
                        "setup_type": plan.get("setup_type"),
                        "acceptance_status": plan.get("quality", {}).get("acceptance_status"),
                    },
                })

    # Position should be restored (possibly with state updates)
    return (True, state_updates, False)
