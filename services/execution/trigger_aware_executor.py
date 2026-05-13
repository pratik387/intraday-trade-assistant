# services/execution/trigger_aware_executor.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import pandas as pd
import uuid

from config.logging_config import get_execution_loggers
from config.filters_setup import load_filters
from services.orders.order_queue import OrderQueue
from utils.time_util import _minute_of_day, _parse_hhmm_to_md
from utils.price_utils import round_to_tick
from diagnostics.diag_event_log import diag_event_log
from services.target_recalc import (
    InvertedSLError,
    UnknownAnchorTypeError,
    recalculate_targets_for_actual_entry,
)

# Import from unified validator
from services.execution.trigger_validation_engine import (
    FastConditionValidator, 
    FastTriggerConditionFactory,
    TriggerCondition,
    ConditionType,
    TradeState
)

logger, trade_logger = get_execution_loggers()

# ======================== DATA CLASSES ========================

@dataclass
class PendingTrade:
    trade_id: str
    symbol: str
    plan: Dict[str, Any]
    state: TradeState
    created_at: pd.Timestamp
    
    # Trigger conditions
    primary_triggers: List[TriggerCondition] = field(default_factory=list)
    must_conditions: List[TriggerCondition] = field(default_factory=list)
    should_conditions: List[TriggerCondition] = field(default_factory=list)
    
    # Lifecycle management
    expiry_time: Optional[pd.Timestamp] = None
    last_validation_ts: Optional[pd.Timestamp] = None
    validation_count: int = 0
    confidence_score: float = 0.0
    
    # Execution tracking
    trigger_price: Optional[float] = None
    trigger_timestamp: Optional[pd.Timestamp] = None

# ======================== MAIN TRIGGER EXECUTOR ========================

class TriggerAwareExecutor:
    """Enhanced trade executor with 1-minute trigger validation"""
    
    def _final_execution_check(self, trade: PendingTrade) -> bool:
        """Final validation before order placement"""
        try:
            # Check if trade is stale using configured limit
            now = self._get_current_time()
            if trade.trigger_timestamp:
                staleness_seconds = (now - trade.trigger_timestamp).total_seconds()
                if staleness_seconds > self.staleness_seconds:
                    logger.debug(f"Trade stale: {trade.symbol} age={staleness_seconds:.1f}s (limit={self.staleness_seconds:.0f}s) - marking expired")
                    trade.state = TradeState.EXPIRED  # Mark as expired to avoid repeated checks
                    return False

            # Block duplicate — position may have been opened by another plan
            # between _add_pending_trade and now.
            # Per-setup wide_open: research setups bypass duplicate-guard so
            # every detector signal becomes a captured trade row.
            from services.config_loader import is_wide_open_for_setup
            if (trade.symbol in self.risk.open_positions
                    and not is_wide_open_for_setup(trade.plan.get("strategy"))):
                logger.warning(f"DUPLICATE_BLOCKED | {trade.symbol} | Position already open at execution time")
                trade.state = TradeState.EXPIRED
                return False

            # Check market hours
            minute_of_day = _minute_of_day(now)
            if self.entry_cutoff_md and minute_of_day >= self.entry_cutoff_md:
                logger.debug(f"Past entry cutoff: {trade.symbol}")
                return False

            # Check risk limits
            if not self.risk.can_open_more():
                logger.debug(f"Risk limit reached: {trade.symbol}")
                return False

            # Check capital availability (if enabled)
            if self.capital_manager:
                plan = trade.plan
                qty = int(plan.get("qty", 0))
                price = trade.trigger_price or plan.get("price", 0)
                cap_segment = plan.get("cap_segment", "unknown")

                # SHADOW TRADE LOGIC: If at capacity, mark as shadow instead of rejecting
                # Shadow trades go through entire pipeline but don't consume capital
                is_shadow = plan.get("shadow", False)  # May already be marked
                if not is_shadow and self.capital_manager.is_at_capacity():
                    trade.plan["shadow"] = True
                    is_shadow = True
                    logger.info(f"SHADOW_TRADE | {trade.symbol} | At max capacity - continuing as shadow trade")

                # SHADOW TRADE: Min notional check
                # Trades below min notional (e.g., 4% of capital) become shadow trades
                # This prevents tiny positions where transaction costs eat profits
                if not is_shadow:
                    notional = qty * price
                    min_notional = self.capital_manager.get_min_notional()
                    if min_notional > 0 and notional < min_notional:
                        trade.plan["shadow"] = True
                        trade.plan["shadow_reason"] = "notional_below_min"
                        is_shadow = True
                        logger.info(
                            f"SHADOW_TRADE | {trade.symbol} | Notional {notional:.0f} < min {min_notional:.0f}"
                        )

                side = plan.get("side", "BUY")
                can_enter, adjusted_qty, reason = self.capital_manager.can_enter_position(
                    trade.symbol, qty, price, cap_segment, shadow=is_shadow, side=side
                )

                if not can_enter:
                    # MIS rejection is a HARD rejection - cannot short non-MIS stocks
                    # Do NOT convert to shadow trade
                    if "mis_not_allowed" in reason:
                        logger.warning(
                            f"TRADE_REJECTED | {trade.symbol} | MIS not allowed for short: {reason}"
                        )
                        return False  # Reject this trade entirely

                    # SHADOW TRADE: Convert capital rejection to shadow trade
                    # This allows tracking theoretical performance even when out of capital
                    if not is_shadow:
                        trade.plan["shadow"] = True
                        trade.plan["shadow_reason"] = f"capital_rejected:{reason}"
                        is_shadow = True
                        logger.info(
                            f"SHADOW_TRADE | {trade.symbol} | Capital rejected: {reason}"
                        )
                    # Re-check with shadow=True to proceed (shadow trades don't need capital)
                    can_enter, adjusted_qty, _ = self.capital_manager.can_enter_position(
                        trade.symbol, qty, price, cap_segment, shadow=True, side=side
                    )

                # Update plan with adjusted quantity if scaled down (not applicable for shadow trades)
                if adjusted_qty != qty and not is_shadow:
                    logger.info(f"Capital scaling: {trade.symbol} qty {qty} -> {adjusted_qty}")
                    trade.plan["qty"] = adjusted_qty
                    trade.plan["_original_qty"] = qty  # Keep original for reference

                    # SHADOW TRADE: Check if SCALED notional is below min
                    # This catches cases where original was above min but scaled is below
                    scaled_notional = adjusted_qty * price
                    if min_notional > 0 and scaled_notional < min_notional:
                        trade.plan["shadow"] = True
                        trade.plan["shadow_reason"] = "scaled_notional_below_min"
                        is_shadow = True
                        logger.info(
                            f"SHADOW_TRADE | {trade.symbol} | Scaled notional {scaled_notional:.0f} < min {min_notional:.0f}"
                        )

            return True

        except Exception as e:
            logger.exception(f"Final execution check failed: {trade.symbol}: {e}")
            return False
    
    def _place_trade_order(self, trade: PendingTrade) -> bool:
        """Place the actual trade order"""
        try:
            plan = trade.plan
            symbol = trade.symbol
            
            # Extract order parameters
            # Derive side from bias if not explicitly set
            side = plan.get("side")
            if not side:
                bias = plan.get("bias", "long")
                side = "SELL" if bias.lower() == "short" else "BUY"
            qty = int(plan.get("qty", 0))
            price = round_to_tick(trade.trigger_price or plan.get("price"))
            
            if qty <= 0:
                logger.warning(f"Invalid qty for {symbol}: {qty}")
                return False

            # Plan-level pre-entry validations (min_entry_sl_distance + RR floor) were
            # removed 2026-05-13. Each setup's min_stop_distance_pct is already enforced
            # by plan_orchestrator via enforce_min_stop_distance, and structural RR
            # is cell-mined per setup (circuit_t1 ~0.3R, delivery_pct 0.25R are
            # validated low-RR designs that the global 0.3 floor was killing).
            hard_sl = plan.get("hard_sl")
            logger.info(f"VALIDATION CHECK: {symbol} entry={price:.2f} hard_sl={hard_sl} side={side}")

            # Check if this is a shadow trade (simulated, no capital consumed)
            is_shadow = plan.get("shadow", False)

            if is_shadow:
                # SHADOW TRADE: Don't place real broker order, generate simulated order_id
                order_id = f"shadow-{trade.trade_id}"
                logger.info(f"SHADOW_ORDER | {symbol} | Simulated order (no broker call) | order_id={order_id}")
            else:
                # Place order with trade_id for tagging (identifies app-placed orders)
                order_args = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "order_type": "MARKET",  # Using market orders for trigger execution
                    "product": "MIS",
                    "variety": "regular",
                    "trade_id": trade.trade_id,  # For order tagging (ITDA_xxx)
                }
                order_id = self.broker.place_order(**order_args)

                # Reconcile with broker - verify position actually exists
                if order_id and hasattr(self.broker, 'reconcile_position'):
                    try:
                        reconciled = self.broker.reconcile_position(
                            symbol=symbol,
                            order_id=order_id,
                            expected_qty=qty,
                            expected_side=side,
                            timeout=0.5
                        )
                        if reconciled is None:
                            # Position not found at broker - order was likely rejected async
                            raise RuntimeError(f"Position not found at broker after order {order_id} - likely rejected")

                        # Use actual fill price from broker
                        broker_fill = reconciled.get("avg_price", price)
                        if broker_fill and broker_fill > 0:
                            slippage = broker_fill - price
                            slippage_bps = (slippage / price) * 10000 if price > 0 else 0
                            logger.info(f"ENTRY_FILL | {symbol} | Broker: {broker_fill:.2f} | Assumed: {price:.2f} | Slippage: {slippage:+.2f} ({slippage_bps:+.1f} bps)")
                            price = round_to_tick(broker_fill)  # Use actual fill price for all downstream
                            qty = reconciled.get("qty", qty)  # Use actual qty if different
                    except RuntimeError:
                        raise  # Re-raise position not found
                    except Exception as e:
                        logger.warning(f"Failed to reconcile position for {symbol}: {e}")

                # Post-fill RR check (removed 2026-05-13): contradicted cell-mined
                # low-RR setups; bad-fill detection now relies on broker reconcile
                # + per-setup hard_sl/structural targets.

            # Log TRIGGER to events.jsonl (single writer: diag_event_log)
            try:
                diag_event_log.log_trigger(
                    symbol=symbol,
                    plan=plan,
                    side=side,
                    qty=qty,
                    price=price,
                    trigger_ts=trade.trigger_timestamp,
                    order_id=order_id,
                    strategy=plan.get('strategy', ''),
                    shadow=plan.get('shadow', False),
                    diagnostics={
                        'confidence_score': trade.confidence_score,
                        'trigger_mode': 'bar' if trade.confidence_score > 0 else 'tick',
                        'trigger_price': trade.trigger_price,
                        'validation_count': trade.validation_count,
                        'entry_zone': plan.get('entry_zone') or plan.get('entry', {}).get('zone', []),
                    },
                )
            except Exception as _diag_err:
                logger.warning("diag_event_log.log_trigger failed for %s: %s", symbol, _diag_err)

            # Log ENTRY fill to events.jsonl (actual execution record)
            try:
                diag_event_log.log_entry_fill(
                    symbol=symbol,
                    plan=plan,
                    side=side,
                    qty=qty,
                    price=price,
                    entry_ts=trade.trigger_timestamp,
                    order_meta={"order_id": order_id},
                )
            except Exception as _diag_err:
                logger.warning("diag_event_log.log_entry_fill failed for %s: %s", symbol, _diag_err)

            # Log TRIGGER to trade_logs.log (human-readable)
            if self.trading_logger:
                self.trading_logger.log_trigger({
                    'symbol': symbol, 'price': price, 'qty': qty,
                    'strategy': plan.get('strategy', ''), 'order_id': order_id, 'side': side,
                })

            # Update risk state
            self.risk.open_positions[symbol] = {
                "side": side,
                "qty": qty,
                "avg_price": price
            }

            # Record position in capital manager (allocate margin)
            # Shadow trades skip margin allocation
            if self.capital_manager:
                cap_segment = plan.get("cap_segment", "unknown")
                is_shadow = plan.get("shadow", False)
                self.capital_manager.enter_position(
                    symbol=symbol,
                    qty=qty,
                    price=price,
                    cap_segment=cap_segment,
                    timestamp=trade.trigger_timestamp,
                    shadow=is_shadow
                )

            # Update shared position store for exit executor
            if self.positions:
                from services.execution.models import Position

                # CRITICAL FIX: Recalculate targets based on ACTUAL entry price
                # Original targets were calculated from entry_ref_price, but actual
                # trigger price may differ. Targets must be R-based from actual entry.
                adjusted_plan = self._recalculate_targets_for_actual_entry(
                    trade.plan, price, side
                )

                # Ensure entry_ts is always set for dashboard display
                # Use trigger timestamp from execution, fallback to plan timestamps
                if trade.trigger_timestamp:
                    adjusted_plan["entry_ts"] = trade.trigger_timestamp.isoformat() if hasattr(trade.trigger_timestamp, 'isoformat') else str(trade.trigger_timestamp)
                elif not adjusted_plan.get("entry_ts"):
                    adjusted_plan["entry_ts"] = adjusted_plan.get("trigger_ts") or pd.Timestamp.now().isoformat()

                pos = Position(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    avg_price=price,
                    plan=adjusted_plan
                )
                self.positions.upsert(pos)

                # Persist position for crash recovery (Phase 5)
                if self.persistence:
                    from broker.kite.kite_broker import APP_ORDER_TAG_PREFIX
                    order_tag = f"{APP_ORDER_TAG_PREFIX}{trade.trade_id[-12:]}"
                    logger.info(f"[PERSIST] Saving position: {symbol} {side} {qty}@{price} trade_id={trade.trade_id}")
                    try:
                        self.persistence.save_position(
                            symbol=symbol,
                            side=side,
                            qty=qty,
                            avg_price=price,
                            trade_id=trade.trade_id,
                            order_id=order_id,
                            order_tag=order_tag,
                            plan=adjusted_plan,
                            state={}  # Initial state (t1_done=False, etc.)
                        )
                        logger.info(f"[PERSIST] Position saved successfully: {symbol}")
                    except Exception as e:
                        logger.error(f"[PERSIST] Failed to save position {symbol}: {e}")
                else:
                    logger.debug(f"[PERSIST] persistence disabled - position {symbol} not persisted (expected in backtest)")

            return True

        except RuntimeError as e:
            # RuntimeError is raised for known rejection reasons (MIS blocked, etc.)
            # Log as warning without traceback - this is expected behavior, not a bug
            error_msg = str(e)
            if "MIS blocked" in error_msg or "MIS orders" in error_msg:
                logger.warning(f"MIS_BLOCKED | {trade.symbol} | {error_msg}")
            else:
                logger.warning(f"Order rejected for {trade.symbol}: {error_msg}")
            return False
        except Exception as e:
            # Unexpected errors get full traceback for debugging
            logger.exception(f"Order placement failed for {trade.symbol}: {e}")
            return False

    def _recalculate_targets_for_actual_entry(
        self, plan: Dict[str, Any], actual_entry: float, side: str
    ) -> Dict[str, Any]:
        """Re-anchor SL/targets/risk to the actual fill price.

        Delegates to services.target_recalc — see that module for anchor-type
        semantics (structural / r_multiple / or_range). The detector tags the
        plan with `target_anchor_type`; default is "structural" (preserve
        target levels, only update rps + actual_entry).

        On InvertedSLError (actual fill on the wrong side of the planned
        stop), logs the structural breach and returns the plan unchanged —
        downstream exit logic will detect the inverted SL and immediate-exit
        at market.
        """
        try:
            return recalculate_targets_for_actual_entry(plan, actual_entry, side)
        except InvertedSLError as e:
            logger.warning(
                f"TARGET_RECALC_INVERTED_SL: {plan.get('symbol')} "
                f"{plan.get('strategy')} {e} — keeping original plan"
            )
            return plan
        except UnknownAnchorTypeError as e:
            # Detector bug — silent fall-through used to keep detect-time
            # targets (broken in production for months on delivery_pct).
            # Fail loud so a future typo cannot ship: outer `except Exception`
            # in execute_trade() catches this and rejects the entry.
            logger.error(
                f"TARGET_RECALC_UNKNOWN_ANCHOR: {plan.get('symbol')} "
                f"{plan.get('strategy')} {e} — REJECTING entry"
            )
            raise
        except Exception as e:
            logger.warning(f"Target recalculation failed: {e}, using original targets")
            return plan

    def _cleanup_expired_trades(self) -> None:
        """Clean up expired and completed trades

        Uses _last_tick_ts (bar timestamp from tick processing) for expiry checks.
        This ensures in backtest mode we check expiry based on the bar being processed,
        not the simulation clock which may have advanced past multiple bars.
        """
        # Use last tick timestamp for accurate expiry checks (critical for backtest)
        # Fall back to _get_current_time() for live mode or initial startup
        now = self._last_tick_ts if self._last_tick_ts else self._get_current_time()

        with self._lock:
            expired_ids = []

            for trade_id, trade in self.pending_trades.items():
                # Remove expired trades
                if trade.expiry_time and now > trade.expiry_time:
                    if trade.state == TradeState.WAITING_TRIGGER:
                        trade.state = TradeState.EXPIRED
                        logger.info(f"EXPIRED: {trade.symbol} {trade.plan.get('strategy', '')} at {now}")
                
                # Remove completed/expired trades
                if trade.state in [TradeState.EXECUTED, TradeState.EXPIRED, TradeState.CANCELLED]:
                    expired_ids.append(trade_id)
            
            # Clean up
            for trade_id in expired_ids:
                del self.pending_trades[trade_id]
    
    def _cancel_pending_for_symbol(self, symbol: str) -> None:
        """Cancel existing pending trades for a symbol"""
        cancelled_count = 0
        for trade in self.pending_trades.values():
            if trade.symbol == symbol and trade.state == TradeState.WAITING_TRIGGER:
                trade.state = TradeState.CANCELLED
                cancelled_count += 1
        
        if cancelled_count > 0:
            logger.debug(f"Cancelled {cancelled_count} pending trades for {symbol}")
    
    def run_forever(self, sleep_ms: int = 200) -> None:
        """Run the executor continuously"""
        try:
            while not self._stop_event.is_set():
                self.run_once()
                time.sleep(max(0.0, sleep_ms / 1000.0))
        except KeyboardInterrupt:
            logger.info("TriggerAwareExecutor: stop (KeyboardInterrupt)")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the executor and clean up"""
        self._stop_event.set()
        logger.info("TriggerAwareExecutor stopped")
    
    def get_pending_trades_summary(self) -> Dict[str, Any]:
        """Get summary of pending trades for monitoring"""
        with self._lock:
            summary = {
                "total_pending": len([t for t in self.pending_trades.values() 
                                    if t.state == TradeState.WAITING_TRIGGER]),
                "total_triggered": len([t for t in self.pending_trades.values() 
                                      if t.state == TradeState.TRIGGERED]),
                "by_symbol": {},
                "by_strategy": {}
            }
            
            for trade in self.pending_trades.values():
                if trade.state in [TradeState.WAITING_TRIGGER, TradeState.TRIGGERED]:
                    # By symbol
                    summary["by_symbol"][trade.symbol] = summary["by_symbol"].get(trade.symbol, 0) + 1
                    
                    # By strategy
                    strategy = trade.plan.get("strategy", "unknown")
                    summary["by_strategy"][strategy] = summary["by_strategy"].get(strategy, 0) + 1
            
            return summary

    def __init__(
        self,
        broker,
        order_queue: OrderQueue,
        risk_state,
        positions,
        get_ltp_ts: Callable[[str], Tuple[Optional[float], Optional[pd.Timestamp]]],
        bar_builder,  # We'll hook into the BarBuilder's 1m callbacks
        trading_logger=None,  # Enhanced logging service
        capital_manager=None,  # Capital & MIS management
        persistence=None,  # Position persistence for crash recovery
        api_server=None  # For checking pause state
    ):
        self.broker = broker
        self.oq = order_queue
        self.trading_logger = trading_logger
        self.capital_manager = capital_manager
        self.persistence = persistence  # For saving positions on entry
        self.api_server = api_server  # For checking pause state
        self.risk = risk_state
        self.positions = positions
        self.get_ltp_ts = get_ltp_ts
        self.bar_builder = bar_builder
        
        # Trigger management - using unified components
        self.pending_trades: Dict[str, PendingTrade] = {}
        self.condition_validator = FastConditionValidator(get_ltp_ts, bar_builder)
        self.trigger_factory = FastTriggerConditionFactory()
        
        # Threading
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        # Track latest tick timestamp for accurate expiry checks in backtest
        self._last_tick_ts: Optional[pd.Timestamp] = None
        
        # Config
        self.cfg = load_filters()
        self.entry_cutoff_md = _parse_hhmm_to_md(self.cfg.get("entry_cutoff_hhmm", "14:45"))
        self.eod_md = _parse_hhmm_to_md(self.cfg.get("eod_squareoff_hhmm", "15:10"))
        
        # Staleness check configuration
        trigger_cfg = self.cfg.get("trigger_system", {})
        self.staleness_seconds = float(trigger_cfg.get("staleness_seconds", 1800))
        
        # Hook into BarBuilder's 1m callback
        self.original_1m_handler = bar_builder._on_1m_close
        bar_builder._on_1m_close = self._enhanced_1m_handler

        # Hook into BarBuilder's on_tick for tick-level logging
        self.original_on_tick = bar_builder.on_tick
        bar_builder.on_tick = self._enhanced_on_tick

        logger.info("TriggerAwareExecutor initialized with unified validation and tick logging")

    def _enhanced_on_tick(self, symbol: str, price: float, volume: float, ts: datetime) -> None:
        """
        Enhanced tick handler that logs ticks for pending trades.

        Note: 'price' parameter is required by bar_builder.on_tick signature but not used here.
        We call broker.get_ltp() with entry_zone instead for polymorphic behavior.
        """
        # Call original on_tick first
        if callable(self.original_on_tick):
            try:
                self.original_on_tick(symbol, price, volume, ts)
            except Exception as e:
                logger.exception(f"Original on_tick failed for {symbol}: {e}")

        # Log tick data for symbols with pending trades
        self._log_tick_for_pending_trades(symbol, ts)
    
    def _enhanced_1m_handler(self, symbol: str, bar_1m: pd.Series) -> None:
        """Enhanced 1m handler that validates triggers AND calls original handler"""
        # Call original handler first
        if callable(self.original_1m_handler):
            try:
                self.original_1m_handler(symbol, bar_1m)
            except Exception as e:
                logger.exception(f"Original 1m handler failed for {symbol}: {e}")
        
        # Validate triggers for this symbol
        self._validate_triggers_1m(symbol, bar_1m)
    
    def run_once(self) -> None:
        """Main execution loop - handles new plans and triggered trades"""
        try:
            # 1. Process new trade plans from queue
            new_item = self.oq.get_next(timeout=0.1)
            if new_item:
                self._add_pending_trade(new_item)

            # 2. Poll ALL pending trades for trigger conditions (critical for backtest)
            # In backtest mode, ticks only flow for scanned symbols, so pending symbols
            # that drop out of the scan never get checked. This ensures all pending
            # trades are actively polled on each cycle.
            self._poll_all_pending_trades()

            # 3. Execute trades that have been triggered
            self._execute_triggered_trades()

            # 4. Clean up expired trades
            self._cleanup_expired_trades()

        except Exception as e:
            logger.exception(f"TriggerAwareExecutor.run_once error: {e}")

    def _poll_all_pending_trades(self) -> None:
        """
        Actively poll ALL pending trades to check if entry zone was touched.

        This is critical for backtest mode where ticks only flow for symbols
        currently being scanned. Without this, pending trades for symbols that
        drop out of the scan shortlist would never be checked for triggers.

        Uses broker.get_ltp() with entry_zone for polymorphic behavior:
        - Live: Returns current LTP (real-time price)
        - Backtest: Checks if bar OHLC touched zone, returns zone price or None
        """
        with self._lock:
            pending = [t for t in self.pending_trades.values()
                      if t.state == TradeState.WAITING_TRIGGER]

        if not pending:
            return

        current_ts = self._last_tick_ts if self._last_tick_ts else self._get_current_time()

        for trade in pending:
            try:
                entry_zone = trade.plan.get("entry_zone") or (trade.plan.get("entry") or {}).get("zone")
                if not entry_zone or len(entry_zone) != 2:
                    continue

                side = "BUY" if trade.plan.get("bias", "long") == "long" else "SELL"

                # Broker handles live vs backtest polymorphically
                # Backtest: get_ltp checks if current bar OHLC touched entry_zone
                try:
                    price = self.broker.get_ltp(
                        trade.symbol,
                        entry_zone=entry_zone,
                        side=side
                    )
                except Exception:
                    # Skip if broker can't get price for this symbol
                    continue

                if price is None:
                    continue

                entry_min, entry_max = sorted(entry_zone)

                # Per-setup wide_open: research setups fire on the first available
                # price without zone-membership check. Prod-ready setups keep zone
                # discipline (price must enter entry_zone to trigger).
                from services.config_loader import is_wide_open_for_setup
                if is_wide_open_for_setup(trade.plan.get("strategy")):
                    self._try_trigger_on_tick(trade, price, current_ts)
                    continue

                # Check if price is in or near entry zone
                in_strict_zone = entry_min <= price <= entry_max
                tolerance = 0.0005 * price  # 0.05% tolerance
                in_near_zone = (entry_min - tolerance <= price <= entry_max + tolerance)

                if in_strict_zone or in_near_zone:
                    zone_status = "IN_ZONE" if in_strict_zone else "NEAR_ZONE"
                    logger.info(
                        f"NEAR_ZONE_TRIGGER: {price:.2f} vs [{entry_min:.2f}, {entry_max:.2f}] "
                        f"tolerance={tolerance:.4f} ({0.05}%)"
                    )

                    # Trigger the trade
                    self._try_trigger_on_tick(trade, price, current_ts)

            except Exception as e:
                logger.debug(f"Poll check failed for {trade.symbol}: {e}")

    def _log_tick_for_pending_trades(self, symbol: str, ts: datetime) -> None:
        """
        Check triggers on tick for pending trades.

        Uses broker.get_ltp() with entry_zone for polymorphic behavior:
        - Live/paper: Returns current LTP (real tick price)
        - Backtest: Checks if bar OHLC touched zone, returns zone price or close
        """
        # Update last tick timestamp for expiry checks (critical for backtest accuracy)
        self._last_tick_ts = pd.Timestamp(ts) if ts else None

        with self._lock:
            # Check if this symbol has any pending trades
            pending_for_symbol = [
                t for t in self.pending_trades.values()
                if t.symbol == symbol and t.state == TradeState.WAITING_TRIGGER
            ]

        if not pending_for_symbol:
            return

        # Log tick arrival for enqueued symbols
        logger.debug(
            f"TICK_RECEIVED [{symbol}]: {len(pending_for_symbol)} pending trade(s) "
            f"ts={ts.strftime('%H:%M:%S')}"
        )

        # Process each pending trade for this symbol
        for trade in pending_for_symbol:
            # Try both key paths for entry_zone (flat and nested)
            entry_zone = trade.plan.get("entry_zone") or (trade.plan.get("entry") or {}).get("zone")
            if not entry_zone or len(entry_zone) != 2:
                continue

            # Get price using broker's polymorphic implementation
            side = "BUY" if trade.plan.get("bias", "long") == "long" else "SELL"

            try:
                # Broker handles live vs backtest:
                # - Live: Returns current LTP
                # - Backtest: Checks if bar OHLC touched entry_zone
                price = self.broker.get_ltp(
                    symbol,
                    entry_zone=entry_zone,
                    side=side
                )
            except Exception:
                # Fallback if broker doesn't support entry_zone
                price, _ = self.get_ltp_ts(symbol)
                if price is None:
                    continue

            entry_min, entry_max = sorted(entry_zone)
            bias = trade.plan.get("bias", "long")
            strategy = trade.plan.get("strategy", "unknown")

            # Per-setup wide_open: research setups fire on the first tick without
            # zone-membership check. Mirror of the poll-path bypass above.
            from services.config_loader import is_wide_open_for_setup
            if is_wide_open_for_setup(strategy):
                logger.info(
                    f"WIDE_OPEN_TICK_TRIGGER: {symbol} {strategy} {bias} "
                    f"price={price:.2f} ts={ts.strftime('%H:%M:%S')} trade_id={trade.trade_id}"
                )
                self._try_trigger_on_tick(trade, price, ts)
                continue

            # Check if returned price is in entry zone
            in_strict_zone = entry_min <= price <= entry_max

            # Add near-zone tolerance (0.05%)
            tolerance = 0.0005 * price
            in_near_zone = (entry_min - tolerance <= price <= entry_max + tolerance)

            # Trigger if price is in or near entry zone
            if in_strict_zone or in_near_zone:
                zone_status = "IN_ZONE" if in_strict_zone else "NEAR_ZONE"
                logger.info(
                    f"TICK_{zone_status}: {symbol} {strategy} {bias} "
                    f"price={price:.2f} zone=[{entry_min:.2f}, {entry_max:.2f}] "
                    f"ts={ts.strftime('%H:%M:%S')} trade_id={trade.trade_id}"
                )

                # Trigger on tick - broker handles live vs backtest polymorphically
                self._try_trigger_on_tick(trade, price, ts)
            else:
                # Log when price is outside entry zone (for debugging enqueued symbols)
                distance_pct = abs(price - entry_min) / entry_min * 100 if price < entry_min else abs(price - entry_max) / entry_max * 100
                logger.debug(
                    f"TICK_WAITING [{symbol}]: {strategy} {bias} "
                    f"price={price:.2f} zone=[{entry_min:.2f}, {entry_max:.2f}] "
                    f"distance={distance_pct:.2f}% trade_id={trade.trade_id}"
                )

    def _try_trigger_on_tick(self, trade: PendingTrade, price: float, ts: datetime) -> None:
        """
        Attempt to trigger a trade based on tick-level price action.

        This bypasses the 1-minute bar validation for faster execution when
        price touches the entry zone.
        """
        try:
            with self._lock:
                # Double-check state (might have been triggered already)
                if trade.state != TradeState.WAITING_TRIGGER:
                    return

                # Mark as triggered
                trade.trigger_price = price
                trade.trigger_timestamp = pd.Timestamp(ts)
                trade.state = TradeState.TRIGGERED

                logger.info(
                    f"TICK_TRIGGERED: {trade.symbol} {trade.plan.get('strategy', '')} "
                    f"price={price:.2f} zone={trade.plan.get('entry_zone')} "
                    f"ts={ts.strftime('%H:%M:%S')} trade_id={trade.trade_id}"
                )

        except Exception as e:
            logger.exception(f"Tick trigger failed for {trade.symbol}: {e}")

    def _add_pending_trade(self, item: Dict[str, Any]) -> None:
        """Convert incoming trade plan to pending trade with triggers"""
        try:
            symbol = str(item.get("symbol", ""))
            plan = item.get("plan", {})
            
            if not symbol or not plan:
                logger.warning("Invalid trade item received")
                return

            # Block duplicate entry — symbol already has an open position.
            # Per-setup wide_open: research setups bypass this guard so every
            # detection produces its own trade outcome for cell-mining capture.
            from services.config_loader import is_wide_open_for_setup
            if symbol in self.risk.open_positions and not is_wide_open_for_setup(plan.get("strategy")):
                logger.info(f"PLAN_SKIP | {symbol} | Already has open position, ignoring duplicate plan")
                return

            # Create trade ID — should always be set upstream (StructureEvent
            # auto-mint → TradePlan → orchestrator plan dict → screener_live
            # logs DECISION). Falling back here means a code path bypassed those
            # mint sites; log loudly so the chain breakage is visible. Safety
            # mint stays so the trade still has a usable id rather than crashing.
            trade_id = plan.get("trade_id")
            if not trade_id:
                logger.error(
                    "TRADE_ID_LATE_MINT | %s | strategy=%s — plan reached trigger_aware_executor "
                    "without trade_id; this breaks event-chain traceability",
                    symbol, plan.get("strategy", "unknown"),
                )
                trade_id = f"{symbol}_{uuid.uuid4().hex[:8]}"
            
            primary_triggers, must_conditions, should_conditions = \
                self.trigger_factory.create_conditions_for_strategy(plan)
            
            # Calculate expiry time (default: 45 minutes)
            now = self._get_current_time()
            expiry_minutes = int(self.cfg.get("trigger_expiry_minutes"))
            expiry_time = now + pd.Timedelta(minutes=expiry_minutes)
            
            # Create pending trade
            pending_trade = PendingTrade(
                trade_id=trade_id,
                symbol=symbol,
                plan=plan,
                state=TradeState.WAITING_TRIGGER,
                created_at=now,
                primary_triggers=primary_triggers,
                must_conditions=must_conditions,
                should_conditions=should_conditions,
                expiry_time=expiry_time
            )
            
            with self._lock:
                # Cancel any existing pending trades for same symbol if configured.
                # Per-setup wide_open: research setups skip the cancel so every
                # detection produces its own trade outcome for cell-mining.
                from services.config_loader import is_wide_open_for_setup
                _wide_open = is_wide_open_for_setup(plan.get("strategy"))
                if not _wide_open and self.cfg.get("cancel_existing_pending", True):
                    self._cancel_pending_for_symbol(symbol)

                self.pending_trades[trade_id] = pending_trade
            
            logger.info(
                f"PENDING: {symbol} {plan.get('strategy', '')} "
                f"triggers={len(primary_triggers)} expires_in={expiry_minutes}m"
            )
            
        except Exception as e:
            logger.exception(f"Failed to add pending trade: {e}")
    
    def _validate_triggers_1m(self, symbol: str, bar_1m: pd.Series) -> None:
        """Validate trigger conditions on 1m bar close using unified validator"""
        with self._lock:
            symbol_trades = [t for t in self.pending_trades.values() 
                           if t.symbol == symbol and t.state == TradeState.WAITING_TRIGGER]
        
        if not symbol_trades:
            return
        
        # Get additional context for validation
        context = self._build_validation_context(symbol, bar_1m)
        
        for trade in symbol_trades:
            try:
                trade.last_validation_ts = pd.Timestamp(bar_1m.name) if hasattr(bar_1m, 'name') else self._get_current_time()
                trade.validation_count += 1
                
                # Validate all condition types using unified validator
                primary_satisfied = self._validate_condition_group(
                    trade.primary_triggers, symbol, bar_1m, context
                )
                must_satisfied = self._validate_condition_group(
                    trade.must_conditions, symbol, bar_1m, context
                )
                should_score = self._calculate_should_score(
                    trade.should_conditions, symbol, bar_1m, context
                )
                
                # Calculate overall confidence
                trade.confidence_score = self._calculate_confidence(
                    primary_satisfied, must_satisfied, should_score
                )
                
                # Check if trade should be triggered - REAL TRADER LOGIC
                if primary_satisfied and must_satisfied:
                    entry_zone = (trade.plan.get("entry_zone"))
                    side = "BUY" if (trade.plan.get("bias", "long") == "long") else "SELL"

                    try:
                        ltp = self.broker.get_ltp(
                            trade.symbol,
                            entry_zone=entry_zone if entry_zone else None,
                            side=side,
                            bar_1m=bar_1m.to_dict() if hasattr(bar_1m, "to_dict") else None,
                        )
                        current_price = float(ltp)
                    except Exception:
                        current_price = float(bar_1m.get("close", 0.0))

                    # CRITICAL: Check if price is within entry zone (like real traders)
                    price_in_zone = self._is_price_in_entry_zone(
                        current_price, entry_zone, trade.plan.get("bias"),
                        setup_type=trade.plan.get("strategy"),
                    )

                    if price_in_zone:
                        # Price is in acceptable entry zone AND conditions met - TRIGGER!
                        trade.trigger_price = current_price
                        trade.trigger_timestamp = trade.last_validation_ts
                        trade.state = TradeState.TRIGGERED

                        logger.info(
                            f"TRIGGERED: {symbol} {trade.plan.get('strategy', '')} "
                            f"price={trade.trigger_price:.2f} zone={entry_zone} confidence={trade.confidence_score:.2f}"
                        )
                    else:
                        # Conditions met but price outside zone - wait like real traders
                        logger.debug(
                            f"CONDITIONS MET but price outside zone: {symbol} "
                            f"price={current_price:.2f} zone={entry_zone} - WAITING"
                        )
                
            except Exception as e:
                logger.exception(f"Trigger validation error for {symbol}: {e}")

    def _is_price_in_entry_zone(
        self, current_price: float, entry_zone: List[float], bias: str,
        setup_type: Optional[str] = None,
    ) -> bool:
        """
        Check if current price is within acceptable entry zone (like real traders).

        Args:
            current_price: Current market price
            entry_zone: [min_price, max_price] from plan
            bias: "long" or "short"
            setup_type: plan["strategy"] used to look up per-setup wide_open

        Returns:
            True if price is in zone, False otherwise
        """
        if not entry_zone or len(entry_zone) != 2:
            # No entry zone defined - allow trigger (fallback behavior)
            return True

        # Per-setup wide_open: research setups bypass zone check.
        from services.config_loader import is_wide_open_for_setup
        if is_wide_open_for_setup(setup_type):
            return True

        min_price, max_price = sorted(entry_zone)  # Ensure min < max

        # P0 ENHANCEMENT: Check strict zone first
        in_zone = min_price <= current_price <= max_price

        if in_zone:
            return True

        # P0 IMPROVEMENT: Near-miss tolerance (0.05% as per plan document)
        tolerance = 0.0005 * current_price  # 0.05% tolerance
        near_zone = (min_price - tolerance <= current_price <= max_price + tolerance)

        if near_zone:
            # Accept near-zone entries - this reduces "no-trigger" scenarios
            logger.info(
                f"NEAR_ZONE_TRIGGER: {current_price:.2f} vs [{min_price:.2f}, {max_price:.2f}] "
                f"tolerance={tolerance:.4f} ({tolerance/current_price*100:.3f}%)"
            )
            return True

        # Still outside tolerance - log for analysis
        distance_from_zone = 0
        if current_price < min_price - tolerance:
            distance_from_zone = (min_price - tolerance) - current_price
        elif current_price > max_price + tolerance:
            distance_from_zone = current_price - (max_price + tolerance)

        logger.debug(
            f"Price outside zone+tolerance: {current_price:.2f} vs [{min_price:.2f}, {max_price:.2f}] "
            f"bias={bias} distance={distance_from_zone:.2f} tolerance={tolerance:.4f}"
        )

        return False

    def _validate_condition_group(
        self, 
        conditions: List[TriggerCondition], 
        symbol: str, 
        bar_1m: pd.Series, 
        context: Dict
    ) -> bool:
        """Validate a group of conditions using unified validator"""
        for condition in conditions:
            current_result = self.condition_validator.validate_condition(
                condition, symbol, bar_1m, context
            )
            
            condition.last_check_ts = pd.Timestamp(bar_1m.name) if hasattr(bar_1m, 'name') and bar_1m.name else pd.Timestamp.now()
            
            if current_result:
                condition.consecutive_hits += 1
            else:
                condition.consecutive_hits = 0
            
            condition.last_result = current_result
            
            # Check if consecutive requirement is met
            if condition.consecutive_hits < condition.required_consecutive:
                return False
        
        return True
    
    def _calculate_should_score(
        self, 
        conditions: List[TriggerCondition], 
        symbol: str, 
        bar_1m: pd.Series, 
        context: Dict
    ) -> float:
        """Calculate score for 'should' conditions (0.0 to 1.0)"""
        if not conditions:
            return 1.0
        
        satisfied_count = 0
        for condition in conditions:
            if self.condition_validator.validate_condition(condition, symbol, bar_1m, context):
                satisfied_count += 1
        
        return satisfied_count / len(conditions)
    
    def _calculate_confidence(self, primary_ok: bool, must_ok: bool, should_score: float) -> float:
        """Calculate overall confidence score"""
        if not (primary_ok and must_ok):
            return 0.0
        
        # Base confidence from mandatory conditions
        base_confidence = 0.7
        
        # Bonus from should conditions
        should_bonus = should_score * 0.3
        
        return base_confidence + should_bonus
    
    def _build_validation_context(self, symbol: str, bar_1m: pd.Series) -> Dict[str, Any]:
        """Build context dict for condition validation"""
        context = {}
        
        try:
            # Get current LTP and timestamp
            ltp, ts = self.get_ltp_ts(symbol)
            context["current_ltp"] = ltp
            context["current_ts"] = ts
            
            # Get VWAP if available in bar
            vwap = bar_1m.get("vwap", 0)
            if vwap > 0:
                context["vwap"] = float(vwap)
            
            # Calculate volume ratio if possible
            volume = float(bar_1m.get("volume", 0))
            if volume > 0 and self.bar_builder:
                # Use the validator's volume ratio calculation
                vol_ratio = self.condition_validator._get_volume_ratio(symbol, volume)
                context["vol_ratio"] = vol_ratio
            
            # Add any pending trade entry zones
            with self._lock:
                for trade in self.pending_trades.values():
                    if trade.symbol == symbol:
                        entry_zone = (trade.plan.get("entry") or {}).get("zone", [])
                        if entry_zone:
                            context["entry_zone"] = entry_zone
                        
                        # Add entry min/max if available
                        entry_data = trade.plan.get("entry", {})
                        if "min" in entry_data:
                            context["entry_min"] = float(entry_data["min"])
                        if "max" in entry_data:
                            context["entry_max"] = float(entry_data["max"])
                        break
            
        except Exception as e:
            logger.warning(f"Failed to build validation context for {symbol}: {e}")
        
        return context
    
    def _is_trading_paused(self) -> bool:
        """Check if trading is paused via API server."""
        if not self.api_server:
            return False
        # Import here to avoid circular imports
        from api.state import SessionState
        return self.api_server.state == SessionState.PAUSED

    def _execute_triggered_trades(self) -> None:
        """Execute trades that have been triggered"""
        # Check if trading is paused - skip new entries but keep processing
        if self._is_trading_paused():
            logger.debug("Trading paused - skipping new entries")
            return

        with self._lock:
            triggered_trades = [t for t in self.pending_trades.values()
                              if t.state == TradeState.TRIGGERED]

        expired_trade_ids = []

        for trade in triggered_trades:
            try:
                # Final pre-execution validation
                if not self._final_execution_check(trade):
                    # Trade was marked as expired in _final_execution_check
                    expired_trade_ids.append(trade.trade_id)
                    continue
                
                # Execute the trade
                success = self._place_trade_order(trade)
                
                if success:
                    trade.state = TradeState.EXECUTED
                    logger.debug(f"EXECUTED: {trade.symbol} {trade.plan.get('strategy', '')}")
                else:
                    trade.state = TradeState.EXPIRED
                    logger.warning(f"EXECUTION_FAILED: {trade.symbol}")
                
            except Exception as e:
                logger.exception(f"Trade execution error: {trade.symbol}: {e}")
                trade.state = TradeState.EXPIRED
                expired_trade_ids.append(trade.trade_id)
        
        # Immediately remove expired trades to prevent repeated processing
        if expired_trade_ids:
            with self._lock:
                for trade_id in expired_trade_ids:
                    if trade_id in self.pending_trades:
                        del self.pending_trades[trade_id]
                        logger.debug(f"Immediately removed expired trade: {trade_id}")
        
    def get_performance_summary(self):
        validator_stats = self.condition_validator.get_performance_stats()
        return {
            "pending_trades": self.get_pending_trades_summary(),
            "validation_performance": validator_stats
        }
        
    def _get_current_time(self) -> pd.Timestamp:
            """Get current time from tick stream - unified for live/backtest"""
            # Get time from latest tick data
            try:
                # Use any symbol that we're tracking
                with self._lock:
                    for trade in self.pending_trades.values():
                        _, ts = self.get_ltp_ts(trade.symbol)
                        if ts:
                            return pd.Timestamp(ts)
                
                # Try to get time from bar builder's latest tick
                if hasattr(self.bar_builder, '_ltp'):
                    for symbol, last_tick in self.bar_builder._ltp.items():
                        if hasattr(last_tick, 'ts') and last_tick.ts:
                            return pd.Timestamp(last_tick.ts)
            except:
                pass
            
            # Fallback - but should be rare in production
            return pd.Timestamp.now()