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
from diagnostics.diag_event_log import diag_event_log
from pipelines.breakout_pipeline import BreakoutPipeline

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

                can_enter, adjusted_qty, reason = self.capital_manager.can_enter_position(
                    trade.symbol, qty, price, cap_segment, shadow=is_shadow
                )

                if not can_enter:
                    # This should only happen for capital insufficiency (not max positions, since shadow handles that)
                    logger.warning(f"Capital check failed: {trade.symbol} - {reason}")
                    return False

                # Update plan with adjusted quantity if scaled down (not applicable for shadow trades)
                if adjusted_qty != qty and not is_shadow:
                    logger.info(f"Capital scaling: {trade.symbol} qty {qty} -> {adjusted_qty}")
                    trade.plan["qty"] = adjusted_qty
                    trade.plan["_original_qty"] = qty  # Keep original for reference

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
            side = plan.get("side", "BUY")
            qty = int(plan.get("qty", 0))
            price = trade.trigger_price or plan.get("price")
            
            if qty <= 0:
                logger.warning(f"Invalid qty for {symbol}: {qty}")
                return False

            # Validate entry price vs hard_sl distance
            hard_sl = plan.get("hard_sl")

            logger.info(f"VALIDATION CHECK: {symbol} entry={price:.2f} hard_sl={hard_sl} side={side}")
    
            if hard_sl is not None:
                # Check minimum distance - configurable to avoid hardcoded trading rules
                from config.filters_setup import load_filters
                filters_config = load_filters()
                # KeyError if missing trading parameters
                min_distance_pct = filters_config["min_entry_sl_distance_pct"]
                min_distance_abs = filters_config["min_entry_sl_distance_abs"]
                min_distance = max(price * min_distance_pct, min_distance_abs)

                if side == "BUY" and price <= (hard_sl + min_distance):
                    logger.warning(f"REJECTED: {symbol} entry {price:.2f} too close to hard_sl {hard_sl:.2f} (min_distance={min_distance:.2f})")
                    return False
                elif side == "SELL" and price >= (hard_sl - min_distance):
                    logger.warning(f"REJECTED: {symbol} entry {price:.2f} too close to hard_sl {hard_sl:.2f} (min_distance={min_distance:.2f})")
                    return False

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

            # REMOVED duplicate trade_logger.info() call for TRIGGER_EXEC
            # Reason: Both trade_logger.info() (removed) and trading_logger.log_trigger() (below)
            #         write to the SAME trade_logs.log file, creating duplicate TRIGGER_EXEC entries
            #
            # Evidence from logs/run_bb5bf6d6_20251013_084000/trade_logs.log:
            #   - Line 1: trade_logger format (basic)
            #   - Line 2: trading_logger format (with diagnostics)
            #
            # Decision: Use trading_logger.log_trigger() as single source of truth
            # Benefits:
            #   - No duplicates in trade_logs.log
            #   - Rich diagnostics (confidence_score, validation_count, entry_zone)
            #   - Consistent with EXIT logging (also uses trading_logger only)

            # Enhanced logging: Log TRIGGER event to events.jsonl
            if self.trading_logger:
                trigger_data = {
                    'symbol': symbol,
                    'trade_id': trade.trade_id,
                    'price': price,
                    'qty': qty,
                    'timestamp': str(trade.trigger_timestamp) if trade.trigger_timestamp else str(pd.Timestamp.now()),
                    'strategy': plan.get('strategy', ''),
                    'setup_type': plan.get('setup_type', ''),
                    'regime': plan.get('regime', ''),
                    'order_id': order_id,
                    'side': side,
                    'shadow': plan.get('shadow', False),  # Shadow trade flag
                    'diagnostics': {
                        'confidence_score': trade.confidence_score,
                        'trigger_price': trade.trigger_price,
                        'validation_count': trade.validation_count,
                        'entry_zone': plan.get('entry', {}).get('zone', [])
                    }
                }
                self.trading_logger.log_trigger(trigger_data)

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
                from services.execution.trade_executor import Position

                # CRITICAL FIX: Recalculate targets based on ACTUAL entry price
                # Original targets were calculated from entry_ref_price, but actual
                # trigger price may differ. Targets must be R-based from actual entry.
                adjusted_plan = self._recalculate_targets_for_actual_entry(
                    trade.plan, price, side
                )

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

            return True
            
        except Exception as e:
            logger.exception(f"Order placement failed for {trade.symbol}: {e}")
            return False

    def _recalculate_targets_for_actual_entry(
        self, plan: Dict[str, Any], actual_entry: float, side: str
    ) -> Dict[str, Any]:
        """
        Recalculate targets based on ACTUAL entry price, not planned entry.

        Problem:
        - Pipeline calculates targets from entry_ref_price (e.g., 256.43)
        - Actual trigger may occur at different price (e.g., 261.40)
        - If targets aren't recalculated, T2 gives wrong R-multiple

        Solution:
        - For ORB setups: Use OR range-based targets (pro standard)
        - For other setups: Use R-multiples from actual entry

        Van Tharp: Targets must be based on YOUR entry, not some theoretical entry.
        """
        import copy
        adjusted_plan = copy.deepcopy(plan)

        try:
            # Check if this is an ORB setup - use OR range-based targets (pro standard)
            # Delegate to breakout pipeline which has the config (no hardcoded values here)
            strategy = plan.get("strategy", "") or ""
            if "orb" in strategy.lower():
                breakout_pipeline = BreakoutPipeline()
                return breakout_pipeline.recalculate_orb_targets_at_trigger(adjusted_plan, actual_entry, side)
            # Get stop data from plan - exec_item now includes full "stop" dict
            stop_data = plan.get("stop", {})
            hard_sl = stop_data.get("hard")
            original_rps = stop_data.get("risk_per_share")

            # Get original entry price
            original_entry = plan.get("entry_ref_price") or plan.get("price")

            # Can't recalculate without stop loss info
            if hard_sl is None or original_rps is None or original_rps <= 0:
                logger.info(f"TARGET_RECALC_SKIP: missing data hard_sl={hard_sl} orig_entry={original_entry} rps={original_rps}")
                return adjusted_plan

            # Calculate ACTUAL risk per share based on actual entry
            if side.upper() == "BUY":
                actual_rps = actual_entry - hard_sl
            else:
                actual_rps = hard_sl - actual_entry

            # Skip if rps is invalid
            if actual_rps <= 0:
                logger.warning(f"Invalid actual_rps={actual_rps:.2f}, using original targets")
                return adjusted_plan

            # Get original targets
            original_targets = plan.get("targets", [])
            if len(original_targets) < 2:
                logger.info(f"TARGET_RECALC_SKIP: less than 2 targets, got {len(original_targets)}")
                return adjusted_plan

            # original_entry was already calculated above when computing original_rps

            # Use explicit R-multiples from plan (rr field) - NOT derived from levels
            # The planner may cap/adjust levels for structure, but rr represents intended R
            t1_orig = original_targets[0].get("level", 0)
            t2_orig = original_targets[1].get("level", 0)

            # Get planned R-multiples directly from targets (preferred)
            # Fall back to deriving from levels only if rr not present
            t1_r = original_targets[0].get("rr") or original_targets[0].get("r_multiple")
            t2_r = original_targets[1].get("rr") or original_targets[1].get("r_multiple")

            # If rr not in plan, derive from levels (legacy fallback)
            if t1_r is None:
                if side.upper() == "BUY":
                    t1_r = (t1_orig - original_entry) / original_rps if original_rps > 0 else 1.5
                else:
                    t1_r = (original_entry - t1_orig) / original_rps if original_rps > 0 else 1.5

            if t2_r is None:
                if side.upper() == "BUY":
                    t2_r = (t2_orig - original_entry) / original_rps if original_rps > 0 else 2.0
                else:
                    t2_r = (original_entry - t2_orig) / original_rps if original_rps > 0 else 2.0

            # Recalculate targets from actual entry using same R-multiples
            if side.upper() == "BUY":
                new_t1 = actual_entry + (t1_r * actual_rps)
                new_t2 = actual_entry + (t2_r * actual_rps)
            else:
                new_t1 = actual_entry - (t1_r * actual_rps)
                new_t2 = actual_entry - (t2_r * actual_rps)

            # Update targets in plan
            adjusted_plan["targets"] = [
                {"level": round(new_t1, 2), "r_multiple": round(t1_r, 2)},
                {"level": round(new_t2, 2), "r_multiple": round(t2_r, 2)}
            ]

            # Update stop info with actual rps - handle both exec_item and full plan structures
            if "stop" in adjusted_plan and isinstance(adjusted_plan["stop"], dict):
                adjusted_plan["stop"]["risk_per_share"] = round(actual_rps, 2)
            # Also store at top level for exec_item structure
            adjusted_plan["risk_per_share"] = round(actual_rps, 2)
            adjusted_plan["actual_entry"] = round(actual_entry, 2)

        except Exception as e:
            logger.warning(f"Target recalculation failed: {e}, using original targets")

        return adjusted_plan

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
        persistence=None  # Position persistence for crash recovery
    ):
        self.broker = broker
        self.oq = order_queue
        self.trading_logger = trading_logger
        self.capital_manager = capital_manager
        self.persistence = persistence  # For saving positions on entry
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

                # Also log to trade logger for detailed analysis
                if self.trading_logger:
                    try:
                        self.trading_logger.log_tick_in_zone(
                            symbol=symbol,
                            price=price,
                            entry_zone=entry_zone,
                            zone_status=zone_status,
                            timestamp=ts,
                            trade_id=trade.trade_id,
                            strategy=strategy,
                            bias=bias
                        )
                    except Exception as e:
                        logger.debug(f"Trading logger tick log failed: {e}")

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
            
            # Create trade ID
            trade_id = plan.get("trade_id") or f"{symbol}_{uuid.uuid4().hex[:8]}"
            
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
                # Cancel any existing pending trades for same symbol if configured
                if self.cfg.get("cancel_existing_pending", True):
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
                    price_in_zone = self._is_price_in_entry_zone(current_price, entry_zone, trade.plan.get("bias"))

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

    def _is_price_in_entry_zone(self, current_price: float, entry_zone: List[float], bias: str) -> bool:
        """
        Check if current price is within acceptable entry zone (like real traders).

        Args:
            current_price: Current market price
            entry_zone: [min_price, max_price] from plan
            bias: "long" or "short"

        Returns:
            True if price is in zone, False otherwise
        """
        if not entry_zone or len(entry_zone) != 2:
            # No entry zone defined - allow trigger (fallback behavior)
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
    
    def _execute_triggered_trades(self) -> None:
        """Execute trades that have been triggered"""
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