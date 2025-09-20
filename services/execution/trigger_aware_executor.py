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
                # Check minimum distance (0.2% of price or 10 paisa minimum)
                min_distance = max(price * 0.002, 0.10)

                if side == "BUY" and price <= (hard_sl + min_distance):
                    logger.warning(f"REJECTED: {symbol} entry {price:.2f} too close to hard_sl {hard_sl:.2f} (min_distance={min_distance:.2f})")
                    return False
                elif side == "SELL" and price >= (hard_sl - min_distance):
                    logger.warning(f"REJECTED: {symbol} entry {price:.2f} too close to hard_sl {hard_sl:.2f} (min_distance={min_distance:.2f})")
                    return False

            # Place order
            order_args = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": "MARKET",  # Using market orders for trigger execution
                "product": "MIS",
                "variety": "regular",
            }
            
            order_id = self.broker.place_order(**order_args)
            
            # Log execution
            trade_logger.info(
                f"TRIGGER_EXEC | {symbol} | {side} {qty} @ {price:.2f} | "
                f"strategy={plan.get('strategy', '')} | order_id={order_id}"
            )
            
            # Update risk state
            self.risk.open_positions[symbol] = {
                "side": side, 
                "qty": qty, 
                "avg_price": price
            }
            
            # Update shared position store for exit executor
            if self.positions:
                from services.execution.trade_executor import Position
                pos = Position(
                    symbol=symbol,
                    side=side, 
                    qty=qty,
                    avg_price=price,
                    plan=trade.plan
                )
                self.positions.upsert(pos)
            
            return True
            
        except Exception as e:
            logger.exception(f"Order placement failed for {trade.symbol}: {e}")
            return False
    
    def _cleanup_expired_trades(self) -> None:
        """Clean up expired and completed trades"""
        now = self._get_current_time()
        
        with self._lock:
            expired_ids = []
            
            for trade_id, trade in self.pending_trades.items():
                # Remove expired trades
                if trade.expiry_time and now > trade.expiry_time:
                    if trade.state == TradeState.WAITING_TRIGGER:
                        trade.state = TradeState.EXPIRED
                        logger.debug(f"EXPIRED: {trade.symbol} {trade.plan.get('strategy', '')}")
                
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
        trading_logger=None  # Enhanced logging service
    ):
        self.broker = broker
        self.oq = order_queue
        self.trading_logger = trading_logger
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
        
        logger.info("TriggerAwareExecutor initialized with unified validation")
    
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
            
            # 2. Execute trades that have been triggered
            self._execute_triggered_trades()
            
            # 3. Clean up expired trades
            self._cleanup_expired_trades()
            
        except Exception as e:
            logger.exception(f"TriggerAwareExecutor.run_once error: {e}")
    
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