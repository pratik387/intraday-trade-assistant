"""
trading_logger.py
----------------
Enhanced logging infrastructure for trading system analytics.

Features:
- Multi-stream logging (events, analytics, performance)
- Trade lifecycle tracking with unique IDs
- Real-time performance metrics
- Analytics-friendly data structure
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class TradeLifecycle:
    """Track a trade through its complete lifecycle"""
    lifecycle_id: str
    trade_id: str
    symbol: str
    stage: str  # DECISION, TRIGGER, EXIT
    timestamp: str
    data: Dict[str, Any]
    elapsed_from_decision: Optional[int] = None  # seconds


class TradingLogger:
    """Enhanced logging service for trading analytics"""

    # ========================================================================
    # ZERODHA INTRADAY EQUITY CHARGES (post Oct 1, 2024 "true-to-label")
    # Reference: https://zerodha.com/charges
    # ========================================================================
    BROKERAGE_RATE = 0.0003         # 0.03% of order value
    BROKERAGE_CAP = 20.0            # Rs 20 per order cap (whichever is lower)
    STT_RATE = 0.00025              # 0.025% on SELL side only
    EXCHANGE_RATE = 0.0000297       # NSE charges 0.00297% (total turnover)
    SEBI_RATE = 0.000001            # Rs 10 per crore (total turnover)
    IPFT_RATE = 0.000001            # Rs 10 per crore NSE Investor Protection Fund (total turnover)
    STAMP_DUTY_RATE = 0.00003       # 0.003% on BUY side
    GST_RATE = 0.18                 # 18% on (brokerage + exchange + SEBI + IPFT)

    @classmethod
    def calculate_trade_fees(cls, entry_price: float, exit_price: float, qty: int,
                            is_entry_order: bool = True) -> Dict[str, float]:
        """
        Calculate Zerodha intraday equity fees for a trade/exit.

        Args:
            entry_price: Entry price per share
            exit_price: Exit price per share
            qty: Quantity traded
            is_entry_order: True if this exit includes entry order fees (first exit of trade)

        Returns:
            Dict with fee breakdown and total
        """
        entry_turnover = entry_price * qty
        exit_turnover = exit_price * qty

        # Brokerage: min(0.03% × order_value, Rs 20) per order
        exit_brokerage = min(cls.BROKERAGE_RATE * exit_turnover, cls.BROKERAGE_CAP)
        brokerage = exit_brokerage
        if is_entry_order:
            entry_brokerage = min(cls.BROKERAGE_RATE * entry_turnover, cls.BROKERAGE_CAP)
            brokerage += entry_brokerage

        # STT: 0.025% on sell side only
        stt = exit_turnover * cls.STT_RATE

        # Exchange/SEBI/IPFT: on each order's turnover
        leg_turnover = exit_turnover + (entry_turnover if is_entry_order else 0)
        exchange = leg_turnover * cls.EXCHANGE_RATE
        sebi = leg_turnover * cls.SEBI_RATE
        ipft = leg_turnover * cls.IPFT_RATE

        # Stamp duty: 0.003% on buy side only
        stamp_duty = entry_turnover * cls.STAMP_DUTY_RATE if is_entry_order else 0

        # GST: 18% on (brokerage + exchange + SEBI + IPFT)
        gst = (brokerage + exchange + sebi + ipft) * cls.GST_RATE

        total = brokerage + stt + exchange + sebi + ipft + stamp_duty + gst

        return {
            'brokerage': round(brokerage, 2),
            'stt': round(stt, 2),
            'exchange': round(exchange, 2),
            'sebi': round(sebi, 2),
            'ipft': round(ipft, 2),
            'stamp_duty': round(stamp_duty, 2),
            'gst': round(gst, 2),
            'total_fees': round(total, 2)
        }

    def __init__(self, session_id: str, log_dir: Path):
        self.session_id = session_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Trade lifecycle tracking
        self.trade_lifecycles: Dict[str, TradeLifecycle] = {}
        self.session_start = datetime.now()

        # Performance tracking
        self.session_stats = {
            'total_decisions': 0,
            'triggered_trades': 0,
            'completed_trades': 0,
            'total_pnl': 0.0,
            'wins': 0,
            'losses': 0,
            'breakevens': 0
        }

        # Initialize loggers
        self._setup_loggers()

    def _setup_loggers(self):
        """Setup multiple logging streams"""
        
        # Events logger (existing format + analytics)
        self.events_logger = self._create_logger(
            'events', 
            self.log_dir / 'events.jsonl'
        )
        
        # Analytics logger (clean, triggered trades only)
        self.analytics_logger = self._create_logger(
            'analytics',
            self.log_dir / 'analytics.jsonl'
        )
        
        # Performance logger (session summaries)
        self.performance_logger = self._create_logger(
            'performance',
            self.log_dir / 'performance.json'
        )
        
        # Trade logs (existing format)
        self.trade_logger = self._create_logger(
            'trade_logs',
            self.log_dir / 'trade_logs.log'
        )
    
    def _create_logger(self, name: str, log_file: Path) -> logging.Logger:
        """Create a logger with file handler"""
        logger = logging.getLogger(f"{self.session_id}_{name}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add file handler
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # JSON format for structured logs
        if name in ['events', 'analytics']:
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s — %(levelname)s — %(name)s — %(message)s'
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
        return logger
    
    def log_decision(self, trade_data: Dict[str, Any]):
        """Log a trading decision with analytics enhancement"""
        
        # Generate unique lifecycle ID
        lifecycle_id = f"{trade_data.get('symbol', 'UNK')}_{uuid.uuid4().hex[:8]}"
        
        # Add analytics fields
        enhanced_data = self._add_analytics_fields(trade_data)
        enhanced_data['lifecycle_id'] = lifecycle_id

        # NOTE: Live session stats updates removed - performance tracking now done in post-processing only
        
        # Log to events stream
        self.events_logger.info(json.dumps(enhanced_data))
        
        # Store lifecycle for tracking
        self.trade_lifecycles[lifecycle_id] = TradeLifecycle(
            lifecycle_id=lifecycle_id,
            trade_id=trade_data.get('trade_id', ''),
            symbol=trade_data.get('symbol', ''),
            stage='DECISION',
            timestamp=enhanced_data.get('ts', ''),
            data=enhanced_data
        )

        # NOTE: Live performance summary updates removed - performance tracking now done in post-processing only
    
    def log_trigger(self, trade_data: Dict[str, Any]):
        """Log a trade trigger execution"""

        # Find matching lifecycle (but don't require it - defensive logging)
        lifecycle_id = self._find_lifecycle_id(trade_data)

        if lifecycle_id:
            # Update existing lifecycle if found
            lifecycle = self.trade_lifecycles[lifecycle_id]
            lifecycle.stage = 'TRIGGER'
            lifecycle.elapsed_from_decision = self._calculate_elapsed(lifecycle)

        # ALWAYS log TRIGGER event to events.jsonl (moved outside if block)
        # This ensures TRIGGER events are logged even if lifecycle tracking fails
        # (e.g., when diag_event_log logs DECISION instead of trading_logger.log_decision)
        trigger_event = {
            'schema_version': 'trade.v1',
            'type': 'TRIGGER',
            'run_id': None,
            'trade_id': trade_data.get('trade_id', ''),
            'symbol': trade_data.get('symbol', ''),
            'ts': trade_data.get('timestamp', str(pd.Timestamp.now())),
            'trigger': {
                'actual_price': trade_data.get('price', 0),
                'qty': trade_data.get('qty', 0),
                'strategy': trade_data.get('strategy', ''),
                'order_id': trade_data.get('order_id', ''),
                'side': trade_data.get('side', 'BUY'),
                'diagnostics': trade_data.get('diagnostics', {})
            }
        }
        self.events_logger.info(json.dumps(trigger_event))

        # Log to trade logs (existing format)
        symbol = trade_data.get('symbol', '')
        qty = trade_data.get('qty', 0)
        price = trade_data.get('price', 0)
        strategy = trade_data.get('strategy', '')
        order_id = trade_data.get('order_id', '')
        side = trade_data.get('side', 'BUY')

        self.trade_logger.info(
            f"TRIGGER_EXEC | {symbol} | {side} {qty} @ {price} | strategy={strategy} | order_id={order_id}"
        )

        # NOTE: Live performance summary updates removed - performance tracking now done in post-processing only
    
    def log_exit(self, trade_data: Dict[str, Any]):
        """Log a trade exit"""
        
        # Find matching lifecycle (create one if none exists)
        lifecycle_id = self._find_or_create_lifecycle_id(trade_data)
        
        if lifecycle_id:
            # Update lifecycle
            lifecycle = self.trade_lifecycles[lifecycle_id]
            lifecycle.stage = 'EXIT'
            lifecycle.elapsed_from_decision = self._calculate_elapsed(lifecycle)

            # NOTE: Live session stats updates removed - performance tracking now done in post-processing only
            # NOTE: Analytics logging removed - analytics.jsonl populated from events.jsonl in post-processing to avoid duplicates
        
        # Log EXIT event to events.jsonl with diagnostics
        exit_event = {
            'schema_version': 'trade.v1',
            'type': 'EXIT',
            'run_id': None,
            'trade_id': trade_data.get('trade_id', ''),
            'symbol': trade_data.get('symbol', ''),
            'ts': trade_data.get('timestamp', str(pd.Timestamp.now())),
            'exit': {
                'price': trade_data.get('exit_price', 0),
                'qty': trade_data.get('qty', 0),
                'reason': trade_data.get('reason', ''),
                'pnl': trade_data.get('pnl', 0),
                'diagnostics': trade_data.get('diagnostics', {})
            }
        }
        self.events_logger.info(json.dumps(exit_event))
        
        # Log to trade logs (existing format)
        symbol = trade_data.get('symbol', '')
        qty = trade_data.get('qty', 0)
        entry_price = trade_data.get('entry_price', 0)
        exit_price = trade_data.get('exit_price', 0)
        pnl = trade_data.get('pnl', 0)
        reason = trade_data.get('reason', '')
        
        self.trade_logger.info(
            f"EXIT | {symbol} | Qty: {qty} | Entry: Rs.{entry_price} | Exit: Rs.{exit_price} | PnL: Rs.{pnl} {reason}"
        )

        # NOTE: Live performance summary updates removed - performance tracking now done in post-processing only
    
    def _add_analytics_fields(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add derived analytics fields to trade data"""
        enhanced = trade_data.copy()
        
        # Add analytics section
        analytics = {}
        
        # Setup quality score (derived from multiple factors)
        if 'features' in enhanced and 'plan' in enhanced:
            rank_score = enhanced['features'].get('rank_score', 0)
            structural_rr = enhanced['plan'].get('quality', {}).get('structural_rr', 0)
            acceptance_status = enhanced['plan'].get('quality', {}).get('acceptance_status', 'poor')

            # Graduated acceptance scoring for analytics
            acceptance_score = {
                "excellent": 1.0,
                "good": 0.6,
                "fair": 0.3,
                "poor": 0.0
            }.get(acceptance_status, 0.0)

            # Simple quality score calculation
            analytics['setup_quality_score'] = (
                rank_score * 3 +
                structural_rr * 2 +
                acceptance_score
            )
        
        # Regime confidence
        if 'decision' in enhanced:
            regime = enhanced['decision'].get('regime', '')
            analytics['regime_confidence'] = 0.8 if regime in ['trend_up', 'trend_down'] else 0.5
        
        # Time decay factor (setups get stale)
        analytics['time_decay_factor'] = 1.0  # Could be enhanced based on market hours
        
        # Execution probability (based on historical data)
        analytics['execution_probability'] = 0.15  # Updated from historical 12.1% + buffer
        
        enhanced['analytics'] = analytics
        return enhanced
    
    def _extract_analytics_fields(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key fields for analytics logging (simplified version)"""
        return {
            'symbol': trade_data.get('symbol', ''),
            'trade_id': trade_data.get('trade_id', ''),
            'timestamp': trade_data.get('timestamp', ''),
            'qty': trade_data.get('qty', 0),
            'entry_price': trade_data.get('entry_price', 0),
            'exit_price': trade_data.get('exit_price', 0),
            'reason': trade_data.get('reason', ''),
        }
    
    def _should_trigger(self, trade_data: Dict[str, Any]) -> bool:
        """Predict if a trade should trigger based on quality filters"""
        
        # Apply the same filters as trade_decision_gate
        if 'features' in trade_data and 'rank_score' in trade_data['features']:
            if trade_data['features']['rank_score'] < 2.0:
                return False
        
        if 'plan' in trade_data and 'quality' in trade_data['plan']:
            structural_rr = trade_data['plan']['quality'].get('structural_rr', 0)
            if structural_rr < 1.2:
                return False
        
        return True
    
    def _extract_analytics_fields(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key fields for analytics logging"""
        return {
            'symbol': trade_data.get('symbol', ''),
            'trade_id': trade_data.get('trade_id', ''),
            'timestamp': trade_data.get('timestamp', ''),
            'setup_type': trade_data.get('setup_type', ''),
            'regime': trade_data.get('regime', ''),
            'entry_price': trade_data.get('price', 0),
            'qty': trade_data.get('qty', 0),
            'risk_amount': trade_data.get('risk_amount', 500),
            'strategy': trade_data.get('strategy', '')
        }
    
    def _find_lifecycle_id(self, trade_data: Dict[str, Any]) -> Optional[str]:
        """Find matching lifecycle ID for a trade"""
        trade_id = trade_data.get('trade_id', '')
        symbol = trade_data.get('symbol', '')
        
        # Search by trade_id first, then symbol
        for lifecycle_id, lifecycle in self.trade_lifecycles.items():
            if lifecycle.trade_id == trade_id or lifecycle.symbol == symbol:
                return lifecycle_id
        
        return None
    
    def _calculate_elapsed(self, lifecycle: TradeLifecycle) -> int:
        """Calculate seconds elapsed from decision"""
        try:
            decision_time = datetime.fromisoformat(lifecycle.timestamp.replace('Z', '+00:00'))
            now = datetime.now(decision_time.tzinfo) if decision_time.tzinfo else datetime.now()
            return int((now - decision_time).total_seconds())
        except:
            return 0
    
    def _find_or_create_lifecycle_id(self, trade_data: Dict[str, Any]) -> Optional[str]:
        """Find existing lifecycle or create new one for this trade"""
        symbol = trade_data.get('symbol', '')
        trade_id = trade_data.get('trade_id', '')
        
        # Try to find existing lifecycle by symbol or trade_id
        for lifecycle_id, lifecycle in self.trade_lifecycles.items():
            if (lifecycle.symbol == symbol and 
                (not trade_id or lifecycle.trade_id == trade_id)):
                return lifecycle_id
        
        # Create new lifecycle for this exit
        lifecycle_id = str(uuid.uuid4())
        lifecycle = TradeLifecycle(
            lifecycle_id=lifecycle_id,
            symbol=symbol,
            trade_id=trade_id,
            timestamp=trade_data.get('timestamp', datetime.now().isoformat()),
            stage='EXIT',  # This is an exit-only lifecycle
            data=trade_data  # Include the trade data
        )
        self.trade_lifecycles[lifecycle_id] = lifecycle
        return lifecycle_id
    
    def _update_performance_summary(self):
        """Update the performance summary file"""

        # Calculate derived metrics
        execution_rate = (
            self.session_stats['triggered_trades'] / self.session_stats['total_decisions']
            if self.session_stats['total_decisions'] > 0 else 0
        )

        win_rate = (
            self.session_stats['wins'] / self.session_stats['completed_trades']
            if self.session_stats['completed_trades'] > 0 else 0
        )

        # Execution quality stats
        avg_slippage = (
            sum(self.slippage_list) / len(self.slippage_list)
            if hasattr(self, 'slippage_list') and self.slippage_list else 0
        )

        summary = {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                **self.session_stats,
                'execution_rate': round(execution_rate, 3),
                'win_rate': round(win_rate, 3)
            },
            'trades': getattr(self, 'trades_list', []),
            'execution': {
                'avg_slippage_bps': round(avg_slippage, 1),
                'total_fees': round(getattr(self, 'total_fees', 0), 2)
            }
        }

        # Add capital stats if available
        capital_report = getattr(self, 'capital_report', None)
        if capital_report and capital_report.get('mode') != 'unlimited':
            cap_stats = capital_report.get('capital_stats', {})
            summary['capital'] = {
                'mode': capital_report.get('mode', 'unknown'),
                'initial': cap_stats.get('initial_capital', 0),
                'max_used': round(cap_stats.get('max_capital_used', 0), 0),
                'max_utilization_pct': round(cap_stats.get('max_utilization_pct', 0), 1)
            }

        # Write to performance file with explicit flush and error handling
        performance_file = self.log_dir / 'performance.json'
        try:
            with open(performance_file, 'w') as f:
                json.dump(summary, f, indent=2)
                f.flush()
                import os
                os.fsync(f.fileno())

            # Verify file was written successfully
            if performance_file.exists() and performance_file.stat().st_size > 0:
                print(f"[analytics] Performance summary written: {performance_file.stat().st_size} bytes")
            else:
                print(f"[analytics] ERROR: Performance file empty or missing after write")

        except Exception as e:
            print(f"[analytics] ERROR: Failed to write performance.json: {e}")
    
    def generate_csv_report(self):
        """Generate CSV report from events.jsonl"""
        try:
            from diagnostics.diagnostics_report_builder import build_csv_from_events
            csv_path = build_csv_from_events(log_dir=self.log_dir)

            # Verify CSV was created successfully
            if csv_path and csv_path.exists() and csv_path.stat().st_size > 0:
                print(f"[analytics] CSV report written: {csv_path.stat().st_size} bytes")
                return csv_path
            else:
                print(f"[analytics] ERROR: CSV file empty or missing after generation")
                return None

        except Exception as e:
            print(f"[analytics] ERROR: Could not generate CSV report: {e}")
            return None
    
    def populate_analytics_from_events(self, capital_report: Optional[Dict[str, Any]] = None):
        """Populate analytics.jsonl and calculate comprehensive performance metrics from events.jsonl

        Args:
            capital_report: Optional capital manager final report (from capital_manager.get_final_report())
        """
        self.capital_report = capital_report  # Store for _update_performance_summary
        try:
            events_file = self.log_dir / 'events.jsonl'
            if not events_file.exists():
                return

            # Clear analytics.jsonl to avoid duplicates from previous real-time logging
            analytics_file = self.log_dir / 'analytics.jsonl'
            try:
                if analytics_file.exists():
                    # Try to delete the file, but if locked, truncate it instead
                    try:
                        analytics_file.unlink()
                    except PermissionError:
                        # File is locked, truncate it instead
                        with open(analytics_file, 'w') as f:
                            f.truncate(0)
            except Exception as e:
                print(f"[analytics] Warning: Could not clear analytics.jsonl: {e}")

            # Recreate analytics logger with fresh file
            self.analytics_logger = self._create_logger(
                'analytics',
                analytics_file
            )

            # Reset session stats for clean calculation
            self.session_stats = {
                'total_decisions': 0,
                'triggered_trades': 0,
                'completed_trades': 0,
                'total_pnl': 0.0,
                'wins': 0,
                'losses': 0,
                'breakevens': 0
            }

            # Trade-by-trade list for performance.json
            self.trades_list = []
            self.total_fees = 0.0
            self.slippage_list = []  # For avg calculation

            # Parse all events and organize by type
            decisions = {}  # trade_id -> decision_event
            triggers = {}   # trade_id -> trigger_event
            exits = {}      # trade_id -> list of exit_events (to handle partial exits)
            shadow_trade_ids = set()  # Track shadow trades to exclude from performance

            with open(events_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        event_type = event.get('type')
                        trade_id = event.get('trade_id')

                        if event_type == 'DECISION':
                            decisions[trade_id] = event
                            self.session_stats['total_decisions'] += 1

                        elif event_type == 'TRIGGER':
                            triggers[trade_id] = event
                            # Check if this is a shadow trade
                            trigger_data = event.get('trigger', {})
                            order_id = trigger_data.get('order_id', '')
                            is_shadow = trigger_data.get('shadow', False) or order_id.startswith('shadow-')
                            if is_shadow:
                                shadow_trade_ids.add(trade_id)

                        elif event_type == 'EXIT':
                            # Support multiple exits per trade (partial exits)
                            if trade_id not in exits:
                                exits[trade_id] = []
                            exits[trade_id].append(event)

                    except Exception:
                        continue

            # Calculate performance metrics by pairing DECISION + EXIT events
            # IMPORTANT: Exclude shadow trades from performance calculations
            for trade_id, exit_events in exits.items():
                # Skip shadow trades - they're simulated and don't reflect real performance
                if trade_id in shadow_trade_ids:
                    continue

                decision_event = decisions.get(trade_id)
                trigger_event = triggers.get(trade_id)

                if decision_event:
                    # Calculate complete trade metrics from ALL exits (sum partial exits)
                    pnl = self._calculate_trade_pnl_all_exits(decision_event, exit_events, trigger_event)

                    # Update session stats
                    self.session_stats['completed_trades'] += 1
                    self.session_stats['total_pnl'] += pnl

                    if pnl > 0:
                        self.session_stats['wins'] += 1
                    elif pnl < 0:
                        self.session_stats['losses'] += 1
                    else:
                        self.session_stats['breakevens'] += 1

                    # Build trade summary for performance.json
                    symbol = decision_event.get('symbol', '').replace('NSE:', '')
                    setup_type = decision_event.get('decision', {}).get('setup_type', '')

                    # Entry price from trigger or decision
                    if trigger_event and 'trigger' in trigger_event:
                        entry_price = trigger_event['trigger'].get('actual_price', 0)
                        ref_price = decision_event['plan']['entry'].get('reference', entry_price)
                        if ref_price > 0:
                            slippage_bps = abs(entry_price - ref_price) / ref_price * 10000
                            self.slippage_list.append(slippage_bps)
                    else:
                        entry_price = decision_event['plan']['entry'].get('reference', 0)

                    # Final exit price (last exit)
                    final_exit = exit_events[-1]
                    exit_price = final_exit['exit'].get('price', 0)
                    exit_reason = final_exit['exit'].get('reason', '')

                    # Calculate fees for this trade
                    total_qty = sum(e['exit'].get('qty', 0) for e in exit_events)
                    if entry_price > 0 and exit_price > 0 and total_qty > 0:
                        fees = self.calculate_trade_fees(entry_price, exit_price, total_qty)
                        self.total_fees += fees['total_fees']

                    # Add to trades list
                    self.trades_list.append({
                        'symbol': symbol,
                        'setup': setup_type,
                        'entry': round(entry_price, 2),
                        'exit': round(exit_price, 2),
                        'pnl': round(pnl, 2),
                        'exit_reason': exit_reason
                    })

                    # FIX: Log analytics for EACH exit event (captures partial exits)
                    # Each exit gets its own analytics record with its specific qty/price/pnl
                    for i, exit_ev in enumerate(exit_events):
                        # Calculate PnL for this specific exit
                        exit_pnl = self._calculate_single_exit_pnl(decision_event, exit_ev, trigger_event)
                        # Mark if this is the final exit
                        is_final = (i == len(exit_events) - 1)
                        analytics_data = self._create_enhanced_analytics(
                            decision_event, exit_ev, trigger_event, exit_pnl
                        )
                        # Add exit sequence info
                        analytics_data['exit_sequence'] = i + 1
                        analytics_data['total_exits'] = len(exit_events)
                        analytics_data['is_final_exit'] = is_final
                        if is_final:
                            analytics_data['total_trade_pnl'] = pnl  # Include total PnL on final exit
                        self.analytics_logger.info(json.dumps(analytics_data))

            # Count triggered trades (exclude shadow trades)
            self.session_stats['triggered_trades'] = len(triggers) - len(shadow_trade_ids)

            # Update performance summary with calculated metrics
            self._update_performance_summary()

            # Generate CSV report for diagnostics
            self.generate_csv_report()

            # Final validation
            print(f"[analytics] Enhanced analytics populated: {self.session_stats['completed_trades']} trades processed")

        except Exception as e:
            print(f"[analytics] ERROR: Could not populate analytics from events: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_trade_pnl_all_exits(self, decision_event: Dict[str, Any], exit_events: list,
                                      trigger_event: Optional[Dict[str, Any]] = None) -> float:
        """Calculate total PnL from all exit events (handles partial exits)

        CRITICAL FIX: This method now ALWAYS recalculates PnL from prices instead of
        trusting 'pnl' fields in exit events, which may be missing or incorrect for partial exits.
        """
        try:
            # Get entry price (prefer actual trigger price, fallback to decision reference)
            if trigger_event and 'trigger' in trigger_event:
                entry_price = float(trigger_event['trigger'].get('actual_price', 0))
            else:
                entry_price = float(decision_event['plan']['entry'].get('reference', 0))

            # Validate entry price
            if entry_price <= 0:
                print(f"[analytics] WARNING: Invalid entry price {entry_price} for {decision_event.get('symbol')}")
                return 0.0

            bias = decision_event['plan'].get('bias', 'long')
            total_pnl = 0.0

            # CRITICAL: Calculate PnL for EACH exit individually (handles partial exits correctly)
            for i, exit_event in enumerate(exit_events):
                exit_price = float(exit_event['exit'].get('price', 0))
                qty = int(exit_event['exit'].get('qty', 0))

                if exit_price <= 0 or qty <= 0:
                    print(f"[analytics] WARNING: Invalid exit data price={exit_price} qty={qty} for exit {i+1}")
                    continue

                # Calculate PnL based on trade direction
                if bias.lower() == 'long':
                    pnl = (exit_price - entry_price) * qty
                else:  # short
                    pnl = (entry_price - exit_price) * qty

                total_pnl += pnl

            return round(total_pnl, 2)

        except Exception as e:
            print(f"[analytics] ERROR: Failed to calculate PnL for {decision_event.get('symbol')}: {e}")
            return 0.0

    def _calculate_single_exit_pnl(self, decision_event: Dict[str, Any], exit_event: Dict[str, Any],
                                   trigger_event: Optional[Dict[str, Any]] = None) -> float:
        """Calculate PnL for a single exit event (used for partial exit analytics)"""
        return self._calculate_trade_pnl(decision_event, exit_event, trigger_event)

    def _calculate_trade_pnl(self, decision_event: Dict[str, Any], exit_event: Dict[str, Any],
                           trigger_event: Optional[Dict[str, Any]] = None) -> float:
        """Calculate PnL for a single exit (legacy method, kept for compatibility)"""
        try:
            # Get entry price (prefer actual trigger price, fallback to decision reference)
            if trigger_event and 'trigger' in trigger_event:
                entry_price = float(trigger_event['trigger'].get('actual_price', 0))
            else:
                entry_price = float(decision_event['plan']['entry'].get('reference', 0))

            # Get exit details
            exit_price = float(exit_event['exit'].get('price', 0))
            qty = int(exit_event['exit'].get('qty', 0))
            bias = decision_event['plan'].get('bias', 'long')

            # Calculate PnL based on trade direction
            if bias.lower() == 'long':
                pnl = (exit_price - entry_price) * qty
            else:  # short
                pnl = (entry_price - exit_price) * qty

            return round(pnl, 2)

        except Exception:
            return 0.0

    def _create_enhanced_analytics(self, decision_event: Dict[str, Any], exit_event: Dict[str, Any],
                                 trigger_event: Optional[Dict[str, Any]], pnl: float) -> Dict[str, Any]:
        """Create enhanced analytics entry combining decision, trigger, and exit data"""
        # Start with basic exit data
        analytics = {
            'symbol': exit_event.get('symbol', ''),
            'trade_id': exit_event.get('trade_id', ''),
            'timestamp': exit_event.get('ts', ''),
            'stage': 'EXIT',
            'qty': exit_event['exit'].get('qty', 0),
            'exit_price': exit_event['exit'].get('price', 0),
            'reason': exit_event['exit'].get('reason', ''),
            'pnl': pnl,
            'lifecycle_id': exit_event.get('trade_id', ''),
        }

        # Add entry information from decision
        if 'plan' in decision_event:
            plan = decision_event['plan']
            analytics.update({
                'entry_reference': plan['entry'].get('reference', 0),
                'bias': plan.get('bias', 'long'),
                'strategy': plan.get('strategy', ''),
                'setup_type': decision_event.get('decision', {}).get('setup_type', ''),
                'regime': decision_event.get('decision', {}).get('regime', ''),
            })

        # Add actual trigger data if available
        if trigger_event and 'trigger' in trigger_event:
            trigger_data = trigger_event['trigger']
            analytics.update({
                'actual_entry_price': trigger_data.get('actual_price', 0),
                'slippage_bps': self._calculate_slippage_bps(
                    analytics.get('entry_reference', 0),
                    trigger_data.get('actual_price', 0)
                ),
                'order_id': trigger_data.get('order_id', ''),
            })

        # Add analytics fields
        analytics['analytics'] = {
            'time_decay_factor': 1.0,
            'execution_probability': 0.15
        }

        # Calculate and add fees
        entry_price = analytics.get('actual_entry_price') or analytics.get('entry_reference', 0)
        exit_price = analytics.get('exit_price', 0)
        qty = analytics.get('qty', 0)

        if entry_price > 0 and exit_price > 0 and qty > 0:
            # For now, assume each exit is a complete trade (is_entry_order=True)
            # TODO: Handle partial exits properly (exit_sequence tracking)
            fees = self.calculate_trade_fees(entry_price, exit_price, qty, is_entry_order=True)
            analytics['fees'] = fees
            analytics['net_pnl'] = round(pnl - fees['total_fees'], 2)

        return analytics

    def _calculate_slippage_bps(self, reference_price: float, actual_price: float) -> float:
        """Calculate slippage in basis points"""
        try:
            if reference_price > 0:
                return abs(actual_price - reference_price) / reference_price * 10000
            return 0.0
        except Exception:
            return 0.0

    def _convert_event_to_analytics(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert an events.jsonl event to analytics format"""
        if event.get('type') != 'EXIT':
            return None
            
        exit_data = event.get('exit', {})
        return {
            'symbol': event.get('symbol', ''),
            'trade_id': event.get('trade_id', ''),
            'timestamp': event.get('ts', ''),
            'stage': 'EXIT',
            'qty': exit_data.get('qty', 0),
            'exit_price': exit_data.get('price', 0),
            'reason': exit_data.get('reason', ''),
            'lifecycle_id': event.get('trade_id', ''),  # Use trade_id as lifecycle_id
        }
    
    def _update_session_stats_from_event(self, event: Dict[str, Any]):
        """Update session stats from event data"""
        if event.get('type') == 'EXIT':
            self.session_stats['completed_trades'] += 1
            # Note: PnL calculation would require entry price, which isn't in exit events
            # This will be a limitation of this approach