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
            'breakevens': 0,
            'rank_scores_triggered': [],
            'rank_scores_skipped': []
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
        
        # Update session stats
        self.session_stats['total_decisions'] += 1
        
        # Track rank scores for analysis
        if 'features' in enhanced_data and 'rank_score' in enhanced_data['features']:
            rank_score = enhanced_data['features']['rank_score']
            if self._should_trigger(enhanced_data):
                self.session_stats['rank_scores_triggered'].append(rank_score)
            else:
                self.session_stats['rank_scores_skipped'].append(rank_score)
        
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
        
        # Update performance summary
        self._update_performance_summary()
    
    def log_trigger(self, trade_data: Dict[str, Any]):
        """Log a trade trigger execution"""
        
        # Find matching lifecycle
        lifecycle_id = self._find_lifecycle_id(trade_data)
        
        if lifecycle_id:
            # Update lifecycle
            lifecycle = self.trade_lifecycles[lifecycle_id]
            lifecycle.stage = 'TRIGGER'
            lifecycle.elapsed_from_decision = self._calculate_elapsed(lifecycle)
            
            # Update session stats
            self.session_stats['triggered_trades'] += 1
            
            # Log to analytics stream (triggered trades only)
            analytics_data = self._add_analytics_fields(trade_data)
            analytics_data['lifecycle_id'] = lifecycle_id
            analytics_data['stage'] = 'TRIGGER'
            self.analytics_logger.info(json.dumps(analytics_data))
        
        # Log to trade logs (existing format)
        symbol = trade_data.get('symbol', '')
        qty = trade_data.get('qty', 0)
        price = trade_data.get('price', 0)
        strategy = trade_data.get('strategy', '')
        order_id = trade_data.get('order_id', '')
        
        self.trade_logger.info(
            f"TRIGGER_EXEC | {symbol} | BUY {qty} @ {price} | strategy={strategy} | order_id={order_id}"
        )
        
        self._update_performance_summary()
    
    def log_exit(self, trade_data: Dict[str, Any]):
        """Log a trade exit"""
        
        # Find matching lifecycle (create one if none exists)
        lifecycle_id = self._find_or_create_lifecycle_id(trade_data)
        
        if lifecycle_id:
            # Update lifecycle
            lifecycle = self.trade_lifecycles[lifecycle_id]
            lifecycle.stage = 'EXIT'
            lifecycle.elapsed_from_decision = self._calculate_elapsed(lifecycle)
            
            # Update session stats with PnL
            pnl = trade_data.get('pnl', 0.0)
            self.session_stats['total_pnl'] += pnl
            self.session_stats['completed_trades'] += 1
            
            if pnl > 0:
                self.session_stats['wins'] += 1
            elif pnl < 0:
                self.session_stats['losses'] += 1
            else:
                self.session_stats['breakevens'] += 1
            
            # Log to analytics stream (exit data)
            analytics_data = self._add_analytics_fields(trade_data)
            analytics_data['lifecycle_id'] = lifecycle_id
            analytics_data['stage'] = 'EXIT'
            analytics_data['pnl'] = pnl
            analytics_data['elapsed_from_decision'] = lifecycle.elapsed_from_decision
            self.analytics_logger.info(json.dumps(analytics_data))
        
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
        
        self._update_performance_summary()
    
    def _add_analytics_fields(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add derived analytics fields to trade data"""
        enhanced = trade_data.copy()
        
        # Add analytics section
        analytics = {}
        
        # Setup quality score (derived from multiple factors)
        if 'features' in enhanced and 'plan' in enhanced:
            rank_score = enhanced['features'].get('rank_score', 0)
            structural_rr = enhanced['plan'].get('quality', {}).get('structural_rr', 0)
            acceptance_ok = enhanced['plan'].get('quality', {}).get('acceptance_ok', False)
            
            # Simple quality score calculation
            analytics['setup_quality_score'] = (
                rank_score * 3 +
                structural_rr * 2 +
                (1.0 if acceptance_ok else 0.0)
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
        
        avg_rank_triggered = (
            sum(self.session_stats['rank_scores_triggered']) / len(self.session_stats['rank_scores_triggered'])
            if self.session_stats['rank_scores_triggered'] else 0
        )
        
        avg_rank_skipped = (
            sum(self.session_stats['rank_scores_skipped']) / len(self.session_stats['rank_scores_skipped'])
            if self.session_stats['rank_scores_skipped'] else 0
        )
        
        summary = {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                **self.session_stats,
                'execution_rate': round(execution_rate, 3),
                'win_rate': round(win_rate, 3),
                'avg_rank_triggered': round(avg_rank_triggered, 2),
                'avg_rank_skipped': round(avg_rank_skipped, 2)
            }
        }
        
        # Write to performance file
        performance_file = self.log_dir / 'performance.json'
        with open(performance_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def generate_csv_report(self):
        """Generate CSV report from events.jsonl"""
        try:
            from diagnostics.diagnostics_report_builder import build_csv_from_events
            csv_path = build_csv_from_events(log_dir=self.log_dir)
            return csv_path
        except Exception as e:
            print(f"Warning: Could not generate CSV report: {e}")
            return None
    
    def populate_analytics_from_events(self):
        """Populate analytics.jsonl from events.jsonl data"""
        try:
            events_file = self.log_dir / 'events.jsonl'
            if not events_file.exists():
                return
                
            with open(events_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get('type') == 'EXIT':
                            # Convert event data to analytics format
                            analytics_data = self._convert_event_to_analytics(event)
                            if analytics_data:
                                self.analytics_logger.info(json.dumps(analytics_data))
                                # Update session stats
                                self._update_session_stats_from_event(event)
                    except Exception as e:
                        continue
            
            # Update performance summary after processing all events
            self._update_performance_summary()
            
        except Exception as e:
            print(f"Warning: Could not populate analytics from events: {e}")
    
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