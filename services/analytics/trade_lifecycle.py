"""
trade_lifecycle.py
-----------------
Trade lifecycle management and performance attribution analysis.

Tracks trades from decision → trigger → exit with detailed analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import statistics
import json


@dataclass
class TradeStage:
    """Individual stage in a trade's lifecycle"""
    stage: str  # DECISION, TRIGGER, EXIT
    timestamp: datetime
    data: Dict[str, Any]
    elapsed_seconds: int = 0


@dataclass
class CompleteTrade:
    """Complete trade lifecycle with performance metrics"""
    lifecycle_id: str
    symbol: str
    trade_id: str
    stages: List[TradeStage] = field(default_factory=list)
    
    # Performance metrics
    entry_price: float = 0.0
    exit_price: float = 0.0
    qty: int = 0
    pnl: float = 0.0
    hold_duration_minutes: int = 0
    setup_type: str = ""
    regime: str = ""
    
    # Quality metrics
    rank_score: float = 0.0
    structural_rr: float = 0.0
    setup_quality_score: float = 0.0
    
    # Attribution
    exit_reason: str = ""
    win_loss: str = ""  # WIN, LOSS, BREAKEVEN
    
    def is_complete(self) -> bool:
        """Check if trade has all required stages"""
        stages = {stage.stage for stage in self.stages}
        return 'DECISION' in stages and 'EXIT' in stages
    
    def add_stage(self, stage: TradeStage):
        """Add a stage to the trade lifecycle"""
        self.stages.append(stage)
        self._update_metrics()
    
    def _update_metrics(self):
        """Update derived metrics when stages are added"""
        if not self.stages:
            return
        
        # Find key stages
        decision_stage = next((s for s in self.stages if s.stage == 'DECISION'), None)
        trigger_stage = next((s for s in self.stages if s.stage == 'TRIGGER'), None)
        exit_stage = next((s for s in self.stages if s.stage == 'EXIT'), None)
        
        # Extract metrics from decision stage
        if decision_stage:
            data = decision_stage.data
            self.setup_type = data.get('decision', {}).get('setup_type', '')
            self.regime = data.get('decision', {}).get('regime', '')
            
            if 'features' in data:
                self.rank_score = data['features'].get('rank_score', 0.0)
            
            if 'plan' in data and 'quality' in data['plan']:
                self.structural_rr = data['plan']['quality'].get('structural_rr', 0.0)
            
            if 'analytics' in data:
                self.setup_quality_score = data['analytics'].get('setup_quality_score', 0.0)
        
        # Extract execution metrics from trigger stage
        if trigger_stage:
            data = trigger_stage.data
            self.entry_price = data.get('price', 0.0)
            self.qty = data.get('qty', 0)
        
        # Extract exit metrics from exit stage
        if exit_stage:
            data = exit_stage.data
            self.exit_price = data.get('exit_price', 0.0)
            self.pnl = data.get('pnl', 0.0)
            self.exit_reason = data.get('reason', '')
            
            # Calculate win/loss
            if self.pnl > 0:
                self.win_loss = 'WIN'
            elif self.pnl < 0:
                self.win_loss = 'LOSS'
            else:
                self.win_loss = 'BREAKEVEN'
        
        # Calculate hold duration
        if trigger_stage and exit_stage:
            duration = exit_stage.timestamp - trigger_stage.timestamp
            self.hold_duration_minutes = int(duration.total_seconds() / 60)


class TradeLifecycleManager:
    """Manages complete trade lifecycles and analytics"""
    
    def __init__(self):
        self.active_trades: Dict[str, CompleteTrade] = {}
        self.completed_trades: List[CompleteTrade] = []
    
    def start_trade(self, lifecycle_id: str, symbol: str, trade_id: str, decision_data: Dict[str, Any]):
        """Start tracking a new trade lifecycle"""
        trade = CompleteTrade(
            lifecycle_id=lifecycle_id,
            symbol=symbol,
            trade_id=trade_id
        )
        
        # Add decision stage
        decision_stage = TradeStage(
            stage='DECISION',
            timestamp=datetime.now(),
            data=decision_data
        )
        trade.add_stage(decision_stage)
        
        self.active_trades[lifecycle_id] = trade
        return trade
    
    def add_trigger(self, lifecycle_id: str, trigger_data: Dict[str, Any]):
        """Add trigger stage to existing trade"""
        if lifecycle_id in self.active_trades:
            trade = self.active_trades[lifecycle_id]
            
            trigger_stage = TradeStage(
                stage='TRIGGER',
                timestamp=datetime.now(),
                data=trigger_data
            )
            trade.add_stage(trigger_stage)
    
    def add_exit(self, lifecycle_id: str, exit_data: Dict[str, Any]):
        """Add exit stage and complete the trade"""
        if lifecycle_id in self.active_trades:
            trade = self.active_trades[lifecycle_id]
            
            exit_stage = TradeStage(
                stage='EXIT',
                timestamp=datetime.now(),
                data=exit_data
            )
            trade.add_stage(exit_stage)
            
            # Move to completed trades
            if trade.is_complete():
                self.completed_trades.append(trade)
                del self.active_trades[lifecycle_id]
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive session analytics"""
        all_trades = list(self.active_trades.values()) + self.completed_trades
        completed = self.completed_trades
        
        if not completed:
            return {'message': 'No completed trades in session'}
        
        # Basic metrics
        total_pnl = sum(trade.pnl for trade in completed)
        wins = [trade for trade in completed if trade.win_loss == 'WIN']
        losses = [trade for trade in completed if trade.win_loss == 'LOSS']
        
        # Performance metrics
        win_rate = len(wins) / len(completed) if completed else 0
        avg_win = statistics.mean([trade.pnl for trade in wins]) if wins else 0
        avg_loss = statistics.mean([trade.pnl for trade in losses]) if losses else 0
        profit_factor = abs(sum(trade.pnl for trade in wins) / sum(trade.pnl for trade in losses)) if losses else float('inf')
        
        # Setup analysis
        setup_performance = {}
        for trade in completed:
            setup = trade.setup_type
            if setup not in setup_performance:
                setup_performance[setup] = {'count': 0, 'pnl': 0, 'wins': 0}
            
            setup_performance[setup]['count'] += 1
            setup_performance[setup]['pnl'] += trade.pnl
            if trade.win_loss == 'WIN':
                setup_performance[setup]['wins'] += 1
        
        # Calculate win rates by setup
        for setup, stats in setup_performance.items():
            stats['win_rate'] = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
        
        # Quality score analysis
        quality_scores = [trade.setup_quality_score for trade in completed if trade.setup_quality_score > 0]
        rank_scores = [trade.rank_score for trade in completed if trade.rank_score > 0]
        
        return {
            'session_summary': {
                'total_trades': len(completed),
                'active_trades': len(self.active_trades),
                'total_pnl': round(total_pnl, 2),
                'win_rate': round(win_rate, 3),
                'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf'
            },
            'performance_breakdown': {
                'wins': len(wins),
                'losses': len(losses),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'largest_win': max([trade.pnl for trade in wins]) if wins else 0,
                'largest_loss': min([trade.pnl for trade in losses]) if losses else 0
            },
            'setup_analysis': setup_performance,
            'quality_metrics': {
                'avg_quality_score': round(statistics.mean(quality_scores), 2) if quality_scores else 0,
                'avg_rank_score': round(statistics.mean(rank_scores), 2) if rank_scores else 0,
                'quality_score_range': [min(quality_scores), max(quality_scores)] if quality_scores else [0, 0]
            },
            'timing_analysis': {
                'avg_hold_minutes': round(statistics.mean([trade.hold_duration_minutes for trade in completed]), 1) if completed else 0,
                'hold_duration_range': [
                    min([trade.hold_duration_minutes for trade in completed]),
                    max([trade.hold_duration_minutes for trade in completed])
                ] if completed else [0, 0]
            }
        }
    
    def export_trades_csv(self) -> str:
        """Export completed trades to CSV format"""
        if not self.completed_trades:
            return "No completed trades to export"
        
        headers = [
            'lifecycle_id', 'symbol', 'setup_type', 'regime', 'entry_price',
            'exit_price', 'qty', 'pnl', 'hold_minutes', 'rank_score',
            'structural_rr', 'quality_score', 'exit_reason', 'win_loss'
        ]
        
        csv_lines = [','.join(headers)]
        
        for trade in self.completed_trades:
            row = [
                trade.lifecycle_id,
                trade.symbol,
                trade.setup_type,
                trade.regime,
                str(trade.entry_price),
                str(trade.exit_price),
                str(trade.qty),
                str(trade.pnl),
                str(trade.hold_duration_minutes),
                str(trade.rank_score),
                str(trade.structural_rr),
                str(trade.setup_quality_score),
                trade.exit_reason,
                trade.win_loss
            ]
            csv_lines.append(','.join(row))
        
        return '\n'.join(csv_lines)