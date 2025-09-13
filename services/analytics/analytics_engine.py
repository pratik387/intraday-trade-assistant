"""
analytics_engine.py
-------------------
Centralized analytics engine that orchestrates all end-of-session analytics generation.

Consolidates:
- Diagnostics CSV generation
- Trade lifecycle analytics
- Session reporting
"""

from pathlib import Path
from typing import Optional
from config.logging_config import get_agent_logger
from diagnostics.diagnostics_report_builder import build_csv_from_events
from services.analytics.trade_lifecycle import TradeLifecycleManager
from services.analytics.session_reporter import generate_session_report, SessionReporter

logger = get_agent_logger()


class AnalyticsEngine:
    """Centralized engine for generating all session analytics"""
    
    def __init__(self, lifecycle_manager: Optional[TradeLifecycleManager] = None):
        self.lifecycle_manager = lifecycle_manager or TradeLifecycleManager()
    
    def generate_end_of_session_analytics(self, include_trade_analytics: bool = True) -> bool:
        """
        Generate comprehensive end-of-session analytics.
        
        Args:
            include_trade_analytics: Whether to generate trade lifecycle analytics
            
        Returns:
            bool: True if all analytics generated successfully
        """
        success = True
        
        # Always generate diagnostics CSV
        success &= self._generate_diagnostics_csv()
        
        # Generate trade analytics if requested and available
        if include_trade_analytics and self.lifecycle_manager:
            success &= self._generate_trade_analytics()
        
        return success
    
    def _generate_diagnostics_csv(self) -> bool:
        """Generate diagnostics CSV from events"""
        try:
            csv_path = build_csv_from_events()
            logger.info("Diagnostics CSV written: %s", csv_path)
            return True
        except Exception as e:
            logger.warning("Failed to build diagnostics CSV: %s", e)
            return False
    
    def _generate_trade_analytics(self) -> bool:
        """Generate trade lifecycle analytics and reports"""
        try:
            from config.logging_config import get_log_directory
            log_dir = get_log_directory()
            
            # Generate session report
            success = generate_session_report(self.lifecycle_manager, log_dir)
            
            if success:
                logger.info("Trade analytics generated successfully")
                
                # Also generate detailed JSON export for further analysis
                reporter = SessionReporter(log_dir)
                analytics_file = reporter.export_detailed_analytics(self.lifecycle_manager, log_dir)
                
                if analytics_file:
                    logger.info("Detailed analytics JSON exported: %s", analytics_file)
                
                return True
            else:
                logger.warning("Failed to generate trade analytics")
                return False
                
        except Exception as e:
            logger.error(f"Error generating trade analytics: {e}")
            return False


def generate_end_of_session_analytics(lifecycle_manager: Optional[TradeLifecycleManager] = None,
                                    include_trade_analytics: bool = True) -> bool:
    """
    Convenience function for generating end-of-session analytics.
    
    Args:
        lifecycle_manager: Optional TradeLifecycleManager instance
        include_trade_analytics: Whether to generate trade lifecycle analytics
        
    Returns:
        bool: True if analytics generated successfully
    """
    engine = AnalyticsEngine(lifecycle_manager)
    return engine.generate_end_of_session_analytics(include_trade_analytics)