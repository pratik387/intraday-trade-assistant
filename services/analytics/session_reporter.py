"""
session_reporter.py
------------------
Session analytics reporting service.

Handles generation and export of trading session analytics, including:
- Session summary logging
- Trade export to CSV
- Performance metrics reporting
- Trading run management
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from config.logging_config import get_agent_logger
from services.analytics.trade_lifecycle import TradeLifecycleManager

logger = get_agent_logger()


class SessionReporter:
    """Service for generating and exporting session analytics"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir
    
    def generate_session_report(self, lifecycle_manager: TradeLifecycleManager, 
                              log_dir: Optional[Path] = None) -> bool:
        """
        Generate comprehensive session analytics report.
        
        Args:
            lifecycle_manager: TradeLifecycleManager instance with session data
            log_dir: Optional override for log directory
            
        Returns:
            bool: True if report generated successfully
        """
        try:
            # Use provided log_dir or instance log_dir
            report_dir = log_dir or self.log_dir
            
            # Generate analytics
            analytics = lifecycle_manager.get_session_analytics()
            
            # Log session summary
            self._log_session_summary(analytics, report_dir)
            
            # Export trades CSV
            self._export_trades_csv(lifecycle_manager, report_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate session analytics: {e}")
            return False
    
    def _log_session_summary(self, analytics: Dict[str, Any], log_dir: Optional[Path] = None):
        """Log session summary to console/logs"""
        
        logger.info("=== SESSION SUMMARY ===")
        
        session_summary = analytics.get('session_summary', {})
        logger.info(f"Total trades: {session_summary.get('total_trades', 0)}")
        logger.info(f"Total PnL: Rs.{session_summary.get('total_pnl', 0)}")
        logger.info(f"Win rate: {session_summary.get('win_rate', 0):.1%}")
        logger.info(f"Profit factor: {session_summary.get('profit_factor', 0)}")
        
        if log_dir:
            logger.info(f"Logs saved to: {log_dir}")
        
        # Additional performance breakdown
        perf = analytics.get('performance_breakdown', {})
        if perf:
            logger.info(f"Wins/Losses: {perf.get('wins', 0)}/{perf.get('losses', 0)}")
            logger.info(f"Avg win: Rs.{perf.get('avg_win', 0)} | Avg loss: Rs.{perf.get('avg_loss', 0)}")
        
        # Quality metrics
        quality = analytics.get('quality_metrics', {})
        if quality:
            logger.info(f"Avg quality score: {quality.get('avg_quality_score', 0)}")
            logger.info(f"Avg rank score: {quality.get('avg_rank_score', 0)}")
        
        # Timing analysis
        timing = analytics.get('timing_analysis', {})
        if timing:
            logger.info(f"Avg hold time: {timing.get('avg_hold_minutes', 0)} minutes")
    
    def _export_trades_csv(self, lifecycle_manager: TradeLifecycleManager, 
                          log_dir: Optional[Path] = None):
        """Export trades to CSV file"""
        
        csv_data = lifecycle_manager.export_trades_csv()
        
        if csv_data == "No completed trades to export":
            logger.info("No completed trades to export")
            return
        
        if not log_dir:
            logger.warning("No log directory specified for CSV export")
            return
        
        try:
            csv_file = log_dir / "trades_summary.csv"
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write(csv_data)
            logger.info(f"Trade summary exported to: {csv_file}")
            
        except Exception as e:
            logger.error(f"Failed to export trades CSV: {e}")
    
    def export_detailed_analytics(self, lifecycle_manager: TradeLifecycleManager, 
                                 log_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Export detailed analytics JSON for further analysis.
        
        Args:
            lifecycle_manager: TradeLifecycleManager instance
            log_dir: Directory to save the analytics file
            
        Returns:
            Path to exported file or None if failed
        """
        if not log_dir:
            logger.warning("No log directory specified for detailed analytics export")
            return None
        
        try:
            analytics = lifecycle_manager.get_session_analytics()
            analytics_file = log_dir / "session_analytics.json"

            with open(analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Detailed analytics exported to: {analytics_file}")
            return analytics_file
            
        except Exception as e:
            logger.error(f"Failed to export detailed analytics: {e}")
            return None


def generate_session_report(lifecycle_manager: TradeLifecycleManager,
                          log_dir: Optional[Path] = None) -> bool:
    """
    Convenience function for generating session analytics report.

    Args:
        lifecycle_manager: TradeLifecycleManager instance with session data
        log_dir: Optional directory for exports

    Returns:
        bool: True if report generated successfully
    """
    reporter = SessionReporter(log_dir)
    return reporter.generate_session_report(lifecycle_manager, log_dir)


# ========== TRADING RUN MANAGEMENT ==========

def start_new_trading_run(description: str = "") -> str:
    """
    Start a new trading run and return run_id.

    Args:
        description: Description for the trading run

    Returns:
        str: Run ID of the started run
    """
    try:
        from services.logging.run_manager import get_run_manager
        run_manager = get_run_manager()
        run_id = run_manager.start_new_run(description)
        logger.info(f"Started new trading run: {run_id} - {description}")
        return run_id
    except Exception as e:
        logger.error(f"Failed to start trading run: {e}")
        raise


def end_current_trading_run() -> Optional[Dict[str, Any]]:
    """
    End the current trading run and return manifest.

    Returns:
        Dict with run information or None if failed
    """
    try:
        from services.logging.run_manager import get_run_manager
        run_manager = get_run_manager()
        manifest = run_manager.end_current_run()

        if manifest:
            logger.info(f"Completed trading run: {manifest['run_id']}")
            logger.info(f"Total sessions: {len(manifest.get('sessions', []))}")
        else:
            logger.warning("No active run to end")

        return manifest
    except Exception as e:
        logger.error(f"Failed to end trading run: {e}")
        return None


def generate_combined_run_report() -> Optional[Path]:
    """
    Generate combined CSV report for the current trading run.

    Returns:
        Path to generated CSV or None if failed
    """
    try:
        from diagnostics.diagnostics_report_builder import build_combined_csv_from_current_run
        csv_path = build_combined_csv_from_current_run()
        logger.info(f"Combined report generated: {csv_path}")
        return Path(csv_path)
    except Exception as e:
        logger.error(f"Failed to generate combined report: {e}")
        return None