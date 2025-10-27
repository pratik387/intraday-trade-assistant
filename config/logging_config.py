# logging_config.py — Singleton logger factory

import logging
import json
from pathlib import Path
from datetime import datetime
import os
import pandas as pd
from pytz import timezone
from services.logging.trading_logger import TradingLogger

india_tz = timezone("Asia/Kolkata")

_agent_logger = None
_trade_logger = None
_trading_logger = None
# Stage JSONL loggers
_scanner_logger = None
_screener_logger = None
_ranking_logger = None
_planning_logger = None
_events_decision_logger = None
_current_log_month = None
_session_id = None
dir_path = None
_global_run_prefix = ""  # Global run prefix to be set before any logger initialization


class JSONLLogger:
    """Helper class for structured JSONL logging at each pipeline stage"""

    def __init__(self, file_path: Path, stage_name: str):
        self.file_path = file_path
        self.stage = stage_name

    def log_accept(self, symbol: str, timestamp: str = None, **data):
        """Log an accept decision with additional data"""
        self._write_jsonl("accept", symbol, timestamp=timestamp, **data)

    def log_reject(self, symbol: str, reason: str, timestamp: str = None, **data):
        """Log a reject decision with reason and additional data"""
        self._write_jsonl("reject", symbol, reason=reason, timestamp=timestamp, **data)

    def _write_jsonl(self, action: str, symbol: str, timestamp: str = None, **data):
        """Write structured JSONL entry"""
        # Ensure parent directory exists before writing
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Use provided timestamp or fall back to current time
        ts = timestamp if timestamp else datetime.now().isoformat()
        entry = {
            "timestamp": ts,
            "stage": self.stage,
            "action": action,
            "symbol": symbol,
            **data
        }
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

def set_global_run_prefix(run_prefix: str):
    """Set the global run prefix before any logger initialization"""
    global _global_run_prefix
    _global_run_prefix = run_prefix

def _initialize_loggers(run_prefix: str = "", force_reinit: bool = False):
    """Initialize all loggers (internal function)"""
    global _agent_logger, _trade_logger, _trading_logger, _session_id, dir_path, _global_run_prefix
    global _scanner_logger, _screener_logger, _ranking_logger, _planning_logger, _events_decision_logger

    # Quick check - if ANY logger is initialized, reuse the existing session
    # UNLESS force_reinit is True (allows re-initialization with different run_prefix)
    if _agent_logger is not None and not force_reinit:
        return

    # Use provided run_prefix, or fall back to global run prefix
    effective_prefix = run_prefix or _global_run_prefix

    # Check if we're in a worker process (ProcessPoolExecutor)
    import multiprocessing as mp
    is_worker_process = mp.current_process().name != 'MainProcess'

    # IMPORTANT: Only create log directories for actual engine runs (with run_prefix)
    # Worker processes get console-only loggers without creating folders
    if not effective_prefix:
        if is_worker_process:
            # Worker process: Create console-only logger (no files)
            _agent_logger = logging.getLogger("agent")
            if not _agent_logger.hasHandlers():
                _agent_logger.setLevel(logging.WARNING)  # Less verbose for workers
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter('%(levelname)s — %(name)s — %(message)s'))
                _agent_logger.addHandler(console_handler)

            # Set minimal globals for worker
            _session_id = "worker_process"
            dir_path = Path(__file__).resolve().parents[1] / "logs"
            return
        else:
            # Main process test/import: Skip initialization completely
            return

    # Create timestamped session directory with run prefix (actual engine runs)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    _session_id = f"{effective_prefix}{timestamp}"
    log_dir = Path(__file__).resolve().parents[1] / "logs" / _session_id
    os.makedirs(log_dir, exist_ok=True)
    dir_path = log_dir

    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(name)s — %(message)s')

    # Agent Logger
    _agent_logger = logging.getLogger("agent")

    # Clear existing handlers if force_reinit (allows switching from console to file logging)
    if force_reinit and _agent_logger.hasHandlers():
        _agent_logger.handlers.clear()

    if not _agent_logger.hasHandlers():
        _agent_logger.setLevel(logging.INFO)
        agent_file = logging.FileHandler(log_dir / "agent.log",  encoding="utf-8")
        agent_file.setFormatter(formatter)
        _agent_logger.addHandler(agent_file)

    # Trade Logger
    _trade_logger = logging.getLogger("trade")
    if not _trade_logger.hasHandlers():
        _trade_logger.setLevel(logging.INFO)
        trade_file = logging.FileHandler(log_dir / "trade_logs.log",  encoding="utf-8")
        trade_file.setFormatter(formatter)
        _trade_logger.addHandler(trade_file)

    # Enhanced Trading Logger
    _trading_logger = TradingLogger(_session_id, log_dir)

    # Stage JSONL Loggers
    _scanner_logger = JSONLLogger(log_dir / "scanning.jsonl", "scanner")
    _screener_logger = JSONLLogger(log_dir / "screening.jsonl", "screener")
    _ranking_logger = JSONLLogger(log_dir / "ranking.jsonl", "ranking")
    _planning_logger = JSONLLogger(log_dir / "planning.jsonl", "planning")
    _events_decision_logger = JSONLLogger(log_dir / "events_decisions.jsonl", "events_decision")

def get_agent_logger(run_prefix: str = "", force_reinit: bool = False):
    """Get the agent logger for general application logging"""
    global _agent_logger
    if _agent_logger is None or force_reinit:
        _initialize_loggers(run_prefix, force_reinit=force_reinit)

    # If _initialize_loggers didn't set a logger (no run_prefix, main process),
    # create a fallback console-only logger (for Lambda, imports, etc.)
    if _agent_logger is None:
        _agent_logger = logging.getLogger("agent")
        if not _agent_logger.hasHandlers():
            _agent_logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            _agent_logger.addHandler(console_handler)

    return _agent_logger

def get_execution_loggers():
    """Get both agent and trade loggers for execution services"""
    global _agent_logger, _trade_logger
    if _agent_logger is None or _trade_logger is None:
        _initialize_loggers()
    return _agent_logger, _trade_logger

def get_trading_logger(run_prefix: str = ""):
    """Get the enhanced trading logger for analytics"""
    global _trading_logger
    if _trading_logger is None:
        _initialize_loggers(run_prefix)
    return _trading_logger

def switch_agent_log_file(month_str: str):
    """Swap agent logger file handler based on backtest month (e.g., '2025-06')"""
    global _agent_logger, _current_log_month
    if not _agent_logger:
        _initialize_loggers()

    if month_str == _current_log_month:
        return

    log_file = dir_path / f"agent.{month_str}.log"

    for h in _agent_logger.handlers[:]:
        _agent_logger.removeHandler(h)

    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(name)s — %(message)s')
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    _agent_logger.addHandler(file_handler)
    _current_log_month = month_str

def get_log_directory():
    global dir_path
    if dir_path is None:
        # Initialize logging if not done already
        _initialize_loggers()
    return dir_path

def get_session_id():
    """Get the current session ID"""
    global _session_id
    if _session_id is None:
        # Initialize logging if not done already
        _initialize_loggers()
    return _session_id


# Stage JSONL Logger Getters

def get_scanner_logger():
    """Get scanner stage logger for energy scanning and shortlisting decisions"""
    global _scanner_logger
    if _scanner_logger is None:
        _initialize_loggers()
    return _scanner_logger


def get_screener_logger():
    """Get screener stage logger for gates and structure detection decisions"""
    global _screener_logger
    if _screener_logger is None:
        _initialize_loggers()
    return _screener_logger


def get_ranking_logger():
    """Get ranking stage logger for scoring and percentile filtering decisions"""
    global _ranking_logger
    if _ranking_logger is None:
        _initialize_loggers()
    return _ranking_logger


def get_planning_logger():
    """Get planning stage logger for trade plan creation and validation decisions"""
    global _planning_logger
    if _planning_logger is None:
        _initialize_loggers()
    return _planning_logger


def get_events_decision_logger():
    """Get events decision logger for final trading decisions"""
    global _events_decision_logger
    if _events_decision_logger is None:
        _initialize_loggers()
    return _events_decision_logger
