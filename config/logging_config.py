# logging_config.py — Singleton logger factory

import logging
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
_current_log_month = None
_session_id = None
dir_path = None

def _initialize_loggers():
    """Initialize all loggers (internal function)"""
    global _agent_logger, _trade_logger, _trading_logger, _session_id, dir_path

    if _agent_logger and _trade_logger and _trading_logger:
        return

    # Create timestamped session directory
    _session_id = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(__file__).resolve().parents[1] / "logs" / _session_id
    os.makedirs(log_dir, exist_ok=True)
    dir_path = log_dir

    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(name)s — %(message)s')

    # Agent Logger
    _agent_logger = logging.getLogger("agent")
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

def get_agent_logger():
    """Get the agent logger for general application logging"""
    global _agent_logger
    if _agent_logger is None:
        _initialize_loggers()
    return _agent_logger

def get_execution_loggers():
    """Get both agent and trade loggers for execution services"""
    global _agent_logger, _trade_logger
    if _agent_logger is None or _trade_logger is None:
        _initialize_loggers()
    return _agent_logger, _trade_logger

def get_trading_logger():
    """Get the enhanced trading logger for analytics"""
    global _trading_logger
    if _trading_logger is None:
        _initialize_loggers()
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
