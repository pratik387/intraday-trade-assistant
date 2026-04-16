# logging_config.py — Singleton logger factory

import atexit
import logging
import json
import threading
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
_timing_logger = None
# Per-event detector decision loggers (for post-OCI gauntlet analysis).
# Capture every non-trivial detector accept / reject with full bar context
# so the 5-stage funnel can be reconstructed offline. See main_detector.py.
_detector_rejections_logger = None
_detector_accepts_logger = None
_current_log_month = None
_session_id = None
dir_path = None
_global_run_prefix = ""  # Global run prefix to be set before any logger initialization


class JSONLLogger:
    """Buffered JSONL logger with persistent file handle.

    Previous implementation opened + closed the file on every write, which
    cost ~224us per call (kernel syscalls: mkdir + open + write + close).
    At ~157k log lines per backtest day, that's ~35s of wasted wall time.

    This implementation keeps a line-buffered (`buffering=1`) file handle open
    for the process lifetime and writes are ~6us each. Benchmarked at 37x
    speedup on local Windows; wins are larger on slower disks (OCI pods).

    Fork safety: ProcessPoolExecutor workers inherit the parent's file handle
    via fork(). If a child process wrote to the inherited Python buffered
    file object, the parent's buffer would be corrupted. We detect fork by
    tracking the PID and transparently reopen the file in the child process.
    Writes use O_APPEND under the hood, so concurrent parent+child appends
    are atomic under PIPE_BUF (4096 bytes) — all our JSONL lines are <1KB.

    Crash safety: line-buffered mode flushes on each '\\n', so at worst the
    last partial line is lost. No data loss for completed runs.
    """

    def __init__(self, file_path: Path, stage_name: str):
        self.file_path = file_path
        self.stage = stage_name
        self._fh = None
        self._fh_pid = None
        self._lock = threading.Lock()
        atexit.register(self._close)

    def _ensure_open(self):
        """Open (or reopen after fork) the file handle. Must be called under self._lock."""
        current_pid = os.getpid()
        if self._fh is None or self._fh_pid != current_pid:
            # First open OR this is a forked child that inherited the parent's handle.
            # Don't close the inherited handle (that would flush/affect the parent's fd).
            # Just drop the reference and open a fresh one for this process.
            self._fh = None
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.file_path, 'a', encoding='utf-8', buffering=1)
            self._fh_pid = current_pid

    def log_accept(self, symbol: str, timestamp: str = None, **data):
        """Log an accept decision with additional data"""
        self._write_jsonl("accept", symbol, timestamp=timestamp, **data)

    def log_reject(self, symbol: str, reason: str, timestamp: str = None, **data):
        """Log a reject decision with reason and additional data"""
        self._write_jsonl("reject", symbol, reason=reason, timestamp=timestamp, **data)

    def log_event(self, **data):
        """Write an arbitrary structured event to this log file.

        Used for timing.jsonl and any other free-form structured logging that
        doesn't fit the accept/reject/symbol shape.
        """
        line = json.dumps(data) + '\n'
        # Fast-path fork check: reset lock if PID changed (inherited lock may be in
        # bad state if forked during a write). Workers are single-threaded per process
        # in ProcessPoolExecutor, so unlocked reset is safe.
        if self._fh_pid is not None and self._fh_pid != os.getpid():
            self._lock = threading.Lock()
            self._fh = None
        with self._lock:
            try:
                self._ensure_open()
                self._fh.write(line)
            except Exception:
                pass  # never let logging break the caller

    def _write_jsonl(self, action: str, symbol: str, timestamp: str = None, **data):
        """Write structured JSONL entry"""
        ts = timestamp if timestamp else datetime.now().isoformat()
        entry = {
            "timestamp": ts,
            "stage": self.stage,
            "action": action,
            "symbol": symbol,
            **data
        }
        line = json.dumps(entry) + '\n'
        if self._fh_pid is not None and self._fh_pid != os.getpid():
            self._lock = threading.Lock()
            self._fh = None
        with self._lock:
            try:
                self._ensure_open()
                self._fh.write(line)
            except Exception:
                pass

    def _close(self):
        """Flush and close the file handle. Registered via atexit in __init__."""
        try:
            if self._fh_pid != os.getpid():
                # Another process registered this atexit via fork inheritance;
                # don't touch — the owning process will handle its own close.
                return
            with self._lock:
                if self._fh is not None:
                    try:
                        self._fh.flush()
                        self._fh.close()
                    except Exception:
                        pass
                    self._fh = None
        except Exception:
            pass

def set_global_run_prefix(run_prefix: str):
    """Set the global run prefix before any logger initialization"""
    global _global_run_prefix
    _global_run_prefix = run_prefix

def _initialize_loggers(run_prefix: str = "", force_reinit: bool = False):
    """Initialize all loggers (internal function)"""
    global _agent_logger, _trade_logger, _trading_logger, _session_id, dir_path, _global_run_prefix
    global _scanner_logger, _screener_logger, _ranking_logger, _planning_logger, _events_decision_logger
    global _timing_logger, _detector_rejections_logger, _detector_accepts_logger

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

    # Configure diag_event_log (single writer for events.jsonl)
    from diagnostics.diag_event_log import diag_event_log
    diag_event_log.set_output(out_dir=str(log_dir), run_id=_session_id)

    # Stage JSONL Loggers
    _scanner_logger = JSONLLogger(log_dir / "scanning.jsonl", "scanner")
    _screener_logger = JSONLLogger(log_dir / "screening.jsonl", "screener")
    _ranking_logger = JSONLLogger(log_dir / "ranking.jsonl", "ranking")
    _planning_logger = JSONLLogger(log_dir / "planning.jsonl", "planning")
    _events_decision_logger = JSONLLogger(log_dir / "events_decisions.jsonl", "events_decision")

    # Timing logger — only used when TRADING_PERF_TIMER=1 in the environment.
    # Always created (cheap) so get_timing_logger() never returns None.
    _timing_logger = JSONLLogger(log_dir / "timing.jsonl", "timing")

    # Per-event detector decision loggers (for post-OCI gauntlet analysis).
    # Trivial rejections (insufficient data, no pattern matched) are filtered
    # out at the call site; only diagnostically useful events are logged.
    _detector_rejections_logger = JSONLLogger(log_dir / "detector_rejections.jsonl", "detector_reject")
    _detector_accepts_logger = JSONLLogger(log_dir / "detector_accepts.jsonl", "detector_accept")

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


def get_timing_logger():
    """Get the performance timing logger (writes to timing.jsonl).

    Used by utils.perf_timer.perf() when TRADING_PERF_TIMER=1. Returns a
    JSONLLogger or None (workers without a run_prefix); caller must guard.
    """
    global _timing_logger
    if _timing_logger is None:
        _initialize_loggers()
    return _timing_logger


def get_detector_rejections_logger():
    """Get the per-event detector rejection logger (detector_rejections.jsonl).

    Captures every non-trivial detector rejection with full bar context
    (symbol, timestamp, detector, reason, regime, cap_segment, hour, vol_z).
    For post-OCI gauntlet funnel reconstruction. Returns None if not
    initialized (e.g., test/import contexts without a run_prefix).
    """
    global _detector_rejections_logger
    if _detector_rejections_logger is None:
        _initialize_loggers()
    return _detector_rejections_logger


def get_detector_accepts_logger():
    """Get the per-event detector accept logger (detector_accepts.jsonl).

    Mirror of detector_rejections.jsonl for accepted detections. Lets
    post-hoc analysis directly compare reject vs accept distributions
    per (regime, cap_segment, hour) without joining tables.
    """
    global _detector_accepts_logger
    if _detector_accepts_logger is None:
        _initialize_loggers()
    return _detector_accepts_logger


# -------------------- Child Process Logger Initialization --------------------

def initialize_child_loggers(log_dir: Path, process_tag: str):
    """
    Initialize loggers for a child process using the parent's log directory.

    Called by the exec child process (spawned via multiprocessing) to write
    logs into the SAME session folder as the parent, avoiding log fragmentation.

    CRITICAL: On Linux, multiprocessing.Process uses fork(). The child inherits
    the parent's module state, including cached logger references like:
        logger, trade_logger = get_execution_loggers()  # in trigger_aware_executor.py
    These point to logging.getLogger("agent") and logging.getLogger("trade"),
    which still have the parent's FileHandlers. We MUST replace handlers on
    these SAME logger objects (not create new names like "agent_exec") so that
    all cached module-level references route to the child's log files.

    Creates:
      - agent_{tag}.log  (replaces parent's agent.log handler)
      - trade_logs.log   (replaces parent's trade handler)
      - TradingLogger    (exec child owns analytics)
      - events_decisions.jsonl (exec child owns decision logging)

    Args:
        log_dir: Parent's log directory (Path object)
        process_tag: Tag for this process (e.g., "exec") — used in agent log filename
    """
    global _agent_logger, _trade_logger, _trading_logger, _session_id, dir_path
    global _events_decision_logger, _timing_logger
    global _detector_rejections_logger, _detector_accepts_logger

    dir_path = log_dir
    _session_id = log_dir.name  # Reuse parent's session ID from directory name

    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(name)s — %(message)s')

    # Agent logger — replace inherited parent handlers on the SAME "agent" logger
    # so all module-level cached references (logger = get_agent_logger()) route here
    _agent_logger = logging.getLogger("agent")
    _agent_logger.handlers.clear()
    _agent_logger.setLevel(logging.INFO)
    agent_file = logging.FileHandler(log_dir / f"agent_{process_tag}.log", encoding="utf-8")
    agent_file.setFormatter(formatter)
    _agent_logger.addHandler(agent_file)

    # Trade logger — replace inherited parent handlers on the SAME "trade" logger
    _trade_logger = logging.getLogger("trade")
    _trade_logger.handlers.clear()
    _trade_logger.setLevel(logging.INFO)
    trade_file = logging.FileHandler(log_dir / "trade_logs.log", encoding="utf-8")
    trade_file.setFormatter(formatter)
    _trade_logger.addHandler(trade_file)

    # Enhanced Trading Logger — exec child owns analytics
    _trading_logger = TradingLogger(_session_id, log_dir)

    # Configure diag_event_log for exec child process
    from diagnostics.diag_event_log import diag_event_log
    diag_event_log.set_output(out_dir=str(log_dir), run_id=_session_id)

    # Events decision JSONL — exec child owns decision logging
    _events_decision_logger = JSONLLogger(log_dir / "events_decisions.jsonl", "events_decision")

    # Timing logger — exec child writes to same timing.jsonl as parent
    # (each process has its own file handle; O_APPEND atomicity handles interleaving)
    _timing_logger = JSONLLogger(log_dir / "timing.jsonl", "timing")

    # Per-event detector decision loggers (exec child writes to same files
    # as parent; O_APPEND atomicity handles interleaving across processes)
    _detector_rejections_logger = JSONLLogger(log_dir / "detector_rejections.jsonl", "detector_reject")
    _detector_accepts_logger = JSONLLogger(log_dir / "detector_accepts.jsonl", "detector_accept")
