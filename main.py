# main.py
from __future__ import annotations
"""
Single entrypoint that wires:
  - ScreenerLive (producer)
  - TriggerAwareExecutor (trigger-based entries)
  - ExitExecutor (exits, LTP-only)
  - Shared PositionStore via TriggerAwareExecutor
  - Broker (live or dry-run)

Dry-run:
  Uses MockBroker + Feather replay for ticks; orders are logged (not sent).
Paper Trading:
  Uses KiteClient + KiteBroker with dry_run=True; live data, simulated orders.
Live:
  Uses KiteClient/KiteBroker.
"""
import sys
import time
import signal
import argparse
import threading
from typing import Optional
from threading import Lock
import pandas as pd

from config.logging_config import get_agent_logger, get_trading_logger
from config.filters_setup import load_filters

from services.orders.order_queue import OrderQueue
from services.screener_live import ScreenerLive
from services.execution.trade_executor import RiskState, Position
from services.execution.exit_executor import ExitExecutor
from services.ingest.tick_router import register_tick_listener
from services.execution.trigger_aware_executor import TriggerAwareExecutor, TradeState
from services.capital_manager import CapitalManager

# Dry-run adapter (patched MockBroker with LTP cache + Feather replay)
from broker.mock.mock_broker import MockBroker

# Live adapters (imported conditionally to avoid kiteconnect cryptography dependency in Lambda)
KiteClient = None
KiteBroker = None
try:
    from broker.kite.kite_client import KiteClient      # WebSocket client
    from broker.kite.kite_broker import KiteBroker      # REST orders/LTP
except ImportError as e:
    # Expected in Lambda where kiteconnect has binary dependencies
    # Dry-run mode uses MockBroker anyway
    pass

logger = get_agent_logger()
trading_logger = get_trading_logger()


# ------------------------ Shared Position Store ------------------------

class _PositionStore:
    """
    Thread-safe in-memory store shared by TriggerAwareExecutor (writer) and ExitExecutor (reader/updater).
    """
    def __init__(self) -> None:
        self._by_sym: dict[str, Position] = {}
        self._lock = threading.RLock()

    def upsert(self, p: Position) -> None:
        with self._lock:
            self._by_sym[p.symbol] = p

    def get(self, sym: str) -> Optional[Position]:
        with self._lock:
            return self._by_sym.get(sym)

    def all(self) -> list[Position]:
        with self._lock:
            return list(self._by_sym.values())

    # --- required by ExitExecutor ---
    def list_open(self) -> dict[str, Position]:
        with self._lock:
            return dict(self._by_sym)

    def close(self, sym: str) -> None:
        with self._lock:
            self._by_sym.pop(sym, None)

    def reduce(self, sym: str, qty_exit: int) -> None:
        """Reduce qty for partial exits; remove if goes to zero."""
        with self._lock:
            p = self._by_sym.get(sym)
            if not p:
                return
            new_qty = int(p.qty) - int(qty_exit)
            if new_qty <= 0:
                self._by_sym.pop(sym, None)
            else:
                p.qty = new_qty
                self._by_sym[sym] = p


# ------------------------ LTP cache (tick clock) ------------------------

class LTPCache:
    def __init__(self):
        self._d = {}
        self._lock = Lock()

    def update(self, symbol: str, ltp: float, ts: pd.Timestamp):
        with self._lock:
            self._d[symbol] = (float(ltp), pd.Timestamp(ts))

    def get_ltp(self, symbol: str):
        with self._lock:
            tup = self._d.get(symbol)
            return float(tup[0]) if tup else None

    def get_ltp_ts(self, symbol: str):
        with self._lock:
            tup = self._d.get(symbol)
            return tup if tup else (None, None)

ltp_cache = LTPCache()


# ------------------------ Dry-run broker wrapper ------------------------

class _DryRunBroker:
    """
    Wraps MockBroker to log orders instead of placing real ones in dry-run.
    Delegates get_ltp/get_ltp_batch to the underlying MockBroker (which is fed by replay).
    """
    def __init__(self, real: MockBroker):
        self._real = real

    def place_order(self, **kwargs):
        logger.info(f"[DRY_RUN] place_order skipped: {kwargs}")
        return "dryrun-order-id"

    def get_ltp(self, symbol: str, **kwargs) -> float:
        return self._real.get_ltp(symbol, **kwargs)

    def get_ltp_batch(self, symbols):
        return self._real.get_ltp_batch(symbols)


# ------------------------ Main orchestration ------------------------

def main() -> int:
    global args, logger, trading_logger  # Access the module-level variables

    # Re-initialize loggers with run_prefix to create log files
    import config.logging_config as logging_config
    from config.logging_config import set_global_run_prefix, get_agent_logger, get_trading_logger

    # Auto-generate run_prefix if not provided
    run_prefix = args.run_prefix
    if not run_prefix:
        if args.paper_trading:
            run_prefix = "paper_"
        elif not args.dry_run:
            run_prefix = "live_"
        # For dry-run (backtests), empty prefix is fine (logs optional)


    set_global_run_prefix(run_prefix)
    logger = get_agent_logger(run_prefix, force_reinit=True)
    trading_logger = get_trading_logger()

    logger.info(f"Session started | Mode: {'Paper' if args.paper_trading else 'Dry-run' if args.dry_run else 'Live'}")

    cfg = load_filters()  # validate early; raises if required keys are missing

    # Apply structure caching if requested (monkey-patches TradeDecisionGate.evaluate)
    # Also set flag in config so worker processes can enable caching
    if args.enable_cache:
        import tools.cached_engine_structures  # noqa: F401
        logger.info("[CACHE] Structure detection caching enabled")
        cfg["_enable_structure_cache"] = True  # Pass flag to worker processes

    # Initialize CapitalManager based on config and mode
    cap_mgmt_cfg = cfg.get('capital_management', {})

    # Auto-enable capital management for paper trading and live trading
    # For backtests, only enable if --with-capital-limits flag is set
    capital_enabled = cap_mgmt_cfg.get('enabled', False)
    if args.paper_trading or (not args.dry_run and not args.paper_trading):
        # Paper trading or live trading: always enable
        capital_enabled = True
        mis_enabled = True
        logger.info("[CAPITAL] Auto-enabled capital management (paper/live mode)")
    elif args.dry_run and args.with_capital_limits:
        # Backtest with capital limits: enable if flag set
        capital_enabled = True
        mis_enabled = True
        logger.info("[CAPITAL] Enabled capital management for realistic backtest (--with-capital-limits)")
    else:
        # Backtest without flag: keep disabled for fast testing
        capital_enabled = False
        mis_enabled = False
        logger.info("[CAPITAL] Disabled capital management for fast backtest")

    capital_manager = CapitalManager(
        enabled=capital_enabled,
        initial_capital=cap_mgmt_cfg.get('initial_capital', 100000),
        mis_enabled=mis_enabled,
        mis_config_path=cap_mgmt_cfg.get('mis_leverage', {}).get('config_file'),
        max_positions=cap_mgmt_cfg.get('max_concurrent_positions', 25)
    )

    # Order queue handles pacing/retries; it reads its own config internally
    oq = OrderQueue()

    # Pick mode
    if args.paper_trading:
        # Paper trading: live data + simulated orders
        from config.env_setup import env
        env.validate_for_paper_trading()  # Validate credentials before starting

        sdk = KiteClient()
        broker = KiteBroker(dry_run=True)
        logger.warning("🧪 PAPER TRADING MODE: Live data, simulated orders (no real trades)")
    elif args.dry_run:
        # Backtesting: historical data + mock broker
        if not args.session_date:
            print("error: --dry-run requires --session-date YYYY-MM-DD", file=sys.stderr)
            sys.exit(2)

        from_date = _hhmm_on(args.session_date, args.from_hhmm)
        to_date   = _hhmm_on(args.session_date, args.to_hhmm)

        # MockBroker replays 1m ticks via FeatherTicker and maintains an internal LTP cache
        sdk = MockBroker(path_json="nse_all.json", from_date=from_date, to_date=to_date)
        sdk.set_session_date(args.session_date)  # prev-day refs (PDH/PDL/PDC) resolve correctly
        broker = _DryRunBroker(sdk)
        logger.warning("📊 BACKTEST MODE: %s %s-%s (historical data replay)", args.session_date, args.from_hhmm, args.to_hhmm)
    else:
        # Live trading: real money
        sdk = KiteClient()
        broker = KiteBroker(dry_run=False)
        logger.warning("💰 LIVE TRADING MODE: Real orders will be placed with real money!")

    # Screener consumes the SDK (WS/ticker) + enqueues entry intents
    screener = ScreenerLive(sdk=sdk, order_queue=oq)

    # Tap the central tick router so entries & exits share the same tick clock
    def _ltp_tap(sym: str, price: float, qty: float, ts_dt):
        ts_pd = pd.Timestamp(ts_dt)  # ensure pandas timestamp
        ltp_cache.update(sym, price, ts_pd)

    register_tick_listener(_ltp_tap)

    # Risk + shared positions
    risk = RiskState(max_concurrent=int(cfg["max_concurrent_positions"]))
    positions = _PositionStore()
    
    trader = TriggerAwareExecutor(
        broker=broker,
        order_queue=oq,
        risk_state=risk,
        positions=positions,
        get_ltp_ts=ltp_cache.get_ltp_ts,
        bar_builder=screener.agg,  # Pass the BarBuilder instance
        trading_logger=trading_logger,  # Enhanced logging
        capital_manager=capital_manager  # Capital & MIS management
    )

    # ExitExecutor is LTP-only; wire it directly to broker.get_ltp and shared store
    exit_exec = ExitExecutor(
        broker=broker,
        positions=positions,
        get_ltp_ts=ltp_cache.get_ltp_ts,   # <- EOD uses tick timestamps, not wall clock
        trading_logger=trading_logger,  # Enhanced logging
        capital_manager=capital_manager  # Capital release on exits
    )

    # Start background threads
    threads: list[threading.Thread] = []

    t_trade = threading.Thread(target=trader.run_forever, name="TriggerAwareExecutor", daemon=True)
    t_trade.start(); threads.append(t_trade)
    logger.info("trigger-aware-executor: started")

    t_exit = threading.Thread(target=exit_exec.run_forever, name="ExitExecutor", daemon=True)
    t_exit.start(); threads.append(t_exit)
    logger.info("exit-executor: started")

    # Add monitoring thread for trigger status
    def monitor_triggers():
        while True:
            try:
                summary = trader.get_pending_trades_summary()
                if summary["total_pending"] > 0 or summary["total_triggered"] > 0:
                    logger.info(
                        f"TRIGGER_STATUS: pending={summary['total_pending']} "
                        f"triggered={summary['total_triggered']} "
                        f"symbols={list(summary['by_symbol'].keys())}"
                    )
                time.sleep(30)  # Log every 30 seconds
            except Exception as e:
                logger.exception(f"Monitor thread error: {e}")
                time.sleep(30)

    t_monitor = threading.Thread(target=monitor_triggers, name="TriggerMonitor", daemon=True)
    t_monitor.start(); threads.append(t_monitor)
    logger.info("trigger-monitor: started")

    # Lifecycle: start screener and block until EOD / request_exit
    _run_until_eod(screener, exit_exec, trader)
    
    return 0


def _run_until_eod(screener: ScreenerLive, exit_exec: ExitExecutor, trader: TriggerAwareExecutor, poll_sec: float = 0.2) -> None:
    logger.info("session start")
    stop = threading.Event()

    def _sig_handler(signum, frame):
        logger.warning(f"signal {signum} received – shutting down")
        stop.set()

    # Handle Ctrl+C / SIGTERM cleanly
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        screener.start()
        while not getattr(screener, "_request_exit", False) and not stop.is_set():
            time.sleep(poll_sec)
    except KeyboardInterrupt:
        logger.info("keyboard interrupt – stopping")
    finally:
        try:
            # Cancel all pending trades before EOD
            with trader._lock:
                for trade in trader.pending_trades.values():
                    if trade.state == TradeState.WAITING_TRIGGER:
                        trade.state = TradeState.CANCELLED
            logger.info("Cancelled all pending trades for EOD")
        except Exception as e:
            logger.warning("Failed to cancel pending trades: %s", e)
        
        try:
            exit_exec.square_off_all_open_positions()
        except Exception as e:
            logger.warning("final EOD sweep failed: %s", e)
        
        try:
            screener.stop()
        except Exception as e:
            logger.warning("screener.stop failed: %s", e)
        
        try:
            trader.stop()
        except Exception as e:
            logger.warning("trader.stop failed: %s", e)
            

    logger.info("session end (EOD)")


def _hhmm_on(day_str: str, hhmm: str) -> str:
    y, m, d = map(int, day_str.split("-"))
    h, mm = map(int, hhmm.split(":"))
    return f"{y:04d}-{m:02d}-{d:02d} {h:02d}:{mm:02d}:00"


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Use mock broker + archived ticks")
    ap.add_argument("--paper-trading", action="store_true", help="Paper trading mode: live data, simulated orders")
    ap.add_argument("--with-capital-limits", action="store_true", help="Enable capital management for backtests (default: disabled for fast testing)")
    ap.add_argument("--session-date", help="YYYY-MM-DD (required with --dry-run)")
    ap.add_argument("--from-hhmm", default="09:10")
    ap.add_argument("--to-hhmm",   default="15:30")
    ap.add_argument("--run-prefix", default="", help="Prefix for session folder names (used by engine.py)")
    ap.add_argument("--enable-cache", action="store_true", help="Enable structure detection caching")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main())