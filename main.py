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
from services.ingest.tick_router import register_tick_listener, register_tick_listener_full
from services.execution.trigger_aware_executor import TriggerAwareExecutor, TradeState
from services.capital_manager import CapitalManager
from services.state import PositionPersistence, BrokerReconciliation, validate_paper_position_on_recovery
from services.state.daily_cache_persistence import DailyCachePersistence
from pipelines.base_pipeline import set_base_config_override, load_base_config
from api import get_api_server, SessionState

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

    def get_ltp_with_level(self, symbol: str, check_level: Optional[float] = None, **kwargs) -> float:
        return self._real.get_ltp_with_level(symbol, check_level=check_level, **kwargs)

    def get_ltp_batch(self, symbols):
        return self._real.get_ltp_batch(symbols)

    @property
    def _last_bar_ohlc(self):
        """Delegate to underlying MockBroker's OHLC cache for T1/T2 detection."""
        return self._real._last_bar_ohlc


# ------------------------ Startup Recovery (Phase 4) ------------------------

def _merge_persisted_state_into_plan(pers_pos) -> dict:
    """
    Merge PersistedPosition.state into plan["_state"] for proper recovery.

    The exit_executor reads state from plan["_state"], but persistence stores
    updates in a separate 'state' field. This merges them back together.
    """
    plan = dict(pers_pos.plan) if pers_pos.plan else {}
    if pers_pos.state:
        # Merge persisted state into plan's _state
        plan_state = plan.get("_state", {})
        plan_state.update(pers_pos.state)
        plan["_state"] = plan_state
    return plan


def startup_recovery(
    broker,
    is_live_mode: bool,
    is_paper_mode: bool,
    log_dir,
    position_store: _PositionStore,
    trading_logger_instance=None
) -> Optional[PositionPersistence]:
    """
    Recover position state on startup.

    For live mode: Reconciles with broker positions
    For paper mode: Validates positions against current price (SL/T1/T2 checks)
    For dry-run (backtest): Returns None (no persistence needed)

    Args:
        broker: KiteBroker instance
        is_live_mode: True if live trading
        is_paper_mode: True if paper trading
        log_dir: Directory for position snapshot
        position_store: PositionStore to populate
        trading_logger_instance: TradingLogger for phantom exit logging

    Returns:
        PositionPersistence instance (or None for backtests)
    """
    from pathlib import Path

    # Backtests don't need persistence
    if not is_live_mode and not is_paper_mode:
        logger.info("[RECOVERY] Backtest mode - persistence disabled")
        return None

    # Use a dedicated state directory for persistence (not session-specific)
    # This allows recovery across sessions
    state_dir = Path(__file__).resolve().parent / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    persistence = PositionPersistence(state_dir)
    persisted = persistence.load_snapshot()

    if not persisted:
        logger.info("[RECOVERY] No persisted positions found")
        return persistence

    logger.info(f"[RECOVERY] Found {len(persisted)} persisted positions")

    if is_live_mode:
        # Live mode: Reconcile with broker
        try:
            reconciliation = BrokerReconciliation(broker)
            result = reconciliation.reconcile(persisted)

            # Log results
            if result.orphaned_app:
                logger.warning(f"[RECOVERY] Positions closed externally: {list(result.orphaned_app.keys())}")
            if result.manual_trades:
                logger.info(f"[RECOVERY] Manual trades detected (not managed): {list(result.manual_trades.keys())}")
            if result.qty_mismatch:
                logger.warning(f"[RECOVERY] QTY MISMATCH - adjusting to broker state: {list(result.qty_mismatch.keys())}")

            # Restore matched positions
            for sym, pers_pos in result.matched.items():
                merged_plan = _merge_persisted_state_into_plan(pers_pos)
                position_store.upsert(Position(
                    symbol=pers_pos.symbol,
                    side=pers_pos.side,
                    qty=pers_pos.qty,
                    avg_price=pers_pos.avg_price,
                    plan=merged_plan,
                ))
                logger.info(f"[RECOVERY] Restored position: {sym} {pers_pos.side} {pers_pos.qty}@{pers_pos.avg_price} state={pers_pos.state}")

            # Handle qty mismatches - trust broker
            for sym, (pers_pos, broker_pos) in result.qty_mismatch.items():
                adjusted_pos = reconciliation.adjust_for_broker(pers_pos, broker_pos)
                merged_plan = _merge_persisted_state_into_plan(adjusted_pos)
                position_store.upsert(Position(
                    symbol=adjusted_pos.symbol,
                    side=adjusted_pos.side,
                    qty=adjusted_pos.qty,
                    avg_price=adjusted_pos.avg_price,
                    plan=merged_plan,
                ))
                # Update persistence with adjusted qty
                persistence.update_position(sym, new_qty=adjusted_pos.qty)
                logger.info(f"[RECOVERY] Adjusted position (broker qty): {sym} {adjusted_pos.qty} state={adjusted_pos.state}")

            # Remove orphaned positions from persistence
            for sym in result.orphaned_app:
                persistence.remove_position(sym)

            logger.info(f"[RECOVERY] Recovered {len(result.matched) + len(result.qty_mismatch)} positions")

        except Exception as e:
            logger.error(f"[RECOVERY] Broker reconciliation failed: {e}")
            # Fall back to persisted state only
            for sym, pers_pos in persisted.items():
                merged_plan = _merge_persisted_state_into_plan(pers_pos)
                position_store.upsert(Position(
                    symbol=pers_pos.symbol,
                    side=pers_pos.side,
                    qty=pers_pos.qty,
                    avg_price=pers_pos.avg_price,
                    plan=merged_plan,
                ))
            logger.warning(f"[RECOVERY] Restored {len(persisted)} positions from snapshot (no broker verification)")

    else:  # Paper mode
        # Paper mode: Validate positions against current price before restoring
        # This handles positions where SL/T1/T2 would have been hit while offline
        restored_count = 0
        phantom_exit_count = 0

        for sym, pers_pos in persisted.items():
            # Get current price to validate position
            try:
                current_price = broker.get_ltp(sym)
                if current_price is None:
                    logger.warning(f"[RECOVERY] Could not get LTP for {sym}, skipping validation")
                    current_price = pers_pos.avg_price  # Fallback to entry price
            except Exception as e:
                logger.warning(f"[RECOVERY] LTP fetch failed for {sym}: {e}, using entry price")
                current_price = pers_pos.avg_price

            # Validate position against current price (SL/T1/T2 checks)
            should_restore, state_updates, phantom_logged = validate_paper_position_on_recovery(
                pers_pos=pers_pos,
                current_price=current_price,
                trading_logger=trading_logger_instance,
                persistence=persistence
            )

            if phantom_logged:
                phantom_exit_count += 1

            if not should_restore:
                continue  # Position was stopped out or T2 hit - already handled

            # Apply state updates (e.g., t1_done=True if T1 was hit)
            if state_updates:
                if pers_pos.state is None:
                    pers_pos.state = {}
                pers_pos.state.update(state_updates)

                # Update quantity if T1 was hit (partial exit happened)
                if "_remaining_qty" in state_updates:
                    pers_pos.qty = state_updates["_remaining_qty"]

                # Update persistence with new state
                persistence.update_position(sym, new_qty=pers_pos.qty, state_updates=state_updates)

            merged_plan = _merge_persisted_state_into_plan(pers_pos)
            position_store.upsert(Position(
                symbol=pers_pos.symbol,
                side=pers_pos.side,
                qty=pers_pos.qty,
                avg_price=pers_pos.avg_price,
                plan=merged_plan,
            ))
            restored_count += 1
            logger.info(
                f"[RECOVERY] Restored position (paper): {sym} {pers_pos.side} {pers_pos.qty}@{pers_pos.avg_price} "
                f"state={pers_pos.state} current_price={current_price:.2f}"
            )

        logger.info(
            f"[RECOVERY] Paper mode recovery complete: restored={restored_count} "
            f"phantom_exits={phantom_exit_count} total_persisted={len(persisted)}"
        )

    return persistence


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

    # Upload any pending sessions from previous runs (paper/live only)
    if args.paper_trading or (not args.dry_run and not args.paper_trading):
        try:
            from pathlib import Path
            from services.state.session_upload_persistence import SessionUploadTracker
            from oci.tools.upload_trading_session import upload_pending_sessions

            state_dir = Path(__file__).resolve().parent / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = Path(__file__).resolve().parent / "logs"

            tracker = SessionUploadTracker(state_dir)
            tracker.cleanup_old_entries(keep_days=30)  # Prevent unbounded growth
            uploaded = upload_pending_sessions(tracker, logs_dir)
            if uploaded > 0:
                logger.info(f"[STARTUP] Uploaded {uploaded} pending session(s) from previous runs")
        except Exception as e:
            logger.warning(f"[STARTUP] Failed to upload pending sessions: {e}")

    cfg = load_filters()  # validate early; raises if required keys are missing

    # Apply structure caching if requested (monkey-patches TradeDecisionGate.evaluate)
    # Also set flag in config so worker processes can enable caching
    if args.enable_cache:
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
        initial_capital=cap_mgmt_cfg['initial_capital'],
        max_positions=cap_mgmt_cfg['max_concurrent_positions'],
        risk_per_trade=cap_mgmt_cfg['risk_per_trade'],
        min_notional=cap_mgmt_cfg['min_notional'],
        capital_utilization=cap_mgmt_cfg['capital_utilization'],
        max_allocation_per_trade=cap_mgmt_cfg['max_allocation_per_trade'],
        mis_enabled=mis_enabled,
        mis_config_path=cap_mgmt_cfg.get('mis_leverage', {}).get('config_file')
    )

    # Set dynamic risk per trade in config (live/paper uses capital %, backtest uses fallback)
    base_cfg = load_base_config()
    fallback_risk = base_cfg.get('risk_per_trade_rupees', 1000.0)
    dynamic_risk = capital_manager.get_risk_per_trade(fallback=fallback_risk)
    set_base_config_override('risk_per_trade_rupees', dynamic_risk)

    # Order queue handles pacing/retries; it reads its own config internally
    oq = OrderQueue()

    def _prewarm_daily_cache(sdk):
        """Pre-warm daily cache: try disk first (2-5s), fallback to API (15min)."""
        cache_persistence = DailyCachePersistence()
        cached_data = cache_persistence.load_today()
        if cached_data:
            sdk.set_daily_cache(cached_data)
        result = sdk.prewarm_daily_cache(days=210)
        # Save to disk if we fetched from API (for next restart)
        if result.get("source") == "api":
            cache_persistence.save(sdk.get_daily_cache())

    # Tick recorder for paper/live trading (records to sidecar format for upload)
    tick_recorder = None

    # Pick mode
    if args.paper_trading:
        # Paper trading: live data + simulated orders
        from config.env_setup import env
        env.validate_for_paper_trading()  # Validate credentials before starting

        sdk = KiteClient()
        broker = KiteBroker(dry_run=True, ltp_cache=ltp_cache)
        logger.warning("🧪 PAPER TRADING MODE: Live data, simulated orders (no real trades)")

        # Initialize tick recorder for paper trading
        try:
            from sidecar.data_collector import TickRecorder as SidecarTickRecorder
            tick_recorder = SidecarTickRecorder(buffer_size=50000)
            logger.info("TICK_RECORDER | Initialized for paper trading session")
        except Exception as e:
            logger.warning(f"TICK_RECORDER | Failed to initialize: {e}")

        if not args.skip_prewarm:
            _prewarm_daily_cache(sdk)
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
        broker = KiteBroker(dry_run=False, ltp_cache=ltp_cache)
        logger.warning("💰 LIVE TRADING MODE: Real orders will be placed with real money!")

        # Initialize tick recorder for live trading
        try:
            from sidecar.data_collector import TickRecorder as SidecarTickRecorder
            tick_recorder = SidecarTickRecorder(buffer_size=50000)
            logger.info("TICK_RECORDER | Initialized for live trading session")
        except Exception as e:
            logger.warning(f"TICK_RECORDER | Failed to initialize: {e}")

        if not args.skip_prewarm:
            _prewarm_daily_cache(sdk)

    # Screener consumes the SDK (WS/ticker) + enqueues entry intents
    screener = ScreenerLive(sdk=sdk, order_queue=oq)

    # Bootstrap from sidecar data (instant startup on late start)
    # This populates ORB cache, daily levels, and 5m bars from pre-collected sidecar data
    bootstrap_result = {"success": False, "skipped": True, "reason": "not_attempted"}
    try:
        from sidecar import maybe_bootstrap_from_sidecar
        bootstrap_result = maybe_bootstrap_from_sidecar(screener)
        if bootstrap_result.get("success"):
            logger.info(f"SIDECAR | Bootstrapped: {bootstrap_result['orb_count']} ORB, "
                       f"{bootstrap_result.get('daily_levels_count', 0)} daily levels, "
                       f"{bootstrap_result['bars_count']} bars for {bootstrap_result['symbols_count']} symbols")
        elif bootstrap_result.get("skipped"):
            logger.debug(f"SIDECAR | Skipped: {bootstrap_result.get('reason', 'unknown')}")
    except Exception as e:
        logger.debug(f"SIDECAR | Bootstrap not available: {e}")

    # Tap the central tick router so entries & exits share the same tick clock
    def _ltp_tap(sym: str, price: float, qty: float, ts_dt):
        ts_pd = pd.Timestamp(ts_dt)  # ensure pandas timestamp
        ltp_cache.update(sym, price, ts_pd)

    register_tick_listener(_ltp_tap)

    # Register tick recorder for paper/live trading (records market data for upload)
    if tick_recorder is not None:
        def _tick_recorder_tap(sym: str, price: float, qty: float, cumvol: int, ts_dt):
            # Adapter: OnTickFull signature → SidecarTickRecorder.on_tick signature
            tick_recorder.on_tick(sym, price, qty, ts_dt, cumvol)

        register_tick_listener_full(_tick_recorder_tap)
        logger.info("TICK_RECORDER | Registered with tick router")

    # Risk + shared positions
    risk = RiskState(max_concurrent=int(cfg["max_concurrent_positions"]))
    positions = _PositionStore()

    # Initialize API server for production monitoring
    api = get_api_server(port=args.health_port)
    api.set_state(SessionState.RECOVERING)
    api.set_position_store(positions)
    api.set_capital_manager(capital_manager)
    api.set_ltp_cache(ltp_cache)
    api.set_kite_client(sdk)  # For broker API calls (funds, etc.)
    api.set_auth_token(args.admin_token)  # For protected endpoints
    api.start()

    # Startup recovery - restore positions from previous session
    from config.logging_config import get_log_directory
    is_live = not args.dry_run and not args.paper_trading
    persistence = startup_recovery(
        broker=broker,
        is_live_mode=is_live,
        is_paper_mode=args.paper_trading,
        log_dir=get_log_directory(),
        position_store=positions,
        trading_logger_instance=trading_logger  # For phantom exit logging in paper mode
    )

    trader = TriggerAwareExecutor(
        broker=broker,
        order_queue=oq,
        risk_state=risk,
        positions=positions,
        get_ltp_ts=ltp_cache.get_ltp_ts,
        bar_builder=screener.agg,  # Pass the BarBuilder instance
        trading_logger=trading_logger,  # Enhanced logging
        capital_manager=capital_manager,  # Capital & MIS management
        persistence=persistence,  # Position persistence for crash recovery
        api_server=api  # For checking pause state
    )

    # ExitExecutor with tick-level validation (like TriggerAwareExecutor)
    exit_exec = ExitExecutor(
        broker=broker,
        positions=positions,
        get_ltp_ts=ltp_cache.get_ltp_ts,   # <- EOD uses tick timestamps, not wall clock
        bar_builder=screener.agg,  # For tick-level exit validation
        trading_logger=trading_logger,  # Enhanced logging
        capital_manager=capital_manager,  # Capital release on exits
        persistence=persistence,  # Position persistence for crash recovery
        api_server=api  # For processing exit requests
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

    # Set health state to trading
    api.set_state(SessionState.TRADING)

    # Lifecycle: start screener and block until EOD / request_exit
    _run_until_eod(screener, exit_exec, trader, api, capital_manager)

    return 0


def _run_until_eod(
    screener: ScreenerLive,
    exit_exec: ExitExecutor,
    trader: TriggerAwareExecutor,
    api_server = None,
    capital_manager = None,
    poll_sec: float = 0.2
) -> None:
    logger.info("session start")
    stop = threading.Event()

    def _sig_handler(signum, frame):
        logger.warning(f"signal {signum} received – shutting down")
        stop.set()

    def _shutdown_via_http():
        """Callback for HTTP shutdown endpoint."""
        logger.info("Shutdown requested via HTTP")
        stop.set()

    # Handle Ctrl+C / SIGTERM cleanly
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # Wire up HTTP shutdown
    if api_server:
        api_server.set_shutdown_callback(_shutdown_via_http)

    try:
        screener.start()
        while not getattr(screener, "_request_exit", False) and not stop.is_set():
            time.sleep(poll_sec)
    except KeyboardInterrupt:
        logger.info("keyboard interrupt – stopping")
    finally:
        # Update health state
        if api_server:
            api_server.set_state(SessionState.SHUTTING_DOWN)

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

        # Save capital report for analytics
        if capital_manager:
            try:
                from config.logging_config import get_log_directory
                log_dir = get_log_directory()
                if log_dir and log_dir.exists():
                    capital_manager.save_final_report(log_dir)
            except Exception as e:
                logger.warning("Failed to save capital report: %s", e)

        # Stop API server
        if api_server:
            api_server.set_state(SessionState.STOPPED)
            api_server.stop()

        # Finalize tick recorder (flush buffers and merge part files)
        _tick_rec = locals().get('tick_recorder')
        if _tick_rec is not None:
            try:
                _tick_rec.finalize()
                logger.info(f"TICK_RECORDER | Finalized: {_tick_rec.tick_count:,} ticks recorded")
            except Exception as e:
                logger.warning(f"TICK_RECORDER | Finalization failed: {e}")

        # Upload current session to OCI (paper/live modes only)
        if args.paper_trading or (not args.dry_run and not args.paper_trading):
            try:
                from pathlib import Path
                from config.logging_config import get_log_directory, get_session_id
                from services.state.session_upload_persistence import SessionUploadTracker
                from oci.tools.upload_trading_session import upload_session

                log_dir = get_log_directory()
                session_id = get_session_id()
                mode = "paper" if args.paper_trading else "live"

                if log_dir and log_dir.exists():
                    logger.info(f"[SHUTDOWN] Uploading session {session_id} to OCI...")
                    state_dir = Path(__file__).resolve().parent / "state"
                    tracker = SessionUploadTracker(state_dir)

                    if upload_session(log_dir, mode):
                        tracker.mark_uploaded(session_id, f"{mode}-trading-logs")
                        logger.info(f"[SHUTDOWN] Session uploaded successfully to {mode}-trading-logs")
                    else:
                        logger.warning(f"[SHUTDOWN] Failed to upload session {session_id}")
            except Exception as e:
                logger.warning(f"[SHUTDOWN] Session upload failed: {e}")

    logger.info("session end (EOD)")


def _hhmm_on(day_str: str, hhmm: str) -> str:
    y, m, d = map(int, day_str.split("-"))
    h, mm = map(int, hhmm.split(":"))
    return f"{y:04d}-{m:02d}-{d:02d} {h:02d}:{mm:02d}:00"


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Use mock broker + archived ticks")
    ap.add_argument("--paper-trading", action="store_true", help="Paper trading mode: live data, simulated orders")
    ap.add_argument("--skip-prewarm", action="store_true", help="Skip daily data pre-warming (faster startup but 11-min delay at 09:40)")
    ap.add_argument("--with-capital-limits", action="store_true", help="Enable capital management for backtests (default: disabled for fast testing)")
    ap.add_argument("--session-date", help="YYYY-MM-DD (required with --dry-run)")
    ap.add_argument("--from-hhmm", default="09:10")
    ap.add_argument("--to-hhmm",   default="15:30")
    ap.add_argument("--run-prefix", default="", help="Prefix for session folder names (used by engine.py)")
    ap.add_argument("--enable-cache", action="store_true", help="Enable structure detection caching")
    ap.add_argument("--health-port", type=int, default=8080, help="Port for health server (default: 8080)")
    ap.add_argument("--admin-token", default=None, help="Admin token for protected endpoints (default: from ADMIN_TOKEN env var)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main())