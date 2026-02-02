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
from pathlib import Path
from typing import Optional
import pandas as pd

from config.logging_config import get_agent_logger, get_trading_logger
from config.filters_setup import load_filters

from services.orders.order_queue import OrderQueue
from services.screener_live import ScreenerLive
from services.execution.models import RiskState
from services.execution.exit_executor import ExitExecutor
from services.ingest.tick_router import register_tick_listener, register_tick_listener_full
from services.execution.trigger_aware_executor import TriggerAwareExecutor, TradeState
from services.capital_manager import CapitalManager
from services.state.position_store import PositionStore
from services.state.recovery import startup_recovery
from services.state.daily_cache_persistence import DailyCachePersistence
from pipelines.base_pipeline import set_base_config_override, load_base_config
from api import get_api_server, SessionState

# Shared LTP cache (supports multi-instance trading via Redis)
from market_data import SharedLTPCache

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



# PositionStore extracted to services/state/position_store.py


# ------------------------ LTP cache (tick clock) ------------------------
# Uses SharedLTPCache which supports multi-instance trading via Redis.
# Mode is determined by market_data_bus.ltp_mode in configuration.json:
#   - standalone: local-only cache (default, backward compatible)
#   - publisher: writes to local AND Redis (LIVE mode)
#   - subscriber: reads from Redis only (PAPER mode with shared data)

def _create_ltp_cache() -> SharedLTPCache:
    """Create LTP cache based on market_data_bus config."""
    try:
        cfg = load_filters()
        mdb_config = cfg.get("market_data_bus", {})
        mode = mdb_config.get("ltp_mode", "standalone")
        redis_url = mdb_config.get("redis_url", "redis://localhost:6379/0")
        return SharedLTPCache(mode=mode, redis_url=redis_url)
    except Exception:
        # Fallback to standalone mode if config loading fails
        return SharedLTPCache(mode="standalone")

ltp_cache = _create_ltp_cache()


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


# startup_recovery extracted to services/state/recovery.py


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

    # Enable shared market data (subscribe to Market Data Service via Redis)
    if args.shared_market_data:
        if "market_data_bus" not in cfg:
            cfg["market_data_bus"] = {}
        cfg["market_data_bus"]["mode"] = "subscriber"
        cfg["market_data_bus"]["ltp_mode"] = "subscriber"
        logger.info("[MARKET_DATA] Shared mode: subscribing to Market Data Service via Redis")

        # Reinitialize global ltp_cache in subscriber mode
        global ltp_cache
        redis_url = cfg["market_data_bus"].get("redis_url", "redis://localhost:6379/0")
        ltp_cache = SharedLTPCache(mode="subscriber", redis_url=redis_url)
        logger.info("[MARKET_DATA] LTP cache in subscriber mode")

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
    mis_fetcher = None  # MIS validation fetcher (paper trading only)

    if args.paper_trading or (not args.dry_run and not args.paper_trading):
        # Paper trading or live trading: always enable
        capital_enabled = True
        mis_enabled = True
        logger.info("[CAPITAL] Auto-enabled capital management (paper/live mode)")

        # Fetch MIS-allowed stocks from Zerodha (paper trading only)
        # Live trading doesn't need this - Zerodha broker will reject non-MIS orders anyway
        if args.paper_trading:
            try:
                from services.state.zerodha_mis_fetcher import ZerodhaMISFetcher
                mis_fetcher = ZerodhaMISFetcher()
                if mis_fetcher.load_from_zerodha(timeout_sec=30):
                    logger.info(f"MIS_FETCHER | Loaded {mis_fetcher.count()} MIS-allowed symbols from Zerodha")
                else:
                    logger.warning("MIS_FETCHER | Failed to fetch MIS list - trades on non-MIS stocks may occur")
                    mis_fetcher = None
            except Exception as e:
                logger.warning(f"MIS_FETCHER | Error initializing: {e}")
                mis_fetcher = None

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

    # Extract risk config (new format with mode, or legacy risk_pct_per_trade)
    risk_cfg = cap_mgmt_cfg.get('risk', {})
    risk_mode = risk_cfg.get('mode')
    risk_fixed_amount = risk_cfg.get('fixed_amount')
    risk_percentage = risk_cfg.get('percentage')

    # CLI override for risk mode (--risk-mode fixed|percentage, --risk-value <amount>)
    if hasattr(args, 'risk_mode') and args.risk_mode:
        risk_mode = args.risk_mode
        logger.info(f"[CAPITAL] Risk mode overridden via CLI: {risk_mode}")
    if hasattr(args, 'risk_value') and args.risk_value is not None:
        if risk_mode == 'fixed':
            risk_fixed_amount = args.risk_value
        else:
            risk_percentage = args.risk_value
        logger.info(f"[CAPITAL] Risk value overridden via CLI: {args.risk_value}")

    capital_manager = CapitalManager(
        enabled=capital_enabled,
        initial_capital=cap_mgmt_cfg['paper_initial_capital'] if (args.paper_trading or args.dry_run) else cap_mgmt_cfg['initial_capital'],
        max_positions=cap_mgmt_cfg['max_concurrent_positions'],
        min_notional_pct=cap_mgmt_cfg['min_notional_pct'],
        capital_utilization=cap_mgmt_cfg['capital_utilization'],
        max_allocation_per_trade=cap_mgmt_cfg['max_allocation_per_trade'],
        risk_mode=risk_mode,
        risk_fixed_amount=risk_fixed_amount,
        risk_percentage=risk_percentage,
        mis_enabled=mis_enabled,
        mis_config_path=cap_mgmt_cfg.get('mis_leverage', {}).get('config_file'),
        mis_fetcher=mis_fetcher,  # Paper trading: validates against Zerodha MIS list
    )

    # Set dynamic risk per trade in config (live/paper uses capital %, backtest uses fallback)
    base_cfg = load_base_config()
    fallback_risk = base_cfg.get('risk_per_trade_rupees', 1000.0)
    dynamic_risk = capital_manager.get_risk_per_trade(fallback=fallback_risk)
    set_base_config_override('risk_per_trade_rupees', dynamic_risk)

    # Determine execution mode: in_process (all-in-one) or scan_only (exec runs separately)
    execution_mode = getattr(args, 'execution_mode', None) or cfg.get("execution_mode", "in_process")
    if args.dry_run:
        execution_mode = "in_process"  # Backtest always forces in_process

    if execution_mode != "in_process":
        logger.info("EXEC_MODE | %s", execution_mode)

    # Order queue: local (in_process), Redis publish (scan_only/separated), Redis consume (exec_only)
    # Each instance MUST have its own dedicated channel to prevent cross-instance plan leakage.
    # - separated: auto-generates unique channel (parent passes to child via args_dict)
    # - scan_only/exec_only: requires --instance-id to pair scan+exec processes
    # - in_process: uses local OrderQueue (no Redis)
    mdb_cfg = cfg.get("market_data_bus", {})
    redis_url = mdb_cfg.get("redis_url", "redis://localhost:6379/0")
    base_queue_key = cfg["trade_plan_queue_key"]

    if execution_mode == "separated":
        import uuid
        queue_key = f"{base_queue_key}:{uuid.uuid4().hex[:8]}"
        from services.execution.redis_plan_queue import RedisPlanPublisher
        oq = RedisPlanPublisher(redis_url=redis_url, queue_key=queue_key)
    elif execution_mode == "scan_only":
        instance_id = getattr(args, 'instance_id', None)
        if not instance_id:
            raise ValueError("--instance-id is required for scan_only mode (e.g., --instance-id=live)")
        queue_key = f"{base_queue_key}:{instance_id}"
        from services.execution.redis_plan_queue import RedisPlanPublisher
        oq = RedisPlanPublisher(redis_url=redis_url, queue_key=queue_key)
    elif execution_mode == "exec_only":
        instance_id = getattr(args, 'instance_id', None)
        if not instance_id:
            raise ValueError("--instance-id is required for exec_only mode (e.g., --instance-id=live)")
        queue_key = f"{base_queue_key}:{instance_id}"
        from services.execution.redis_plan_queue import RedisPlanConsumer
        oq = RedisPlanConsumer(redis_url=redis_url, queue_key=queue_key)
    else:
        queue_key = base_queue_key
        oq = OrderQueue()

    def _prewarm_daily_cache(sdk):
        """Pre-warm daily cache: Redis (1-3s) -> rolling disk (2-5s) -> API (15min)."""
        from datetime import datetime as _dt

        mdb_cfg = cfg.get("market_data_bus", {})
        dc_cfg = mdb_cfg.get("daily_cache_redis", {})
        redis_enabled = dc_cfg.get("enabled", False)

        cache_persistence = DailyCachePersistence()
        loaded_from = None

        # Step 1: Try Redis (shared by MDS publisher)
        if redis_enabled:
            try:
                from market_data.market_data_bus import MarketDataBus
                redis_url = mdb_cfg["redis_url"]
                bus = MarketDataBus(mode="subscriber", redis_url=redis_url)
                today = _dt.now().date().isoformat()
                redis_cache = bus.get_daily_cache(today)
                bus.shutdown()
                if redis_cache:
                    sdk.set_daily_cache(redis_cache)
                    loaded_from = "redis"
                    logger.info(f"DAILY_CACHE | Loaded {len(redis_cache)} symbols from Redis")
            except Exception as e:
                logger.warning(f"DAILY_CACHE | Redis load failed: {e}")

        # Step 2: Try rolling disk cache (today's or yesterday's)
        if loaded_from is None:
            cached_data = cache_persistence.load_latest()
            if cached_data:
                sdk.set_daily_cache(cached_data)
                loaded_from = "disk"

        # Step 3: API fallback (90% threshold skips if rolling cache loaded)
        result = sdk.prewarm_daily_cache(days=210)
        if result.get("source") == "api":
            loaded_from = "api"
            cache_persistence.save(sdk.get_daily_cache())

        # Step 4: Cooperative publish to Redis (seed for other instances)
        if redis_enabled and loaded_from in ("disk", "api"):
            try:
                from market_data.market_data_bus import MarketDataBus
                redis_url = mdb_cfg["redis_url"]
                bus = MarketDataBus(mode="publisher", redis_url=redis_url)
                today = _dt.now().date().isoformat()
                cache = sdk.get_daily_cache()
                if cache:
                    bus.publish_daily_cache(today, cache, dc_cfg)
                bus.shutdown()
            except Exception as e:
                logger.warning(f"DAILY_CACHE | Redis publish failed (non-fatal): {e}")

        logger.info(f"DAILY_CACHE | Prewarm complete, source={loaded_from}")

    # Tick recorder for paper/live trading (records to parquet for upload)
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
        tick_cfg = cfg.get("tick_recording", {})
        if tick_cfg.get("enabled", True):
            try:
                from market_data.tick_recorder import TickRecorder
                tick_recorder = TickRecorder(buffer_size=tick_cfg.get("buffer_size", 50000))
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
        tick_cfg = cfg.get("tick_recording", {})
        if tick_cfg.get("enabled", True):
            try:
                from market_data.tick_recorder import TickRecorder
                tick_recorder = TickRecorder(buffer_size=tick_cfg.get("buffer_size", 50000))
                logger.info("TICK_RECORDER | Initialized for live trading session")
            except Exception as e:
                logger.warning(f"TICK_RECORDER | Failed to initialize: {e}")

        if not args.skip_prewarm:
            _prewarm_daily_cache(sdk)

    # ---------- exec_only: full execution pipeline, no screener ----------
    if execution_mode == "exec_only":
        from market_data import BarSubscriber
        from services.execution.exec_heartbeat import ExecHeartbeatPublisher

        # BarSubscriber as tick source (receives from MDS via Redis — own GIL isolation)
        bar_subscriber = BarSubscriber(redis_url=redis_url, symbols=None)
        bar_subscriber.init_redis()

        # Wire LTP cache to BarSubscriber ticks
        _original_on_tick = bar_subscriber.on_tick
        def _ltp_tap_on_tick(symbol, price, volume, ts):
            _original_on_tick(symbol, price, volume, ts)
            ltp_cache.update(symbol, float(price), pd.Timestamp(ts))
        bar_subscriber.on_tick = _ltp_tap_on_tick

        # Risk + shared positions
        risk = RiskState(max_concurrent=int(cfg["max_concurrent_positions"]))
        positions = PositionStore()

        # API server
        api = get_api_server(port=args.health_port)
        api.set_state(SessionState.RECOVERING)
        api.set_position_store(positions)
        api.set_capital_manager(capital_manager)
        api.set_ltp_cache(ltp_cache)
        api.set_kite_client(sdk)
        api.set_auth_token(args.admin_token)

        state_dir = Path(__file__).resolve().parent / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        api.set_state_dir(state_dir)
        api.start()

        # WebSocket server
        from api.websocket_server import WebSocketServer, LTPBatcher
        ws_port = args.ws_port if args.ws_port else args.health_port + 1
        ws_server = WebSocketServer(port=ws_port)
        ws_server.start()
        api.set_websocket_server(ws_server)

        positions.set_api_server(api)
        ltp_batcher = LTPBatcher(api.broadcast_ws, interval_ms=500)
        ltp_cache.set_ltp_batcher(ltp_batcher)

        # Startup recovery
        from config.logging_config import get_log_directory
        is_live = not args.dry_run and not args.paper_trading
        persistence = startup_recovery(
            broker=broker,
            is_live_mode=is_live,
            is_paper_mode=args.paper_trading,
            log_dir=get_log_directory(),
            position_store=positions,
            trading_logger_instance=trading_logger,
        )

        # Executors — use BarSubscriber (duck-typed BarBuilder) and RedisPlanConsumer (duck-typed OrderQueue)
        trader = TriggerAwareExecutor(
            broker=broker,
            order_queue=oq,           # RedisPlanConsumer
            risk_state=risk,
            positions=positions,
            get_ltp_ts=ltp_cache.get_ltp_ts,
            bar_builder=bar_subscriber,  # BarSubscriber (duck-typed BarBuilder)
            trading_logger=trading_logger,
            capital_manager=capital_manager,
            persistence=persistence,
            api_server=api,
        )
        exit_exec = ExitExecutor(
            broker=broker,
            positions=positions,
            get_ltp_ts=ltp_cache.get_ltp_ts,
            bar_builder=bar_subscriber,
            trading_logger=trading_logger,
            capital_manager=capital_manager,
            persistence=persistence,
            api_server=api,
        )

        # Start BarSubscriber (begin receiving ticks from MDS)
        bar_subscriber.start()
        logger.info("EXEC_ONLY | BarSubscriber started — receiving ticks from MDS")

        # Start executor threads
        threads: list[threading.Thread] = []
        t_trade = threading.Thread(target=trader.run_forever, name="TriggerAwareExecutor", daemon=True)
        t_trade.start(); threads.append(t_trade)
        logger.info("trigger-aware-executor: started")

        t_exit = threading.Thread(target=exit_exec.run_forever, name="ExitExecutor", daemon=True)
        t_exit.start(); threads.append(t_exit)
        logger.info("exit-executor: started")

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
                    time.sleep(30)
                except Exception as e:
                    logger.exception(f"Monitor thread error: {e}")
                    time.sleep(30)

        t_monitor = threading.Thread(target=monitor_triggers, name="TriggerMonitor", daemon=True)
        t_monitor.start(); threads.append(t_monitor)

        # Heartbeat publisher — scan process checks this before enqueuing
        hb_cfg = cfg["exec_heartbeat"]
        heartbeat = ExecHeartbeatPublisher(
            redis_url=redis_url,
            key=hb_cfg["key"],
            interval_sec=hb_cfg["interval_sec"],
            ttl_sec=hb_cfg["ttl_sec"],
        )
        heartbeat.start()

        api.set_state(SessionState.TRADING)
        logger.info("EXEC_ONLY | All components started — ready for trade plans from Redis")

        # Run until EOD / signal
        stop = threading.Event()

        def _sig_handler(signum, frame):
            logger.warning(f"signal {signum} received – shutting down exec_only")
            stop.set()

        def _shutdown_via_http():
            logger.info("Shutdown requested via HTTP")
            stop.set()

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
        if api:
            api.set_shutdown_callback(_shutdown_via_http)

        try:
            while not stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("keyboard interrupt – stopping exec_only")
        finally:
            api.set_state(SessionState.SHUTTING_DOWN)

            try:
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
                trader.stop()
            except Exception as e:
                logger.warning("trader.stop failed: %s", e)

            if capital_manager:
                try:
                    log_dir = get_log_directory()
                    if log_dir and log_dir.exists():
                        capital_manager.save_final_report(log_dir)
                except Exception:
                    pass

            heartbeat.stop()
            bar_subscriber.shutdown()
            oq.shutdown()
            api.set_state(SessionState.STOPPED)
            api.stop()

        logger.info("EXEC_ONLY | Session end")
        return 0

    # ---------- separated: spawn exec child, then run scan in this process ----------
    if execution_mode == "separated":
        import multiprocessing as mp
        from config.logging_config import get_log_directory
        from services.execution.exec_process import run_exec_child

        log_dir = get_log_directory()
        stop_event = mp.Event()
        args_dict = {
            "paper_trading": args.paper_trading,
            "dry_run": args.dry_run,
            "health_port": args.health_port,
            "ws_port": args.ws_port,
            "admin_token": args.admin_token,
            "queue_key": queue_key,  # Auto-generated unique channel for this instance
        }

        exec_child = mp.Process(
            target=run_exec_child,
            args=(str(log_dir), stop_event, args_dict),
            name="ExecChild",
            daemon=False,  # Not daemon — we join it on shutdown
        )
        exec_child.start()
        logger.info("SEPARATED | Exec child spawned (PID=%d)", exec_child.pid)

        # Wait for exec child heartbeat before starting scan
        time.sleep(3)
        try:
            from services.execution.exec_heartbeat import ExecHeartbeatChecker
            hb_cfg = cfg["exec_heartbeat"]
            hb = ExecHeartbeatChecker(redis_url=redis_url, key=hb_cfg["key"])
            if hb.is_alive():
                logger.info("SEPARATED | Exec child heartbeat OK")
            else:
                logger.warning("SEPARATED | Exec child heartbeat not yet detected — continuing anyway")
            hb.shutdown()
        except Exception as e:
            logger.warning("SEPARATED | Heartbeat check failed: %s", e)

        # Parent becomes scan process
        screener = ScreenerLive(sdk=sdk, order_queue=oq)
        logger.info("SEPARATED | Starting screener (exec runs in child process PID=%d)", exec_child.pid)

        stop = threading.Event()

        def _sig_handler(signum, frame):
            logger.warning(f"signal {signum} received – shutting down separated mode")
            stop.set()

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        try:
            screener.start()
            while not getattr(screener, "_request_exit", False) and not stop.is_set():
                # Also check if exec child crashed
                if not exec_child.is_alive():
                    logger.error("SEPARATED | Exec child died (exit code=%s) — stopping scan", exec_child.exitcode)
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("keyboard interrupt – stopping separated mode")
        finally:
            try:
                screener.stop()
            except Exception as e:
                logger.warning("screener.stop failed: %s", e)
            if hasattr(oq, 'shutdown'):
                oq.shutdown()

            # Signal exec child to stop and wait for it
            logger.info("SEPARATED | Signalling exec child to stop")
            stop_event.set()
            exec_child.join(timeout=30)
            if exec_child.is_alive():
                logger.warning("SEPARATED | Exec child did not exit in 30s — terminating")
                exec_child.terminate()
                exec_child.join(timeout=5)

        logger.info("SEPARATED | Session end")
        return 0

    # ---------- scan_only / in_process: need ScreenerLive ----------

    # Screener consumes the SDK (WS/ticker) + enqueues entry intents
    screener = ScreenerLive(sdk=sdk, order_queue=oq)

    # Late-start warmup is now handled by Redis bar backfill in screener_live.py
    # (see late_start_warmup config and _late_start_backfill method)

    # ---------- scan_only: early return (no executors, no positions) ----------
    if execution_mode == "scan_only":
        # Check exec process heartbeat (non-blocking, advisory only)
        try:
            from services.execution.exec_heartbeat import ExecHeartbeatChecker
            hb_cfg = cfg["exec_heartbeat"]
            hb = ExecHeartbeatChecker(redis_url=redis_url, key=hb_cfg["key"])
            if hb.is_alive():
                logger.info("SCAN_ONLY | Exec process heartbeat OK")
            else:
                logger.warning("SCAN_ONLY | Exec process heartbeat not detected — plans will queue in Redis")
            hb.shutdown()
        except Exception as e:
            logger.warning("SCAN_ONLY | Heartbeat check failed: %s", e)

        # Simplified run loop: scan + publish plans, no local execution
        logger.info("SCAN_ONLY | Starting screener (no local executors)")
        stop = threading.Event()

        def _sig_handler(signum, frame):
            logger.warning(f"signal {signum} received – shutting down scan_only")
            stop.set()

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        try:
            screener.start()
            while not getattr(screener, "_request_exit", False) and not stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("keyboard interrupt – stopping scan_only")
        finally:
            try:
                screener.stop()
            except Exception as e:
                logger.warning("screener.stop failed: %s", e)
            if hasattr(oq, 'shutdown'):
                oq.shutdown()

        logger.info("SCAN_ONLY | Session end")
        return 0

    # ---------- in_process: full execution pipeline below ----------

    # Tap the central tick router so entries & exits share the same tick clock
    def _ltp_tap(sym: str, price: float, qty: float, ts_dt):
        ts_pd = pd.Timestamp(ts_dt)  # ensure pandas timestamp
        ltp_cache.update(sym, price, ts_pd)

    register_tick_listener(_ltp_tap)

    # Register tick recorder for paper/live trading (records market data for upload)
    if tick_recorder is not None:
        def _tick_recorder_tap(sym: str, price: float, qty: float, cumvol: int, ts_dt):
            # Adapter: OnTickFull signature → TickRecorder.on_tick signature
            tick_recorder.on_tick(sym, price, qty, ts_dt, cumvol)

        register_tick_listener_full(_tick_recorder_tap)
        logger.info("TICK_RECORDER | Registered with tick router")

    # Risk + shared positions
    risk = RiskState(max_concurrent=int(cfg["max_concurrent_positions"]))
    positions = PositionStore()

    # Initialize API server for production monitoring
    api = get_api_server(port=args.health_port)
    api.set_state(SessionState.RECOVERING)
    api.set_position_store(positions)
    api.set_capital_manager(capital_manager)
    api.set_ltp_cache(ltp_cache)
    api.set_kite_client(sdk)  # For broker API calls (funds, etc.)
    api.set_auth_token(args.admin_token)  # For protected endpoints

    # Set state directory for closed trades persistence
    state_dir = Path(__file__).resolve().parent / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    api.set_state_dir(state_dir)

    api.start()

    # Initialize WebSocket server for real-time dashboard updates
    from api.websocket_server import WebSocketServer, LTPBatcher
    ws_port = args.ws_port if args.ws_port else args.health_port + 1
    ws_server = WebSocketServer(port=ws_port)
    ws_server.start()
    api.set_websocket_server(ws_server)

    # Wire up position store and LTP cache for WebSocket broadcasts
    positions.set_api_server(api)
    ltp_batcher = LTPBatcher(api.broadcast_ws, interval_ms=500)  # Throttled LTP updates
    ltp_cache.set_ltp_batcher(ltp_batcher)

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

    # Set state to trading
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
        # Update API state
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
    ap.add_argument("--ws-port", type=int, default=None, help="Port for WebSocket server (default: health-port + 1)")
    ap.add_argument("--admin-token", default=None, help="Admin token for protected endpoints (default: from ADMIN_TOKEN env var)")
    # Risk mode configuration (overrides config file)
    ap.add_argument("--risk-mode", choices=["fixed", "percentage"], default=None,
                    help="Risk calculation mode: 'fixed' for fixed Rs. amount, 'percentage' for %% of capital")
    ap.add_argument("--risk-value", type=float, default=None,
                    help="Risk value: Rs. amount if mode=fixed (e.g., 1000), or decimal %% if mode=percentage (e.g., 0.01 for 1%%)")
    # Shared market data (subscribe to standalone Market Data Service)
    ap.add_argument("--shared-market-data", action="store_true",
                    help="Subscribe to Market Data Service via Redis (requires: python -m market_data.market_data_service)")
    ap.add_argument("--execution-mode", choices=["in_process", "separated", "scan_only", "exec_only"], default=None,
                    help="in_process=all-in-one (default), separated=auto-spawn exec child process, scan_only/exec_only=manual two-terminal")
    ap.add_argument("--instance-id", default=None,
                    help="Unique instance identifier for Redis plan queue channel (e.g., live, paper1, paper2). "
                         "Required when running multiple instances to prevent cross-instance plan leakage.")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main())