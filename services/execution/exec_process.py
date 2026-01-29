"""
Execution Child Process — spawned by main.py in 'separated' mode.

Receives the parent's log directory and creates execution components
(broker, executors, API) in a separate GIL-isolated process.

This module's run_exec_child() is the target for multiprocessing.Process.
It must be a top-level function in a separate module for Windows 'spawn' compatibility.
"""
from __future__ import annotations

import os
import time
import signal
import threading
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def run_exec_child(
    log_dir_str: str,
    stop_event,  # multiprocessing.Event — set by parent on shutdown
    args_dict: Dict[str, Any],
) -> None:
    """
    Entry point for the execution child process.

    Creates broker, BarSubscriber, executors, API server, and heartbeat.
    Runs until stop_event is set, signal received, or EOD.

    Args:
        log_dir_str: Path to parent's log directory (string for pickling)
        stop_event: multiprocessing.Event — parent sets this to request shutdown
        args_dict: Serializable subset of argparse args (paper_trading, health_port, etc.)
    """
    log_dir = Path(log_dir_str)

    # 1. Initialize child process logging (shared directory, separate agent log)
    from config.logging_config import initialize_child_loggers, get_agent_logger, get_trading_logger
    initialize_child_loggers(log_dir, "exec")
    logger = get_agent_logger()
    trading_logger = get_trading_logger()

    logger.info("EXEC_CHILD | Process started (PID=%d, parent=%d)", os.getpid(), os.getppid())

    try:
        _run_exec_loop(logger, trading_logger, stop_event, args_dict)
    except Exception as e:
        logger.exception("EXEC_CHILD | Fatal error: %s", e)
    finally:
        logger.info("EXEC_CHILD | Process exiting (PID=%d)", os.getpid())


def _run_exec_loop(logger, trading_logger, stop_event, args_dict: Dict[str, Any]) -> None:
    """Core execution loop — separated for clean error handling."""

    from config.filters_setup import load_filters
    from config.logging_config import get_log_directory
    from services.capital_manager import CapitalManager
    from services.execution.trade_executor import RiskState
    from services.execution.trigger_aware_executor import TriggerAwareExecutor, TradeState
    from services.execution.exit_executor import ExitExecutor
    from services.execution.redis_plan_queue import RedisPlanConsumer
    from services.execution.exec_heartbeat import ExecHeartbeatPublisher
    from services.state.position_store import PositionStore
    from services.state.recovery import startup_recovery
    from market_data import BarSubscriber, SharedLTPCache
    from pipelines.base_pipeline import set_base_config_override, load_base_config
    from api import get_api_server, SessionState

    cfg = load_filters()
    paper_trading = args_dict["paper_trading"]
    is_live = not args_dict["dry_run"] and not paper_trading

    # --- Market data config ---
    mdb_cfg = cfg.get("market_data_bus", {})
    redis_url = mdb_cfg.get("redis_url", "redis://localhost:6379/0")
    queue_key = cfg["trade_plan_queue_key"]

    # --- RedisPlanConsumer (duck-typed OrderQueue) ---
    plan_consumer = RedisPlanConsumer(redis_url=redis_url, queue_key=queue_key)

    # --- LTP cache (subscriber mode — populated by BarSubscriber ticks) ---
    ltp_cache = SharedLTPCache(mode="subscriber", redis_url=redis_url)

    # --- Broker ---
    if paper_trading:
        from broker.kite.kite_client import KiteClient
        from broker.kite.kite_broker import KiteBroker
        sdk = KiteClient()
        broker = KiteBroker(dry_run=True, ltp_cache=ltp_cache)
        logger.info("EXEC_CHILD | Paper trading broker initialized")
    elif is_live:
        from broker.kite.kite_client import KiteClient
        from broker.kite.kite_broker import KiteBroker
        sdk = KiteClient()
        broker = KiteBroker(dry_run=False, ltp_cache=ltp_cache)
        logger.info("EXEC_CHILD | Live trading broker initialized")
    else:
        logger.error("EXEC_CHILD | Backtest mode not supported in separated execution")
        return

    # --- Capital manager ---
    cap_mgmt_cfg = cfg.get('capital_management', {})
    risk_cfg = cap_mgmt_cfg.get('risk', {})

    mis_fetcher = None
    if paper_trading:
        try:
            from services.state.zerodha_mis_fetcher import ZerodhaMISFetcher
            mis_fetcher = ZerodhaMISFetcher()
            if mis_fetcher.load_from_zerodha(timeout_sec=30):
                logger.info("EXEC_CHILD | MIS_FETCHER loaded %d symbols", mis_fetcher.count())
            else:
                mis_fetcher = None
        except Exception as e:
            logger.warning("EXEC_CHILD | MIS_FETCHER error: %s", e)

    capital_manager = CapitalManager(
        enabled=True,
        initial_capital=cap_mgmt_cfg['initial_capital'],
        max_positions=cap_mgmt_cfg['max_concurrent_positions'],
        min_notional_pct=cap_mgmt_cfg['min_notional_pct'],
        capital_utilization=cap_mgmt_cfg['capital_utilization'],
        max_allocation_per_trade=cap_mgmt_cfg['max_allocation_per_trade'],
        risk_mode=risk_cfg.get('mode'),
        risk_fixed_amount=risk_cfg.get('fixed_amount'),
        risk_percentage=risk_cfg.get('percentage'),
        mis_enabled=True,
        mis_config_path=cap_mgmt_cfg.get('mis_leverage', {}).get('config_file'),
        mis_fetcher=mis_fetcher,
    )

    # Set dynamic risk per trade
    base_cfg = load_base_config()
    fallback_risk = base_cfg.get('risk_per_trade_rupees', 1000.0)
    dynamic_risk = capital_manager.get_risk_per_trade(fallback=fallback_risk)
    set_base_config_override('risk_per_trade_rupees', dynamic_risk)

    # --- BarSubscriber (tick source from MDS via Redis) ---
    bar_subscriber = BarSubscriber(redis_url=redis_url, symbols=None)
    bar_subscriber.init_redis()

    # Wire LTP cache to BarSubscriber ticks
    _original_on_tick = bar_subscriber.on_tick
    def _ltp_tap_on_tick(symbol, price, volume, ts):
        _original_on_tick(symbol, price, volume, ts)
        ltp_cache.update(symbol, float(price), pd.Timestamp(ts))
    bar_subscriber.on_tick = _ltp_tap_on_tick

    # --- Risk + position store ---
    risk = RiskState(max_concurrent=int(cfg["max_concurrent_positions"]))
    positions = PositionStore()

    # --- API server (exec child owns the dashboard) ---
    health_port = args_dict["health_port"]
    api = get_api_server(port=health_port)
    api.set_state(SessionState.RECOVERING)
    api.set_position_store(positions)
    api.set_capital_manager(capital_manager)
    api.set_ltp_cache(ltp_cache)
    api.set_kite_client(sdk)
    api.set_auth_token(args_dict.get("admin_token"))

    state_dir = Path(__file__).resolve().parents[2] / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    api.set_state_dir(state_dir)
    api.start()

    # WebSocket server
    from api.websocket_server import WebSocketServer, LTPBatcher
    ws_port = args_dict.get("ws_port") or (health_port + 1)
    ws_server = WebSocketServer(port=ws_port)
    ws_server.start()
    api.set_websocket_server(ws_server)

    positions.set_api_server(api)
    ltp_batcher = LTPBatcher(api.broadcast_ws, interval_ms=500)
    ltp_cache.set_ltp_batcher(ltp_batcher)

    # --- Startup recovery ---
    persistence = startup_recovery(
        broker=broker,
        is_live_mode=is_live,
        is_paper_mode=paper_trading,
        log_dir=get_log_directory(),
        position_store=positions,
        trading_logger_instance=trading_logger,
    )

    # --- Executors ---
    trader = TriggerAwareExecutor(
        broker=broker,
        order_queue=plan_consumer,
        risk_state=risk,
        positions=positions,
        get_ltp_ts=ltp_cache.get_ltp_ts,
        bar_builder=bar_subscriber,
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

    # --- Start BarSubscriber (begin receiving ticks from MDS) ---
    bar_subscriber.start()
    logger.info("EXEC_CHILD | BarSubscriber started — receiving ticks from MDS")

    # --- Start executor threads ---
    threads: list[threading.Thread] = []

    t_trade = threading.Thread(target=trader.run_forever, name="TriggerAwareExecutor", daemon=True)
    t_trade.start()
    threads.append(t_trade)
    logger.info("EXEC_CHILD | TriggerAwareExecutor started")

    t_exit = threading.Thread(target=exit_exec.run_forever, name="ExitExecutor", daemon=True)
    t_exit.start()
    threads.append(t_exit)
    logger.info("EXEC_CHILD | ExitExecutor started")

    def monitor_triggers():
        while not stop_event.is_set():
            try:
                summary = trader.get_pending_trades_summary()
                if summary["total_pending"] > 0 or summary["total_triggered"] > 0:
                    logger.info(
                        "TRIGGER_STATUS: pending=%d triggered=%d symbols=%s",
                        summary['total_pending'],
                        summary['total_triggered'],
                        list(summary['by_symbol'].keys()),
                    )
                stop_event.wait(timeout=30)
            except Exception as e:
                logger.exception("EXEC_CHILD | Monitor thread error: %s", e)
                stop_event.wait(timeout=30)

    t_monitor = threading.Thread(target=monitor_triggers, name="TriggerMonitor", daemon=True)
    t_monitor.start()
    threads.append(t_monitor)

    # --- Heartbeat publisher ---
    hb_cfg = cfg["exec_heartbeat"]
    heartbeat = ExecHeartbeatPublisher(
        redis_url=redis_url,
        key=hb_cfg["key"],
        interval_sec=hb_cfg["interval_sec"],
        ttl_sec=hb_cfg["ttl_sec"],
    )
    heartbeat.start()

    api.set_state(SessionState.TRADING)
    logger.info("EXEC_CHILD | All components started — ready for trade plans from Redis")

    # --- HTTP shutdown callback ---
    local_stop = threading.Event()

    def _shutdown_via_http():
        logger.info("EXEC_CHILD | Shutdown requested via HTTP")
        local_stop.set()

    api.set_shutdown_callback(_shutdown_via_http)

    # --- Block until parent signals stop, local stop, or Ctrl+C ---
    try:
        while not stop_event.is_set() and not local_stop.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("EXEC_CHILD | KeyboardInterrupt — shutting down")
    finally:
        api.set_state(SessionState.SHUTTING_DOWN)

        # Cancel pending trades
        try:
            with trader._lock:
                for trade in trader.pending_trades.values():
                    if trade.state == TradeState.WAITING_TRIGGER:
                        trade.state = TradeState.CANCELLED
            logger.info("EXEC_CHILD | Cancelled all pending trades for EOD")
        except Exception as e:
            logger.warning("EXEC_CHILD | Failed to cancel pending trades: %s", e)

        # EOD sweep
        try:
            exit_exec.square_off_all_open_positions()
        except Exception as e:
            logger.warning("EXEC_CHILD | Final EOD sweep failed: %s", e)

        try:
            trader.stop()
        except Exception as e:
            logger.warning("EXEC_CHILD | trader.stop failed: %s", e)

        # Save capital report
        try:
            log_dir = get_log_directory()
            if log_dir and log_dir.exists():
                capital_manager.save_final_report(log_dir)
        except Exception:
            pass

        # Cleanup
        heartbeat.stop()
        bar_subscriber.shutdown()
        plan_consumer.shutdown()
        api.set_state(SessionState.STOPPED)
        api.stop()

    logger.info("EXEC_CHILD | Session end")
