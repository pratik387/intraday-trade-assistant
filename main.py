from __future__ import annotations
"""
Orchestrator — the ONLY entrypoint.
Keeps Screener focused on screening. This file owns:
  - Config + logging bootstrap
  - WebSocket adapter (KiteClient) and Broker (KiteBroker)
  - OrderQueue with rate limits
  - TradeExecutor & ExitExecutor threads
  - Position store & RiskState
  - Clean shutdown

Strict contracts:
  - Symbols are always "EXCH:TRADINGSYMBOL" (e.g., "NSE:RELIANCE")
  - Intraday only: product MIS, variety regular, validity DAY
  - No hidden defaults: load_filters() must provide required keys

Optional: set DRY_RUN=1 in env to log orders without placing.
"""
import signal
import sys
import threading
import time
from typing import Optional
import argparse

from config.logging_config import get_loggers
from config.filters_setup import load_filters

from services.orders.order_queue import OrderQueue
from services.screener_live import ScreenerLive
from services.execution.trade_executor import TradeExecutor, RiskState, Position

from services.execution.exit_executor import ExitExecutor

# Strict adapters
from broker.kite.kite_client import KiteClient  # WS only
from broker.kite.kite_broker import KiteBroker  # REST orders/LTP
from broker.mock.mock_broker import MockBroker

logger, _ = get_loggers()


class _PositionStore:
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


class _DryRunBroker:
    """Wrapper that logs orders instead of placing them when DRY_RUN=1."""
    def __init__(self, real):
        self._real = real
    def place_order(self, **kwargs):
        logger.info(f"[DRY_RUN] place_order skipped: {kwargs}")
        return "dryrun-order-id"
    def get_ltp(self, symbol: str, **kwargs) -> float:
        return self._real.get_ltp(symbol, **kwargs)
    def get_ltp_batch(self, symbols):
        return self._real.get_ltp_batch(symbols)


def main() -> int:
    # ---- bootstrap ----
    cfg = load_filters()  # HARD fail if keys missing

    # Order queue: OrderQueue reads its own config via load_filters() (no kwargs accepted)
    oq = OrderQueue()

    if args.dry_run:
        if not args.session_date:
            print("error: --dry-run requires --session-date YYYY-MM-DD", file=sys.stderr)
            sys.exit(2)

        from_date = _hhmm_on(args.session_date, args.from_hhmm)
        to_date   = _hhmm_on(args.session_date, args.to_hhmm)

        sdk = MockBroker(path_json="nse_all.json", from_date=from_date, to_date=to_date)
        sdk.set_session_date(args.session_date)  # PDH/PDL/PDC = day before session_date
        broker = _DryRunBroker(sdk)
        logger.warning("DRY RUN: %s %s-%s", args.session_date, args.from_hhmm, args.to_hhmm)
    else:
        sdk = KiteClient()
        broker = KiteBroker()
        
    # Screener: screens only
    screener = ScreenerLive(sdk=sdk, order_queue=oq)

    # Risk + positions
    risk = RiskState(max_concurrent=int(cfg["max_concurrent_positions"]))
    positions = _PositionStore()

    # Executors    
    trader = TradeExecutor(
        broker=broker,
        order_queue=oq,
        risk_state=risk,
        positions=positions,
        )

    exit_exec = None
    if ExitExecutor is not None:
        for ctor in (
            lambda: ExitExecutor(),
            lambda: ExitExecutor(broker=broker, order_queue=oq),
            lambda: ExitExecutor(broker=broker, order_queue=oq, get_df5m=getattr(screener, "_bars5", {}).get),
        ):
            try:
                exit_exec = ctor()
                logger.info("ExitExecutor constructed")
                break
            except TypeError:
                continue
            except Exception as e:
                logger.warning(f"ExitExecutor ctor failed: {e}")

    threads: list[threading.Thread] = []

    t_trade = threading.Thread(target=trader.run_forever, name="TradeExecutor", daemon=True)
    t_trade.start(); threads.append(t_trade)
    logger.info("trade-executor: started")

    if exit_exec is not None and hasattr(exit_exec, "run_forever"):
        t_exit = threading.Thread(target=exit_exec.run_forever, name="ExitExecutor", daemon=True)
        t_exit.start(); threads.append(t_exit)
        logger.info("exit-executor: started")
    else:
        logger.info("exit-executor: not started")
        
    # ---- lifecycle ----
    stop = threading.Event()
    def _sig_handler(signum, frame):
        logger.warning(f"signal {signum} received — shutting down")
        stop.set()
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    run_until_eod(screener)

    return 0

def run_until_eod(screener: ScreenerLive, poll_sec: float = 0.2) -> None:
    """
    Start the screener and block until it requests exit (EOD),
    then stop it cleanly. Works for both live and dry runs.
    """
    logger.info("session start")
    try:
        screener.start()
        while not getattr(screener, "_request_exit", False):
            time.sleep(poll_sec)
    except KeyboardInterrupt:
        logger.info("keyboard interrupt — stopping")
    finally:
        try:
            screener.stop()
        except Exception as e:
            logger.warning("screener.stop failed: %s", e)
    logger.info("session end (EOD)")
    
def _hhmm_on(day_str: str, hhmm: str) -> str:
    y, m, d = map(int, day_str.split("-"))
    h, mm = map(int, hhmm.split(":"))
    return f"{y:04d}-{m:02d}-{d:02d} {h:02d}:{mm:02d}:00"

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Use mock broker + archived ticks")
    ap.add_argument("--session-date", help="YYYY-MM-DD (required with --dry-run)")
    ap.add_argument("--from-hhmm", default="09:00")
    ap.add_argument("--to-hhmm",   default="16:00")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main())
