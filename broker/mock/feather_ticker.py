import time, threading
from typing import Callable, Optional, Dict, List, Iterable
import pandas as pd
from config.logging_config import get_agent_logger

logger = get_agent_logger()

class FeatherTicker:
    """
    Mimics Zerodha KiteTicker API for the mock:
      - on_ticks(cb): cb(ws, ticks: List[dict])
      - on_connect(cb), on_close(cb)
      - connect(threaded=True), close()
      - subscribe(tokens), unsubscribe(tokens), set_mode(mode, tokens)

    Replays cached OHLCV (with 'date' column) from FeatherTickLoader.
    Emits ticks: {instrument_token, last_price, last_quantity, timestamp}.
    """

    def __init__(
        self,
        *,
        loader,
        tok2sym: Dict[int, str],
        sym2tok: Dict[str, int],
        replay_sleep: float = 0.01,
        use_close_as_price: bool = True,
    ):
        self.loader = loader
        self._tok2sym = dict(tok2sym)
        self._sym2tok = dict(sym2tok)
        self._sleep = float(replay_sleep)
        self._use_close = bool(use_close_as_price)

        self._on_ticks_cb: Optional[Callable] = None
        self._on_connect_cb: Optional[Callable] = None
        self._on_close_cb: Optional[Callable] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._subs: set[int] = set()
        self._mode: Dict[int, str] = {}

        self._data: Optional[Dict[str, pd.DataFrame]] = None
        self._timeline: Optional[List[pd.Timestamp]] = None

    # ---------------- Kite-like API ----------------
    def on_ticks(self, fn: Callable):   self._on_ticks_cb = fn
    def on_connect(self, fn: Callable): self._on_connect_cb = fn
    def on_close(self, fn: Callable):   self._on_close_cb = fn

    def subscribe(self, tokens: List[int]):   self._subs.update(int(t) for t in tokens or [])
    def unsubscribe(self, tokens: List[int]): [self._subs.discard(int(t)) for t in tokens or []]
    def set_mode(self, mode: str, tokens: List[int]):
        m = (mode or "").lower()
        for t in tokens or []:
            self._mode[int(t)] = m

    def connect(self, threaded: bool = True):
        if self._thread and self._thread.is_alive():
            logger.warning("FeatherTicker.connect called twice; ignoring"); return
        self._running = True
        if threaded:
            self._thread = threading.Thread(target=self._run, name="FeatherTicker", daemon=True)
            self._thread.start()
        else:
            self._run()

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ---------------- Internals ----------------
    def _build_timeline(self):
        assert self._data is not None and len(self._data) > 0
        # Use the first symbol's minute grid (all equities share the same 1-min grid)
        first_sym = next(iter(self._data.keys()))
        df0 = self._data[first_sym]
        if "date" not in df0.columns and not isinstance(df0.index, pd.DatetimeIndex):
            raise ValueError("FeatherTicker: expected 'date' column or DatetimeIndex")
        # We will index each DF by 'date' for fast lookup per minute.
        for sym, df in self._data.items():
            if "date" in df.columns:
                self._data[sym] = df.set_index("date", drop=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"FeatherTicker: {sym} has no time index/column")
        self._timeline = sorted(self._data[first_sym].index.unique().tolist())

    def _run(self):
        try:
            if self._data is None:
                logger.info("FeatherTicker: Calling loader.load_all() from ticker thread...")
                self._data = self.loader.load_all()  # { "NSE:XYZ": df([...,'date',...]) }
                if not self._data:
                    logger.warning("FeatherTicker: no data loaded"); return
                logger.info(f"FeatherTicker: Loaded {len(self._data)} symbols, building timeline...")
                self._build_timeline()
                logger.info("FeatherTicker: Timeline built, calling on_connect callback...")

            if callable(self._on_connect_cb):
                try:
                    import sys
                    logger.info("FeatherTicker: About to call on_connect callback...")
                    sys.stdout.flush()  # Force flush logs
                    self._on_connect_cb(self)
                    logger.info("FeatherTicker: on_connect callback returned successfully")
                    sys.stdout.flush()
                except Exception:
                    logger.exception("FeatherTicker.on_connect failed")
                    sys.stdout.flush()

            for ts in self._timeline or []:
                if not self._running: break

                batch: List[dict] = []
                tokens: Iterable[int] = (self._subs or self._sym2tok.values())

                for tok in tokens:
                    sym = self._tok2sym.get(int(tok))
                    if not sym: continue
                    df = self._data.get(sym)
                    if df is None or df.empty: continue
                    # O(1) lookup since we indexed by 'date'
                    try:
                        r = df.loc[ts]
                    except KeyError:
                        continue
                    # r may be a Series (single row) or DataFrame (dup minute); take first
                    if not isinstance(r, pd.Series):
                        r = r.iloc[0]
                    price = float(r["close"]) if self._use_close else float(r.get("last_price", r["close"]))
                    qty = int(r.get("volume", 0))

                    batch.append({
                        "instrument_token": int(tok),
                        "last_price": price,
                        "last_quantity": qty,
                        "timestamp": ts,   # naive datetime
                        "ohlc": {
                            "open": float(r.get("open", price)),
                            "high": float(r.get("high", price)),
                            "low": float(r.get("low", price)),
                            "close": price
                        }
                    })

                if batch and callable(self._on_ticks_cb):
                    try: self._on_ticks_cb(self, batch)
                    except Exception: logger.exception("FeatherTicker.on_ticks failed")

                time.sleep(self._sleep)

        except Exception:
            logger.exception("FeatherTicker loop crashed")
        finally:
            if callable(self._on_close_cb):
                try: self._on_close_cb(self, 1000, "closed")
                except Exception: logger.exception("FeatherTicker.on_close failed")
