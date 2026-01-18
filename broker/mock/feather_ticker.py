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
        # CRITICAL FIX: Build timeline from UNION of all symbols' timestamps
        # Using only first symbol misses early data for illiquid stocks that start trading later
        # We need all timestamps to ensure ORB period (09:15-09:30) data is included
        all_timestamps = set()
        for sym, df in self._data.items():
            if "date" in df.columns:
                self._data[sym] = df.set_index("date", drop=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"FeatherTicker: {sym} has no time index/column")
            all_timestamps.update(self._data[sym].index.unique().tolist())
        self._timeline = sorted(all_timestamps)

    def _run(self):
        try:
            if self._data is None:
                self._data = self.loader.load_all()  # { "NSE:XYZ": df([...,'date',...]) }
                if not self._data:
                    logger.warning("FeatherTicker: no data loaded"); return
                self._build_timeline()

            if callable(self._on_connect_cb):
                try: self._on_connect_cb(self)
                except Exception: logger.exception("FeatherTicker.on_connect failed")

            for ts in self._timeline or []:
                if not self._running: break

                batch: List[dict] = []

                # Process symbols with tokens (equities)
                tokens: Iterable[int] = (self._subs or self._sym2tok.values())
                processed_symbols: set[str] = set()

                for tok in tokens:
                    sym = self._tok2sym.get(int(tok))
                    if not sym: continue
                    processed_symbols.add(sym)
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

                # Process symbols without tokens (e.g., index symbols like NSE:NIFTY 50)
                # These don't have instrument tokens but are loaded in self._data
                for sym, df in self._data.items():
                    if sym in processed_symbols:
                        continue  # Already processed via token
                    if df is None or df.empty:
                        continue
                    try:
                        r = df.loc[ts]
                    except KeyError:
                        continue
                    if not isinstance(r, pd.Series):
                        r = r.iloc[0]
                    price = float(r["close"]) if self._use_close else float(r.get("last_price", r["close"]))
                    qty = int(r.get("volume", 0))

                    # Use a synthetic negative token for index symbols (won't conflict with real tokens)
                    # The symbol is passed via the 'symbol' key for TickRouter to handle
                    batch.append({
                        "instrument_token": -1,  # Synthetic token for symbols without real tokens
                        "symbol": sym,           # Pass symbol directly for TickRouter
                        "last_price": price,
                        "last_quantity": qty,
                        "timestamp": ts,
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
