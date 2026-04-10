import time, threading
from typing import Callable, Optional, Dict, List, Iterable
import pandas as pd
from config.logging_config import get_agent_logger

logger = get_agent_logger()

# IST offset in nanoseconds (5h30m) for converting IST-naive → UTC epoch ms
_IST_OFFSET_NS = (5 * 3600 + 30 * 60) * 1_000_000_000

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
        enriched_5m: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.loader = loader
        self._tok2sym = dict(tok2sym)
        self._sym2tok = dict(sym2tok)
        self._sleep = float(replay_sleep)
        self._use_close = bool(use_close_as_price)

        self._on_ticks_cb: Optional[Callable] = None
        self._on_connect_cb: Optional[Callable] = None
        self._on_close_cb: Optional[Callable] = None

        # I1 candle callback (same interface as UpstoxTickerAdapter.on_i1_candle)
        # Assigned by WSClient._wire_callbacks() if listener is registered
        self.on_i1_candle: Optional[Callable] = None

        # Direct 5m enriched bar callback — bypasses LiveTickHandler aggregation
        # When set, fires at 5m boundaries with precomputed enriched bars
        self.on_5m_enriched: Optional[Callable] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._subs: set[int] = set()
        self._mode: Dict[int, str] = {}

        self._data: Optional[Dict[str, pd.DataFrame]] = None
        self._timeline: Optional[List[pd.Timestamp]] = None

        # Precomputed enriched 5m bars: {symbol: DataFrame with DatetimeIndex}
        self._enriched_5m = enriched_5m or {}

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
    def _fire_enriched_5m(self, bar_start: pd.Timestamp) -> None:
        """Fire on_5m_enriched callback ONCE per 5m boundary to trigger the scan.
        The scan itself processes all shortlisted symbols using precomputed data."""
        cb = self.on_5m_enriched
        if not callable(cb):
            return
        for sym, df in self._enriched_5m.items():
            try:
                bar = df.loc[bar_start]
            except KeyError:
                continue
            if not isinstance(bar, pd.Series):
                bar = bar.iloc[0]
            bar.name = bar_start
            try:
                cb(sym, bar)
            except Exception:
                logger.exception("FeatherTicker.on_5m_enriched callback failed for %s", sym)
            return

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

                    o = float(r.get("open", r["close"]))
                    h = float(r.get("high", r["close"]))
                    l = float(r.get("low", r["close"]))
                    c = float(r["close"])
                    total_vol = int(r.get("volume", 0))

                    ohlc_dict = {"open": o, "high": h, "low": l, "close": c}
                    tok_int = int(tok)

                    # Emit I1 candle (broker-constructed 1m bar) for BarBuilder.on_i1_candle()
                    # Same path as live Upstox WebSocket — ensures backtest/live parity
                    if callable(self.on_i1_candle):
                        epoch_ms = str((ts.value - _IST_OFFSET_NS) // 1_000_000)
                        self.on_i1_candle(sym, o, h, l, c, total_vol, epoch_ms)

                    # Adaptive 4-tick OHLC simulation (NautilusTrader method):
                    # Emit O, H, L, C as 4 synthetic ticks so BarBuilder
                    # reconstructs proper candle shapes from stored 1m bars.
                    # Ordering: if bar is bullish (close > open), price likely
                    # dipped first → O → L → H → C.  If bearish → O → H → L → C.
                    if abs(h - l) < 1e-9:
                        # Flat bar (O=H=L=C) — single tick is enough
                        batch.append({
                            "instrument_token": tok_int,
                            "last_price": c,
                            "last_quantity": total_vol,
                            "timestamp": ts,
                            "ohlc": ohlc_dict,
                        })
                    else:
                        bullish = c >= o  # includes doji as bullish
                        price_seq = [o, l, h, c] if bullish else [o, h, l, c]

                        # Distribute volume: 25% each, remainder to close
                        vol_quarter = total_vol // 4
                        vol_remainder = total_vol - vol_quarter * 3
                        vol_seq = [vol_quarter, vol_quarter, vol_quarter, vol_remainder]

                        for price, vol in zip(price_seq, vol_seq):
                            batch.append({
                                "instrument_token": tok_int,
                                "last_price": price,
                                "last_quantity": vol,
                                "timestamp": ts,
                                "ohlc": ohlc_dict,
                            })

                if batch and callable(self._on_ticks_cb):
                    try: self._on_ticks_cb(self, batch)
                    except Exception: logger.exception("FeatherTicker.on_ticks failed")

                # Fire precomputed enriched 5m bars at 5m boundaries
                # LiveTickHandler fires _on_5m_close when minute % 5 == 0 (e.g., 09:35 closes 09:30 bar)
                # We replicate that timing: at ts=09:35, fire the 09:30 enriched bar
                if self._enriched_5m and callable(self.on_5m_enriched):
                    ts_minute = ts.minute if hasattr(ts, 'minute') else ts.to_pydatetime().minute
                    if ts_minute % 5 == 0:
                        from datetime import timedelta as _td
                        bar_start = ts.floor("5min") - _td(minutes=5)
                        self._fire_enriched_5m(bar_start)

                time.sleep(self._sleep)

        except Exception:
            logger.exception("FeatherTicker loop crashed")
        finally:
            if callable(self._on_close_cb):
                try: self._on_close_cb(self, 1000, "closed")
                except Exception: logger.exception("FeatherTicker.on_close failed")
