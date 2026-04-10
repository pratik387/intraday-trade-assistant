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

        # Scan callback: fires at 5m boundaries with enriched bars dict
        # Signature: on_5m_scan(api_df5_cache: Dict[str, DataFrame])
        self.on_5m_scan: Optional[Callable] = None

        # Legacy: enriched 5m replay callback (kept for backward compat)
        self.on_5m_enriched: Optional[Callable] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._subs: set[int] = set()
        self._mode: Dict[int, str] = {}

        self._data: Optional[Dict[str, pd.DataFrame]] = None
        self._timeline: Optional[List[pd.Timestamp]] = None

        # Native 5m bars from disk: {symbol: DataFrame[open,high,low,close,volume]}
        # Loaded from _5minutes.feather files (same data as paper's API fetch)
        self._native_5m: Dict[str, pd.DataFrame] = {}

        # Legacy precomputed enriched 5m bars (kept for backward compat)
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
    def load_native_5m(self, cache_dir: str = "cache/ohlcv_archive") -> None:
        """Load native 5m bars from disk for all symbols.
        These are the same bars the paper API fetch returns."""
        from pathlib import Path
        import time as _t

        _t0 = _t.perf_counter()
        cache = Path(cache_dir)
        for sym, tok in self._sym2tok.items():
            tsym = sym.split(":", 1)[-1].strip().upper() if ":" in sym else sym
            for suffix in [f"{tsym}.NS", tsym]:
                path = cache / suffix / f"{suffix}_5minutes.feather"
                if path.exists():
                    try:
                        df = pd.read_feather(path)
                        df["date"] = pd.to_datetime(df["date"])
                        if getattr(df["date"].dt, "tz", None) is not None:
                            df["date"] = df["date"].dt.tz_localize(None)
                        df = df.set_index("date").sort_index()
                        df = df[["open", "high", "low", "close", "volume"]].astype(float)
                        self._native_5m[sym] = df
                    except Exception:
                        pass
                    break
        logger.info("NATIVE_5M_CACHE | Loaded %d symbols (%.1fs)",
                   len(self._native_5m), _t.perf_counter() - _t0)

    def _fire_5m_scan(self, bar_start: pd.Timestamp) -> None:
        """At 5m boundary, load native 5m bars up to this point, enrich, fire scan callback.
        This is the backtest equivalent of paper's API fetch + enrich."""
        cb = self.on_5m_scan
        if not callable(cb):
            return

        from services.indicators.bar_enrichment import enrich_5m_bars
        warmup_bars = 30
        scan_ts = bar_start
        today = scan_ts.normalize()

        api_df5_cache: Dict[str, pd.DataFrame] = {}
        for sym, df_all in self._native_5m.items():
            df_up_to = df_all[df_all.index <= scan_ts]
            today_bars = df_up_to[df_up_to.index >= today]

            if len(today_bars) < 1:
                continue

            prev_bars = df_up_to[df_up_to.index < today]
            if len(prev_bars) >= warmup_bars:
                warmup = prev_bars.tail(warmup_bars)
                combined = pd.concat([warmup, today_bars])
                enriched = enrich_5m_bars(combined, session_date=today)
            else:
                enriched = enrich_5m_bars(today_bars)

            if not enriched.empty:
                api_df5_cache[sym] = enriched

        try:
            cb(api_df5_cache)
        except Exception:
            logger.exception("FeatherTicker._fire_5m_scan failed at %s", bar_start)

    def _fire_enriched_5m(self, bar_start: pd.Timestamp) -> None:
        """Legacy: Fire on_5m_enriched callback for backward compat."""
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

                # Fire scan at 5m boundaries
                # At ts=09:20 (minute % 5 == 0), fire scan for bar 09:15
                ts_minute = ts.minute if hasattr(ts, 'minute') else ts.to_pydatetime().minute
                if ts_minute % 5 == 0:
                    from datetime import timedelta as _td
                    bar_start = ts.floor("5min") - _td(minutes=5)

                    # New path: native 5m from disk → enrich → scan callback
                    if self._native_5m and callable(self.on_5m_scan):
                        self._fire_5m_scan(bar_start)
                    # Legacy path: precomputed enriched 5m replay
                    elif self._enriched_5m and callable(self.on_5m_enriched):
                        self._fire_enriched_5m(bar_start)

                time.sleep(self._sleep)

        except Exception:
            logger.exception("FeatherTicker loop crashed")
        finally:
            if callable(self._on_close_cb):
                try: self._on_close_cb(self, 1000, "closed")
                except Exception: logger.exception("FeatherTicker.on_close failed")
