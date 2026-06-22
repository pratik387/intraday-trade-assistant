# broker/mock/mock_broker.py
from __future__ import annotations
import os
import json
from typing import Any, List, Dict, Iterable, Optional, Union, Callable
from pathlib import Path
from datetime import datetime, date
import threading

import pandas as pd

from broker.mock.feather_tick_loader import FeatherTickLoader
from broker.mock.feather_ticker import FeatherTicker
from config.logging_config import get_agent_logger

logger = get_agent_logger()
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = ROOT / "cache" / "ohlcv_archive"


class _Inst:
    def __init__(self, token: int, exch: str, tsym: str, instrument_type: str):
        self.token = token
        self.exch = exch
        self.tsym = tsym
        self.instrument_type = instrument_type


class MockBroker:
    """
    Dry-run broker that:
      • Replays 1m 'ticks' via FeatherTicker
      • Maintains a thread-safe last-price cache (symbol -> LTP) from the replay
      • Serves get_ltp(symbol) from that cache (or from explicit kwarg 'ltp' for backward-compat)

    Note: No changes required in FeatherTicker/FeatherTickLoader.
    """

    def __init__(self, path_json: str = "nse_all.json", from_date: str = None, to_date: str = None,
                 slippage_bps: float = None, data_sdk: Optional[Any] = None):
        if slippage_bps is None:
            raise ValueError("MockBroker requires slippage_bps from config (fees_slippage_bps)")
        self._slippage_frac = slippage_bps / 10_000  # Convert bps to fraction
        self._sym2inst: Dict[str, _Inst] = {}
        self._tok2sym: Dict[int, str] = {}
        self._equity_instruments: List[str] = []
        self._from_date = from_date
        self._to_date = to_date

        # Paper-trading data source: when set, get_daily / get_intraday_5m
        # delegate to this client (e.g. UpstoxDataClient) for LIVE data
        # instead of returning stale archive bars. Orders stay simulated
        # on MockBroker. Leave as None for backtest mode (archive lookup).
        self._data_sdk = data_sdk

        # Optional prefetched {symbol: today's-5m DataFrame} populated once per
        # EOD run (run_eod batch-fetches the universe via the data SDK's async
        # 5m batch, mirroring overnight_handlers). get_intraday_5m checks this
        # FIRST, so fetch_daily_window's per-symbol synthesis is a cache hit
        # instead of ~1 live API call per symbol over the whole MTF universe.
        self._intraday_5m_prefetch: Dict[str, Any] = {}

        # Paper-order sequence — feeds place_order's returned order_id.
        # Long-running paper-trade uses KiteBroker(dry_run=True); the
        # overnight cron uses MockBroker (it's a short-lived process with
        # data_sdk wired for live data), so MockBroker carries its own
        # tiny paper-order simulator for the close_dn_overnight_long flow.
        self._paper_order_seq: int = 0
        self._paper_order_lock = threading.RLock()

        # Daily cache (tz-naive), namespaced by session date
        self._daily_cache_day: Optional[str] = None      # e.g. "YYYY-MM-DD" (session key)
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._daily_lock = threading.RLock()
        self._dry_session_date: Optional[date] = None    # slice strictly < this date

        # --- live LTP cache for replay ---
        self._lp_lock = threading.RLock()
        self._last_price: Dict[str, float] = {}          # updated by ticker proxy
        self._last_bar_ohlc: Dict[str, dict] = {}        # OHLC data for intrabar checks

        # Paper mode: when a live data SDK is wired, skip the nse_all.json
        # load entirely. The instrument/token maps are dead weight there —
        # every consumer (screener_live, main.py) reads sdk.get_symbol_map()
        # which goes to the SDK (UpstoxDataClient) directly, never to the
        # broker. Backtest mode still loads the JSON because MockBroker IS
        # the SDK there.
        if self._data_sdk is not None:
            logger.info(
                "MockBroker: data_sdk wired — skipping nse_all.json instrument "
                "load (instrument lookups delegate to the SDK in paper mode)"
            )
        else:
            self._load_instruments(path_json)

    # -------------------- instruments --------------------
    def _load_instruments(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"MockBroker: JSON file not found: {path}")

        with open(path, "r") as f:
            raw = json.load(f)

        for i in raw:
            tsym = i.get("symbol")
            token = i.get("instrument_token")
            if not tsym or not token:
                continue

            exch = "NSE"
            itype = "EQ"

            inst = _Inst(token=token, exch=exch, tsym=tsym.replace(".NS", ""), instrument_type=itype)

            sym_key = f"{exch}:{inst.tsym}"
            self._sym2inst[sym_key] = inst
            self._tok2sym[token] = sym_key

        self._equity_instruments = list(self._sym2inst.keys())
        logger.info(f"MockBroker: loaded {len(self._equity_instruments)} instruments from {path}")

    def list_equities(self) -> List[str]:
        return self._equity_instruments

    def get_symbol_map(self) -> Dict[str, int]:
        # Paper mode: tokens come from the live SDK (Upstox/Kite). Backtest
        # mode: tokens come from nse_all.json (loaded at init).
        if self._data_sdk is not None and hasattr(self._data_sdk, "get_symbol_map"):
            return self._data_sdk.get_symbol_map()
        return {s: i.token for s, i in self._sym2inst.items()}

    def get_token_map(self) -> Dict[int, str]:
        if self._data_sdk is not None and hasattr(self._data_sdk, "get_token_map"):
            return self._data_sdk.get_token_map()
        return dict(self._tok2sym)

    def list_symbols(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[str]:
        # Paper-mode: prefer the live data SDK's symbol list. It's current
        # (refreshed on every startup from the Upstox instrument API) vs
        # nse_all.json which can be weeks/months stale. Falls back to the
        # nse_all-derived list for backtest mode (no data_sdk).
        if self._data_sdk is not None and exchange == "NSE" and instrument_type == "EQ":
            live_syms = getattr(self._data_sdk, "_equity_instruments", None)
            if live_syms:
                return list(live_syms)
        return [
            f"{i.exch}:{i.tsym}"
            for i in self._sym2inst.values()
            if i.exch == exchange and i.instrument_type == instrument_type
        ]

    def list_tokens(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[int]:
        return [
            i.token
            for i in self._sym2inst.values()
            if i.exch == exchange and i.instrument_type == instrument_type
        ]

    def resolve_tokens(self, symbols: Iterable[str]) -> List[int]:
        # Paper mode: delegate to the live SDK so tokens reflect the current
        # broker-side instrument set, not a 7-week-stale snapshot.
        if self._data_sdk is not None and hasattr(self._data_sdk, "resolve_tokens"):
            return self._data_sdk.resolve_tokens(symbols)
        out: List[int] = []
        miss = 0
        for s in symbols:
            inst = self._sym2inst.get(str(s).upper())
            if inst:
                out.append(inst.token)
            else:
                miss += 1
        if miss:
            logger.warning(f"MockBroker.resolve_tokens: {miss} symbols not found")
        return out

    # -------------------- paper-mode orders --------------------
    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        product: str = "MIS",
        variety: str = "regular",
        validity: str = "DAY",
        tag: Optional[str] = None,
        trade_id: Optional[str] = None,
        check_margins: bool = False,
    ) -> str:
        """Simulated paper-mode order placement.

        Returns a sequential paper order_id; does no margin check, no
        broker call, no persistence. The overnight cron uses this to mint
        order_ids for close_dn_overnight_long MOC BUY + AMO SELL. Actual
        fill price is determined later by _paper_fill_price_entry /
        _paper_fill_price_exit reading the data_sdk's intraday/historical
        bars.
        """
        side_u = (side or "").upper()
        if side_u not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY/SELL, got {side!r}")
        if int(qty) <= 0:
            raise ValueError(f"qty must be > 0, got {qty}")
        with self._paper_order_lock:
            self._paper_order_seq += 1
            seq = self._paper_order_seq
        order_id = f"PAPER_OVNT_{seq:08d}"
        logger.info(
            "[PAPER] MockBroker order: %s %s %d %s (product=%s variety=%s) -> %s",
            side_u, symbol, int(qty), order_type, product, variety, order_id,
        )
        return order_id

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Paper-mode order status: always COMPLETE.

        The overnight handler's _live_poll_fill polls this for the fill
        price; in paper mode we don't track fills here (the handler uses
        _paper_fill_price_entry to derive the price from data instead).
        Returning COMPLETE with average_price=0.0 short-circuits the
        polling loop without claiming a specific fill price.
        """
        return {"order_id": str(order_id), "status": "COMPLETE", "average_price": 0.0}

    def place_gtt_stop(self, *, symbol: str, qty: int, trigger_price: float,
                       limit_price: float, product: str = "MTF") -> str:
        """Paper-mode GTT stop: mint a fake trigger id, place nothing.

        Lets the overnight place-exit cron exercise the full AMO+GTT flow in
        paper (the GTT is a live-only catastrophe failsafe; in paper the
        position still exits via the simulated next-open AMO fill). Without
        this, run_place_exit would log a gtt_failed for every paper position.
        """
        with self._paper_order_lock:
            self._paper_order_seq += 1
            gid = f"PAPER_GTT_{self._paper_order_seq:08d}"
        logger.info(
            "[PAPER] MockBroker GTT stop: %s qty=%d trig=%s lim=%s (product=%s) -> %s",
            symbol, int(qty), trigger_price, limit_price, product, gid,
        )
        return gid

    def cancel_gtt(self, gtt_id: str) -> bool:
        """Paper-mode GTT cancel: always succeeds (nothing was really placed)."""
        logger.info("[PAPER] MockBroker cancel GTT: %s", gtt_id)
        return True

    def set_intraday_5m_prefetch(self, mapping: Dict[str, Any]) -> None:
        """Stash a batch-fetched {symbol: today's-5m DataFrame} map (keys may be
        'NSE:SYM' or bare). get_intraday_5m serves from here before the live API."""
        self._intraday_5m_prefetch = mapping or {}

    def _prefetched_5m(self, symbol: str) -> Optional[pd.DataFrame]:
        pf = self._intraday_5m_prefetch
        if not pf:
            return None
        bare = str(symbol).replace("NSE:", "").upper()
        for key in (symbol, bare, f"NSE:{bare}"):
            hit = pf.get(key)
            if hit is not None:
                return hit
        return None

    # -------------------- ticker (with last-price proxy) --------------------
    def get_intraday_5m(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch today's 5m bars: batch-warmed prefetch map first, then the wired
        data SDK (paper-mode only).

        Returns None when no SDK is configured — callers should treat that
        as "fall through to archive/backtest path".
        """
        hit = self._prefetched_5m(symbol)
        if hit is not None:
            return hit
        if self._data_sdk is None or not hasattr(self._data_sdk, "get_intraday_5m"):
            return None
        try:
            return self._data_sdk.get_intraday_5m(symbol)
        except Exception as e:
            logger.warning("MockBroker.get_intraday_5m via data_sdk failed for %s: %s", symbol, e)
            return None

    def _load_enriched_5m(self) -> Dict[str, pd.DataFrame]:
        """Load precomputed enriched 5m bars from monthly cache or individual files."""
        from pathlib import Path
        import time as _t

        t0 = _t.perf_counter()
        result = {}

        from_dt = pd.to_datetime(self._from_date)
        to_dt = pd.to_datetime(self._to_date)

        # Try monthly pre-aggregated file first (fast path)
        preagg_dir = Path("backtest-cache-download/monthly")
        if from_dt.year == to_dt.year and from_dt.month == to_dt.month:
            monthly_file = preagg_dir / f"{from_dt.year}_{from_dt.month:02d}_5m_enriched.feather"
            if monthly_file.exists():
                try:
                    df_all = pd.read_feather(monthly_file)
                    df_all["date"] = pd.to_datetime(df_all["date"])
                    if df_all["date"].dt.tz is not None:
                        df_all["date"] = df_all["date"].dt.tz_localize(None)
                    df_all = df_all[(df_all["date"] >= from_dt) & (df_all["date"] <= to_dt)]

                    for sym_raw, group in df_all.groupby("symbol"):
                        sym = f"NSE:{sym_raw}"
                        df_sym = group.drop(columns=["symbol"]).set_index("date").sort_index()
                        result[sym] = df_sym

                    elapsed = _t.perf_counter() - t0
                    logger.info("ENRICHED_5M | Loaded %d symbols from %s (%.1fs)",
                               len(result), monthly_file.name, elapsed)
                    return result
                except Exception as e:
                    logger.warning("ENRICHED_5M | Monthly file load failed: %s", e)

        # Fallback: individual symbol files
        cache_dir = Path("cache/ohlcv_archive")
        for sym in self._equity_instruments:
            tsym = sym.split(":", 1)[-1].strip().upper()
            for suffix in [f"{tsym}.NS", tsym]:
                path = cache_dir / suffix / f"{suffix}_5minutes_enriched.feather"
                if path.exists():
                    try:
                        df = pd.read_feather(path)
                        df["date"] = pd.to_datetime(df["date"])
                        if df["date"].dt.tz is not None:
                            df["date"] = df["date"].dt.tz_localize(None)
                        df = df[(df["date"] >= from_dt) & (df["date"] <= to_dt)]
                        df = df.set_index("date").sort_index()
                        result[sym] = df
                    except Exception:
                        pass
                    break

        elapsed = _t.perf_counter() - t0
        logger.info("ENRICHED_5M | Loaded %d symbols from individual files (%.1fs)", len(result), elapsed)
        return result

    def make_ticker(self):
        """
        Returns a FeatherTicker proxy that updates the broker's last-price cache
        and forwards ticks/connect/close to the client's callbacks.
        """
        if not self._from_date or not self._to_date:
            raise ValueError("MockBroker: from_date and to_date must be set")

        loader = FeatherTickLoader(
            from_date=self._from_date,
            to_date=self._to_date,
            symbols=self._equity_instruments,
        )

        tok2sym = self.get_token_map()
        sym2tok = {v: k for k, v in tok2sym.items()}

        # Load precomputed enriched 5m bars for direct replay (skip LiveTickHandler aggregation)
        enriched_5m = self._load_enriched_5m()

        inner = FeatherTicker(
            loader=loader,
            tok2sym=tok2sym,
            sym2tok=sym2tok,
            replay_sleep=0.01,
            use_close_as_price=True,
            enriched_5m=enriched_5m,
        )

        broker_self = self  # capture for closures

        class _ProxyTicker:
            def __init__(self, inner_ticker: FeatherTicker):
                self._inner = inner_ticker

                # KiteTicker-style: Attributes for callback assignment
                # Usage: ticker.on_ticks = my_callback (attribute assignment, not method call)
                self.on_ticks: Optional[Callable] = None
                self.on_connect: Optional[Callable] = None
                self.on_close: Optional[Callable] = None

                # Bind our wrappers to inner so we always see ticks/connect/close
                self._inner.on_ticks(self._mux_on_ticks)
                self._inner.on_connect(self._mux_on_connect)
                self._inner.on_close(self._mux_on_close)

            # --- mux wrappers ---
            def _mux_on_connect(self, ws):
                try:
                    if callable(self.on_connect):
                        self.on_connect(ws, None)  # response=None for MockBroker
                except Exception:
                    logger.exception("ProxyTicker.on_connect (client) failed")

            def _mux_on_close(self, ws, code, reason):
                try:
                    if callable(self.on_close):
                        self.on_close(ws, code, reason)
                except Exception:
                    logger.exception("ProxyTicker.on_close (client) failed")

            def _mux_on_ticks(self, ws, ticks: List[dict]):
                # Update broker's last-price cache and OHLC data
                try:
                    with broker_self._lp_lock:
                        for t in ticks or []:
                            tok = int(t.get("instrument_token"))
                            sym = tok2sym.get(tok)
                            if not sym:
                                continue
                            lp = t.get("last_price")
                            if lp is None:
                                continue
                            broker_self._last_price[sym] = float(lp)

                            # Cache OHLC data for intrabar checks
                            ohlc = t.get("ohlc")
                            if ohlc and isinstance(ohlc, dict):
                                broker_self._last_bar_ohlc[sym] = {
                                    "open": float(ohlc.get("open", lp)),
                                    "high": float(ohlc.get("high", lp)),
                                    "low": float(ohlc.get("low", lp)),
                                    "close": float(ohlc.get("close", lp))
                                }
                except Exception:
                    logger.exception("ProxyTicker: failed updating last price/OHLC cache")

                # Forward to the client's callback unchanged
                try:
                    if callable(self.on_ticks):
                        self.on_ticks(ws, ticks)
                except Exception:
                    logger.exception("ProxyTicker.on_ticks (client) failed")

            # --- I1 candle callback (forwarded to FeatherTicker) ---
            @property
            def on_i1_candle(self):
                return self._inner.on_i1_candle

            @on_i1_candle.setter
            def on_i1_candle(self, cb):
                self._inner.on_i1_candle = cb

            # --- Enriched 5m bar callback (forwarded to FeatherTicker) ---
            @property
            def on_5m_enriched(self):
                return self._inner.on_5m_enriched

            @on_5m_enriched.setter
            def on_5m_enriched(self, cb):
                self._inner.on_5m_enriched = cb

            # --- pass-through API ---
            def subscribe(self, tokens):  self._inner.subscribe(tokens)
            def unsubscribe(self, tokens): self._inner.unsubscribe(tokens)
            def set_mode(self, mode, tokens): self._inner.set_mode(mode, tokens)
            def connect(self, threaded=True): self._inner.connect(threaded=threaded)
            def close(self): self._inner.close()

        return _ProxyTicker(inner)

    # -------------------- daily OHLCV (tz-naive) --------------------
    def _archive_key(self, symbol: str) -> str:
        """'NSE:BATAINDIA' -> 'BATAINDIA.NS'; pass through '*.NS'/'*.BSE'."""
        s = symbol.upper()
        if s.endswith(".NS") or s.endswith(".BSE"):
            return s
        if ":" in s:
            exch, sym = s.split(":", 1)
            if exch == "NSE":
                return f"{sym}.NS"
            if exch == "BSE":
                return f"{sym}.BSE"
            return sym
        return s

    def _load_from_consolidated_daily(self, symbol: str, consolidated_path: Path) -> pd.DataFrame:
        """
        Load daily data for a symbol from consolidated daily cache.

        This is used in OCI mode where individual symbol files don't exist.
        The consolidated cache contains all symbols' daily data in one file.
        """
        # Load consolidated cache (cached at class level to avoid repeated reads)
        if not hasattr(self, '_consolidated_daily_cache'):
            logger.info(f"Loading consolidated daily cache from {consolidated_path}")
            try:
                self._consolidated_daily_cache = pd.read_feather(consolidated_path)
                logger.info(f"Loaded consolidated daily cache: {len(self._consolidated_daily_cache):,} rows, {self._consolidated_daily_cache['symbol'].nunique()} symbols")
            except Exception as e:
                logger.error(f"Failed to load consolidated daily cache: {e}")
                self._consolidated_daily_cache = pd.DataFrame()
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Extract symbol data - normalize symbol to plain format (RELIANCE, not NSE:RELIANCE)
        df_all = self._consolidated_daily_cache
        # Consolidated cache stores plain symbols, so strip prefix/suffix
        plain_symbol = symbol.upper()
        if ":" in plain_symbol:
            plain_symbol = plain_symbol.split(":", 1)[1]  # NSE:RELIANCE -> RELIANCE
        if plain_symbol.endswith(".NS"):
            plain_symbol = plain_symbol[:-3]  # RELIANCE.NS -> RELIANCE
        elif plain_symbol.endswith(".BSE"):
            plain_symbol = plain_symbol[:-4]  # RELIANCE.BSE -> RELIANCE

        symbol_data = df_all[df_all['symbol'] == plain_symbol].copy()

        if symbol_data.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Normalize timestamp and filter before session date
        ts = pd.to_datetime(symbol_data['ts'])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)

        symbol_data['date'] = ts

        # Filter to before session date (if set)
        if self._dry_session_date:
            symbol_data = symbol_data[ts < pd.Timestamp(self._dry_session_date)]

        # Sort and prepare final columns
        symbol_data = symbol_data.sort_values('date')
        symbol_data = symbol_data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Use date as the index — matches the fallback (individual-files) path
        # and the contract assumed by detectors that read df_daily.index
        # (e.g. circuit_t1_fade_short reads idx.date for T-1 lookup).
        symbol_data['date'] = symbol_data['date'].dt.normalize()
        symbol_data = symbol_data.set_index('date')

        return symbol_data

    def _reset_daily_cache_if_new_day(self) -> None:
        # Cache namespace tied to the session date (or today if unset).
        # Without the None-guard, callers that forget to set_session_date()
        # (e.g. the overnight cron handler) crash with AttributeError, which
        # then gets silently swallowed by upstream `except Exception` blocks
        # and leaves the universe empty without explanation. Default to
        # today so the cache namespace stays valid.
        from datetime import date as _date
        sd = self._dry_session_date if self._dry_session_date is not None else _date.today()
        key = sd.isoformat()
        with self._daily_lock:
            if self._daily_cache_day != key:
                self._daily_cache.clear()
                self._daily_cache_day = key

    def set_session_date(self, d: Union[str, date, datetime, None]) -> None:
        """
        Set the 'as-of' session date for dry runs.
        All daily data returned will be strictly BEFORE this date.
        Pass None to revert to today().
        """
        if d is None:
            sd = None
        elif isinstance(d, date) and not isinstance(d, datetime):
            sd = d
        elif isinstance(d, datetime):
            sd = d.date()
        else:
            sd = datetime.fromisoformat(str(d)).date()
        with self._daily_lock:
            self._dry_session_date = sd
            self._daily_cache_day = None  # force new cache namespace

    def get_daily(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Return last `days` completed daily bars (tz-naive) for `symbol`.

        Strategy:
        0. If a live data SDK is wired (paper mode), use it FIRST — the
           local archive is for backtests, not for paper-trading current
           sessions.
        1. Check cache
        2. Try consolidated daily cache (OCI mode)
        3. Fall back to individual files (local mode)
        """
        # Paper-mode live data path
        if self._data_sdk is not None:
            try:
                df = self._data_sdk.get_daily(symbol, days=days)
                if df is not None and not df.empty:
                    with self._daily_lock:
                        self._daily_cache[symbol] = df
                    return df.tail(int(days)).copy()
            except Exception as e:
                logger.warning(
                    "MockBroker.get_daily via data_sdk failed for %s: %s; "
                    "falling back to archive (likely empty for current session)",
                    symbol, e,
                )

        self._reset_daily_cache_if_new_day()
        with self._daily_lock:
            cached = self._daily_cache.get(symbol)
            if cached is not None and len(cached) >= min(days, len(cached)):
                return cached.tail(int(days)).copy()

        # Try consolidated daily cache first (for OCI)
        consolidated_path = OUTPUT_BASE.parent / "preaggregate" / "consolidated_daily.feather"
        if consolidated_path.exists():
            try:
                df = self._load_from_consolidated_daily(symbol, consolidated_path)
                if not df.empty:
                    # Cache and return
                    with self._daily_lock:
                        self._daily_cache[symbol] = df
                    return df.tail(int(days)).copy()
            except Exception as e:
                logger.warning(f"get_daily: consolidated cache failed for {symbol}, falling back to individual: {e}")

        # Fall back to individual files (local mode)
        key = self._archive_key(symbol)
        dirp = OUTPUT_BASE / key
        if not dirp.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        files = sorted(dirp.glob(f"{key}_1days.feather"), key=lambda p: p.stat().st_mtime)
        if not files:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        fp = files[-1]

        # Read + normalize
        try:
            df = pd.read_feather(fp)
        except Exception as e:
            logger.exception("get_daily: failed to read %s: %s", fp, e)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        dates = pd.to_datetime(df["date"], errors="coerce")
        try:
            # if tz-aware (e.g., +05:30), drop tz to make naive
            if getattr(getattr(dates, "dt", None), "tz", None) is not None:
                dates = dates.dt.tz_localize(None)
        except Exception:
            pass
        df["date"] = dates.dt.normalize()
        df = df.dropna(subset=["date"]).set_index("date")

        # lower-case and keep OHLCV; fill any missing columns
        df.rename(columns=str.lower, inplace=True)
        for c in ("open", "high", "low", "close", "volume"):
            if c not in df.columns:
                df[c] = pd.NA
        df = df[["open", "high", "low", "close", "volume"]]

        # de-duplicate dates (keep last), sort
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # slice strictly before session date (so prev-day is 'yesterday' for that session)
        cutoff = self._dry_session_date
        df = df[df.index.date < cutoff]

        with self._daily_lock:
            self._daily_cache[symbol] = df

        return df.tail(int(days)).copy()

    def fetch_daily_window(self, symbols, start_date, end_date) -> pd.DataFrame:
        """Trailing adjusted daily bars for a basket, as the long panel the
        multi-day CNC/MTF ranker consumes (LiveDailyPanelProvider's fetch_fn).

        Assembles per-symbol bars via `get_daily` (live SDK in paper mode) into
        a [date, symbol, open, high, low, close, volume] frame windowed to
        [start_date, end_date] inclusive. Unknown / data-less symbols are
        skipped (never raises). The calendar span is requested as the bar count
        (an intentional over-fetch — `get_daily` tails what it returns — so the
        window is fully covered even for thinly-traded illiquid names).
        """
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        end_norm = end.normalize()
        # Request the calendar span as the BAR count (intentional over-fetch:
        # get_daily tails what it returns). +5 is a calendar-day holiday buffer for
        # the day->bar-count conversion, NOT a trading threshold.
        days = max(1, (end - start).days + 5)
        cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
        frames = []
        for sym in symbols:
            bare = str(sym).replace("NSE:", "").upper()
            # get_daily / the data SDK resolve the NSE:-PREFIXED symbol
            # (_instrument_key_for keys _sym2inst by 'NSE:SYMBOL'); a bare symbol
            # is "unknown". Pass the prefixed form; the output panel stays bare.
            df = self.get_daily(f"NSE:{bare}", days=days)
            have_end = False
            if df is not None and not df.empty:
                d = df.reset_index()
                date_col = "date" if "date" in d.columns else d.columns[0]
                d = d.rename(columns={date_col: "date"})
                d["symbol"] = bare
                missing = [c for c in ("open", "high", "low", "close", "volume") if c not in d.columns]
                if missing:
                    logger.warning("fetch_daily_window: %s missing %s; skip", bare, missing)
                    continue
                d["date"] = pd.to_datetime(d["date"])
                have_end = bool((d["date"].dt.normalize() == end_norm).any())
                frames.append(d[cols])
            # get_daily DROPS the current (partial) day's bar, but the
            # cross-sectional ranker requires a row dated exactly == the session
            # date (end). Synthesize end's daily bar from that day's 5m so live
            # entries can fire. _synth_daily_from_5m self-filters to `end`, so it
            # no-ops for any symbol whose 5m isn't on that day.
            if not have_end:
                synth = self._synth_daily_from_5m(bare, end_norm)
                if synth is not None:
                    frames.append(pd.DataFrame([synth])[cols])
        if not frames:
            return pd.DataFrame(columns=cols)
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])
        if getattr(out["date"].dt, "tz", None) is not None:
            out["date"] = out["date"].dt.tz_localize(None)
        mask = (out["date"] >= start) & (out["date"] <= end)
        return out.loc[mask].reset_index(drop=True)

    def _synth_daily_from_5m(self, bare: str, day: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """Build a single daily OHLCV bar for `day` from that day's intraday 5m
        (open=first, high=max, low=min, close=last, volume=sum). Returns None if
        no 5m for `day` is available (so the ranker simply skips that symbol).
        Used to supply the current session's bar, which get_daily drops as partial.
        """
        try:
            df5 = self.get_intraday_5m(f"NSE:{bare}")
            if df5 is None or df5.empty:
                df5 = self.get_intraday_5m(bare)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("fetch_daily_window: 5m fetch failed for %s: %s", bare, e)
            return None
        if df5 is None or df5.empty:
            return None
        idx = pd.to_datetime(df5.index)
        if getattr(idx, "tz", None) is not None:
            # Canonical IST strip (utils/time_util): convert, don't reinterpret.
            idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        sub = df5[idx.normalize() == day.normalize()]
        if sub.empty:
            return None
        return {
            "date": day.normalize(), "symbol": bare,
            "open": float(sub["open"].iloc[0]), "high": float(sub["high"].max()),
            "low": float(sub["low"].min()), "close": float(sub["close"].iloc[-1]),
            "volume": float(sub["volume"].sum()),
        }

    def get_prevday_levels(self, symbol: str) -> Dict[str, float]:
        """Previous trading day's high/low/close from stored daily df."""
        df = self.get_daily(symbol, days=2)
        if df is None or df.empty:
            return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}
        row = df.iloc[-1]
        return {"PDH": float(row.high), "PDL": float(row.low), "PDC": float(row.close)}

    def prewarm_daily_cache(self, symbols: List[str], days: int = 210) -> None:
        """
        Pre-load daily data for all symbols into cache at session start.
        Eliminates repeated disk I/O during bar processing (saves ~6s per bar).

        Call once at initialization to populate _daily_cache for all symbols.
        Subsequent get_daily() calls will return from cache instantly.
        """
        import time
        logger.info(f"MockBroker: Pre-warming daily cache for {len(symbols)} symbols ({days} days)")
        start = time.time()

        success = 0
        for i, symbol in enumerate(symbols):
            try:
                self.get_daily(symbol, days=days)  # Populates _daily_cache
                success += 1
                if (i + 1) % 500 == 0:
                    logger.info(f"  Pre-warmed {i+1}/{len(symbols)} symbols...")
            except Exception as e:
                logger.warning(f"  Failed to pre-warm {symbol}: {e}")

        elapsed = time.time() - start
        logger.info(f"MockBroker: Daily cache pre-warmed ({success}/{len(symbols)} symbols) in {elapsed:.1f}s")

    # -------------------- LTP API (now works without kwargs) --------------------
    def get_ltp(self, symbol: str, **kwargs) -> float:
        """
        Backtest-friendly LTP with multiple logic paths:
        1. If kwargs['ltp'] provided, return it (compat).
        2. If kwargs['check_level'] provided (for intrabar SL/target checks):
           - If level is within current bar's [low, high], return the level
           - Otherwise, return bar close
        3. If caller passed entry_zone + bar_1m, and the bar touched the zone,
            synthesize an in-zone price deterministically (entry logic).
        4. Else, return last cached price from replay (bar close).
        """
        # 0) explicit override for backwards-compat
        v = kwargs.get("ltp")
        if v is not None:
            return float(v)

        sym = str(symbol).upper()

        # Try cached (your replay writes close into _last_price)
        with self._lp_lock:
            cached = self._last_price.get(sym)
            bar_ohlc = self._last_bar_ohlc.get(sym)

        # 1) Intrabar check_level logic (for SL/target exits)
        check_level = kwargs.get("check_level")
        if check_level is not None and bar_ohlc:
            level = float(check_level)
            low = float(bar_ohlc["low"])
            high = float(bar_ohlc["high"])

            # If level within [low, high], return it; else return close
            if low <= level <= high:
                return level
            # Level not touched in this bar, return close
            return float(bar_ohlc["close"])

        # 2) Entry zone logic (for entries)
        entry_zone = kwargs.get("entry_zone")
        bar = kwargs.get("bar_1m")
        if entry_zone and isinstance(entry_zone, (list, tuple)) and len(entry_zone) == 2 and isinstance(bar, dict):
            lo, hi = sorted(map(float, entry_zone))

            # Pull OHLC(VWAP optional) from provided bar dict
            open_  = float(bar.get("open",  cached or 0.0))
            high   = float(bar.get("high",  cached or 0.0))
            low    = float(bar.get("low",   cached or 0.0))
            close  = float(bar.get("close", cached or 0.0))
            # vwap = float(bar.get("vwap", close))  # not used in hardcoded path

            touched = (high >= lo) and (low <= hi)
            if touched:
                # Hardcoded deterministic fill:
                if lo <= open_ <= hi:
                    fill = min(max(open_, lo), hi)  # open clamped to zone
                else:
                    # direction by bar body: up -> use zone low, down -> zone high
                    fill = lo if close >= open_ else hi

                # Side-aware slippage from config (fees_slippage_bps)
                side = (kwargs.get("side") or "").upper()
                slip = self._slippage_frac
                if side == "BUY":
                    fill *= (1.0 + slip)
                elif side == "SELL":
                    fill *= (1.0 - slip)
                else:
                    # default slight penalty if side unknown
                    fill *= (1.0 + slip)

                # Clamp to zone after slippage to remain conservative
                return float(min(max(fill, lo), hi))

        # 3) Fallback to cached replay price (typically bar close)
        if cached is not None:
            return float(cached)

        raise TypeError("MockBroker.get_ltp: no cached LTP and no bar/zone kwargs provided")

    def get_ltp_with_level(self, symbol: str, check_level: Optional[float] = None, **kwargs) -> float:
        """
        Get LTP with intrabar level checking (for exit executor).

        In backtest mode, if check_level is provided, checks if current bar's OHLC touched the level.
        If touched, returns the level; otherwise returns bar close.

        Args:
            symbol: Trading symbol
            check_level: SL/target level to check against bar OHLC

        Returns:
            Level if bar touched it, otherwise bar close
        """
        # Just call get_ltp with check_level parameter - existing logic handles it
        return self.get_ltp(symbol, check_level=check_level, **kwargs)

    def get_ltp_batch(self, symbols: Iterable[str]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        with self._lp_lock:
            for s in symbols or []:
                key = str(s).upper()
                out[key] = float(self._last_price[key]) if key in self._last_price else None
        return out
