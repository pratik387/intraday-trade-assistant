# broker/mock/mock_broker.py
from __future__ import annotations
import os
import json
from typing import List, Dict, Iterable, Optional, Union, Callable
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

    def __init__(self, path_json: str = "nse_all.json", from_date: str = None, to_date: str = None):
        self._sym2inst: Dict[str, _Inst] = {}
        self._tok2sym: Dict[int, str] = {}
        self._equity_instruments: List[str] = []
        self._from_date = from_date
        self._to_date = to_date

        # Daily cache (tz-naive), namespaced by session date
        self._daily_cache_day: Optional[str] = None      # e.g. "YYYY-MM-DD" (session key)
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._daily_lock = threading.RLock()
        self._dry_session_date: Optional[date] = None    # slice strictly < this date

        # --- live LTP cache for replay ---
        self._lp_lock = threading.RLock()
        self._last_price: Dict[str, float] = {}          # updated by ticker proxy
        self._last_bar_ohlc: Dict[str, dict] = {}        # OHLC data for intrabar checks

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
        return {s: i.token for s, i in self._sym2inst.items()}

    def get_token_map(self) -> Dict[int, str]:
        return dict(self._tok2sym)

    def list_symbols(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[str]:
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

    # -------------------- ticker (with last-price proxy) --------------------
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

        inner = FeatherTicker(
            loader=loader,
            tok2sym=tok2sym,
            sym2tok=sym2tok,
            replay_sleep=0.01,
            use_close_as_price=True,
        )

        broker_self = self  # capture for closures

        class _ProxyTicker:
            def __init__(self, inner_ticker: FeatherTicker):
                self._inner = inner_ticker
                self._client_on_ticks: Optional[Callable] = None
                self._client_on_connect: Optional[Callable] = None
                self._client_on_close: Optional[Callable] = None

                # Bind our wrappers to inner so we always see ticks/connect/close
                self._inner.on_ticks(self._mux_on_ticks)
                self._inner.on_connect(self._mux_on_connect)
                self._inner.on_close(self._mux_on_close)

            # --- client registration (we forward to these after caching LTP) ---
            def on_ticks(self, fn: Callable):   self._client_on_ticks = fn
            def on_connect(self, fn: Callable): self._client_on_connect = fn
            def on_close(self, fn: Callable):   self._client_on_close = fn

            # --- mux wrappers ---
            def _mux_on_connect(self, ws):
                try:
                    if callable(self._client_on_connect):
                        self._client_on_connect(ws)
                except Exception:
                    logger.exception("ProxyTicker.on_connect (client) failed")

            def _mux_on_close(self, ws, code, reason):
                try:
                    if callable(self._client_on_close):
                        self._client_on_close(ws, code, reason)
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
                    if callable(self._client_on_ticks):
                        self._client_on_ticks(ws, ticks)
                except Exception:
                    logger.exception("ProxyTicker.on_ticks (client) failed")

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

    def _reset_daily_cache_if_new_day(self) -> None:
        # Cache namespace tied to the session date (or today if unset)
        key = (self._dry_session_date).isoformat()
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
        Reads the latest <KEY>_days_*.feather under OUTPUT_BASE/<KEY>/,
        normalizes columns, de-dupes dates, and slices strictly before session date.
        """
        self._reset_daily_cache_if_new_day()
        with self._daily_lock:
            cached = self._daily_cache.get(symbol)
            if cached is not None and len(cached) >= min(days, len(cached)):
                return cached.tail(int(days)).copy()

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

    def get_prevday_levels(self, symbol: str) -> Dict[str, float]:
        """Previous trading day's high/low/close from stored daily df."""
        df = self.get_daily(symbol, days=2)
        if df is None or df.empty:
            return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}
        row = df.iloc[-1]
        return {"PDH": float(row.high), "PDL": float(row.low), "PDC": float(row.close)}

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

                # Fixed 5 bps slippage; side-aware if provided
                side = (kwargs.get("side") or "").upper()
                slip = 0.0005  # 5 bps hardcoded
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


    def get_ltp_batch(self, symbols: Iterable[str]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        with self._lp_lock:
            for s in symbols or []:
                key = str(s).upper()
                out[key] = float(self._last_price[key]) if key in self._last_price else None
        return out
