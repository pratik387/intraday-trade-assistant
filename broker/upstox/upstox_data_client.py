"""
Upstox Data Client — drop-in replacement for KiteClient (market data only).

Provides the same duck-typed interface as KiteClient so that ScreenerLive,
WSClient, BarBuilder, and all downstream pipeline code work without changes.

Orders still go through KiteBroker (Zerodha). This client only handles:
- Symbol/token universe (fetched live from Upstox at startup)
- WebSocket tick stream (via UpstoxTickerAdapter)
- Daily historical OHLCV (via Upstox public REST API, no auth)
- 1m historical for late-start warmup

Usage:
    sdk = UpstoxDataClient()
    sdk.make_ticker()          # Returns UpstoxTickerAdapter (KiteTicker-compatible)
    sdk.get_symbol_map()       # {"NSE:RELIANCE": 123456, ...}
    sdk.get_daily("NSE:RELIANCE", 210)  # Daily OHLCV DataFrame
"""

from __future__ import annotations

import gzip
import json
import threading
import asyncio
import time
import random
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from config.env_setup import env
from config.logging_config import get_agent_logger
from utils.time_util import _now_naive_ist

logger = get_agent_logger()

ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS_CACHE_PATH = ROOT / "cache" / "upstox_nse_instruments.json"

# Upstox public instrument file (no auth, updated daily by Upstox)
UPSTOX_NSE_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

# Upstox public historical API (no auth required)
UPSTOX_HIST_BASE = "https://api.upstox.com/v3/historical-candle"
UPSTOX_HEADERS = {"Accept": "application/json"}


@dataclass(frozen=True)
class _UpstoxInst:
    int_token: int          # deterministic int from crc32(instrument_key)
    instrument_key: str     # e.g. "NSE_EQ|INE002A01018"
    trading_symbol: str     # e.g. "RELIANCE"


class UpstoxDataClient:
    """Drop-in replacement for KiteClient — Upstox data, same interface."""

    def __init__(self) -> None:
        self.access_token = getattr(env, "UPSTOX_ACCESS_TOKEN", None) or ""

        # Internal maps (same structure as KiteClient)
        self._sym2inst: Dict[str, _UpstoxInst] = {}   # "NSE:RELIANCE" -> _UpstoxInst
        self._tok2sym: Dict[int, str] = {}             # int_token -> "NSE:RELIANCE"
        self._key2int: Dict[str, int] = {}             # instrument_key -> int_token
        self._int2key: Dict[int, str] = {}             # int_token -> instrument_key
        self._equity_instruments: List[str] = []

        # Daily history cache (same as KiteClient)
        self._daily_cache_day: Optional[str] = None
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._daily_cache_lock = threading.RLock()

        # Rate limiter for REST API (Upstox official limit: 50 RPS, 500/min, 2000/30min)
        self._rps = 50.0
        self._rl_min_dt = 1.0 / self._rps
        self._rl_last = 0.0
        self._rl_lock = threading.Lock()

        self._load_instruments()

    # ─── Instrument loading ────────────────────────────────────────────────

    def _load_instruments(self) -> None:
        """
        Fetch NSE instruments from Upstox public API and build internal maps.

        Primary: GET NSE.json.gz from Upstox (no auth, updated daily).
        Fallback: cached local file from last successful fetch.
        """
        instruments = self._fetch_instruments_from_api()
        if not instruments:
            instruments = self._load_instruments_from_cache()
        if not instruments:
            raise RuntimeError(
                "UpstoxDataClient: failed to load instruments from Upstox API "
                "and no local cache available."
            )

        self._build_maps(instruments)

    def _fetch_instruments_from_api(self) -> Optional[List[dict]]:
        """Fetch NSE instruments from Upstox public CDN (gzipped JSON array)."""
        try:
            resp = requests.get(UPSTOX_NSE_INSTRUMENTS_URL, timeout=30)
            resp.raise_for_status()
            instruments = json.loads(gzip.decompress(resp.content))
            logger.info(
                f"UpstoxDataClient: fetched {len(instruments)} NSE instruments from Upstox API"
            )
            # Cache locally for fallback
            self._save_instruments_cache(instruments)
            return instruments
        except Exception as e:
            logger.warning(f"UpstoxDataClient: API fetch failed: {e}")
            return None

    def _load_instruments_from_cache(self) -> Optional[List[dict]]:
        """Load instruments from local cache file (fallback)."""
        if not INSTRUMENTS_CACHE_PATH.exists():
            return None
        try:
            with open(INSTRUMENTS_CACHE_PATH, "r", encoding="utf-8") as f:
                instruments = json.load(f)
            logger.info(
                f"UpstoxDataClient: loaded {len(instruments)} instruments from cache "
                f"({INSTRUMENTS_CACHE_PATH.name})"
            )
            return instruments
        except Exception as e:
            logger.warning(f"UpstoxDataClient: cache load failed: {e}")
            return None

    def _save_instruments_cache(self, instruments: List[dict]) -> None:
        """Save fetched instruments to local cache for fallback."""
        try:
            INSTRUMENTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Only cache NSE_EQ to keep file small (~300KB vs 50MB)
            nse_eq = [i for i in instruments if i.get("segment") == "NSE_EQ"]
            with open(INSTRUMENTS_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(nse_eq, f)
            logger.debug(f"UpstoxDataClient: cached {len(nse_eq)} NSE_EQ instruments to disk")
        except Exception as e:
            logger.warning(f"UpstoxDataClient: cache save failed: {e}")

    def _build_maps(self, instruments: List[dict]) -> None:
        """Build symbol/token maps from instrument list (JSON array)."""
        self._sym2inst.clear()
        self._tok2sym.clear()
        self._key2int.clear()
        self._int2key.clear()
        self._equity_instruments.clear()

        count = 0
        collisions = 0

        for item in instruments:
            # Filter: NSE_EQ only, EQ instrument type
            segment = item.get("segment", "")
            itype = item.get("instrument_type", "")
            tsym = item.get("trading_symbol", "")
            ikey = item.get("instrument_key", "")

            if segment != "NSE_EQ" or itype != "EQ":
                continue
            if not tsym or not ikey:
                continue

            # ETF / bond / T-bill filter (same logic as KiteClient)
            tsym_upper = tsym.upper()
            name = item.get("name", "").strip().lower()
            if (
                tsym_upper.endswith("ETF")
                or ("etf" in name.split() if name else False)
                or tsym_upper.startswith(tuple(str(i) for i in range(10)))
                or any(x in tsym_upper for x in ["SDL", "NCD", "GSEC", "GOI", "GS", "T-BILL"])
                or "-" in tsym
            ):
                continue

            # Deterministic int token from instrument_key
            int_token = zlib.crc32(ikey.encode()) & 0xFFFFFFFF

            # Handle CRC32 collisions (extremely rare but possible)
            if int_token in self._tok2sym:
                collisions += 1
                int_token = (int_token + collisions) & 0xFFFFFFFF

            sym_key = f"NSE:{tsym}"
            inst = _UpstoxInst(
                int_token=int_token,
                instrument_key=ikey,
                trading_symbol=tsym,
            )

            self._sym2inst[sym_key] = inst
            self._tok2sym[int_token] = sym_key
            self._key2int[ikey] = int_token
            self._int2key[int_token] = ikey
            self._equity_instruments.append(sym_key)
            count += 1

        if collisions:
            logger.warning(f"UpstoxDataClient: {collisions} CRC32 token collisions resolved")

        logger.info(f"UpstoxDataClient: {count} NSE EQ instruments ready")

    # ─── WebSocket ticker ──────────────────────────────────────────────────

    def make_ticker(self):
        """Return UpstoxTickerAdapter (KiteTicker-compatible interface)."""
        from broker.upstox.upstox_ticker_adapter import UpstoxTickerAdapter
        return UpstoxTickerAdapter(
            access_token=self.access_token,
            key_to_int=dict(self._key2int),
            int_to_key=dict(self._int2key),
            tok_to_sym=dict(self._tok2sym),
        )

    # ─── Symbol/token universe (same interface as KiteClient) ──────────────

    def list_equities(self) -> List[str]:
        """All NSE EQ symbols as ['NSE:RELIANCE', ...]."""
        return list(self._equity_instruments)

    def get_symbol_map(self) -> Dict[str, int]:
        """'NSE:RELIANCE' -> int_token."""
        return {s: inst.int_token for s, inst in self._sym2inst.items()}

    def get_token_map(self) -> Dict[int, str]:
        """int_token -> 'NSE:RELIANCE'."""
        return dict(self._tok2sym)

    def resolve_tokens(self, symbols: Iterable[str]) -> List[int]:
        """Map ['NSE:RELIANCE', ...] -> [int_token, ...]; unknowns skipped."""
        out: List[int] = []
        miss = 0
        for s in symbols:
            inst = self._sym2inst.get(str(s).upper())
            if inst:
                out.append(inst.int_token)
            else:
                miss += 1
        if miss:
            logger.debug("UpstoxDataClient.resolve_tokens: %d symbols not found", miss)
        return out

    # ─── Funds (not applicable — data-only client) ─────────────────────────

    def get_funds(self) -> dict:
        """Not applicable for data-only client. Returns empty."""
        return {
            "available_cash": 0,
            "available_margin": 0,
            "used_margin": 0,
            "net": 0,
            "error": "UpstoxDataClient is data-only; funds via KiteBroker"
        }

    # ─── Rate limiter ──────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        """Thread-safe RPS limiter for historical API calls."""
        with self._rl_lock:
            now = time.monotonic()
            dt = now - self._rl_last
            if dt < self._rl_min_dt:
                time.sleep(self._rl_min_dt - dt)
            self._rl_last = time.monotonic()

    def set_hist_rate_limit(self, rps: float) -> None:
        """Tune historical RPS dynamically."""
        self._rps = max(0.5, float(rps))
        self._rl_min_dt = 1.0 / self._rps

    # ─── Daily historical data ─────────────────────────────────────────────

    def _reset_daily_cache_if_new_day(self) -> None:
        today = _now_naive_ist().date().isoformat()
        with self._daily_cache_lock:
            if self._daily_cache_day != today:
                self._daily_cache.clear()
                self._daily_cache_day = today

    def get_daily_cache(self) -> Dict[str, pd.DataFrame]:
        """Get current daily cache (for persistence layer)."""
        with self._daily_cache_lock:
            return dict(self._daily_cache)

    def set_daily_cache(self, cache: Dict[str, pd.DataFrame]) -> None:
        """Inject daily cache from external source (Redis/disk)."""
        today = _now_naive_ist().date().isoformat()
        with self._daily_cache_lock:
            self._daily_cache = cache
            self._daily_cache_day = today
        logger.info(f"DAILY_CACHE | Injected {len(cache)} symbols from external source")

    def _instrument_key_for(self, symbol: str) -> str:
        """Resolve NSE:SYMBOL to Upstox instrument_key."""
        inst = self._sym2inst.get(symbol.upper())
        if not inst:
            raise KeyError(f"UpstoxDataClient: unknown symbol {symbol}")
        return inst.instrument_key

    def _fetch_daily_candles(self, instrument_key: str, days: int) -> pd.DataFrame:
        """
        Fetch daily candles from Upstox public REST API (no auth).
        Returns tz-naive daily DataFrame with columns: [open, high, low, close, volume].
        """
        now_ist = _now_naive_ist()
        end_date = now_ist.date()
        start_date = (now_ist - timedelta(days=max(days + 10, 30))).date()

        url = (
            f"{UPSTOX_HIST_BASE}/"
            f"{instrument_key}/days/1/{end_date.isoformat()}/{start_date.isoformat()}"
        )

        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        for attempt in range(3):
            try:
                self._rate_limit()
                resp = requests.get(url, headers=UPSTOX_HEADERS, timeout=15)

                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"UPSTOX_DAILY | Rate limited, waiting {wait}s")
                    time.sleep(wait)
                    continue

                if resp.status_code == 400:
                    logger.warning(f"UPSTOX_DAILY | 400 for {instrument_key}")
                    return empty

                resp.raise_for_status()
                candles = resp.json().get("data", {}).get("candles", [])
                if not candles:
                    return empty

                df = pd.DataFrame(candles)
                df.columns = ["date", "open", "high", "low", "close", "volume", "_"]
                df = df.drop(columns=["_"])
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

                # Strip timezone to naive IST
                if getattr(df["date"].dt, "tz", None) is not None:
                    df["date"] = df["date"].dt.tz_localize(None)

                df["date"] = df["date"].dt.normalize()
                df = df.sort_values("date").reset_index(drop=True)
                df = df.set_index("date")
                df = df[["open", "high", "low", "close", "volume"]].astype(float)

                # Drop today's partial bar if present
                if not df.empty and df.index[-1].date() == _now_naive_ist().date():
                    df = df.iloc[:-1]

                return df.tail(days).copy()

            except Exception as e:
                if attempt == 2:
                    logger.exception("UPSTOX_DAILY | Failed for %s: %s", instrument_key, e)
                    return empty
                time.sleep(0.5 + 0.4 * attempt + random.random() * 0.2)

        return empty

    def get_daily(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Return last `days` daily bars for `symbol` (tz-naive).
        Cached per day; at most one API call per symbol per session.
        """
        self._reset_daily_cache_if_new_day()
        with self._daily_cache_lock:
            cached = self._daily_cache.get(symbol)
            if cached is not None and len(cached) >= min(days, len(cached)):
                return cached.tail(days).copy()

        ikey = self._instrument_key_for(symbol)
        df = self._fetch_daily_candles(ikey, days)
        with self._daily_cache_lock:
            self._daily_cache[symbol] = df
        return df

    def get_prevday_levels(self, symbol: str) -> Dict[str, float]:
        """Previous trading day's high/low/close: {'PDH', 'PDL', 'PDC'}."""
        df = self.get_daily(symbol, days=2)
        if df is None or df.empty:
            return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}
        row = df.iloc[-1]
        return {"PDH": float(row.high), "PDL": float(row.low), "PDC": float(row.close)}

    def prewarm_daily_cache(self, symbols: List[str] = None, days: int = 210) -> dict:
        """
        Pre-warm daily data cache by fetching from Upstox REST API.
        Same interface as KiteClient.prewarm_daily_cache().
        """
        import time as time_module

        if symbols is None:
            symbols = self._equity_instruments

        total = len(symbols)

        # Check if cache already populated
        if len(self._daily_cache) >= total * 0.9:
            logger.info(f"PREWARM_DAILY | Cache already has {len(self._daily_cache)} symbols, skipping API fetch")
            return {
                "success": len(self._daily_cache),
                "failed": 0,
                "total": total,
                "elapsed_seconds": 0.0,
                "source": "cache"
            }

        logger.info(f"PREWARM_DAILY | Starting pre-warm for {total} symbols ({days} days each)")
        logger.info(f"PREWARM_DAILY | Estimated time: {total / self._rps / 60:.1f} minutes at {self._rps} RPS")

        start_time = time_module.perf_counter()
        success_count = 0
        fail_count = 0

        for i, symbol in enumerate(symbols):
            try:
                df = self.get_daily(symbol, days=days)
                if df is not None and not df.empty:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                if fail_count <= 5:
                    logger.warning(f"PREWARM_DAILY | Failed for {symbol}: {e}")

            if (i + 1) % 500 == 0:
                elapsed = time_module.perf_counter() - start_time
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"PREWARM_DAILY | Progress: {i+1}/{total} ({success_count} ok, {fail_count} fail) | "
                    f"Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s"
                )

        elapsed = time_module.perf_counter() - start_time
        logger.info(
            f"PREWARM_DAILY | Complete: {success_count}/{total} symbols cached | "
            f"Failed: {fail_count} | Time: {elapsed:.1f}s ({elapsed/60:.1f} min)"
        )

        return {
            "success": success_count,
            "failed": fail_count,
            "total": total,
            "elapsed_seconds": elapsed,
            "source": "api"
        }

    # ─── Intraday 1m historical (for late-start warmup) ────────────────────

    def get_historical_1m(
        self, symbol: str, from_dt: datetime, to_dt: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical 1-minute OHLCV from Upstox.
        Used for late-start warmup when server starts after 09:30.

        Returns DataFrame with [open, high, low, close, volume], index=datetime.
        Returns None if no data or API error.
        """
        ikey = self._instrument_key_for(symbol)

        # Upstox 1m API requires chunking into 28-day windows
        # For intraday warmup this is typically same-day, so single call
        from_str = from_dt.strftime("%Y-%m-%d")
        to_str = to_dt.strftime("%Y-%m-%d")

        url = (
            f"{UPSTOX_HIST_BASE}/"
            f"{ikey}/minutes/1/{to_str}/{from_str}"
        )

        for attempt in range(3):
            try:
                self._rate_limit()
                resp = requests.get(url, headers=UPSTOX_HEADERS, timeout=15)

                if resp.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                if resp.status_code == 400:
                    logger.warning(f"UPSTOX_1M | 400 for {symbol}")
                    return None

                resp.raise_for_status()
                candles = resp.json().get("data", {}).get("candles", [])
                if not candles:
                    return None

                df = pd.DataFrame(candles)
                df.columns = ["date", "open", "high", "low", "close", "volume", "_"]
                df = df.drop(columns=["_"])
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

                # Strip timezone to naive IST
                if getattr(df["date"].dt, "tz", None) is not None:
                    df["date"] = df["date"].dt.tz_localize(None)

                df = df.sort_values("date").reset_index(drop=True)
                df = df.set_index("date")
                df = df[["open", "high", "low", "close", "volume"]].astype(float)

                # Filter to requested time range
                df = df[(df.index >= from_dt) & (df.index <= to_dt)]

                logger.info(f"UPSTOX_1M | Fetched {len(df)} rows for {symbol}")
                return df

            except Exception as e:
                if attempt == 2:
                    logger.exception("UPSTOX_1M | Failed for %s: %s", symbol, e)
                    return None
                time.sleep(0.5 + 0.4 * attempt + random.random() * 0.2)

        return None

    # ─── Intraday 5m bars (same pipeline as Historical API) ──────────────

    def get_intraday_5m(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch today's completed 5-minute OHLCV bars from Upstox V3 intraday endpoint.

        Uses the same pipeline as the Historical API — guarantees backtest-live parity
        for structure detection and planning. No auth required.

        Returns DataFrame with [open, high, low, close, volume], index=datetime (IST-naive).
        Returns None if no data or API error.
        """
        try:
            ikey = self._instrument_key_for(symbol)
        except KeyError:
            return None
        url = f"{UPSTOX_HIST_BASE}/intraday/{ikey}/minutes/5"

        for attempt in range(3):
            try:
                self._rate_limit()
                resp = requests.get(url, headers=UPSTOX_HEADERS, timeout=10)

                if resp.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                if resp.status_code == 400:
                    return None

                resp.raise_for_status()
                candles = resp.json().get("data", {}).get("candles", [])
                if not candles:
                    return None

                df = pd.DataFrame(candles)
                df.columns = ["date", "open", "high", "low", "close", "volume", "_"]
                df = df.drop(columns=["_"])
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

                # Strip timezone to naive IST
                if getattr(df["date"].dt, "tz", None) is not None:
                    df["date"] = df["date"].dt.tz_localize(None)

                df = df.sort_values("date").reset_index(drop=True)
                df = df.set_index("date")
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
                return df

            except Exception as e:
                if attempt == 2:
                    logger.debug("UPSTOX_5M_INTRADAY | Failed for %s: %s", symbol, e)
                    return None
                time.sleep(0.5 + 0.4 * attempt + random.random() * 0.2)

        return None

    async def async_fetch_intraday_5m_batch(
        self, symbols: list, concurrency: int = 50, rps: float = 45.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch today's 5m intraday bars for multiple symbols concurrently.

        Uses aiohttp with async rate limiting to stay within Upstox's 50 RPS
        limit while maximizing throughput via concurrent HTTP connections.
        1000 symbols in ~22s (vs 107s sequential).

        Args:
            symbols: List of "NSE:SYMBOL" strings
            concurrency: Max concurrent HTTP connections (default 50)
            rps: Requests per second limit (default 45, headroom below 50)

        Returns:
            Dict mapping symbol -> DataFrame [open, high, low, close, volume].
            Symbols that fail or have no data are silently omitted.
        """
        import aiohttp
        from aiolimiter import AsyncLimiter

        # Pre-resolve instrument keys (sync, fast — uses in-memory dict)
        sym_to_url = {}
        skipped = 0
        for sym in symbols:
            try:
                ikey = self._instrument_key_for(sym)
                sym_to_url[sym] = f"{UPSTOX_HIST_BASE}/intraday/{ikey}/minutes/5"
            except KeyError:
                skipped += 1

        if skipped > 0:
            logger.debug("ASYNC_5M | Skipped %d unknown symbols", skipped)
        if not sym_to_url:
            return {}

        limiter = AsyncLimiter(rps, 1.0)
        sem = asyncio.Semaphore(concurrency)
        results: Dict[str, pd.DataFrame] = {}
        retries_429 = 0

        async def _fetch_one(session: aiohttp.ClientSession, sym: str, url: str):
            nonlocal retries_429
            for attempt in range(3):
                try:
                    async with sem:
                        async with limiter:
                            async with session.get(
                                url, timeout=aiohttp.ClientTimeout(total=10)
                            ) as resp:
                                if resp.status == 429:
                                    retries_429 += 1
                                    await asyncio.sleep(2 ** (attempt + 1))
                                    continue
                                if resp.status == 400:
                                    return
                                resp.raise_for_status()
                                body = await resp.json()
                except Exception:
                    if attempt == 2:
                        return
                    await asyncio.sleep(0.5 + 0.4 * attempt)
                    continue

                candles = body.get("data", {}).get("candles", [])
                if not candles:
                    return

                df = pd.DataFrame(
                    candles,
                    columns=["date", "open", "high", "low", "close", "volume", "_"],
                )
                df = df.drop(columns=["_"])
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                if getattr(df["date"].dt, "tz", None) is not None:
                    df["date"] = df["date"].dt.tz_localize(None)
                df = df.sort_values("date").set_index("date")
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
                results[sym] = df
                return

        connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300)
        async with aiohttp.ClientSession(
            connector=connector, headers=UPSTOX_HEADERS
        ) as session:
            tasks = [_fetch_one(session, sym, url) for sym, url in sym_to_url.items()]
            await asyncio.gather(*tasks, return_exceptions=True)

        if retries_429 > 0:
            logger.warning(
                "ASYNC_5M | %d 429-throttle retries during batch of %d symbols",
                retries_429, len(sym_to_url),
            )

        return results
