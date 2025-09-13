# kite_client.py
# Zerodha Kite SDK wrapper for managing instrument metadata, token/symbol maps, and equity universe filtering.
# Efficient in-memory maps for symbol/token resolution, with optional CSV fallback.

from typing import Dict, List, Iterable, Optional
from dataclasses import dataclass
from kiteconnect import KiteConnect
from config.env_setup import env
from kiteconnect import KiteTicker
from config.logging_config import get_agent_logger
from datetime import datetime, timedelta
import threading, time, random
import pandas as pd

logger = get_agent_logger()

@dataclass(frozen=True)
class _Inst:
    token: int
    tsym: str            # e.g. 'RELIANCE'

class KiteClient:
    def __init__(self):
        self.api_key = env.KITE_API_KEY
        self.access_token = env.KITE_ACCESS_TOKEN
        if not self.api_key or not self.access_token:
            raise RuntimeError("KiteBroker: KITE_API_KEY and KITE_ACCESS_TOKEN are required")
        self._kc = KiteConnect(api_key=self.api_key)
        self._kc.set_access_token(self.access_token)
        self._sym2inst: Dict[str, _Inst] = {}
        self._tok2sym: Dict[int, str] = {}
        self._equity_instruments: List[str] = []
                # ---- Daily history: per-day in-memory cache + simple rate limiter (tz-naive) ----
        self._daily_cache_day: Optional[str] = None   # "YYYY-MM-DD"
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._daily_cache_lock = threading.RLock()

        # conservative historical RPS (~2.5 rps)
        self._rps = 2.5
        self._rl_min_dt = 1.0 / float(self._rps)
        self._rl_last = 0.0
        self._rl_lock = threading.Lock()


        self._load_instruments()
        
    def make_ticker(self):
        try:
            return KiteTicker(api_key=self.api_key, access_token=self.access_token)
        except Exception as e:
            raise RuntimeError(f"KiteClient: failed to create KiteTicker: {e}")

    def _load_instruments(self) -> None:
        """Load instruments from Kite API or CSV and build internal maps."""
        try:
            raw = self._kc.instruments()
        except Exception as e:
            raise RuntimeError(f"KiteClient: failed to load instruments: {e}")

        self._sym2inst.clear()
        self._tok2sym.clear()
        self._equity_instruments.clear()

        for i in raw:
            exch = i.get("exchange", "")
            seg = i.get("segment", "")
            itype = i.get("instrument_type", "")
            name = i.get("name", "").strip().lower()
            tsym = i.get("tradingsymbol", "")
            token = i.get("instrument_token")
            expiry = i.get("expiry")

            if not exch or not seg or not itype or not tsym or not token:
                continue

            # Safe equity filter
            if (
                exch != "NSE" or
                seg != "NSE" or
                itype != "EQ" or
                expiry or
                not name or len(name) < 3 or
                tsym.upper().startswith(tuple(str(i) for i in range(10))) or  # starts with digit
                any(x in tsym.upper() for x in ["SDL", "NCD", "GSEC", "GOI", "GS", "T-BILL"]) or
                "-" in tsym
            ):
                continue

            inst = _Inst(
                token=token,
                tsym=tsym
            )

            sym_key = f"{exch}:{tsym}"
            self._sym2inst[sym_key] = inst
            self._tok2sym[token] = sym_key
            self._equity_instruments.append(sym_key)

        logger.info(f"KiteClient: loaded {len(self._sym2inst)} total; NSE EQ = {len(self._equity_instruments)}")


    def list_equities(self) -> List[str]:
        """All NSE:EQ symbols as ['NSE:RELIANCE', ...]."""
        return self._equity_instruments

    def list_symbols(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[str]:
        """Return symbols like 'EXCH:TSYM' filtered by exchange/instrument_type."""
        e = exchange.upper()
        it = (instrument_type or "").upper()
        return [
            f"{i.exch}:{i.tsym}"
            for i in self._sym2inst.values()
            if i.exch == e and (not it or i.instrument_type == it)
        ]

    def list_tokens(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[int]:
        """Return instrument tokens filtered by exchange/instrument_type."""
        e = exchange.upper()
        it = (instrument_type or "").upper()
        return [
            i.token
            for i in self._sym2inst.values()
            if i.exch == e and (not it or i.instrument_type == it)
        ]

    def get_symbol_map(self) -> Dict[str, int]:
        return {s: i.token for s, i in self._sym2inst.items()}

    def get_token_map(self) -> Dict[int, str]:
        return dict(self._tok2sym)

    def resolve_tokens(self, symbols: Iterable[str]) -> List[int]:
        """Map ['EXCH:TSYM', ...] -> [token, ...]; unknowns skipped."""
        out: List[int] = []
        miss = 0
        for s in symbols:
            inst = self._sym2inst.get(str(s).upper())
            if inst:
                out.append(inst.token)
            else:
                miss += 1
        if miss:
            logger.debug("KiteClient.resolve_tokens: %d symbols not found", miss)
        return out
    
        # ----------------------- Daily (historical) helpers -----------------------

    def _rate_limit(self) -> None:
        """Simple thread-safe RPS limiter for historical_data calls."""
        with self._rl_lock:
            now = time.monotonic()
            dt = now - self._rl_last
            if dt < self._rl_min_dt:
                time.sleep(self._rl_min_dt - dt)
            self._rl_last = time.monotonic()

    def _reset_daily_cache_if_new_day(self) -> None:
        today = datetime.now().date().isoformat()  # tz-naive
        with self._daily_cache_lock:
            if self._daily_cache_day != today:
                self._daily_cache.clear()
                self._daily_cache_day = today

    def _token_for(self, symbol: str) -> int:
        inst = self._sym2inst.get(symbol.upper())
        if not inst:
            raise KeyError(f"KiteClient: unknown symbol {symbol}")
        return inst.token

    def _historical_daily_df(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Fetch daily candles from Kite for `symbol` and return tz-naive daily DataFrame.
        Index: naive dates (normalized), columns: ['open','high','low','close','volume'].
        Drops today's partial if present.
        """
        token = self._token_for(symbol)
        end_date = datetime.now().date()  # naive
        start_date = (datetime.now() - timedelta(days=max(days + 5, 20))).date()  # pad for weekends/holidays

        for attempt in range(3):
            try:
                self._rate_limit()
                candles = self._kc.historical_data(
                    instrument_token=token,
                    from_date=start_date.isoformat(),
                    to_date=end_date.isoformat(),
                    interval="day",
                )
                if not candles:
                    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

                df = pd.DataFrame(candles)
                # 'date' from Kite is typically tz-naive; normalize and drop any tz if present
                dates = pd.to_datetime(df["date"], errors="coerce", utc=False)
                # If somehow tz-aware slipped in, strip it to make naive
                if getattr(dates.dt, "tz", None) is not None:
                    dates = dates.dt.tz_localize(None)
                df["date"] = dates.dt.normalize()
                df.set_index("date", inplace=True)

                df = df[["open", "high", "low", "close", "volume"]].astype(float)

                # Drop today's partial bar if present
                if not df.empty and df.index[-1].date() == datetime.now().date():
                    df = df.iloc[:-1]

                return df.tail(days).copy()
            except Exception as e:
                if attempt == 2:
                    logger.exception("KiteClient._historical_daily_df failed for %s: %s", symbol, e)
                    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                time.sleep(0.5 + 0.4 * attempt + random.random() * 0.2)

        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def get_daily(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Return last `days` daily bars for `symbol` (tz-naive).
        Cached per day; at most one broker call per symbol per session.
        """
        self._reset_daily_cache_if_new_day()
        with self._daily_cache_lock:
            cached = self._daily_cache.get(symbol)
            if cached is not None and len(cached) >= min(days, len(cached)):
                return cached.tail(days).copy()

        df = self._historical_daily_df(symbol, days)
        with self._daily_cache_lock:
            self._daily_cache[symbol] = df
        return df

    def get_prevday_levels(self, symbol: str) -> Dict[str, float]:
        """
        Previous trading day's high/low/close as tz-naive floats:
        {'PDH': ..., 'PDL': ..., 'PDC': ...}
        """
        df = self.get_daily(symbol, days=2)
        if df is None or df.empty:
            return {"PDH": float("nan"), "PDL": float("nan"), "PDC": float("nan")}
        row = df.iloc[-1]  # last completed day
        return {"PDH": float(row.high), "PDL": float(row.low), "PDC": float(row.close)}

    def set_hist_rate_limit(self, rps: float) -> None:
        """Optional: tune historical RPS dynamically."""
        self._rps = max(0.5, float(rps))
        self._rl_min_dt = 1.0 / self._rps

