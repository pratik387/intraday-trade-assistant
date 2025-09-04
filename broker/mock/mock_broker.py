import os
import json
from typing import List, Dict, Iterable, Optional, Union

from broker.mock.feather_tick_loader import FeatherTickLoader
from broker.mock.feather_ticker import FeatherTicker
from config.logging_config import get_loggers
from pathlib import Path
from datetime import datetime, date
import pandas as pd
from threading import RLock

logger, _ = get_loggers()
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = ROOT / "cache" / "ohlcv_archive"

class _Inst:
    def __init__(self, token: int, exch: str, tsym: str, instrument_type: str):
        self.token = token
        self.exch = exch
        self.tsym = tsym
        self.instrument_type = instrument_type


class MockBroker:
    def __init__(self, path_json: str = "nse_all.json", from_date: str = None, to_date: str = None):
        self._sym2inst: Dict[str, _Inst] = {}
        self._tok2sym: Dict[int, str] = {}
        self._equity_instruments: List[str] = []
        self._from_date = from_date
        self._to_date = to_date

        # Daily cache (tz-naive), namespaced by session date
        self._daily_cache_day: Optional[str] = None      # e.g. "YYYY-MM-DD" (session key)
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._daily_lock = RLock()
        self._dry_session_date: Optional[date] = None    # slice strictly < this date

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

    # -------------------- ticker --------------------
    def make_ticker(self) -> FeatherTicker:
        if not self._from_date or not self._to_date:
            raise ValueError("MockBroker: from_date and to_date must be set")

        loader = FeatherTickLoader(
            from_date=self._from_date,
            to_date=self._to_date,
            symbols=self._equity_instruments,
        )

        tok2sym = self.get_token_map()
        sym2tok = {v: k for k, v in tok2sym.items()}

        return FeatherTicker(
            loader=loader,
            tok2sym=tok2sym,
            sym2tok=sym2tok,
            replay_sleep=0.01,
            use_close_as_price=True,
        )

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
        key = (self._dry_session_date or datetime.now().date()).isoformat()
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
        files = sorted(dirp.glob(f"{key}_days_*.feather"), key=lambda p: p.stat().st_mtime)
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
        cutoff = self._dry_session_date or datetime.now().date()
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
    
    def get_ltp(self, symbol: str, **kwargs) -> float:
        """DRY-RUN: caller must pass last price as `ltp`."""
        v = kwargs.get("ltp")
        if v is None:
            raise TypeError("MockBroker.get_ltp requires 'ltp' kwarg in dry-run")
        return float(v)
