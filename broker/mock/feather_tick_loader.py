import os, glob
from typing import List, Optional, Dict
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.logging_config import get_agent_logger

logger = get_agent_logger()

class FeatherTickLoader:
    """
    Load cached OHLCV without relying on dates in the filename.

    For each 'EXCH:TSYM':
      • Look under base_path/<TSYM>.NS/*.feather, else base_path/<TSYM>/*.feather
      • Prefer files containing '_5m_' in name; else any *.feather
      • Pick the newest file by mtime
      • Keep 'date' column (no renaming), parse to naive datetimes (IST wall time preserved)
      • Sort by 'date' and optionally slice rows by from_date/to_date (inclusive)
    """

    def __init__(
        self,
        *,
        from_date: Optional[str],
        to_date: Optional[str],
        symbols: List[str],
        base_path: str = "cache/ohlcv_archive",
    ):
        self.from_raw = from_date
        self.to_raw = to_date
        self.symbols = symbols
        self.base_path = base_path

    # ---------- FS ----------
    def _sym_dirs(self, symbol: str) -> List[str]:
        tsym = symbol.split(":", 1)[-1].strip().upper()
        return [
            os.path.join(self.base_path, f"{tsym}.NS"),
            os.path.join(self.base_path, tsym),
        ]

    def _pick_feather(self, dirs: List[str]) -> Optional[str]:
        cands: List[str] = []
        for d in dirs:
            if os.path.isdir(d):
                cands += glob.glob(os.path.join(d, "*.feather"))
        if not cands:
            return None
        five = [p for p in cands if "_5m_" in os.path.basename(p).lower()]
        pool = five or cands
        pool.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return pool[0]

    # ---------- DF ----------
    def _parse_date_naive(self, s: pd.Series) -> pd.Series:
        """Parse to datetime; if tz-aware, drop tz while preserving IST wall time."""
        dt = pd.to_datetime(s, errors="coerce")
        # If tz-aware, coerce to naive IST wall time: convert to Asia/Kolkata, then drop tz.
        try:
            if getattr(dt.dt, "tz", None) is not None:
                dt = dt.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        except Exception:
            # If tz_convert fails (already localized differently), try a bare drop (best effort)
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                pass
        return dt

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" not in df.columns:
            raise ValueError("FeatherTickLoader: expected 'date' in cached file")
        df = df.copy()
        df["date"] = self._parse_date_naive(df["date"])
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    def _slice(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.from_raw is None and self.to_raw is None:
            return df
        lo = pd.to_datetime(self.from_raw) if self.from_raw else None
        hi = pd.to_datetime(self.to_raw) if self.to_raw else None
        # inputs are naive; df['date'] is naive → direct compare
        if lo is None: lo = df["date"].iloc[0]
        if hi is None: hi = df["date"].iloc[-1]
        return df[(df["date"] >= lo) & (df["date"] <= hi)]

    # ---------- Public ----------
    def _load_one(self, sym: str):
        try:
            path = self._pick_feather(self._sym_dirs(sym))
            if not path:
                return None
            df = pd.read_feather(path)
            df = self._normalize(df)   # parses 'date' (naive), sorts
            df = self._slice(df)       # trims to from_date..to_date inclusive
            if df.empty:
                return None
            # logger.info(f"FeatherTickLoader: loaded {len(df)} rows for {sym} from {path}")
            return sym, df
        except Exception as e:
            # keep it quiet; the main loop will count successes
            return None

    def load_all(self) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        symbols = list(self.symbols)

        # Good default for one SSD/NVMe. If you’ve got a very fast disk, 12–16 is OK.
        MAX_WORKERS = 8

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(self._load_one, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    sym, df = res
                    out[sym] = df

        # optional: a single summary log
        try:
            logger.info("FeatherTickLoader: loaded %d/%d symbols (window %s..%s)",
                        len(out), len(symbols), self.from_raw, self.to_raw)
        except Exception:
            pass

        return out
