"""IV-rank lookup service for options_vol_iv_rank_revert detector.

Reads the daily IV-rank parquet (built by tools/option_chain/build_iv_rank.py
from option_chain bhavcopies) and exposes per-(symbol, date) IV-rank lookup.

The detector calls `get_iv_rank(symbol, session_date)` once per qualifying
bar; the service caches the parquet in memory once per process.

Source: data/options_iv/<from_year>_<to_year>_iv_rank.parquet
Columns: session_date, symbol, atm_iv, iv_rank, iv_rank_252d_min,
         iv_rank_252d_max

In live mode, the daily compute job appends a row per F&O symbol every
trading day post-bhavcopy publish (~17:30 IST). In backtest, the historical
parquet covers 2023-01-02 onward.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from config.logging_config import get_agent_logger

logger = get_agent_logger()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_IV_DIR = _REPO_ROOT / "data" / "options_iv"


class IVRankService:
    """In-memory IV-rank lookup. Singleton pattern via module-level instance.

    Lazy-loads the parquet on first access; raises FileNotFoundError if the
    parquet is missing (no silent defaults — fail fast per CLAUDE.md rule 1).
    """

    def __init__(self, parquet_path: Optional[Path] = None):
        self._path = parquet_path
        self._lookup: Optional[pd.Series] = None

    def _resolve_path(self) -> Path:
        if self._path is not None:
            return self._path
        # Pick the newest *_iv_rank.parquet under data/options_iv/
        if not _DEFAULT_IV_DIR.exists():
            raise FileNotFoundError(
                f"IV-rank directory missing: {_DEFAULT_IV_DIR}. "
                f"Run tools/option_chain/build_iv_rank.py to populate."
            )
        candidates = sorted(_DEFAULT_IV_DIR.glob("*_iv_rank.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"No *_iv_rank.parquet found in {_DEFAULT_IV_DIR}. "
                f"Run tools/option_chain/build_iv_rank.py to populate."
            )
        return candidates[-1]

    def _ensure_loaded(self) -> None:
        if self._lookup is not None:
            return
        path = self._resolve_path()
        logger.info(f"IV_RANK_SERVICE: loading {path}")
        df = pd.read_parquet(path)
        df = df.dropna(subset=["iv_rank"])
        df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
        # Strip NSE: prefix from symbol if present (parquet uses bare symbols)
        df["symbol"] = df["symbol"].astype(str).str.replace("NSE:", "", regex=False)
        self._lookup = df.set_index(["symbol", "session_date"])["iv_rank"]
        logger.info(
            f"IV_RANK_SERVICE: loaded {len(self._lookup):,} rows "
            f"({self._lookup.index.get_level_values('symbol').nunique()} symbols, "
            f"{self._lookup.index.get_level_values('session_date').min()} → "
            f"{self._lookup.index.get_level_values('session_date').max()})"
        )

    def get_iv_rank(self, symbol: str, session_date: date) -> Optional[float]:
        """Return iv_rank computed on the most recent bhavcopy ON OR BEFORE
        `session_date - 1` (T-1 EOD signal applied to T+0 entry).

        Returns None if no IV-rank coverage exists for the symbol on/before
        the lookup window (e.g., new F&O listing without 252-day history).
        """
        self._ensure_loaded()
        bare = symbol.replace("NSE:", "")
        # We want the iv_rank computed at the close of T-1 (or earlier).
        # session_date is T+0; lookup the most recent (symbol, d) where
        # d < session_date.
        try:
            sym_idx = self._lookup.loc[bare]
        except KeyError:
            return None
        # sym_idx is a Series indexed by session_date
        prior = sym_idx.index[sym_idx.index < session_date]
        if len(prior) == 0:
            return None
        last = prior.max()
        return float(sym_idx.loc[last])

    def get_high_iv_rank_symbols(
        self, session_date: date, threshold: float, prefix: str = "NSE:",
    ) -> list:
        """All symbols whose most recent T-1 iv_rank >= threshold.

        Used by EnergyScanner to prepend setup-priority candidates so that
        the IV-rank detector reaches the per-bar evaluation regardless of
        the symbol's generic OHLCV momentum/MR rank. Returns symbols with
        the supplied `prefix` (default "NSE:") to match scanner conventions.
        """
        self._ensure_loaded()
        df = self._lookup.reset_index()
        df = df[df["session_date"] < session_date]
        if df.empty:
            return []
        # Most recent iv_rank per symbol on or before T-1
        latest = df.sort_values("session_date").groupby("symbol").tail(1)
        qualifying = latest[latest["iv_rank"] >= threshold]["symbol"].tolist()
        return [f"{prefix}{s}" for s in qualifying]


# Module-level singleton — first .get_iv_rank() loads the parquet.
_singleton: Optional[IVRankService] = None


def get_iv_rank_service() -> IVRankService:
    global _singleton
    if _singleton is None:
        # Allow override via env var for tests
        path_env = os.environ.get("IV_RANK_PARQUET_PATH")
        if path_env:
            _singleton = IVRankService(Path(path_env))
        else:
            _singleton = IVRankService()
    return _singleton


def reset_for_tests() -> None:
    """Reset singleton — used by unit tests that point at fixture parquets."""
    global _singleton
    _singleton = None
