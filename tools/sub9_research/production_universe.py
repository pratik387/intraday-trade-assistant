"""Production-aligned universe gate for sanity scripts.

Background: existing sanity scripts apply a window-level coverage filter
(`days_per_sym >= 80% of window days`) which silently rejects newly-tracked
SME symbols that production fires on. This causes sanity PF claims to be
inflated by survivorship — computed only on the "older tracked" cohort.

This module provides a reusable `ProductionUniverseGate` that mirrors the
EXACT logic of `services/setup_universe.py` builders. Use it INSTEAD OF
window-level filters in any new sanity script.

Lessons informing this design:
  - #16 — sanity universe ≠ production universe; verify monthly-feather coverage
  - #17 — legacy intraday-MIS filters (`min_trading_days`, `min_daily_avg_volume`)
         are inappropriate for cell-locked setups; set to 0 to disable
  - #18 — universe-data-source asymmetry is the dominant root cause of
         sanity-OCI parity gap; per-bar detector logic matches identically

Usage:
    from tools.sub9_research.production_universe import ProductionUniverseGate

    gate = ProductionUniverseGate(
        accepted_caps={"unknown"},               # match cell_lock_cap_segment
        require_mis=True,                         # match production builder
        min_trading_days_required=0,              # zero post-Lesson #17
        min_daily_avg_volume=0,                   # zero post-Lesson #17
        mtf_snapshot_path=None,                   # only for MTF setups
        exclude_etf=False,                        # MTF setups: True
    )

    # Per-signal-date filter (replaces window-level coverage check):
    if not gate.is_eligible("RPPINFRA", session_date=date(2026, 3, 30)):
        continue
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Set

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DAILY_PATH = _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather"
_NSE_ALL = _REPO_ROOT / "nse_all.json"


def _normalize_symbol(s: str) -> str:
    """Mirror services.symbol_metadata._normalize_symbol: strip NSE: prefix + .NS suffix."""
    if ":" in s:
        s = s.split(":")[-1]
    if "." in s:
        s = s.split(".")[0]
    return s


@dataclass(frozen=True)
class _NseRow:
    cap_segment: str            # 'large_cap' | 'mid_cap' | 'small_cap' | 'micro_cap' | 'unknown'
    mis_enabled: bool


class ProductionUniverseGate:
    """Per-session-date universe filter matching production's setup_universe.py builders.

    Mirrors the logic of these production functions byte-for-byte:
      - services/setup_universe.below_vwap_volume_revert_long_universe
      - services/setup_universe.close_dn_overnight_long_universe
      - services/setup_universe.gap_fade_universe
      - services/setup_universe.long_panic_gap_down_universe
      - services/setup_universe.or_window_failure_fade_short_universe
      - services/setup_universe.circuit_t1_universe
      - services/setup_universe.delivery_pct_universe

    Caches consolidated_daily + nse_all on first use.
    """

    def __init__(
        self,
        *,
        accepted_caps: Set[str],
        require_mis: bool = True,
        require_mtf: bool = False,
        min_trading_days_required: int = 0,
        min_daily_avg_volume: float = 0,
        mtf_snapshot_path: Optional[Path] = None,
        exclude_etf: bool = False,
    ):
        if not accepted_caps:
            raise ValueError("accepted_caps must be a non-empty set")
        if require_mtf and mtf_snapshot_path is None:
            raise ValueError("require_mtf=True requires mtf_snapshot_path")
        self._accepted_caps = set(accepted_caps)
        self._require_mis = bool(require_mis)
        self._require_mtf = bool(require_mtf)
        self._min_days = int(min_trading_days_required)
        self._min_vol = float(min_daily_avg_volume)
        self._exclude_etf = bool(exclude_etf)
        self._mtf_snapshot_path = mtf_snapshot_path

        self._daily_df: Optional[pd.DataFrame] = None
        self._nse_all: Optional[dict] = None
        self._mtf: Optional[dict] = None  # bare -> {'category': str, 'leverage': float} or None

    # ------------- Lazy data loaders -------------

    def _load_daily(self) -> pd.DataFrame:
        if self._daily_df is None:
            df = pd.read_feather(_DAILY_PATH)
            df["ts"] = pd.to_datetime(df["ts"])
            df["d"] = df["ts"].dt.date
            self._daily_df = df
        return self._daily_df

    def _load_nse_all(self) -> dict:
        if self._nse_all is None:
            with open(_NSE_ALL, encoding="utf-8") as f:
                entries = json.load(f)
            out = {}
            for e in entries:
                if not isinstance(e, dict):
                    continue
                bare = _normalize_symbol(e.get("symbol", ""))
                if bare:
                    out[bare] = _NseRow(
                        cap_segment=str(e.get("cap_segment", "unknown")),
                        mis_enabled=bool(e.get("mis_enabled", False)),
                    )
            self._nse_all = out
        return self._nse_all

    def _load_mtf(self) -> dict:
        if self._mtf is None:
            if self._mtf_snapshot_path is None:
                self._mtf = {}
                return self._mtf
            with open(self._mtf_snapshot_path, encoding="utf-8") as f:
                entries = json.load(f)
            out = {}
            for e in entries:
                if not isinstance(e, dict):
                    continue
                bare = _normalize_symbol(e.get("tradingsymbol", ""))
                if bare:
                    out[bare] = {
                        "category": str(e.get("category", "non_categorized")),
                        "leverage": float(e.get("leverage", 1.0)),
                    }
            self._mtf = out
        return self._mtf

    # ------------- Cap segment lookup (matches services.symbol_metadata.get_cap_segment) -------------

    def _cap_segment(self, bare: str) -> str:
        """Same default as services.symbol_metadata: missing -> 'unknown'."""
        nse_all = self._load_nse_all()
        row = nse_all.get(bare)
        return row.cap_segment if row else "unknown"

    def _mis_enabled(self, bare: str) -> bool:
        nse_all = self._load_nse_all()
        row = nse_all.get(bare)
        return row.mis_enabled if row else False

    def _mtf_eligible(self, bare: str) -> bool:
        mtf = self._load_mtf()
        info = mtf.get(bare)
        if info is None:
            return False
        if self._exclude_etf and info["category"] == "etf":
            return False
        return True

    # ------------- Public API -------------

    def is_eligible(self, bare_symbol: str, session_date: date) -> bool:
        """Return True if symbol passes ALL production universe filters on session_date.

        Mirrors:
          1. cap_segment in accepted_caps  (default 'unknown' if missing)
          2. mis_enabled (if require_mis=True)
          3. MTF eligibility (if require_mtf=True)
          4. min trading days in consolidated_daily PRIOR to session_date
          5. avg daily volume in consolidated_daily PRIOR to session_date
        """
        bare = _normalize_symbol(bare_symbol)

        # Filter 1: cap segment
        cap = self._cap_segment(bare)
        if cap not in self._accepted_caps:
            return False

        # Filter 2: MIS
        if self._require_mis and not self._mis_enabled(bare):
            return False

        # Filter 3: MTF
        if self._require_mtf and not self._mtf_eligible(bare):
            return False

        # Filters 4 + 5: daily history (only if thresholds > 0)
        if self._min_days > 0 or self._min_vol > 0:
            daily = self._load_daily()
            sym_rows = daily[(daily.symbol == bare) & (daily.d < session_date)]
            if self._min_days > 0 and len(sym_rows) < self._min_days:
                return False
            if self._min_vol > 0:
                if sym_rows.empty:
                    return False
                if float(sym_rows.volume.mean()) < self._min_vol:
                    return False

        return True

    def filter_symbols(self, symbols: Iterable[str], session_date: date) -> Set[str]:
        """Convenience: filter a list of symbols by per-date eligibility.

        Returns the BARE (normalized) symbol set that passes.
        """
        return {
            _normalize_symbol(s)
            for s in symbols
            if self.is_eligible(s, session_date)
        }
