"""Universe-parity simulator: production (OCI) vs sanity (monthly 5m).

Investigates the divergence flagged by OCI vs sanity comparison for
`below_vwap_volume_revert_long`:
  - Sanity HO (Dec 2025-Apr 2026): 577 cell-locked trades, PF 1.17
  - OCI HO (same range):            141 completed trades, PF 0.96
  - (sym, date) overlap:            6.9%

Goal: identify which side (production universe / sanity universe) is
the source of divergence on specific (sym, date) pairs.

Method:
  1. For a sample (sym, date), simulate the PRODUCTION universe build
     using consolidated_daily.feather + nse_all.json. Check whether the
     symbol passes the cap/MIS/volume/coverage filters.
  2. Check whether the symbol exists in the monthly_YYYY_MM_5m_enriched.feather
     for that date.
  3. Compare. The mismatch tells us which side is causing the divergence.

Specific cases probed:
  Sanity-only (sanity fires, OCI doesn't):
    2025-12-22  NSE:BALAJITELE
    2026-03-23  NSE:GOLDADD       (ETF — should be excluded)
    2026-02-06  NSE:CEMPRO
    2026-01-27  NSE:SETL
    2026-01-20  NSE:JITFINFRA
  OCI-only (OCI fires, sanity doesn't):
    2026-03-30  NSE:RPPINFRA
    2026-03-30  NSE:SIMPLEXINF
    2026-02-05  NSE:TRACXN
    2026-01-09  NSE:KAMOPAINTS
    2026-01-14  NSE:CIFL
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DAILY_PATH = _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather"
_MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"
_NSE_ALL = _REPO_ROOT / "nse_all.json"


# Cell-lock for below_vwap_volume_revert_long
_REQUIRED_CAP = "unknown"
_MIN_DAILY_AVG_VOL = 50000
_MIN_TRADING_DAYS = 30


@dataclass
class SymbolUniverseStatus:
    """Per-symbol diagnostic for why each side did or didn't include it."""
    symbol_bare: str
    target_date: date

    # nse_all.json side
    in_nse_all: bool = False
    cap_segment: Optional[str] = None
    mis_enabled: bool = False

    # Production (consolidated_daily.feather) side
    in_consolidated_daily: bool = False
    daily_rows_before_target: int = 0
    daily_avg_volume: Optional[float] = None
    passes_prod_filter: bool = False
    prod_rejection_reason: Optional[str] = None

    # Sanity (monthly 5m feather) side
    in_monthly_feather: bool = False
    monthly_filename: Optional[str] = None

    def summary(self) -> str:
        prod = "YES" if self.passes_prod_filter else "NO "
        san = "YES" if self.in_monthly_feather else "NO "
        return (
            f"{self.symbol_bare:<15} {str(self.target_date):>12} | "
            f"nse_all={self.in_nse_all} cap={self.cap_segment} mis={self.mis_enabled} | "
            f"PROD={prod} ({self.prod_rejection_reason or 'passes'}) | "
            f"SANITY={san}"
        )


class UniverseSimulator:
    """Reuses the production universe-builder logic but lets us inspect each
    symbol's filter outcome, not just the final symbol set.
    """

    def __init__(self):
        self._daily_df: Optional[pd.DataFrame] = None
        self._nse_all: Optional[Dict[str, dict]] = None  # keyed by bare symbol

    def _load_daily(self) -> pd.DataFrame:
        if self._daily_df is None:
            print(f"Loading {_DAILY_PATH}...")
            df = pd.read_feather(_DAILY_PATH)
            df["ts"] = pd.to_datetime(df["ts"])
            df["d"] = df["ts"].dt.date
            self._daily_df = df
            print(f"  Loaded {len(df):,} rows, {df.symbol.nunique()} symbols")
        return self._daily_df

    def _load_nse_all(self) -> Dict[str, dict]:
        if self._nse_all is None:
            with open(_NSE_ALL, encoding="utf-8") as f:
                entries = json.load(f)
            out = {}
            for e in entries:
                if not isinstance(e, dict):
                    continue
                sym = e.get("symbol", "")
                # Strip 'NSE:' prefix AND '.NS' suffix (matches services.symbol_metadata._normalize_symbol)
                if ":" in sym:
                    sym = sym.split(":")[-1]
                if "." in sym:
                    sym = sym.split(".")[0]
                if sym:
                    out[sym] = e
            self._nse_all = out
        return self._nse_all

    def diagnose(self, symbol: str, target_date: date) -> SymbolUniverseStatus:
        """Run all filter checks for ONE (symbol, date)."""
        bare = symbol.replace("NSE:", "")
        s = SymbolUniverseStatus(symbol_bare=bare, target_date=target_date)

        # nse_all.json check
        nse_all = self._load_nse_all()
        if bare in nse_all:
            e = nse_all[bare]
            s.in_nse_all = True
            s.cap_segment = e.get("cap_segment")
            s.mis_enabled = bool(e.get("mis_enabled", False))

        # Production daily check
        daily = self._load_daily()
        sym_rows = daily[(daily.symbol == bare) & (daily.d < target_date)]
        s.in_consolidated_daily = len(sym_rows) > 0
        s.daily_rows_before_target = len(sym_rows)
        if not sym_rows.empty:
            s.daily_avg_volume = float(sym_rows.volume.mean())

        # Production filter (mirrors services/setup_universe.py:below_vwap_volume_revert_long_universe)
        if not s.in_nse_all:
            s.prod_rejection_reason = "not in nse_all.json"
        elif s.cap_segment != _REQUIRED_CAP:
            s.prod_rejection_reason = f"cap_segment={s.cap_segment!r} != {_REQUIRED_CAP!r}"
        elif not s.mis_enabled:
            s.prod_rejection_reason = "mis_enabled=False"
        elif not s.in_consolidated_daily:
            s.prod_rejection_reason = "no rows in consolidated_daily"
        elif s.daily_rows_before_target < _MIN_TRADING_DAYS:
            s.prod_rejection_reason = (
                f"only {s.daily_rows_before_target} prior trading days "
                f"(need >= {_MIN_TRADING_DAYS})"
            )
        elif (s.daily_avg_volume or 0) < _MIN_DAILY_AVG_VOL:
            s.prod_rejection_reason = (
                f"avg_vol={s.daily_avg_volume:.0f} < {_MIN_DAILY_AVG_VOL}"
            )
        else:
            s.passes_prod_filter = True

        # Sanity (monthly 5m) check
        yyyy_mm = target_date.strftime("%Y_%m")
        monthly_path = _MONTHLY_DIR / f"{yyyy_mm}_5m_enriched.feather"
        if monthly_path.exists():
            s.monthly_filename = monthly_path.name
            try:
                mdf = pd.read_feather(monthly_path, columns=["symbol"])
                s.in_monthly_feather = bare in set(mdf.symbol.unique())
            except Exception:
                pass

        return s


def main():
    sim = UniverseSimulator()

    # Sample mismatches from the earlier comparison
    sanity_only = [
        (date(2025, 12, 22), "NSE:BALAJITELE"),
        (date(2026, 3, 23), "NSE:GOLDADD"),
        (date(2026, 2, 6),  "NSE:CEMPRO"),
        (date(2026, 1, 27), "NSE:SETL"),
        (date(2026, 1, 20), "NSE:JITFINFRA"),
    ]
    oci_only = [
        (date(2026, 3, 30), "NSE:RPPINFRA"),
        (date(2026, 3, 30), "NSE:SIMPLEXINF"),
        (date(2026, 2, 5),  "NSE:TRACXN"),
        (date(2026, 1, 9),  "NSE:KAMOPAINTS"),
        (date(2026, 1, 14), "NSE:CIFL"),
    ]

    print("=" * 100)
    print("SANITY-ONLY cases (sanity fires; OCI does not)")
    print("=" * 100)
    print(f"{'symbol':<15} {'target_date':>12} | nse_all=? cap=? mis=? | PROD | SANITY")
    print("-" * 100)
    for d, sym in sanity_only:
        s = sim.diagnose(sym, d)
        print(s.summary())

    print()
    print("=" * 100)
    print("OCI-ONLY cases (OCI fires; sanity does not)")
    print("=" * 100)
    print(f"{'symbol':<15} {'target_date':>12} | nse_all=? cap=? mis=? | PROD | SANITY")
    print("-" * 100)
    for d, sym in oci_only:
        s = sim.diagnose(sym, d)
        print(s.summary())

    # ----- Aggregate: production vs sanity universe SIZE on one specific date -----
    target = date(2026, 3, 30)
    print()
    print("=" * 100)
    print(f"AGGREGATE universe sizes on {target}")
    print("=" * 100)

    daily = sim._load_daily()
    nse_all = sim._load_nse_all()

    # Production: filter daily_dict
    prod_qual = set()
    history_floor = target - timedelta(days=120)  # broad date window for prior history
    grouped = daily[daily.d < target].groupby("symbol")
    for bare, ddf in grouped:
        if bare not in nse_all:
            continue
        e = nse_all[bare]
        if e.get("cap_segment") != _REQUIRED_CAP:
            continue
        if not e.get("mis_enabled", False):
            continue
        if len(ddf) < _MIN_TRADING_DAYS:
            continue
        if float(ddf.volume.mean()) < _MIN_DAILY_AVG_VOL:
            continue
        prod_qual.add(bare)

    # Sanity: read monthly_2026_03 feather, get all symbols
    yyyy_mm = target.strftime("%Y_%m")
    monthly_path = _MONTHLY_DIR / f"{yyyy_mm}_5m_enriched.feather"
    if monthly_path.exists():
        mdf = pd.read_feather(monthly_path, columns=["symbol"])
        # Sanity universe = symbols that pass the SAME filters but using monthly data sources
        # For simpler comparison: just the bare symbol set in the feather, intersected with nse_all+cap+mis+history
        sanity_all_in_feather = set(mdf.symbol.unique())
        # Apply nse_all + cap=unknown + mis filter on top
        sanity_qual = set()
        for bare in sanity_all_in_feather:
            if bare not in nse_all:
                continue
            e = nse_all[bare]
            if e.get("cap_segment") != _REQUIRED_CAP:
                continue
            if not e.get("mis_enabled", False):
                continue
            sanity_qual.add(bare)
    else:
        sanity_qual = set()

    overlap = prod_qual & sanity_qual
    prod_only = prod_qual - sanity_qual
    sanity_only_set = sanity_qual - prod_qual

    print(f"  Production universe (consolidated_daily + nse_all filters): {len(prod_qual)} symbols")
    print(f"  Sanity universe (monthly feather symbols + nse_all filters): {len(sanity_qual)} symbols")
    print(f"  Overlap: {len(overlap)} ({len(overlap)/max(1,len(prod_qual))*100:.1f}% of prod)")
    print(f"  Production-only (not in sanity): {len(prod_only)}")
    print(f"  Sanity-only (not in production): {len(sanity_only_set)}")
    print()
    print(f"  Sample production-only symbols (first 10): {sorted(list(prod_only))[:10]}")
    print(f"  Sample sanity-only symbols (first 10): {sorted(list(sanity_only_set))[:10]}")


if __name__ == "__main__":
    main()
