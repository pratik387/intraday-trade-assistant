"""Session-start cross-day RVOL baseline, computed live from Historical API.

Replaces dependency on `data/cross_day_rvol/rvol_baseline.parquet` for the
live/paper path. The parquet path is preserved for backtest (see
`services/cross_day_rvol_enrichment.py` dispatch).

Live/paper flow:
  1. ScreenerLive daily-seed: after `_setup_universes["delivery_pct_anomaly_short"]`
     is computed (~5-30 qualifying symbols), call:

        rt = RuntimeRvolBaseline()
        asyncio.run(rt.populate(sdk, qualifying_symbols, session_date))
        cross_day_rvol_enrichment.set_runtime_baseline(rt)

  2. Detector's `_cross_day_rvol(df_5m, session_date, symbol)` calls
     `cross_day_rvol_enrichment.get_baseline_vol(...)` which now consults
     this runtime singleton first, falls through to parquet only if missing.

Data shape — per (symbol, hhmm) → mean volume of last `rolling_days`
prior-session same-tod bars. Today's date is encoded implicitly (a single
RuntimeRvolBaseline instance is single-session — re-instantiated next day).

Why a class instead of module-level state:
  Backtests in a single python process can replay multiple sessions back-
  to-back. Backtest uses the parquet path (skips populate), so this instance
  never gets touched in dry-run. But keeping the API class-based means a
  future multi-session-per-process live mode (rare) wouldn't accidentally
  reuse yesterday's baselines.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date as _date, timedelta
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


logger = logging.getLogger(__name__)


class RuntimeRvolBaseline:
    """In-memory baseline keyed by (bare_symbol, hhmm) → mean volume."""

    def __init__(self) -> None:
        self._lookup: Dict[Tuple[str, int], float] = {}
        self._n_symbols_loaded: int = 0
        self._n_symbols_missing_data: int = 0
        self._n_429s: int = 0
        self._populated: bool = False

    @staticmethod
    def _bare(symbol: str) -> str:
        return symbol.replace("NSE:", "") if symbol.startswith("NSE:") else symbol

    def get_baseline_vol(
        self, symbol: str, session_date: _date, hhmm: int,
    ) -> Optional[float]:
        """O(1) lookup. `session_date` is unused here (single-session baseline).
        Returns None if (symbol, hhmm) not in cache or value <= 0."""
        val = self._lookup.get((self._bare(symbol), int(hhmm)))
        if val is None or val <= 0:
            return None
        return val

    def is_populated(self) -> bool:
        return self._populated

    def stats(self) -> dict:
        return {
            "populated": self._populated,
            "n_symbols_loaded": self._n_symbols_loaded,
            "n_symbols_missing_data": self._n_symbols_missing_data,
            "n_429s": self._n_429s,
            "n_lookup_entries": len(self._lookup),
        }

    async def populate(
        self,
        sdk,
        symbols: Iterable[str],
        session_date: _date,
        hhmm_window: Tuple[int, int] = (930, 1000),
        rolling_days: int = 20,
    ) -> dict:
        """Fetch ~30 prior calendar days of 5m bars per symbol; compute mean
        volume per hhmm within `hhmm_window` across last `rolling_days`
        unique trading dates seen.

        `sdk` must expose `async_fetch_historical_5m_batch(symbols, from_date,
        to_date, concurrency, rps)` returning a Dict[str, DataFrame] with a
        DatetimeIndex and a `volume` column. Both KiteClient (post this
        commit) and UpstoxDataClient satisfy this contract.

        Returns the stats dict — logged by the caller.
        """
        sym_list = list({s for s in symbols if s})
        if not sym_list:
            logger.info("RUNTIME_RVOL | no symbols to populate — skipping")
            self._populated = True
            return self.stats()

        # Calendar buffer for 20 trading days = ~30 calendar days; pad to 40
        # to absorb long weekends + NSE holidays.
        from_date = session_date - timedelta(days=rolling_days * 2)
        to_date = session_date - timedelta(days=1)

        if not hasattr(sdk, "async_fetch_historical_5m_batch"):
            raise RuntimeError(
                f"sdk type={type(sdk).__name__} has no async_fetch_historical_5m_batch; "
                f"cannot populate runtime rvol baseline"
            )

        # Determine RPS / concurrency per sdk. Conservative defaults if the
        # sdk doesn't carry hints — Upstox can sustain 40 RPS; Kite caps at
        # ~3 RPS per docstring.
        sdk_type = type(sdk).__name__
        if "Kite" in sdk_type:
            rps, concurrency = 2.5, 3
        else:
            rps, concurrency = 20.0, 30

        logger.info(
            "RUNTIME_RVOL | populating %d symbols from %s to %s "
            "(sdk=%s rps=%.1f concurrency=%d)",
            len(sym_list), from_date, to_date, sdk_type, rps, concurrency,
        )

        bars_map = await sdk.async_fetch_historical_5m_batch(
            symbols=sym_list,
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
            concurrency=concurrency,
            rps=rps,
        )

        # The SDK exposes _last_batch_429s on Upstox; mirror if present.
        self._n_429s = int(getattr(sdk, "_last_batch_429s", 0) or 0)

        hhmm_min, hhmm_max = hhmm_window

        for sym, df in bars_map.items():
            if df is None or df.empty:
                self._n_symbols_missing_data += 1
                continue
            # df is expected DatetimeIndex; compute (date, hhmm) per row
            try:
                idx = df.index
                if not isinstance(idx, pd.DatetimeIndex):
                    idx = pd.to_datetime(idx)
                hhmm_arr = idx.hour * 100 + idx.minute
                date_arr = idx.date
                tmp = pd.DataFrame({
                    "date": date_arr,
                    "hhmm": hhmm_arr,
                    "volume": df["volume"].astype(float).values,
                })
            except Exception as e:
                logger.warning("RUNTIME_RVOL | %s: bar dataframe shape error %s",
                               sym, e)
                self._n_symbols_missing_data += 1
                continue

            # Restrict to active window
            tmp = tmp[(tmp["hhmm"] >= hhmm_min) & (tmp["hhmm"] <= hhmm_max)]
            if tmp.empty:
                self._n_symbols_missing_data += 1
                continue

            # Keep only the last `rolling_days` unique trading dates seen
            unique_dates = sorted(tmp["date"].unique())
            if len(unique_dates) < 5:
                # Below the production min_periods floor in build_baseline.py;
                # the parquet path also drops these. Be conservative.
                self._n_symbols_missing_data += 1
                continue
            keep_dates = set(unique_dates[-rolling_days:])
            tmp = tmp[tmp["date"].isin(keep_dates)]
            if tmp.empty:
                self._n_symbols_missing_data += 1
                continue

            # Per-hhmm mean volume
            bare = self._bare(sym)
            for hhmm, grp in tmp.groupby("hhmm"):
                mean_vol = float(grp["volume"].mean())
                if mean_vol > 0:
                    self._lookup[(bare, int(hhmm))] = mean_vol
            self._n_symbols_loaded += 1

        self._populated = True
        result = self.stats()
        logger.info("RUNTIME_RVOL | done %s", result)
        return result
