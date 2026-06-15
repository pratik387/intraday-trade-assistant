"""Daily-panel provider for multi-day CNC/MTF cross-sectional setups.

Returns the clean, corporate-action-ADJUSTED daily OHLCV panel that
`CrossSectionalRanker` consumes — source-agnostic so live and backtest share
ONE code path (avoids the live/backtest divergence that dirty `consolidated_
daily` caused; see memory `feedback_clean_data_before_cross_sectional`).

Backtest: slice the clean feather (cache/preaggregate/clean_daily_from5m.feather,
resampled from the verified CA-adjusted 5m monthly feathers).

Live: fetch the trailing window of adjusted daily bars for the MTF universe via
an injected fetch_fn (Upstox/Kite historical daily is split-adjusted at source,
so it stays consistent with the backtest panel). The fetch_fn is injected (not
hard-bound) so the windowing/assembly logic is unit-testable.

IST-naive throughout. Trailing-window size is config-driven (no hardcoded
defaults): we need max(lookback, shock_lookback) + buffer trading days.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd

from config.logging_config import get_agent_logger

logger = get_agent_logger()

_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def _window_calendar_days(config: Dict[str, Any]) -> int:
    """Calendar-day span to cover the required trailing TRADING days.

    The ranker needs ~max(lookback, shock_lookback)+2 clean rows AT-OR-BEFORE
    as_of per symbol. Illiquid names trade with gaps and some months are sparse
    (e.g. Feb-2026), so we load a generous margin (≈2x the shock window) to
    guarantee enough rows even for thinly-traded symbols. Daily data is small;
    over-fetching is cheap and avoids dropping valid names in sparse periods.
    """
    # The "signal lookback" depends on the selection mode: trailing-return setups
    # need `lookback_days`; near-period-low setups need the (much longer)
    # `low_lookback_days` (e.g. 252). Fail-fast on selection_mode (CLAUDE.md rule
    # 1) — a silent default here would silently under-fetch the window for a
    # mis-keyed near-low setup (5-day vs 252-day) -> empty basket, no error.
    if str(config["selection_mode"]) == "near_period_low":
        sig_lookback = int(config["low_lookback_days"])
    else:
        sig_lookback = int(config["lookback_days"])
    need_trading = int(config["shock_lookback_days"]) * 2 + sig_lookback + 10
    # ~5 trading days per 7 calendar days, plus a holiday buffer.
    return math.ceil(need_trading * 7 / 5) + 10


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    return df[_COLS].sort_values(["symbol", "date"]).reset_index(drop=True)


class DailyPanelProvider:
    """Interface: return the clean daily panel up to (and including) as_of."""

    def get_panel(self, as_of: date) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError


class FeatherDailyPanelProvider(DailyPanelProvider):
    """Backtest backend: slice the clean adjusted daily feather."""

    def __init__(self, feather_path: Path, config: Dict[str, Any]):
        self._path = Path(feather_path)
        self._win = _window_calendar_days(config)
        self._cache: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._cache is None:
            if not self._path.exists():
                raise FileNotFoundError(f"clean daily panel not found: {self._path}")
            self._cache = _normalize(pd.read_feather(self._path))
        return self._cache

    def get_panel(self, as_of: date) -> pd.DataFrame:
        df = self._load()
        lo = pd.Timestamp(as_of) - pd.Timedelta(days=self._win)
        hi = pd.Timestamp(as_of)
        out = df[(df["date"] >= lo) & (df["date"].dt.normalize() <= hi)]
        return out.reset_index(drop=True)


class LiveDailyPanelProvider(DailyPanelProvider):
    """Live backend: fetch trailing adjusted daily bars for the MTF universe.

    fetch_fn(symbols, start_date, end_date) -> DataFrame[_COLS]. Injected so the
    cron wires the real data client and tests can mock it. Symbols are the
    MTF-eligible bare tickers (the only names we can leverage / would rank).
    """

    def __init__(
        self,
        fetch_fn: Callable[[Iterable[str], date, date], pd.DataFrame],
        mtf_symbols: Iterable[str],
        config: Dict[str, Any],
    ):
        self._fetch = fetch_fn
        self._symbols = sorted({str(s).replace("NSE:", "").upper() for s in mtf_symbols})
        self._win = _window_calendar_days(config)

    def get_panel(self, as_of: date) -> pd.DataFrame:
        start = as_of - timedelta(days=self._win)
        raw = self._fetch(self._symbols, start, as_of)
        if raw is None or len(raw) == 0:
            logger.warning("LiveDailyPanelProvider: fetch returned no bars for %s..%s", start, as_of)
            return pd.DataFrame(columns=_COLS)
        df = _normalize(raw)
        df = df[df["date"].dt.normalize() <= pd.Timestamp(as_of)]
        return df.reset_index(drop=True)


def make_provider(config: Dict[str, Any], *, dry_run: bool,
                  fetch_fn=None, mtf_symbols=None, repo_root: Optional[Path] = None) -> DailyPanelProvider:
    """Factory: feather in backtest (dry_run), live fetcher otherwise."""
    if dry_run:
        root = repo_root or Path(__file__).resolve().parents[1]
        return FeatherDailyPanelProvider(root / config["data_source"], config)
    if fetch_fn is None or mtf_symbols is None:
        raise ValueError("live DailyPanelProvider needs fetch_fn and mtf_symbols")
    return LiveDailyPanelProvider(fetch_fn, mtf_symbols, config)
