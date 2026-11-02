"""Results context of a symbol on a session, from data known before the open.

Audit 2026-11-01 (re-run 11-02): up_spike_fade_short trades taken after the
company's results were public made half of the honest backtest's P&L (117 of
1,241 trades, PF 3.2, higher in every year); trades with no results nearby were
about break-even after live costs. This module tags each signal so the forward
record can test that split before any sizing change. LOGGING ONLY.

Tags, from data/earnings_calendar/earnings_events.parquet (refreshed at start):
  results_preopen      results released after the previous close or before
                       09:15 today (the reaction day is today)
  board_meeting_today  a results board meeting is scheduled today; SEBI LODR
                       Reg 29 makes it public days ahead, release time unknown
  none                 neither
  unknown              the calendar could not be read

Both tags are knowable before the open, so the same call is point-in-time in a
backtest. Intraday releases without a scheduled meeting are not seen.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_PARQUET = Path(__file__).resolve().parents[1] / "data" / "earnings_calendar" / "earnings_events.parquet"
_OPEN = time(9, 15)


def _bare(sym: str) -> str:
    return str(sym).split(":", 1)[-1]


@lru_cache(maxsize=4)
def _tags_cached(session_iso: str, path: str) -> Dict[str, str]:
    ev = pd.read_parquet(path, columns=["symbol", "announce_time", "trade_date", "bm_date",
                                        "announce_time_class", "announce_date"])
    sess = date.fromisoformat(session_iso)
    open_ts = datetime.combine(sess, _OPEN)
    sym = ev["symbol"].map(_bare)
    trade_day = ev["trade_date"].astype(str).str[:10]
    # "scheduled" rows carry a placeholder 09:00 time: a meeting, not a release.
    preopen = ((trade_day == session_iso) & (pd.to_datetime(ev["announce_time"]) < open_ts)
               & (ev["announce_time_class"] != "scheduled"))
    on_day = pd.to_datetime(ev["announce_date"]).dt.date == sess
    # The calendar keeps one row per (symbol, day), the release winning over the
    # scheduled meeting; a release dated today during or after market hours means
    # its board meeting is today, which Reg 29 made public days ahead. Live, at
    # start, today's meetings are still "scheduled" rows (nothing released yet).
    meeting = ((ev["bm_date"].astype(str).str[:10] == session_iso)
               | (on_day & ev["announce_time_class"].isin(["scheduled", "intraday", "AMC"])))
    tags = {s: "board_meeting_today" for s in sym[meeting]}
    tags.update({s: "results_preopen" for s in sym[preopen]})
    return tags


def _tags_for(session_iso: str, path: str) -> Tuple[Dict[str, str], Optional[str]]:
    """Errors are returned, never cached: a read during the other daemon's refresh
    must not tag the whole session unknown."""
    try:
        return _tags_cached(session_iso, path), None
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"


_tags_for.cache_clear = _tags_cached.cache_clear


def results_context(symbol: str, session_date, path: Path = _PARQUET) -> str:
    """Tag for `symbol` on `session_date` (date or ISO string); see module doc."""
    iso = str(session_date)[:10]
    tags, err = _tags_for(iso, str(path))
    if err is not None:
        logger.warning("RESULTS_CONTEXT | calendar unreadable (%s); tagging unknown", err)
        return "unknown"
    return tags.get(_bare(symbol), "none")
