"""Earnings-reaction-day enrichment for `df_daily`.

Adds an `is_earnings_reaction_day` boolean column to a per-symbol daily
frame, marking the session on which the market first traded a quarterly
result. Mirrors `services/delivery_pct_enrichment.py` exactly: load the
parquet once, cache a dict for O(1) lookup, enrich in place, never raise.

Consumer: `structures/earnings_downshock_continuation_short_structure.py`
reads the column off `ctx.df_daily` so the detector itself does no I/O
(same contract `delivery_pct_anomaly_short` uses for `delivery_pct`).

REACTION-DAY ASSIGNMENT (must stay identical to the research reference,
`tools/sub9_research/sanity_earnings_downshock_continuation_short.py`):
  announce_time_class in `reaction_same_day_classes` (intraday / BMO)
      -> the announcement date IS the reaction day
  announce_time_class == "scheduled" with announce hour < reaction_same_day_hour_max
      -> the announcement date IS the reaction day
  otherwise (AMC, or scheduled at/after the hour cutoff)
      -> the NEXT trading day is the reaction day
Both class lists and the hour cutoff come from the setup's config block —
no defaults here (CLAUDE.md rule 1).

DE-DUPLICATION: standalone and consolidated filings for the same company
and session are separate rows in the parquet. They are collapsed to one
(symbol, reaction_date) key. This is load-bearing: leaving them in
inflated the research era_A expectancy from +0.057% to +0.183%.

DATA DEPENDENCY (must exist on the OCI/VM box):
  data/earnings_calendar/earnings_events.parquet
Repaired 2026-07-28 (`tools/earnings_calendar/repair_post_fr_window.py`)
after NSE retired the "Financial Result Updates" subject in Mar-2025 and
the stream migrated to "Outcome of Board Meeting". A stale copy silently
degrades reaction-day assignment for ~1/3 of post-Mar-2025 events, which
is exactly the failure the data-health monitor in
`jobs/check_edge_integrity.py` watches for. Missing file = no fires (the
detector rejects with a loud reason), never a crash.

All timestamps IST-naive (CLAUDE.md rule 2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_PARQUET = _REPO / "data" / "earnings_calendar" / "earnings_events.parquet"

_REACTION_KEYS: Optional[Set[Tuple[str, object]]] = None
_LOAD_ATTEMPTED = False
_LOAD_ERROR: Optional[str] = None


def _bare(sym: str) -> str:
    return str(sym).replace("NSE:", "").replace(".NS", "").strip().upper()


def _load(
    same_day_classes: Tuple[str, ...],
    same_day_hour_max: int,
) -> Optional[Set[Tuple[str, object]]]:
    """Build the {(bare_symbol, reaction_date)} set once. None on failure."""
    global _REACTION_KEYS, _LOAD_ATTEMPTED, _LOAD_ERROR
    if _LOAD_ATTEMPTED:
        return _REACTION_KEYS
    _LOAD_ATTEMPTED = True

    if not _PARQUET.exists():
        _LOAD_ERROR = f"missing {_PARQUET}"
        return None
    try:
        cols = ["symbol", "announce_date", "announce_time", "announce_time_class"]
        df = pd.read_parquet(_PARQUET, columns=cols)
        df["symbol"] = df["symbol"].map(_bare)
        df["announce_date"] = pd.to_datetime(df["announce_date"]).dt.normalize()
        hour = pd.to_datetime(df["announce_time"], errors="coerce").dt.hour
        cls = df["announce_time_class"].astype(str)

        # Same-day requires BOTH: an eligible class AND an announcement early
        # enough in the session to actually be traded. The hour cutoff applies
        # TO the same-day classes — it is not an escape hatch for "scheduled".
        # Validated against the research reference: PSB intraday 15:27 and
        # CAMPUS intraday 15:26 both react the NEXT session (3-4 min before the
        # close is not tradeable), and "scheduled" rows carry a SYNTHETIC 09:00
        # timestamp from the calendar scrape, so their hour is meaningless and
        # they must never qualify as same-day. Missing hour -> next session.
        same_day = (
            cls.isin(same_day_classes)
            & hour.notna()
            & (hour < int(same_day_hour_max))
        )
        # Next-trading-day resolution is deferred to the caller, which owns
        # the symbol's real session calendar (df_daily). Here we emit the
        # ANCHOR date plus the same-day flag; `enrich` walks it forward.
        _REACTION_KEYS = set(
            zip(df["symbol"], df["announce_date"].dt.date, same_day.astype(bool))
        )
        return _REACTION_KEYS
    except Exception as exc:  # never raise into the detector path
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def load_error() -> Optional[str]:
    """Why the lookup is empty, for logging by the caller."""
    return _LOAD_ERROR


def enrich(
    daily_df: pd.DataFrame,
    symbol: str,
    *,
    same_day_classes,
    same_day_hour_max: int,
) -> pd.DataFrame:
    """Add `is_earnings_reaction_day` to a per-symbol daily frame.

    The frame's own index supplies the session calendar, so an AMC filing
    resolves to the next REAL trading session for this symbol (holidays and
    suspensions included) rather than a naive calendar +1 day.

    Returns the frame with the column added (all-False when the calendar is
    unavailable). Never raises.
    """
    if daily_df is None or daily_df.empty:
        return daily_df

    keys = _load(tuple(same_day_classes), int(same_day_hour_max))
    if not keys:
        daily_df = daily_df.copy()
        daily_df["is_earnings_reaction_day"] = False
        return daily_df

    bare = _bare(symbol)
    sessions = pd.to_datetime(pd.Index(daily_df.index)).normalize()
    session_dates = list(sessions.date)
    session_pos = {d: i for i, d in enumerate(session_dates)}

    flags = [False] * len(session_dates)
    for sym, anchor_date, is_same_day in keys:
        if sym != bare:
            continue
        if is_same_day:
            # Same-day classes resolve to the announcement date ONLY if it is
            # a real trading session. Filings land on Saturdays (board meetings
            # are commonly held at weekends) and are still classed "intraday";
            # those roll forward to the next session, matching the research
            # reference. Validated: without this, 2 of 25 spot-checked Stage-4
            # trades were dropped (SIYSIL 2023-01-28 Sat, SKIPPER 2023-02-04
            # Sat) — a ~8% systematic under-fire vs the validated cohort.
            i = session_pos.get(anchor_date)
            if i is None:
                i = next(
                    (j for j, d in enumerate(session_dates) if d > anchor_date),
                    None,
                )
        else:
            # next REAL session strictly after the anchor
            i = next(
                (j for j, d in enumerate(session_dates) if d > anchor_date),
                None,
            )
        if i is not None:
            flags[i] = True  # de-dup is implicit: repeated filings set the same slot

    out = daily_df.copy()
    out["is_earnings_reaction_day"] = flags
    return out
