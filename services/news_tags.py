"""Pure tagging for the multiday news gate (R3 results-day skip + bulk tilt).

Design: specs/2026-10-06-multiday-news-skip-and-bulk-tilt-design.md §2, §4.

Everything here is a pure function over DataFrames the caller has already
loaded (announcements, event calendar, bulk deals) plus a caller-supplied
NSE trading-day index. No I/O, no network, no wall clock. Every threshold,
window and regex comes from the ``news_gate`` config block passed in and is
read with ``[]`` so a missing key raises ``KeyError`` (fail fast at startup).

Timestamps: IST-naive throughout. Any tz-aware input is coerced to IST
wall-time and stripped via ``utils.time_util._to_naive_ist``.

Business-day stepping uses ``trading_days`` (the real NSE calendar), never
``pandas.tseries.offsets.BDay`` — NSE holidays are not weekends.

Feed-availability policy (spec §3) lives in the CALLER: a stale/missing feed
is passed here as an empty frame, which yields "no tags" (skip fails OPEN,
tilt fails CLOSED at 1.0). This module never raises on empty frames.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

from utils.time_util import IST_TZ, _parse_hhmm_str, _to_naive_ist

# Category label that marks a filing as a realised results filing. The regex
# for it lives in cfg["category_patterns"]; this is only the LABEL the
# results_reaction leg looks for, and tag_symbol_day verifies the pattern
# list actually declares it (ValueError otherwise).
RESULTS_CATEGORY = "results"
OTHER_CATEGORY = "other"

# Keys every caller must supply. validate_news_gate_cfg touches each one so a
# half-written config dies at startup, not on the first signal day.
REQUIRED_CFG_KEYS: Tuple[str, ...] = (
    "enabled",
    "results_days_before",
    "results_days_after",
    "shallow_drop_sma50_pct",
    "bulk_overrides_skip",
    "bulk_deal_window_bdays",
    "filing_cutoff_hhmm",
    "results_cluster_categories",
    "category_patterns",
    "scheduled_purpose_regex",
    "bulk_deal_risk_multiplier",
)


@dataclass(frozen=True)
class NewsTags:
    results_reaction: bool
    results_scheduled: bool
    bulk_deal: bool
    bulk_side: Optional[str]
    categories: frozenset
    results_cluster: bool


EMPTY_TAGS = NewsTags(
    results_reaction=False,
    results_scheduled=False,
    bulk_deal=False,
    bulk_side=None,
    categories=frozenset(),
    results_cluster=False,
)


# ----------------------------------------------------------------- helpers


def _bare(symbol: str) -> str:
    """'NSE:XYZ' / 'XYZ.NS' / 'xyz' -> 'XYZ'."""
    return str(symbol).replace("NSE:", "").replace(".NS", "").strip().upper()


def validate_news_gate_cfg(cfg: dict) -> None:
    """Touch every required key so a missing one raises KeyError up front."""
    for key in REQUIRED_CFG_KEYS:
        cfg[key]
    _parse_hhmm_str(cfg["filing_cutoff_hhmm"])
    patterns = _as_pattern_pairs(cfg["category_patterns"])
    if not any(name == RESULTS_CATEGORY for name, _ in patterns):
        raise ValueError(
            f"news_gate.category_patterns must declare a '{RESULTS_CATEGORY}' "
            f"category; got {[n for n, _ in patterns]}"
        )


def _as_pattern_pairs(patterns: Iterable[Sequence[str]]) -> list:
    """Config JSON gives [[name, regex], ...]; accept any 2-sequence."""
    out = []
    for entry in patterns:
        if len(entry) != 2:
            raise ValueError(f"category pattern must be (name, regex); got {entry!r}")
        name, regex = entry
        out.append((str(name), re.compile(str(regex), re.IGNORECASE)))
    return out


def _naive_ts(value) -> pd.Timestamp:
    return _to_naive_ist(pd.Timestamp(value))


def _naive_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(IST_TZ).dt.tz_localize(None)
    return dt


def _trading_index(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(trading_days))
    if idx.tz is not None:
        idx = idx.tz_convert(IST_TZ).tz_localize(None)
    idx = idx.normalize()
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("trading_days must be a sorted, de-duplicated DatetimeIndex")
    return idx


def _position(day: pd.Timestamp, idx: pd.DatetimeIndex) -> int:
    pos = idx.searchsorted(day)
    if pos >= len(idx) or idx[pos] != day:
        raise ValueError(f"{day.date()} is not in trading_days")
    return int(pos)


def _step(day: pd.Timestamp, k: int, idx: pd.DatetimeIndex) -> pd.Timestamp:
    """Trading day k steps from `day` (k may be negative). Raises if the
    calendar does not cover the target — the caller owns calendar range."""
    target = _position(day, idx) + k
    if target < 0 or target >= len(idx):
        raise ValueError(
            f"trading_days does not cover {day.date()} {k:+d} trading days "
            f"(calendar spans {idx[0].date()}..{idx[-1].date()})"
        )
    return idx[target]


def _window(day: pd.Timestamp, back: int, fwd: int, idx: pd.DatetimeIndex) -> set:
    """{day - back .. day + fwd} in trading days, inclusive."""
    return {_step(day, k, idx) for k in range(-back, fwd + 1)}


def _rows_for(df: Optional[pd.DataFrame], symbol: str) -> pd.DataFrame:
    """Rows of `df` for the bare symbol; empty frame on None/empty input."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    sym = df["symbol"].map(_bare)
    # reset_index: feeds are often concatenated parquet chunks with duplicate
    # labels, and the boolean masks below must align positionally.
    return df.loc[(sym == symbol).values].reset_index(drop=True)


# ---------------------------------------------------------------- public


def bucket_filing(desc: str, text: str, patterns: list) -> str:
    """First (name, regex) whose regex matches the lowered desc+text; else 'other'."""
    hay = f"{'' if desc is None else desc} {'' if text is None else text}".lower()
    for name, regex in _as_pattern_pairs(patterns):
        if regex.search(hay):
            return name
    return OTHER_CATEGORY


def reaction_day(
    an_dt: pd.Timestamp, cutoff_hhmm: str, trading_days: pd.DatetimeIndex
) -> pd.Timestamp:
    """Reaction day R of a filing.

    Filing on a trading day strictly before the cutoff -> that day.
    Filing at/after the cutoff, or on a non-trading day -> next trading day.
    Raises ValueError if the calendar does not extend to R.
    """
    r = _reaction_day_or_none(an_dt, cutoff_hhmm, _trading_index(trading_days))
    if r is None:
        raise ValueError(
            f"trading_days does not cover the reaction day of a filing at {an_dt}"
        )
    return r


def _reaction_day_or_none(
    an_dt: pd.Timestamp, cutoff_hhmm: str, idx: pd.DatetimeIndex
) -> Optional[pd.Timestamp]:
    ts = _naive_ts(an_dt)
    day = ts.normalize()
    hh, mm = _parse_hhmm_str(cutoff_hhmm)
    before_cutoff = (ts.hour * 60 + ts.minute) < (hh * 60 + mm)
    pos = idx.searchsorted(day)
    on_trading_day = pos < len(idx) and idx[pos] == day
    if on_trading_day and before_cutoff:
        return idx[pos]
    nxt = pos + 1 if on_trading_day else pos
    if nxt >= len(idx):
        return None
    return idx[nxt]


def tag_symbol_day(
    symbol: str,
    day: pd.Timestamp,
    *,
    announcements: pd.DataFrame,
    event_calendar: pd.DataFrame,
    bulk_deals: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    cfg: dict,
) -> NewsTags:
    """Tag signal day T for one symbol. See module docstring for semantics."""
    validate_news_gate_cfg(cfg)
    idx = _trading_index(trading_days)
    sym = _bare(symbol)
    t = _naive_ts(day).normalize()
    _position(t, idx)  # T must itself be a trading day (fail fast)

    days_before = int(cfg["results_days_before"])
    days_after = int(cfg["results_days_after"])
    bulk_w = int(cfg["bulk_deal_window_bdays"])
    cutoff = cfg["filing_cutoff_hhmm"]
    patterns = cfg["category_patterns"]
    cluster_cats = {str(c) for c in cfg["results_cluster_categories"]}
    purpose_re = re.compile(str(cfg["scheduled_purpose_regex"]), re.IGNORECASE)

    # ---- announcements: categories on T-1..T+1, results_reaction via R
    # T == R - k (k in 1..days_before)  <=>  R in {T+1 .. T+days_before}
    # T == R + k (k in 1..days_after)   <=>  R in {T-days_after .. T-1}
    reaction_set = _window(t, days_after, days_before, idx)
    category_days = _window(t, 1, 1, idx)
    categories: set = set()
    results_reaction = False
    ann = _rows_for(announcements, sym)
    if len(ann):
        an_dt = _naive_series(ann["an_dt"])
        an_day = an_dt.dt.normalize()
        # Superset pre-filter in calendar days; R(D) >= D always, and R(D) is
        # at most the next trading day after D, so a week of slack is plenty.
        lo = min(min(reaction_set), min(category_days)) - pd.Timedelta(days=7)
        hi = max(max(reaction_set), max(category_days))
        keep = (an_day >= lo) & (an_day <= hi)
        for ts, d, desc, text in zip(
            an_dt[keep], an_day[keep], ann.loc[keep, "desc"], ann.loc[keep, "text"]
        ):
            cat = bucket_filing(desc, text, patterns)
            if d in category_days:
                categories.add(cat)
            if cat == RESULTS_CATEGORY and not results_reaction:
                r = _reaction_day_or_none(ts, cutoff, idx)
                if r is not None and r in reaction_set:
                    results_reaction = True

    # ---- scheduled results board meeting on T or T+1..T+days_before
    results_scheduled = False
    cal = _rows_for(event_calendar, sym)
    if len(cal):
        sched_days = _window(t, 0, days_before, idx)
        meet = _naive_series(cal["meeting_date"]).dt.normalize()
        purpose_ok = cal["purpose"].astype(str).map(lambda p: bool(purpose_re.search(p)))
        results_scheduled = bool((meet.isin(sched_days) & purpose_ok.values).any())

    # ---- bulk deals on T-w..T+w, either side
    bulk_deal = False
    bulk_side: Optional[str] = None
    bd = _rows_for(bulk_deals, sym)
    if len(bd):
        bulk_days = _window(t, bulk_w, bulk_w, idx)
        bd_day = _naive_series(bd["date"]).dt.normalize()
        hit = bd.loc[bd_day.isin(bulk_days).values]
        if len(hit):
            bulk_deal = True
            sides = {str(s).strip().upper() for s in hit["side"]}
            bulk_side = "BOTH" if len(sides) > 1 else next(iter(sides))

    results_cluster = (
        results_reaction
        or results_scheduled
        or bool(categories & cluster_cats)
    )
    return NewsTags(
        results_reaction=results_reaction,
        results_scheduled=results_scheduled,
        bulk_deal=bulk_deal,
        bulk_side=bulk_side,
        categories=frozenset(categories),
        results_cluster=results_cluster,
    )


def gate_verdict(
    tags: NewsTags, dist_sma50_pct: Optional[float], cfg: dict
) -> Tuple[str, str]:
    """Rule R3 -> ("take" | "skip", reason).

    skip iff enabled and (results_reaction or results_scheduled)
             and dist_sma50_pct is known and > shallow_drop_sma50_pct
             and not (bulk_overrides_skip and bulk_deal).
    A missing SMA50 fails OPEN ("take", "sma50_unavailable").
    """
    if not bool(cfg["enabled"]):
        return "take", "gate_disabled"
    legs = []
    if tags.results_reaction:
        legs.append("results_reaction_day")
    if tags.results_scheduled:
        legs.append("results_scheduled")
    if not legs:
        return "take", "no_results_event"
    if dist_sma50_pct is None:
        return "take", "sma50_unavailable"
    if float(dist_sma50_pct) <= float(cfg["shallow_drop_sma50_pct"]):
        return "take", "deep_drop"
    if bool(cfg["bulk_overrides_skip"]) and tags.bulk_deal:
        return "take", "bulk_overrides"
    return "skip", "+".join(legs)


def tilt_multiplier(tags: NewsTags, cfg: dict) -> float:
    """Bulk tilt: cfg multiplier when bulk-deal tagged, else 1.0."""
    mult = float(cfg["bulk_deal_risk_multiplier"])
    return mult if tags.bulk_deal else 1.0
