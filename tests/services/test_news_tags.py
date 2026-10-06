"""Tests for services/news_tags.py — the pure tagging layer of the multiday
news gate (spec 2026-10-06 §2/§4/§6).

Calendar used throughout (IST-naive). 2026-10-02 (Fri) is Gandhi Jayanti, a
NSE holiday, so the trading sequence around the signal day is:

    Sep 29 Tue, Sep 30 Wed, Oct 1 Thu, [Oct 2 Fri HOLIDAY], [Oct 3-4 weekend],
    Oct 5 Mon, Oct 6 Tue, Oct 7 Wed, Oct 8 Thu, Oct 9 Fri, Oct 12 Mon
"""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from services.news_tags import (
    NewsTags,
    REQUIRED_CFG_KEYS,
    bucket_filing,
    gate_verdict,
    reaction_day,
    tag_symbol_day,
    tilt_multiplier,
    validate_news_gate_cfg,
)

TRADING_DAYS = pd.DatetimeIndex(pd.to_datetime([
    "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01",
    "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09",
    "2026-10-12", "2026-10-13",
]))

T = pd.Timestamp("2026-10-06")  # signal day (Tue)

CFG = {
    "enabled": True,
    "results_days_before": 1,
    "results_days_after": 0,
    "shallow_drop_sma50_pct": -20.0,
    "bulk_overrides_skip": True,
    "bulk_deal_window_bdays": 1,
    "filing_cutoff_hhmm": "15:30",
    "results_cluster_categories": ["results", "investor/analyst", "board meeting"],
    "category_patterns": [
        ["results", "financial result|results|outcome of board meeting.*result"],
        ["pledge/SAST", "pledge|encumbr|sast|substantial acquisition|invocation|release of"],
        ["price/volume clarification", "clarification|price movement|volume movement|spurt"],
        ["resignation/auditor", "resign|auditor|cessation|change in auditor"],
        ["order/contract win", "order|contract|bagged|award|letter of intent"],
        ["fund raise/pref/rights/QIP", "preferential|rights issue|qip|fund rais|allotment|warrant|conversion"],
        ["rating", "credit rating|rating action|downgrade|upgrade"],
        ["insolvency/default/legal", "insolvency|nclt|default|litigation|legal|sebi order|penalty|fraud|investigation|show cause"],
        ["bonus/split/dividend", "bonus|split|sub-division|dividend|buyback|buy back"],
        ["investor/analyst", "investor presentation|analyst|conference call|earnings call|press release|media"],
        ["board meeting", "board meeting"],
    ],
    "scheduled_purpose_regex": "financial result",
    "bulk_deal_risk_multiplier": 2.0,
}


# ------------------------------------------------------------ builders

def _ann(rows):
    """rows: [(symbol, 'YYYY-MM-DD HH:MM', desc, text), ...]"""
    return pd.DataFrame(
        [{"symbol": s, "an_dt": pd.Timestamp(dt), "desc": d, "text": x} for s, dt, d, x in rows],
        columns=["symbol", "an_dt", "desc", "text"],
    )


def _cal(rows):
    """rows: [(symbol, 'YYYY-MM-DD', purpose), ...]"""
    return pd.DataFrame(
        [{"symbol": s, "meeting_date": pd.Timestamp(d), "purpose": p} for s, d, p in rows],
        columns=["symbol", "meeting_date", "purpose"],
    )


def _bulk(rows):
    """rows: [('YYYY-MM-DD', symbol, side), ...]"""
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "symbol": s, "side": side} for d, s, side in rows],
        columns=["date", "symbol", "side"],
    )


EMPTY_ANN = _ann([])
EMPTY_CAL = _cal([])
EMPTY_BULK = _bulk([])


def tag(symbol="XYZ", day=T, *, announcements=EMPTY_ANN, event_calendar=EMPTY_CAL,
        bulk_deals=EMPTY_BULK, cfg=CFG, trading_days=TRADING_DAYS):
    return tag_symbol_day(
        symbol, day,
        announcements=announcements, event_calendar=event_calendar,
        bulk_deals=bulk_deals, trading_days=trading_days, cfg=cfg,
    )


def _tags(**kw) -> NewsTags:
    base = dict(results_reaction=False, results_scheduled=False, bulk_deal=False,
                bulk_side=None, categories=frozenset(), results_cluster=False)
    base.update(kw)
    return NewsTags(**base)


# ------------------------------------------------------------ bucket_filing

def test_bucket_filing_first_match_wins():
    # "Outcome of Board Meeting ... results" matches both 'results' and
    # 'board meeting'; 'results' is listed first so it must win.
    assert bucket_filing("Outcome of Board Meeting", "approved financial results",
                         CFG["category_patterns"]) == "results"


def test_bucket_filing_board_meeting_without_results_is_board_meeting():
    assert bucket_filing("Board Meeting Intimation", "to consider fund raising",
                         CFG["category_patterns"]) == "fund raise/pref/rights/QIP"
    assert bucket_filing("Board Meeting Intimation", "to consider other matters",
                         CFG["category_patterns"]) == "board meeting"


def test_bucket_filing_case_insensitive_and_other():
    assert bucket_filing("PLEDGE", "", CFG["category_patterns"]) == "pledge/SAST"
    assert bucket_filing("Change in Directorate", "appointment of CFO",
                         CFG["category_patterns"]) == "other"


def test_bucket_filing_none_fields_do_not_crash():
    assert bucket_filing(None, None, CFG["category_patterns"]) == "other"


# ------------------------------------------------------------ reaction_day

def test_reaction_day_before_cutoff_is_same_day():
    assert reaction_day(pd.Timestamp("2026-10-06 15:29"), "15:30", TRADING_DAYS) == T


def test_reaction_day_at_or_after_cutoff_rolls_to_next_trading_day():
    assert reaction_day(pd.Timestamp("2026-10-06 15:30"), "15:30", TRADING_DAYS) == pd.Timestamp("2026-10-07")
    assert reaction_day(pd.Timestamp("2026-10-06 19:45"), "15:30", TRADING_DAYS) == pd.Timestamp("2026-10-07")


def test_reaction_day_on_holiday_and_weekend_is_next_trading_day():
    # Oct 2 (holiday, before cutoff), Oct 3 (Sat) -> Mon Oct 5
    assert reaction_day(pd.Timestamp("2026-10-02 10:00"), "15:30", TRADING_DAYS) == pd.Timestamp("2026-10-05")
    assert reaction_day(pd.Timestamp("2026-10-03 11:00"), "15:30", TRADING_DAYS) == pd.Timestamp("2026-10-05")


def test_reaction_day_after_cutoff_before_holiday_skips_the_gap():
    # Oct 1 (Thu) 18:00 -> next trading day is Mon Oct 5 (Fri is a holiday)
    assert reaction_day(pd.Timestamp("2026-10-01 18:00"), "15:30", TRADING_DAYS) == pd.Timestamp("2026-10-05")


def test_reaction_day_beyond_calendar_raises():
    with pytest.raises(ValueError):
        reaction_day(pd.Timestamp("2026-10-13 16:00"), "15:30", TRADING_DAYS)


# ------------------------------------------------------------ results_reaction

RESULTS_FILING_R_T = _ann([("XYZ", "2026-10-06 09:00", "Financial Results", "Q2 unaudited")])
# filed after cutoff on Oct 6 -> R = Oct 7, so T = Oct 6 is R-1
RESULTS_FILING_R_T_PLUS_1 = _ann([("XYZ", "2026-10-06 18:00", "Financial Results", "Q2 unaudited")])
# filed Oct 5 before cutoff -> R = Oct 5, so T = Oct 6 is R+1
RESULTS_FILING_R_T_MINUS_1 = _ann([("XYZ", "2026-10-05 10:00", "Financial Results", "Q2 unaudited")])


def test_results_reaction_on_R():
    t = tag(announcements=RESULTS_FILING_R_T)
    assert t.results_reaction and t.results_cluster
    assert "results" in t.categories


def test_results_reaction_on_R_minus_1_with_days_before_1():
    assert tag(announcements=RESULTS_FILING_R_T_PLUS_1).results_reaction


def test_results_reaction_not_on_R_minus_1_when_days_before_0():
    cfg = {**CFG, "results_days_before": 0}
    assert not tag(announcements=RESULTS_FILING_R_T_PLUS_1, cfg=cfg).results_reaction


def test_results_reaction_not_on_R_plus_1_with_days_after_0():
    t = tag(announcements=RESULTS_FILING_R_T_MINUS_1)
    assert not t.results_reaction
    # the filing is still within T-1..T+1 so the category is visible
    assert "results" in t.categories and t.results_cluster


def test_results_reaction_on_R_plus_1_with_days_after_1():
    cfg = {**CFG, "results_days_after": 1}
    assert tag(announcements=RESULTS_FILING_R_T_MINUS_1, cfg=cfg).results_reaction


def test_results_reaction_steps_over_holiday_not_calendar_days():
    # Filing Thu Oct 1 after cutoff -> R = Mon Oct 5. Signal day Oct 5 is R.
    ann = _ann([("XYZ", "2026-10-01 20:00", "Outcome of Board Meeting", "financial results approved")])
    assert tag(day=pd.Timestamp("2026-10-05"), announcements=ann).results_reaction
    # Signal day Oct 1 is R-1 in TRADING days (not 4 calendar days before).
    assert tag(day=pd.Timestamp("2026-10-01"), announcements=ann).results_reaction


def test_non_results_filing_does_not_set_results_reaction():
    ann = _ann([("XYZ", "2026-10-06 09:00", "Pledge", "creation of pledge")])
    t = tag(announcements=ann)
    assert not t.results_reaction and not t.results_cluster
    assert t.categories == frozenset({"pledge/SAST"})


def test_other_symbol_filings_are_ignored():
    ann = _ann([("ABC", "2026-10-06 09:00", "Financial Results", "")])
    assert tag(symbol="XYZ", announcements=ann) == tag()


def test_categories_window_is_T_minus_1_to_T_plus_1_trading_days():
    ann = _ann([
        ("XYZ", "2026-10-05 10:00", "Pledge", ""),                 # T-1
        ("XYZ", "2026-10-07 10:00", "Credit Rating", "downgrade"),  # T+1
        ("XYZ", "2026-10-08 10:00", "Order", "bagged order"),       # T+2 -> excluded
        ("XYZ", "2026-10-01 10:00", "Resignation", ""),             # T-2 (Oct 1) -> excluded
    ])
    t = tag(announcements=ann)
    assert t.categories == frozenset({"pledge/SAST", "rating"})


def test_results_cluster_from_investor_category_without_reaction():
    ann = _ann([("XYZ", "2026-10-07 10:00", "Investor Presentation", "")])
    t = tag(announcements=ann)
    assert not t.results_reaction and t.results_cluster


# ------------------------------------------------------------ results_scheduled

def test_results_scheduled_for_meeting_on_T_plus_1():
    cal = _cal([("XYZ", "2026-10-07", "Financial Results")])
    t = tag(event_calendar=cal)
    assert t.results_scheduled and t.results_cluster


def test_results_scheduled_for_meeting_on_T():
    cal = _cal([("XYZ", "2026-10-06", "Financial Results/Dividend")])
    assert tag(event_calendar=cal).results_scheduled


def test_results_scheduled_not_for_meeting_on_T_plus_3():
    cal = _cal([("XYZ", "2026-10-09", "Financial Results")])
    assert not tag(event_calendar=cal).results_scheduled


def test_results_scheduled_not_for_meeting_on_T_minus_1():
    cal = _cal([("XYZ", "2026-10-05", "Financial Results")])
    assert not tag(event_calendar=cal).results_scheduled


def test_results_scheduled_ignores_non_results_purpose():
    cal = _cal([("XYZ", "2026-10-07", "Fund Raising")])
    assert not tag(event_calendar=cal).results_scheduled


def test_results_scheduled_T_plus_1_steps_over_holiday():
    # Signal day Oct 1 (Thu); T+1 trading day is Mon Oct 5.
    cal = _cal([("XYZ", "2026-10-05", "Financial Results")])
    assert tag(day=pd.Timestamp("2026-10-01"), event_calendar=cal).results_scheduled


# ------------------------------------------------------------ bulk deals

def test_bulk_window_T_minus_1_and_T_plus_1_and_sides():
    t = tag(bulk_deals=_bulk([("2026-10-05", "XYZ", "BUY")]))
    assert t.bulk_deal and t.bulk_side == "BUY"
    t = tag(bulk_deals=_bulk([("2026-10-07", "XYZ", "SELL")]))
    assert t.bulk_deal and t.bulk_side == "SELL"
    t = tag(bulk_deals=_bulk([("2026-10-06", "XYZ", "BUY"), ("2026-10-06", "XYZ", "SELL")]))
    assert t.bulk_deal and t.bulk_side == "BOTH"


def test_bulk_outside_window_is_not_tagged():
    t = tag(bulk_deals=_bulk([("2026-10-08", "XYZ", "BUY"), ("2026-10-01", "XYZ", "SELL")]))
    assert not t.bulk_deal and t.bulk_side is None


def test_bulk_window_is_trading_days():
    # Signal day Oct 5 (Mon): T-1 trading day is Thu Oct 1.
    t = tag(day=pd.Timestamp("2026-10-05"), bulk_deals=_bulk([("2026-10-01", "XYZ", "SELL")]))
    assert t.bulk_deal
    # With window 0, only T counts.
    cfg = {**CFG, "bulk_deal_window_bdays": 0}
    t = tag(day=pd.Timestamp("2026-10-05"), bulk_deals=_bulk([("2026-10-01", "XYZ", "SELL")]), cfg=cfg)
    assert not t.bulk_deal


def test_bulk_does_not_set_results_cluster():
    t = tag(bulk_deals=_bulk([("2026-10-06", "XYZ", "BUY")]))
    assert t.bulk_deal and not t.results_cluster


# ------------------------------------------------------------ symbol normalisation

@pytest.mark.parametrize("query", ["XYZ", "NSE:XYZ", "xyz", " NSE:xyz "])
@pytest.mark.parametrize("feed_sym", ["XYZ", "NSE:XYZ"])
def test_symbol_prefix_normalised_both_sides(query, feed_sym):
    ann = _ann([(feed_sym, "2026-10-06 09:00", "Financial Results", "")])
    bulk = _bulk([("2026-10-06", feed_sym, "BUY")])
    cal = _cal([(feed_sym, "2026-10-07", "Financial Results")])
    t = tag(symbol=query, announcements=ann, bulk_deals=bulk, event_calendar=cal)
    assert t.results_reaction and t.bulk_deal and t.results_scheduled


# ------------------------------------------------------------ empty / degenerate input

def test_empty_frames_yield_no_tags():
    t = tag(announcements=pd.DataFrame(), event_calendar=pd.DataFrame(), bulk_deals=pd.DataFrame())
    assert t == _tags()


def test_signal_day_not_a_trading_day_raises():
    with pytest.raises(ValueError):
        tag(day=pd.Timestamp("2026-10-02"))


def test_tz_aware_inputs_are_coerced_to_ist_naive():
    # 03:30 UTC == 09:00 IST on Oct 6 -> before cutoff, R = T
    ann = pd.DataFrame({
        "symbol": ["XYZ"],
        "an_dt": [pd.Timestamp("2026-10-06 03:30", tz="UTC")],
        "desc": ["Financial Results"], "text": [""],
    })
    assert tag(announcements=ann).results_reaction


# ------------------------------------------------------------ gate_verdict truth table

SHALLOW = -7.2   # > -20 -> shallow
DEEP = -25.0     # <= -20 -> deep


@pytest.mark.parametrize("tags_kw, dist, expected", [
    # no results event -> take regardless of depth
    (dict(), SHALLOW, ("take", "no_results_event")),
    (dict(bulk_deal=True), SHALLOW, ("take", "no_results_event")),
    # reaction leg, shallow -> skip
    (dict(results_reaction=True), SHALLOW, ("skip", "results_reaction_day")),
    # scheduled leg, shallow -> skip
    (dict(results_scheduled=True), SHALLOW, ("skip", "results_scheduled")),
    # both legs -> skip, reason names both
    (dict(results_reaction=True, results_scheduled=True), SHALLOW,
     ("skip", "results_reaction_day+results_scheduled")),
    # deep drop keeps
    (dict(results_reaction=True), DEEP, ("take", "deep_drop")),
    (dict(results_scheduled=True), DEEP, ("take", "deep_drop")),
    # boundary: dist == threshold is NOT shallow (skip only if dist > threshold)
    (dict(results_reaction=True), -20.0, ("take", "deep_drop")),
    # bulk overrides the skip
    (dict(results_reaction=True, bulk_deal=True), SHALLOW, ("take", "bulk_overrides")),
    (dict(results_scheduled=True, bulk_deal=True), SHALLOW, ("take", "bulk_overrides")),
    # sma50 unknown -> fail open
    (dict(results_reaction=True), None, ("take", "sma50_unavailable")),
    (dict(results_scheduled=True, bulk_deal=True), None, ("take", "sma50_unavailable")),
    # cluster-only (investor/analyst) is NOT a skip trigger
    (dict(results_cluster=True, categories=frozenset({"investor/analyst"})), SHALLOW,
     ("take", "no_results_event")),
])
def test_gate_verdict_truth_table(tags_kw, dist, expected):
    assert gate_verdict(_tags(**tags_kw), dist, CFG) == expected


def test_gate_verdict_bulk_override_disabled_still_skips():
    cfg = {**CFG, "bulk_overrides_skip": False}
    assert gate_verdict(_tags(results_reaction=True, bulk_deal=True), SHALLOW, cfg) == \
        ("skip", "results_reaction_day")


@pytest.mark.parametrize("tags_kw, dist", [
    (dict(results_reaction=True), SHALLOW),
    (dict(results_scheduled=True), SHALLOW),
    (dict(results_reaction=True, results_scheduled=True), None),
    (dict(), DEEP),
])
def test_gate_verdict_disabled_always_takes(tags_kw, dist):
    cfg = {**CFG, "enabled": False}
    assert gate_verdict(_tags(**tags_kw), dist, cfg) == ("take", "gate_disabled")


def test_gate_verdict_end_to_end_from_tagging():
    ann = _ann([("NSE:XYZ", "2026-10-06 18:00", "Financial Results", "")])  # R = Oct 7, T = R-1
    t = tag(symbol="NSE:XYZ", announcements=ann)
    assert gate_verdict(t, SHALLOW, CFG) == ("skip", "results_reaction_day")
    t2 = tag(symbol="NSE:XYZ", announcements=ann, bulk_deals=_bulk([("2026-10-07", "XYZ", "SELL")]))
    assert gate_verdict(t2, SHALLOW, CFG) == ("take", "bulk_overrides")


# ------------------------------------------------------------ tilt_multiplier

def test_tilt_multiplier():
    assert tilt_multiplier(_tags(bulk_deal=True, bulk_side="BUY"), CFG) == 2.0
    assert tilt_multiplier(_tags(bulk_deal=True, bulk_side="SELL"), CFG) == 2.0
    assert tilt_multiplier(_tags(), CFG) == 1.0
    assert tilt_multiplier(_tags(bulk_deal=True), {**CFG, "bulk_deal_risk_multiplier": 1.0}) == 1.0


# ------------------------------------------------------------ config fail-fast

@pytest.mark.parametrize("missing", REQUIRED_CFG_KEYS)
def test_missing_config_key_raises_keyerror(missing):
    cfg = copy.deepcopy(CFG)
    del cfg[missing]
    with pytest.raises(KeyError):
        tag(cfg=cfg)


def test_missing_gate_keys_raise_in_gate_verdict_and_tilt():
    with pytest.raises(KeyError):
        gate_verdict(_tags(results_reaction=True), SHALLOW, {k: v for k, v in CFG.items() if k != "shallow_drop_sma50_pct"})
    with pytest.raises(KeyError):
        gate_verdict(_tags(), SHALLOW, {k: v for k, v in CFG.items() if k != "enabled"})
    with pytest.raises(KeyError):
        tilt_multiplier(_tags(bulk_deal=True), {k: v for k, v in CFG.items() if k != "bulk_deal_risk_multiplier"})


def test_patterns_without_results_category_rejected():
    cfg = copy.deepcopy(CFG)
    cfg["category_patterns"] = [p for p in cfg["category_patterns"] if p[0] != "results"]
    with pytest.raises(ValueError):
        validate_news_gate_cfg(cfg)


def test_bad_cutoff_string_rejected():
    with pytest.raises(ValueError):
        validate_news_gate_cfg({**CFG, "filing_cutoff_hhmm": "1530"})
